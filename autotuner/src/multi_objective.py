"""
multi_objective.py — Otimização Multi-Objetivo para o HCPA Autotuner.

Modela o problema de treinamento como um problema de otimização multi-objetivo,
considerando simultaneamente:

  1. Qualidade preditiva  : val_AUC, sensibilidade, especificidade
  2. Eficiência temporal  : throughput (img/s), tempo de treinamento
  3. Eficiência de memória: utilização de VRAM (penaliza uso excessivo)

Os pesos padrão foram calibrados a partir da análise dos resultados experimentais:
  - AUC domina (0.60) porque é o KPI principal do projeto médico
  - Throughput importa (0.25) para viabilidade computacional
  - Memória importa menos (0.15) mas penaliza risco de OOM

O scorer produz:
  - `composite_score`: escalar [0, 1] para comparação de épocas/configs
  - `ParetoFrontier` : conjunto de soluções não-dominadas ao longo do treino
  - `MultiObjectiveSignal`: sinais de ajuste derivados do score multi-objetivo
  - Histórico de scores para detecção de plateau multi-dimensional
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .offline_knowledge import OfflineKnowledgeBase, get_knowledge_base


# ═══════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectiveWeights:
    """
    Pesos para cada objetivo na função de score composto.

    Invariante: auc_weight + throughput_weight + memory_weight == 1.0
    """
    auc_weight: float = 0.60
    throughput_weight: float = 0.25
    memory_weight: float = 0.15

    def __post_init__(self):
        total = self.auc_weight + self.throughput_weight + self.memory_weight
        if abs(total - 1.0) > 1e-6:
            # Normalizar automaticamente
            self.auc_weight /= total
            self.throughput_weight /= total
            self.memory_weight /= total

    @classmethod
    def quality_focused(cls) -> "ObjectiveWeights":
        """AUC domina — para cenário onde precisão é crítica."""
        return cls(auc_weight=0.80, throughput_weight=0.10, memory_weight=0.10)

    @classmethod
    def efficiency_focused(cls) -> "ObjectiveWeights":
        """Throughput em destaque — para benchmark de performance."""
        return cls(auc_weight=0.40, throughput_weight=0.45, memory_weight=0.15)

    @classmethod
    def balanced(cls) -> "ObjectiveWeights":
        """Balanceado — padrão."""
        return cls(auc_weight=0.60, throughput_weight=0.25, memory_weight=0.15)


@dataclass
class EpochObjectives:
    """Valores dos objetivos em uma época."""
    epoch: int
    val_auc: float
    train_loss: float
    val_loss: float
    throughput_img_s: float
    gpu_mem_used_mb: float
    gpu_mem_total_mb: float

    @property
    def gpu_mem_pct(self) -> float:
        if self.gpu_mem_total_mb > 0:
            return self.gpu_mem_used_mb / self.gpu_mem_total_mb
        return 0.0

    @property
    def is_valid(self) -> bool:
        return (not math.isnan(self.val_auc) and not math.isinf(self.val_auc)
                and not math.isnan(self.train_loss))


@dataclass
class CompositeScore:
    """Score composto multi-objetivo para uma época."""
    epoch: int
    composite: float          # score final ponderado [0, 1]
    auc_score: float          # sub-score AUC normalizado [0, 1]
    throughput_score: float   # sub-score throughput normalizado [0, 1]
    memory_score: float       # sub-score memória normalizado [0, 1]
    weights: ObjectiveWeights
    raw_objectives: EpochObjectives

    def dominates(self, other: "CompositeScore") -> bool:
        """
        Retorna True se este score domina 'other' no sentido de Pareto.
        (melhor ou igual em todos os objetivos, estritamente melhor em pelo menos um)
        """
        return (
            self.auc_score >= other.auc_score
            and self.throughput_score >= other.throughput_score
            and self.memory_score >= other.memory_score
            and (
                self.auc_score > other.auc_score
                or self.throughput_score > other.throughput_score
                or self.memory_score > other.memory_score
            )
        )

    def __str__(self) -> str:
        return (
            f"E{self.epoch} composite={self.composite:.4f} "
            f"[AUC={self.auc_score:.3f} thr={self.throughput_score:.3f} mem={self.memory_score:.3f}]"
        )


@dataclass
class MultiObjectiveSignal:
    """
    Sinal gerado pelo scorer multi-objetivo para guiar o controller.

    Indica quais objetivos estão sub-desempenhando e que tipo de ação pode ajudar.
    """
    epoch: int
    auc_lagging: bool          # AUC abaixo da expectativa histórica
    throughput_lagging: bool   # throughput abaixo do baseline
    memory_pressure: bool      # uso de memória > 85%
    composite_plateau: bool    # score composto não melhora há N épocas
    suggested_action: str      # texto descritivo da ação sugerida
    priority: int              # 0=info, 1=warn, 2=urgent

    def has_action(self) -> bool:
        return bool(self.suggested_action)

    def __str__(self) -> str:
        flags = []
        if self.auc_lagging:
            flags.append("AUC↓")
        if self.throughput_lagging:
            flags.append("thr↓")
        if self.memory_pressure:
            flags.append("mem!")
        if self.composite_plateau:
            flags.append("plateau")
        flag_str = ",".join(flags) if flags else "OK"
        prio = ["INFO", "WARN", "URGENT"][self.priority]
        return (
            f"[MultiObj E{self.epoch} {prio}] flags=[{flag_str}] "
            f"→ {self.suggested_action or 'sem ação'}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ParetoFrontier
# ═══════════════════════════════════════════════════════════════════════════

class ParetoFrontier:
    """
    Mantém a fronteira de Pareto dos scores compostos ao longo do treinamento.

    Permite identificar quais épocas/configs foram Pareto-ótimas e qual
    checkpoint seria mais indicado dado um vetor de preferências.
    """

    def __init__(self):
        self._frontier: List[CompositeScore] = []
        self._all_scores: List[CompositeScore] = []

    def update(self, score: CompositeScore):
        """Adiciona score e atualiza a fronteira de Pareto."""
        self._all_scores.append(score)
        # Remover da fronteira scores dominados pelo novo
        self._frontier = [s for s in self._frontier if not score.dominates(s)]
        # Adicionar novo apenas se não for dominado por nenhum existente
        if not any(s.dominates(score) for s in self._frontier):
            self._frontier.append(score)

    @property
    def frontier(self) -> List[CompositeScore]:
        return list(self._frontier)

    def best_by_objective(
        self,
        weights: Optional[ObjectiveWeights] = None,
    ) -> Optional[CompositeScore]:
        """
        Retorna o ponto da fronteira com maior score composto
        segundo os pesos fornecidos (ou os pesos do próprio score).
        """
        if not self._frontier:
            return None
        if weights is None:
            return max(self._frontier, key=lambda s: s.composite)
        return max(
            self._frontier,
            key=lambda s: (
                weights.auc_weight * s.auc_score
                + weights.throughput_weight * s.throughput_score
                + weights.memory_weight * s.memory_score
            ),
        )

    def best_auc_epoch(self) -> Optional[CompositeScore]:
        if not self._all_scores:
            return None
        return max(self._all_scores, key=lambda s: s.auc_score)

    def best_composite_epoch(self) -> Optional[CompositeScore]:
        if not self._all_scores:
            return None
        return max(self._all_scores, key=lambda s: s.composite)

    def summary(self) -> str:
        if not self._frontier:
            return "[ParetoFrontier] Vazia"
        lines = [f"[ParetoFrontier] {len(self._frontier)} ponto(s) não-dominados:"]
        for s in sorted(self._frontier, key=lambda x: x.composite, reverse=True):
            lines.append(f"  {s}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  MultiObjectiveScorer
# ═══════════════════════════════════════════════════════════════════════════

class MultiObjectiveScorer:
    """
    Calcula e rastreia scores multi-objetivos para o HCPA Autotuner.

    Usa os perfis históricos do OfflineKnowledgeBase como referência de
    normalização: throughput de 1000 img/s = score 1.0 (ajustável por GPU).

    Gera MultiObjectiveSignal recommending actions para o controller.
    """

    def __init__(
        self,
        stack: str,
        mode: str,
        gpu_name: str = "default",
        weights: Optional[ObjectiveWeights] = None,
        kb: Optional[OfflineKnowledgeBase] = None,
        # Thresholds de alerta
        auc_lag_threshold: float = 0.02,      # AUC abaixo do esperado por este delta
        throughput_lag_ratio: float = 0.60,    # throughput < 60% do baseline
        memory_pressure_pct: float = 0.85,     # mem > 85% do total
        composite_plateau_patience: int = 10,  # épocas sem melhora no score composto
    ):
        self.stack = stack
        self.mode = mode
        self.gpu_name = gpu_name
        self.weights = weights or ObjectiveWeights.balanced()
        self.kb = kb or get_knowledge_base()
        self.auc_lag_threshold = auc_lag_threshold
        self.throughput_lag_ratio = throughput_lag_ratio
        self.memory_pressure_pct = memory_pressure_pct
        self.composite_plateau_patience = composite_plateau_patience

        # Carregar perfil histórico para normalização
        variant_key = f"{stack}_{mode}"
        self._profile = self.kb.get_variant_profile(variant_key, gpu_name)
        self._convergence = self.kb.estimate_convergence(stack, mode, gpu_name)

        # Referências de normalização baseadas no perfil histórico
        # AUC: normalizado entre 0.80 (mínimo aceitável) e 1.0
        self._auc_min = 0.80
        self._auc_max = 1.0
        # Throughput: referência = média histórica para a variante × GPU
        self._thr_ref = max(1.0, self._profile.mean_throughput_img_s)
        # Memória: memória total disponível (atualizado dinamicamente)
        self._mem_ref = max(1.0, self._profile.mean_peak_mem_mb)

        self._scores: List[CompositeScore] = []
        self._pareto = ParetoFrontier()
        self._best_composite = -1.0
        self._composite_plateau = 0
        self._total_epochs = 200

    def set_total_epochs(self, n: int):
        self._total_epochs = n

    def update_throughput_reference(self, thr_ref: float):
        """Atualiza referência de throughput (útil após primeiras épocas)."""
        if thr_ref > self._thr_ref:
            self._thr_ref = thr_ref

    def _normalize_auc(self, val_auc: float) -> float:
        """Normaliza AUC para [0, 1]. AUC >= 0.99 → 1.0, AUC=0.80 → 0.0."""
        if math.isnan(val_auc):
            return 0.0
        norm = (val_auc - self._auc_min) / (self._auc_max - self._auc_min)
        return max(0.0, min(1.0, norm))

    def _normalize_throughput(self, thr: float) -> float:
        """
        Normaliza throughput relativo ao histórico.
        thr = _thr_ref → score 1.0; thr acima → score > 1.0 (cap em 1.0).
        """
        if thr <= 0 or self._thr_ref <= 0:
            return 0.5   # desconhecido → neutro
        ratio = thr / self._thr_ref
        # score logístico: penaliza abaixo de 60% severamente
        if ratio >= 1.0:
            return 1.0
        if ratio >= 0.80:
            return 0.80 + (ratio - 0.80) * (0.20 / 0.20)
        if ratio >= 0.60:
            return 0.50 + (ratio - 0.60) * (0.30 / 0.20)
        return ratio / 0.60 * 0.50

    def _normalize_memory(self, mem_used_mb: float, mem_total_mb: float) -> float:
        """
        Score de memória: menor uso relativo é melhor.
        mem = 0%  → 1.0; mem = 100% → 0.0
        """
        if mem_total_mb <= 0:
            return 0.8   # desconhecido → neutro-positivo
        pct = mem_used_mb / mem_total_mb
        # Inverte e suaviza: 0% → 1.0, 80% → 0.5, 95%+ → 0.1
        if pct <= 0.50:
            return 1.0 - pct * 0.2          # leve penalidade até 50%
        if pct <= 0.85:
            return 0.90 - (pct - 0.50) * (0.50 / 0.35)
        return max(0.0, 0.40 - (pct - 0.85) * (0.40 / 0.15))

    def score_epoch(
        self,
        epoch: int,
        val_auc: float,
        train_loss: float,
        val_loss: float,
        throughput_img_s: float,
        gpu_mem_used_mb: float = 0.0,
        gpu_mem_total_mb: float = 0.0,
    ) -> CompositeScore:
        """
        Calcula o score multi-objetivo para a época dada.

        Args:
            epoch: Número da época atual
            val_auc: AUC de validação
            train_loss: Loss de treino
            val_loss: Loss de validação
            throughput_img_s: Imagens/segundo observadas
            gpu_mem_used_mb: Memória GPU usada
            gpu_mem_total_mb: Memória GPU total

        Returns:
            CompositeScore com todos os sub-scores
        """
        objs = EpochObjectives(
            epoch=epoch,
            val_auc=val_auc,
            train_loss=train_loss,
            val_loss=val_loss,
            throughput_img_s=throughput_img_s,
            gpu_mem_used_mb=gpu_mem_used_mb,
            gpu_mem_total_mb=gpu_mem_total_mb if gpu_mem_total_mb > 0 else self._mem_ref,
        )

        auc_sc = self._normalize_auc(val_auc)
        thr_sc = self._normalize_throughput(throughput_img_s)
        mem_sc = self._normalize_memory(gpu_mem_used_mb, gpu_mem_total_mb or self._mem_ref)

        composite = (
            self.weights.auc_weight * auc_sc
            + self.weights.throughput_weight * thr_sc
            + self.weights.memory_weight * mem_sc
        )

        score = CompositeScore(
            epoch=epoch,
            composite=composite,
            auc_score=auc_sc,
            throughput_score=thr_sc,
            memory_score=mem_sc,
            weights=self.weights,
            raw_objectives=objs,
        )

        self._scores.append(score)
        self._pareto.update(score)

        # Atualizar plateau do score composto
        if composite > self._best_composite + 1e-5:
            self._best_composite = composite
            self._composite_plateau = 0
        else:
            self._composite_plateau += 1

        # Atualizar referência de throughput dinamicamente
        if throughput_img_s > self._thr_ref * 0.8:
            # Usar média móvel para estabilizar
            self._thr_ref = 0.8 * self._thr_ref + 0.2 * throughput_img_s

        return score

    def generate_signal(
        self,
        score: CompositeScore,
        total_epochs: int = 200,
    ) -> MultiObjectiveSignal:
        """
        Gera sinal de ação baseado no score multi-objetivo atual.

        Considera desempenho relativo ao histórico e tendência recente.
        """
        epoch = score.epoch
        objs = score.raw_objectives

        # ── 1. AUC lagando atrás do esperado? ──
        expected_auc = self._expected_auc_at_epoch(epoch, total_epochs)
        auc_lagging = (
            not math.isnan(objs.val_auc)
            and objs.val_auc < expected_auc - self.auc_lag_threshold
        )

        # ── 2. Throughput abaixo do baseline? ──
        thr_ref = self._profile.mean_throughput_img_s
        throughput_lagging = (
            objs.throughput_img_s > 0
            and thr_ref > 0
            and objs.throughput_img_s < thr_ref * self.throughput_lag_ratio
        )

        # ── 3. Pressão de memória? ──
        memory_pressure = (
            objs.gpu_mem_total_mb > 0
            and objs.gpu_mem_pct > self.memory_pressure_pct
        )

        # ── 4. Plateau composto? ──
        composite_plateau = self._composite_plateau >= self.composite_plateau_patience

        # ── Sintetizar sinal e ação sugerida ──
        suggested_action, priority = self._decide_action(
            score=score,
            auc_lagging=auc_lagging,
            throughput_lagging=throughput_lagging,
            memory_pressure=memory_pressure,
            composite_plateau=composite_plateau,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        return MultiObjectiveSignal(
            epoch=epoch,
            auc_lagging=auc_lagging,
            throughput_lagging=throughput_lagging,
            memory_pressure=memory_pressure,
            composite_plateau=composite_plateau,
            suggested_action=suggested_action,
            priority=priority,
        )

    def _expected_auc_at_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        Modela curva de AUC esperada baseada no perfil histórico.
        Curva assintótica: cresce de 0.5 para mean_auc_best com função logística.
        """
        progress = epoch / max(1, total_epochs)
        target = self._profile.mean_auc_best
        floor = 0.50
        # Logística simples parametrizada pelo plateau esperado
        plateau_frac = self._convergence.expected_plateau_epoch / max(1, total_epochs)
        # Parâmetro k: quanto mais rápido o plateau, mais íngreme a curva
        k = 6.0 / max(0.1, plateau_frac)
        mid = plateau_frac * 0.6
        sigmoid = 1.0 / (1.0 + math.exp(-k * (progress - mid)))
        return floor + (target - floor) * sigmoid

    def _decide_action(
        self,
        score: CompositeScore,
        auc_lagging: bool,
        throughput_lagging: bool,
        memory_pressure: bool,
        composite_plateau: bool,
        epoch: int,
        total_epochs: int,
    ) -> Tuple[str, int]:
        """
        Decide ação recomendada baseada nos sinais ativos.

        Retorna (ação_texto, prioridade).
        Hierarquia: memória (urgente) > AUC+plateau > throughput.
        """
        # Urgente: risco de OOM
        if memory_pressure:
            return (
                "REDUZIR batch_size (pressão de memória > 85%); "
                f"mem atual={score.raw_objectives.gpu_mem_pct:.1%}",
                2,
            )

        # AUC lagging + plateau composto: situação critica
        if auc_lagging and composite_plateau:
            return (
                "AUC abaixo do histórico E plateau composto: reduzir LR e/ou ativar mixup",
                2,
            )

        # Só AUC lagging: LR pode estar muito alto
        if auc_lagging and epoch > 20:
            progress = epoch / max(1, total_epochs)
            if progress < 0.3:
                return (
                    f"AUC={score.raw_objectives.val_auc:.4f} abaixo do esperado "
                    f"({self._expected_auc_at_epoch(epoch, total_epochs):.4f}): "
                    "verificar LR e augmentation",
                    1,
                )
            else:
                return (
                    f"AUC lagging (progresso={progress:.0%}): considerar label_smoothing reduzido",
                    1,
                )

        # Plateau composto (AUC + throughput estagnados)
        if composite_plateau and epoch > 30:
            if score.throughput_score < 0.7:
                return (
                    f"Score composto em plateau por {self._composite_plateau} épocas "
                    "e throughput baixo: avaliar batch_size ou augmentation",
                    1,
                )
            return (
                f"Score composto em plateau por {self._composite_plateau} épocas: "
                "reduzir LR gradualmente",
                1,
            )

        # Throughput lagging (GPU subutilizada)
        if throughput_lagging:
            thr = score.raw_objectives.throughput_img_s
            ref = self._profile.mean_throughput_img_s
            return (
                f"Throughput {thr:.0f} img/s << histórico {ref:.0f} img/s: "
                "verificar batch_size, num_workers ou bottleneck de I/O",
                1,
            )

        return ("", 0)

    # ──────────────────────────────────────────────────────────────────────
    #  Propriedades e utilitários
    # ──────────────────────────────────────────────────────────────────────

    @property
    def pareto_frontier(self) -> ParetoFrontier:
        return self._pareto

    @property
    def best_composite_score(self) -> float:
        return self._best_composite

    @property
    def composite_plateau_epochs(self) -> int:
        return self._composite_plateau

    def latest_score(self) -> Optional[CompositeScore]:
        return self._scores[-1] if self._scores else None

    def score_history(self) -> List[CompositeScore]:
        return list(self._scores)

    def is_composite_improving(self, window: int = 5) -> bool:
        """Retorna True se o score composto melhorou nas últimas 'window' épocas."""
        if len(self._scores) < window:
            return True
        recent = self._scores[-window:]
        oldest_worst = min(s.composite for s in recent[:window // 2])
        newest_best = max(s.composite for s in recent[window // 2:])
        return newest_best > oldest_worst + 1e-4

    def get_adaptive_lr_factor(self) -> float:
        """
        Retorna fator de ajuste de LR baseado no score multi-objetivo.
        
        Se o score composto está em plateau mas AUC ainda está crescendo,
        sugere redução moderada (0.7x). Se tudo em plateau, redução maior (0.5x).
        """
        if self._composite_plateau < self.composite_plateau_patience:
            return 1.0   # sem ajuste

        # Verificar se AUC ainda cresce
        if len(self._scores) >= 5:
            auc_scores = [s.auc_score for s in self._scores[-5:]]
            auc_improving = auc_scores[-1] > auc_scores[0] + 1e-4
            if auc_improving:
                return 0.7   # plateau leve de score, AUC ainda melhora
        return 0.5    # plateau completo

    def get_gradient_accumulation_suggestion(
        self,
        current_bs: int,
        current_thr: float,
    ) -> Optional[int]:
        """
        Sugere gradient_accumulation steps quando:
        - batch_size atual é pequeno (< 32) E
        - throughput ainda tem margem

        Se GA > 1, permite simular batch maior sem gastar mais memória.
        """
        if current_bs >= 64:
            return None   # BS suficiente
        effective_bs = current_bs
        target_effective_bs = 64
        ga_steps = max(1, target_effective_bs // effective_bs)
        if ga_steps > 8:
            ga_steps = 8   # limitar overhead
        return ga_steps if ga_steps > 1 else None

    def summary(self) -> str:
        """Relatório compacto do scorer."""
        if not self._scores:
            return "[MultiObjective] Sem dados ainda."
        latest = self._scores[-1]
        lines = [
            f"[MultiObjective] {self.stack}_{self.mode} @ {self.gpu_name}",
            f"  Pesos: AUC={self.weights.auc_weight:.2f} thr={self.weights.throughput_weight:.2f} mem={self.weights.memory_weight:.2f}",
            f"  Score atual: {latest}",
            f"  Melhor score: {self._best_composite:.4f}",
            f"  Plateau composto: {self._composite_plateau} épocas",
            self._pareto.summary(),
        ]
        return "\n".join(lines)

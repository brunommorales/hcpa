"""
convergence_tracker.py — Rastreamento e predição de convergência para o HCPA Autotuner.

Modela a trajetória de convergência do treinamento epoch-by-epoch e:
  1. Detecta padrões de convergência: rápida, lenta, estagnada, instável
  2. Prediz o AUC final com base nos primeiros N épocas (modelo de saturação)
  3. Adapta dinamicamente a paciência do plateau no controller
  4. Detecta "loss landscape instability" (alta variância nos últimos K épocas)
  5. Compara convergência atual com padrões históricos das 6 variantes

Os parâmetros de calibração foram derivados da análise dos 162 runs históricos:
  - tensorflow_base: convergência rápida (AUC 0.95 em ~100s, ~5 épocas)
  - tensorflow_opt:  convergência média (~70-80s)
  - pytorch_opt:     convergência lenta (plateau após época ~120)
  - monai_opt:       alta variabilidade no final (instabilidade EMA)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .offline_knowledge import OfflineKnowledgeBase, ConvergenceEstimate, get_knowledge_base


# ═══════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceState:
    """Estado atual de convergência estimado pelo tracker."""
    epoch: int
    predicted_final_auc: float     # AUC prevista ao final das 200 épocas
    predicted_best_auc: float      # AUC prevista do melhor checkpoint
    convergence_rate: float        # AUC/época média nas últimas K épocas
    is_converging: bool            # True se ainda está melhorando visivelmente
    is_unstable: bool              # True se oscilação > threshold
    plateau_epochs: int            # épocas sem melhora significativa
    recommended_patience: int      # paciência recomendada para LR reduction
    confidence: float              # confiança na predição [0, 1]

    @property
    def phase(self) -> str:
        """Fase de treinamento estimada."""
        if self.is_unstable:
            return "unstable"
        if self.plateau_epochs > self.recommended_patience:
            return "plateau"
        if self.convergence_rate > 0.003:
            return "rapid_improvement"
        if self.convergence_rate > 0.0005:
            return "gradual_improvement"
        return "marginal"

    def __str__(self) -> str:
        return (
            f"[Convergence E{self.epoch}] "
            f"pred_final={self.predicted_final_auc:.4f} "
            f"pred_best={self.predicted_best_auc:.4f} "
            f"rate={self.convergence_rate:.5f}/epoch "
            f"phase={self.phase} "
            f"plateau={self.plateau_epochs} "
            f"patience={self.recommended_patience}"
        )


@dataclass
class InstabilityEvent:
    """Evento de instabilidade detectado."""
    epoch: int
    metric: str             # "loss" | "auc" | "both"
    severity: float         # desvio-padrão normalizado
    description: str

    def is_severe(self) -> bool:
        return self.severity > 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  ConvergenceTracker
# ═══════════════════════════════════════════════════════════════════════════

class ConvergenceTracker:
    """
    Rastreia e prediz convergência do treinamento.

    Usa modelo de saturação exponencial para prever AUC final:
        AUC(t) ≈ AUC_max × (1 - exp(-t / τ))

    Parâmetros τ e AUC_max são ajustados online via mínimos quadrados
    não-linear simplificado (2 pontos) nos primeiros epochs.
    """

    def __init__(
        self,
        stack: str,
        mode: str,
        gpu_name: str = "default",
        total_epochs: int = 200,
        kb: Optional[OfflineKnowledgeBase] = None,
        # Janelas de análise
        rate_window: int = 5,          # janela para calcular taxa de melhora
        instability_window: int = 8,   # janela para detectar instabilidade
        # Thresholds
        instability_cv_threshold: float = 0.05,  # coef. variação do AUC
        min_improvement_delta: float = 5e-5,      # delta mínimo para "melhorando"
    ):
        self.stack = stack
        self.mode = mode
        self.gpu_name = gpu_name
        self.total_epochs = total_epochs
        self.rate_window = rate_window
        self.instability_window = instability_window
        self.instability_cv_threshold = instability_cv_threshold
        self.min_improvement_delta = min_improvement_delta

        # Knowledge base
        self.kb = kb or get_knowledge_base()
        variant_key = f"{stack}_{mode}"
        self._hist_estimate: ConvergenceEstimate = self.kb.estimate_convergence(
            stack, mode, gpu_name, total_epochs
        )
        self._profile = self.kb.get_variant_profile(variant_key, gpu_name)

        # Histórico de métricas por época
        self._auc_history: List[Tuple[int, float]] = []    # (epoch, val_auc)
        self._loss_history: List[Tuple[int, float]] = []   # (epoch, train_loss)
        self._instability_events: List[InstabilityEvent] = []

        # Estado do modelo de predição
        self._auc_max_est: Optional[float] = None          # AUC_max estimado
        self._tau_est: Optional[float] = None              # constante de tempo estimada
        self._best_auc_seen: float = -1.0
        self._plateau_counter: int = 0
        self._plateau_patience: int = self._hist_estimate.suggested_plateau_patience

    # ──────────────────────────────────────────────────────────────────────
    #  API principal
    # ──────────────────────────────────────────────────────────────────────

    def record(self, epoch: int, val_auc: float, train_loss: float) -> ConvergenceState:
        """
        Registra métricas de uma época e retorna estado de convergência atualizado.

        Args:
            epoch: Número da época
            val_auc: AUC de validação
            train_loss: Loss de treino

        Returns:
            ConvergenceState com análise atual
        """
        if not math.isnan(val_auc) and not math.isinf(val_auc):
            self._auc_history.append((epoch, val_auc))
        if not math.isnan(train_loss) and not math.isinf(train_loss):
            self._loss_history.append((epoch, train_loss))

        # Atualizar melhor AUC e plateau
        if val_auc > self._best_auc_seen + self.min_improvement_delta:
            self._best_auc_seen = val_auc
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1

        # Atualizar modelo de predição
        self._update_prediction_model()

        # Detectar instabilidade
        unstable = self._detect_instability(epoch)

        # Calcular taxa de convergência recente
        rate = self._compute_convergence_rate()

        # Predições
        pred_final, pred_best = self._predict_final_auc(epoch)

        # Adaptar paciência dinamicamente
        patience = self._adaptive_patience(epoch, rate, unstable)
        self._plateau_patience = patience

        # Confiança: cresce com número de pontos
        n_pts = len(self._auc_history)
        confidence = min(1.0, n_pts / 30.0) * (0.5 + 0.5 * (1.0 if not unstable else 0.3))

        return ConvergenceState(
            epoch=epoch,
            predicted_final_auc=pred_final,
            predicted_best_auc=pred_best,
            convergence_rate=rate,
            is_converging=rate > self.min_improvement_delta,
            is_unstable=unstable,
            plateau_epochs=self._plateau_counter,
            recommended_patience=patience,
            confidence=confidence,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Modelo de predição
    # ──────────────────────────────────────────────────────────────────────

    def _update_prediction_model(self):
        """
        Ajusta modelo AUC(t) = A × (1 - e^(-t/τ)) + 0.50

        Com pelo menos 3 pontos, estima A e τ por método de 2 pontos
        (para manter simplicidade e robustez numérica).
        """
        n = len(self._auc_history)
        if n < 3:
            # Usar estimativa histórica
            self._auc_max_est = self._profile.mean_auc_best
            self._tau_est = self.total_epochs * 0.3
            return

        # Usar mediana dos primeiros e últimas metades para robustez
        first_half = self._auc_history[:max(2, n // 3)]
        last_half = self._auc_history[-(n // 3 + 1):]

        t1 = statistics.mean([pt[0] for pt in first_half])
        a1 = statistics.median([pt[1] for pt in first_half])
        t2 = statistics.mean([pt[0] for pt in last_half])
        a2 = statistics.median([pt[1] for pt in last_half])

        # Clamp para evitar log de negativo
        floor = 0.5
        if a2 >= 1.0 or a1 >= a2 - 1e-6 or t1 >= t2:
            self._auc_max_est = max(a2 * 1.02, self._profile.mean_auc_best)
            self._tau_est = self._tau_est or self.total_epochs * 0.3
            return

        # Estimar A (AUC_max - floor): AUC(t) - floor = A × (1-e^{-t/τ})
        try:
            A = self._auc_max_est or (self._profile.mean_auc_best - floor)
            if A <= 0:
                A = 0.5
            r1 = (a1 - floor) / A if A > 0 else 0.5
            r2 = (a2 - floor) / A if A > 0 else 0.8
            r1 = max(1e-6, min(1 - 1e-6, r1))
            r2 = max(1e-6, min(1 - 1e-6, r2))
            # -t/τ = log(1 - r)  →  τ = -Δt / (log(1-r2) - log(1-r1))
            tau_new = -(t2 - t1) / (math.log(1 - r2) - math.log(1 - r1) + 1e-12)
            if 0 < tau_new < self.total_epochs * 3:
                self._tau_est = 0.7 * (self._tau_est or tau_new) + 0.3 * tau_new
            # Estimativa de A pelo best observado
            self._auc_max_est = max(self._best_auc_seen * 1.01, self._profile.mean_auc_best)
        except (ValueError, ZeroDivisionError):
            self._tau_est = self._tau_est or self.total_epochs * 0.3
            self._auc_max_est = self._auc_max_est or self._profile.mean_auc_best

    def _predict_final_auc(self, current_epoch: int) -> Tuple[float, float]:
        """
        Prediz AUC na época final e melhor AUC esperado.

        Returns: (predicted_final, predicted_best)
        """
        if self._auc_max_est is None or self._tau_est is None:
            return self._profile.mean_auc, self._profile.mean_auc_best

        floor = 0.5
        A = max(0.0, self._auc_max_est - floor)
        tau = max(1.0, self._tau_est)

        # Final = AUC na última época
        auc_at_final = floor + A * (1 - math.exp(-self.total_epochs / tau))
        # Best = assintota do modelo (AUC_max)
        auc_best = self._auc_max_est

        # Clamp razoável
        auc_at_final = max(self._best_auc_seen * 0.95, min(1.0, auc_at_final))
        auc_best = max(self._best_auc_seen, min(1.0, auc_best))

        return auc_at_final, auc_best

    def _compute_convergence_rate(self) -> float:
        """Taxa média de melhora do AUC nas últimas `rate_window` épocas."""
        n = len(self._auc_history)
        if n < 2:
            return 0.0
        window = min(self.rate_window, n)
        recent = self._auc_history[-window:]
        if len(recent) < 2:
            return 0.0
        delta_auc = recent[-1][1] - recent[0][1]
        delta_epoch = recent[-1][0] - recent[0][0]
        if delta_epoch <= 0:
            return 0.0
        return max(0.0, delta_auc / delta_epoch)

    # ──────────────────────────────────────────────────────────────────────
    #  Detecção de instabilidade
    # ──────────────────────────────────────────────────────────────────────

    def _detect_instability(self, epoch: int) -> bool:
        """
        Detecta instabilidade como alta variância do AUC na janela recente.
        
        Usa coeficiente de variação (CV): std / mean.
        CV > instability_cv_threshold → instável.
        """
        n = len(self._auc_history)
        if n < self.instability_window:
            return False
        window_aucs = [v for _, v in self._auc_history[-self.instability_window:]]
        if len(window_aucs) < 3:
            return False
        mean_auc = statistics.mean(window_aucs)
        if mean_auc < 1e-6:
            return False
        std_auc = statistics.stdev(window_aucs)
        cv = std_auc / mean_auc
        if cv > self.instability_cv_threshold:
            event = InstabilityEvent(
                epoch=epoch,
                metric="auc",
                severity=cv / self.instability_cv_threshold,
                description=(
                    f"CV do AUC = {cv:.4f} (threshold={self.instability_cv_threshold}) "
                    f"em {self.instability_window} épocas"
                ),
            )
            # Adicionar apenas se não há evento recente
            if not self._instability_events or self._instability_events[-1].epoch < epoch - 5:
                self._instability_events.append(event)
            return True

        # Verificar também oscillação na loss
        n_loss = len(self._loss_history)
        if n_loss >= self.instability_window:
            window_losses = [v for _, v in self._loss_history[-self.instability_window:]]
            mean_loss = statistics.mean(window_losses)
            std_loss = statistics.stdev(window_losses)
            if mean_loss > 1e-6 and std_loss / mean_loss > self.instability_cv_threshold * 2:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    #  Paciência adaptativa
    # ──────────────────────────────────────────────────────────────────────

    def _adaptive_patience(self, epoch: int, rate: float, unstable: bool) -> int:
        """
        Calcula paciência adaptativa para LR reduction.

        Regras:
        - Instabilidade → aumentar paciência (evitar reduzir LR quando oscilando)
        - Taxa de melhora alta → aumentar paciência (deixar melhorar)
        - Taxa baixa + plateau longo → reduzir paciência (agir mais cedo)
        - Calibrado pelo estimate histórico da variante
        """
        base = self._hist_estimate.suggested_plateau_patience

        # Fração de progresso
        progress_frac = epoch / max(1, self.total_epochs)

        # Fase inicial: mais paciência
        if progress_frac < 0.2:
            patience = base + 4

        # Fase de progresso rápido: deixar correr
        elif rate > 0.003:
            patience = base + 3

        # Fase de progresso lento: paciência base
        elif rate > 0.0005:
            patience = base

        # Fase quase estagnada: reduzir um pouco para acelerar ajuste
        else:
            patience = max(5, base - 2)

        # Instabilidade: nunca reduzir LR no meio de oscilação
        if unstable:
            patience = max(patience, base + 5)

        # No final (progresso > 80%), paciência menor (queremos ajustes finos)
        if progress_frac > 0.80:
            patience = max(5, patience - 2)

        return patience

    # ──────────────────────────────────────────────────────────────────────
    #  Early stopping inteligente
    # ──────────────────────────────────────────────────────────────────────

    def should_early_stop(
        self,
        min_epochs: int = 40,
        target_auc: float = 0.995,
    ) -> Tuple[bool, str]:
        """
        Avalia se é seguro parar o treino mais cedo baseado em:
        1. AUC atual ≥ target_auc (excelente resultado)
        2. Predição de AUC final não vai melhorar significativamente
        3. Mínimo de épocas atingido

        Returns:
            (should_stop: bool, reason: str)
        """
        epoch = self._auc_history[-1][0] if self._auc_history else 0
        n = len(self._auc_history)

        if epoch < min_epochs or n < 15:
            return False, ""

        current_auc = self._best_auc_seen
        pred_final, pred_best = self._predict_final_auc(epoch)

        # Caso 1: já atingiu target excelente
        if current_auc >= target_auc:
            return True, f"AUC atual {current_auc:.4f} ≥ target {target_auc}"

        # Caso 2: predição sugere que não vai melhorar muito mais
        remaining_epochs = self.total_epochs - epoch
        remaining_improvement = pred_best - current_auc
        if remaining_epochs < 20 and remaining_improvement < 0.002:
            return True, (
                f"Predição indica melhoria residual {remaining_improvement:.4f} "
                f"em {remaining_epochs} épocas restantes"
            )

        return False, ""

    # ──────────────────────────────────────────────────────────────────────
    #  Propriedades e utilitários
    # ──────────────────────────────────────────────────────────────────────

    @property
    def best_auc_seen(self) -> float:
        return self._best_auc_seen

    @property
    def plateau_counter(self) -> int:
        return self._plateau_counter

    @property
    def current_patience(self) -> int:
        return self._plateau_patience

    @property
    def instability_events(self) -> List[InstabilityEvent]:
        return list(self._instability_events)

    def get_recent_auc_trend(self, window: int = 10) -> str:
        """Retorna descrição textual da tendência recente do AUC."""
        rate = self._compute_convergence_rate()
        if rate > 0.003:
            return "improving_fast"
        if rate > 0.0005:
            return "improving_slow"
        if abs(rate) <= self.min_improvement_delta:
            return "plateau"
        return "degrading" if rate < -1e-4 else "stable"

    def summary(self) -> str:
        """Relatório compacto do tracker."""
        n = len(self._auc_history)
        if n == 0:
            return "[ConvergenceTracker] Sem dados."
        epoch, last_auc = self._auc_history[-1]
        rate = self._compute_convergence_rate()
        pred_final, pred_best = self._predict_final_auc(epoch)
        lines = [
            f"[ConvergenceTracker] {self.stack}_{self.mode} @ {epoch}/{self.total_epochs} épocas",
            f"  AUC atual: {last_auc:.4f} (best={self._best_auc_seen:.4f})",
            f"  Predição final: {pred_final:.4f} | best: {pred_best:.4f}",
            f"  Taxa: {rate:.5f}/epoch | tendência: {self.get_recent_auc_trend()}",
            f"  Plateau: {self._plateau_counter} épocas | paciência: {self._plateau_patience}",
            f"  Instabilidade: {len(self._instability_events)} evento(s)",
            f"  τ={self._tau_est:.1f} | AUC_max_est={self._auc_max_est:.4f}" if self._tau_est else "  Modelo: aguardando dados",
        ]
        return "\n".join(lines)

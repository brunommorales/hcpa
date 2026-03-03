"""
controller.py — Controller/Tuner agnóstico de framework.

Aplica ajustes SOMENTE dentro do Espaço de Configuração Derivado,
guiado por telemetria GPU, sinais do treino, conhecimento offline
e scoring multi-objetivo.

Módulos integrados (v2):
  - offline_knowledge : warm-start baseado em resultados históricos
  - multi_objective   : scoring composto AUC × throughput × memória
  - convergence_tracker: predição de convergência e paciência adaptativa

Cada ajuste é logado com razão explícita (evento/sinal).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .derived_space import DerivedConfigSpace, ParamSpec, get_initial_config
from .safety import HealthSignal, SafetyManager
from .gpu_monitor import GPUMonitor, GPUSnapshot
from .offline_knowledge import OfflineKnowledgeBase, get_knowledge_base, WarmStartRecommendation
from .multi_objective import MultiObjectiveScorer, ObjectiveWeights, CompositeScore
from .convergence_tracker import ConvergenceTracker, ConvergenceState


@dataclass
class TuningAction:
    """Registro de um ajuste aplicado pelo controller."""
    epoch: int
    param_name: str
    old_value: Any
    new_value: Any
    reason: str  # evento/sinal que motivou o ajuste
    timestamp: float = 0.0

    def __str__(self):
        return (
            f"[Tuning E{self.epoch}] {self.param_name}: {self.old_value} -> {self.new_value} "
            f"| razão: {self.reason}"
        )


class AutoTuneController:
    """
    Controller principal de autoajuste online (v2 com conhecimento offline).

    Regras invioláveis:
    - Só ajusta parâmetros presentes no DerivedConfigSpace.
    - Cada ajuste é logado com razão explícita.
    - Possui rollback para última configuração estável.
    - Respeita os limites (range/choices) definidos no espaço.

    Novidades v2:
    - Integra OfflineKnowledgeBase para paciência adaptativa e warm-start
    - Integra MultiObjectiveScorer para sinalização de eficiência
    - Integra ConvergenceTracker para predição de AUC final
    - Políticas de ajuste estendidas:
      6. GradientAccumulation: sugere acumulação quando OOM reduz batch
      7. WeightDecay:          aumenta weight_decay se overfitting detectado
      8. Multi-objetivo:       ajusta baseado em score composto em plateau
    """

    def __init__(
        self,
        space: DerivedConfigSpace,
        safety: SafetyManager,
        gpu_monitor: Optional[GPUMonitor] = None,
        enable_tuning: bool = True,
        # Stack e modo para integração com knowledge base
        stack: str = "",
        mode: str = "",
        gpu_name: str = "",
        total_epochs: int = 200,
        # Pesos multi-objetivo (None = usar padrão balanced)
        objective_weights: Optional[ObjectiveWeights] = None,
        # Thresholds para decisão (sobrescritos dinamicamente pelo offline KB)
        lr_reduction_factor: float = 0.5,
        lr_increase_factor: float = 1.2,
        mem_pressure_threshold_pct: float = 90.0,
        underutilization_threshold_pct: float = 30.0,
        plateau_patience: int = 8,
        plateau_min_delta: float = 1e-4,
        min_tune_epoch: int = 3,
        lr_cooldown_epochs: int = 2,
        min_lr_floor: float = 1e-5,
        # GPU compute capability para adaptar thresholds (ex: "8.9" para RTX 4070)
        gpu_compute_capability: Optional[str] = None,
        # VRAM total em MB para adaptar thresholds automaticamente
        gpu_vram_mb: Optional[float] = None,
    ):
        self.space = space
        self.safety = safety
        self.gpu_monitor = gpu_monitor
        self.enable_tuning = enable_tuning
        self.stack = stack
        self.mode = mode
        self.gpu_name = gpu_name
        self.total_epochs = total_epochs
        self.gpu_compute_capability = gpu_compute_capability
        self.gpu_vram_mb = gpu_vram_mb

        # ── Adaptar thresholds com base no hardware real ──
        # GPUs com ≤16 GB VRAM operam mais próximo do limite → threshold mais conservador.
        # GPUs pré-Ampere (CC < 8.0) não suportam TF32, logo podem usar mais memória.
        _effective_mem_threshold = mem_pressure_threshold_pct
        if gpu_vram_mb is not None and gpu_vram_mb <= 16384.0:
            # Para GPUs com ≤16 GB, usar 82% em vez de 90% para ter margem de escape
            _effective_mem_threshold = min(mem_pressure_threshold_pct, 82.0)
        if gpu_vram_mb is not None and gpu_vram_mb <= 12288.0:
            # Para GPUs ≤12 GB (RTX 4070, RTX 3060, etc.), ser ainda mais conservador
            _effective_mem_threshold = min(_effective_mem_threshold, 78.0)
        try:
            cc_float = float(gpu_compute_capability) if gpu_compute_capability else None
        except (ValueError, TypeError):
            cc_float = None
        # Ampere+ (CC >= 8.0): TF32 é padrão em PyTorch—usa menos memória que FP32
        # A100/H100 (CC >= 8.0) também têm BF16 nativo
        self._has_tf32 = cc_float is not None and cc_float >= 8.0
        self._has_bf16 = cc_float is not None and cc_float >= 8.0
        self._has_flash_attn = cc_float is not None and cc_float >= 8.0

        # Thresholds base (podem ser adaptativamente sobrescritos)
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_increase_factor = lr_increase_factor
        self.mem_pressure_threshold_pct = _effective_mem_threshold
        self.underutilization_threshold_pct = underutilization_threshold_pct
        self.plateau_min_delta = plateau_min_delta
        self.min_tune_epoch = min_tune_epoch
        self.lr_cooldown_epochs = lr_cooldown_epochs
        self.min_lr_floor = min_lr_floor

        # ── Módulos de conhecimento offline e scoring (v2) ──
        self._kb: Optional[OfflineKnowledgeBase] = None
        self._mo_scorer: Optional[MultiObjectiveScorer] = None
        self._conv_tracker: Optional[ConvergenceTracker] = None
        self._warm_start_rec: Optional[WarmStartRecommendation] = None

        if stack and mode:
            try:
                self._kb = get_knowledge_base()
                # Obter paciência adaptativa do knowledge base
                conv_est = self._kb.estimate_convergence(stack, mode, gpu_name, total_epochs)
                plateau_patience = conv_est.suggested_plateau_patience

                self._mo_scorer = MultiObjectiveScorer(
                    stack=stack,
                    mode=mode,
                    gpu_name=gpu_name,
                    weights=objective_weights,
                    kb=self._kb,
                )
                self._mo_scorer.set_total_epochs(total_epochs)

                self._conv_tracker = ConvergenceTracker(
                    stack=stack,
                    mode=mode,
                    gpu_name=gpu_name,
                    total_epochs=total_epochs,
                    kb=self._kb,
                )
            except Exception as exc:
                print(f"[Controller] Aviso: falha ao inicializar módulos v2: {exc}")

        self.plateau_patience = plateau_patience

        # Estado
        self._current_config: Dict[str, Any] = {}
        self._action_log: List[TuningAction] = []
        self._epoch_metrics: List[Dict[str, Any]] = []
        self._plateau_counter = 0
        self._best_val_auc = -1.0
        self._initialized = False
        self._last_epoch = -1
        self._lr_cooldown_until = -1

        # Estado v2
        self._latest_composite_score: Optional[CompositeScore] = None
        self._latest_convergence_state: Optional[ConvergenceState] = None
        self._mo_cooldown_until = -1        # cooldown para ações multi-objetivo
        self._wd_increased = False          # marcador para weight_decay já aumentado
        self._ga_applied = False            # marcador para gradient_accumulation

    def initialize(self, mode: str):
        """
        Inicializa config com valores base ou opt.

        Se knowledge base disponível, aplica warm-start baseado em
        resultados históricos para a variante × GPU.
        """
        self._current_config = get_initial_config(self.space, mode)

        # Tentar aplicar warm-start do knowledge base
        if self._kb and self.stack and self.gpu_name:
            try:
                gpu_mem = None
                if self.gpu_monitor and hasattr(self.gpu_monitor, 'latest') and self.gpu_monitor.latest:
                    gpu_mem = self.gpu_monitor.latest.memory_total_mb
                self._warm_start_rec = self._kb.get_warm_start(
                    stack=self.stack,
                    mode=self.mode,
                    gpu_name=self.gpu_name,
                    gpu_mem_mb=gpu_mem,
                    current_config=self._current_config,
                )
                # Aplicar overrides de warm-start apenas para parâmetros no espaço derivado
                applied = []
                for k, v in self._warm_start_rec.config_overrides.items():
                    if k in self.space.params:
                        self._current_config[k] = v
                        applied.append(f"{k}={v}")
                if applied:
                    print(f"[Controller] Warm-start aplicado ({len(applied)} params): "
                          f"{', '.join(applied)}")
                print(f"[Controller] {self._warm_start_rec.summary()}")
            except Exception as exc:
                print(f"[Controller] Aviso: warm-start falhou: {exc}")

        self._initialized = True

    def set_config(self, config: Dict[str, Any]):
        """Define config manualmente (para retomar de checkpoint)."""
        self._current_config = copy.deepcopy(config)
        self._initialized = True

    @property
    def config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._current_config)

    @property
    def action_log(self) -> List[TuningAction]:
        return list(self._action_log)

    def _validate_adjustment(self, param_name: str, new_value: Any) -> bool:
        """Verifica se o ajuste é válido dentro do espaço derivado."""
        spec = self.space.get(param_name)
        if spec is None:
            return False  # parâmetro não está no espaço derivado
        if not spec.tunable_online:
            return False  # não ajustável online
        if spec.param_type in ("int", "float") and spec.range is not None:
            if float(new_value) < spec.range_min or float(new_value) > spec.range_max:
                return False
        if spec.choices is not None and new_value not in spec.choices:
            return False
        return True

    def _apply_adjustment(self, epoch: int, param_name: str, new_value: Any, reason: str) -> bool:
        """Aplica um ajuste se válido e loga."""
        if not self._validate_adjustment(param_name, new_value):
            return False
        old_value = self._current_config.get(param_name)
        if old_value == new_value:
            return False  # sem mudança
        self._current_config[param_name] = new_value
        action = TuningAction(
            epoch=epoch,
            param_name=param_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            timestamp=time.time(),
        )
        self._action_log.append(action)
        return True

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_auc: float,
        throughput: float,
        gpu_mem_used_mb: float = 0.0,
        gpu_mem_total_mb: float = 0.0,
    ) -> List[TuningAction]:
        """
        Chamado no fim de cada época. Avalia sinais e aplica ajustes.

        v2: integra multi-objetivo e convergência adaptativa.

        Returns:
            Lista de ações tomadas nesta época.
        """
        if not self.enable_tuning or not self._initialized:
            return []

        if epoch <= self._last_epoch:
            return []
        self._last_epoch = epoch

        # Coletar telemetria GPU
        gpu_snap = self.gpu_monitor.latest if self.gpu_monitor else None
        mem_used = gpu_snap.memory_used_mb if gpu_snap else gpu_mem_used_mb or None
        mem_total = gpu_snap.memory_total_mb if gpu_snap else gpu_mem_total_mb or None

        # Registrar sinal de saúde
        signal = HealthSignal(
            epoch=epoch,
            loss=train_loss,
            val_loss=val_loss,
            val_auc=val_auc,
            gpu_mem_used_mb=mem_used,
            gpu_mem_total_mb=mem_total,
            throughput_img_s=throughput,
            is_nan=math.isnan(train_loss) or math.isnan(val_loss),
            is_inf=math.isinf(train_loss) or math.isinf(val_loss),
        )
        self.safety.record_signal(signal)

        # Registrar métricas
        self._epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "throughput": throughput,
            "gpu_mem_used_mb": mem_used,
        })

        actions_this_epoch: List[TuningAction] = []

        # ═══════════════════════════════════════════════════════════════════
        # v2: Atualizar módulos de scoring e convergência
        # ═══════════════════════════════════════════════════════════════════
        mo_signal = None
        conv_state = None
        adaptive_lr_factor = None

        if self._conv_tracker and not math.isnan(val_auc) and not math.isnan(train_loss):
            try:
                conv_state = self._conv_tracker.record(epoch, val_auc, train_loss)
                self._latest_convergence_state = conv_state
                # Adaptar paciência dinamicamente conforme tracker
                self.plateau_patience = conv_state.recommended_patience
            except Exception:
                pass

        if self._mo_scorer and not math.isnan(val_auc):
            try:
                mem_u = float(mem_used or 0)
                mem_t = float(mem_total or 0)
                cs = self._mo_scorer.score_epoch(
                    epoch=epoch,
                    val_auc=val_auc,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    throughput_img_s=throughput,
                    gpu_mem_used_mb=mem_u,
                    gpu_mem_total_mb=mem_t,
                )
                self._latest_composite_score = cs
                mo_signal = self._mo_scorer.generate_signal(cs, self.total_epochs)
                adaptive_lr_factor = self._mo_scorer.get_adaptive_lr_factor()
            except Exception:
                pass

        # ─── ROLLBACK se necessário ───
        if self.safety.should_rollback():
            rollback_cfg = self.safety.get_rollback_config()
            if rollback_cfg:
                for key, val in rollback_cfg.items():
                    spec = self.space.get(key)
                    if spec and spec.tunable_online and self._current_config.get(key) != val:
                        if self._apply_adjustment(epoch, key, val,
                                                   reason="ROLLBACK: NaN/divergência detectada"):
                            actions_this_epoch.append(self._action_log[-1])
                return actions_this_epoch

        # ─── Atualizar checkpoint estável se melhorou ───
        if not math.isnan(val_auc) and val_auc > self._best_val_auc + self.plateau_min_delta:
            self._best_val_auc = val_auc
            self._plateau_counter = 0
            self.safety.update_stable_checkpoint(
                config=self._current_config,
                epoch=epoch,
                val_auc=val_auc,
                val_loss=val_loss,
            )
        else:
            self._plateau_counter += 1

        # ═══════════════════════════════════════════════════════════════════
        # AJUSTES BASEADOS EM SINAIS (5 políticas originais + 3 novas v2)
        # ═══════════════════════════════════════════════════════════════════

        # ─── 1. Pressão de memória → reduzir batch_size ───
        if self.safety.detect_oom_risk():
            bs_spec = self.space.get("batch_size")
            if bs_spec and bs_spec.tunable_online:
                current_bs = self._current_config.get("batch_size", 32)
                new_bs = max(int(bs_spec.range_min or 8), current_bs // 2)
                if self._apply_adjustment(epoch, "batch_size", new_bs,
                                           reason=f"OOM_RISK: GPU mem >{self.mem_pressure_threshold_pct}%"):
                    actions_this_epoch.append(self._action_log[-1])

        # ─── 2. Plateau de AUC → reduzir LR (paciência adaptativa v2) ───
        lr_key = self._get_lr_key()
        effective_patience = (
            conv_state.recommended_patience if conv_state else self.plateau_patience
        )
        if (
            self._plateau_counter >= effective_patience
            and epoch >= self.min_tune_epoch
            and epoch > self._lr_cooldown_until
            and lr_key
        ):
            lr_spec = self.space.get(lr_key)
            if lr_spec and lr_spec.tunable_online:
                current_lr = float(self._current_config.get(lr_key, 1e-4))
                min_allowed = max(float(lr_spec.range_min or 0.0), self.min_lr_floor)
                # Usar fator adaptativo do multi-objetivo se disponível
                factor = adaptive_lr_factor if adaptive_lr_factor and adaptive_lr_factor < 1.0 else self.lr_reduction_factor
                new_lr = max(min_allowed, current_lr * factor)
                reason_detail = (
                    f"plateau={self._plateau_counter} épocas, "
                    f"patience={effective_patience}, "
                    f"fator={'mo_adaptive' if adaptive_lr_factor else 'fixo'}={factor:.2f}"
                )
                if self._apply_adjustment(epoch, lr_key, new_lr,
                                           reason=f"PLATEAU_LR: {reason_detail}"):
                    actions_this_epoch.append(self._action_log[-1])
                    self._plateau_counter = 0
                    self._lr_cooldown_until = epoch + self.lr_cooldown_epochs

        # ─── 2b. LR já no floor + estagnação longa → solicitar rollback ───────────
        # Quando o LR não pode mais ser reduzido (já está no mínimo) e a AUC
        # continua plana por muito tempo, não há mais ajuste possível dentro do
        # espaço atual. Melhor voltar ao checkpoint estável e tentar nova configuração.
        if lr_key and not signal.is_nan and not signal.is_inf:
            _cur_lr_floor = float(self._current_config.get(lr_key, 1.0))
            if (
                _cur_lr_floor <= self.min_lr_floor * 2.0    # LR próximo ou abaixo do floor
                and self._plateau_counter >= effective_patience * 2   # estagnação dupla
                and self._best_val_auc > 0.5                            # já teve um checkpoint bom
                and epoch >= self.min_tune_epoch + 10
                and epoch > self._lr_cooldown_until
            ):
                self.safety.request_rollback("lr_floor_stagnation")
                print(
                    f"  [Controller] LR no floor ({_cur_lr_floor:.2e} ≤ "
                    f"{self.min_lr_floor * 2.0:.2e}) com plateau de "
                    f"{self._plateau_counter} épocas → rollback solicitado para época estável."
                )

        # ─── 3. GPU subutilizada → ativar mixup/augmentation ───
        if gpu_snap and gpu_snap.utilization_gpu_pct < self.underutilization_threshold_pct:
            mixup_spec = self.space.get("mixup_alpha")
            if mixup_spec and mixup_spec.tunable_online:
                current_mixup = float(self._current_config.get("mixup_alpha", 0.0))
                if current_mixup <= 0.0 and float(mixup_spec.opt_value or 0.0) > 0.0:
                    new_mixup = float(mixup_spec.opt_value)
                    if self._apply_adjustment(epoch, "mixup_alpha", new_mixup,
                                               reason=f"UNDERUTIL: GPU util {gpu_snap.utilization_gpu_pct:.0f}%"):
                        actions_this_epoch.append(self._action_log[-1])

        # ─── 4. Loss spike → aumentar label_smoothing ───
        if len(self._epoch_metrics) >= 2:
            prev_loss = self._epoch_metrics[-2]["train_loss"]
            curr_loss = self._epoch_metrics[-1]["train_loss"]
            if curr_loss > prev_loss * 2.0 and not signal.is_nan:
                ls_spec = self.space.get("label_smoothing")
                if ls_spec and ls_spec.tunable_online:
                    current_ls = float(self._current_config.get("label_smoothing", 0.0))
                    if current_ls < 0.1:
                        new_ls = min(float(ls_spec.range_max or 0.2), current_ls + 0.02)
                        if self._apply_adjustment(epoch, "label_smoothing", new_ls,
                                                   reason=f"LOSS_SPIKE: {prev_loss:.4f}->{curr_loss:.4f}"):
                            actions_this_epoch.append(self._action_log[-1])

        # ─── 5. Gradient clipping — reduzir se divergência ───
        clip_spec = self.space.get("clip_grad_norm") or self.space.get("grad_clip_norm")
        if clip_spec and clip_spec.tunable_online and signal.is_diverging and not signal.is_nan:
            clip_key = clip_spec.name
            current_clip = float(self._current_config.get(clip_key, 1.0))
            new_clip = max(0.1, current_clip * 0.5)
            if self._apply_adjustment(epoch, clip_key, new_clip,
                                       reason="DIVERGENCE: reduzindo clip_grad_norm"):
                actions_this_epoch.append(self._action_log[-1])

        # ═══════════════════════════════════════════════════════════════════
        # POLÍTICAS v2 — Derivadas de conhecimento offline e multi-objetivo
        # ═══════════════════════════════════════════════════════════════════

        # ─── 6. Gradient Accumulation — quando OOM forçou BS pequeno ───
        # Compensa batch menor com gradient_accumulation para manter
        # batch efetivo próximo ao histórico ótimo (96)
        if not self._ga_applied:
            ga_spec = self.space.get("gradient_accumulation")
            if ga_spec and ga_spec.tunable_online:
                current_bs = int(self._current_config.get("batch_size", 96))
                current_thr = throughput
                ga_suggested = None
                if self._mo_scorer:
                    ga_suggested = self._mo_scorer.get_gradient_accumulation_suggestion(
                        current_bs, current_thr
                    )
                if ga_suggested and ga_suggested > 1:
                    max_ga = int(ga_spec.range_max or 8)
                    new_ga = min(ga_suggested, max_ga)
                    if self._apply_adjustment(epoch, "gradient_accumulation", new_ga,
                                               reason=f"SMALL_BATCH: BS={current_bs}, GA={new_ga} "
                                               f"→ effective_batch={current_bs * new_ga}"):
                        actions_this_epoch.append(self._action_log[-1])
                        self._ga_applied = True

        # ─── 7. Weight Decay — aumentar se overfitting detectado ───
        # Sinal: val_loss crescendo enquanto train_loss decresce (overfitting)
        if (
            not self._wd_increased
            and len(self._epoch_metrics) >= 6
            and epoch >= self.min_tune_epoch + 5
        ):
            recent = self._epoch_metrics[-6:]
            train_losses = [m["train_loss"] for m in recent]
            val_losses = [m["val_loss"] for m in recent if m.get("val_loss") is not None]
            if (len(val_losses) >= 4
                and train_losses[-1] < train_losses[0] * 0.90    # train melhorou ≥10%
                and val_losses[-1] > val_losses[0] + 0.02         # val piorou
                and not math.isnan(val_losses[-1])):
                wd_spec = self.space.get("weight_decay")
                if wd_spec and wd_spec.tunable_online:
                    current_wd = float(self._current_config.get("weight_decay", 1e-4))
                    max_wd = float(wd_spec.range_max or 1e-2)
                    new_wd = min(max_wd, current_wd * 2.0)
                    if self._apply_adjustment(
                        epoch, "weight_decay", new_wd,
                        reason=f"OVERFIT: train_loss↓ ({train_losses[0]:.4f}→{train_losses[-1]:.4f}) "
                               f"val_loss↑ ({val_losses[0]:.4f}→{val_losses[-1]:.4f})"
                    ):
                        actions_this_epoch.append(self._action_log[-1])
                        self._wd_increased = True

        # ─── 8. Multi-objetivo: ação baseada em sinal composto ───
        # Emitir aviso/ação quando o scorer multi-objetivo detecta anomalia
        # Não sobrepor ajustes já feitos acima (verificar cooldown)
        if (
            mo_signal
            and mo_signal.has_action()
            and mo_signal.priority >= 1
            and epoch > self._mo_cooldown_until
            and not actions_this_epoch   # não empilhar ações no mesmo epoch
        ):
            # AUC muito abaixo do esperado → reduzir LR pela metade
            if mo_signal.auc_lagging and lr_key and epoch > 20:
                lr_spec = self.space.get(lr_key)
                if lr_spec and lr_spec.tunable_online and epoch > self._lr_cooldown_until:
                    current_lr = float(self._current_config.get(lr_key, 1e-4))
                    min_allowed = max(float(lr_spec.range_min or 0.0), self.min_lr_floor)
                    new_lr = max(min_allowed, current_lr * 0.7)
                    if self._apply_adjustment(
                        epoch, lr_key, new_lr,
                        reason=f"MO_AUC_LAG: {mo_signal.suggested_action[:80]}"
                    ):
                        actions_this_epoch.append(self._action_log[-1])
                        self._mo_cooldown_until = epoch + self.lr_cooldown_epochs * 2
                        self._lr_cooldown_until = epoch + self.lr_cooldown_epochs

            # Throughput lagging → tentar ativar augmentation menor
            elif mo_signal.throughput_lagging and epoch > 10:
                aug_spec = self.space.get("augment")
                if aug_spec and aug_spec.tunable_online:
                    current_aug = self._current_config.get("augment", True)
                    if current_aug:  # já ativo; registrar aviso
                        print(f"  [MO] {mo_signal}")
                        self._mo_cooldown_until = epoch + 10  # cooldown sem ação

        return actions_this_epoch

    def _get_lr_key(self) -> Optional[str]:
        """Retorna a chave do LR no espaço derivado dependendo do stack."""
        for key in ("learning_rate", "lrate", "fine_tune_lr"):
            if key in self.space.params:
                return key
        return None

    def get_state(self) -> Dict[str, Any]:
        """Retorna estado serializável do controller para checkpoint (v2)."""
        state = {
            "current_config": copy.deepcopy(self._current_config),
            "best_val_auc": self._best_val_auc,
            "plateau_counter": self._plateau_counter,
            "plateau_patience": self.plateau_patience,
            "action_log": [
                {
                    "epoch": a.epoch,
                    "param_name": a.param_name,
                    "old_value": a.old_value,
                    "new_value": a.new_value,
                    "reason": a.reason,
                    "timestamp": a.timestamp,
                }
                for a in self._action_log
            ],
            "epoch_metrics_count": len(self._epoch_metrics),
            # v2 state
            "stack": self.stack,
            "mode": self.mode,
            "gpu_name": self.gpu_name,
            "total_epochs": self.total_epochs,
            "ga_applied": self._ga_applied,
            "wd_increased": self._wd_increased,
        }
        # Incluir score composto mais recente se disponível
        if self._latest_composite_score:
            cs = self._latest_composite_score
            state["latest_composite"] = {
                "epoch": cs.epoch,
                "composite": cs.composite,
                "auc_score": cs.auc_score,
                "throughput_score": cs.throughput_score,
                "memory_score": cs.memory_score,
            }
        if self._latest_convergence_state:
            cv = self._latest_convergence_state
            state["latest_convergence"] = {
                "epoch": cv.epoch,
                "predicted_final_auc": cv.predicted_final_auc,
                "predicted_best_auc": cv.predicted_best_auc,
                "phase": cv.phase,
                "plateau_epochs": cv.plateau_epochs,
            }
        return state

    def load_state(self, state: Dict[str, Any]):
        """Restaura estado do controller de checkpoint (v2)."""
        if "current_config" in state:
            self._current_config = state["current_config"]
            self._initialized = True
        self._best_val_auc = state.get("best_val_auc", -1.0)
        self._plateau_counter = state.get("plateau_counter", 0)
        if "plateau_patience" in state:
            self.plateau_patience = state["plateau_patience"]
        # v2
        self._ga_applied = state.get("ga_applied", False)
        self._wd_increased = state.get("wd_increased", False)

    # ──────────────────────────────────────────────────────────────────────
    #  Propriedades v2
    # ──────────────────────────────────────────────────────────────────────

    @property
    def latest_composite_score(self) -> Optional[CompositeScore]:
        """Score composto multi-objetivo mais recente."""
        return self._latest_composite_score

    @property
    def latest_convergence_state(self) -> Optional[ConvergenceState]:
        """Estado de convergência mais recente."""
        return self._latest_convergence_state

    @property
    def warm_start_recommendation(self) -> Optional[WarmStartRecommendation]:
        """Recomendação de warm-start usada na inicialização."""
        return self._warm_start_rec

    def get_knowledge_summary(self) -> str:
        """Relatório do conhecimento offline para a variante atual."""
        if self._kb and self.stack:
            return self._kb.summary_report(self.gpu_name)
        return "[OfflineKB] Não inicializado."

    def get_multi_objective_summary(self) -> str:
        """Relatório do scorer multi-objetivo."""
        if self._mo_scorer:
            return self._mo_scorer.summary()
        return "[MultiObjective] Não inicializado."

    def get_convergence_summary(self) -> str:
        """Relatório do tracker de convergência."""
        if self._conv_tracker:
            return self._conv_tracker.summary()
        return "[ConvergenceTracker] Não inicializado."

    def summary(self) -> str:
        lines = [f"[Controller v2] {len(self._action_log)} ações de tuning realizadas:"]
        for a in self._action_log:
            lines.append(f"  {a}")
        if self._mo_scorer:
            lines.append("")
            lines.append(self._mo_scorer.summary())
        if self._conv_tracker:
            lines.append("")
            lines.append(self._conv_tracker.summary())
        return "\n".join(lines)

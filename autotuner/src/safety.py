"""
safety.py — Mecanismos de segurança: rollback, checkpoint, detecção de OOM/NaN/divergência.

Estes são os ÚNICOS mecanismos "extras" permitidos (não derivados das variantes) -
usados exclusivamente para segurança e recuperação.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HealthSignal:
    """Sinal de saúde coletado a cada época/step."""
    epoch: int
    loss: float
    val_loss: Optional[float] = None
    val_auc: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_mem_total_mb: Optional[float] = None
    throughput_img_s: Optional[float] = None
    is_nan: bool = False
    is_inf: bool = False
    is_oom: bool = False

    @property
    def is_healthy(self) -> bool:
        return not self.is_nan and not self.is_inf and not self.is_oom

    @property
    def is_diverging(self) -> bool:
        """Perda muito grande ou crescente."""
        return self.loss > 100.0 or self.is_nan or self.is_inf


@dataclass
class StableCheckpoint:
    """Guarda a última configuração+estado estáveis para rollback."""
    config: Dict[str, Any]
    epoch: int
    val_auc: float
    val_loss: float
    model_state_path: Optional[str] = None  # path do checkpoint do modelo
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "epoch": self.epoch,
            "val_auc": self.val_auc,
            "val_loss": self.val_loss,
            "model_state_path": self.model_state_path,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StableCheckpoint":
        return cls(**d)


class SafetyManager:
    """Gerencia rollback, detecção de falhas e checkpoints do controller."""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_nan_tolerance: int = 3,
        max_loss_spike_ratio: float = 5.0,
        mem_headroom_pct: float = 10.0,
        min_epoch_for_rollback: int = 5,
        low_auc_floor: float = 0.50,
        min_auc_drop: float = 0.05,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_nan_tolerance = max_nan_tolerance
        self.max_loss_spike_ratio = max_loss_spike_ratio
        self.mem_headroom_pct = mem_headroom_pct
        # Épocas mínimas antes de permitir qualquer rollback.
        # Previne rollback prematuro durante o warmup inicial (LR warmup, batch norm,
        # etc.) quando a loss pode oscilar legitimamente.
        self.min_epoch_for_rollback = min_epoch_for_rollback
        self.low_auc_floor = low_auc_floor
        self.min_auc_drop = min_auc_drop

        self._history: List[HealthSignal] = []
        self._nan_count = 0
        self._best_checkpoint: Optional[StableCheckpoint] = None
        self._last_stable: Optional[StableCheckpoint] = None
        self._rollback_flag: bool = False
        self._rollback_reason: Optional[str] = None

    def record_signal(self, signal: HealthSignal):
        """Registra um sinal de saúde."""
        self._history.append(signal)
        if signal.is_nan or signal.is_inf:
            self._nan_count += 1
        else:
            self._nan_count = 0  # reset contagem consecutiva

    def should_rollback(self) -> bool:
        """Verifica se deve fazer rollback.

        Nunca aciona rollback antes de ``min_epoch_for_rollback`` épocas terem sido
        registradas — durante o warmup a loss pode oscilar legitimamente e um
        rollback prematuro seria mais prejudicial que útil.
        Mesmo após o período de warmup, NaNs acumulados (>= max_nan_tolerance)
        sempre justificam rollback (dados corrompidos não têm recuperação suave).
        """
        # Se alguém já solicitou rollback explicitamente, respeitar.
        if self._rollback_flag:
            return True

        n_epochs = len(self._history)
        past_warmup = n_epochs >= self.min_epoch_for_rollback

        latest = self._history[-1] if self._history else None
        reason: Optional[str] = None

        # NaN/Inf: rollback imediato, mesmo antes do warmup (melhor voltar ao último
        # estado saudável do que continuar acumulando NaN).
        if latest and (latest.is_nan or latest.is_inf):
            reason = "nan_or_inf"
        elif past_warmup and self._nan_count >= self.max_nan_tolerance:
            reason = "nan_accumulated"

        # Colapso de AUC: se cair para ~aleatório (<= low_auc_floor) e estiver pior
        # que o melhor checkpoint por min_auc_drop, recuar.
        if (
            reason is None
            and past_warmup
            and latest
            and latest.val_auc is not None
            and latest.val_auc <= self.low_auc_floor
            and self._best_checkpoint
            and (self._best_checkpoint.val_auc - latest.val_auc) >= self.min_auc_drop
        ):
            reason = "auc_collapse"

        # 3 épocas consecutivas divergindo → provável explosão de gradiente
        if reason is None and past_warmup and n_epochs >= 3:
            recent = self._history[-3:]
            if all(s.is_diverging for s in recent):
                reason = "divergence_3_epochs"

        # Spike isolado de loss × ratio — só após warmup E só se temos checkpoint saudável
        if reason is None and past_warmup and n_epochs >= 2 and self._last_stable is not None:
            prev = self._history[-2]
            curr = self._history[-1]
            if prev.loss > 0 and curr.loss > prev.loss * self.max_loss_spike_ratio:
                reason = "loss_spike"

        if reason:
            self._rollback_flag = True
            self._rollback_reason = reason
            return True
        return False

    def request_rollback(self, reason: str = "manual"):
        """Permite sinalizar rollback imediato fora do fluxo de análise."""
        self._rollback_flag = True
        self._rollback_reason = reason

    def detect_oom_risk(self) -> bool:
        """Detecta risco de OOM baseado na telemetria."""
        if not self._history:
            return False
        latest = self._history[-1]
        if latest.gpu_mem_used_mb and latest.gpu_mem_total_mb:
            usage_pct = (latest.gpu_mem_used_mb / latest.gpu_mem_total_mb) * 100
            return usage_pct > (100.0 - self.mem_headroom_pct)
        return False

    def update_stable_checkpoint(self, config: Dict[str, Any], epoch: int,
                                  val_auc: float, val_loss: float,
                                  model_state_path: Optional[str] = None):
        """Atualiza o checkpoint estável se a métrica melhorou."""
        ckpt = StableCheckpoint(
            config=copy.deepcopy(config),
            epoch=epoch,
            val_auc=val_auc,
            val_loss=val_loss,
            model_state_path=model_state_path,
            timestamp=time.time(),
        )
        self._last_stable = ckpt
        if self._best_checkpoint is None or val_auc > self._best_checkpoint.val_auc:
            self._best_checkpoint = ckpt

    def get_rollback_config(self) -> Optional[Dict[str, Any]]:
        """Retorna a última configuração estável para rollback."""
        if self._last_stable:
            return copy.deepcopy(self._last_stable.config)
        return None

    def get_best_checkpoint(self) -> Optional[StableCheckpoint]:
        return self._best_checkpoint

    def consume_rollback_flag(self) -> (bool, Optional[str]):
        """Retorna e limpa o flag de rollback pendente."""
        flag, reason = self._rollback_flag, self._rollback_reason
        self._rollback_flag = False
        self._rollback_reason = None
        return flag, reason

    def save_controller_state(self, controller_state: Dict[str, Any]):
        """Salva estado do controller em disco para retomada."""
        path = self.checkpoint_dir / "controller_state.json"
        data = {
            "controller_state": controller_state,
            "best_checkpoint": self._best_checkpoint.to_dict() if self._best_checkpoint else None,
            "last_stable": self._last_stable.to_dict() if self._last_stable else None,
            "history_len": len(self._history),
            "nan_count": self._nan_count,
        }
        # Serialize safely
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.rename(path)

    def load_controller_state(self) -> Optional[Dict[str, Any]]:
        """Carrega estado do controller de disco."""
        path = self.checkpoint_dir / "controller_state.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("best_checkpoint"):
                self._best_checkpoint = StableCheckpoint.from_dict(data["best_checkpoint"])
            if data.get("last_stable"):
                self._last_stable = StableCheckpoint.from_dict(data["last_stable"])
            self._nan_count = data.get("nan_count", 0)
            return data.get("controller_state")
        except Exception:
            return None

    @property
    def history(self) -> List[HealthSignal]:
        return list(self._history)

    def reset(self):
        self._history.clear()
        self._nan_count = 0

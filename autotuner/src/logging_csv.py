"""
logging_csv.py — Logging mínimo em CSV com telemetria GPU e configuração aplicada.

Campos adicionais v2 (multi-objective / convergence):
  composite_score       — score composto [0,1] ponderando AUC + throughput + memória
  auc_score_mo          — componente AUC normalizada [0,1]
  throughput_score_mo   — componente throughput normalizada [0,1]
  memory_score_mo       — componente eficiência de memória [0,1]
  convergence_phase     — fase atual: rapid_improvement / gradual / marginal / plateau / unstable
  predicted_final_auc   — AUC final prevista pelo modelo de saturação
  convergence_tau       — constante de tempo τ do modelo de saturação (épocas)
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


CSV_FIELDS = [
    "timestamp",
    "epoch",
    "stage",
    "train_loss",
    "train_auc",
    "train_sens",
    "train_spec",
    "train_throughput_img_s",
    "train_elapsed_s",
    "val_loss",
    "val_auc",
    "val_sens",
    "val_spec",
    "val_throughput_img_s",
    "val_elapsed_s",
    "lr",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_util_pct",
    "gpu_temp_c",
    "gpu_power_w",
    "tuning_actions",
    "config_snapshot",
    "total_train_time_s",
    # ── v2: multi-objective ──────────────────────────────────────────────────
    "composite_score",
    "auc_score_mo",
    "throughput_score_mo",
    "memory_score_mo",
    # ── v2: convergence ──────────────────────────────────────────────────────
    "convergence_phase",
    "predicted_final_auc",
    "convergence_tau",
]


class CSVLogger:
    """Logger em CSV com métricas de treino, telemetria GPU e config aplicada."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None

    def open(self):
        mode = "w"
        self._file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def log_epoch(
        self,
        epoch: int,
        stage: str,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        lr: float,
        gpu_snapshot: Optional[Dict[str, Any]] = None,
        tuning_actions: Optional[List[str]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        total_train_time_s: Optional[float] = None,
        # ── v2 multi-objective / convergence ───────────────────────────────
        composite_score_info: Optional[Dict[str, Any]] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
    ):
        """Registra uma época no CSV.

        Args:
            composite_score_info: dict com chaves opcionais:
                ``composite``, ``auc_score``, ``throughput_score``, ``memory_score``
                (retornado por MultiObjectiveScorer.score_epoch()).
            convergence_info: dict com chaves opcionais:
                ``phase``, ``predicted_final_auc``, ``tau``
                (retornado por ConvergenceTracker.record()).
        """
        if self._writer is None:
            self.open()

        gpu = gpu_snapshot or {}
        mo = composite_score_info or {}
        cv = convergence_info or {}

        row = {
            "timestamp": time.time(),
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_metrics.get("loss"),
            "train_auc": train_metrics.get("auc"),
            "train_sens": train_metrics.get("sensitivity"),
            "train_spec": train_metrics.get("specificity"),
            "train_throughput_img_s": train_metrics.get("throughput"),
            "train_elapsed_s": train_metrics.get("elapsed_s"),
            "val_loss": val_metrics.get("loss"),
            "val_auc": val_metrics.get("auc"),
            "val_sens": val_metrics.get("sensitivity"),
            "val_spec": val_metrics.get("specificity"),
            "val_throughput_img_s": val_metrics.get("throughput"),
            "val_elapsed_s": val_metrics.get("elapsed_s"),
            "lr": lr,
            "gpu_mem_used_mb": gpu.get("memory_used_mb"),
            "gpu_mem_total_mb": gpu.get("memory_total_mb"),
            "gpu_util_pct": gpu.get("utilization_gpu_pct"),
            "gpu_temp_c": gpu.get("temperature_c"),
            "gpu_power_w": gpu.get("power_draw_w"),
            "tuning_actions": "|".join(tuning_actions) if tuning_actions else "",
            "config_snapshot": str(config_snapshot) if config_snapshot else "",
            "total_train_time_s": total_train_time_s,
            # v2 fields
            "composite_score": mo.get("composite"),
            "auc_score_mo": mo.get("auc_score"),
            "throughput_score_mo": mo.get("throughput_score"),
            "memory_score_mo": mo.get("memory_score"),
            "convergence_phase": cv.get("phase"),
            "predicted_final_auc": cv.get("predicted_final_auc"),
            "convergence_tau": cv.get("tau"),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None

    def __del__(self):
        self.close()

"""
tensorflow_backend.py — Backend para variantes TensorFlow (tensorflow_base, tensorflow_opt).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BackendBase

_TENSORFLOW_BASE_ARGS = {
    "tfrec_dir", "dataset", "results", "exec", "img_sizes", "batch_size",
    "epochs", "lrate", "num_thresholds", "verbose", "model", "augment",
    "no-augment", "normalize", "cores", "log-gpu-mem", "no-log-gpu-mem",
    "warmup_epochs", "min_lr", "label_smoothing",
}

_TENSORFLOW_OPT_ARGS = _TENSORFLOW_BASE_ARGS | {
    "num_classes", "wait_epochs", "show_files", "cache_dir",
    "freeze_epochs", "fine_tune_lr_factor", "fine_tune_at", "fine_tune_lr",
    "scheduler", "grad_clip_norm", "mixup_alpha", "cutmix_alpha",
    "focal_gamma", "pos_weight", "fundus_crop_ratio",
    "freeze_bn", "no-freeze_bn", "fine_tune_schedule",
    "channels_last", "h2d_uint8", "tta_views",
    "mixed_precision", "no-mixed_precision",
    "use_dali", "dali_threads", "dali_layout", "dali_seed",
    "recompute_backbone", "jit_compile", "auc_target",
}


class TensorFlowBackend(BackendBase):
    STACK_NAME = "tensorflow"

    def __init__(self, variant_path: Path, mode: str):
        super().__init__(variant_path, mode)
        self._accepted_args = _TENSORFLOW_OPT_ARGS if mode == "opt" else _TENSORFLOW_BASE_ARGS

    def get_entry_point(self) -> str:
        return str(self.variant_path / "dr_hcpa_v2_2024.py")

    def build_command(self, config: Dict[str, Any]) -> List[str]:
        cmd = [sys.executable, self.get_entry_point()]
        cmd.extend(self.config_to_cli_args(config))
        return cmd

    def config_to_cli_args(self, config: Dict[str, Any]) -> List[str]:
        args: List[str] = []
        filtered = self._filter_applicable_config(config)
        for key, value in filtered.items():
            if isinstance(value, bool):
                if key == "augment":
                    args.append("--augment" if value else "--no-augment")
                elif key == "freeze_bn":
                    args.append("--freeze_bn" if value else "--no-freeze_bn")
                elif key == "mixed_precision":
                    args.append("--mixed_precision" if value else "--no-mixed_precision")
                elif key in ("channels_last", "h2d_uint8", "use_dali",
                             "recompute_backbone", "jit_compile"):
                    if value:
                        args.append(f"--{key}")
                elif key in ("log_gpu_mem",):
                    args.append("--log-gpu-mem" if value else "--no-log-gpu-mem")
                continue
            if value is None:
                continue
            cli_key = f"--{key}"
            args.extend([cli_key, str(value)])
        return args

    def _filter_applicable_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize key names for matching
        applicable = {}
        for k, v in config.items():
            norm_k = k.replace("-", "_")
            if norm_k in self._accepted_args or k in self._accepted_args:
                applicable[k] = v
        return applicable

    def parse_epoch_metrics(self, output: str) -> Dict[str, Any]:
        """Parseia a saída do Keras fit para métricas de época.

        Formatos esperados (rank prefix opcional):
          Epoch 3/200
          115/115 [====] - 9s 76ms/step - loss: 0.09 - AUC: 0.75 - val_loss: 0.07 - val_AUC: 0.92 - lr: 1.2e-04 - throughput_img_s: 1259.17
        """
        metrics: Dict[str, Any] = {}
        # Strip rank prefix (e.g. "0: ")
        line = re.sub(r"^\d+:\s*", "", output.strip())

        # Epoch header line: "Epoch N/M"
        epoch_match = re.match(r"Epoch\s+(\d+)/(\d+)", line)
        if epoch_match:
            metrics["epoch"] = int(epoch_match.group(1))
            metrics["total_epochs"] = int(epoch_match.group(2))
            return metrics

        # Keras progress bar line with metrics
        if re.match(r"\d+/\d+", line) and "loss:" in line:
            # Aceita números ou 'nan' para não perder épocas divergentes
            num = r"(?:nan|[\d.eE+-]+)"
            patterns = {
                "train_loss": rf"(?<![a-z_])loss:\s*({num})",
                "train_auc": rf"(?<![a-z_])AUC:\s*({num})",
                "val_loss": rf"val_loss:\s*({num})",
                "val_auc": rf"val_AUC:\s*({num})",
                "lr": rf"\blr:\s*({num})",
                "throughput": rf"throughput_img_s:\s*({num})",
                "epoch_time_sec": rf"epoch_time_sec:\s*({num})",
                "val_throughput": rf"val_throughput_img_s:\s*({num})",
                "gpu_mem_peak_mb": rf"gpu_mem_peak_mb:\s*({num})",
            }
            for key, pat in patterns.items():
                match = re.search(pat, line, flags=re.IGNORECASE)
                if match:
                    val_str = match.group(1)
                    metrics[key] = float("nan") if val_str.lower() == "nan" else float(val_str)
            # Mark as a metrics line (epoch will be injected by main.py)
            if metrics:
                metrics["_is_metrics_line"] = True
        return metrics

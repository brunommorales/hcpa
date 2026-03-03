"""
monai_backend.py — Backend para variantes MONAI (monai_base, monai_opt).

MONAI é tratado como backend próprio mesmo sendo torch-based,
porque o pipeline/treino (TrainConfig dataclass, DALI opcional, etc.) é diferente.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BackendBase

# Argumentos CLI aceitos pelo train.py de cada variante MONAI (extraídos da auditoria)
_MONAI_BASE_CLI_ARGS = {
    "results", "tfrec_dir", "image_size", "num_classes", "batch_size",
    "epochs", "learning_rate", "weight_decay", "model", "normalize",
    "fundus_crop_ratio", "augment", "no_augment", "mixup_alpha",
    "cutmix_alpha", "label_smoothing", "channels_last", "no_channels_last",
    "amp", "no_amp", "compile", "no_compile", "scheduler",
    "grad_clip_norm", "pos_weight", "ema_decay", "ema_on_cpu",
    "gradient_accumulation", "log_every", "num_workers", "seed",
    "use_fake_data",
}

_MONAI_OPT_CLI_ARGS = _MONAI_BASE_CLI_ARGS | {
    "min_lr", "warmup_epochs", "use_dali", "no_dali",
}


class MonaiBackend(BackendBase):
    STACK_NAME = "monai"

    def __init__(self, variant_path: Path, mode: str):
        super().__init__(variant_path, mode)
        self._accepted_args = _MONAI_OPT_CLI_ARGS if mode == "opt" else _MONAI_BASE_CLI_ARGS

    def get_entry_point(self) -> str:
        return str(self.variant_path / "train.py")

    def build_command(self, config: Dict[str, Any]) -> List[str]:
        cmd = [sys.executable, self.get_entry_point()]
        cmd.extend(self.config_to_cli_args(config))
        return cmd

    def config_to_cli_args(self, config: Dict[str, Any]) -> List[str]:
        args: List[str] = []
        filtered = self._filter_applicable_config(config)

        # Mapear chaves do espaço derivado para CLI do MONAI
        KEY_MAP = {
            "learning_rate": "learning_rate",
            "model_name": "model",
            "results_dir": "results",
        }

        for key, value in filtered.items():
            cli_key = KEY_MAP.get(key, key)
            if isinstance(value, bool):
                if key == "augment":
                    args.append("--augment" if value else "--no_augment")
                elif key == "channels_last":
                    args.append("--channels_last" if value else "--no_channels_last")
                elif key == "amp":
                    args.append("--amp" if value else "--no_amp")
                elif key == "compile":
                    args.append("--compile" if value else "--no_compile")
                elif key == "use_dali":
                    args.append("--use_dali" if value else "--no_dali")
                elif key == "ema_on_cpu":
                    if value:
                        args.append("--ema_on_cpu")
                elif key == "use_fake_data":
                    if value:
                        args.append("--use_fake_data")
                continue
            if value is None:
                continue
            args.extend([f"--{cli_key}", str(value)])
        return args

    def _filter_applicable_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filtra para chaves aceitas pelo CLI MONAI."""
        KEY_MAP_INV = {
            "learning_rate": "learning_rate",
            "model_name": "model",
            "results_dir": "results",
        }
        applicable = {}
        for k, v in config.items():
            mapped = KEY_MAP_INV.get(k, k)
            if mapped in self._accepted_args or k in self._accepted_args:
                applicable[k] = v
        return applicable

    def parse_epoch_metrics(self, output: str) -> Dict[str, Any]:
        """Parseia saída do treino MONAI."""
        metrics: Dict[str, Any] = {}
        # MONAI output format: [E{n}] train_loss=X val_loss=X val_auc=X ...
        patterns = {
            "train_loss": r"train_loss[=:]\s*([\d.]+)",
            "val_loss": r"val_loss[=:]\s*([\d.]+)",
            "val_auc": r"val_auc[=:]\s*([\d.]+)",
            "throughput": r"throughput[=:]\s*([\d.]+)",
            "lr": r"lr[=:]\s*([\d.e+-]+)",
        }
        for key, pat in patterns.items():
            match = re.search(pat, output, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))
        return metrics

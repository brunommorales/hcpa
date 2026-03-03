"""
pytorch_backend.py — Backend para variantes PyTorch (pytorch_base, pytorch_opt).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BackendBase

# Argumentos CLI aceitos por cada variante (extraídos da auditoria)
_PYTORCH_BASE_ARGS = {
    "tfrec_dir", "dataset", "results", "exec", "img_sizes", "batch_size",
    "epochs", "lrate", "num_thresholds", "verbose", "model", "normalize",
    "augment", "no-augment", "cores", "seed", "clip_grad_norm",
    "freeze_epochs", "fine_tune_lr_factor", "fine_tune_lr",
    "warmup_epochs", "min_lr", "label_smoothing",
}

_PYTORCH_OPT_ARGS = _PYTORCH_BASE_ARGS | {
    "mixup_alpha", "cutmix_alpha", "focal_gamma", "pos_weight",
    "tta_views", "fundus_crop_ratio",
} - {"num_thresholds"}


class PyTorchBackend(BackendBase):
    STACK_NAME = "pytorch"

    def __init__(self, variant_path: Path, mode: str):
        super().__init__(variant_path, mode)
        self._accepted_args = _PYTORCH_OPT_ARGS if mode == "opt" else _PYTORCH_BASE_ARGS

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
            cli_key = f"--{key}"
            if isinstance(value, bool):
                if key == "augment":
                    args.append("--augment" if value else "--no-augment")
                continue
            if value is None:
                continue
            args.extend([cli_key, str(value)])
        return args

    def _filter_applicable_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filtra para incluir apenas args CLI aceitos pela variante."""
        return {k: v for k, v in config.items() if k in self._accepted_args}

    def parse_epoch_metrics(self, output: str) -> Dict[str, Any]:
        """Parseia a saída do treino PyTorch para métricas de época."""
        metrics: Dict[str, Any] = {}
        # Formato: [freeze/finetune E{n}/{N}] train_loss=X val_loss=X trainAUC=X valAUC=X thr=X img/s lr=X
        pattern = (
            r"\[(\w+)\s+E(\d+)/(\d+)\]\s+"
            r"train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+"
            r"trainAUC=([\d.]+)\s+valAUC=([\d.]+)\s+"
            r"thr=([\d.]+)\s+img/s\s+lr=([\d.e+-]+)"
        )
        match = re.search(pattern, output)
        if match:
            metrics["stage"] = match.group(1)
            metrics["epoch"] = int(match.group(2))
            metrics["train_loss"] = float(match.group(4))
            metrics["val_loss"] = float(match.group(5))
            metrics["train_auc"] = float(match.group(6))
            metrics["val_auc"] = float(match.group(7))
            metrics["throughput"] = float(match.group(8))
            metrics["lr"] = float(match.group(9))
        return metrics

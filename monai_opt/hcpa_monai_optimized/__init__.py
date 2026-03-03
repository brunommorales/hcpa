"""monai_opt (pacote hcpa_monai_optimized) — versão de performance (PyTorch/MONAI)."""

from .config import TrainConfig, VariantDefaults
from .data import create_loaders, find_tfrec_splits
from .metrics import EpochStats, compute_binary_metrics
from .models import build_model
from .training import train_and_evaluate, evaluate_checkpoint, benchmark, OptimizedDefaults
from .utils import (
    apply_ema,
    ensure_dir,
    get_device,
    maybe_compile,
    move_to_device,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    time_since,
    to_channels_last,
    update_ema,
)

__all__ = [
    "TrainConfig",
    "VariantDefaults",
    "create_loaders",
    "find_tfrec_splits",
    "EpochStats",
    "compute_binary_metrics",
    "build_model",
    "train_and_evaluate",
    "evaluate_checkpoint",
    "benchmark",
    "OptimizedDefaults",
    "apply_ema",
    "ensure_dir",
    "get_device",
    "maybe_compile",
    "move_to_device",
    "save_checkpoint",
    "load_checkpoint",
    "set_seed",
    "time_since",
    "to_channels_last",
    "update_ema",
]

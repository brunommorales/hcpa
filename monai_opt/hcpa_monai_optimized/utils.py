from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def save_checkpoint(
    path: Path,
    *,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]],
    scheduler_state: Optional[Dict[str, Any]],
    scaler_state: Optional[Dict[str, Any]],
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "scaler": scaler_state,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def apply_ema(model: torch.nn.Module, ema_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Swap model params with EMA state returning backup."""
    backup: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in ema_state:
                continue
            backup[name] = param.detach().clone()
            param.data.copy_(ema_state[name])
    return backup


def update_ema(state: Dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            new_val = param.detach()
            if name in state:
                state[name].mul_(decay).add_(new_val, alpha=1 - decay)
            else:
                state[name] = new_val.clone()


def time_since(start: float) -> float:
    return time.perf_counter() - start


def to_channels_last(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled:
        return model.to(memory_format=torch.channels_last)
    return model


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model, mode="reduce-overhead")
    return model

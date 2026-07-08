"""
Helpers de estabilidade numérica compartilhados entre variantes de treino.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from contextlib import nullcontext


def is_primary_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def format_metric(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def resolve_amp_dtype(*, enable_amp: bool, requested: str = "auto") -> torch.dtype | None:
    if not enable_amp or not torch.cuda.is_available():
        return None

    normalized = str(requested).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized != "auto":
        raise ValueError(f"Unknown AMP dtype: {requested}")

    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(enabled: bool, amp_dtype: torch.dtype | None):
    if enabled and amp_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def tensor_is_finite(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    return bool(torch.isfinite(tensor).all().item())


def clip_gradients_and_check_finite(
    model: torch.nn.Module,
    *,
    max_norm: float,
) -> tuple[bool, float]:
    params_with_grad = [param for param in model.parameters() if param.grad is not None]
    if not params_with_grad:
        return True, 0.0

    clip_value = float("inf") if max_norm <= 0 else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(
        params_with_grad,
        max_norm=clip_value,
        error_if_nonfinite=False,
    )
    total_norm_value = float(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
    return math.isfinite(total_norm_value), total_norm_value


def sanitize_prediction_tensors(
    targets: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    targets = targets.view(-1)
    probs = probs.view(-1)
    finite_mask = torch.isfinite(targets) & torch.isfinite(probs)
    invalid_count = int((~finite_mask).sum().item())
    if invalid_count == 0:
        return targets, probs, 0
    return targets[finite_mask], probs[finite_mask], invalid_count


def apply_finetune_lr_transition(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    target_lr: float,
) -> float:
    if target_lr <= 0:
        raise ValueError(f"target_lr must be positive, got {target_lr}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = target_lr

    if scheduler is not None:
        scheduler.base_lr = target_lr
        scheduler.current_epoch = max(scheduler.current_epoch, scheduler.warmup_epochs)

    return target_lr

"""Utilities for RETFound-Green fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_shared.training_utils_opt import count_parameters
from retfound_green.model import RETFOUND_GREEN_PATCH_SIZE


def validate_retfound_green_args(img_size: int) -> None:
    if img_size <= 0:
        raise ValueError(f"img_size ({img_size}) must be positive.")
    if img_size % RETFOUND_GREEN_PATCH_SIZE != 0:
        raise ValueError(
            f"img_size ({img_size}) must be divisible by patch_size ({RETFOUND_GREEN_PATCH_SIZE})."
        )


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    return trainable, total


def model_summary(
    model: nn.Module,
    model_name: str = "RETFound-Green",
    *,
    img_size: int,
    feature_dim: int,
    backbone_model: str,
    freeze_backbone: bool,
    checkpoint_path: str | None,
) -> str:
    trainable, total = count_model_parameters(model)
    grid_size = img_size // RETFOUND_GREEN_PATCH_SIZE
    num_patches = grid_size * grid_size
    checkpoint_label = checkpoint_path or "none"

    return f"""
╔════════════════════════════════════════════════════════════════╗
║  MODEL SUMMARY: {model_name:<40}║
╚════════════════════════════════════════════════════════════════╝

Parametros:
  Treinaveis:     {trainable:>15,}
  Totais:         {total:>15,}
  Taxa treino:    {trainable/total*100:>14.1f}%

Backbone:
  Model:          {backbone_model}
  Image size:     {img_size} x {img_size}
  Patch size:     {RETFOUND_GREEN_PATCH_SIZE} x {RETFOUND_GREEN_PATCH_SIZE}
  Patch grid:     {grid_size} x {grid_size}
  Num patches:    {num_patches}
  Feature dim:    {feature_dim}
  Frozen:         {"yes" if freeze_backbone else "no"}
  Weights:        {checkpoint_label}

Estrutura:
{model}
"""


def print_gpu_memory_info() -> None:
    if not torch.cuda.is_available():
        print("GPU nao disponivel")
        return

    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

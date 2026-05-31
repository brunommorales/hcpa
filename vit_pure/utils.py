"""Utilities for Pure ViT."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_shared.training_utils_opt import count_parameters


def validate_vit_args(img_size: int, patch_size: int) -> None:
    if img_size % patch_size != 0:
        raise ValueError(
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
        )


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    return trainable, total


def model_summary(
    model: nn.Module,
    model_name: str = "ViT-B/16",
    *,
    img_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    use_cls_token: bool = True,
) -> str:
    trainable, total = count_model_parameters(model)
    grid_size = img_size // patch_size
    num_patches = grid_size * grid_size
    sequence_length = num_patches + (1 if use_cls_token else 0)

    return f"""
╔════════════════════════════════════════════════════════════════╗
║  MODEL SUMMARY: {model_name:<40}║
╚════════════════════════════════════════════════════════════════╝

Parâmetros:
  Treináveis:     {trainable:>15,}
  Totais:         {total:>15,}
  Taxa treino:    {trainable/total*100:>14.1f}%

Patch Tokenization:
  Source:         raw image patches
  Image size:     {img_size} x {img_size}
  Patch size:     {patch_size} x {patch_size}
  Patch grid:     {grid_size} x {grid_size}
  Num patches:    {num_patches}
  Seq length:     {sequence_length}
  Pooling:        {"cls" if use_cls_token else "mean"}
  Embed dim:      {embed_dim}
  Layers/Heads:   {num_layers} / {num_heads}

Estrutura:
{model}
"""


def print_gpu_memory_info() -> None:
    if not torch.cuda.is_available():
        print("GPU não disponível")
        return

    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

from __future__ import annotations

import timm
import torch
from torch import nn

from .config import TrainConfig


class TimmClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
            drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.backbone(x)


def build_model(cfg: TrainConfig) -> nn.Module:
    return TimmClassifier(cfg.model_name, cfg.num_classes, cfg.pretrained, dropout=cfg.dropout)


__all__ = ["build_model", "TimmClassifier"]

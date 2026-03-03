from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import Inception_V3_Weights

from .config import TrainConfig


class InceptionV3Classifier(nn.Module):
    """Minimal InceptionV3 head without timm/DALI."""

    def __init__(self, num_classes: int, pretrained: bool, dropout: float = 0.0):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.inception_v3(weights=weights, aux_logits=True)
        in_features = backbone.fc.in_features
        head: nn.Module
        if dropout > 0:
            head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        else:
            head = nn.Linear(in_features, num_classes)
        backbone.fc = head
        # disable aux head to keep single-output path
        backbone.aux_logits = False
        backbone.AuxLogits = None
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        out = self.backbone(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_name.lower() != "inception_v3":
        raise ValueError("Este variant usa apenas inception_v3 (sem timm).")
    return InceptionV3Classifier(cfg.num_classes, cfg.pretrained, dropout=cfg.dropout)


__all__ = ["build_model", "InceptionV3Classifier"]

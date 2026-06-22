"""
RETFound-Green backbone wrapper for diabetic retinopathy classification.

This module uses the public RETFound-Green backbone weights and adds a
lightweight binary classification head on top of the pooled feature vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import timm
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "RETFound-Green requires `timm` and its torchvision dependency. "
        "Install the module requirements before importing this package."
    ) from exc

import torch
import torch.nn as nn


RETFOUND_GREEN_BACKBONE_MODEL = "vit_small_patch14_reg4_dinov2"
RETFOUND_GREEN_PATCH_SIZE = 14
RETFOUND_GREEN_IMG_SIZE = 392
RETFOUND_GREEN_FEATURE_DIM = 384
ALLOWED_UNEXPECTED_PREFIXES = ("head.", "classifier.")


def _unwrap_checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "backbone_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)}")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def _filter_problematic_keys(keys: list[str]) -> list[str]:
    return [key for key in keys if not key.startswith(ALLOWED_UNEXPECTED_PREFIXES)]


class RetFoundGreenClassifier(nn.Module):
    """Binary classifier built on top of the RETFound-Green public backbone."""

    def __init__(
        self,
        *,
        img_size: int = RETFOUND_GREEN_IMG_SIZE,
        backbone_model: str = RETFOUND_GREEN_BACKBONE_MODEL,
        checkpoint_path: Optional[str] = None,
        head_dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.img_size = int(img_size)
        self.backbone_model = str(backbone_model)
        self.patch_size = RETFOUND_GREEN_PATCH_SIZE
        self.loaded_checkpoint_path: Optional[str] = None

        self.backbone = timm.create_model(
            self.backbone_model,
            pretrained=False,
            img_size=(self.img_size, self.img_size),
            num_classes=0,
        )
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0, global_pool="avg")
        elif hasattr(self.backbone, "global_pool"):
            self.backbone.global_pool = "avg"

        self.feature_dim = int(getattr(self.backbone, "num_features", RETFOUND_GREEN_FEATURE_DIM))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 0))
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, 1)
        self._init_head()

        if checkpoint_path:
            self.load_backbone_checkpoint(checkpoint_path)

        self.set_backbone_trainable(not freeze_backbone)

    def _init_head(self) -> None:
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = bool(trainable)

    def load_backbone_checkpoint(self, checkpoint_path: str) -> None:
        resolved_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(resolved_path, map_location="cpu")
        state_dict = _normalize_state_dict_keys(_unwrap_checkpoint_state_dict(checkpoint))
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)

        missing_keys = _filter_problematic_keys(list(incompatible.missing_keys))
        unexpected_keys = _filter_problematic_keys(list(incompatible.unexpected_keys))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint RETFound-Green nao encaixou no backbone esperado.\n"
                f"Checkpoint: {resolved_path}\n"
                f"Missing keys: {missing_keys}\n"
                f"Unexpected keys: {unexpected_keys}"
            )

        self.loaded_checkpoint_path = str(resolved_path)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(self.head_dropout(features))

    def forward_with_patch_analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        features = self.forward_features(x)
        logits = self.classifier(self.head_dropout(features))
        grid_size = self.img_size // self.patch_size
        num_patches = grid_size * grid_size

        analysis = {
            "token_source": "retfound_green_backbone_features",
            "backbone_model": self.backbone_model,
            "patch_size": self.patch_size,
            "grid_size": grid_size,
            "num_patches": num_patches,
            "sequence_length": num_patches + self.num_prefix_tokens,
            "feature_dim": self.feature_dim,
            "checkpoint_path": self.loaded_checkpoint_path,
        }
        return logits, analysis


def create_retfound_green_model(
    *,
    img_size: int = RETFOUND_GREEN_IMG_SIZE,
    backbone_model: str = RETFOUND_GREEN_BACKBONE_MODEL,
    checkpoint_path: Optional[str] = None,
    head_dropout: float = 0.0,
    freeze_backbone: bool = False,
    device: Optional[torch.device | str] = "cuda",
) -> RetFoundGreenClassifier:
    model = RetFoundGreenClassifier(
        img_size=img_size,
        backbone_model=backbone_model,
        checkpoint_path=checkpoint_path,
        head_dropout=head_dropout,
        freeze_backbone=freeze_backbone,
    )
    return model.to(device)

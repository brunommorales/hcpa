"""
Hybrid Token Reduction Model - CNN + token pruning + modern vision Transformer.

A selecao de tokens preserva a posicao espacial original antes do Transformer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_simple.model import (
    PositionalEncoding,
    TransformerEncoder,
    extract_last_feature_map,
    feature_map_to_tokens,
)
from hybrid_token_reduction.token_selector import TokenSelector


class HybridTokenReduction(nn.Module):
    """Modelo hibrido com pruning de tokens e positional encoding 2D correto."""

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 2048,
        transformer_dropout: float = 0.1,
        keep_ratio: float = 0.5,
        keep_k: Optional[int] = None,
        aggregation: str = "mean",
        use_cls_token: bool = False,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = int(d_model)
        self.aggregation = aggregation
        self.use_cls_token = bool(use_cls_token)
        self.keep_ratio = float(keep_ratio)
        self.current_keep_ratio = float(keep_ratio)

        self.token_projection = nn.Linear(self.d_model, self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(transformer_dropout)
        self.token_selector = TokenSelector(
            token_dim=self.d_model,
            keep_ratio=keep_ratio,
            keep_k=keep_k,
            include_cls_token=self.use_cls_token,
        )
        self.pos_encoding = PositionalEncoding(
            self.d_model,
            use_cls_token=self.use_cls_token,
        )
        self.transformer = TransformerEncoder(
            d_model=self.d_model,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=transformer_dropout,
            enable_flash_attention=enable_flash_attention,
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.register_parameter("cls_token", None)

        self.classifier = nn.Linear(self.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.token_projection.weight, std=0.02)
        nn.init.zeros_(self.token_projection.bias)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        features = extract_last_feature_map(self.backbone(x))
        channels = features.size(1)
        if channels != self.d_model:
            raise ValueError(
                f"Dimensao do backbone ({channels}) difere de d_model ({self.d_model}). "
                "Ajuste backbone_dim/config para o backbone escolhido."
            )

        tokens, height, width = feature_map_to_tokens(features)
        tokens = self.token_projection(tokens)
        tokens = self.token_norm(tokens)
        return tokens, height, width

    def _pool_tokens(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            return encoded_tokens[:, 0, :]
        return encoded_tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, height, width = self._prepare_tokens(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        selected_tokens, selected_indices, _ = self.token_selector(tokens)
        selected_tokens = self.pos_encoding(
            selected_tokens,
            spatial_shape=(height, width),
            token_indices=selected_indices,
            include_cls_token=self.use_cls_token,
        )
        selected_tokens = self.input_dropout(selected_tokens)

        encoded_tokens, _ = self.transformer(selected_tokens, return_attention=False)
        encoded_tokens = self.final_norm(encoded_tokens)
        pooled = self._pool_tokens(encoded_tokens)
        return self.classifier(pooled)

    def forward_with_token_analysis(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        tokens, height, width = self._prepare_tokens(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        selected_tokens, selected_indices, scores = self.token_selector(tokens)
        actual_tokens = tokens.size(1)
        selected_tokens = self.pos_encoding(
            selected_tokens,
            spatial_shape=(height, width),
            token_indices=selected_indices,
            include_cls_token=self.use_cls_token,
        )
        encoded_tokens, attention_weights = self.transformer(selected_tokens, return_attention=True)
        encoded_tokens = self.final_norm(encoded_tokens)
        pooled = self._pool_tokens(encoded_tokens)
        logits = self.classifier(pooled)

        analysis = {
            "num_input_tokens": actual_tokens,
            "num_selected_tokens": selected_tokens.size(1),
            "compression_ratio": selected_tokens.size(1) / actual_tokens,
            "kept_ratio": selected_tokens.size(1) / actual_tokens,
            "token_scores": scores,
            "selected_indices": selected_indices,
            "attention_weights": attention_weights,
            "spatial_shape": (height, width),
        }
        return logits, analysis

    def get_effective_reduction(self) -> float:
        n_typical = 10000
        k_typical = max(1, int(n_typical * self.current_keep_ratio))
        return (n_typical ** 2) / (k_typical ** 2)

    def set_keep_ratio(self, keep_ratio: float) -> None:
        self.current_keep_ratio = float(keep_ratio)
        self.token_selector.keep_ratio = float(keep_ratio)


def create_hybrid_token_reduction_model(
    backbone: nn.Module,
    backbone_feature_dim: int,
    keep_ratio: float = 0.5,
    keep_k: Optional[int] = None,
    num_transformer_layers: int = 4,
    num_heads: int = 4,
    use_cls_token: bool = False,
    enable_flash_attention: bool = False,
    device: str = "cuda",
) -> HybridTokenReduction:
    """Factory function para criar modelo com token reduction."""
    model = HybridTokenReduction(
        backbone=backbone,
        d_model=backbone_feature_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        d_ff=backbone_feature_dim * 4,
        transformer_dropout=0.1,
        keep_ratio=keep_ratio,
        keep_k=keep_k,
        aggregation="mean",
        use_cls_token=use_cls_token,
        enable_flash_attention=enable_flash_attention,
    )
    return model.to(device)

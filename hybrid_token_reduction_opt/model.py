"""
Hybrid Token Reduction Model - optimized variant with modern vision Transformer blocks.

A selecao de tokens preserva a posicao espacial original antes do Transformer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_simple.model import (
    PositionalEncoding,
    extract_last_feature_map,
    feature_map_to_tokens,
)
from hybrid_token_reduction_opt.token_selector import TokenSelector


class FusedSelfAttention(nn.Module):
    """Self-attention with a packed QKV projection and SDPA fast path."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        enable_flash_attention: bool,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisivel por num_heads")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.enable_flash_attention = bool(enable_flash_attention)
        self.last_attention_backend = "standard"

        self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_attention_backend = "standard"
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        return context, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        q, k, v = self._project_qkv(x)
        dropout_p = self.attn_dropout.p if self.training else 0.0

        if (
            self.enable_flash_attention
            and not return_attention
            and mask is None
            and hasattr(F, "scaled_dot_product_attention")
        ):
            self.last_attention_backend = "sdpa"
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_weights = None
        else:
            context, attn_weights = self._standard_attention(q, k, v, mask)

        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        output = self.out_dropout(output)
        return output, attn_weights


class OptimizedTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer using fused QKV self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        enable_flash_attention: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = FusedSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            enable_flash_attention=enable_flash_attention,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_weights = self.attention(
            self.norm1(x),
            mask=mask,
            return_attention=return_attention,
        )
        x = x + attn_output
        x = x + self.feed_forward(self.norm2(x))
        return x, attn_weights


class OptimizedTransformerEncoder(nn.Module):
    """Stack of optimized Transformer encoder layers."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        enable_flash_attention: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                OptimizedTransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    enable_flash_attention=enable_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_attention_weights: list[torch.Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)
        return x, all_attention_weights


class HybridTokenReduction(nn.Module):
    """Modelo hibrido com pruning de tokens e encoder moderno otimizado."""

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
        enable_flash_attention: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = int(d_model)
        self.aggregation = aggregation
        self.use_cls_token = bool(use_cls_token)
        self.keep_ratio = float(keep_ratio)
        self.current_keep_ratio = float(keep_ratio)
        self.enable_flash_attention = bool(enable_flash_attention)

        self.token_projection = nn.Linear(self.d_model, self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(transformer_dropout)
        self.token_selector = TokenSelector(
            token_dim=self.d_model,
            keep_ratio=keep_ratio,
            keep_k=keep_k,
            include_cls_token=False,
        )
        self.pos_encoding = PositionalEncoding(
            self.d_model,
            use_cls_token=self.use_cls_token,
        )
        self.transformer = OptimizedTransformerEncoder(
            d_model=self.d_model,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=transformer_dropout,
            enable_flash_attention=self.enable_flash_attention,
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

    def _extract_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        features = extract_last_feature_map(self.backbone(x))
        channels = features.size(1)
        if channels != self.d_model:
            raise ValueError(
                f"Dimensao do backbone ({channels}) difere de d_model ({self.d_model}). "
                "Ajuste backbone_dim/config para o backbone escolhido."
            )

        tokens, height, width = feature_map_to_tokens(features)
        return tokens, height, width

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.token_projection(tokens)
        tokens = self.token_norm(tokens)
        return tokens

    def _select_project_and_position(
        self,
        tokens: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_tokens, selected_indices, scores = self.token_selector(tokens)
        selected_tokens = self._project_tokens(selected_tokens)

        if self.use_cls_token:
            batch_size = selected_tokens.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            selected_tokens = torch.cat([cls_tokens, selected_tokens], dim=1)
            cls_indices = torch.zeros(
                (batch_size, 1),
                dtype=selected_indices.dtype,
                device=selected_indices.device,
            )
            selected_indices = torch.cat([cls_indices, selected_indices + 1], dim=1)

        selected_tokens = self.pos_encoding(
            selected_tokens,
            spatial_shape=spatial_shape,
            token_indices=selected_indices,
            include_cls_token=self.use_cls_token,
        )
        return selected_tokens, selected_indices, scores

    def _pool_tokens(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            return encoded_tokens[:, 0, :]
        return encoded_tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, height, width = self._extract_tokens(x)
        selected_tokens, _, _ = self._select_project_and_position(
            tokens,
            spatial_shape=(height, width),
        )
        selected_tokens = self.input_dropout(selected_tokens)

        encoded_tokens, _ = self.transformer(selected_tokens, return_attention=False)
        encoded_tokens = self.final_norm(encoded_tokens)
        pooled = self._pool_tokens(encoded_tokens)
        return self.classifier(pooled)

    def forward_with_token_analysis(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        tokens, height, width = self._extract_tokens(x)
        selected_tokens, selected_indices, scores = self._select_project_and_position(
            tokens,
            spatial_shape=(height, width),
        )
        actual_tokens = tokens.size(1) + (1 if self.use_cls_token else 0)
        selected_tokens = self.input_dropout(selected_tokens)
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
        n_typical = 64
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
    enable_flash_attention: bool = True,
    device: str = "cuda",
) -> HybridTokenReduction:
    """Factory function para criar modelo otimizado com token reduction."""
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

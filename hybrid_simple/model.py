"""
Hybrid Simple Model - CNN Backbone + Modern Vision Transformer Encoder.

Arquitetura:
  Imagem -> CNN Backbone -> Feature Map [B, C, H, W]
         -> Flatten espacial -> [B, N, D] tokens
         -> Projecao/Norm -> Positional Encoding 2D learnable
         -> Transformer Encoder pre-norm
         -> Agregacao (mean pooling ou CLS token)
         -> Linear Head -> Logit
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN_PACKAGE = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN_PACKAGE = False


def _flash_sdp_context():
    """Prefer the CUDA flash kernel when PyTorch exposes the SDP context manager."""
    if not hasattr(torch.backends, "cuda"):
        return nullcontext()

    sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp_kernel is None:
        return nullcontext()

    try:
        return sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    except TypeError:
        return nullcontext()


def extract_last_feature_map(features: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Normaliza o retorno do backbone para um unico mapa [B, C, H, W]."""
    if isinstance(features, (list, tuple)):
        if len(features) == 0:
            raise ValueError("Backbone retornou lista de features vazia.")
        features = features[-1]

    if not torch.is_tensor(features):
        raise TypeError(f"Backbone retornou tipo invalido: {type(features)}")
    if features.ndim != 4:
        raise ValueError(f"Esperado feature map 4D [B, C, H, W], recebido shape={tuple(features.shape)}")
    return features


def feature_map_to_tokens(features: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Converte [B, C, H, W] em tokens [B, N, C] mantendo a grade espacial."""
    features = extract_last_feature_map(features)
    _, _, height, width = features.shape
    tokens = features.flatten(2).transpose(1, 2).contiguous()
    return tokens, height, width


class PositionalEncoding(nn.Module):
    """
    Learnable 2D positional encoding for CNN feature grids.

    Usa embeddings fatorados por linha e coluna, mais robustos para modelos
    hibridos do que um encoding 1D senoidal aplicado na ordem do flatten.
    """

    def __init__(
        self,
        d_model: int,
        max_height: int = 64,
        max_width: int = 64,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.use_cls_token = bool(use_cls_token)

        self.row_embed = nn.Parameter(torch.zeros(1, self.max_height, self.d_model))
        self.col_embed = nn.Parameter(torch.zeros(1, self.max_width, self.d_model))
        if self.use_cls_token:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.register_parameter("cls_pos_embed", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
        if self.cls_pos_embed is not None:
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    @staticmethod
    def _resize_axis_embedding(embedding: torch.Tensor, size: int) -> torch.Tensor:
        if embedding.size(1) == size:
            return embedding
        resized = F.interpolate(
            embedding.transpose(1, 2),
            size=size,
            mode="linear",
            align_corners=False,
        )
        return resized.transpose(1, 2)

    def build_position_sequence(
        self,
        height: int,
        width: int,
        *,
        include_cls_token: bool = False,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        row = self._resize_axis_embedding(self.row_embed, height)
        col = self._resize_axis_embedding(self.col_embed, width)

        spatial_pos = row[:, :, None, :] + col[:, None, :, :]
        spatial_pos = spatial_pos.reshape(1, height * width, self.d_model)

        if include_cls_token:
            if self.cls_pos_embed is None:
                raise RuntimeError("CLS positional embedding nao foi inicializado.")
            spatial_pos = torch.cat([self.cls_pos_embed, spatial_pos], dim=1)

        if dtype is not None:
            spatial_pos = spatial_pos.to(dtype=dtype)
        return spatial_pos

    def forward(
        self,
        x: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        token_indices: Optional[torch.Tensor] = None,
        include_cls_token: bool = False,
    ) -> torch.Tensor:
        height, width = spatial_shape
        positions = self.build_position_sequence(
            height,
            width,
            include_cls_token=include_cls_token,
            dtype=x.dtype,
        )

        if token_indices is not None:
            positions = positions.expand(x.size(0), -1, -1)
            positions = torch.gather(
                positions,
                dim=1,
                index=token_indices.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
        else:
            positions = positions[:, : x.size(1), :]

        return x + positions


class MultiHeadAttention(nn.Module):
    """Multi-head attention com fast paths para Flash Attention/SDPA."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisivel por num_heads")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.enable_flash_attention = bool(enable_flash_attention)
        self.last_attention_backend = "standard"

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _reshape_projected(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _can_use_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_attention: bool,
    ) -> bool:
        if not self.enable_flash_attention or return_attention or mask is not None:
            return False
        if not HAS_FLASH_ATTN_PACKAGE and not hasattr(F, "scaled_dot_product_attention"):
            return False
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        return True

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dropout_p = self.attn_dropout.p if self.training else 0.0

        if HAS_FLASH_ATTN_PACKAGE:
            self.last_attention_backend = "flash_attn"
            q_flash = q.transpose(1, 2).contiguous()
            k_flash = k.transpose(1, 2).contiguous()
            v_flash = v.transpose(1, 2).contiguous()
            out = flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=dropout_p,
                softmax_scale=self.head_dim ** -0.5,
                causal=False,
            )
            return out.transpose(1, 2).contiguous()

        self.last_attention_backend = "sdpa"
        with _flash_sdp_context():
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )

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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape

        q = self._reshape_projected(self.q_proj(q), batch_size, q_len)
        k = self._reshape_projected(self.k_proj(k), batch_size, k_len)
        v = self._reshape_projected(self.v_proj(v), batch_size, v_len)

        if self._can_use_flash_attention(q, k, v, mask, return_attention):
            try:
                context = self._flash_attention(q, k, v)
                attn_weights = None
            except (RuntimeError, TypeError, AttributeError):
                context, attn_weights = self._standard_attention(q, k, v, mask)
        else:
            context, attn_weights = self._standard_attention(q, k, v, mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.out_proj(context)
        output = self.out_dropout(output)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer em estilo moderno pre-norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout,
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
        norm_x = self.norm1(x)
        attn_output, attn_weights = self.attention(
            norm_x,
            norm_x,
            norm_x,
            mask,
            return_attention=return_attention,
        )
        x = x + attn_output
        x = x + self.feed_forward(self.norm2(x))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack de camadas Transformer encoder."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
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
            x, attn_weights = layer(x, mask, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)
        return x, all_attention_weights


class HybridSimple(nn.Module):
    """Modelo hibrido CNN + Transformer com encoding posicional 2D learnable."""

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 2048,
        transformer_dropout: float = 0.1,
        aggregation: str = "mean",
        use_cls_token: bool = False,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = int(d_model)
        self.aggregation = aggregation
        self.use_cls_token = bool(use_cls_token)

        self.token_projection = nn.Linear(self.d_model, self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(transformer_dropout)
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

    def _apply_positional_encoding(
        self,
        tokens: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.pos_encoding(
            tokens,
            spatial_shape=spatial_shape,
            token_indices=token_indices,
            include_cls_token=self.use_cls_token,
        )

    def _pool_tokens(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            return encoded_tokens[:, 0, :]
        return encoded_tokens.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, height, width = self._prepare_tokens(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = self._apply_positional_encoding(tokens, spatial_shape=(height, width))
        tokens = self.input_dropout(tokens)
        encoded_tokens, _ = self.transformer(tokens, return_attention=False)
        encoded_tokens = self.final_norm(encoded_tokens)
        pooled = self._pool_tokens(encoded_tokens)
        return self.classifier(pooled)

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        tokens, height, width = self._prepare_tokens(x)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = self._apply_positional_encoding(tokens, spatial_shape=(height, width))
        tokens = self.input_dropout(tokens)
        encoded_tokens, all_attn_weights = self.transformer(tokens, return_attention=True)
        encoded_tokens = self.final_norm(encoded_tokens)
        pooled = self._pool_tokens(encoded_tokens)
        logits = self.classifier(pooled)
        return logits, all_attn_weights


def create_hybrid_simple_model(
    backbone: nn.Module,
    backbone_feature_dim: int,
    num_transformer_layers: int = 4,
    num_heads: int = 4,
    use_cls_token: bool = False,
    enable_flash_attention: bool = False,
    device: str = "cuda",
) -> HybridSimple:
    """Factory function para criar modelo hibrido simples."""
    model = HybridSimple(
        backbone=backbone,
        d_model=backbone_feature_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        d_ff=backbone_feature_dim * 4,
        transformer_dropout=0.1,
        aggregation="mean",
        use_cls_token=use_cls_token,
        enable_flash_attention=enable_flash_attention,
    )
    return model.to(device)

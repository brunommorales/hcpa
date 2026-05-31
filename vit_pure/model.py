"""
Pure Vision Transformer for diabetic retinopathy classification.

This module keeps the architecture close to the canonical ViT-B/16 layout:
- patch embedding directly from raw pixels
- learnable CLS token and positional embeddings
- pre-norm Transformer encoder blocks
- MLP expansion ratio of 4
- final LayerNorm followed by a linear classification head
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VIT_B16_PATCH_SIZE = 16
VIT_B16_EMBED_DIM = 768
VIT_B16_NUM_LAYERS = 12
VIT_B16_NUM_HEADS = 12
VIT_B16_MLP_RATIO = 4.0
VIT_B16_NORM_EPS = 1e-6


class PatchEmbedding(nn.Module):
    """Convert raw images into a sequence of patch tokens."""

    def __init__(
        self,
        img_size: int = 320,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = int(embed_dim)

        # Each convolution window is one raw image patch.
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected input size {(self.img_size, self.img_size)}, got {tuple(x.shape[-2:])}."
            )

        x = self.proj(x)  # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened raw patches before projection for analysis/debugging."""
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected input size {(self.img_size, self.img_size)}, got {tuple(x.shape[-2:])}."
            )

        patches = torch.nn.functional.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patches.transpose(1, 2)  # [B, N, patch_dim]


class ViTAttention(nn.Module):
    """Multi-head self-attention in standard ViT form."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.enable_flash_attention = bool(enable_flash_attention)
        self.last_attention_backend = "standard"

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def _can_use_sdp_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attention: bool,
    ) -> bool:
        if not self.enable_flash_attention or return_attention:
            return False
        if not hasattr(F, "scaled_dot_product_attention"):
            return False
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        return True

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self._can_use_sdp_attention(q, k, v, return_attention):
            dropout_p = self.attn_drop.p if self.training else 0.0
            try:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
                self.last_attention_backend = "sdpa"
                x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x, None
            except RuntimeError:
                self.last_attention_backend = "standard"
                pass

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.last_attention_backend = "standard"

        x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn if return_attention else None


class ViTMLP(nn.Module):
    """Feed-forward block used in ViT."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ViTBlock(nn.Module):
    """Transformer block with pre-norm, matching standard ViT practice."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_eps: float = VIT_B16_NORM_EPS,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = ViTAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            enable_flash_attention=enable_flash_attention,
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = ViTMLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class PureViT(nn.Module):
    """
    Vision Transformer with tokens generated from raw image patches.

    Flow:
    - Image [B, 3, H, W]
    - PatchEmbedding over raw pixels -> [B, N, D]
    - Optional CLS token + learnable positional embeddings
    - Transformer encoder
    - CLS pooling or mean pooling
    - Linear head -> binary logit
    """

    def __init__(
        self,
        img_size: int = 320,
        patch_size: int = VIT_B16_PATCH_SIZE,
        in_chans: int = 3,
        embed_dim: int = VIT_B16_EMBED_DIM,
        num_transformer_layers: int = VIT_B16_NUM_LAYERS,
        num_heads: int = VIT_B16_NUM_HEADS,
        mlp_ratio: float = VIT_B16_MLP_RATIO,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        norm_eps: float = VIT_B16_NORM_EPS,
        enable_flash_attention: bool = False,
    ):
        super().__init__()

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_cls_token = bool(use_cls_token)
        self.enable_flash_attention = bool(enable_flash_attention)

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        num_tokens = self.num_patches + (1 if self.use_cls_token else 0)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("cls_token", None)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    enable_flash_attention=self.enable_flash_attention,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.classifier = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(self._init_module)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.pos_drop(tokens)
        return tokens

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        tokens = self._prepare_tokens(x)
        attention_weights = [] if return_attentions else None
        for block in self.blocks:
            tokens, attn_weights = block(tokens, return_attention=return_attentions)
            if attention_weights is not None and attn_weights is not None:
                attention_weights.append(attn_weights)
        return self.norm(tokens), attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_tokens, _ = self.forward_features(x, return_attentions=False)

        if self.use_cls_token:
            pooled = encoded_tokens[:, 0, :]
        else:
            pooled = encoded_tokens.mean(dim=1)

        return self.classifier(pooled)

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        encoded_tokens, attention_weights = self.forward_features(x, return_attentions=True)

        if self.use_cls_token:
            pooled = encoded_tokens[:, 0, :]
        else:
            pooled = encoded_tokens.mean(dim=1)

        logits = self.classifier(pooled)
        return logits, attention_weights

    def forward_with_patch_analysis(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raw_patches = self.patch_embed.patchify(x)
        logits, attention_weights = self.forward_with_attention(x)

        analysis = {
            "token_source": "raw_image_patches",
            "patch_size": self.patch_size,
            "grid_size": self.grid_size,
            "num_patches": self.num_patches,
            "sequence_length": self.num_patches + (1 if self.use_cls_token else 0),
            "raw_patch_shape": tuple(raw_patches.shape),
            "attention_weights": attention_weights,
        }
        return logits, analysis


def create_pure_vit_model(
    img_size: int = 320,
    patch_size: int = VIT_B16_PATCH_SIZE,
    embed_dim: int = VIT_B16_EMBED_DIM,
    num_transformer_layers: int = VIT_B16_NUM_LAYERS,
    num_heads: int = VIT_B16_NUM_HEADS,
    mlp_ratio: float = VIT_B16_MLP_RATIO,
    dropout: float = 0.1,
    use_cls_token: bool = True,
    enable_flash_attention: bool = False,
    device: Optional[torch.device | str] = "cuda",
) -> PureViT:
    model = PureViT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        use_cls_token=use_cls_token,
        enable_flash_attention=enable_flash_attention,
    )
    return model.to(device)

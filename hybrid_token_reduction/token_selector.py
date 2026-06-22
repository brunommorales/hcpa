"""
Token Selector Module - Seleção learnable de tokens importantes.

Estratégia para reduzir custo computacional do Transformer de O(n²) para O(k²),
onde k << n (apenas os tokens mais relevantes são processados).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class TokenSelector(nn.Module):
    """
    Seletor de tokens baseado em learned scoring.

    O módulo aprende a atribuir uma pontuação de importância a cada token
    e seleciona os top-k mais importantes.

    Fluxo:
    - Input: [B, N, D] tokens
    - Scoring: Linear(D, 1) → [B, N, 1]
    - Top-K: Seleciona indices dos k maiores scores
    - Output: [B, K, D] tokens reduzidos

    Esta arquitetura é eficiente porque:
    1. Scoring é O(N) - apenas uma passagem linear
    2. Top-K é O(N log k)
    3. Indexing é O(K)
    Total: O(N) para seleção, depois O(k²) no Transformer em vez de O(n²)

    Args:
        token_dim: Dimensão dos tokens (D)
        keep_ratio: Proporção de tokens a manter (ex: 0.5 para 50%, 0.25 para 25%)
        keep_k: Número absoluto de tokens a manter (se especificado, sobrescreve keep_ratio)
        include_cls_token: Se True e CLS token estiver presente, sempre mantém (útil para ViT)
    """

    def __init__(
        self,
        token_dim: int,
        keep_ratio: float = 0.5,
        keep_k: Optional[int] = None,
        include_cls_token: bool = False,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.keep_ratio = keep_ratio
        self.keep_k = keep_k
        self.include_cls_token = include_cls_token

        # Scoring network: Linear para gerar score escalar por token
        self.score_layer = nn.Linear(token_dim, 1)

    def forward(self, tokens: torch.Tensor, score_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Seleciona tokens importantes.

        Args:
            tokens: [B, N, D] tensor de tokens (N pode incluir CLS token no início)
            score_override: [B, N] tensor de scores customizados (para debugging/análise)

        Returns:
            selected_tokens: [B, K, D] tokens selecionados
            selected_indices: [B, K] índices originais dos tokens selecionados
            scores: [B, N] scores de cada token (para visualização/análise)
        """
        B, N, D = tokens.shape

        # Se override de scores for fornecido, usar
        if score_override is not None:
            scores = score_override  # [B, N]
        else:
            # Gerar scores
            scores = self.score_layer(tokens).squeeze(-1)  # [B, N]

        # Handle CLS token
        cls_score = None
        if self.include_cls_token and N > 1:
            # Guardar score do CLS token (índice 0)
            cls_score = scores[:, 0:1]  # [B, 1]
            scores_for_topk = scores[:, 1:]  # [B, N-1]
            tokens_for_topk = tokens[:, 1:, :]  # [B, N-1, D]
            cls_token = tokens[:, 0:1, :]  # [B, 1, D]
            n_without_cls = N - 1
        else:
            scores_for_topk = scores
            tokens_for_topk = tokens
            cls_token = None
            n_without_cls = N

        # Computar quantos tokens manter
        if self.keep_k is not None:
            k = min(self.keep_k, n_without_cls)
        else:
            k = max(1, int(n_without_cls * self.keep_ratio))

        # Top-K selection followed by index sorting to preserve raster order.
        # This keeps the spatial sequence stable after pruning.
        _, top_indices = torch.topk(scores_for_topk, k=k, dim=1)  # [B, K]
        top_indices, _ = torch.sort(top_indices, dim=1)

        # Gather selected tokens
        selected_tokens = torch.gather(
            tokens_for_topk,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        )  # [B, K, D]

        # Se tinha CLS token, concatenar novamente
        if cls_token is not None:
            selected_tokens = torch.cat([cls_token, selected_tokens], dim=1)  # [B, K+1, D]
            # Índices: 0 é CLS, depois K outros
            selected_indices_with_cls = torch.full((B, 1), 0, dtype=top_indices.dtype, device=top_indices.device)
            selected_indices = torch.cat([selected_indices_with_cls, top_indices + 1], dim=1)  # [B, K+1]
        else:
            selected_indices = top_indices  # [B, K]

        return selected_tokens, selected_indices, scores

    def compute_keep_ratio(self, tokens: torch.Tensor) -> float:
        """
        Computa razão atual de retenção baseada na configuração.

        Args:
            tokens: [B, N, D]

        Returns:
            Proporção de tokens que será retida
        """
        N = tokens.size(1)

        if self.include_cls_token and N > 1:
            n_without_cls = N - 1
        else:
            n_without_cls = N

        if self.keep_k is not None:
            k = min(self.keep_k, n_without_cls)
        else:
            k = max(1, int(n_without_cls * self.keep_ratio))

        if self.include_cls_token:
            return (k + 1) / N  # +1 por CLS token
        else:
            return k / N


class DynamicTokenSelector(nn.Module):
    """
    Variante: Token selector com número dinâmico de tokens.

    Ao invés de top-k fixo, aprende dinamicamente quantos tokens manter.
    Menos usado na prática, mas útil para pesquisa.
    """

    def __init__(
        self,
        token_dim: int,
        min_kept_ratio: float = 0.1,
        max_kept_ratio: float = 1.0,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.min_kept_ratio = min_kept_ratio
        self.max_kept_ratio = max_kept_ratio

        # Layers para predizer kept ratio
        self.ratio_predictor = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, 1),
            nn.Sigmoid(),  # Output em [0, 1]
        )

        # Scoring layer
        self.score_layer = nn.Linear(token_dim, 1)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Seleciona número dinâmico de tokens.

        Args:
            tokens: [B, N, D]

        Returns:
            selected_tokens: [B, K, D]
            selected_indices: [B, K]
            keep_ratio: [B, 1] razão predita
        """
        B, N, D = tokens.shape

        # Predizer keep ratio (média sobre todos os tokens de um batch)
        predicted_ratio_logits = self.ratio_predictor(tokens.mean(dim=1))  # [B, 1]
        predicted_ratio = (
            self.min_kept_ratio
            + (self.max_kept_ratio - self.min_kept_ratio) * predicted_ratio_logits
        )  # [B, 1]

        # Computar K dinâmico por sample
        k_per_sample = torch.clamp(
            (predicted_ratio * N).long().squeeze(-1),
            min=1,
            max=N - 1,
        )  # [B]

        # Gerar scores
        scores = self.score_layer(tokens).squeeze(-1)  # [B, N]

        # Top-K selection (com K variável por batch é mais complexo)
        # Por simplicidade, usar K mínimo do batch
        k_actual = k_per_sample.min().item()

        top_scores, top_indices = torch.topk(scores, k=k_actual, dim=1)  # [B, K]

        # Gather
        selected_tokens = torch.gather(
            tokens,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, K, D]

        return selected_tokens, top_indices, predicted_ratio


def visualize_token_importance(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualiza distribuição de importância dos tokens (para debugging).

    Args:
        tokens: [B, N, D]
        scores: [B, N]
        save_path: Se fornecido, salva figura em PNG
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        scores_np = scores.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histograma dos scores
        axes[0].hist(scores_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Token Importance Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Token Importance Scores')
        axes[0].grid(True, alpha=0.3)

        # Scores por posição (média do batch)
        avg_scores = scores_np.mean(axis=0)
        axes[1].plot(avg_scores, marker='o', linestyle='-', linewidth=2)
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Average Importance Score')
        axes[1].set_title('Token Importance by Position')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Token importance visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Matplotlib não disponível para visualização")

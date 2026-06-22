"""
Utilitários para Hybrid Token Reduction.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_shared.training_utils_opt import count_parameters
from hybrid_simple.utils import get_backbone_feature_dim, modify_backbone_for_feature_extraction


def apply_mixup_cutmix(xb, yb, mixup_alpha: float, cutmix_alpha: float):
    """
    Aplica Mixup ou CutMix de forma aleatória ao batch.

    Mixup: Mistura duas imagens linearmente
    CutMix: Corta uma região de uma imagem e cola of outra

    Args:
        xb: [B, 3, H, W] tensor de imagens
        yb: [B, 1] ou [B] tensor de labels (0.0 ou 1.0)
        mixup_alpha: Alpha para Beta(alpha, alpha) em Mixup
        cutmix_alpha: Alpha para Beta(alpha, alpha) em CutMix

    Returns:
        xb_mixed: [B, 3, H, W] imagens mixas
        yb_mixed: [B, 1] ou [B] labels mixos
    """
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return xb, yb

    device = xb.device
    # Decidir entre Mixup ou CutMix
    use_mixup = mixup_alpha > 0.0 and cutmix_alpha > 0.0 and (torch.rand((), device=device) < 0.5).item()
    if mixup_alpha > 0.0 and cutmix_alpha <= 0.0:
        use_mixup = True
    if cutmix_alpha > 0.0 and mixup_alpha <= 0.0:
        use_mixup = False

    # Permutação aleatória para pega a segunda imagem
    perm = torch.randperm(xb.size(0), device=xb.device)
    x2, y2 = xb[perm], yb[perm]

    if use_mixup:
        # Mixup: λ * x + (1-λ) * x2
        alpha = torch.tensor(mixup_alpha, device=device, dtype=torch.float32)
        lam = torch.distributions.Beta(alpha, alpha).sample()
        xb_mixed = lam * xb + (1.0 - lam) * x2
        yb_mixed = lam * yb + (1.0 - lam) * y2
        return xb_mixed, yb_mixed

    # CutMix: Corta região de x2 e cola em xb
    alpha = torch.tensor(cutmix_alpha, device=device, dtype=torch.float32)
    lam = torch.distributions.Beta(alpha, alpha).sample()
    _, _, H, W = xb.shape

    # Gerar bounding box aleatório
    cut_ratio = torch.sqrt(torch.clamp(1.0 - lam, min=0.0)).item()
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = int(torch.randint(0, W, (), device=device).item())
    cy = int(torch.randint(0, H, (), device=device).item())

    x1 = max(cx - cut_w // 2, 0)
    x2b = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2b = min(cy + cut_h // 2, H)

    # Copiar e modificar
    xb_mixed = xb.clone()
    xb_mixed[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]

    # Ajustar lambda baseado na área efetivamente removida
    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / float(W * H))
    yb_mixed = lam_adj * yb + (1.0 - lam_adj) * y2

    return xb_mixed, yb_mixed


def compute_token_reduction_stats(keep_ratio: float, typical_n: int = 64) -> dict:
    """
    Computa estatísticas de redução de tokens.

    Args:
        keep_ratio: Proporção de tokens a manter
        typical_n: Número típico de tokens de entrada. InceptionV3 em 299px
            gera 8x8 tokens no último feature map.

    Returns:
        Dict com estatísticas
    """
    k = max(1, int(typical_n * keep_ratio))

    # Complexidade Transformer é O(n²) para n tokens
    original_complexity = typical_n ** 2
    reduced_complexity = k ** 2
    complexity_reduction_factor = original_complexity / reduced_complexity

    # Memória (aproximadamente proporcional a complexidade)
    original_memory = typical_n ** 2  # Aprox.
    reduced_memory = k ** 2
    memory_reduction_percent = (1 - reduced_memory / original_memory) * 100

    # Tempo (Transformer é dominante, mas não único)
    # Não é linear, mas heurística: speedup menos que memory reduction
    speedup_heuristic = (complexity_reduction_factor - 1) * 0.7 + 1  # Conservador

    return {
        "keep_ratio": keep_ratio,
        "num_input_tokens": typical_n,
        "num_output_tokens": k,
        "compression_factor": typical_n / k,
        "original_transformer_complexity": original_complexity,
        "reduced_transformer_complexity": reduced_complexity,
        "complexity_reduction_factor": complexity_reduction_factor,
        "memory_savings_percent": memory_reduction_percent,
        "speedup_heuristic": speedup_heuristic,
    }


def model_summary_with_token_analysis(
    model: nn.Module,
    model_name: str = "Hybrid Token Reduction",
    keep_ratio: float = 0.5,
) -> str:
    """
    Gera resumo do modelo com análise de token reduction.

    Args:
        model: Modelo
        model_name: Nome para exibição
        keep_ratio: Proporção de redução

    Returns:
        String formatada
    """
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    token_stats = compute_token_reduction_stats(keep_ratio)

    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  MODEL SUMMARY: {model_name:<40}║
╚════════════════════════════════════════════════════════════════╝

Parâmetros:
  Treináveis:         {trainable:>15,}
  Totais:             {total:>15,}
  Taxa treino:        {trainable/total*100:>14.1f}%

Token Reduction Analysis:
  Keep Ratio:         {token_stats['keep_ratio']:>14.1%}
  Compression Factor: {token_stats['compression_factor']:>14.1f}x
  Tokens: {token_stats['num_input_tokens']} → {token_stats['num_output_tokens']}

Transformer Complexity Reduction:
  Original:           O(n²) = O({token_stats['original_transformer_complexity']:>12,})
  Reduced:            O(k²) = O({token_stats['reduced_transformer_complexity']:>12,})
  Reduction Factor:   {token_stats['complexity_reduction_factor']:>14.1f}x
  Memory Savings:     {token_stats['memory_savings_percent']:>14.1f}%
  Speedup (heuristic):{token_stats['speedup_heuristic']:>14.1f}x
"""
    return summary


def analyze_token_selection(
    model: nn.Module,
    input_batch: torch.Tensor,
    device: str = "cuda",
) -> dict:
    """
    Analisa padrão de seleção de tokens do modelo.

    Args:
        model: Modelo com forward_with_token_analysis
        input_batch: [B, 3, H, W] tensor
        device: Device

    Returns:
        Dict com análise
    """
    model.eval()
    with torch.no_grad():
        input_batch = input_batch.to(device)

        _, analysis = model.forward_with_token_analysis(input_batch)

        # Processar análise
        scores = analysis["token_scores"]  # [B, N]
        kept_ratio = analysis["kept_ratio"]
        compression = 1 / kept_ratio

        # Estatísticas dos scores
        scores_np = scores.cpu().numpy()

        return {
            "kept_ratio": kept_ratio,
            "compression_factor": compression,
            "scores_mean": scores_np.mean(),
            "scores_std": scores_np.std(),
            "scores_min": scores_np.min(),
            "scores_max": scores_np.max(),
            "num_input_tokens": analysis["num_input_tokens"],
            "num_selected_tokens": analysis["num_selected_tokens"],
            "raw_scores": scores,
            "selected_indices": analysis["selected_indices"],
        }


def format_token_analysis_report(analysis: dict, model_name: str = "Token Reduction") -> str:
    """
    Formata relatório legível de análise de token selection.

    Args:
        analysis: Dict retornado por analyze_token_selection
        model_name: Nome do modelo

    Returns:
        String formatada
    """
    report = f"""
╔════════════════════════════════════════════════════════════════╗
║  TOKEN SELECTION ANALYSIS: {model_name:<25}║
╚════════════════════════════════════════════════════════════════╝

Compression:
  Kept Ratio:         {analysis['kept_ratio']:.1%}
  Compression Factor: {analysis['compression_factor']:.1f}x
  Tokens: {analysis['num_input_tokens']} → {analysis['num_selected_tokens']}

Token Importance Scores:
  Mean:               {analysis['scores_mean']:.4f}
  Std Dev:            {analysis['scores_std']:.4f}
  Min:                {analysis['scores_min']:.4f}
  Max:                {analysis['scores_max']:.4f}
  Range:              {analysis['scores_max'] - analysis['scores_min']:.4f}
"""
    return report

"""
Utilitários para Hybrid Token Reduction.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_shared.training_utils import count_parameters
from hybrid_simple.utils import get_backbone_feature_dim, modify_backbone_for_feature_extraction


def compute_token_reduction_stats(keep_ratio: float, typical_n: int = 10000) -> dict:
    """
    Computa estatísticas de redução de tokens.

    Args:
        keep_ratio: Proporção de tokens a manter
        typical_n: Número típico de tokens de entrada (para imagem, H*W)

    Returns:
        Dict com estatísticas
    """
    k = int(typical_n * keep_ratio)

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

"""
Compare Models - Benchmark comparativo entre os 3 modelos.

Compara:
1. CNN Baseline (pytorch_opt)
2. Hybrid Simple (CNN + Transformer completo)
3. Hybrid Token Reduction (CNN + Transformer com seleção de tokens)

Foco: Trade-off entre desempenho clínico e custo computacional.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import argparse
from typing import List, Dict, Tuple

sys.path.insert(0, '/home/users/bmmorales/projects/hcpa')

from hybrid_shared.metrics import compute_metrics
from hybrid_shared.profiling import PerformanceProfiler
from hybrid_shared.data_loader_bridge import get_data_loaders, get_backbone
from hybrid_shared.training_utils import count_parameters

from hybrid_simple.model import create_hybrid_simple_model
from hybrid_simple.utils import modify_backbone_for_feature_extraction as modify_backbone

from hybrid_token_reduction.model import create_hybrid_token_reduction_model
from hybrid_token_reduction.utils import compute_token_reduction_stats


def get_cnn_baseline(
    backbone_name: str = "inception_v3",
    img_size: int = 299,
    device: str = "cuda",
) -> Tuple[nn.Module, str]:
    """
    Carrega CNN baseline do pytorch_opt.

    Args:
        backbone_name: Nome do backbone
        img_size: Tamanho da imagem
        device: Device

    Returns:
        (modelo, nome descritivo)
    """
    backbone = get_backbone(
        model_name=backbone_name,
        img_size=img_size,
        pretrained=True,
        freeze_backbone=False,
    )

    return backbone.to(device), f"CNN Baseline ({backbone_name})"


def get_hybrid_simple(
    backbone_name: str = "inception_v3",
    backbone_dim: int = 2048,
    img_size: int = 299,
    num_transformer_layers: int = 4,
    num_heads: int = 4,
    device: str = "cuda",
) -> Tuple[nn.Module, str]:
    """
    Carrega Hybrid Simple.
    """
    backbone = get_backbone(
        model_name=backbone_name,
        img_size=img_size,
        pretrained=True,
    )
    backbone = modify_backbone(backbone, backbone_name)

    model = create_hybrid_simple_model(
        backbone=backbone,
        backbone_feature_dim=backbone_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        use_cls_token=False,
        device=device,
    )

    return model, f"Hybrid Simple ({backbone_name})"


def get_hybrid_token_reduction(
    backbone_name: str = "inception_v3",
    backbone_dim: int = 2048,
    img_size: int = 299,
    num_transformer_layers: int = 4,
    num_heads: int = 4,
    keep_ratio: float = 0.5,
    device: str = "cuda",
) -> Tuple[nn.Module, str]:
    """
    Carrega Hybrid Token Reduction.
    """
    backbone = get_backbone(
        model_name=backbone_name,
        img_size=img_size,
        pretrained=True,
    )
    backbone = modify_backbone(backbone, backbone_name)

    model = create_hybrid_token_reduction_model(
        backbone=backbone,
        backbone_feature_dim=backbone_dim,
        keep_ratio=keep_ratio,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        use_cls_token=False,
        device=device,
    )

    return model, f"Hybrid Token Reduction (keep_ratio={keep_ratio})"


def benchmark_model(
    model: nn.Module,
    loader,
    device: torch.device,
    num_batches: int = None,
    enable_amp: bool = True,
) -> Dict:
    """
    Executa benchmark no modelo.

    Args:
        model: Modelo
        loader: DataLoader
        device: Device
        num_batches: Número de batches para benchmark (None = todos)
        enable_amp: Usar AMP

    Returns:
        Dict com métricas de performance
    """
    model.eval()

    profiler = PerformanceProfiler(device=str(device))

    with torch.no_grad():
        profiler.start_epoch()

        for batch_idx, batch in enumerate(loader):
            if num_batches and batch_idx >= num_batches:
                break

            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = batch[1]

            images = images.to(device)
            labels = labels.to(device)

            profiler.start_batch()

            if enable_amp:
                from torch.cuda.amp import autocast
                with autocast(dtype=torch.float16):
                    _ = model(images)
            else:
                _ = model(images)

            profiler.end_batch(images.size(0))

        epoch_metrics = profiler.end_epoch()

    return epoch_metrics


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    enable_amp: bool = True,
) -> Dict:
    """
    Avalia métricas clínicas do modelo.

    Args:
        model: Modelo
        loader: DataLoader
        device: Device
        enable_amp: Usar AMP

    Returns:
        Dict com AUC, sensibilidade, especificidade
    """
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = batch[1]

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(-1)

            if enable_amp:
                from torch.cuda.amp import autocast
                with autocast(dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits.squeeze(-1)).cpu()
            all_targets.append(labels.squeeze(-1).cpu())
            all_probs.append(probs)

    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    metrics = compute_metrics(all_targets, all_probs)
    return metrics


def create_comparison_report(results: List[Dict]) -> str:
    """
    Cria relatório formatado de comparação.

    Args:
        results: Lista de dicts com resultados

    Returns:
        String formatada
    """
    df = pd.DataFrame(results)

    report = """
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                    COMPARAÇÃO DE MODELOS - RETINOPATIA DIABÉTICA                                        ║
║                    Trade-off: Desempenho Clínico vs. Custo Computacional                                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

"""

    # Tabela de desempenho clínico
    report += "MÉTRICAS CLÍNICAS\n"
    report += "─" * 90 + "\n"
    clinical_cols = ["Model", "AUC", "Sensitivity", "Specificity", "Parameters"]
    if all(col in df.columns for col in clinical_cols):
        report += df[clinical_cols].to_string(index=False)
    report += "\n\n"

    # Tabela de desempenho computacional
    report += "PERFORMANCE COMPUTACIONAL\n"
    report += "─" * 90 + "\n"
    perf_cols = ["Model", "Time/Epoch (s)", "Throughput (img/s)", "Memory Avg (MB)"]
    if all(col in df.columns for col in perf_cols):
        report += df[perf_cols].to_string(index=False)
    report += "\n\n"

    # Tabela de eficiência
    report += "MÉTRICAS DE EFICIÊNCIA\n"
    report += "─" * 90 + "\n"
    if "AUC/Time" in df.columns and "AUC/Memory" in df.columns:
        eff_cols = ["Model", "AUC/Time", "AUC/Memory (MB)", "Score Eficiência"]
        report += df[eff_cols].to_string(index=False)
    report += "\n\n"

    # Ranking
    report += "RANKING\n"
    report += "─" * 90 + "\n"

    if "AUC" in df.columns:
        auc_rank = df.sort_values("AUC", ascending=False)
        report += "\n🏆 Por AUC:\n"
        for i, (_, row) in enumerate(auc_rank.iterrows(), 1):
            report += f"  {i}. {row['Model']:<40} AUC={row['AUC']:.4f}\n"

    if "Throughput (img/s)" in df.columns:
        throughput_rank = df.sort_values("Throughput (img/s)", ascending=False)
        report += "\n⚡ Por Throughput:\n"
        for i, (_, row) in enumerate(throughput_rank.iterrows(), 1):
            report += f"  {i}. {row['Model']:<40} {row['Throughput (img/s)']:.1f} img/s\n"

    if "AUC/Time" in df.columns:
        eff_rank = df.sort_values("AUC/Time", ascending=False)
        report += "\n💡 Por Eficiência (AUC/Time):\n"
        for i, (_, row) in enumerate(eff_rank.iterrows(), 1):
            report += f"  {i}. {row['Model']:<40} {row['AUC/Time']:.6f} AUC/s\n"

    report += "\n"
    return report


def main():
    parser = argparse.ArgumentParser(description="Comparação de modelos")
    parser.add_argument("--tfrec_dir", type=str,
                       default="/home/users/bmmorales/projects/hcpa/pytorch_opt/data/all-tfrec",
                       help="Diretório com TFRecords")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=299, help="Tamanho da imagem")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers")
    parser.add_argument("--backbone", type=str, default="inception_v3", help="Backbone")
    parser.add_argument("--enable_amp", action="store_true", default=True, help="Enable AMP")
    parser.add_argument("--benchmark_batches", type=int, default=50,
                       help="Número de batches para benchmark (None = todos)")
    parser.add_argument("--output_dir", type=str, default="/home/users/bmmorales/projects/hcpa/comparison_results",
                       help="Diretório para salvar resultados")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*90)
    print("COMPARAÇÃO DE MODELOS - RETINOPATIA DIABÉTICA")
    print("="*90)
    print(f"\nDevice: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch Size: {args.batch_size}")

    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)

    # Carregar dataloaders
    print("\nCarregando dataloaders...")
    _, val_loader, test_loader = get_data_loaders(
        tfrec_dir=args.tfrec_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        dataset_name="all",
        use_dali=True,
        augment=False,
        num_workers=args.num_workers,
        rank=0,
        world_size=1,
        enable_amp=args.enable_amp,
    )

    results = []

    # 1. CNN Baseline
    print("\n" + "="*90)
    print("1. CNN BASELINE")
    print("="*90)

    cnn_model, cnn_name = get_cnn_baseline(args.backbone, args.img_size, device)
    print(f"Modelo: {cnn_name}")
    print(f"Parâmetros: {count_parameters(cnn_model, trainable_only=True):,}")

    print("Benchmark...")
    cnn_perf = benchmark_model(cnn_model, val_loader, device, args.benchmark_batches, args.enable_amp)

    print("Avaliação...")
    cnn_metrics = evaluate_model(cnn_model, test_loader, device, args.enable_amp)

    cnn_result = {
        "Model": cnn_name,
        "AUC": cnn_metrics["auc"],
        "Sensitivity": cnn_metrics["sensitivity"],
        "Specificity": cnn_metrics["specificity"],
        "Parameters": count_parameters(cnn_model, trainable_only=True),
        "Time/Epoch (s)": cnn_perf["epoch_time"],
        "Throughput (img/s)": cnn_perf["throughput"],
        "Memory Avg (MB)": cnn_perf["memory_avg_mb"],
    }

    # 2. Hybrid Simple
    print("\n" + "="*90)
    print("2. HYBRID SIMPLE")
    print("="*90)

    simple_model, simple_name = get_hybrid_simple(args.backbone, img_size=args.img_size, device=device)
    print(f"Modelo: {simple_name}")
    print(f"Parâmetros: {count_parameters(simple_model, trainable_only=True):,}")

    print("Benchmark...")
    simple_perf = benchmark_model(simple_model, val_loader, device, args.benchmark_batches, args.enable_amp)

    print("Avaliação...")
    simple_metrics = evaluate_model(simple_model, test_loader, device, args.enable_amp)

    simple_result = {
        "Model": simple_name,
        "AUC": simple_metrics["auc"],
        "Sensitivity": simple_metrics["sensitivity"],
        "Specificity": simple_metrics["specificity"],
        "Parameters": count_parameters(simple_model, trainable_only=True),
        "Time/Epoch (s)": simple_perf["epoch_time"],
        "Throughput (img/s)": simple_perf["throughput"],
        "Memory Avg (MB)": simple_perf["memory_avg_mb"],
    }

    # 3. Hybrid Token Reduction (múltiplas keep_ratios)
    keep_ratios = [0.75, 0.5, 0.25]
    token_reduction_results = []

    for keep_ratio in keep_ratios:
        print("\n" + "="*90)
        print(f"3.{keep_ratios.index(keep_ratio)+1}. HYBRID TOKEN REDUCTION (keep_ratio={keep_ratio})")
        print("="*90)

        token_model, token_name = get_hybrid_token_reduction(
            args.backbone, img_size=args.img_size,
            keep_ratio=keep_ratio, device=device
        )
        print(f"Modelo: {token_name}")
        print(f"Parâmetros: {count_parameters(token_model, trainable_only=True):,}")

        # Token info
        token_stats = compute_token_reduction_stats(keep_ratio)
        print(f"Redução de complexidade: {token_stats['complexity_reduction_factor']:.1f}x")
        print(f"Economia de memória: {token_stats['memory_savings_percent']:.1f}%")

        print("Benchmark...")
        token_perf = benchmark_model(token_model, val_loader, device, args.benchmark_batches, args.enable_amp)

        print("Avaliação...")
        token_metrics = evaluate_model(token_model, test_loader, device, args.enable_amp)

        token_result = {
            "Model": token_name,
            "AUC": token_metrics["auc"],
            "Sensitivity": token_metrics["sensitivity"],
            "Specificity": token_metrics["specificity"],
            "Parameters": count_parameters(token_model, trainable_only=True),
            "Time/Epoch (s)": token_perf["epoch_time"],
            "Throughput (img/s)": token_perf["throughput"],
            "Memory Avg (MB)": token_perf["memory_avg_mb"],
        }

        token_reduction_results.append(token_result)

    # Compilar resultados
    results.append(cnn_result)
    results.append(simple_result)
    results.extend(token_reduction_results)

    # Calcular métricas derivadas
    for result in results:
        result["AUC/Time"] = result["AUC"] / (result["Time/Epoch (s)"] + 1e-7)
        result["AUC/Memory (MB)"] = result["AUC"] / (result["Memory Avg (MB)"] + 1e-7)
        result["Score Eficiência"] = (
            result["AUC"] / result["Time/Epoch (s)"] *
            (5000 / result["Memory Avg (MB)"])  # Normalize by typical memory
        )

    # Imprimir relatório
    report = create_comparison_report(results)
    print(report)

    # Salvar relatório
    report_path = os.path.join(args.output_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Relatório salvo em: {report_path}")

    # Salvar tabela CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tabela CSV salva em: {csv_path}")

    # Resumo em JSON
    import json
    json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON salvo em: {json_path}")

    print("\n✅ Comparação completa!")


if __name__ == "__main__":
    main()

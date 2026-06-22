"""
Avaliação do modelo Hybrid Token Reduction.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_token_reduction.model import HybridTokenReduction
from hybrid_token_reduction.config import get_args
from hybrid_token_reduction.utils import analyze_token_selection, format_token_analysis_report

from hybrid_shared.metrics import compute_metrics, find_optimal_threshold
from hybrid_shared.data_loader_bridge import get_data_loaders, get_backbone, unpack_batch
from hybrid_simple.utils import modify_backbone_for_feature_extraction


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    enable_amp: bool = True,
    collect_token_analysis: bool = False,
) -> dict:
    """
    Avalia modelo com análise opcional de token selection.

    Args:
        model: Modelo
        loader: DataLoader
        device: Device
        enable_amp: Usar AMP
        collect_token_analysis: Coletar análise de tokens (mais lento)

    Returns:
        Dictionary com métricas
    """
    model.eval()

    all_targets = []
    all_probs = []
    token_analyses = []
    amp_enabled = enable_amp and device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, labels = unpack_batch(batch)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            if amp_enabled:
                with autocast(dtype=torch.float16):
                    if collect_token_analysis and hasattr(model, 'forward_with_token_analysis'):
                        logits, analysis = model.forward_with_token_analysis(images)
                    else:
                        logits = model(images)
            else:
                if collect_token_analysis and hasattr(model, 'forward_with_token_analysis'):
                    logits, analysis = model.forward_with_token_analysis(images)
                else:
                    logits = model(images)

            logits = logits.view(-1)
            probs = torch.sigmoid(logits)

            all_targets.append(labels)
            all_probs.append(probs)

            if collect_token_analysis and 'analysis' in locals():
                token_analyses.append(analysis)

            print(f"  Batch {batch_idx + 1}/{len(loader)}")

    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    metrics = compute_metrics(all_targets, all_probs)

    opt_threshold, opt_f1 = find_optimal_threshold(all_targets, all_probs, metric="f1")
    metrics["optimal_threshold"] = opt_threshold
    metrics["optimal_f1"] = opt_f1

    if token_analyses:
        # Agregar análises
        avg_compression = np.mean([a['compression_ratio'] for a in token_analyses])
        metrics["avg_compression_ratio"] = avg_compression

    return metrics


def main():
    """Main evaluation script."""
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("HYBRID TOKEN REDUCTION - EVALUATION")
    print("="*70)

    # Load checkpoint
    checkpoint_path = os.path.join(args.results_dir, f"best_model_exec{args.exec_id}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Create model
    backbone = get_backbone(
        model_name=args.backbone,
        img_size=args.img_size,
        pretrained=True,
    )
    backbone = modify_backbone_for_feature_extraction(backbone, args.backbone)

    model = HybridTokenReduction(
        backbone=backbone,
        d_model=args.backbone_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        keep_ratio=args.keep_ratio,
        keep_k=args.keep_k,
        use_cls_token=args.use_cls_token,
    ).to(device)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Load dataloaders
    print("Loading dataloaders...")
    _, val_loader, test_loader = get_data_loaders(
        tfrec_dir=args.tfrec_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        dataset_name=args.dataset,
        use_dali=args.enable_dali,
        augment=False,
        num_workers=args.num_workers,
        rank=0,
        world_size=1,
        enable_amp=args.enable_amp,
        model_name=args.backbone,
    )

    # Evaluate validation
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, args.enable_amp, collect_token_analysis=False)

    print(f"\n{'Validation Results':-^70}")
    print(f"AUC: {val_metrics['auc']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall: {val_metrics['recall']:.4f}")
    print(f"Specificity: {val_metrics['specificity']:.4f}")
    print(f"F1: {val_metrics['f1']:.4f}")
    print(f"Optimal Threshold: {val_metrics['optimal_threshold']:.4f}")
    print(f"Optimal F1: {val_metrics['optimal_f1']:.4f}")

    # Evaluate test
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, args.enable_amp, collect_token_analysis=True)

    print(f"\n{'Test Results':-^70}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Optimal Threshold: {test_metrics['optimal_threshold']:.4f}")
    print(f"Optimal F1: {test_metrics['optimal_f1']:.4f}")

    if "avg_compression_ratio" in test_metrics:
        print(f"Average Compression Ratio: {test_metrics['avg_compression_ratio']:.1f}x")

    # Save results
    results_file = os.path.join(args.results_dir, f"eval_results_exec{args.exec_id}.txt")
    with open(results_file, 'w') as f:
        f.write("HYBRID TOKEN REDUCTION - EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Keep Ratio: {args.keep_ratio}\n")
        f.write(f"Keep K: {args.keep_k}\n\n")

        f.write("VALIDATION SET\n")
        f.write(f"AUC: {val_metrics['auc']:.4f}\n")
        f.write(f"Precision: {val_metrics['precision']:.4f}\n")
        f.write(f"Recall: {val_metrics['recall']:.4f}\n")
        f.write(f"Specificity: {val_metrics['specificity']:.4f}\n")
        f.write(f"F1: {val_metrics['f1']:.4f}\n")
        f.write(f"Optimal Threshold: {val_metrics['optimal_threshold']:.4f}\n")
        f.write(f"Optimal F1: {val_metrics['optimal_f1']:.4f}\n\n")

        f.write("TEST SET\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Specificity: {test_metrics['specificity']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")
        f.write(f"Optimal Threshold: {test_metrics['optimal_threshold']:.4f}\n")
        f.write(f"Optimal F1: {test_metrics['optimal_f1']:.4f}\n")

        if "avg_compression_ratio" in test_metrics:
            f.write(f"Average Compression Ratio: {test_metrics['avg_compression_ratio']:.1f}x\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

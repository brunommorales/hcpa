"""Evaluation script for RETFound-Green fine-tuning."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retfound_green.config import get_args
from retfound_green.model import RetFoundGreenClassifier
from retfound_green.utils import validate_retfound_green_args

from hybrid_shared.data_loader_bridge_opt import get_data_loaders
from hybrid_shared.metrics import compute_metrics, find_optimal_threshold
from hybrid_shared.training_utils_opt import configure_cuda_runtime


# RETFound-Green uses mean/std = 0.5 on [0, 1], numerically identical to the
# Inception-style normalization already supported by the shared loader.
RETFOUND_GREEN_NORMALIZATION_MODEL = "inception_v3"


def unwrap_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch["label"]
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 1 and isinstance(batch[0], dict):
            images = batch[0]["image"]
            labels = batch[0]["label"]
        elif len(batch) == 2 and not isinstance(batch[0], dict):
            images, labels = batch
        elif len(batch) > 0 and isinstance(batch[0], dict):
            images = batch[0]["image"]
            labels = batch[0]["label"]
        else:
            raise TypeError(f"Unsupported batch structure: {type(batch)}")
    else:
        raise TypeError(f"Unsupported batch structure: {type(batch)}")

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).float().view(-1)
    return images, labels


def normalize_state_dict_keys(state_dict: dict) -> dict:
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    enable_amp: bool = True,
    collect_patch_analysis: bool = False,
) -> dict:
    model.eval()

    all_targets = []
    all_probs = []
    patch_analyses = []
    amp_enabled = enable_amp and device.type == "cuda"

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, labels = unwrap_batch(batch, device)

            if amp_enabled:
                with autocast(dtype=torch.float16):
                    if collect_patch_analysis and hasattr(model, "forward_with_patch_analysis"):
                        logits, analysis = model.forward_with_patch_analysis(images)
                    else:
                        logits = model(images)
                        analysis = None
            else:
                if collect_patch_analysis and hasattr(model, "forward_with_patch_analysis"):
                    logits, analysis = model.forward_with_patch_analysis(images)
                else:
                    logits = model(images)
                    analysis = None

            probs = torch.sigmoid(logits.squeeze(-1))
            all_targets.append(labels)
            all_probs.append(probs)

            if analysis is not None:
                patch_analyses.append(analysis)

            print(f"  Batch {batch_idx + 1}/{len(loader)}")

    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    metrics = compute_metrics(all_targets, all_probs)
    opt_threshold, opt_f1 = find_optimal_threshold(all_targets, all_probs, metric="f1")
    metrics["optimal_threshold"] = opt_threshold
    metrics["optimal_f1"] = opt_f1

    if patch_analyses:
        metrics["token_source"] = patch_analyses[0]["token_source"]
        metrics["patch_size"] = patch_analyses[0]["patch_size"]
        metrics["num_patches"] = patch_analyses[0]["num_patches"]
        metrics["sequence_length"] = patch_analyses[0]["sequence_length"]
        metrics["feature_dim"] = patch_analyses[0]["feature_dim"]
        metrics["backbone_model"] = patch_analyses[0]["backbone_model"]

    return metrics


def main() -> None:
    args = get_args()
    validate_retfound_green_args(args.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_runtime()

    print("=" * 70)
    print("RETFound-Green - EVALUATION")
    print("=" * 70)

    checkpoint_path = os.path.join(args.results_dir, f"best_model_exec{args.exec_id}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    model = RetFoundGreenClassifier(
        img_size=args.img_size,
        backbone_model=args.backbone_model,
        checkpoint_path=None,
        head_dropout=args.head_dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = normalize_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)

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
        model_name=RETFOUND_GREEN_NORMALIZATION_MODEL,
    )

    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, args.enable_amp, collect_patch_analysis=False)

    print(f"\n{'Validation Results':-^70}")
    print(f"AUC: {val_metrics['auc']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall: {val_metrics['recall']:.4f}")
    print(f"Specificity: {val_metrics['specificity']:.4f}")
    print(f"F1: {val_metrics['f1']:.4f}")
    print(f"Optimal Threshold: {val_metrics['optimal_threshold']:.4f}")
    print(f"Optimal F1: {val_metrics['optimal_f1']:.4f}")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, args.enable_amp, collect_patch_analysis=True)

    print(f"\n{'Test Results':-^70}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Specificity: {test_metrics['specificity']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Optimal Threshold: {test_metrics['optimal_threshold']:.4f}")
    print(f"Optimal F1: {test_metrics['optimal_f1']:.4f}")
    if "token_source" in test_metrics:
        print(f"Token Source: {test_metrics['token_source']}")
        print(f"Patch Size: {test_metrics['patch_size']}")
        print(f"Num Patches: {test_metrics['num_patches']}")
        print(f"Sequence Length: {test_metrics['sequence_length']}")
        print(f"Feature Dim: {test_metrics['feature_dim']}")
        print(f"Backbone Model: {test_metrics['backbone_model']}")

    results_file = os.path.join(args.results_dir, f"eval_results_exec{args.exec_id}.txt")
    with open(results_file, "w", encoding="utf-8") as handle:
        handle.write("RETFound-Green - EVALUATION RESULTS\n")
        handle.write("=" * 70 + "\n\n")
        handle.write("VALIDATION SET\n")
        handle.write(f"AUC: {val_metrics['auc']:.4f}\n")
        handle.write(f"Precision: {val_metrics['precision']:.4f}\n")
        handle.write(f"Recall: {val_metrics['recall']:.4f}\n")
        handle.write(f"Specificity: {val_metrics['specificity']:.4f}\n")
        handle.write(f"F1: {val_metrics['f1']:.4f}\n")
        handle.write(f"Optimal Threshold: {val_metrics['optimal_threshold']:.4f}\n")
        handle.write(f"Optimal F1: {val_metrics['optimal_f1']:.4f}\n\n")

        handle.write("TEST SET\n")
        handle.write(f"AUC: {test_metrics['auc']:.4f}\n")
        handle.write(f"Precision: {test_metrics['precision']:.4f}\n")
        handle.write(f"Recall: {test_metrics['recall']:.4f}\n")
        handle.write(f"Specificity: {test_metrics['specificity']:.4f}\n")
        handle.write(f"F1: {test_metrics['f1']:.4f}\n")
        handle.write(f"Optimal Threshold: {test_metrics['optimal_threshold']:.4f}\n")
        handle.write(f"Optimal F1: {test_metrics['optimal_f1']:.4f}\n")
        if "token_source" in test_metrics:
            handle.write(f"Token Source: {test_metrics['token_source']}\n")
            handle.write(f"Patch Size: {test_metrics['patch_size']}\n")
            handle.write(f"Num Patches: {test_metrics['num_patches']}\n")
            handle.write(f"Sequence Length: {test_metrics['sequence_length']}\n")
            handle.write(f"Feature Dim: {test_metrics['feature_dim']}\n")
            handle.write(f"Backbone Model: {test_metrics['backbone_model']}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

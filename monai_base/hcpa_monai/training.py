from __future__ import annotations

import csv
import json
import os
import time
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from .config import TrainConfig
from .models import build_model
from .data import create_loaders
from .utils import (
    ensure_dir,
    get_device,
    maybe_compile,
    move_to_device,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    to_channels_last,
    apply_ema,
    update_ema,
)
from .metrics import EpochStats


@dataclass
class BaselineDefaults:
    """Preset baseline sem otimizações (para comparação com versão optimized)."""

    def to_config(self, results_dir: Path, tfrec_dir: Path) -> TrainConfig:
        return TrainConfig(
            results_dir=results_dir,
            tfrec_dir=tfrec_dir,
            image_size=299,
            model_name="inception_v3",
            pretrained=True,
            dropout=0.2,
            batch_size=96,
            eval_batch_size=96,
            epochs=200,
            learning_rate=3e-4,
            min_lr=0.0,
            warmup_epochs=0,
            weight_decay=1e-4,
            scheduler="none",
            grad_clip_norm=1.0,
            augment=True,
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            label_smoothing=0.0,
            fundus_crop_ratio=0.9,
            normalize="inception",
            channels_last=True,
            amp=True,
            compile=False,
            use_dali=False,
            num_workers=8,
            host_prefetch=2,
            device_prefetch=2,
            log_every=50,
            save_every=1,
            ema_decay=0.0,
            ema_on_cpu=False,
            gradient_accumulation=1,
            seed=2026,
        )


def _init_distributed() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _class_weights(pos_weight: Optional[float], device: torch.device, num_classes: int) -> Optional[torch.Tensor]:
    if pos_weight is None:
        return None
    if num_classes == 2:
        return torch.tensor([1.0, pos_weight], device=device)
    w = torch.ones(num_classes, device=device)
    w[-1] = pos_weight
    return w


def _loss_fn(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    soft_targets: Optional[torch.Tensor],
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    if soft_targets is not None:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()
    return F.cross_entropy(logits, labels.long(), weight=class_weights, label_smoothing=label_smoothing)


def _prob_positive(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=-1)[:, 1]


def _mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return images, hard_labels, soft_targets."""
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, labels, None

    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    labels_onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
    mixed_targets = labels_onehot.clone()
    lam = 1.0

    if cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[perm, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        mixed_targets = lam * labels_onehot + (1 - lam) * labels_onehot[perm]
    elif mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images[perm]
        mixed_targets = lam * labels_onehot + (1 - lam) * labels_onehot[perm]

    hard_labels = labels
    return images, hard_labels, mixed_targets


def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def _build_scheduler(cfg: TrainConfig, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    return None


def train_and_evaluate(cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    rank, world_size = _init_distributed()
    device = get_device()

    train_loader, eval_loader, meta = create_loaders(cfg, rank=rank, world_size=world_size)
    steps_per_epoch = max(1, meta.get("train_items", cfg.batch_size * 10) // cfg.batch_size)

    model = build_model(cfg)
    model = to_channels_last(model, cfg.channels_last)
    model = model.to(device)
    model = maybe_compile(model, cfg.compile)
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if device.type == "cuda" else None, static_graph=True)

    fused = bool(torch.cuda.is_available())
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=fused)
    scheduler = _build_scheduler(cfg, optimizer, steps_per_epoch)
    scaler = GradScaler(enabled=cfg.amp)
    class_weights = _class_weights(cfg.pos_weight, device, cfg.num_classes)

    ema_state: Dict[str, torch.Tensor] = {}
    best_auc = -float("inf")
    # value used to decide whether to save a checkpoint; falls back to val loss when AUC is NaN
    best_selector = -float("inf")
    best_epoch = 0
    ensure_dir(cfg.results_dir)

    metrics_csv = cfg.results_dir / "metrics.csv"
    ckpt_path = cfg.results_dir / "checkpoint.pt"
    if rank == 0:
        with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "epoch",
                "stage",
                "train_loss",
                "train_auc",
                "train_sens",
                "train_spec",
                "train_accuracy",
                "train_throughput_img_s",
                "train_gpu_mem_alloc_mb",
                "train_gpu_mem_reserved_mb",
                "val_loss",
                "val_auc",
                "val_sens",
                "val_spec",
                "val_accuracy",
                "val_throughput_img_s",
                "val_gpu_mem_alloc_mb",
                "val_gpu_mem_reserved_mb",
                "lr",
                "elapsed_s",
            ])
            writer.writeheader()

    start_time = time.perf_counter()
    last_eval_arrays = (np.array([]), np.array([]))  # probs, labels
    last_eval_arrays = (np.array([]), np.array([]))  # probs, labels
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        train_stats = EpochStats()
        t_epoch = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            batch = move_to_device(batch, device)
            images = batch["image"]
            labels = batch["label"].view(-1)

            images, labels, soft_targets = _mixup_cutmix(
                images,
                labels,
                cfg.num_classes,
                cfg.mixup_alpha,
                cfg.cutmix_alpha,
            )

            with autocast(enabled=cfg.amp):
                logits = model(images)
                loss = _loss_fn(
                    logits,
                    labels,
                    soft_targets=soft_targets,
                    class_weights=class_weights,
                    label_smoothing=cfg.label_smoothing,
                )
                loss = loss / cfg.gradient_accumulation

            scaler.scale(loss).backward()

            if step % cfg.gradient_accumulation == 0:
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if cfg.ema_decay > 0:
                    update_ema(ema_state, model.module if isinstance(model, DDP) else model, cfg.ema_decay)
                if scheduler is not None and cfg.scheduler != "onecycle":
                    scheduler.step()

            probs = _prob_positive(logits.detach())
            train_stats.update(float(loss.detach()) * cfg.gradient_accumulation, probs.cpu().numpy(), labels.cpu().numpy())
            global_step += 1

            if cfg.log_every and global_step % cfg.log_every == 0 and rank == 0:
                print(f"[opt][epoch {epoch} step {global_step}] loss={loss.item()*cfg.gradient_accumulation:.4f}")

        train_elapsed = time.perf_counter() - t_epoch
        train_metrics = train_stats.aggregate(threshold=cfg.threshold)
        train_throughput = meta.get("train_items", cfg.batch_size * steps_per_epoch) / max(train_elapsed, 1e-6)

        # eval (optionally with EMA)
        model.eval()
        backup_state = None
        if cfg.ema_decay > 0 and ema_state and cfg.ema_on_cpu is False:
            backup_state = apply_ema(model.module if isinstance(model, DDP) else model, ema_state)

        eval_stats = EpochStats()
        with torch.inference_mode(), autocast(enabled=cfg.amp):
            for batch in eval_loader:
                batch = move_to_device(batch, device)
                logits = model(batch["image"])
                loss = _loss_fn(
                    logits,
                    batch["label"].view(-1),
                    soft_targets=None,
                    class_weights=class_weights,
                    label_smoothing=0.0,
                )
                probs = _prob_positive(logits)
                eval_stats.update(float(loss), probs.cpu().numpy(), batch["label"].view(-1).cpu().numpy())
        eval_metrics = eval_stats.aggregate(threshold=cfg.threshold)
        last_eval_arrays = eval_stats.stack()
        val_throughput = meta.get("eval_items", cfg.eval_batch_size * 10) / max((time.perf_counter() - t_epoch), 1e-6)

        # Quick sanity check: flag single-class validation splits that make AUC undefined
        if rank == 0 and last_eval_arrays[1].size > 0:
            labels_arr = last_eval_arrays[1]
            pos = int(np.sum(labels_arr == 1))
            neg = int(np.sum(labels_arr == 0))
            if pos == 0 or neg == 0:
                print(
                    f"[opt][epoch {epoch}] WARNING: validation split missing {'positives' if pos == 0 else 'negatives'} "
                    f"(pos={pos}, neg={neg}); AUC will be NaN."
                )

        if backup_state is not None:
            # restore original weights
            apply_ema(model.module if isinstance(model, DDP) else model, backup_state)

        lr_value = optimizer.param_groups[0]["lr"]
        if scheduler is not None and cfg.scheduler == "onecycle":
            scheduler.step()

        if rank == 0:
            row = {
                "epoch": epoch,
                "stage": "train",
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "train_sens": train_metrics["sensitivity"],
                "train_spec": train_metrics["specificity"],
                "train_accuracy": train_metrics["accuracy"],
                "train_throughput_img_s": train_throughput,
                "train_gpu_mem_alloc_mb": _gpu_mem_mb()[0],
                "train_gpu_mem_reserved_mb": _gpu_mem_mb()[1],
                "val_loss": eval_metrics["loss"],
                "val_auc": eval_metrics["auc"],
                "val_sens": eval_metrics["sensitivity"],
                "val_spec": eval_metrics["specificity"],
                "val_accuracy": eval_metrics["accuracy"],
                "val_throughput_img_s": val_throughput,
                "val_gpu_mem_alloc_mb": _gpu_mem_mb()[0],
                "val_gpu_mem_reserved_mb": _gpu_mem_mb()[1],
                "lr": lr_value,
                "elapsed_s": time.perf_counter() - start_time,
            }
            with metrics_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writerow(row)

            val_auc = eval_metrics["auc"]
            # When AUC is NaN (single-class validation), fall back to minimizing validation loss
            selector_metric = val_auc if math.isfinite(val_auc) else -eval_metrics["loss"]
            improved = selector_metric > best_selector

            # Always save something so eval.py can run even if AUC stays NaN
            if improved or (epoch == 1 and not ckpt_path.exists()):
                best_selector = selector_metric
                if math.isfinite(val_auc):
                    best_auc = val_auc
                    best_epoch = epoch
                save_checkpoint(
                    ckpt_path,
                    model_state=model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    scaler_state=scaler.state_dict() if cfg.amp else None,
                    epoch=epoch,
                    best_metric=best_auc,
                    config=asdict(cfg),
                )
            print(
                f"[opt][epoch {epoch}] train_loss={train_metrics['loss']:.4f} "
                f"val_loss={eval_metrics['loss']:.4f} val_auc={eval_metrics['auc']:.4f} "
                f"best_auc={best_auc:.4f} (epoch {best_epoch})"
            )

    final = {
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "train_items": meta.get("train_items", 0),
        "eval_items": meta.get("eval_items", 0),
    }
    if rank == 0:
        _save_roc_plots(cfg.results_dir, last_eval_arrays, prefix="val")
        (cfg.results_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return final


def evaluate_checkpoint(results_dir: Path) -> Dict[str, float]:
    ckpt_path = Path(results_dir) / "checkpoint.pt"
    payload = load_checkpoint(ckpt_path, map_location=get_device())
    cfg = TrainConfig(**payload.get("config", {}))
    model = build_model(cfg).to(get_device())
    model.load_state_dict(payload["model"])

    cfg.use_fake_data = cfg.use_fake_data or not cfg.tfrec_dir.exists()
    cfg.batch_size = cfg.eval_batch_size

    _, eval_loader, meta = create_loaders(cfg, rank=0, world_size=1)
    model.eval()
    eval_stats = EpochStats()
    with torch.inference_mode(), autocast(enabled=cfg.amp):
        for batch in eval_loader:
            batch = move_to_device(batch, get_device())
            logits = model(batch["image"])
            loss = _loss_fn(logits, batch["label"].view(-1), soft_targets=None, class_weights=None, label_smoothing=0.0)
            probs = _prob_positive(logits)
            eval_stats.update(float(loss), probs.cpu().numpy(), batch["label"].view(-1).cpu().numpy())
    metrics = eval_stats.aggregate(threshold=cfg.threshold)
    metrics["eval_items"] = meta.get("eval_items", 0)
    _save_roc_plots(Path(results_dir), eval_stats.stack(), prefix="val")
    return metrics


def benchmark(cfg: TrainConfig, *, warmup_steps: int = 20, measure_steps: int = 200) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = get_device()
    loader, _, _ = create_loaders(cfg, rank=0, world_size=1)
    model = build_model(cfg).to(device)
    model.eval()

    it = iter(loader)
    times: List[float] = []
    with torch.inference_mode(), autocast(enabled=cfg.amp):
        for _ in range(warmup_steps):
            batch = move_to_device(next(it), device)
            _ = model(batch["image"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(measure_steps):
            batch = move_to_device(next(it), device)
            t0 = time.perf_counter()
            _ = model(batch["image"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    mean_t = float(np.mean(times))
    throughput = cfg.batch_size / mean_t if mean_t > 0 else float("nan")
    return {"latency_s": mean_t, "throughput_img_s": throughput}


def _gpu_mem_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    return float(alloc), float(reserved)


def _save_roc_plots(results_dir: Path, arrays: tuple[np.ndarray, np.ndarray], prefix: str) -> None:
    probs, labels = arrays
    if probs.size == 0:
        return
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    sens = tpr
    spec = 1.0 - fpr
    results_dir.mkdir(parents=True, exist_ok=True)
    # thresholds CSV
    with (results_dir / f"{prefix}_thresholds.csv").open("w", encoding="utf-8") as fh:
        fh.write("thresholds,fpr,tpr,sens,spec\n")
        for th, fp, tp, se, sp in zip(thresholds, fpr, tpr, sens, spec):
            fh.write(f"{th},{fp},{tp},{se},{sp}\n")
    # ROC PDF
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.savefig(results_dir / f"{prefix}_roc.pdf", format="pdf", bbox_inches="tight")
    plt.close()


__all__ = [
    "train_and_evaluate",
    "evaluate_checkpoint",
    "benchmark",
    "BaselineDefaults",
]

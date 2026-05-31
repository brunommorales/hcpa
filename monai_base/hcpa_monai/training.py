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
from .metrics import EpochStats, compute_binary_metrics


COMMON_CSV_FIELDS = [
    "epoch",
    "stage",
    "train_loss",
    "train_auc",
    "train_precision",
    "train_recall",
    "train_f1",
    "train_sens",
    "train_spec",
    "train_spec_at_sens95",
    "train_throughput_img_s",
    "train_elapsed_s",
    "train_avg_batch_time_ms",
    "train_inference_latency_ms_img",
    "train_inference_latency_ms_batch",
    "train_gpu_mem_avg_mb",
    "val_loss",
    "val_auc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_sens",
    "val_spec",
    "val_spec_at_sens95",
    "val_throughput_img_s",
    "val_elapsed_s",
    "val_avg_batch_time_ms",
    "val_inference_latency_ms_img",
    "val_inference_latency_ms_batch",
    "val_gpu_mem_avg_mb",
    "test_loss",
    "test_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_sens",
    "test_spec",
    "test_spec_at_sens95",
    "test_throughput_img_s",
    "test_elapsed_s",
    "test_avg_batch_time_ms",
    "test_inference_latency_ms_img",
    "test_inference_latency_ms_batch",
    "test_gpu_mem_avg_mb",
    "lr",
    "total_train_time_s",
]


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


def _ddp_concat_variable_length(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t.detach().cpu()

    n_local = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(n_local) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, n_local)
    maxn = int(torch.stack(sizes).max().item())
    if maxn == 0:
        return torch.empty(0, dtype=t.dtype)

    pad = maxn - t.shape[0]
    if pad > 0:
        pad_value = -1.0 if t.dtype.is_floating_point else -1
        padding = torch.full((pad,), pad_value, device=t.device, dtype=t.dtype)
        t = torch.cat([t, padding], dim=0)

    buffers = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(buffers, t)
    out = torch.cat(buffers, dim=0)
    mask = out >= (0.0 if out.dtype.is_floating_point else 0)
    return out[mask].detach().cpu()


def _empty_metrics() -> Dict[str, float]:
    return {
        "loss": float("nan"),
        "auc": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "specificity_at_sens95": float("nan"),
        "f1": float("nan"),
        "accuracy": float("nan"),
    }


def _aggregate_epoch_stats(
    stats: EpochStats,
    loss_sum: float,
    sample_count: int,
    *,
    threshold: float,
    device: torch.device,
) -> tuple[Dict[str, float], tuple[np.ndarray, np.ndarray], int]:
    probs_np, labels_np = stats.stack()
    probs_t = torch.as_tensor(probs_np, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    global_probs = _ddp_concat_variable_length(probs_t).numpy()
    global_labels = _ddp_concat_variable_length(labels_t).numpy().astype(np.int32)

    loss_count = torch.tensor([float(loss_sum), float(sample_count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
    global_loss_sum = float(loss_count[0].item())
    global_sample_count = int(loss_count[1].item())

    if global_sample_count <= 0 or global_probs.size == 0:
        return _empty_metrics(), (global_probs, global_labels), global_sample_count

    metrics = compute_binary_metrics(global_probs, global_labels, threshold=threshold)
    metrics["loss"] = global_loss_sum / max(global_sample_count, 1)
    return metrics, (global_probs, global_labels), global_sample_count


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
            writer = csv.DictWriter(handle, fieldnames=COMMON_CSV_FIELDS)
            writer.writeheader()

    start_time = time.perf_counter()
    last_eval_arrays = (np.array([]), np.array([]))  # probs, labels
    last_eval_metrics = None
    last_val_throughput = float("nan")
    last_val_elapsed = float("nan")
    last_val_avg_batch_time_ms = float("nan")
    last_val_inference_latency_ms_img = float("nan")
    last_val_inference_latency_ms_batch = float("nan")
    last_val_mem_avg = float("nan")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        train_stats = EpochStats()
        t_epoch = time.perf_counter()
        train_batches = 0
        train_loss_sum = 0.0
        train_sample_count = 0
        train_memory_samples_mb: list[float] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            train_batches = step
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
            loss_value = float(loss.detach()) * cfg.gradient_accumulation
            batch_size = int(labels.shape[0])
            train_loss_sum += loss_value * batch_size
            train_sample_count += batch_size
            train_stats.update(loss_value, probs.cpu().numpy(), labels.cpu().numpy())
            global_step += 1
            train_memory_samples_mb.append(_gpu_mem_current_mb())

            if cfg.log_every and global_step % cfg.log_every == 0 and rank == 0:
                print(f"[opt][epoch {epoch} step {global_step}] loss={loss.item()*cfg.gradient_accumulation:.4f}")

        train_elapsed = time.perf_counter() - t_epoch
        if world_size > 1:
            elapsed_tensor = torch.tensor([train_elapsed], device=device, dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            train_elapsed = float(elapsed_tensor.item())
        train_metrics, _, train_global_samples = _aggregate_epoch_stats(
            train_stats,
            train_loss_sum,
            train_sample_count,
            threshold=cfg.threshold,
            device=device,
        )
        train_throughput = train_global_samples / max(train_elapsed, 1e-6)
        train_avg_batch_time_ms = (train_elapsed / max(train_batches, 1)) * 1000.0 if train_batches > 0 else 0.0
        train_mem_avg = _mean_valid(train_memory_samples_mb)
        if world_size > 1:
            mem_tensor = torch.tensor(
                [0.0 if math.isnan(train_mem_avg) else train_mem_avg],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.SUM)
            train_mem_avg = float(mem_tensor.item()) / max(world_size, 1)

        # eval (optionally with EMA)
        model.eval()
        backup_state = None
        if cfg.ema_decay > 0 and ema_state and cfg.ema_on_cpu is False:
            backup_state = apply_ema(model.module if isinstance(model, DDP) else model, ema_state)

        eval_stats = EpochStats()
        val_start = time.perf_counter()
        val_batches = 0
        val_loss_sum = 0.0
        val_sample_count = 0
        val_memory_samples_mb: list[float] = []
        val_inference_time_s = 0.0
        val_inference_samples = 0
        with torch.inference_mode(), autocast(enabled=cfg.amp):
            for batch in eval_loader:
                val_batches += 1
                batch = move_to_device(batch, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = time.perf_counter()
                logits = model(batch["image"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                val_inference_time_s += time.perf_counter() - inference_start
                val_inference_samples += int(batch["image"].shape[0])
                labels = batch["label"].view(-1)
                loss = _loss_fn(
                    logits,
                    labels,
                    soft_targets=None,
                    class_weights=class_weights,
                    label_smoothing=0.0,
                )
                probs = _prob_positive(logits)
                loss_value = float(loss)
                batch_size = int(labels.shape[0])
                val_loss_sum += loss_value * batch_size
                val_sample_count += batch_size
                eval_stats.update(loss_value, probs.cpu().numpy(), labels.cpu().numpy())
                val_memory_samples_mb.append(_gpu_mem_current_mb())
        val_elapsed = time.perf_counter() - val_start
        if world_size > 1:
            elapsed_tensor = torch.tensor([val_elapsed], device=device, dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            val_elapsed = float(elapsed_tensor.item())
        eval_metrics, last_eval_arrays, val_global_samples = _aggregate_epoch_stats(
            eval_stats,
            val_loss_sum,
            val_sample_count,
            threshold=cfg.threshold,
            device=device,
        )
        val_throughput = val_global_samples / max(val_elapsed, 1e-6)
        val_avg_batch_time_ms = (val_elapsed / max(val_batches, 1)) * 1000.0 if val_batches > 0 else 0.0
        val_inference_latency_ms_img = (
            val_inference_time_s / val_inference_samples * 1000.0
            if val_inference_samples > 0
            else float("nan")
        )
        val_inference_latency_ms_batch = (
            val_inference_time_s / val_batches * 1000.0 if val_batches > 0 else float("nan")
        )
        val_mem_avg = _mean_valid(val_memory_samples_mb)
        if world_size > 1:
            mem_tensor = torch.tensor(
                [
                    0.0 if math.isnan(val_mem_avg) else val_mem_avg,
                    0.0 if math.isnan(val_inference_latency_ms_img) else val_inference_latency_ms_img,
                    0.0 if math.isnan(val_inference_latency_ms_batch) else val_inference_latency_ms_batch,
                    1.0 if val_inference_samples > 0 else 0.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.SUM)
            val_mem_avg = float(mem_tensor[0].item()) / max(world_size, 1)
            latency_ranks = max(float(mem_tensor[3].item()), 1.0)
            if mem_tensor[3].item() > 0:
                val_inference_latency_ms_img = float(mem_tensor[1].item()) / latency_ranks
                val_inference_latency_ms_batch = float(mem_tensor[2].item()) / latency_ranks

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
            last_eval_metrics = eval_metrics
            last_val_throughput = val_throughput
            last_val_elapsed = val_elapsed
            last_val_avg_batch_time_ms = val_avg_batch_time_ms
            last_val_inference_latency_ms_img = val_inference_latency_ms_img
            last_val_inference_latency_ms_batch = val_inference_latency_ms_batch
            last_val_mem_avg = val_mem_avg
            row = {
                "epoch": epoch,
                "stage": "train",
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_sens": train_metrics["sensitivity"],
                "train_spec": train_metrics["specificity"],
                "train_spec_at_sens95": train_metrics["specificity_at_sens95"],
                "train_throughput_img_s": train_throughput,
                "train_elapsed_s": train_elapsed,
                "train_avg_batch_time_ms": train_avg_batch_time_ms,
                "train_inference_latency_ms_img": float("nan"),
                "train_inference_latency_ms_batch": float("nan"),
                "train_gpu_mem_avg_mb": train_mem_avg,
                "val_loss": eval_metrics["loss"],
                "val_auc": eval_metrics["auc"],
                "val_precision": eval_metrics["precision"],
                "val_recall": eval_metrics["recall"],
                "val_f1": eval_metrics["f1"],
                "val_sens": eval_metrics["sensitivity"],
                "val_spec": eval_metrics["specificity"],
                "val_spec_at_sens95": eval_metrics["specificity_at_sens95"],
                "val_throughput_img_s": val_throughput,
                "val_elapsed_s": val_elapsed,
                "val_avg_batch_time_ms": val_avg_batch_time_ms,
                "val_inference_latency_ms_img": val_inference_latency_ms_img,
                "val_inference_latency_ms_batch": val_inference_latency_ms_batch,
                "val_gpu_mem_avg_mb": val_mem_avg,
                "test_loss": float("nan"),
                "test_auc": float("nan"),
                "test_precision": float("nan"),
                "test_recall": float("nan"),
                "test_f1": float("nan"),
                "test_sens": float("nan"),
                "test_spec": float("nan"),
                "test_spec_at_sens95": float("nan"),
                "test_throughput_img_s": float("nan"),
                "test_elapsed_s": float("nan"),
                "test_avg_batch_time_ms": float("nan"),
                "test_inference_latency_ms_img": float("nan"),
                "test_inference_latency_ms_batch": float("nan"),
                "test_gpu_mem_avg_mb": float("nan"),
                "lr": lr_value,
                "total_train_time_s": float("nan"),
            }
            with metrics_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COMMON_CSV_FIELDS)
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
        if last_eval_metrics is not None:
            final_row = {
                "epoch": cfg.epochs,
                "stage": "final_test",
                "train_loss": float("nan"),
                "train_auc": float("nan"),
                "train_precision": float("nan"),
                "train_recall": float("nan"),
                "train_f1": float("nan"),
                "train_sens": float("nan"),
                "train_spec": float("nan"),
                "train_spec_at_sens95": float("nan"),
                "train_throughput_img_s": float("nan"),
                "train_elapsed_s": float("nan"),
                "train_avg_batch_time_ms": float("nan"),
                "train_inference_latency_ms_img": float("nan"),
                "train_inference_latency_ms_batch": float("nan"),
                "train_gpu_mem_avg_mb": float("nan"),
                "val_loss": float("nan"),
                "val_auc": float("nan"),
                "val_precision": float("nan"),
                "val_recall": float("nan"),
                "val_f1": float("nan"),
                "val_sens": float("nan"),
                "val_spec": float("nan"),
                "val_spec_at_sens95": float("nan"),
                "val_throughput_img_s": float("nan"),
                "val_elapsed_s": float("nan"),
                "val_avg_batch_time_ms": float("nan"),
                "val_inference_latency_ms_img": float("nan"),
                "val_inference_latency_ms_batch": float("nan"),
                "val_gpu_mem_avg_mb": float("nan"),
                "test_loss": last_eval_metrics["loss"],
                "test_auc": last_eval_metrics["auc"],
                "test_precision": last_eval_metrics["precision"],
                "test_recall": last_eval_metrics["recall"],
                "test_f1": last_eval_metrics["f1"],
                "test_sens": last_eval_metrics["sensitivity"],
                "test_spec": last_eval_metrics["specificity"],
                "test_spec_at_sens95": last_eval_metrics["specificity_at_sens95"],
                "test_throughput_img_s": last_val_throughput,
                "test_elapsed_s": last_val_elapsed,
                "test_avg_batch_time_ms": last_val_avg_batch_time_ms,
                "test_inference_latency_ms_img": last_val_inference_latency_ms_img,
                "test_inference_latency_ms_batch": last_val_inference_latency_ms_batch,
                "test_gpu_mem_avg_mb": last_val_mem_avg,
                "lr": float("nan"),
                "total_train_time_s": time.perf_counter() - start_time,
            }
            with metrics_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COMMON_CSV_FIELDS)
                writer.writerow(final_row)
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


def _gpu_mem_current_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    return float(max(alloc, reserved))


def _mean_valid(values: list[float]) -> float:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.mean(valid)) if valid else float("nan")


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

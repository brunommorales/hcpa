"""
Training script for Pure Vision Transformer.

Tokens are generated directly from raw image patches. No CNN backbone is used.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vit_pure.config import get_args
from vit_pure.model import create_pure_vit_model
from vit_pure.utils import model_summary, print_gpu_memory_info, validate_vit_args

from hybrid_shared.data_loader_bridge_opt import get_data_loaders
from hybrid_shared.metrics import compute_metrics, find_optimal_threshold
from hybrid_shared.profiling import PerformanceProfiler, format_profiling_report
from hybrid_shared.results import (
    build_terminal_epoch_summary,
    build_training_summary,
    build_terminal_final_summary,
    compute_avg_gpu_memory_mb,
    make_metrics_row,
    save_metrics_history_csv,
    save_roc_curve_artifacts,
)
from hybrid_shared.training_utils_opt import (
    EMAManager,
    LinearWarmupCosineAnnealingScheduler,
    compute_loss,
    create_optimizer,
    ddp_concat_variable_length,
    ensure_cuda_ready,
)
from hybrid_shared.runtime_profiling import RuntimeProfiler
from hybrid_shared.stability import (
    autocast_context,
    clip_gradients_and_check_finite,
    format_metric,
    is_primary_process,
    resolve_amp_dtype,
    sanitize_prediction_tensors,
    tensor_is_finite,
)


VIT_NORMALIZATION_MODEL = "vit_base_patch16_224"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def unwrap_batch(batch, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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

    if device is not None:
        images = images.to(device)
        labels = labels.to(device)
    labels = labels.float().view(-1, 1)
    return images, labels


def extract_labels(batch) -> torch.Tensor:
    """Extract labels without moving full image tensors to the target device."""
    if isinstance(batch, dict):
        labels = batch["label"]
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 1 and isinstance(batch[0], dict):
            labels = batch[0]["label"]
        elif len(batch) == 2 and not isinstance(batch[0], dict):
            _, labels = batch
        elif len(batch) > 0 and isinstance(batch[0], dict):
            labels = batch[0]["label"]
        else:
            raise TypeError(f"Unsupported batch structure: {type(batch)}")
    else:
        raise TypeError(f"Unsupported batch structure: {type(batch)}")

    return labels.float().view(-1)


def estimate_pos_weight(loader, device: torch.device) -> float:
    """
    Estimate neg/pos over the training split.

    This is used to keep the ViT from collapsing to the negative class when
    trained from scratch on the imbalanced DR dataset.
    """
    pos_count = 0.0
    total_count = 0.0

    for batch in loader:
        labels = extract_labels(batch)
        pos_count += float(labels.sum().item())
        total_count += float(labels.numel())

    if hasattr(loader, "reset") and callable(loader.reset):
        try:
            loader.reset()
        except Exception:
            pass

    if dist.is_initialized():
        counts = torch.tensor([pos_count, total_count], device=device, dtype=torch.float64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        pos_count = float(counts[0].item())
        total_count = float(counts[1].item())

    neg_count = max(total_count - pos_count, 0.0)
    if pos_count <= 0.0:
        return 1.0
    return max(1.0, neg_count / pos_count)


def load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    target_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    target_model.load_state_dict(state_dict)


def forward_loss_with_optional_amp(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    enable_amp: bool,
    amp_dtype: torch.dtype | None,
    pos_weight: float,
    label_smoothing: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(enable_amp, amp_dtype):
        logits = model(images)
        loss = compute_loss(
            logits.squeeze(-1),
            labels.squeeze(-1),
            label_smoothing=label_smoothing,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
        )
    return logits, loss


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    args,
    scaler,
    pos_weight: float,
    ema_manager: EMAManager | None = None,
    profiler: PerformanceProfiler | None = None,
    runtime_profiler: RuntimeProfiler | None = None,
    stage_name: str = "train",
    return_predictions: bool = False,
) -> dict:
    if train:
        model.train()
        if ema_manager:
            ema_manager.model.train()
    else:
        model.eval()

    if profiler:
        profiler.start_epoch()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    all_targets = []
    all_probs = []
    num_batches = 0
    memory_samples_mb = []
    inference_time_s = 0.0
    inference_batches = 0
    inference_samples = 0

    amp_dtype = getattr(args, "resolved_amp_dtype", None)
    use_amp = bool(args.enable_amp and amp_dtype is not None)
    use_grad_scaler = bool(use_amp and scaler.is_enabled())
    skipped_batches = 0
    amp_fallback_batches = 0
    total_samples = 0
    epoch_start_time = time.perf_counter()
    grad_accum_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1))) if train else 1
    accumulated_microbatches = 0
    optimizer_steps = 0

    if train:
        optimizer.zero_grad(set_to_none=True)

    try:
        total_loader_batches = len(loader)
    except TypeError:
        total_loader_batches = None

    def finish_optimizer_step(batch_number: int) -> None:
        nonlocal accumulated_microbatches, optimizer_steps, skipped_batches
        if accumulated_microbatches <= 0:
            return

        with runtime_profiler.range(f"{stage_name}/optimizer_step") if runtime_profiler else nullcontext():
            if use_grad_scaler:
                scaler.unscale_(optimizer)
            grads_finite, _ = clip_gradients_and_check_finite(
                model,
                max_norm=args.clip_grad_norm,
            )
            if not grads_finite:
                skipped_batches += accumulated_microbatches
                optimizer.zero_grad(set_to_none=True)
                if use_grad_scaler:
                    scaler.update()
                if is_primary_process():
                    print(
                        f"[WARN] {stage_name}: gradientes não-finitos acumulados até batch {batch_number}; "
                        f"descartando {accumulated_microbatches} microbatches."
                    )
                accumulated_microbatches = 0
                return

            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulated_microbatches = 0

        if ema_manager:
            with runtime_profiler.range(f"{stage_name}/ema_update") if runtime_profiler else nullcontext():
                ema_manager.update()

    loader_iter = iter(loader)
    batch_idx = 0
    while True:
        try:
            batch = runtime_profiler.fetch_next(loader_iter, stage_name) if runtime_profiler else next(loader_iter)
        except StopIteration:
            break
        is_last_batch = (
            total_loader_batches is not None
            and batch_idx + 1 >= total_loader_batches
        )

        if profiler:
            profiler.start_batch()

        with runtime_profiler.range(f"{stage_name}/batch") if runtime_profiler else nullcontext():
            with runtime_profiler.range(f"{stage_name}/batch_unpack") if runtime_profiler else nullcontext():
                images, labels = unwrap_batch(batch, None)

            with runtime_profiler.range(f"{stage_name}/h2d") if runtime_profiler else nullcontext():
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                metric_labels = labels.detach().clone()

            if train:
                use_amp_for_batch = use_amp
                with runtime_profiler.range(f"{stage_name}/forward") if runtime_profiler else nullcontext():
                    logits, loss = forward_loss_with_optional_amp(
                        model,
                        images,
                        labels,
                        enable_amp=use_amp_for_batch,
                        amp_dtype=amp_dtype,
                        pos_weight=pos_weight,
                        label_smoothing=args.label_smoothing,
                        focal_gamma=args.focal_gamma,
                    )

                if not (tensor_is_finite(logits) and tensor_is_finite(loss)) and use_amp_for_batch:
                    amp_fallback_batches += 1
                    if is_primary_process():
                        print(
                            f"[WARN] {stage_name}: batch {batch_idx + 1} gerou não-finito em AMP; "
                            "reexecutando em FP32."
                        )
                    with runtime_profiler.range(f"{stage_name}/forward_fp32_fallback") if runtime_profiler else nullcontext():
                        logits, loss = forward_loss_with_optional_amp(
                            model,
                            images,
                            labels,
                            enable_amp=False,
                            amp_dtype=None,
                            pos_weight=pos_weight,
                            label_smoothing=args.label_smoothing,
                            focal_gamma=args.focal_gamma,
                        )
                    use_amp_for_batch = False

                if not (tensor_is_finite(logits) and tensor_is_finite(loss)):
                    skipped_batches += 1
                    if is_primary_process():
                        print(
                            f"[WARN] {stage_name}: batch {batch_idx + 1} permanece não-finito após fallback; "
                            "ignorando batch."
                        )
                    if profiler:
                        profiler.end_batch(images.size(0))
                    if runtime_profiler:
                        runtime_profiler.step(stage_name)
                    batch_idx += 1
                    continue

                loss_for_backward = loss / float(grad_accum_steps)
                with runtime_profiler.range(f"{stage_name}/backward") if runtime_profiler else nullcontext():
                    if use_grad_scaler:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()
                accumulated_microbatches += 1

                if accumulated_microbatches >= grad_accum_steps or is_last_batch:
                    finish_optimizer_step(batch_idx + 1)
            else:
                with torch.no_grad():
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with runtime_profiler.range(f"{stage_name}/forward") if runtime_profiler else nullcontext():
                        logits, loss = forward_loss_with_optional_amp(
                            model,
                            images,
                            labels,
                            enable_amp=use_amp,
                            amp_dtype=amp_dtype,
                            pos_weight=pos_weight,
                            label_smoothing=0.0,
                            focal_gamma=0.0,
                        )
                    if not (tensor_is_finite(logits) and tensor_is_finite(loss)) and use_amp:
                        amp_fallback_batches += 1
                        if is_primary_process():
                            print(
                                f"[WARN] {stage_name}: batch {batch_idx + 1} gerou não-finito em AMP; "
                                "reexecutando em FP32."
                            )
                        with runtime_profiler.range(f"{stage_name}/forward_fp32_fallback") if runtime_profiler else nullcontext():
                            logits, loss = forward_loss_with_optional_amp(
                                model,
                                images,
                                labels,
                                enable_amp=False,
                                amp_dtype=None,
                                pos_weight=pos_weight,
                                label_smoothing=0.0,
                                focal_gamma=0.0,
                            )
                    if not (tensor_is_finite(logits) and tensor_is_finite(loss)):
                        skipped_batches += 1
                        if is_primary_process():
                            print(
                                f"[WARN] {stage_name}: batch {batch_idx + 1} permanece não-finito após fallback; "
                                "ignorando batch."
                            )
                        if profiler:
                            profiler.end_batch(images.size(0))
                        if runtime_profiler:
                            runtime_profiler.step(stage_name)
                        batch_idx += 1
                        continue
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_time_s += time.perf_counter() - inference_start
                    inference_batches += 1
                    inference_samples += int(images.size(0))

            with runtime_profiler.range(f"{stage_name}/metrics") if runtime_profiler else nullcontext():
                detached_logits = logits.detach().float()
                total_loss = total_loss + loss.detach().float()
                all_probs.append(torch.sigmoid(detached_logits).view_as(metric_labels))
                all_targets.append(metric_labels.float())
                num_batches += 1
                total_samples += int(images.size(0))

            if profiler:
                profiler.end_batch(images.size(0))
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
                allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                memory_samples_mb.append(max(allocated_mb, reserved_mb))

        if runtime_profiler:
            runtime_profiler.step(stage_name)

        if (batch_idx + 1) % 10 == 0 and args.verbose:
            total_batches_label = total_loader_batches if total_loader_batches is not None else "?"
            print(f"  Batch {batch_idx + 1}/{total_batches_label}")
        batch_idx += 1

    if train:
        finish_optimizer_step(batch_idx)

    avg_loss = float((total_loss / max(1, num_batches)).item()) if num_batches > 0 else float("nan")

    if profiler:
        epoch_profile = profiler.end_epoch()
    else:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        else:
            memory_samples_mb = [0.0]
        epoch_time = time.perf_counter() - epoch_start_time
        epoch_profile = {
            "epoch_time": epoch_time,
            "avg_batch_time_ms": (epoch_time / max(num_batches, 1)) * 1000.0 if num_batches > 0 else 0.0,
            "throughput": total_samples / (epoch_time + 1e-7),
            "memory_avg_mb": (
                sum(memory_samples_mb) / len(memory_samples_mb)
                if memory_samples_mb
                else 0.0
            ),
            "num_samples": total_samples,
        }

    epoch_profile["inference_latency_ms_img"] = (
        inference_time_s / inference_samples * 1000.0 if inference_samples > 0 else float("nan")
    )
    epoch_profile["inference_latency_ms_batch"] = (
        inference_time_s / inference_batches * 1000.0 if inference_batches > 0 else float("nan")
    )

    global_samples = total_samples
    if dist.is_initialized():
        samples_tensor = torch.tensor([total_samples], device=device, dtype=torch.float64)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        global_samples = int(samples_tensor.item())
        elapsed_tensor = torch.tensor([epoch_profile["epoch_time"]], device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        epoch_profile["epoch_time"] = float(elapsed_tensor.item())
        mem_tensor = torch.tensor(
            [epoch_profile["memory_avg_mb"]],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(mem_tensor, op=dist.ReduceOp.SUM)
        epoch_profile["memory_avg_mb"] = float(mem_tensor.item()) / max(dist.get_world_size(), 1)
        latency_tensor = torch.tensor(
            [
                0.0 if np.isnan(epoch_profile["inference_latency_ms_img"]) else epoch_profile["inference_latency_ms_img"],
                0.0 if np.isnan(epoch_profile["inference_latency_ms_batch"]) else epoch_profile["inference_latency_ms_batch"],
                1.0 if inference_samples > 0 else 0.0,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(latency_tensor, op=dist.ReduceOp.SUM)
        latency_ranks = max(float(latency_tensor[2].item()), 1.0)
        if latency_tensor[2].item() > 0:
            epoch_profile["inference_latency_ms_img"] = float(latency_tensor[0].item()) / latency_ranks
            epoch_profile["inference_latency_ms_batch"] = float(latency_tensor[1].item()) / latency_ranks
        epoch_profile["avg_batch_time_ms"] = (
            epoch_profile["epoch_time"] / max(num_batches, 1)
        ) * 1000.0 if num_batches > 0 else 0.0
    epoch_profile["num_samples"] = global_samples
    epoch_profile["throughput"] = global_samples / (epoch_profile["epoch_time"] + 1e-7)
    if all_targets and all_probs:
        all_targets = torch.cat(all_targets, dim=0).squeeze(-1)
        all_probs = torch.cat(all_probs, dim=0).squeeze(-1)

        if dist.is_initialized():
            all_targets = ddp_concat_variable_length(all_targets, device=device).view(-1)
            all_probs = ddp_concat_variable_length(all_probs, device=device).view(-1)
        else:
            all_targets = all_targets.view(-1)
            all_probs = all_probs.view(-1)

        all_targets, all_probs, invalid_prediction_count = sanitize_prediction_tensors(all_targets, all_probs)
        if invalid_prediction_count > 0 and is_primary_process():
            print(
                f"[WARN] {stage_name}: descartando {invalid_prediction_count} predições não finitas antes das métricas."
            )
        metrics = compute_metrics(all_targets, all_probs)
    else:
        all_targets = torch.empty(0, device=device)
        all_probs = torch.empty(0, device=device)
        invalid_prediction_count = 0
        nan = float("nan")
        metrics = {
            "auc": nan,
            "precision": nan,
            "sensitivity": nan,
            "recall": nan,
            "specificity": nan,
            "f1": nan,
        }

    results = {
        "loss": avg_loss,
        "auc": metrics["auc"],
        "precision": metrics["precision"],
        "sensitivity": metrics["sensitivity"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
        "skipped_batches": skipped_batches,
        "amp_fallback_batches": amp_fallback_batches,
        "invalid_prediction_count": invalid_prediction_count,
        "gradient_accumulation_steps": grad_accum_steps,
        "optimizer_steps": optimizer_steps,
        **epoch_profile,
    }
    if return_predictions:
        results["targets"] = all_targets
        results["probs"] = all_probs
    return results


def get_state_dict(model: nn.Module) -> dict:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def main() -> None:
    args = get_args()
    validate_vit_args(args.img_size, args.patch_size)
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    ensure_cuda_ready()
    amp_dtype = resolve_amp_dtype(enable_amp=args.enable_amp)
    args.resolved_amp_dtype = amp_dtype

    set_seed(args.seed)
    rank, world_size, local_rank, device = init_distributed()

    if rank == 0:
        print(f"Device: {device}")
        print(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")
        amp_dtype_name = (
            "off" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
        )
        print(f"AMP: {args.enable_amp} ({amp_dtype_name})")
        print(f"Flash/SDPA attention: {args.enable_flash_attention}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print_gpu_memory_info()

        # Aviso sobre batch_size para ViT puro
        if args.batch_size > 48:
            print(f"\n[WARNING] Batch size {args.batch_size} é alto para Pure ViT com {args.img_size}x{args.img_size} imagens.")
            print(f"         Recomendado: batch_size <= 32-48")
            print(f"         Se receber CUDA OOM, reduza batch_size ou use --gradient_accumulation_steps")
            print(f"         Exemplo: --batch_size 32 --gradient_accumulation_steps 3\n")

    os.makedirs(args.results_dir, exist_ok=True)
    runtime_profiler = RuntimeProfiler(args.results_dir, rank=rank, project_name="vit_pure")
    metrics_csv_path = os.path.join(args.results_dir, f"metrics_exec{args.exec_id}.csv")
    metrics_rows = []

    if rank == 0:
        print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        tfrec_dir=args.tfrec_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        dataset_name=args.dataset,
        use_dali=args.enable_dali,
        augment=args.augment,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
        enable_amp=args.enable_amp,
        model_name=VIT_NORMALIZATION_MODEL,
    )

    effective_pos_weight = float(args.pos_weight)
    if args.auto_pos_weight and effective_pos_weight <= 1.0:
        effective_pos_weight = estimate_pos_weight(train_loader, device)
    effective_pos_weight = max(1.0, effective_pos_weight)

    if rank == 0:
        print(f"Effective positive-class weight: {effective_pos_weight:.4f}")

    model = create_pure_vit_model(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
        enable_flash_attention=args.enable_flash_attention,
        device=device,
    )

    if rank == 0:
        print(
            model_summary(
                model,
                "ViT-B/16",
                img_size=args.img_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_layers=args.num_transformer_layers,
                num_heads=args.num_heads,
                use_cls_token=args.use_cls_token,
            )
        )

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr=args.lrate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.enable_cosine:
        scheduler = LinearWarmupCosineAnnealingScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            min_lr=1e-6,
        )

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=bool(args.enable_amp and amp_dtype == torch.float16),
    )
    ema_manager = EMAManager(model, decay=args.ema_decay) if args.enable_ema else None

    profiler_train = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None
    profiler_val = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None

    best_auc = 0.0
    best_epoch = 0
    overall_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'=' * 70}")
            print(
                f"Patch source: raw image patches | patch={args.patch_size} | "
                f"grid={args.img_size // args.patch_size}x{args.img_size // args.patch_size}"
            )
            print("Training...")

        if scheduler is not None:
            scheduler.step()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            args=args,
            scaler=scaler,
            pos_weight=effective_pos_weight,
            ema_manager=ema_manager,
            profiler=profiler_train,
            runtime_profiler=runtime_profiler,
            stage_name="train",
        )

        if rank == 0:
            print("Validating...")
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            args=args,
            scaler=scaler,
            pos_weight=effective_pos_weight,
            profiler=profiler_val,
            runtime_profiler=runtime_profiler,
            stage_name="val",
            return_predictions=True,
        )

        if rank == 0:
            val_opt_threshold, val_opt_f1 = find_optimal_threshold(
                val_metrics["targets"],
                val_metrics["probs"],
                metric="f1",
            )
            val_metrics["optimal_threshold"] = val_opt_threshold
            val_metrics["optimal_f1"] = val_opt_f1

            print(
                f"\nTrain Loss: {format_metric(train_metrics['loss'])} | AUC: {format_metric(train_metrics['auc'])} | "
                f"Precision: {format_metric(train_metrics['precision'])} | Sensitivity: {format_metric(train_metrics['sensitivity'])} | "
                f"Spec: {format_metric(train_metrics['specificity'])} | "
                f"F1: {format_metric(train_metrics['f1'])}"
            )
            print(
                f"Val   Loss: {format_metric(val_metrics['loss'])} | AUC: {format_metric(val_metrics['auc'])} | "
                f"Precision: {format_metric(val_metrics['precision'])} | Sensitivity: {format_metric(val_metrics['sensitivity'])} | "
                f"Spec: {format_metric(val_metrics['specificity'])} | "
                f"F1: {format_metric(val_metrics['f1'])}"
            )
            if "throughput" in train_metrics:
                print(f"Throughput: {train_metrics['throughput']:.1f} img/s")
            if train_metrics.get("skipped_batches", 0) > 0 or train_metrics.get("amp_fallback_batches", 0) > 0:
                print(
                    f"Train stability: skipped_batches={train_metrics.get('skipped_batches', 0)} | "
                    f"amp_fallbacks={train_metrics.get('amp_fallback_batches', 0)}"
                )
            if val_metrics.get("skipped_batches", 0) > 0 or val_metrics.get("amp_fallback_batches", 0) > 0:
                print(
                    f"Val stability: skipped_batches={val_metrics.get('skipped_batches', 0)} | "
                    f"amp_fallbacks={val_metrics.get('amp_fallback_batches', 0)}"
                )
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            metrics_rows.append(
                make_metrics_row(
                    epoch=epoch,
                    stage="train",
                    lr=optimizer.param_groups[0]["lr"],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            )
            save_metrics_history_csv(metrics_csv_path, metrics_rows)

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_epoch = epoch
                checkpoint_path = os.path.join(args.results_dir, f"best_model_exec{args.exec_id}.pt")
                torch.save(get_state_dict(model), checkpoint_path)
                print(f"Saved best model: {checkpoint_path}")

    best_checkpoint_path = os.path.join(args.results_dir, f"best_model_exec{args.exec_id}.pt")
    loaded_best_checkpoint = False
    if os.path.exists(best_checkpoint_path):
        if rank == 0:
            print(f"\nLoading best checkpoint for final evaluation: {best_checkpoint_path}")
        state_dict = torch.load(best_checkpoint_path, map_location=device)
        load_model_state_dict(model, state_dict)
        loaded_best_checkpoint = True
    elif rank == 0:
        print(f"\n[WARN] Best checkpoint not found; evaluating final model state: {best_checkpoint_path}")

    test_ema_manager = ema_manager if (ema_manager and not loaded_best_checkpoint) else None
    if test_ema_manager:
        test_ema_manager.apply_shadow()

    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer,
        device,
        train=False,
        args=args,
        scaler=scaler,
        pos_weight=effective_pos_weight,
        profiler=profiler_val,
        runtime_profiler=runtime_profiler,
        stage_name="test",
        return_predictions=True,
    )

    if test_ema_manager:
        test_ema_manager.restore()

    if rank == 0:
        opt_threshold, opt_f1 = find_optimal_threshold(test_metrics["targets"], test_metrics["probs"], metric="f1")
        test_metrics["optimal_threshold"] = opt_threshold
        test_metrics["optimal_f1"] = opt_f1

        roc_artifacts = save_roc_curve_artifacts(
            test_metrics["targets"],
            test_metrics["probs"],
            args.results_dir,
            args.dataset,
            args.exec_id,
        )
        total_train_time_s = round(time.perf_counter() - overall_start_time, 1)
        metrics_rows.append(
            make_metrics_row(
                epoch=args.epochs,
                stage="final_test",
                lr=optimizer.param_groups[0]["lr"],
                test_metrics=test_metrics,
                total_train_time_s=total_train_time_s,
            )
        )
        save_metrics_history_csv(metrics_csv_path, metrics_rows)

        avg_gpu_memory_mb = compute_avg_gpu_memory_mb(
            metrics_rows=metrics_rows,
            eval_metrics=test_metrics,
            profiler=profiler_train,
        )
        print(
            build_terminal_final_summary(
                eval_metrics=test_metrics,
                total_train_time_s=total_train_time_s,
                avg_gpu_memory_mb=avg_gpu_memory_mb,
            )
        )

        summary_path = os.path.join(args.results_dir, f"summary_exec{args.exec_id}.txt")
        summary_text = build_training_summary(
            model_name="ViT-B/16",
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            eval_split_name="test",
            eval_metrics=test_metrics,
            profiler=profiler_train,
            total_train_time_s=total_train_time_s,
            metadata=[
                ("Token source", "raw image patches"),
                ("Variant", "ViT-B/16"),
                ("Patch size", args.patch_size),
                ("Image size", args.img_size),
                ("Num patches", (args.img_size // args.patch_size) ** 2),
                ("Flash/SDPA attention", args.enable_flash_attention),
                ("Dataset", args.dataset),
                ("Exec ID", args.exec_id),
            ],
            artifacts=[
                ("Metrics CSV", metrics_csv_path),
                ("ROC thresholds CSV", roc_artifacts["thresholds_csv_path"]),
                ("ROC PDF", roc_artifacts["roc_pdf_path"]),
            ],
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(summary_text)

    runtime_profiler.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

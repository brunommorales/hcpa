"""
Training script for RETFound-Green fine-tuning.

Uses the public RETFound-Green backbone weights and trains a binary head for
diabetic retinopathy classification in the existing hcpa pipeline.
"""

from __future__ import annotations

import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELF_DIR = Path(__file__).resolve().parent  # gpu_energy.py vive ao lado do script (robusto ao mount do container)
for _p in (str(SELF_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from retfound_green.config import get_args
from retfound_green.model import (
    RETFOUND_GREEN_PATCH_SIZE,
    create_retfound_green_model,
)
from retfound_green.utils import (
    model_summary,
    print_gpu_memory_info,
    validate_retfound_green_args,
)

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
    clip_gradients,
    compute_loss,
    create_optimizer,
    ddp_concat_variable_length,
    ensure_cuda_ready,
)
from hybrid_shared.runtime_profiling import RuntimeProfiler


# RETFound-Green uses mean/std = 0.5 on [0, 1], numerically identical to the
# Inception-style normalization already supported by the shared loader.
RETFOUND_GREEN_NORMALIZATION_MODEL = "inception_v3"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # ensure_cuda_ready() sets benchmark=True; override for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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


from gpu_energy import GpuTelemetry

_GPU_TELE = None


def _gpu_tele() -> GpuTelemetry:
    """Singleton lazy: criado após o device CUDA estar definido (resolve a GPU correta)."""
    global _GPU_TELE
    if _GPU_TELE is None:
        _GPU_TELE = GpuTelemetry()
    return _GPU_TELE


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    args,
    scaler: GradScaler,
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

    total_loss = torch.zeros((), device=device)
    all_targets = []
    all_probs = []
    num_batches = 0
    memory_samples_mb = []
    power_samples_w = []
    util_samples_pct = []
    mem_util_samples_pct = []
    _tele = _gpu_tele()
    inference_time_s = 0.0
    inference_batches = 0
    inference_samples = 0

    amp_enabled = args.enable_amp and device.type == "cuda"
    total_samples = 0
    epoch_start_time = time.perf_counter()
    _energy_start_j = _tele.energy_j()

    loader_iter = iter(loader)
    batch_idx = 0
    while True:
        try:
            batch = runtime_profiler.fetch_next(loader_iter, stage_name) if runtime_profiler else next(loader_iter)
        except StopIteration:
            break

        if profiler:
            profiler.start_batch()

        with runtime_profiler.range(f"{stage_name}/batch") if runtime_profiler else nullcontext():
            with runtime_profiler.range(f"{stage_name}/batch_unpack") if runtime_profiler else nullcontext():
                images, labels = unwrap_batch(batch, None)

            with runtime_profiler.range(f"{stage_name}/h2d") if runtime_profiler else nullcontext():
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            if train:
                with runtime_profiler.range(f"{stage_name}/optimizer_zero_grad") if runtime_profiler else nullcontext():
                    optimizer.zero_grad(set_to_none=True)

                amp_context = autocast(dtype=torch.float16) if amp_enabled else nullcontext()
                with runtime_profiler.range(f"{stage_name}/forward") if runtime_profiler else nullcontext():
                    with amp_context:
                        logits = model(images)
                        loss = compute_loss(
                            logits.squeeze(-1),
                            labels.squeeze(-1),
                            label_smoothing=args.label_smoothing,
                            pos_weight=pos_weight,
                            focal_gamma=args.focal_gamma,
                        )

                if amp_enabled:
                    with runtime_profiler.range(f"{stage_name}/backward") if runtime_profiler else nullcontext():
                        scaler.scale(loss).backward()
                    with runtime_profiler.range(f"{stage_name}/optimizer_step") if runtime_profiler else nullcontext():
                        if args.clip_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            clip_gradients(model, args.clip_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    with runtime_profiler.range(f"{stage_name}/backward") if runtime_profiler else nullcontext():
                        loss.backward()
                    with runtime_profiler.range(f"{stage_name}/optimizer_step") if runtime_profiler else nullcontext():
                        if args.clip_grad_norm > 0:
                            clip_gradients(model, args.clip_grad_norm)
                        optimizer.step()

                if ema_manager:
                    with runtime_profiler.range(f"{stage_name}/ema_update") if runtime_profiler else nullcontext():
                        ema_manager.update()
            else:
                with torch.no_grad():
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with runtime_profiler.range(f"{stage_name}/forward") if runtime_profiler else nullcontext():
                        if amp_enabled:
                            with autocast(dtype=torch.float16):
                                logits = model(images)
                        else:
                            logits = model(images)

                        loss = compute_loss(
                            logits.squeeze(-1),
                            labels.squeeze(-1),
                            label_smoothing=0.0,
                            pos_weight=pos_weight,
                            focal_gamma=0.0,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_time_s += time.perf_counter() - inference_start
                    inference_batches += 1
                    inference_samples += int(images.size(0))

            with runtime_profiler.range(f"{stage_name}/metrics") if runtime_profiler else nullcontext():
                detached_logits = logits.detach()
                total_loss = total_loss + loss.detach()
                all_probs.append(torch.sigmoid(detached_logits))
                all_targets.append(labels.detach())
                num_batches += 1
                total_samples += int(images.size(0))

            if profiler:
                profiler.end_batch(images.size(0))
            elif device.type == "cuda":
                torch.cuda.synchronize(device)
                allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                _used_mb = _tele.memory_used_mb()
                memory_samples_mb.append(
                    _used_mb if _used_mb is not None else max(allocated_mb, reserved_mb)
                )
                _pw = _tele.power_w()
                if _pw is not None:
                    power_samples_w.append(_pw)
                _u = _tele.util_pct()
                if _u is not None:
                    util_samples_pct.append(_u)
                _mu = _tele.mem_util_pct()
                if _mu is not None:
                    mem_util_samples_pct.append(_mu)

        if runtime_profiler:
            runtime_profiler.step(stage_name)

        if (batch_idx + 1) % 10 == 0 and args.verbose:
            print(f"  Batch {batch_idx + 1}/{len(loader)}")
        batch_idx += 1

    all_targets = torch.cat(all_targets, dim=0).squeeze(-1)
    all_probs = torch.cat(all_probs, dim=0).squeeze(-1)

    if dist.is_initialized():
        all_targets = ddp_concat_variable_length(all_targets, device=device).view(-1)
        all_probs = ddp_concat_variable_length(all_probs, device=device).view(-1)
    else:
        all_targets = all_targets.view(-1)
        all_probs = all_probs.view(-1)

    metrics = compute_metrics(all_targets, all_probs)
    avg_loss = float((total_loss / max(1, num_batches)).item())

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
    # Energia (J) pelo contador NVML e memória REAL (pico) — rank-local (1 GPU/processo).
    _energy_end_j = _tele.energy_j()
    epoch_profile["energy_j"] = (
        max(0.0, _energy_end_j - _energy_start_j)
        if (_energy_start_j is not None and _energy_end_j is not None)
        else float("nan")
    )
    epoch_profile["memory_peak_mb"] = max(memory_samples_mb) if memory_samples_mb else float("nan")
    epoch_profile["avg_power_w"] = (
        sum(power_samples_w) / len(power_samples_w) if power_samples_w else float("nan")
    )
    epoch_profile["gpu_util_pct"] = (
        sum(util_samples_pct) / len(util_samples_pct) if util_samples_pct else float("nan")
    )
    epoch_profile["mem_util_pct"] = (
        sum(mem_util_samples_pct) / len(mem_util_samples_pct) if mem_util_samples_pct else float("nan")
    )
    # tempo GPU-ATIVA (kernel-only) = tempo da época x util/100 (remove ociosidade/CPU)
    epoch_profile["busy_time_s"] = (
        epoch_profile["epoch_time"] * epoch_profile["gpu_util_pct"] / 100.0
        if util_samples_pct else float("nan")
    )
    results = {
        "loss": avg_loss,
        "auc": metrics["auc"],
        "precision": metrics["precision"],
        "sensitivity": metrics["sensitivity"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
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


def load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    target_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    target_model.load_state_dict(state_dict)


def main() -> None:
    args = get_args()
    validate_retfound_green_args(args.img_size)

    checkpoint_path = Path(args.backbone_checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Pesos RETFound-Green nao encontrados.\n"
            f"Caminho esperado: {checkpoint_path}\n"
            "Baixe com:\n"
            "wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/"
            "retfoundgreen_statedict.pth -O "
            f"{checkpoint_path}"
        )

    ensure_cuda_ready()
    set_seed(args.seed)
    rank, world_size, local_rank, device = init_distributed()

    if rank == 0:
        print(f"Device: {device}")
        print(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")
        print_gpu_memory_info()

    os.makedirs(args.results_dir, exist_ok=True)
    runtime_profiler = RuntimeProfiler(args.results_dir, rank=rank, project_name="retfound_green")
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
        model_name=RETFOUND_GREEN_NORMALIZATION_MODEL,
    )

    effective_pos_weight = float(args.pos_weight)
    if args.auto_pos_weight and effective_pos_weight <= 1.0:
        effective_pos_weight = estimate_pos_weight(train_loader, device)
    effective_pos_weight = max(1.0, effective_pos_weight)

    if rank == 0:
        print(f"Effective positive-class weight: {effective_pos_weight:.4f}")

    # LP-FT: se freeze_epochs > 0, iniciar com backbone congelado independente de freeze_backbone
    initial_freeze = args.freeze_backbone or (args.freeze_epochs > 0)
    model = create_retfound_green_model(
        img_size=args.img_size,
        backbone_model=args.backbone_model,
        checkpoint_path=str(checkpoint_path),
        head_dropout=args.head_dropout,
        freeze_backbone=initial_freeze,
        device=device,
    )
    if args.freeze_epochs > 0:
        args.freeze_backbone = True

    if rank == 0:
        print(
            model_summary(
                model,
                "RETFound-Green",
                img_size=args.img_size,
                feature_dim=model.feature_dim,
                backbone_model=args.backbone_model,
                freeze_backbone=args.freeze_backbone,
                checkpoint_path=str(checkpoint_path),
            )
        )

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # LR discriminativo: head (aleatória) usa LR cheio; backbone (pré-treinado) usa LR/10
    _base = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    _head_params = list(_base.head_dropout.parameters()) + list(_base.classifier.parameters())
    _backbone_params = list(_base.backbone.parameters())
    _param_groups = [
        {"params": _head_params, "lr": args.lrate},
        {"params": _backbone_params, "lr": args.lrate * 0.1},
    ]
    optimizer = optim.AdamW(_param_groups, weight_decay=args.weight_decay, eps=1e-7)
    scheduler = None
    if args.enable_cosine:
        scheduler = LinearWarmupCosineAnnealingScheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            min_lr=1e-6,
        )

    amp_enabled = args.enable_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    ema_manager = EMAManager(model, decay=args.ema_decay) if args.enable_ema else None

    profiler_train = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None
    profiler_val = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None

    best_auc = 0.0
    best_epoch = 0
    overall_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        # LP-FT: unfreeze backbone after freeze_epochs warm-up epochs
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            base_model = model.module if hasattr(model, "module") else model
            base_model.set_backbone_trainable(True)
            args.freeze_backbone = False
            if rank == 0:
                print(f"[LP-FT] Epoch {epoch + 1}: backbone descongelado para fine-tuning completo")

        # Warmup fix: step antes do treino para que a época 0 rode com LR de warmup, não LR cheio
        if scheduler is not None:
            scheduler.step()

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'=' * 70}")
            print(
                f"Backbone: {args.backbone_model} | img={args.img_size} | "
                f"patch={RETFOUND_GREEN_PATCH_SIZE} | frozen={args.freeze_backbone}"
            )
            print("Training...")
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
                f"\nTrain Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc']:.4f} | "
                f"Precision: {train_metrics['precision']:.4f} | Sensitivity: {train_metrics['sensitivity']:.4f} | "
                f"Spec: {train_metrics['specificity']:.4f} | F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"Val   Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | Sensitivity: {val_metrics['sensitivity']:.4f} | "
                f"Spec: {val_metrics['specificity']:.4f} | F1: {val_metrics['f1']:.4f}"
            )
            if "throughput" in train_metrics:
                print(f"Throughput: {train_metrics['throughput']:.1f} img/s")
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
                checkpoint_out = os.path.join(args.results_dir, f"best_model_exec{args.exec_id}.pt")
                torch.save(get_state_dict(model), checkpoint_out)
                print(f"Saved best model: {checkpoint_out}")

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
            model_name="RETFound-Green",
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            eval_split_name="test",
            eval_metrics=test_metrics,
            profiler=profiler_train,
            total_train_time_s=total_train_time_s,
            metadata=[
                ("Token source", "retfound_green_backbone_features"),
                ("Variant", args.backbone_model),
                ("Patch size", RETFOUND_GREEN_PATCH_SIZE),
                ("Image size", args.img_size),
                ("Num patches", (args.img_size // RETFOUND_GREEN_PATCH_SIZE) ** 2),
                ("Freeze backbone", args.freeze_backbone),
                ("Checkpoint", str(checkpoint_path)),
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

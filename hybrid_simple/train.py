"""
Treinamento para Hybrid Simple Model.

Script que integra:
- Dataset do pytorch_opt (DALI/PyTorch)
- Modelo HybridSimple
- Métricas e profiling
- Training loop com AMP, EMA, DDP
"""

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
SELF_DIR = Path(__file__).resolve().parent  # gpu_energy.py vive ao lado do script (robusto ao mount do container)
for _p in (str(SELF_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure hybrid_shared can be imported
HYBRID_SHARED_PATH = PROJECT_ROOT / "hybrid_shared"
if str(HYBRID_SHARED_PATH) not in sys.path:
    sys.path.insert(0, str(HYBRID_SHARED_PATH))

from hybrid_simple.model import create_hybrid_simple_model
from hybrid_simple.config import get_args
from hybrid_simple.utils import get_backbone_feature_dim, modify_backbone_for_feature_extraction, model_summary, print_gpu_memory_info

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
from hybrid_shared.runtime_profiling import RuntimeProfiler
from hybrid_shared.training_utils import (
    compute_loss,
    create_optimizer,
    LinearWarmupCosineAnnealingScheduler,
    EMAManager,
    clip_gradients,
    ddp_concat_variable_length,
    ensure_cuda_ready,
)
from hybrid_shared.data_loader_bridge import get_data_loaders, get_backbone, unpack_batch
from hybrid_shared.stability import (
    apply_finetune_lr_transition,
    autocast_context,
    clip_gradients_and_check_finite,
    format_metric,
    is_primary_process,
    resolve_amp_dtype,
    sanitize_prediction_tensors,
    tensor_is_finite,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple:
    """Initialize distributed training."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        # Abortar ALTO se nao houver GPU. Sem isto o treino roda na CPU em
        # silencio e grava um CSV cuja energia/tempo nao tem relacao com GPU
        # nenhuma (aconteceu no chuc: singularity --nv sem /usr/sbin/ldconfig.real).
        if not torch.cuda.is_available() and os.environ.get("HCPA_ALLOW_CPU") != "1":
            raise SystemExit(
                "[Hardware] Execucao requer GPU, mas torch.cuda.is_available() e False.\n"
                "  Cheque: singularity exec --nv | /usr/sbin/ldconfig.real | driver vs container.\n"
                "  Para forcar CPU (debug apenas): HCPA_ALLOW_CPU=1"
            )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return rank, world_size, local_rank, device


def get_state_dict(model: nn.Module) -> dict:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


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
    label_smoothing: float,
    pos_weight: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(enable_amp, amp_dtype):
        logits = model(images)
        loss = compute_loss(
            logits.view(-1),
            labels.view(-1),
            label_smoothing=label_smoothing,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
        )
    return logits, loss


from gpu_energy import GpuTelemetry, EnergyScope, DEFAULT_SAMPLE_INTERVAL_S

_SAMPLE_INTERVAL_S = DEFAULT_SAMPLE_INTERVAL_S

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
    scaler,
    ema_manager: EMAManager = None,
    profiler: PerformanceProfiler = None,
    runtime_profiler: RuntimeProfiler | None = None,
    stage_name: str = "train",
    return_predictions: bool = False,
) -> dict:
    """
    Executa uma época de treinamento ou validação.

    Args:
        model: Modelo
        loader: DataLoader
        optimizer: Otimizador
        device: Device
        train: Se True, modo treino; se False, modo avaliação
        args: Argumentos
        scaler: GradScaler para AMP
        ema_manager: EMA manager (opcional)
        profiler: Performance profiler (opcional)

    Returns:
        Dictionary com métricas
    """
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
    total_samples = 0
    memory_samples_mb = []
    _tele = _gpu_tele()
    inference_time_s = 0.0
    inference_batches = 0
    inference_samples = 0
    amp_dtype = getattr(args, "resolved_amp_dtype", None)
    use_amp = bool(args.enable_amp and amp_dtype is not None)
    use_grad_scaler = bool(use_amp and scaler.is_enabled())
    skipped_batches = 0
    amp_fallback_batches = 0
    epoch_start_time = time.perf_counter()
    # Telemetria em THREAD DE FUNDO (nunca no laço de batches): o overhead não
    # depende do número de batches, então util%/potência ficam comparáveis entre
    # abordagens. A versão anterior amostrava por batch e ainda sincronizava a GPU
    # a cada batch — o que serializava o pipeline e inflava a ociosidade medida.
    _scope = EnergyScope(_tele, _SAMPLE_INTERVAL_S)
    _scope.start()

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
                images, labels = unpack_batch(batch)

            with runtime_profiler.range(f"{stage_name}/h2d") if runtime_profiler else nullcontext():
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().view(-1, 1)
                metric_labels = labels.detach().clone()

            if train:
                with runtime_profiler.range(f"{stage_name}/optimizer_zero_grad") if runtime_profiler else nullcontext():
                    optimizer.zero_grad(set_to_none=True)

                use_amp_for_batch = use_amp
                with runtime_profiler.range(f"{stage_name}/forward") if runtime_profiler else nullcontext():
                    logits, loss = forward_loss_with_optional_amp(
                        model,
                        images,
                        labels,
                        enable_amp=use_amp_for_batch,
                        amp_dtype=amp_dtype,
                        label_smoothing=args.label_smoothing,
                        pos_weight=args.pos_weight,
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
                            label_smoothing=args.label_smoothing,
                            pos_weight=args.pos_weight,
                            focal_gamma=args.focal_gamma,
                        )
                    use_amp_for_batch = False

                if not (tensor_is_finite(logits) and tensor_is_finite(loss)):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
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

                if use_amp_for_batch and use_grad_scaler:
                    with runtime_profiler.range(f"{stage_name}/backward") if runtime_profiler else nullcontext():
                        scaler.scale(loss).backward()
                    with runtime_profiler.range(f"{stage_name}/optimizer_step") if runtime_profiler else nullcontext():
                        scaler.unscale_(optimizer)
                        grads_finite, _ = clip_gradients_and_check_finite(
                            model,
                            max_norm=args.clip_grad_norm,
                        )
                        if not grads_finite:
                            skipped_batches += 1
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            if is_primary_process():
                                print(
                                    f"[WARN] {stage_name}: batch {batch_idx + 1} gerou gradientes não-finitos; "
                                    "pulando update."
                                )
                            if profiler:
                                profiler.end_batch(images.size(0))
                            if runtime_profiler:
                                runtime_profiler.step(stage_name)
                            batch_idx += 1
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    with runtime_profiler.range(f"{stage_name}/backward") if runtime_profiler else nullcontext():
                        loss.backward()
                    with runtime_profiler.range(f"{stage_name}/optimizer_step") if runtime_profiler else nullcontext():
                        grads_finite, _ = clip_gradients_and_check_finite(
                            model,
                            max_norm=args.clip_grad_norm,
                        )
                        if not grads_finite:
                            skipped_batches += 1
                            optimizer.zero_grad(set_to_none=True)
                            if is_primary_process():
                                print(
                                    f"[WARN] {stage_name}: batch {batch_idx + 1} gerou gradientes não-finitos; "
                                    "pulando update."
                                )
                            if profiler:
                                profiler.end_batch(images.size(0))
                            if runtime_profiler:
                                runtime_profiler.step(stage_name)
                            batch_idx += 1
                            continue
                        optimizer.step()

                if ema_manager:
                    with runtime_profiler.range(f"{stage_name}/ema_update") if runtime_profiler else nullcontext():
                        ema_manager.update()

            else:
                with torch.no_grad():
                    # sem synchronize por-batch: serializava o pipeline de validacao
                    # e a latencia de inferencia nao entra no schema de 33 colunas.
                    inference_start = time.perf_counter()
                    with runtime_profiler.range(f"{stage_name}/forward") if runtime_profiler else nullcontext():
                        logits, loss = forward_loss_with_optional_amp(
                            model,
                            images,
                            labels,
                            enable_amp=use_amp,
                            amp_dtype=amp_dtype,
                            label_smoothing=0.0,
                            pos_weight=args.pos_weight,
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
                                label_smoothing=0.0,
                                pos_weight=args.pos_weight,
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
                    inference_time_s += time.perf_counter() - inference_start
                    inference_batches += 1
                    inference_samples += int(images.size(0))

            with runtime_profiler.range(f"{stage_name}/metrics") if runtime_profiler else nullcontext():
                detached_logits = logits.detach().float()
                total_loss = total_loss + loss.detach().float()
                probs = torch.sigmoid(detached_logits).view_as(metric_labels)
                all_probs.append(probs)
                all_targets.append(metric_labels.float())
                num_batches += 1
                total_samples += int(images.size(0))

            if profiler:
                profiler.end_batch(images.size(0))

        if runtime_profiler:
            runtime_profiler.step(stage_name)

        if (batch_idx + 1) % 10 == 0 and args.verbose:
            print(f"  Batch {batch_idx + 1}/{len(loader)}")
        batch_idx += 1

    avg_loss = float((total_loss / max(num_batches, 1)).item()) if num_batches > 0 else float("nan")

    # As métricas da época são calculadas AQUI, antes de fechar o cronômetro e o
    # EnergyScope, para que a janela de medição seja a mesma de todas as outras
    # abordagens (no TF elas são streaming, dentro do train_step: não dá para
    # excluí-las). Ver METRICAS_COLETADAS.txt, PARTE 6.
    if all_targets and all_probs:
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        all_probs = torch.cat(all_probs, dim=0).view(-1)

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
    _scope.stop()
    _nan = float("nan")

    def _or_nan(v):
        return float(v) if v is not None else _nan

    epoch_profile["energy_j"] = _or_nan(_scope.energy_j)
    epoch_profile["memory_peak_mb"] = _or_nan(_scope.peak_mem_mb)
    epoch_profile["avg_power_w"] = _or_nan(_scope.avg_power_w)
    epoch_profile["gpu_util_pct"] = _or_nan(_scope.avg_util_pct)
    epoch_profile["mem_util_pct"] = _or_nan(_scope.avg_mem_util_pct)
    # tempo GPU-ATIVA (kernel-only) = tempo da época x util/100 (remove ociosidade/CPU).
    # Usa epoch_time (reduzido entre ranks no DDP), não o elapsed interno do scope.
    epoch_profile["busy_time_s"] = (
        epoch_profile["epoch_time"] * _scope.avg_util_pct / 100.0
        if _scope.avg_util_pct is not None else _nan
    )


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
        **epoch_profile,
    }
    if return_predictions:
        results["targets"] = all_targets
        results["probs"] = all_probs
    return results


def main():
    """Main training script."""
    args = get_args()
    ensure_cuda_ready()

    # Initialize
    set_seed(args.seed)
    rank, world_size, local_rank, device = init_distributed()
    amp_dtype = resolve_amp_dtype(enable_amp=args.enable_amp)
    args.resolved_amp_dtype = amp_dtype

    # Print info
    if rank == 0:
        print(f"Device: {device}")
        print(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")
        amp_dtype_name = (
            "off" if amp_dtype is None else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
        )
        print(f"AMP: {args.enable_amp} ({amp_dtype_name})")
        print_gpu_memory_info()

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    runtime_profiler = RuntimeProfiler(args.results_dir, rank=rank, project_name="hybrid_simple")
    metrics_csv_path = os.path.join(args.results_dir, f"metrics_exec{args.exec_id}.csv")
    metrics_rows = []

    # Load dataloaders
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
        model_name=args.backbone,
    )

    # Load backbone CNN
    if rank == 0:
        print(f"Loading backbone: {args.backbone}")
    backbone = get_backbone(
        model_name=args.backbone,
        img_size=args.img_size,
        pretrained=True,
        freeze_backbone=False,
    )

    # Modify backbone for feature extraction
    backbone = modify_backbone_for_feature_extraction(backbone, args.backbone)

    # Create model
    model = create_hybrid_simple_model(
        backbone=backbone,
        backbone_feature_dim=args.backbone_dim,
        transformer_dim=args.transformer_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        use_cls_token=args.use_cls_token,
        device=device,
    )

    if rank == 0:
        print(model_summary(model, f"HybridSimple ({args.backbone})"))

    # DDP wrapping
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Optimizer and scheduler
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

    # AMP
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=bool(args.enable_amp and amp_dtype == torch.float16),
    )

    # EMA
    ema_manager = None
    if args.enable_ema:
        ema_manager = EMAManager(model, decay=args.ema_decay)

    # Profiling
    profiler_train = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None
    profiler_val = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None

    # Training loop
    best_auc = 0.0
    best_epoch = 0
    best_state_dict = None  # best-checkpoint EM MEMÓRIA (RAM), sem IO em disco
    overall_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*70}")

        if scheduler is not None:
            scheduler.step()

        # Freeze backbone in first epochs
        freeze_backbone = epoch < args.freeze_backbone_epochs
        stage_label = "freeze" if freeze_backbone else "finetune"
        for param in model.module.backbone.parameters() if world_size > 1 else model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        # Adjust LR for fine-tuning
        if epoch == args.freeze_backbone_epochs and epoch > 0:
            finetune_lr = args.lrate * args.fine_tune_lr_factor
            apply_finetune_lr_transition(
                optimizer,
                scheduler,
                target_lr=finetune_lr,
            )

        if rank == 0:
            print("Training...")
        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            train=True, args=args, scaler=scaler,
            ema_manager=ema_manager, profiler=profiler_train,
            runtime_profiler=runtime_profiler, stage_name="train",
        )

        if rank == 0:
            print("Validating...")
        val_metrics = run_epoch(
            model, val_loader, optimizer, device,
            train=False, args=args, scaler=scaler,
            profiler=profiler_val, runtime_profiler=runtime_profiler, stage_name="val",
        )

        if rank == 0:
            print(f"\nTrain Loss: {format_metric(train_metrics['loss'])} | AUC: {format_metric(train_metrics['auc'])} | "
                  f"Precision: {format_metric(train_metrics['precision'])} | Sensitivity: {format_metric(train_metrics['sensitivity'])} | "
                  f"Spec: {format_metric(train_metrics['specificity'])} | "
                  f"F1: {format_metric(train_metrics['f1'])}")
            print(f"Val   Loss: {format_metric(val_metrics['loss'])} | AUC: {format_metric(val_metrics['auc'])} | "
                  f"Precision: {format_metric(val_metrics['precision'])} | Sensitivity: {format_metric(val_metrics['sensitivity'])} | "
                  f"Spec: {format_metric(val_metrics['specificity'])} | "
                  f"F1: {format_metric(val_metrics['f1'])}")

            if 'throughput' in train_metrics:
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
                    stage=stage_label,
                    lr=optimizer.param_groups[0]["lr"],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            )
            save_metrics_history_csv(metrics_csv_path, metrics_rows)

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_epoch = epoch

                # Best-checkpoint EM MEMÓRIA (RAM, clone na CPU) — sem IO em disco.
                # Antes: torch.save() a CADA melhora da val -> escrita em disco dentro
                # do total_train_time_s (confundidor de tempo). Padrão do pytorch_opt.
                best_state_dict = {k: v.detach().cpu().clone() for k, v in get_state_dict(model).items()}
                if rank == 0:
                    print(f"[BEST] checkpoint em memória atualizado: epoch={best_epoch} val_auc={best_auc:.6f}")

    loaded_best_checkpoint = False
    if best_state_dict is not None:
        if rank == 0:
            print(f"\nRestaurando best checkpoint em memória para avaliação final (epoch={best_epoch}, val_auc={best_auc:.6f})")
        load_model_state_dict(model, best_state_dict)
        loaded_best_checkpoint = True
    elif rank == 0:
        print("\n[WARN] Sem best checkpoint em memória; avaliando estado final do modelo.")

    test_ema_manager = ema_manager if (ema_manager and not loaded_best_checkpoint) else None
    if test_ema_manager:
        test_ema_manager.apply_shadow()

    test_metrics = run_epoch(
        model, test_loader, optimizer, device,
        train=False, args=args, scaler=scaler,
        profiler=profiler_val, runtime_profiler=runtime_profiler, stage_name="test",
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

        # Save summary
        summary_path = os.path.join(args.results_dir, f"summary_exec{args.exec_id}.txt")
        summary_text = build_training_summary(
            model_name="HybridSimple",
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            eval_split_name="test",
            eval_metrics=test_metrics,
            profiler=profiler_train,
            total_train_time_s=total_train_time_s,
            metadata=[
                ("Backbone", args.backbone),
                ("Dataset", args.dataset),
                ("Exec ID", args.exec_id),
            ],
            artifacts=[
                ("Metrics CSV", metrics_csv_path),
                ("ROC thresholds CSV", roc_artifacts["thresholds_csv_path"]),
                ("ROC PDF", roc_artifacts["roc_pdf_path"]),
            ],
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

    # Cleanup
    runtime_profiler.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

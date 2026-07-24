"""
Treinamento para Hybrid Token Reduction Model.

Script que integra:
- Token selector aprendendo importância
- Redução dinâmica de tokens antes do Transformer
- Comparação com baseline (sem seleção)
- Análise de padrões de seleção
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.amp
from contextlib import contextmanager, nullcontext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELF_DIR = Path(__file__).resolve().parent  # gpu_energy.py vive ao lado do script (robusto ao mount do container)
for _p in (str(SELF_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hybrid_token_reduction_opt.model import create_hybrid_token_reduction_model
from hybrid_token_reduction_opt.config import get_args
from hybrid_token_reduction_opt.utils import (
    compute_token_reduction_stats,
    model_summary_with_token_analysis,
    analyze_token_selection,
    format_token_analysis_report,
    apply_mixup_cutmix,
)

from hybrid_shared.metrics import compute_metrics, find_optimal_threshold
from hybrid_shared.profiling import PerformanceProfiler, format_profiling_report
from hybrid_shared.results import (
    build_training_summary,
    make_metrics_row,
    save_metrics_history_csv,
    save_roc_curve_artifacts,
)
from hybrid_shared.runtime_profiling import RuntimeProfiler
from hybrid_shared.training_utils_opt import (
    compute_loss,
    create_optimizer,
    LinearWarmupCosineAnnealingScheduler,
    EMAManager,
    ddp_concat_variable_length,
    ensure_cuda_ready,
)
from hybrid_shared.data_loader_bridge_opt import get_data_loaders, get_backbone, unpack_batch
from hybrid_simple.utils import modify_backbone_for_feature_extraction


def set_seed(seed: int) -> None:
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple:
    """Initialize DDP."""
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


def format_metric(value: float) -> str:
    """Formats numeric metrics while keeping NaN readable in logs."""
    if isinstance(value, float) and value != value:
        return "n/a"
    return f"{value:.4f}"


def is_primary_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def resolve_amp_dtype(args) -> torch.dtype | None:
    """Resolve AMP dtype preferring bf16 on modern CUDA GPUs for stability."""
    if not getattr(args, "enable_amp", False) or not torch.cuda.is_available():
        return None

    requested = str(getattr(args, "amp_dtype", "auto")).lower()
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested == "auto":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unknown AMP dtype: {requested}")


def autocast_context(enabled: bool, amp_dtype: torch.dtype | None):
    if enabled and amp_dtype is not None:
        return torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
    return nullcontext()


def tensor_is_finite(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    return bool(torch.isfinite(tensor).all().item())


def clip_gradients_and_check_finite(
    model: nn.Module,
    *,
    max_norm: float,
    check_finite: bool,
) -> tuple[bool, float]:
    params_with_grad = [param for param in model.parameters() if param.grad is not None]
    if not params_with_grad:
        return True, 0.0

    if max_norm <= 0 and not check_finite:
        return True, 0.0

    clip_value = float("inf") if max_norm <= 0 else float(max_norm)
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(
            params_with_grad,
            max_norm=clip_value,
            error_if_nonfinite=False,
            foreach=True,
        )
    except TypeError:
        total_norm = torch.nn.utils.clip_grad_norm_(
            params_with_grad,
            max_norm=clip_value,
            error_if_nonfinite=False,
        )

    if not check_finite:
        return True, 0.0

    total_norm_value = float(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
    return bool(np.isfinite(total_norm_value)), total_norm_value


def sanitize_prediction_tensors(
    targets: torch.Tensor,
    probs: torch.Tensor,
    *,
    stage_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    targets = targets.view(-1)
    probs = probs.view(-1)
    finite_mask = torch.isfinite(targets) & torch.isfinite(probs)
    invalid_count = int((~finite_mask).sum().item())
    if invalid_count == 0:
        return targets, probs, 0

    if is_primary_process():
        print(
            f"[WARN] {stage_name}: descartando {invalid_count} predições não finitas antes das métricas."
        )
    return targets[finite_mask], probs[finite_mask], invalid_count


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


def apply_finetune_lr_transition(
    optimizer: torch.optim.Optimizer,
    scheduler: LinearWarmupCosineAnnealingScheduler | None,
    *,
    target_lr: float,
) -> float:
    """
    Align optimizer and scheduler state when switching from frozen backbone to finetuning.

    Cosine warmup otherwise keeps pushing the LR back toward the pre-unfreeze value,
    which is exactly the pattern that made this variant collapse to the negative class.
    """
    if target_lr <= 0:
        raise ValueError(f"target_lr must be positive, got {target_lr}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = target_lr

    if scheduler is not None:
        scheduler.base_lr = target_lr
        scheduler.current_epoch = max(scheduler.current_epoch, scheduler.warmup_epochs)

    return target_lr


@contextmanager
def ema_shadow_scope(ema_manager: EMAManager | None):
    """
    Avalia temporariamente o modelo com pesos EMA.

    Mantém validação, checkpoint e teste final no mesmo estado de pesos para
    evitar misturar o melhor checkpoint "raw" com o shadow EMA de outra época.
    """
    if ema_manager is None:
        yield
        return

    ema_manager.apply_shadow()
    try:
        yield
    finally:
        ema_manager.restore()


def unwrap_model_for_state_dict(model: nn.Module) -> nn.Module:
    """Return the real model behind DDP and torch.compile wrappers."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def get_state_dict(model: nn.Module) -> dict:
    return unwrap_model_for_state_dict(model).state_dict()


def load_model_state_dict(model: nn.Module, state_dict: dict) -> None:
    unwrap_model_for_state_dict(model).load_state_dict(state_dict)


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
    scaler: "torch.amp.GradScaler",
    ema_manager: EMAManager = None,
    profiler: PerformanceProfiler = None,
    runtime_profiler: RuntimeProfiler | None = None,
    stage_name: str = "train",
    return_predictions: bool = False,
) -> dict:
    """
    Executa uma época.
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

    total_loss = torch.zeros((), device=device)
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
    if amp_dtype is None:
        amp_dtype = resolve_amp_dtype(args)
    use_amp = bool(args.enable_amp and amp_dtype is not None)
    use_grad_scaler = bool(use_amp and scaler.is_enabled())
    skipped_batches = 0
    amp_fallback_batches = 0
    use_channels_last = bool(getattr(args, "use_channels_last", False))
    check_finite = bool(getattr(args, "enable_finite_checks", False))
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
                if use_channels_last and images.ndim == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True).float().view(-1, 1)
                metric_labels = labels.detach()

            if train and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
                with runtime_profiler.range(f"{stage_name}/augment") if runtime_profiler else nullcontext():
                    images, labels = apply_mixup_cutmix(
                        images,
                        labels,
                        mixup_alpha=args.mixup_alpha,
                        cutmix_alpha=args.cutmix_alpha,
                    )
                    if use_channels_last and images.ndim == 4:
                        images = images.contiguous(memory_format=torch.channels_last)

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

                if check_finite and not (tensor_is_finite(logits) and tensor_is_finite(loss)):
                    if use_amp_for_batch:
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

                if check_finite and not (tensor_is_finite(logits) and tensor_is_finite(loss)):
                    skipped_batches += 1
                    if is_primary_process():
                        print(
                            f"[WARN] {stage_name}: batch {batch_idx + 1} permanece não-finito após fallback; "
                            "ignorando batch."
                        )
                    optimizer.zero_grad(set_to_none=True)
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
                            check_finite=check_finite,
                        )
                        if not grads_finite:
                            skipped_batches += 1
                            if is_primary_process():
                                print(
                                    f"[WARN] {stage_name}: batch {batch_idx + 1} gerou gradientes não-finitos; "
                                    "pulando update."
                                )
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
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
                            check_finite=check_finite,
                        )
                        if not grads_finite:
                            skipped_batches += 1
                            if is_primary_process():
                                print(
                                    f"[WARN] {stage_name}: batch {batch_idx + 1} gerou gradientes não-finitos; "
                                    "pulando update."
                                )
                            optimizer.zero_grad(set_to_none=True)
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
                with torch.inference_mode():
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

                    if check_finite and not (tensor_is_finite(logits) and tensor_is_finite(loss)) and use_amp:
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

                    if check_finite and not (tensor_is_finite(logits) and tensor_is_finite(loss)):
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
                probs = torch.sigmoid(detached_logits).view_as(metric_labels).float()
                total_loss = total_loss + loss.detach().float()
                all_probs.append(probs)
                all_targets.append(metric_labels.float())
                num_batches += 1
                total_samples += int(images.size(0))

            if profiler:
                profiler.end_batch(images.size(0))

        if runtime_profiler:
            runtime_profiler.step(stage_name)

        if (batch_idx + 1) % 10 == 0 and args.verbose:
            print(f"  Batch {batch_idx + 1}/{len(loader)}: Loss={loss.item():.4f}")
        batch_idx += 1

    def finish_epoch_profile() -> dict:
        if profiler:
            epoch_profile = profiler.end_epoch()
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            else:
                memory_samples_mb.append(0.0)

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

        epoch_profile["num_samples"] = global_samples
        epoch_profile["throughput"] = global_samples / (epoch_profile["epoch_time"] + 1e-7)
        epoch_profile["avg_batch_time_ms"] = (
            epoch_profile["epoch_time"] / max(num_batches, 1)
        ) * 1000.0 if num_batches > 0 else 0.0
        _scope.stop()   # idempotente
        _nan = float("nan")
        _u_avg = _scope.avg_util_pct
        epoch_profile["energy_j"] = _scope.energy_j if _scope.energy_j is not None else _nan
        epoch_profile["memory_peak_mb"] = _scope.peak_mem_mb if _scope.peak_mem_mb is not None else _nan
        epoch_profile["avg_power_w"] = _scope.avg_power_w if _scope.avg_power_w is not None else _nan
        epoch_profile["gpu_util_pct"] = _u_avg if _u_avg is not None else _nan
        epoch_profile["mem_util_pct"] = (
            _scope.avg_mem_util_pct if _scope.avg_mem_util_pct is not None else _nan
        )
        # tempo GPU-ATIVA (kernel-only) = tempo da época x util/100 (remove ociosidade/CPU).
        # Usa epoch_time (reduzido entre ranks no DDP), não o elapsed interno do scope.
        epoch_profile["busy_time_s"] = (
            epoch_profile["epoch_time"] * _u_avg / 100.0 if _u_avg is not None else _nan
        )
        return epoch_profile

    if not all_targets or not all_probs:
        epoch_profile = finish_epoch_profile()
        nan_metrics = {
            "loss": float("nan"),
            "auc": float("nan"),
            "precision": float("nan"),
            "sensitivity": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "skipped_batches": skipped_batches,
            "amp_fallback_batches": amp_fallback_batches,
            **epoch_profile,
        }
        if return_predictions:
            nan_metrics["targets"] = torch.empty(0, device=device)
            nan_metrics["probs"] = torch.empty(0, device=device)
        return nan_metrics

    gathered_targets = torch.cat(all_targets, dim=0).view(-1)
    gathered_probs = torch.cat(all_probs, dim=0).view(-1)

    if dist.is_initialized():
        gathered_targets = ddp_concat_variable_length(gathered_targets, device=device).view(-1)
        gathered_probs = ddp_concat_variable_length(gathered_probs, device=device).view(-1)
    else:
        gathered_targets = gathered_targets.view(-1)
        gathered_probs = gathered_probs.view(-1)

    avg_loss = float((total_loss / max(num_batches, 1)).item())
    gathered_targets, gathered_probs, invalid_prediction_count = sanitize_prediction_tensors(
        gathered_targets,
        gathered_probs,
        stage_name=stage_name,
    )
    if gathered_targets.numel() == 0 or gathered_probs.numel() == 0:
        epoch_profile = finish_epoch_profile()
        nan_metrics = {
            "loss": avg_loss,
            "auc": float("nan"),
            "precision": float("nan"),
            "sensitivity": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "skipped_batches": skipped_batches,
            "amp_fallback_batches": amp_fallback_batches,
            "invalid_prediction_count": invalid_prediction_count,
            **epoch_profile,
        }
        if return_predictions:
            nan_metrics["targets"] = gathered_targets
            nan_metrics["probs"] = gathered_probs
        return nan_metrics
    metrics = compute_metrics(gathered_targets, gathered_probs)

    if train and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
        metrics["auc"] = float("nan")

    epoch_profile = finish_epoch_profile()

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
        results["targets"] = gathered_targets
        results["probs"] = gathered_probs
    return results


def main():
    """Main training."""
    args = get_args()
    ensure_cuda_ready()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    rank, world_size, local_rank, device = init_distributed()
    amp_dtype = resolve_amp_dtype(args)
    args.resolved_amp_dtype = amp_dtype
    args.use_channels_last = bool(args.enable_channels_last and device.type == "cuda" and not args.enable_dali)
    amp_dtype_name = (
        "off"
        if amp_dtype is None
        else ("bf16" if amp_dtype == torch.bfloat16 else "fp16")
    )

    if rank == 0:
        print(f"Device: {device}")
        print(f"Rank: {rank}/{world_size}")
        print(f"torch.compile: {args.enable_compile}")
        print(f"AMP: {args.enable_amp} ({amp_dtype_name})")
        print(f"channels_last: {args.use_channels_last}")
        if args.enable_channels_last and args.enable_dali and device.type == "cuda":
            print("[INFO] channels_last desativado com DALI para evitar conversão por batch.")
        print(f"EMA: {args.enable_ema}")
        if args.enable_ema:
            print(f"EMA update every: {args.ema_update_every} optimizer step(s)")
        print(f"Finite checks: {args.enable_finite_checks}")
        print(f"Performance profiler: {args.enable_performance_profiler}")
        if args.enable_ema:
            print("Validation/test use EMA shadow weights; best checkpoint stores the EMA state.")
        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            print(
                "Train metrics use original hard labels for comparability; "
                "AUC de treino fica como n/a quando Mixup/CutMix estiver ativo."
            )

        # Print token reduction stats
        token_stats = compute_token_reduction_stats(args.keep_ratio)
        print("\n" + "="*70)
        print("TOKEN REDUCTION STATISTICS")
        print("="*70)
        print(f"Keep Ratio: {token_stats['keep_ratio']:.1%}")
        print(f"Compression: {token_stats['num_input_tokens']} → {token_stats['num_output_tokens']} tokens")
        print(f"Complexity reduction: {token_stats['complexity_reduction_factor']:.1f}x")
        print(f"Memory savings: {token_stats['memory_savings_percent']:.1f}%")

    os.makedirs(args.results_dir, exist_ok=True)
    runtime_profiler = RuntimeProfiler(args.results_dir, rank=rank, project_name="hybrid_token_reduction_opt")
    metrics_csv_path = os.path.join(args.results_dir, f"metrics_exec{args.exec_id}.csv")
    metrics_rows = []
    wall_clock_start = time.perf_counter()

    # Load data
    if rank == 0:
        print("\nLoading dataloaders...")
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

    # Load backbone
    if rank == 0:
        print(f"Loading backbone: {args.backbone}")
    backbone = get_backbone(
        model_name=args.backbone,
        img_size=args.img_size,
        pretrained=True,
        freeze_backbone=False,
    )
    backbone = modify_backbone_for_feature_extraction(backbone, args.backbone)

    # Create model
    model = create_hybrid_token_reduction_model(
        backbone=backbone,
        backbone_feature_dim=args.backbone_dim,
        transformer_dim=args.transformer_dim,
        keep_ratio=args.keep_ratio,
        keep_k=args.keep_k,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        use_cls_token=args.use_cls_token,
        enable_flash_attention=args.enable_flash_attention,
        device=device,
    )
    if args.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if rank == 0:
        print(model_summary_with_token_analysis(
            model,
            f"HybridTokenReduction ({args.backbone}, keep_ratio={args.keep_ratio})",
            keep_ratio=args.keep_ratio,
        ))

    if args.enable_compile:
        if rank == 0 and args.token_warmup_epochs > 0:
            print(
                "[WARN] torch.compile com token_warmup_epochs > 0 pode recompilar quando "
                "o keep_ratio muda; mantenha desativado para throughput previsivel."
            )
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            if rank == 0:
                print("Compiling model with torch.compile(mode='reduce-overhead')...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                if rank == 0:
                    print(f"[WARN] torch.compile falhou; seguindo sem compilar: {exc}")
        elif rank == 0:
            print("[WARN] torch.compile solicitado, mas indisponivel neste ambiente.")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Optimizer, scheduler, AMP, EMA
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

    scaler = torch.amp.GradScaler(device='cuda', enabled=bool(args.enable_amp and amp_dtype == torch.float16))

    ema_manager = None
    if args.enable_ema:
        ema_manager = EMAManager(model, decay=args.ema_decay, update_every=args.ema_update_every)

    profiler_train = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None
    profiler_val = PerformanceProfiler(device=str(device)) if args.enable_performance_profiler else None

    # Training loop
    best_auc = 0.0
    best_epoch = 0
    best_state_dict = None  # best-checkpoint EM MEMÓRIA (RAM), sem IO em disco

    for epoch in range(args.epochs):
        if scheduler is not None:
            scheduler.step()

        # ============ Token Selector Warmup ============
        # Progressivamente ramp-up keep_ratio: 1.0 → target_ratio
        if epoch < args.token_warmup_epochs:
            # Linear interpolation: 1.0 → target_ratio over token_warmup_epochs
            progress = epoch / args.token_warmup_epochs
            current_keep_ratio = 1.0 - (1.0 - args.keep_ratio) * progress
        else:
            current_keep_ratio = args.keep_ratio

        # Update model's token selector
        model_to_update = model.module if world_size > 1 else model
        model_to_update.set_keep_ratio(current_keep_ratio)

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs} | Keep Ratio: {current_keep_ratio:.2%}")
            print(f"{'='*70}")

        # Freeze backbone
        freeze_backbone = epoch < args.freeze_backbone_epochs
        stage_label = "freeze" if freeze_backbone else "finetune"
        for param in model.module.backbone.parameters() if world_size > 1 else model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        if epoch == args.freeze_backbone_epochs and epoch > 0:
            finetune_lr = args.lrate * args.fine_tune_lr_factor
            current_lr = apply_finetune_lr_transition(
                optimizer,
                scheduler,
                target_lr=finetune_lr,
            )
            if rank == 0:
                scheduler_note = (
                    "cosine scheduler realinhado para não ramp-up de volta ao LR pré-unfreeze."
                    if scheduler is not None
                    else "scheduler desativado."
                )
                print(
                    f"[INFO] Backbone unfrozen at epoch {epoch + 1}; finetune LR ajustado para "
                    f"{current_lr:.2e} (fine_tune_lr_factor={args.fine_tune_lr_factor}). "
                    f"{scheduler_note}"
                )

        # Train
        if rank == 0:
            print("Training...")
        train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            train=True, args=args, scaler=scaler,
            ema_manager=ema_manager, profiler=profiler_train,
            runtime_profiler=runtime_profiler, stage_name="train",
        )

        # Validation
        if rank == 0:
            print("Validating...")
        with ema_shadow_scope(ema_manager):
            val_metrics = run_epoch(
                model, val_loader, optimizer, device,
                train=False, args=args, scaler=scaler,
                profiler=profiler_val, runtime_profiler=runtime_profiler, stage_name="val",
            )

            if rank == 0 and val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_epoch = epoch

                # Best-checkpoint EM MEMÓRIA (RAM, clone na CPU) — sem IO em disco.
                # get_state_dict aqui já captura o estado EMA (shadow), preservando a
                # semântica original. Antes: torch.save() a cada melhora (IO).
                best_state_dict = {k: v.detach().cpu().clone() for k, v in get_state_dict(model).items()}
                print(f"[BEST] checkpoint em memória atualizado: epoch={best_epoch} val_auc={best_auc:.6f}")

        if rank == 0:
            print(
                f"\nTrain Loss: {train_metrics['loss']:.4f} | AUC: {format_metric(train_metrics['auc'])} | "
                f"Precision: {format_metric(train_metrics['precision'])} | "
                f"Recall: {format_metric(train_metrics['recall'])} | "
                f"Spec: {format_metric(train_metrics['specificity'])} | "
                f"F1: {format_metric(train_metrics['f1'])}"
            )
            print(
                f"Val   Loss: {val_metrics['loss']:.4f} | AUC: {format_metric(val_metrics['auc'])} | "
                f"Precision: {format_metric(val_metrics['precision'])} | "
                f"Recall: {format_metric(val_metrics['recall'])} | "
                f"Spec: {format_metric(val_metrics['specificity'])} | "
                f"F1: {format_metric(val_metrics['f1'])}"
            )
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

    loaded_best_checkpoint = False
    if best_state_dict is not None:
        if rank == 0:
            print(f"\nRestaurando best checkpoint em memória para avaliação final (epoch={best_epoch}, val_auc={best_auc:.6f})")
        load_model_state_dict(model, best_state_dict)
        loaded_best_checkpoint = True
    elif rank == 0:
        print("\n[WARN] Sem best checkpoint em memória; avaliando estado final do modelo.")

    # Final evaluation
    if rank == 0:
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70)
        if ema_manager and loaded_best_checkpoint:
            print("Using the saved EMA checkpoint directly for test evaluation.")
        elif ema_manager:
            print("[WARN] Best checkpoint missing; falling back to current EMA weights for test evaluation.")

    test_ema_manager = ema_manager if (ema_manager and not loaded_best_checkpoint) else None
    with ema_shadow_scope(test_ema_manager):
        test_metrics = run_epoch(
            model, test_loader, optimizer, device,
            train=False, args=args, scaler=scaler,
            profiler=profiler_val, runtime_profiler=runtime_profiler, stage_name="test",
            return_predictions=True,
        )

    if rank == 0:
        valid_targets, valid_probs, invalid_prediction_count = sanitize_prediction_tensors(
            test_metrics["targets"],
            test_metrics["probs"],
            stage_name="test_final",
        )
        test_metrics["targets"] = valid_targets
        test_metrics["probs"] = valid_probs
        test_metrics["invalid_prediction_count"] = invalid_prediction_count

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
        total_train_time_s = round(time.perf_counter() - wall_clock_start, 1)
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

        print(f"\nTest AUC: {test_metrics['auc']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test Specificity: {test_metrics['specificity']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        if test_metrics.get("skipped_batches", 0) > 0 or test_metrics.get("amp_fallback_batches", 0) > 0:
            print(
                f"Test stability: skipped_batches={test_metrics.get('skipped_batches', 0)} | "
                f"amp_fallbacks={test_metrics.get('amp_fallback_batches', 0)} | "
                f"invalid_predictions={test_metrics.get('invalid_prediction_count', 0)}"
            )

        if profiler_train is not None:
            print(format_profiling_report(
                f"HybridTokenReduction (keep_ratio={args.keep_ratio})",
                test_metrics['auc'],
                test_metrics['recall'],
                test_metrics['specificity'],
                profiler_train,
            ))

        # Save summary
        summary_path = os.path.join(args.results_dir, f"summary_exec{args.exec_id}.txt")
        summary_text = build_training_summary(
            model_name="HybridTokenReduction",
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            eval_split_name="test",
            eval_metrics=test_metrics,
            profiler=profiler_train,
            total_train_time_s=total_train_time_s,
            metadata=[
                ("Backbone", args.backbone),
                ("Keep Ratio", args.keep_ratio),
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

        print(f"\nSummary saved to: {summary_path}")

    runtime_profiler.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

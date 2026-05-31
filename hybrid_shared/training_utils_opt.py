"""
Training utilities for optimized variants.

Mantem os helpers estaveis do shared atual, mas isola decisoes de runtime
e optimizer que so devem afetar as variantes opt.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW

from hybrid_shared.training_utils import (
    LinearWarmupCosineAnnealingScheduler,
    clip_gradients,
    compute_loss,
    count_parameters,
    create_scheduler,
    ddp_concat_variable_length,
    ensure_cuda_ready as ensure_cuda_ready_base,
)


ADAMW_SUPPORTS_FOREACH = "foreach" in inspect.signature(AdamW).parameters
ADAMW_SUPPORTS_FUSED = "fused" in inspect.signature(AdamW).parameters


class EMAManager:
    """
    EMA otimizado para variantes opt.

    Evita realocar clones em cada batch; o shadow fica alocado e é atualizado
    in-place. `apply_shadow`/`restore` usam `copy_` em vez de trocar `.data`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, update_every: int = 1):
        self.model = model
        self.decay = float(decay)
        self.update_every = max(1, int(update_every))
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._init_shadow()

    def _iter_float_params(self):
        for name, param in self.model.named_parameters():
            if param.is_floating_point():
                yield name, param

    def _init_shadow(self) -> None:
        with torch.no_grad():
            for name, param in self._iter_float_params():
                self.shadow[name] = param.detach().clone()

    def update(self) -> None:
        self.num_updates += 1
        if self.num_updates % self.update_every != 0:
            return

        blend = 1.0 - self.decay
        with torch.no_grad():
            for name, param in self._iter_float_params():
                if not param.requires_grad:
                    continue
                value = param.detach()
                shadow = self.shadow.get(name)
                if shadow is None or shadow.shape != value.shape:
                    self.shadow[name] = value.clone()
                    continue
                shadow.lerp_(value.to(device=shadow.device, dtype=shadow.dtype), blend)

    def apply_shadow(self) -> None:
        self.backup = {}
        with torch.no_grad():
            for name, param in self._iter_float_params():
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                self.backup[name] = param.detach().clone()
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))

    def restore(self) -> None:
        if not self.backup:
            return
        with torch.no_grad():
            for name, param in self._iter_float_params():
                backup = self.backup.get(name)
                if backup is not None:
                    param.copy_(backup.to(device=param.device, dtype=param.dtype))
        self.backup.clear()


def configure_cuda_runtime(*, enable_tf32: bool = True, enable_cudnn_benchmark: bool = True) -> None:
    """
    Ativa knobs de runtime baratos para as variantes otimizadas.
    """
    if not torch.cuda.is_available():
        return

    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def ensure_cuda_ready() -> None:
    """
    Mantem a validacao do shared base e aplica runtime otimizado quando houver CUDA.
    """
    ensure_cuda_ready_base()
    configure_cuda_runtime()


def resolve_adamw_impl(requested_impl: str = "auto", *, use_cuda: bool) -> list[str]:
    """
    Resolve a ordem de tentativas para AdamW, priorizando kernels mais eficientes.
    """
    requested = str(requested_impl).lower()
    if requested == "auto":
        candidates = ["fused", "foreach", "default"] if use_cuda else ["default"]
    elif requested == "fused":
        candidates = ["fused", "foreach", "default"] if use_cuda else ["foreach", "default"]
    elif requested == "foreach":
        candidates = ["foreach", "default"]
    else:
        candidates = ["default"]

    supported = []
    for candidate in candidates:
        if candidate == "fused" and not ADAMW_SUPPORTS_FUSED:
            continue
        if candidate == "foreach" and not ADAMW_SUPPORTS_FOREACH:
            continue
        supported.append(candidate)

    if "default" not in supported:
        supported.append("default")

    return supported


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs,
) -> optim.Optimizer:
    """
    Cria otimizador para variantes opt.

    Para AdamW, tenta `fused` e depois `foreach` automaticamente quando CUDA
    estiver disponivel. Os demais otimizadores mantem comportamento simples.
    """
    params = list(model.parameters())
    if not params:
        raise RuntimeError("Model has no parameters to optimize.")

    optimizer_name = optimizer_name.lower()
    use_cuda = any(param.is_cuda for param in params)

    extra_kwargs = dict(kwargs)
    eps = extra_kwargs.pop("eps", 1e-7)
    optimizer_impl = extra_kwargs.pop("optimizer_impl", "auto")
    extra_kwargs.pop("foreach", None)
    extra_kwargs.pop("fused", None)

    if optimizer_name == "adamw":
        last_error = None
        for impl_name in resolve_adamw_impl(optimizer_impl, use_cuda=use_cuda):
            adamw_kwargs = dict(lr=lr, eps=eps, weight_decay=weight_decay, **extra_kwargs)
            if impl_name == "foreach":
                adamw_kwargs["foreach"] = True
            elif impl_name == "fused":
                adamw_kwargs["fused"] = True

            try:
                return optim.AdamW(params, **adamw_kwargs)
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            print(f"[WARN] AdamW impl='{optimizer_impl}' falhou; usando default: {last_error}")
        return optim.AdamW(params, lr=lr, eps=eps, weight_decay=weight_decay, **extra_kwargs)

    if optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, **extra_kwargs)

    if optimizer_name == "sgd":
        momentum = extra_kwargs.pop("momentum", 0.9)
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, **extra_kwargs)

    if optimizer_name == "rmsprop":
        momentum = extra_kwargs.pop("momentum", 0.0)
        return optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            **extra_kwargs,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


__all__ = [
    "EMAManager",
    "LinearWarmupCosineAnnealingScheduler",
    "clip_gradients",
    "compute_loss",
    "configure_cuda_runtime",
    "count_parameters",
    "create_optimizer",
    "create_scheduler",
    "ddp_concat_variable_length",
    "ensure_cuda_ready",
    "resolve_adamw_impl",
]

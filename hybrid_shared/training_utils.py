"""
Training Utilities - Loss, optimizers, schedulers, EMA.

Imita e reutiliza implementação do pytorch_opt.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from typing import Callable, Optional


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """
    Computa loss com BCE + label smoothing + pos_weight + focal loss (opcional).

    Args:
        logits: Output do modelo (antes de sigmoid), shape [B]
        targets: Labels binários [0, 1], shape [B]
        label_smoothing: Smoothing factor (0 desativa)
        pos_weight: Peso para classe positiva no BCE
        focal_gamma: Gamma para Focal Loss (0 desativa)

    Returns:
        Loss escalar
    """
    # Label smoothing
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # BCE com pos_weight
    pos_weight_tensor = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight_tensor, reduction="none"
    )

    # Focal loss (se gamma > 0)
    if focal_gamma > 0:
        prob = torch.sigmoid(logits)
        # Para classe positiva: (1 - probs)^gamma
        # Para classe negativa: probs^gamma
        focal_weight = torch.where(
            targets > 0.5,
            (1 - prob) ** focal_gamma,
            prob ** focal_gamma,
        )
        loss = loss * focal_weight

    return loss.mean()


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs,
) -> optim.Optimizer:
    """
    Cria otimizador.

    Args:
        model: Modelo PyTorch
        optimizer_name: "adamw", "adam", "sgd", "rmsprop"
        lr: Learning rate
        weight_decay: L2 regularization
        **kwargs: Argumentos adicionais específicos do otimizador

    Returns:
        Otimizador PyTorch
    """
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=kwargs.get("momentum", 0.9), **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


class LinearWarmupCosineAnnealingScheduler:
    """
    Learning rate scheduler: linear warmup + cosine annealing.

    Similar to pytorch_opt implementation.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        # suporta tanto single-group (optimizer.defaults tem "lr") quanto multi-group
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.base_lr = self.base_lrs[0]  # atributo legado
        self.current_epoch = 0

    def step(self) -> None:
        """Atualiza LR para a época atual, mantendo a proporção entre param groups."""
        epoch = self.current_epoch

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < self.warmup_epochs:
                lr = self.min_lr + (base_lr - self.min_lr) * (epoch / max(self.warmup_epochs, 1))
            else:
                progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            param_group["lr"] = lr

        self.current_epoch += 1


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "linear_warmup_cosine",
    warmup_epochs: int = 5,
    total_epochs: int = 100,
    min_lr: float = 1e-6,
    **kwargs,
) -> LinearWarmupCosineAnnealingScheduler:
    """
    Cria scheduler de learning rate.

    Args:
        optimizer: Otimizador PyTorch
        scheduler_name: Nome do scheduler suportado
        warmup_epochs: Épocas de warmup
        total_epochs: Épocas totais
        min_lr: Learning rate mínimo
        **kwargs: Reservado para compatibilidade futura

    Returns:
        Scheduler configurado
    """
    if scheduler_name.lower() != "linear_warmup_cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return LinearWarmupCosineAnnealingScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr=min_lr,
    )


class EMAManager:
    """
    Exponential Moving Average para atualizar pesos do modelo.

    Mantém uma versão "exponentially averaged" dos pesos durante treinamento.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: Modelo PyTorch
            decay: Decay factor (0.999 típico)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Copia pesos iniciais
        self._update_shadow()

    def _update_shadow(self) -> None:
        """Atualiza shadow weights com valores atuais do modelo."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone().detach()
                else:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data.clone().detach()

    def apply_shadow(self) -> None:
        """Substitui pesos do modelo pelos shadow weights."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                param.data = self.shadow[name].clone().detach()

    def restore(self) -> None:
        """Restaura pesos originais (desfaz apply_shadow)."""
        if not self.backup:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone().detach()
        self.backup.clear()

    def update(self) -> None:
        """Atualiza shadow weights (chamar após cada batch de treino)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data.clone().detach()
                else:
                    self.shadow[name] = param.data.clone().detach()


def clip_gradients(model: nn.Module, clip_value: float = 1.0) -> None:
    """
    Clipa norma dos gradientes.

    Args:
        model: Modelo PyTorch
        clip_value: Valor máximo da norma
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Conta número de parâmetros.

    Args:
        model: Modelo PyTorch
        trainable_only: Contar apenas parâmetros treináveis

    Returns:
        Número de parâmetros
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def ensure_cuda_ready() -> None:
    """
    Falha cedo quando o scheduler alocou GPU, mas o runtime PyTorch não tem CUDA.
    """
    requested_markers = []
    for env_name in ("SLURM_GPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
        env_value = os.environ.get(env_name)
        if env_value and env_value not in {"0", "(null)", "none", "None"}:
            requested_markers.append(f"{env_name}={env_value}")

    if not requested_markers or torch.cuda.is_available():
        return

    raise RuntimeError(
        "GPU(s) foram solicitadas para o job, mas torch.cuda.is_available() retornou False. "
        f"Ambiente: {', '.join(requested_markers)}. "
        f"Torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, torch_file={torch.__file__}. "
        "Isto normalmente indica um PyTorch CPU-only ou execução fora do container CUDA."
    )


def ddp_concat_variable_length(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Concatena tensores 1D/2D de tamanho variavel entre ranks, semelhante ao
    padrao usado nas versoes base/opt.

    Args:
        tensor: Tensor local a ser agregado.
        device: Device de comunicacao. Necessario quando o backend for NCCL.

    Returns:
        Tensor concatenado de todos os ranks.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)

    original_dim = tensor.dim()
    if original_dim == 1:
        tensor = tensor.unsqueeze(-1)

    comm_device = tensor.device
    if comm_device.type == "cpu":
        backend = dist.get_backend()
        if backend == "nccl":
            if device is None:
                raise ValueError("device must be provided for NCCL all_gather")
            comm_device = device
            tensor = tensor.to(comm_device)

    tensor = tensor.contiguous()
    local_size = torch.tensor([tensor.shape[0]], device=comm_device, dtype=torch.long)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_sizes, local_size)

    max_size = int(max(size.item() for size in gathered_sizes))
    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tuple(tensor.shape[1:])
        padding = torch.zeros(pad_shape, device=comm_device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)

    pieces = []
    for gathered_tensor, size in zip(gathered_tensors, gathered_sizes):
        pieces.append(gathered_tensor[: size.item()])

    output = torch.cat(pieces, dim=0)
    if original_dim == 1:
        output = output.squeeze(-1)
    return output

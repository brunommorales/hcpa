# -*- coding: utf-8 -*-
"""
Treinamento PyTorch distribuído (DDP) com leitura de TFRecords e coleta de métricas:
- AUC, sensibilidade, especificidade
- Throughput (img/s) por época
- Tempo de época e tempo total
- Memória de GPU (peak/current via nvidia-smi, fallback CUDA)
- ROC/thresholds exportados
"""
import argparse
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List
from os.path import join

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import timm
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from lib.dali_pipeline import build_dali_pipeline, create_dali_iterator


# ---------------------------
# Args
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento PyTorch distribuído com TFRecord + DDP")
    parser.add_argument("--tfrec_dir", type=str, default="./data/all-tfrec", help="Diretório com TFRecords")
    parser.add_argument("--dataset", type=str, default="all", help="Nome lógico do dataset")
    parser.add_argument("--results", type=str, default="./results/all", help="Diretório de saída dos resultados")
    parser.add_argument("--exec", type=int, default=0, help="ID da execução")
    parser.add_argument("--img_sizes", type=int, default=299, help="Tamanho das imagens (quadradas)")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch por GPU (batch global = batch_size * GPUs)")
    parser.add_argument("--epochs", type=int, default=200, help="Épocas totais")
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate (fase inicial ou única)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (apenas no rank 0)")
    parser.add_argument("--model", type=str, default="inception_v3", help="Backbone timm (ex: inception_v3)")
    parser.add_argument(
        "--normalize",
        type=str,
        default="preprocess",
        choices=["preprocess", "raw255", "unit"],
        help="Normalização aplicada após o decode",
    )
    parser.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Habilita augmentações simples (flip/rot/brilho/contraste/saturação)",
    )
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Desabilita augmentações")
    parser.set_defaults(augment=True)
    parser.add_argument("--cores", type=int, default=0, help="Define OMP_NUM_THREADS/threads do PyTorch (<=0 mantém)")
    parser.add_argument("--seed", type=int, default=42, help="Seed base para reprodutibilidade")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Valor de clipping de norma de gradiente (<=0 desativa clipping)",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=3,
        help="Épocas iniciais treinando apenas a cabeça (backbone congelado)",
    )
    parser.add_argument(
        "--fine_tune_lr_factor",
        type=float,
        default=0.1,
        help="Fator aplicado ao LR ao liberar o backbone (<=0 mantém o mesmo LR)",
    )
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=5e-4,
        help="Learning rate absoluto para o fine-tune (sobrepõe o fator se >0); em freeze_epochs=0 vira LR único",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Número de épocas de warmup para o learning rate scheduler",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Learning rate mínimo para o cosine annealing",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help="Alpha da distribuição Beta para mixup (<=0 desativa)",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=0.0,
        help="Alpha da distribuição Beta para cutmix (<=0 desativa)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing aplicado ao alvo (0 desativa)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="Gamma da focal loss aplicada sobre BCE (0 desativa)",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Peso da classe positiva (pos_weight do BCEWithLogits)",
    )
    parser.add_argument(
        "--tta_views",
        type=int,
        default=1,
        help="Número de vistas para TTA na avaliação final (1 desativa)",
    )
    parser.add_argument(
        "--fundus_crop_ratio",
        type=float,
        default=0.9,
        help="Crop central antes do resize (<=0 ou >1 desativa)",
    )
    return parser.parse_args()


# Normalização ImageNet padrão (para ResNet, VGG, etc.)
IMAGENET_MEAN_255 = [x * 255.0 for x in (0.485, 0.456, 0.406)]
IMAGENET_STD_255 = [x * 255.0 for x in (0.229, 0.224, 0.225)]

# Normalização InceptionV3/Xception: (x / 127.5) - 1.0 = (x - 127.5) / 127.5
INCEPTION_MEAN_255 = [127.5, 127.5, 127.5]
INCEPTION_STD_255 = [127.5, 127.5, 127.5]

# Mapeamento de backbone para normalização correta
BACKBONE_NORMALIZATION = {
    "inception_v3": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "inception_resnet_v2": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "xception": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    # Modelos com normalização ImageNet padrão
    "resnet50": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "resnet101": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b0": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b1": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b2": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b3": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b4": (IMAGENET_MEAN_255, IMAGENET_STD_255),
}

def get_normalization_for_backbone(model_name: str):
    """Retorna (mean, std) apropriados para o backbone."""
    model_key = model_name.lower().replace("-", "_")
    if model_key in BACKBONE_NORMALIZATION:
        return BACKBONE_NORMALIZATION[model_key]
    # Para InceptionV3 e variantes, usar normalização [-1, 1]
    if "inception" in model_key or "xception" in model_key:
        return (INCEPTION_MEAN_255, INCEPTION_STD_255)
    # Default: ImageNet
    return (IMAGENET_MEAN_255, IMAGENET_STD_255)


class EMAManager:
    """EMA simples para suavizar pesos e aproximar o pipeline TF."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, start_epoch: int = 0):
        self.model = model
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.shadow = {}
        self.backup = {}
        self._init_shadow()

    def _init_shadow(self):
        self.shadow = {name: p.detach().clone() for name, p in self.model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


def _read_gpu_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        vals = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            vals.append(float(line))
        if vals:
            return max(vals)
    except Exception:
        return None
    return None


# ---------------------------
# TFRecord loader (DALI)
# ---------------------------
def list_split_files(base_dir: str, split_prefix: str) -> list[str]:
    base = Path(base_dir)
    return sorted(str(p) for p in base.glob(f"{split_prefix}*.tfrec"))


def infer_index_file(tfrec_path: str) -> str:
    p = Path(tfrec_path)
    candidates = [p.with_suffix(".tfrec.idx"), p.with_suffix(".idx"), Path(str(p) + ".idx")]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Índice não encontrado para {tfrec_path} (esperado .tfrec.idx ou .idx)")


def build_dali_loader(
    tfrec_files: list[str],
    *,
    batch_size: int,
    image_size: int,
    device_id: int,
    shard_id: int,
    num_shards: int,
    augment: bool,
    seed: int,
    fundus_crop_ratio: float,
    random_shuffle: bool,
    model_name: str = "inception_v3",
):
    """Constrói loader DALI com normalização apropriada para o backbone."""
    idx_files = [infer_index_file(f) for f in tfrec_files]
    # Usa normalização correta para o backbone
    mean, std = get_normalization_for_backbone(model_name)
    pipe = build_dali_pipeline(
        tfrec_files=tfrec_files,
        idx_files=idx_files,
        batch_size=batch_size,
        image_size=image_size,
        num_threads=4,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=seed,
        enable_augment=augment,
        output_layout="NCHW",
        fundus_crop_ratio=fundus_crop_ratio,
        mean=mean,
        std=std,
        random_shuffle=random_shuffle,
    )
    return create_dali_iterator([pipe], auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)


# ---------------------------
# Métricas auxiliares
# ---------------------------
def compute_sens_spec(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5):
    y_true = y_true.float()
    y_pred = (y_prob >= threshold).float()
    tp = torch.sum(y_true * y_pred)
    fn = torch.sum(y_true * (1.0 - y_pred))
    tn = torch.sum((1.0 - y_true) * (1.0 - y_pred))
    fp = torch.sum((1.0 - y_true) * y_pred)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    return sens.item(), spec.item()


# ---------------------------
# Mixup / Cutmix helpers
# ---------------------------
def rand_bbox(width: int, height: int, lam: float):
    r = np.sqrt(1.0 - lam)
    cut_w = int(width * r)
    cut_h = int(height * r)
    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(xb, yb, mixup_alpha: float, cutmix_alpha: float):
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return xb, yb
    use_mixup = mixup_alpha > 0.0 and cutmix_alpha > 0.0 and (np.random.rand() < 0.5)
    if mixup_alpha > 0.0 and cutmix_alpha <= 0.0:
        use_mixup = True
    if cutmix_alpha > 0.0 and mixup_alpha <= 0.0:
        use_mixup = False
    perm = torch.randperm(xb.size(0), device=xb.device)
    x2, y2 = xb[perm], yb[perm]
    if use_mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        xb = lam * xb + (1.0 - lam) * x2
        yb = lam * yb + (1.0 - lam) * y2
        return xb, yb
    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    _, _, H, W = xb.shape
    x1, y1, x2b, y2b = rand_bbox(W, H, lam)
    xb[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]
    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / float(W * H))
    yb = lam_adj * yb + (1.0 - lam_adj) * y2
    return xb, yb


# ---------------------------
# Modelo
# ---------------------------
def create_backbone(model_name: str, img_size: int):
    """
    Cria backbone puro PyTorch usando timm com 1 logit de saída.
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=1,
        in_chans=3,
        global_pool="avg",
    )
    return model


def parameters_of_trainable(model: torch.nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            yield p


def set_backbone_trainable(model: torch.nn.Module, trainable: bool):
    for name, param in model.named_parameters():
        head_like = ("fc" in name) or ("classifier" in name) or ("head" in name)
        param.requires_grad = True if head_like else trainable


def unwrap_compiled_module(model: torch.nn.Module):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def ddp_concat_variable_length(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    local_len = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.int64)
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    max_len = int(torch.max(torch.stack(lengths)).item())
    padded = torch.zeros((max_len,) + tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
    padded[: tensor.shape[0]] = tensor
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    slices = []
    for g, l in zip(gathered, lengths):
        slices.append(g[: int(l.item())])
    return torch.cat(slices, dim=0)


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    
    # Otimizações CUDA para performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")

    if args.cores > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.cores)
        torch.set_num_threads(args.cores)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    MODEL_NAME = args.model
    IMG = args.img_sizes
    LOCAL_BS = args.batch_size
    GLOBAL_BATCH = LOCAL_BS * max(world_size, 1)
    EPOCHS = args.epochs
    LR = args.lrate
    FREEZE_EPOCHS = max(0, args.freeze_epochs)
    FINE_TUNE_LR = args.fine_tune_lr if args.fine_tune_lr > 0 else LR * max(args.fine_tune_lr_factor, 1e-6)
    MIXUP_ALPHA = max(0.0, float(args.mixup_alpha))
    CUTMIX_ALPHA = max(0.0, float(args.cutmix_alpha))
    LABEL_SMOOTHING = max(0.0, float(args.label_smoothing))
    FOCAL_GAMMA = max(0.0, float(args.focal_gamma))
    POS_WEIGHT = max(0.0, float(args.pos_weight))
    TTA_VIEWS = max(1, int(args.tta_views))
    FUNDUS_CROP_RATIO = args.fundus_crop_ratio if 0.0 < float(args.fundus_crop_ratio) <= 1.0 else 1.0
    VERBOSE = args.verbose if rank == 0 else 0
    HEAD_WEIGHT_DECAY = 1e-4
    FINE_TUNE_WEIGHT_DECAY = 1e-5

    train_files = list_split_files(args.tfrec_dir, "train")
    valid_files = list_split_files(args.tfrec_dir, "test")
    if not train_files or not valid_files:
        raise SystemExit("É necessário ter TFRecords de treino e validação.")

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{MODEL_NAME}-{args.exec}.csv"
    csv_fields = [
        "epoch",
        "stage",
        "train_loss",
        "train_auc",
        "train_sens",
        "train_spec",
        "train_throughput_img_s",
        "train_elapsed_s",
        "train_gpu_mem_alloc_mb",
        "train_gpu_mem_reserved_mb",
        "val_loss",
        "val_auc",
        "val_sens",
        "val_spec",
        "val_throughput_img_s",
        "val_elapsed_s",
        "val_gpu_mem_alloc_mb",
        "val_gpu_mem_reserved_mb",
        "lr",
        "total_train_time_s",
    ]
    csv_writer = None
    if rank == 0:
        csv_writer = pd.DataFrame(columns=csv_fields)

    dali_seed = args.seed + rank
    if rank == 0:
        norm_mean, norm_std = get_normalization_for_backbone(MODEL_NAME)
        print(f"[DALI] Ativo: decode/resize/augment na GPU | shuffle=train | layout=NCHW")
        print(f"[DALI] Normalização para {MODEL_NAME}: mean={norm_mean}, std={norm_std}")

    train_loader = build_dali_loader(
        train_files,
        batch_size=LOCAL_BS,
        image_size=IMG,
        device_id=local_rank if use_cuda else 0,
        shard_id=rank,
        num_shards=world_size,
        augment=args.augment,
        seed=dali_seed,
        fundus_crop_ratio=FUNDUS_CROP_RATIO,
        random_shuffle=True,
        model_name=MODEL_NAME,
    )
    valid_loader = build_dali_loader(
        valid_files,
        batch_size=LOCAL_BS,
        image_size=IMG,
        device_id=local_rank if use_cuda else 0,
        shard_id=rank,
        num_shards=world_size,
        augment=False,
        seed=dali_seed + 999,
        fundus_crop_ratio=FUNDUS_CROP_RATIO,
        random_shuffle=False,
        model_name=MODEL_NAME,
    )

    model = create_backbone(MODEL_NAME, IMG)
    if torch.cuda.is_available():
        try:
            # Usar reduce-overhead para melhor estabilidade numérica
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            if rank == 0:
                print(f"[WARN] torch.compile falhou; seguindo sem compilar: {exc}")

    ddp_model = model.to(device)
    core_model = ddp_model
    if world_size > 1:
        ddp_model = DDP(ddp_model, device_ids=[local_rank] if use_cuda else None, find_unused_parameters=True)
        core_model = ddp_model.module
    use_amp = use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pos_weight_tensor = torch.tensor(POS_WEIGHT, device=device) if POS_WEIGHT > 0 else None

    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """BCE com label smoothing, pos_weight e focal opcional."""
        tgt = targets
        if LABEL_SMOOTHING > 0.0:
            tgt = tgt * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, tgt, pos_weight=pos_weight_tensor, reduction="none"
        )
        if FOCAL_GAMMA > 0.0:
            prob = torch.sigmoid(logits)
            pt = prob * tgt + (1.0 - prob) * (1.0 - tgt)
            focal_factor = (1.0 - pt).pow(FOCAL_GAMMA)
            loss = loss * focal_factor
        return loss.mean()

    def rebuild_optimizer(lr: float, weight_decay: float):
        params = list(parameters_of_trainable(core_model))
        if len(params) == 0:
            raise RuntimeError("Nenhum parâmetro treinável encontrado ao criar o otimizador.")
        return AdamW(params, lr=lr, eps=1e-7, weight_decay=weight_decay), params

    if FREEZE_EPOCHS > 0:
        set_backbone_trainable(core_model, False)
        current_stage = "freeze"
        opt, trainable_params = rebuild_optimizer(LR, HEAD_WEIGHT_DECAY)
    else:
        set_backbone_trainable(core_model, True)
        current_stage = "finetune"
        opt, trainable_params = rebuild_optimizer(FINE_TUNE_LR, FINE_TUNE_WEIGHT_DECAY)
    if rank == 0:
        print(
            f"[Train] freeze_epochs={FREEZE_EPOCHS} base_lr={LR:.2e} fine_tune_lr={FINE_TUNE_LR:.2e} stage_inicial={current_stage}"
        )
    clip_norm = max(0.0, float(args.clip_grad_norm))
    
    # LR Scheduler com Warmup + Cosine Annealing
    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 3)))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    
    def create_scheduler(optimizer, base_lr, total_epochs, warmup_ep):
        """Cria scheduler com warmup linear + cosine annealing."""
        if warmup_ep <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
        
        def warmup_fn(epoch):
            if epoch < warmup_ep:
                return (epoch + 1) / warmup_ep
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_ep), eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_ep])
    
    scheduler = create_scheduler(opt, LR if FREEZE_EPOCHS > 0 else FINE_TUNE_LR, EPOCHS, warmup_epochs)
    if rank == 0:
        print(f"[Scheduler] Warmup epochs={warmup_epochs}, min_lr={min_lr:.2e}, Cosine Annealing")
    ema_start_epoch = max(0, int(0.6 * EPOCHS))
    ema_helper = EMAManager(core_model, decay=0.999, start_epoch=ema_start_epoch) if rank == 0 else None
    if rank == 0 and ema_helper is not None:
        print(f"[EMA] decay=0.999 start_epoch={ema_start_epoch}")

    def mixup_alpha_for_epoch(epoch: int) -> float:
        """Replica o decaimento progressivo usado no script TF."""
        if MIXUP_ALPHA <= 0.0:
            return 0.0
        eff_total = max(1, EPOCHS - FREEZE_EPOCHS)
        progress = (epoch - FREEZE_EPOCHS) / eff_total
        if progress < 0.0:
            decay = 1.0
        elif progress < 0.6:
            decay = 1.0
        else:
            decay = max(0.0, 1.0 - (progress - 0.6) / 0.4)
        return MIXUP_ALPHA * decay

    def run_epoch(
        loader,
        train: bool,
        epoch_idx: Optional[int] = None,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        ema_helper: Optional[EMAManager] = None,
    ) -> Tuple[float, Optional[float], float, float, float, float, Optional[float], Optional[float]]:
        if loader is None:
            return 0.0, float("nan"), float("nan"), float("nan"), 0.0, 0.0, None, None
        ddp_model.train() if train else ddp_model.eval()

        track_memory = use_cuda and (device.type == "cuda")
        if track_memory:
            torch.cuda.reset_peak_memory_stats(device)

        running_loss = 0.0
        n_samples = 0
        all_probs = []
        all_labels = []
        t_start = time.time()
        for batch in loader:
            data = batch[0] if isinstance(batch, list) else batch
            xb = data["image"]  # já em GPU, layout NCHW
            yb_original = data["label"].squeeze(-1)  # Labels originais (binários)
            yb = yb_original.clone()

            if train:
                xb, yb = apply_mixup_cutmix(xb, yb, mixup_alpha, cutmix_alpha)

            if train:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = ddp_model(xb).squeeze(1)
                    loss = compute_loss(logits, yb)
                if not torch.isfinite(loss):
                    if rank == 0:
                        where = f" epoch={epoch_idx}" if epoch_idx is not None else ""
                        print(f"[WARN] loss não finita{where}; batch ignorado.")
                    continue
                scaler.scale(loss).backward()
                if clip_norm > 0.0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
                scaler.step(opt)
                scaler.update()
                if ema_helper is not None and epoch_idx is not None and epoch_idx >= ema_helper.start_epoch:
                    ema_helper.update()
            else:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    logits = ddp_model(xb).squeeze(1)
                    loss = compute_loss(logits, yb)

            running_loss += loss.item() * xb.shape[0]
            n_samples += xb.shape[0]
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_probs.append(probs.detach())
                # Usar labels originais (binários) para métricas, não labels mixados
                all_labels.append(yb_original.detach())

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        probs_all = ddp_concat_variable_length(all_probs)
        labels_all = ddp_concat_variable_length(all_labels)

        sens_val, spec_val = float("nan"), float("nan")
        try:
            sens_val, spec_val = compute_sens_spec(labels_all, probs_all)
        except Exception:
            pass
        try:
            auc_val = roc_auc_score(labels_all.cpu().numpy(), probs_all.cpu().numpy())
        except Exception:
            auc_val = float("nan")
        epoch_loss = running_loss / max(n_samples, 1)
        if track_memory:
            torch.cuda.synchronize(device)
            smi_used = _read_gpu_used_mb()
            if smi_used is not None:
                peak_mem_alloc = smi_used
                peak_mem_reserved = smi_used
            else:
                peak_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                peak_mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        else:
            peak_mem_alloc = None
            peak_mem_reserved = None
        elapsed = time.time() - t_start
        global_samples = n_samples
        if dist.is_initialized():
            n_samples_tensor = torch.tensor([n_samples], device=device, dtype=torch.float64)
            dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
            global_samples = n_samples_tensor.item()
        throughput = global_samples / elapsed if elapsed > 0 else 0.0
        return epoch_loss, auc_val, sens_val, spec_val, throughput, elapsed, peak_mem_alloc, peak_mem_reserved

    best_val_auc = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        if current_stage == "freeze" and epoch >= FREEZE_EPOCHS:
            set_backbone_trainable(core_model, True)
            opt, trainable_params = rebuild_optimizer(FINE_TUNE_LR, FINE_TUNE_WEIGHT_DECAY)
            # Recria scheduler para fase de fine-tuning
            scheduler = create_scheduler(opt, FINE_TUNE_LR, EPOCHS - epoch, warmup_ep=warmup_epochs)
            current_stage = "finetune"
            if rank == 0:
                print(f"[Stage] Liberando backbone no epoch {epoch} | lr={FINE_TUNE_LR:.2e}")

        mixup_alpha_now = mixup_alpha_for_epoch(epoch)
        cutmix_alpha_now = CUTMIX_ALPHA

        train_loss, train_auc, train_sens, train_spec, train_thpt, train_elapsed, train_mem_alloc, train_mem_reserved = run_epoch(
            train_loader,
            train=True,
            epoch_idx=epoch,
            mixup_alpha=mixup_alpha_now,
            cutmix_alpha=cutmix_alpha_now,
            ema_helper=ema_helper,
        )
        val_loss, val_auc, val_sens, val_spec, val_thpt, val_elapsed, val_mem_alloc, val_mem_reserved = run_epoch(
            valid_loader, train=False, epoch_idx=epoch, mixup_alpha=0.0, cutmix_alpha=0.0, ema_helper=None
        )
        
        # Atualiza learning rate scheduler
        scheduler.step()

        if rank == 0:
            stage_label = current_stage
            print(
                f"[{stage_label} E{epoch}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"trainAUC={train_auc:.4f} valAUC={val_auc:.4f} thr={train_thpt:.1f} img/s lr={opt.param_groups[0]['lr']:.2e}"
            )
            if csv_writer is not None:
                new_row = {
                    "epoch": epoch,
                    "stage": stage_label,
                    "train_loss": train_loss,
                    "train_auc": train_auc,
                    "train_sens": train_sens,
                    "train_spec": train_spec,
                    "train_throughput_img_s": train_thpt,
                    "train_elapsed_s": train_elapsed,
                    "train_gpu_mem_alloc_mb": train_mem_alloc,
                    "train_gpu_mem_reserved_mb": train_mem_reserved,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_sens": val_sens,
                    "val_spec": val_spec,
                    "val_throughput_img_s": val_thpt,
                    "val_elapsed_s": val_elapsed,
                    "val_gpu_mem_alloc_mb": val_mem_alloc,
                    "val_gpu_mem_reserved_mb": val_mem_reserved,
                    "lr": opt.param_groups[0]["lr"],
                    "total_train_time_s": None,
                }
                csv_writer.loc[len(csv_writer)] = new_row
                csv_writer.to_csv(csv_path, index=False)

        if (rank == 0) and (val_auc > best_val_auc):
            best_val_auc = val_auc
            best_epoch = epoch

    if rank == 0:
        eval_start = time.time()
        # garante início limpo do iterador DALI
        try:
            valid_loader.reset()
        except Exception:
            pass
        model_for_eval = unwrap_compiled_module(core_model)
        if ema_helper is not None:
            ema_helper.apply()

        model_for_eval.eval()

        y_true = []
        y_score = []
        with torch.no_grad():
            for batch in valid_loader:
                data = batch[0] if isinstance(batch, list) else batch
                xb = data["image"]
                yb = data["label"].squeeze(-1)
                views = [xb]
                if TTA_VIEWS > 1:
                    for view_idx in range(1, TTA_VIEWS):
                        mode = view_idx % 4
                        if mode == 1:
                            views.append(torch.flip(xb, dims=[3]))  # horizontal
                        elif mode == 2:
                            views.append(torch.flip(xb, dims=[2]))  # vertical
                        elif mode == 3:
                            views.append(torch.rot90(xb, k=1, dims=(2, 3)))
                        else:
                            views.append(xb)

                probs_acc = None
                for view in views:
                    logits = model_for_eval(view).squeeze(1)
                    probs_view = torch.sigmoid(logits)
                    probs_acc = probs_view if probs_acc is None else probs_acc + probs_view
                probs = (probs_acc / float(len(views))).cpu().numpy()
                y_score.append(probs)
                y_true.append(yb.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        sens_final, spec_final = compute_sens_spec(torch.from_numpy(y_true), torch.from_numpy(y_score))
        eval_elapsed = time.time() - eval_start
        val_throughput = (len(y_true) / eval_elapsed) if eval_elapsed > 0 else 0.0

        thresholds_df = pd.DataFrame(
            {
                "thresholds": thresholds,
                "tpr": tpr,
                "fpr": fpr,
            }
        )
        thresholds_df["sens"] = thresholds_df["tpr"]
        thresholds_df["spec"] = 1.0 - thresholds_df["fpr"]
        thresholds_path = results_dir / f"{args.dataset}-{args.exec}-thresholds.csv"
        thresholds_df.to_csv(thresholds_path, index=False, encoding="utf-8")

        plt.figure()
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="best")
        pdf_path = results_dir / f"{args.dataset}-{args.exec}.pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close()

        final_elapsed = round(time.time() - start_time, 1)
        if csv_writer is not None:
            final_row = {
                "epoch": EPOCHS,
                "stage": "final_eval",
                "train_loss": np.nan,
                "train_auc": np.nan,
                "train_sens": np.nan,
                "train_spec": np.nan,
                "train_throughput_img_s": np.nan,
                "train_elapsed_s": np.nan,
                "train_gpu_mem_alloc_mb": None,
                "train_gpu_mem_reserved_mb": None,
                "val_loss": float("nan"),
                "val_auc": auc_val,
                "val_sens": sens_final,
                "val_spec": spec_final,
                "val_throughput_img_s": val_throughput,
                "val_elapsed_s": eval_elapsed,
                "val_gpu_mem_alloc_mb": None,
                "val_gpu_mem_reserved_mb": None,
                "lr": opt.param_groups[0]["lr"],
                "total_train_time_s": final_elapsed,
            }
            csv_writer.loc[len(csv_writer)] = final_row
            csv_writer.to_csv(csv_path, index=False)
        print(f"Valid AUC (final): {auc_val:.4f} | best_val_auc: {best_val_auc:.4f} @ epoch {best_epoch}")
        print(f"{args.dataset},{args.exec},{auc_val:.6f},{final_elapsed}")
        if ema_helper is not None:
            ema_helper.restore()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

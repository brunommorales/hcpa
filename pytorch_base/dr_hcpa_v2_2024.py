# -*- coding: utf-8 -*-
"""
Treinamento PyTorch distribuído com leitura direta de TFRecords.

Este script mantém o pipeline base sem DALI/EMA, mas permite alinhar a
receita de treino com a versão otimizada para comparações controladas.
"""
import argparse
import io
import os
import random
import struct
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
from os.path import join

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import timm
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tfrecord import example_pb2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import Adam, AdamW, SGD, RMSprop, Adadelta
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.v2 as T

from lib.runtime_profiling import RuntimeProfiler


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


# ---------------------------
# Args
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Treino básico em GPU com DDP")
    parser.add_argument("--tfrec_dir", type=str, default="/home/users/bmmorales/projects/hcpa/data/all-tfrec", help="Diretório com TFRecords")
    parser.add_argument("--dataset", type=str, default="all", help="Nome lógico do dataset")
    parser.add_argument("--results", type=str, default="./results/all", help="Diretório de saída dos resultados")
    parser.add_argument("--exec", type=int, default=0, help="ID da execução")
    parser.add_argument("--img_sizes", type=int, default=299, help="Tamanho das imagens (quadradas)")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch por GPU (batch global = batch_size * GPUs)")
    parser.add_argument("--epochs", type=int, default=200, help="Épocas totais")
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=["adam", "adamw", "sgd_mom", "rmsprop", "adadelta"],
        help="Otimizador a usar (padrão: adamw)"
    )
    parser.add_argument("--num_thresholds", type=int, default=200, help="Número de thresholds para ROC")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (apenas no rank 0)")
    parser.add_argument(
        "--model",
        type=str,
        default="inception_v3",
        help="Backbone timm (ex.: inception_v3)",
    )
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
        default=0,
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
        help="Learning rate absoluto para o fine-tune (sobrepõe o fator se >0)",
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
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing para regularização (0.0 desativa, 0.1 recomendado)",
    )
    parser.add_argument(
        "--fundus_crop_ratio",
        type=float,
        default=1.0,
        help="Crop central antes do resize (<=0 ou >1 desativa)",
    )
    parser.add_argument("--enable_amp", dest="use_amp", action="store_true", help="Habilita AMP/FP16")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false", help="Desabilita AMP/FP16")
    parser.set_defaults(use_amp=False)
    parser.add_argument(
        "--enable_cosine",
        dest="use_cosine_scheduler",
        action="store_true",
        help="Habilita warmup + cosine annealing",
    )
    parser.add_argument(
        "--disable_cosine",
        dest="use_cosine_scheduler",
        action="store_false",
        help="Mantém learning rate constante",
    )
    parser.set_defaults(use_cosine_scheduler=False)
    return parser.parse_args()


# --- Normalização por backbone ---
IMAGENET_MEAN_255 = [x * 255.0 for x in (0.485, 0.456, 0.406)]
IMAGENET_STD_255 = [x * 255.0 for x in (0.229, 0.224, 0.225)]

INCEPTION_MEAN_255 = [127.5, 127.5, 127.5]
INCEPTION_STD_255 = [127.5, 127.5, 127.5]

BACKBONE_NORMALIZATION = {
    "inceptionv3": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "inception_v3": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "inceptionresnetv2": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "inception_resnet_v2": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "xception": (INCEPTION_MEAN_255, INCEPTION_STD_255),
    "resnet50": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "resnet101": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnetb0": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b0": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnetb1": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b1": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnetb2": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b2": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnetb3": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b3": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnetb4": (IMAGENET_MEAN_255, IMAGENET_STD_255),
    "efficientnet_b4": (IMAGENET_MEAN_255, IMAGENET_STD_255),
}


def _canonical_model_key(model_name: str) -> str:
    return str(model_name).lower().replace("-", "_")


def get_normalization_for_backbone(model_name: str):
    key = _canonical_model_key(model_name)
    if key in BACKBONE_NORMALIZATION:
        return BACKBONE_NORMALIZATION[key]
    if "inception" in key or "xception" in key:
        return (INCEPTION_MEAN_255, INCEPTION_STD_255)
    return (IMAGENET_MEAN_255, IMAGENET_STD_255)


# ---------------------------
# GPU-based Augmentations
# ---------------------------
class GPUAugmentation:
    """
    Augmentações otimizadas para execução na GPU usando torchvision.transforms.v2.
    Aplicadas após transferência para GPU para evitar gargalo CPU-GPU.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.15),
                T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0),
            ])
        else:
            self.transforms = None
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica augmentações em batch de imagens.
        Args:
            x: Tensor [B, C, H, W] ou [B, H, W, C] em float32
        Returns:
            Tensor augmentado no mesmo formato
        """
        if not self.enabled or self.transforms is None:
            return x
        # Garante formato channel-first antes das transforms v2 para evitar erro de canais=H/W
        if x.dim() == 4 and x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1).contiguous()
        x = self.transforms(x)
        return x.clamp_(0.0, 255.0)


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
# TFRecord loader (PyTorch)
# ---------------------------
class RetinaTFRecord(Dataset):
    """
    Lê TFRecords com chaves:
      - "imagem": bytes JPEG
      - "retinopatia": int (0/1)
    """

    def __init__(
        self,
        tfrecord_paths,
        index_paths,
        img_size=299,
        model_name="inception_v3",
        normalize_mode="preprocess",
        augment=False,
        apply_normalization=True,
        fundus_crop_ratio=1.0,
    ):
        if len(tfrecord_paths) != len(index_paths):
            raise ValueError("Listas de TFRecords e índices possuem tamanhos distintos.")

        self.img_size = img_size
        self.model_name = model_name
        self.normalize_mode = normalize_mode
        self.apply_normalization = bool(apply_normalization)
        self.fundus_crop_ratio = fundus_crop_ratio if 0.0 < float(fundus_crop_ratio) <= 1.0 else 1.0
        self.norm_mean, self.norm_std = get_normalization_for_backbone(model_name)
        self.augment = augment
        self._entries = []  # lista de tuplas (path, offset)
        self._file_handles = {}  # cache de file handles

        for tf_path, idx_path in zip(tfrecord_paths, index_paths):
            tf_path = Path(tf_path)
            idx_path = Path(idx_path)

            if not tf_path.exists():
                raise FileNotFoundError(f"TFRecord ausente: {tf_path}")
            if not idx_path.exists():
                raise FileNotFoundError(f"Arquivo .idx ausente: {idx_path}")

            offsets = np.loadtxt(str(idx_path), dtype=np.int64, usecols=0, ndmin=1)
            offsets = np.atleast_1d(offsets).astype(np.int64).tolist()
            for off in offsets:
                self._entries.append((tf_path, int(off)))

        if len(self._entries) == 0:
            raise RuntimeError("Nenhuma entrada encontrada no conjunto de TFRecords/idx.")

    def _center_crop_fundus(self, img: Image.Image) -> Image.Image:
        if self.fundus_crop_ratio >= 0.999:
            return img
        width, height = img.size
        crop_w = max(1, int(round(width * self.fundus_crop_ratio)))
        crop_h = max(1, int(round(height * self.fundus_crop_ratio)))
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        return img.crop((left, top, left + crop_w, top + crop_h))

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        if not self.augment:
            return img
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
        if np.random.rand() < 0.1:
            img = ImageOps.flip(img)
        if np.random.rand() < 0.25:
            angle = float(np.random.uniform(-10.0, 10.0))
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        if np.random.rand() < 0.3:
            img = ImageEnhance.Contrast(img).enhance(float(np.random.uniform(0.9, 1.1)))
        if np.random.rand() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(float(np.random.uniform(0.9, 1.1)))
        if np.random.rand() < 0.2:
            img = ImageEnhance.Color(img).enhance(float(np.random.uniform(0.9, 1.1)))
        return img

    def __len__(self):
        return len(self._entries)

    def _open(self, path: Path):
        p = str(path)
        fh = self._file_handles.get(p)
        if fh is None:
            fh = open(p, "rb")
            self._file_handles[p] = fh
        return fh

    def _read_record_at(self, tf_path: Path, offset: int) -> bytes:
        fh = self._open(tf_path)
        fh.seek(offset)
        length_bytes = fh.read(8)
        if len(length_bytes) != 8:
            raise EOFError(f"Offset inválido: {offset}")
        length = struct.unpack("<Q", length_bytes)[0]
        _ = fh.read(4)  # crc len
        data = fh.read(length)
        _ = fh.read(4)  # crc data
        return data

    def __getitem__(self, idx):
        tf_path, offset = self._entries[idx]
        data = self._read_record_at(tf_path, offset)
        ex = example_pb2.Example()
        ex.ParseFromString(data)

        feats = ex.features.feature
        if ("imagem" not in feats) or ("retinopatia" not in feats):
            raise KeyError("TFRecord não contém as chaves esperadas 'imagem'/'retinopatia'.")

        img_bytes = feats["imagem"].bytes_list.value[0]
        label = float(feats["retinopatia"].int64_list.value[0])

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = self._center_crop_fundus(img)
        if self.img_size and img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        img = self._apply_augmentations(img)

        x = np.asarray(img, dtype=np.float32)  # [H,W,C], 0..255
        if self.apply_normalization:
            x = self._normalize_array(x)

        x = torch.from_numpy(x)  # [H,W,C]
        # Transforma para channel-first para compatibilidade com PyTorch e augmentação na GPU
        if x.dim() == 3:
            x = x.permute(2, 0, 1).contiguous()  # [C,H,W]
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def _normalize_array(self, image: np.ndarray) -> np.ndarray:
        if self.normalize_mode == "preprocess":
            mean = np.asarray(self.norm_mean, dtype=np.float32)
            std = np.asarray(self.norm_std, dtype=np.float32)
            return (image - mean) / std
        if self.normalize_mode == "unit":
            return image / 255.0
        return image

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handles"] = {}
        return state

    def __del__(self):
        try:
            for fh in self._file_handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
        except Exception:
            pass


# ---------------------------
# Modelo
# ---------------------------
def create_backbone(model_name: str, input_shape: Tuple[int, int, int]):
    """Cria backbone puro PyTorch usando timm com 1 logit de saída."""
    timm_key = _canonical_model_key(model_name)
    if not hasattr(timm, "is_model") or not timm.is_model(timm_key):
        raise ValueError(f"Backbone timm desconhecido: {model_name}")
    model = timm.create_model(
        timm_key,
        pretrained=True,
        num_classes=1,
        in_chans=3,
        global_pool="avg",
    )
    return model, "timm"


def parameters_of_trainable(model) -> Iterable[torch.Tensor]:
    for p in model.parameters():
        if p.requires_grad:
            yield p


def unwrap_compiled_module(module):
    """Recupera módulo original (caso torch.compile seja usado em outro contexto)."""
    current = module
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        next_mod = getattr(current, "_orig_mod", None)
        if next_mod is None:
            next_mod = getattr(current, "_original_module", None)
        if next_mod is None or next_mod is current:
            break
        current = next_mod
    return current or module


# ---------------------------
# Util: concat DDP de tamanho variável (para AUC global)
# ---------------------------
def ddp_concat_variable_length(t: torch.Tensor):
    if not dist.is_initialized():
        return t.detach().cpu()
    n_local = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(n_local) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, n_local)
    maxn = int(torch.stack(sizes).max().item())
    pad = maxn - t.shape[0]
    if pad > 0:
        pad_val = -1.0 if t.dtype.is_floating_point else -1
        t = torch.cat([t, torch.full((pad,), pad_val, device=t.device, dtype=t.dtype)], dim=0)
    bufs = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(bufs, t)
    out = torch.cat(bufs, dim=0)
    if out.dtype.is_floating_point:
        mask = out >= 0.0
    else:
        mask = out >= 0
    return out[mask].detach().cpu()


SPECIFICITY_TARGET_SENSITIVITY = 0.95


def specificity_at_sensitivity(
    labels: torch.Tensor,
    probs: torch.Tensor,
    target_sensitivity: float = SPECIFICITY_TARGET_SENSITIVITY,
) -> float:
    labels = labels.reshape(-1).to(dtype=torch.float32)
    probs = probs.reshape(-1).to(device=labels.device, dtype=torch.float32)
    finite = torch.isfinite(labels) & torch.isfinite(probs)
    labels = labels[finite]
    probs = probs[finite].clamp(1e-7, 1.0 - 1e-7)
    if labels.numel() == 0 or probs.numel() == 0:
        return float("nan")

    positives = labels > 0.5
    n_pos = int(positives.sum().item())
    n_neg = int((~positives).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    target = float(target_sensitivity)
    if target <= 0.0:
        return 1.0
    if target > 1.0:
        return float("nan")

    order = torch.argsort(probs, descending=True, stable=True)
    sorted_scores = probs[order]
    sorted_true = positives[order]
    group_end = torch.ones(sorted_scores.numel(), device=sorted_scores.device, dtype=torch.bool)
    if sorted_scores.numel() > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_idxs = torch.nonzero(group_end, as_tuple=False).flatten()
    tp = torch.cumsum(sorted_true.to(torch.float64), dim=0)[threshold_idxs]
    fp = torch.cumsum((~sorted_true).to(torch.float64), dim=0)[threshold_idxs]
    tpr = torch.cat([torch.zeros(1, device=probs.device, dtype=torch.float64), tp / float(n_pos)])
    fpr = torch.cat([torch.zeros(1, device=probs.device, dtype=torch.float64), fp / float(n_neg)])

    target_tensor = torch.tensor(target, device=probs.device, dtype=torch.float64)
    target_idxs = torch.nonzero(tpr >= target_tensor, as_tuple=False).flatten()
    if target_idxs.numel() == 0:
        return 0.0
    idx = int(target_idxs[0].item())
    if idx == 0:
        fpr_at_target = fpr[0]
    else:
        tpr_lo, tpr_hi = tpr[idx - 1], tpr[idx]
        fpr_lo, fpr_hi = fpr[idx - 1], fpr[idx]
        if float(torch.abs(tpr_hi - tpr_lo).item()) <= 1e-12:
            fpr_at_target = torch.minimum(fpr_lo, fpr_hi)
        else:
            fraction = (target_tensor - tpr_lo) / (tpr_hi - tpr_lo)
            fpr_at_target = fpr_lo + fraction * (fpr_hi - fpr_lo)
    return float(max(0.0, min(1.0, 1.0 - float(fpr_at_target.item()))))


def compute_binary_metrics(labels: torch.Tensor, probs: torch.Tensor, threshold: float = 0.5):
    """Compute threshold metrics and specificity at 95% sensitivity."""
    preds = (probs >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    specificity = specificity_at_sensitivity(labels, probs)
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    return {
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "specificity_at_sens95": specificity,
        "f1": f1,
    }


# ---------------------------
# Helpers
# ---------------------------
def infer_idx(p: Path) -> Path:
    cand1 = p.with_suffix(".idx")
    cand2 = Path(str(p) + ".idx")
    return cand2 if cand2.exists() else cand1


def set_global_seed(seed: Optional[int]):
    if seed is None:
        return
    try:
        s = int(seed)
    except Exception:
        return
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def init_distributed(use_cuda: bool) -> Tuple[int, int, int]:
    if not dist.is_initialized():
        if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
            backend = "nccl" if use_cuda else "gloo"
            dist.init_process_group(backend=backend)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, rank, local_rank


def set_backbone_trainable(model, trainable: bool):
    """Congela ou libera apenas o backbone."""
    for name, param in model.named_parameters():
        head_like = ("fc" in name) or ("classifier" in name) or ("head" in name)
        param.requires_grad = True if head_like else trainable


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Otimizações CUDA para performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cores = max(0, int(getattr(args, "cores", 0)))
    if cores > 0:
        os.environ["OMP_NUM_THREADS"] = str(cores)
        try:
            torch.set_num_threads(cores)
        except Exception:
            pass

    IMG = int(args.img_sizes)
    EPOCHS = int(args.epochs)
    LR = float(args.lrate)
    freeze_epochs = max(0, int(getattr(args, "freeze_epochs", 0)))
    ft_factor = float(getattr(args, "fine_tune_lr_factor", 0.0))
    ft_lr_override = float(getattr(args, "fine_tune_lr", -1.0))
    fine_tune_lr = ft_lr_override if ft_lr_override > 0 else (LR * ft_factor if ft_factor > 0 else LR)
    VERBOSE = args.verbose
    MODEL_NAME = args.model
    IMAGE_SIZE = (IMG, IMG, 3)
    HEAD_WEIGHT_DECAY = 1e-4
    FINE_TUNE_WEIGHT_DECAY = 1e-5

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise SystemExit("[Hardware] Execução requer GPU, mas nenhuma está disponível.")
    use_cuda = True

    world_size, rank, local_rank = init_distributed(use_cuda)
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    seed_base = int(getattr(args, "seed", 42))
    set_global_seed(seed_base + rank)

    if rank == 0:
        chosen = f"cuda:{local_rank}"
        print(f"[Hardware] usando {chosen} | world_size={world_size}")
        if cores > 0:
            print(f"[Hardware] OMP_NUM_THREADS={cores}")
        print(f"[Seed] base={seed_base} rank={rank} -> {seed_base + rank}")

    LOCAL_BS = int(args.batch_size)  # batch por GPU/rank
    GLOBAL_BS = LOCAL_BS * max(world_size, 1)

    # listar TFRecords e idx
    tfrec_dir = Path(args.tfrec_dir)
    train_paths = sorted(tfrec_dir.glob("train*.tfrec"))
    val_paths = sorted(tfrec_dir.glob("val*.tfrec")) + sorted(tfrec_dir.glob("valid*.tfrec"))
    test_paths = sorted(tfrec_dir.glob("test*.tfrec"))
    valid_paths = val_paths or test_paths
    test_paths = test_paths or valid_paths
    if rank == 0:
        print(
            f"Encontrados {len(train_paths)} TFRecords de treino, "
            f"{len(valid_paths)} de validação e {len(test_paths)} de teste."
        )
    if not train_paths or not valid_paths or not test_paths:
        raise SystemExit("É necessário ao menos um TFRecord de treino e um de validação/teste.")

    train_idx = [infer_idx(p) for p in train_paths]
    valid_idx = [infer_idx(p) for p in valid_paths]
    test_idx = [infer_idx(p) for p in test_paths]

    # Augmentação será feita na GPU, então desabilita no Dataset
    train_ds = RetinaTFRecord(
        [str(p) for p in train_paths],
        [str(p) for p in train_idx],
        img_size=IMG,
        model_name=MODEL_NAME,
        normalize_mode=args.normalize,
        augment=bool(args.augment),
        apply_normalization=True,
        fundus_crop_ratio=float(getattr(args, "fundus_crop_ratio", 1.0)),
    )
    valid_ds = RetinaTFRecord(
        [str(p) for p in valid_paths],
        [str(p) for p in valid_idx],
        img_size=IMG,
        model_name=MODEL_NAME,
        normalize_mode=args.normalize,
        augment=False,
        apply_normalization=True,
        fundus_crop_ratio=float(getattr(args, "fundus_crop_ratio", 1.0)),
    )
    test_ds = RetinaTFRecord(
        [str(p) for p in test_paths],
        [str(p) for p in test_idx],
        img_size=IMG,
        model_name=MODEL_NAME,
        normalize_mode=args.normalize,
        augment=False,
        apply_normalization=True,
        fundus_crop_ratio=float(getattr(args, "fundus_crop_ratio", 1.0)),
    )

    gpu_augment = GPUAugmentation(enabled=False)
    if rank == 0:
        print(
            f"[Augmentation] dataset_side={args.augment} gpu_side=False "
            f"crop_ratio={float(getattr(args, 'fundus_crop_ratio', 1.0)):.2f}"
        )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    valid_sampler = DistributedSampler(valid_ds, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if world_size > 1 else None

    num_workers = min(8, os.cpu_count() or 4)
    if num_workers < 1:
        num_workers = 1
    train_loader = DataLoader(
        train_ds,
        batch_size=LOCAL_BS,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=LOCAL_BS,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=LOCAL_BS,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    if rank == 0:
        print(f"[DataLoader] num_workers={num_workers}, persistent_workers=True, prefetch_factor=4")

    model, model_backend = create_backbone(MODEL_NAME, IMAGE_SIZE)
    if rank == 0 and VERBOSE:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] backend={model_backend} total_params={total_params} trainable_params={trainable_params_count}")

    ddp_model = model.to(device)
    core_model = ddp_model
    if world_size > 1:
        ddp_model = DDP(ddp_model, device_ids=[local_rank] if use_cuda else None, find_unused_parameters=True)
        core_model = ddp_model.module

    def rebuild_optimizer(lr: float, weight_decay: float):
        params = list(parameters_of_trainable(core_model))
        if len(params) == 0:
            raise RuntimeError("Nenhum parâmetro treinável encontrado ao criar o otimizador.")
        opt_name = str(getattr(args, "optimizer", "adamw")).lower()
        if opt_name in ("adamw", "adam_w"):
            return AdamW(params, lr=lr, eps=1e-7, weight_decay=weight_decay), params
        if opt_name == "adam":
            return Adam(params, lr=lr, eps=1e-7, weight_decay=weight_decay), params
        if opt_name == "sgd" or opt_name == "sgd_mom" or opt_name == "sgd-mom":
            return SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay), params
        if opt_name == "rmsprop":
            return RMSprop(params, lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay), params
        if opt_name == "adadelta":
            return Adadelta(params, lr=lr, rho=0.95, eps=1e-6, weight_decay=weight_decay), params
        return AdamW(params, lr=lr, eps=1e-7, weight_decay=weight_decay), params

    # congela backbone se necessário e cria otimizador da fase atual
    if freeze_epochs > 0:
        set_backbone_trainable(core_model, False)
        current_stage = "freeze"
        opt, trainable_params = rebuild_optimizer(LR, HEAD_WEIGHT_DECAY)
    else:
        set_backbone_trainable(core_model, True)
        current_stage = "finetune"
        opt, trainable_params = rebuild_optimizer(fine_tune_lr, FINE_TUNE_WEIGHT_DECAY)
    if rank == 0:
        print(
            f"[Train] freeze_epochs={freeze_epochs} base_lr={LR:.2e} "
            f"fine_tune_lr={fine_tune_lr:.2e} stage_inicial={current_stage}"
        )
    
    # Label smoothing para regularização
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))
    
    def smooth_labels(targets, smoothing):
        """Aplica label smoothing em targets binários."""
        if smoothing <= 0.0:
            return targets
        # Para classificação binária: labels suavizados
        # 0 -> smoothing/2, 1 -> 1 - smoothing/2
        return targets * (1.0 - smoothing) + smoothing / 2.0
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    if rank == 0 and label_smoothing > 0:
        print(f"[Label Smoothing] Ativado com valor={label_smoothing:.3f}")
    
    clip_norm = max(0.0, float(getattr(args, "clip_grad_norm", 1.0)))
    
    # Automatic Mixed Precision
    amp_requested = bool(getattr(args, "use_amp", False))
    use_amp = use_cuda and amp_requested
    scaler = GradScaler(enabled=use_amp)
    if rank == 0:
        print(f"[AMP] requested={amp_requested} active={use_amp}")
    
    # LR Scheduler com Warmup + Cosine Annealing
    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 3)))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    use_cosine_scheduler = bool(getattr(args, "use_cosine_scheduler", False))
    
    def create_scheduler(optimizer, base_lr, total_epochs, warmup_ep):
        """Cria scheduler com warmup linear + cosine annealing."""
        if not use_cosine_scheduler:
            return None
        if warmup_ep <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
        
        # Warmup linear
        def warmup_fn(epoch):
            if epoch < warmup_ep:
                return (epoch + 1) / warmup_ep
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_ep), eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_ep])
    
    scheduler = create_scheduler(opt, LR if freeze_epochs > 0 else fine_tune_lr, EPOCHS, warmup_epochs)
    if rank == 0:
        if use_cosine_scheduler:
            print(f"[Scheduler] Warmup epochs={warmup_epochs}, min_lr={min_lr:.2e}, Cosine Annealing")
        else:
            print("[Scheduler] Desativado; learning rate constante.")

    def normalize_batch(xb: torch.Tensor) -> torch.Tensor:
        return xb.float()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    runtime_profiler = RuntimeProfiler(results_dir, rank=rank, project_name="pytorch_base")
    csv_path = results_dir / f"{MODEL_NAME}-{args.exec}.csv"
    csv_writer = None
    csv_fields = COMMON_CSV_FIELDS
    if rank == 0:
        csv_writer = pd.DataFrame(columns=csv_fields)

    def run_epoch(
        loader: DataLoader,
        train: bool,
        epoch_idx: Optional[int] = None,
    ):
        stage_name = "train" if train else "val"
        if loader is None:
            return 0.0, float("nan"), {
                "precision": float("nan"),
                "recall": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
                "f1": float("nan"),
            }, 0.0, 0.0, 0.0, float("nan"), float("nan"), None
        if train:
            ddp_model.train()
        else:
            ddp_model.eval()

        track_memory = use_cuda and (device.type == "cuda")

        running_loss = 0.0
        n_samples = 0
        all_probs = []
        all_labels = []
        memory_samples_mb = []
        inference_time_s = 0.0
        inference_batches = 0
        inference_samples = 0
        t_start = time.time()
        num_batches = 0
        loader_iter = iter(loader)
        while True:
            try:
                xb, yb = runtime_profiler.fetch_next(loader_iter, stage_name)
            except StopIteration:
                break

            with runtime_profiler.range(f"{stage_name}/batch"):
                with runtime_profiler.range(f"{stage_name}/h2d"):
                    xb = xb.to(device, non_blocking=True).float()
                    yb = yb.to(device, non_blocking=True).float()

                if train:
                    with runtime_profiler.range(f"{stage_name}/augment"):
                        xb = gpu_augment(xb)

                with runtime_profiler.range(f"{stage_name}/preprocess"):
                    xb = normalize_batch(xb)
                    xb = xb.contiguous(memory_format=torch.channels_last)

                if train:
                    yb_smooth = smooth_labels(yb, label_smoothing)
                    with runtime_profiler.range(f"{stage_name}/optimizer_zero_grad"):
                        opt.zero_grad(set_to_none=True)
                    with runtime_profiler.range(f"{stage_name}/forward"):
                        with autocast(enabled=use_amp, dtype=torch.float16):
                            logits = ddp_model(xb)
                            loss = loss_fn(logits.squeeze(1), yb_smooth)
                    if not torch.isfinite(loss):
                        if rank == 0:
                            where = f" epoch={epoch_idx}" if epoch_idx is not None else ""
                            print(f"[WARN] loss não finita{where}; batch ignorado.")
                        runtime_profiler.step(stage_name)
                        continue
                    with runtime_profiler.range(f"{stage_name}/backward"):
                        scaler.scale(loss).backward()
                    with runtime_profiler.range(f"{stage_name}/optimizer_step"):
                        if clip_norm > 0.0:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
                        scaler.step(opt)
                        scaler.update()
                else:
                    if track_memory:
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with runtime_profiler.range(f"{stage_name}/forward"):
                        with torch.no_grad(), autocast(enabled=use_amp, dtype=torch.float16):
                            logits = ddp_model(xb)
                            loss = loss_fn(logits.squeeze(1), yb)
                    if track_memory:
                        torch.cuda.synchronize(device)
                    inference_time_s += time.perf_counter() - inference_start
                    inference_batches += 1
                    inference_samples += int(xb.shape[0])

                with runtime_profiler.range(f"{stage_name}/metrics"):
                    running_loss += loss.item() * xb.shape[0]
                    n_samples += xb.shape[0]
                    num_batches += 1
                    with torch.no_grad():
                        probs = torch.sigmoid(logits).squeeze(1)
                        all_probs.append(probs.detach())
                        all_labels.append(yb.detach())

                if track_memory:
                    torch.cuda.synchronize(device)
                    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    memory_samples_mb.append(max(allocated_mb, reserved_mb))

            runtime_profiler.step(stage_name)

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        probs_all = ddp_concat_variable_length(all_probs)
        labels_all = ddp_concat_variable_length(all_labels)

        metric_values = {
            "precision": float("nan"),
            "recall": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
        }
        try:
            metric_values.update(compute_binary_metrics(labels_all, probs_all))
        except Exception:
            pass
        try:
            auc_val = roc_auc_score(labels_all.numpy(), probs_all.numpy())
        except Exception:
            auc_val = float("nan")
        epoch_loss = running_loss / max(n_samples, 1)
        memory_avg_mb = (
            sum(memory_samples_mb) / len(memory_samples_mb)
            if memory_samples_mb
            else (0.0 if not track_memory else None)
        )
        inference_latency_ms_img = (
            inference_time_s / inference_samples * 1000.0 if inference_samples > 0 else float("nan")
        )
        inference_latency_ms_batch = (
            inference_time_s / inference_batches * 1000.0 if inference_batches > 0 else float("nan")
        )
        elapsed = time.time() - t_start
        avg_batch_time_ms = (elapsed / max(num_batches, 1)) * 1000.0 if num_batches > 0 else 0.0
        # throughput global: soma samples de todos os ranks
        global_samples = n_samples
        if dist.is_initialized():
            n_samples_tensor = torch.tensor([n_samples], device=device, dtype=torch.float64)
            dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
            global_samples = n_samples_tensor.item()
            elapsed_tensor = torch.tensor([elapsed], device=device, dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            elapsed = float(elapsed_tensor.item())
            mem_tensor = torch.tensor(
                [0.0 if memory_avg_mb is None else memory_avg_mb],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(mem_tensor, op=dist.ReduceOp.SUM)
            memory_avg_mb = float(mem_tensor.item()) / max(world_size, 1)
            latency_tensor = torch.tensor(
                [
                    0.0 if np.isnan(inference_latency_ms_img) else inference_latency_ms_img,
                    0.0 if np.isnan(inference_latency_ms_batch) else inference_latency_ms_batch,
                    1.0 if inference_samples > 0 else 0.0,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(latency_tensor, op=dist.ReduceOp.SUM)
            latency_ranks = max(float(latency_tensor[2].item()), 1.0)
            if latency_tensor[2].item() > 0:
                inference_latency_ms_img = float(latency_tensor[0].item()) / latency_ranks
                inference_latency_ms_batch = float(latency_tensor[1].item()) / latency_ranks
            avg_batch_time_ms = (elapsed / max(num_batches, 1)) * 1000.0 if num_batches > 0 else 0.0
        throughput = global_samples / elapsed if elapsed > 0 else 0.0
        return (
            epoch_loss,
            auc_val,
            metric_values,
            throughput,
            elapsed,
            avg_batch_time_ms,
            inference_latency_ms_img,
            inference_latency_ms_batch,
            memory_avg_mb,
        )

    best_val_auc = -1.0
    best_state_dict = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        if current_stage == "freeze" and epoch >= freeze_epochs:
            set_backbone_trainable(core_model, True)
            opt, trainable_params = rebuild_optimizer(fine_tune_lr, FINE_TUNE_WEIGHT_DECAY)
            scheduler = create_scheduler(opt, fine_tune_lr, EPOCHS - epoch, warmup_ep=warmup_epochs)
            current_stage = "finetune"
            if rank == 0:
                print(f"[Stage] Liberando backbone no epoch {epoch} | lr={fine_tune_lr:.2e}")

        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        (
            train_loss,
            train_auc,
            train_epoch_metrics,
            train_thpt,
            train_elapsed,
            train_avg_batch_time_ms,
            train_inference_latency_ms_img,
            train_inference_latency_ms_batch,
            train_mem_avg,
        ) = run_epoch(
            train_loader, train=True, epoch_idx=epoch
        )
        (
            val_loss,
            val_auc,
            val_epoch_metrics,
            val_thpt,
            val_elapsed,
            val_avg_batch_time_ms,
            val_inference_latency_ms_img,
            val_inference_latency_ms_batch,
            val_mem_avg,
        ) = run_epoch(
            valid_loader, train=False, epoch_idx=epoch
        )
        
        if scheduler is not None:
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
                    "train_precision": train_epoch_metrics["precision"],
                    "train_recall": train_epoch_metrics["recall"],
                    "train_f1": train_epoch_metrics["f1"],
                    "train_sens": train_epoch_metrics["sensitivity"],
                    "train_spec": train_epoch_metrics["specificity"],
                    "train_spec_at_sens95": train_epoch_metrics["specificity_at_sens95"],
                    "train_throughput_img_s": train_thpt,
                    "train_elapsed_s": train_elapsed,
                    "train_avg_batch_time_ms": train_avg_batch_time_ms,
                    "train_inference_latency_ms_img": train_inference_latency_ms_img,
                    "train_inference_latency_ms_batch": train_inference_latency_ms_batch,
                    "train_gpu_mem_avg_mb": train_mem_avg,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_precision": val_epoch_metrics["precision"],
                    "val_recall": val_epoch_metrics["recall"],
                    "val_f1": val_epoch_metrics["f1"],
                    "val_sens": val_epoch_metrics["sensitivity"],
                    "val_spec": val_epoch_metrics["specificity"],
                    "val_spec_at_sens95": val_epoch_metrics["specificity_at_sens95"],
                    "val_throughput_img_s": val_thpt,
                    "val_elapsed_s": val_elapsed,
                    "val_avg_batch_time_ms": val_avg_batch_time_ms,
                    "val_inference_latency_ms_img": val_inference_latency_ms_img,
                    "val_inference_latency_ms_batch": val_inference_latency_ms_batch,
                    "val_gpu_mem_avg_mb": val_mem_avg,
                    "test_loss": np.nan,
                    "test_auc": np.nan,
                    "test_precision": np.nan,
                    "test_recall": np.nan,
                    "test_f1": np.nan,
                    "test_sens": np.nan,
                    "test_spec": np.nan,
                    "test_spec_at_sens95": np.nan,
                    "test_throughput_img_s": np.nan,
                    "test_elapsed_s": np.nan,
                    "test_avg_batch_time_ms": np.nan,
                    "test_inference_latency_ms_img": np.nan,
                    "test_inference_latency_ms_batch": np.nan,
                    "test_gpu_mem_avg_mb": np.nan,
                    "lr": opt.param_groups[0]["lr"],
                    "total_train_time_s": None,
                }
                csv_writer.loc[len(csv_writer)] = new_row
                csv_writer.to_csv(csv_path, index=False)

        if (rank == 0) and (val_auc > best_val_auc):
            best_val_auc = val_auc
            # guarda melhor estado em memória para evitar escrita em disco
            to_save = unwrap_compiled_module(core_model)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in to_save.state_dict().items()}

    if rank == 0:
        eval_start = time.time()
        eval_loader = DataLoader(
            test_ds,
            batch_size=LOCAL_BS,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=use_cuda,
            drop_last=False,
        )
        model_for_eval = unwrap_compiled_module(core_model)
        if best_state_dict is not None:
            try:
                model_for_eval.load_state_dict(best_state_dict, strict=False)
            except Exception as exc:
                print(f"[WARN] falha ao aplicar melhor estado em memória: {exc}")

        model_for_eval.eval()

        y_true = []
        y_score = []
        eval_num_batches = 0
        eval_memory_samples_mb = []
        eval_inference_time_s = 0.0
        eval_inference_samples = 0
        with torch.no_grad():
            eval_iter = iter(eval_loader)
            while True:
                try:
                    xb, yb = runtime_profiler.fetch_next(eval_iter, "eval")
                except StopIteration:
                    break
                with runtime_profiler.range("eval/batch"):
                    with runtime_profiler.range("eval/h2d"):
                        xb = xb.to(device, non_blocking=True).float()
                    with runtime_profiler.range("eval/preprocess"):
                        xb = normalize_batch(xb)
                        xb = xb.contiguous(memory_format=torch.channels_last)
                    if use_cuda and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with runtime_profiler.range("eval/forward"):
                        logits = model_for_eval(xb)
                    if use_cuda and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    eval_inference_time_s += time.perf_counter() - inference_start
                    eval_inference_samples += int(xb.shape[0])
                    with runtime_profiler.range("eval/metrics"):
                        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                        y_score.append(probs)
                        y_true.append(yb.numpy())
                        eval_num_batches += 1
                    if use_cuda and device.type == "cuda":
                        allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                        eval_memory_samples_mb.append(max(allocated_mb, reserved_mb))
                runtime_profiler.step("eval")
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        final_metrics = compute_binary_metrics(torch.from_numpy(y_true), torch.from_numpy(y_score))
        eval_elapsed = time.time() - eval_start
        test_throughput = (len(test_ds) / eval_elapsed) if eval_elapsed > 0 else 0.0
        test_avg_batch_time_ms = (eval_elapsed / max(eval_num_batches, 1)) * 1000.0 if eval_num_batches > 0 else 0.0
        test_inference_latency_ms_img = (
            eval_inference_time_s / eval_inference_samples * 1000.0
            if eval_inference_samples > 0
            else float("nan")
        )
        test_inference_latency_ms_batch = (
            eval_inference_time_s / eval_num_batches * 1000.0
            if eval_num_batches > 0
            else float("nan")
        )
        test_mem_avg = (
            sum(eval_memory_samples_mb) / len(eval_memory_samples_mb)
            if eval_memory_samples_mb
            else None
        )

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
                "stage": "final_test",
                "train_loss": np.nan,
                "train_auc": np.nan,
                "train_precision": np.nan,
                "train_recall": np.nan,
                "train_f1": np.nan,
                "train_sens": np.nan,
                "train_spec": np.nan,
                "train_spec_at_sens95": np.nan,
                "train_throughput_img_s": np.nan,
                "train_elapsed_s": np.nan,
                "train_avg_batch_time_ms": np.nan,
                "train_inference_latency_ms_img": np.nan,
                "train_inference_latency_ms_batch": np.nan,
                "train_gpu_mem_avg_mb": np.nan,
                "val_loss": float("nan"),
                "val_auc": np.nan,
                "val_precision": np.nan,
                "val_recall": np.nan,
                "val_f1": np.nan,
                "val_sens": np.nan,
                "val_spec": np.nan,
                "val_spec_at_sens95": np.nan,
                "val_throughput_img_s": np.nan,
                "val_elapsed_s": np.nan,
                "val_avg_batch_time_ms": np.nan,
                "val_inference_latency_ms_img": np.nan,
                "val_inference_latency_ms_batch": np.nan,
                "val_gpu_mem_avg_mb": np.nan,
                "test_loss": float("nan"),
                "test_auc": auc_val,
                "test_precision": final_metrics["precision"],
                "test_recall": final_metrics["recall"],
                "test_f1": final_metrics["f1"],
                "test_sens": final_metrics["sensitivity"],
                "test_spec": final_metrics["specificity"],
                "test_spec_at_sens95": final_metrics["specificity_at_sens95"],
                "test_throughput_img_s": test_throughput,
                "test_elapsed_s": eval_elapsed,
                "test_avg_batch_time_ms": test_avg_batch_time_ms,
                "test_inference_latency_ms_img": test_inference_latency_ms_img,
                "test_inference_latency_ms_batch": test_inference_latency_ms_batch,
                "test_gpu_mem_avg_mb": test_mem_avg,
                "lr": opt.param_groups[0]["lr"],
                "total_train_time_s": final_elapsed,
            }
            csv_writer.loc[len(csv_writer)] = final_row
            csv_writer.to_csv(csv_path, index=False)
        print(f"Test AUC (final): {auc_val:.4f}")
        print(f"{args.dataset},{args.exec},{auc_val:.6f},{final_elapsed}")

    runtime_profiler.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

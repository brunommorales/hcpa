# -*- coding: utf-8 -*-
"""
Versão simplificada do treinamento PyTorch distribuído.
- Sem fases de fine-tuning ou agendas especiais.
- Sem DALI, mixup, EMA, AMP ou outras otimizações.
- Mantém leitura direta de TFRecords e DDP para usar múltiplas GPUs.
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
os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from keras import applications as _apps
from keras import layers
import keras
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tfrecord import example_pb2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF


# ---------------------------
# Args
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Treino básico em GPU com DDP")
    parser.add_argument("--tfrec_dir", type=str, default="./data/all", help="Diretório com TFRecords")
    parser.add_argument("--dataset", type=str, default="all", help="Nome lógico do dataset")
    parser.add_argument("--results", type=str, default="./results/all", help="Diretório de saída dos resultados")
    parser.add_argument("--exec", type=int, default=0, help="ID da execução")
    parser.add_argument("--img_sizes", type=int, default=299, help="Tamanho das imagens (quadradas)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch por GPU (batch global = batch_size * GPUs)")
    parser.add_argument("--epochs", type=int, default=50, help="Épocas totais")
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_thresholds", type=int, default=200, help="Número de thresholds para ROC")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity (apenas no rank 0)")
    parser.add_argument("--model", type=str, default="InceptionV3", help="Backbone keras.applications")
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
        default=0.01,
        help="Fator aplicado ao LR ao liberar o backbone (<=0 mantém o mesmo LR)",
    )
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=-1.0,
        help="Learning rate absoluto para o fine-tune (sobrepõe o fator se >0)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=3,
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
    return parser.parse_args()


# --- Normalização por backbone ---
_PREPROCESS_MAP = {
    "InceptionV3": _apps.inception_v3.preprocess_input,
    "Xception": _apps.xception.preprocess_input,
    "EfficientNetB0": _apps.efficientnet.preprocess_input,
    "EfficientNetB1": _apps.efficientnet.preprocess_input,
    "EfficientNetB2": _apps.efficientnet.preprocess_input,
    "EfficientNetB3": _apps.efficientnet.preprocess_input,
    "EfficientNetB4": _apps.efficientnet.preprocess_input,
    "EfficientNetB5": _apps.efficientnet.preprocess_input,
    "EfficientNetB6": _apps.efficientnet.preprocess_input,
    "EfficientNetB7": _apps.efficientnet.preprocess_input,
    "ResNet50": _apps.resnet.preprocess_input,
    "ResNet101": _apps.resnet.preprocess_input,
    "ResNet152": _apps.resnet.preprocess_input,
    "ResNet50V2": _apps.resnet_v2.preprocess_input,
    "ResNet101V2": _apps.resnet_v2.preprocess_input,
    "ResNet152V2": _apps.resnet_v2.preprocess_input,
    "VGG16": _apps.vgg16.preprocess_input,
    "VGG19": _apps.vgg19.preprocess_input,
    "DenseNet121": _apps.densenet.preprocess_input,
    "DenseNet169": _apps.densenet.preprocess_input,
    "DenseNet201": _apps.densenet.preprocess_input,
    "MobileNet": _apps.mobilenet.preprocess_input,
    "MobileNetV2": _apps.mobilenet_v2.preprocess_input,
    "MobileNetV3Small": _apps.mobilenet_v3.preprocess_input,
    "MobileNetV3Large": _apps.mobilenet_v3.preprocess_input,
    "NASNetMobile": _apps.nasnet.preprocess_input,
    "NASNetLarge": _apps.nasnet.preprocess_input,
    "ConvNeXtTiny": _apps.convnext.preprocess_input,
    "ConvNeXtSmall": _apps.convnext.preprocess_input,
    "ConvNeXtBase": _apps.convnext.preprocess_input,
    "ConvNeXtLarge": _apps.convnext.preprocess_input,
    "ConvNeXtXLarge": _apps.convnext.preprocess_input,
}


def get_preprocess_fn(model_name: str):
    return _PREPROCESS_MAP.get(model_name, None)


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
        return self.transforms(x)


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

    def __init__(self, tfrecord_paths, index_paths, img_size=299, model_name="InceptionV3", normalize_mode="preprocess",
                 augment=False):
        if len(tfrecord_paths) != len(index_paths):
            raise ValueError("Listas de TFRecords e índices possuem tamanhos distintos.")

        self.img_size = img_size
        self.normalize_mode = normalize_mode
        self.pre_fn = get_preprocess_fn(model_name) if normalize_mode == "preprocess" else None
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

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        if not self.augment:
            return img
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
        if np.random.rand() < 0.15:
            img = ImageOps.flip(img)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-15.0, 15.0)
            img = img.rotate(angle, resample=Image.BILINEAR)
        if np.random.rand() < 0.35:
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.85, 1.15))
        if np.random.rand() < 0.35:
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.85, 1.15))
        if np.random.rand() < 0.3:
            img = ImageEnhance.Color(img).enhance(np.random.uniform(0.85, 1.15))
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
        label = int(feats["retinopatia"].int64_list.value[0])

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.img_size and img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        img = self._apply_augmentations(img)

        x = np.asarray(img, dtype=np.float32)  # [H,W,C], 0..255
        if self.normalize_mode == "preprocess" and self.pre_fn is not None:
            x = self.pre_fn(x)
        elif self.normalize_mode == "unit":
            x = x / 255.0  # 0..1
        # raw255 mantém 0..255

        x = torch.from_numpy(x)  # [H,W,C]
        # Transforma para channel-first para compatibilidade com PyTorch e augmentação na GPU
        if x.dim() == 3:
            x = x.permute(2, 0, 1).contiguous()  # [C,H,W]
        y = torch.tensor(label, dtype=torch.long)
        return x, y

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
# Modelo (Keras 3 sobre torch)
# ---------------------------
def create_backbone(model_name: str, input_shape: Tuple[int, int, int]):
    """
    Usa keras.applications (compatível com Keras 3) para criar o backbone sem topo.
    Todos os layers ficam treináveis desde o início.
    """
    apps = keras.applications
    factory = {
        "Xception": apps.Xception,
        "VGG16": apps.VGG16,
        "VGG19": apps.VGG19,
        "ResNet50": apps.ResNet50,
        "ResNet50V2": apps.ResNet50V2,
        "ResNet101": apps.ResNet101,
        "ResNet101V2": apps.ResNet101V2,
        "ResNet152": apps.ResNet152,
        "ResNet152V2": apps.ResNet152V2,
        "InceptionV3": apps.InceptionV3,
        "InceptionResNetV2": apps.InceptionResNetV2,
        "MobileNet": apps.MobileNet,
        "MobileNetV2": apps.MobileNetV2,
        "MobileNetV3Small": apps.MobileNetV3Small,
        "MobileNetV3Large": apps.MobileNetV3Large,
        "DenseNet121": apps.DenseNet121,
        "DenseNet169": apps.DenseNet169,
        "DenseNet201": apps.DenseNet201,
        "NASNetMobile": apps.NASNetMobile,
        "NASNetLarge": apps.NASNetLarge,
        "EfficientNetB0": apps.EfficientNetB0,
        "EfficientNetB1": apps.EfficientNetB1,
        "EfficientNetB2": apps.EfficientNetB2,
        "EfficientNetB3": apps.EfficientNetB3,
        "EfficientNetB4": apps.EfficientNetB4,
        "EfficientNetB5": apps.EfficientNetB5,
        "EfficientNetB6": apps.EfficientNetB6,
        "EfficientNetB7": apps.EfficientNetB7,
        "ConvNeXtTiny": apps.ConvNeXtTiny,
        "ConvNeXtSmall": apps.ConvNeXtSmall,
        "ConvNeXtBase": apps.ConvNeXtBase,
        "ConvNeXtLarge": apps.ConvNeXtLarge,
        "ConvNeXtXLarge": apps.ConvNeXtXLarge,
    }
    if model_name not in factory:
        raise ValueError(f"Backbone desconhecido: {model_name}")

    inputs = keras.Input(shape=input_shape)
    base = factory[model_name](weights="imagenet", include_top=False, input_tensor=inputs)
    for layer in base.layers:
        layer.trainable = True
    x = layers.GlobalAveragePooling2D()(base.output)
    outputs = layers.Dense(1, activation=None, dtype="float32")(x)  # logit
    model = keras.Model(inputs, outputs)
    model.base_model = base
    return model


def parameters_of_trainable(model) -> Iterable[torch.Tensor]:
    for w in model.weights:
        if not getattr(w, "trainable", False):
            continue
        t = getattr(w, "value", None)
        if isinstance(t, torch.Tensor):
            if not t.requires_grad:
                t.requires_grad = True
            yield t
        elif isinstance(w, torch.Tensor):
            if not w.requires_grad:
                w.requires_grad = True
            yield w


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


def compute_sens_spec(labels: torch.Tensor, probs: torch.Tensor, threshold: float = 0.5):
    """Compute sensitivity/specificity at a fixed threshold (default 0.5)."""
    preds = (probs >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    sens = (tp / (tp + fn + 1e-8)).item()
    spec = (tn / (tn + fp + 1e-8)).item()
    return sens, spec


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
    """Congela ou libera apenas o backbone (camadas do modelo base)."""
    base = getattr(model, "base_model", None)
    if base is None:
        return
    for layer in base.layers:
        layer.trainable = trainable
    for w in getattr(base, "weights", []):
        t = getattr(w, "value", None)
        if isinstance(t, torch.Tensor):
            t.requires_grad = trainable
        elif isinstance(w, torch.Tensor):
            w.requires_grad = trainable


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
    valid_paths = (
        sorted(tfrec_dir.glob("test*.tfrec"))
        + sorted(tfrec_dir.glob("val*.tfrec"))
        + sorted(tfrec_dir.glob("valid*.tfrec"))
    )
    if rank == 0:
        print(f"Encontrados {len(train_paths)} TFRecords de treino e {len(valid_paths)} de validação.")
    if not train_paths or not valid_paths:
        raise SystemExit("É necessário ao menos um TFRecord de treino e um de validação.")

    train_idx = [infer_idx(p) for p in train_paths]
    valid_idx = [infer_idx(p) for p in valid_paths]

    # Augmentação será feita na GPU, então desabilita no Dataset
    train_ds = RetinaTFRecord(
        [str(p) for p in train_paths],
        [str(p) for p in train_idx],
        img_size=IMG,
        model_name=MODEL_NAME,
        normalize_mode=args.normalize,
        augment=False,  # Augmentação será feita na GPU
    )
    valid_ds = RetinaTFRecord(
        [str(p) for p in valid_paths],
        [str(p) for p in valid_idx],
        img_size=IMG,
        model_name=MODEL_NAME,
        normalize_mode=args.normalize,
        augment=False,
    )
    
    # GPU Augmentation - aplicada no loop de treino
    gpu_augment = GPUAugmentation(enabled=args.augment)
    if rank == 0:
        print(f"[Augmentation] GPU augmentation habilitado: {args.augment}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    valid_sampler = DistributedSampler(valid_ds, shuffle=False) if world_size > 1 else None

    num_workers = max(4, cores) if cores > 0 else min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=LOCAL_BS,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,  # Drop last para batches uniformes
        persistent_workers=True if num_workers > 0 else False,  # Mantém workers vivos
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch mais batches
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=LOCAL_BS,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=max(2, num_workers // 2),
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    if rank == 0:
        print(f"[DataLoader] num_workers={num_workers}, persistent_workers=True, prefetch_factor=4")

    model = create_backbone(MODEL_NAME, IMAGE_SIZE)
    if rank == 0 and VERBOSE:
        model.summary()

    ddp_model = model.to(device)
    core_model = ddp_model
    if world_size > 1:
        ddp_model = DDP(ddp_model, device_ids=[local_rank] if use_cuda else None, find_unused_parameters=True)
        core_model = ddp_model.module

    def rebuild_optimizer(lr: float):
        params = list(parameters_of_trainable(core_model))
        if len(params) == 0:
            raise RuntimeError("Nenhum parâmetro treinável encontrado ao criar o otimizador.")
        # TensorFlow usa epsilon padrão 1e-7; alinhamos para reduzir diferenças numéricas.
        return Adam(params, lr=lr, eps=1e-7), params

    # congela backbone se necessário e cria otimizador da fase atual
    if freeze_epochs > 0:
        set_backbone_trainable(core_model, False)
        current_stage = "freeze"
        opt, trainable_params = rebuild_optimizer(LR)
    else:
        set_backbone_trainable(core_model, True)
        current_stage = "finetune"
        opt, trainable_params = rebuild_optimizer(fine_tune_lr)
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
    use_amp = use_cuda
    scaler = GradScaler(enabled=use_amp)
    if rank == 0:
        print(f"[AMP] Mixed Precision habilitado: {use_amp}")
    
    # LR Scheduler com Warmup + Cosine Annealing
    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 3)))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    
    def create_scheduler(optimizer, base_lr, total_epochs, warmup_ep):
        """Cria scheduler com warmup linear + cosine annealing."""
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
        print(f"[Scheduler] Warmup epochs={warmup_epochs}, min_lr={min_lr:.2e}, Cosine Annealing")

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{MODEL_NAME}-{args.exec}.csv"
    csv_writer = None
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
    if rank == 0:
        csv_writer = pd.DataFrame(columns=csv_fields)

    def run_epoch(
        loader: DataLoader,
        train: bool,
        epoch_idx: Optional[int] = None,
    ) -> Tuple[float, Optional[float], float, float, float, float, Optional[float], Optional[float]]:
        if loader is None:
            return 0.0, float("nan"), float("nan"), float("nan"), 0.0, 0.0, None, None
        if train:
            ddp_model.train()
        else:
            ddp_model.eval()

        track_memory = use_cuda and (device.type == "cuda")
        if track_memory:
            torch.cuda.reset_peak_memory_stats(device)

        running_loss = 0.0
        n_samples = 0
        all_probs = []
        all_labels = []
        t_start = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            
            # GPU Augmentation apenas no treino
            if train:
                xb = gpu_augment(xb)

            # Modelo Keras (backend torch) espera NHWC; garante conversão sempre
            if xb.dim() == 4 and xb.shape[1] in (1, 3):
                xb = xb.permute(0, 2, 3, 1).contiguous()
            xb = xb.contiguous(memory_format=torch.channels_last)

            if train:
                # Aplica label smoothing nos targets de treino
                yb_smooth = smooth_labels(yb, label_smoothing)
                opt.zero_grad(set_to_none=True)
                # AMP: autocast para forward pass
                with autocast(enabled=use_amp, dtype=torch.float16):
                    logits = ddp_model(xb)
                    loss = loss_fn(logits.squeeze(1), yb_smooth)
                if not torch.isfinite(loss):
                    if rank == 0:
                        where = f" epoch={epoch_idx}" if epoch_idx is not None else ""
                        print(f"[WARN] loss não finita{where}; batch ignorado.")
                    continue
                # AMP: scale loss e backward
                scaler.scale(loss).backward()
                if clip_norm > 0.0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad(), autocast(enabled=use_amp, dtype=torch.float16):
                    logits = ddp_model(xb)
                    loss = loss_fn(logits.squeeze(1), yb)

            running_loss += loss.item() * xb.shape[0]
            n_samples += xb.shape[0]
            with torch.no_grad():
                probs = torch.sigmoid(logits).squeeze(1)
                all_probs.append(probs.detach())
                all_labels.append(yb.detach())

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
            auc_val = roc_auc_score(labels_all.numpy(), probs_all.numpy())
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
        # throughput global: soma samples de todos os ranks
        global_samples = n_samples
        if dist.is_initialized():
            n_samples_tensor = torch.tensor([n_samples], device=device, dtype=torch.float64)
            dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
            global_samples = n_samples_tensor.item()
        throughput = global_samples / elapsed if elapsed > 0 else 0.0
        return epoch_loss, auc_val, sens_val, spec_val, throughput, elapsed, peak_mem_alloc, peak_mem_reserved

    best_val_auc = -1.0
    best_state_dict = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        if current_stage == "freeze" and epoch >= freeze_epochs:
            set_backbone_trainable(core_model, True)
            opt, trainable_params = rebuild_optimizer(fine_tune_lr)
            # Recria scheduler para fase de fine-tuning
            scheduler = create_scheduler(opt, fine_tune_lr, EPOCHS - epoch, warmup_ep=1)
            current_stage = "finetune"
            if rank == 0:
                print(f"[Stage] Liberando backbone no epoch {epoch} | lr={fine_tune_lr:.2e}")

        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        train_loss, train_auc, train_sens, train_spec, train_thpt, train_elapsed, train_mem_alloc, train_mem_reserved = run_epoch(
            train_loader, train=True, epoch_idx=epoch
        )
        val_loss, val_auc, val_sens, val_spec, val_thpt, val_elapsed, val_mem_alloc, val_mem_reserved = run_epoch(
            valid_loader, train=False, epoch_idx=epoch
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
            # guarda melhor estado em memória para evitar escrita em disco
            to_save = unwrap_compiled_module(core_model)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in to_save.state_dict().items()}

    if rank == 0:
        eval_start = time.time()
        eval_loader = DataLoader(
            valid_ds,
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
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device, non_blocking=True).float()
                if xb.dim() == 4 and xb.shape[1] in (1, 3):
                    xb = xb.permute(0, 2, 3, 1).contiguous()
                xb = xb.contiguous(memory_format=torch.channels_last)
                logits = model_for_eval(xb)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                y_score.append(probs)
                y_true.append(yb.numpy())
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        sens_final, spec_final = compute_sens_spec(torch.from_numpy(y_true), torch.from_numpy(y_score))
        eval_elapsed = time.time() - eval_start
        val_throughput = (len(valid_ds) / eval_elapsed) if eval_elapsed > 0 else 0.0

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
        print(f"Valid AUC (final): {auc_val:.4f}")
        print(f"{args.dataset},{args.exec},{auc_val:.6f},{final_elapsed}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

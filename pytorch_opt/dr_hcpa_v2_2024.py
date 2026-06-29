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
import io
import inspect
import os
import random
import struct
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
from torch.optim import AdamW, Adam, SGD, RMSprop, Adadelta
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import timm
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from PIL import Image, ImageEnhance, ImageOps
from tfrecord import example_pb2
from lib.dali_pipeline import build_dali_pipeline, create_dali_iterator
from lib.runtime_profiling import RuntimeProfiler

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SELF_DIR = Path(__file__).resolve().parent  # gpu_energy.py vive ao lado do script (robusto ao mount do container)
for _p in (str(_SELF_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from gpu_energy import GpuTelemetry, EnergyScope

_GPU_TELE = None


def _gpu_tele() -> GpuTelemetry:
    """Singleton lazy de telemetria NVML (energia/potência/memória real)."""
    global _GPU_TELE
    if _GPU_TELE is None:
        _GPU_TELE = GpuTelemetry()
    return _GPU_TELE


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
    "train_gpu_mem_peak_mb",
    "train_energy_j",
    "train_avg_power_w",
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
    "val_gpu_mem_peak_mb",
    "val_energy_j",
    "val_avg_power_w",
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
    "test_gpu_mem_peak_mb",
    "test_energy_j",
    "test_avg_power_w",
    "lr",
    "total_train_time_s",
]


# ---------------------------
# Args
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento PyTorch distribuído com TFRecord + DDP")
    parser.add_argument("--tfrec_dir", type=str, default="/home/users/bmmorales/projects/hcpa/data/all-tfrec", help="Diretório com TFRecords")
    parser.add_argument("--dataset", type=str, default="all", help="Nome lógico do dataset")
    parser.add_argument("--train_csv", type=str, default="", help="CSV de treino com image_path e binary_label")
    parser.add_argument("--val_csv", type=str, default="", help="CSV de validação com image_path e binary_label")
    parser.add_argument("--csv_image_col", type=str, default="image_path", help="Coluna do caminho da imagem no CSV")
    parser.add_argument("--csv_label_col", type=str, default="binary_label", help="Coluna do rótulo no CSV")
    parser.add_argument("--train_split", type=str, default="train", help="Prefixo lógico do split de treino")
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        help="Prefixo lógico do split de validação (fallback: val -> valid -> test)",
    )
    parser.add_argument("--results", type=str, default="./results/all", help="Diretório de saída dos resultados")
    parser.add_argument("--exec", type=int, default=0, help="ID da execução")
    parser.add_argument("--img_sizes", type=int, default=299, help="Tamanho das imagens (quadradas)")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch por GPU (batch global = batch_size * GPUs)")
    parser.add_argument("--epochs", type=int, default=200, help="Épocas totais")
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate (fase inicial ou única)")
    parser.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=["adamw", "adam", "sgd_mom", "rmsprop", "adadelta"],
        help="Otimizador a usar (padrão: adamw)",
    )
    parser.add_argument(
        "--adamw_impl",
        type=str,
        default="auto",
        choices=["auto", "default", "foreach", "fused"],
        help="Implementação do AdamW: auto prefere fused em CUDA, depois foreach",
    )
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
        "--loss_finite_check_interval",
        type=int,
        default=0,
        help="Checa loss finita a cada N steps no modo sem AMP (0 desativa para evitar sync por batch)",
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
    parser.add_argument("--enable_dali", dest="use_dali", action="store_true", help="Habilita input pipeline com DALI")
    parser.add_argument("--disable_dali", dest="use_dali", action="store_false", help="Usa DataLoader PyTorch sem DALI")
    parser.set_defaults(use_dali=False)
    parser.add_argument("--enable_amp", dest="use_amp", action="store_true", help="Habilita AMP/FP16")
    parser.add_argument("--disable_amp", dest="use_amp", action="store_false", help="Desabilita AMP/FP16")
    parser.set_defaults(use_amp=True)
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
    parser.set_defaults(use_cosine_scheduler=True)
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

ADAMW_SUPPORTS_FOREACH = "foreach" in inspect.signature(AdamW).parameters
ADAMW_SUPPORTS_FUSED = "fused" in inspect.signature(AdamW).parameters

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


def resolve_split_files(
    base_dir: str,
    preferred_prefix: str,
    fallback_prefixes: Optional[list[str]] = None,
) -> tuple[list[str], Optional[str]]:
    prefixes: list[str] = []
    for prefix in [preferred_prefix] + list(fallback_prefixes or []):
        prefix = str(prefix).strip()
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)

    for prefix in prefixes:
        files = list_split_files(base_dir, prefix)
        if files:
            return files, prefix
    return [], None


def default_validation_fallbacks(preferred_prefix: str) -> list[str]:
    ordered = ["val", "valid", "test"]
    return [prefix for prefix in ordered if prefix != preferred_prefix]


def infer_index_file(tfrec_path: str) -> str:
    p = Path(tfrec_path)
    candidates = [p.with_suffix(".tfrec.idx"), p.with_suffix(".idx"), Path(str(p) + ".idx")]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Índice não encontrado para {tfrec_path} (esperado .tfrec.idx ou .idx)")


class RetinaTFRecordDataset(Dataset):
    """Fallback sem DALI que lê TFRecords diretamente via DataLoader."""

    def __init__(
        self,
        tfrecord_paths: list[str],
        index_paths: list[str],
        *,
        img_size: int,
        normalize_mode: str,
        augment: bool,
        fundus_crop_ratio: float,
        model_name: str,
    ):
        if len(tfrecord_paths) != len(index_paths):
            raise ValueError("Listas de TFRecords e índices possuem tamanhos distintos.")

        self.img_size = int(img_size)
        self.normalize_mode = normalize_mode
        self.augment = bool(augment)
        self.fundus_crop_ratio = fundus_crop_ratio if 0.0 < float(fundus_crop_ratio) <= 1.0 else 1.0
        self.norm_mean, self.norm_std = get_normalization_for_backbone(model_name)
        self._entries: list[tuple[Path, int]] = []
        self._file_handles: dict[str, object] = {}

        for tf_path, idx_path in zip(tfrecord_paths, index_paths):
            tf_path = Path(tf_path)
            idx_path = Path(idx_path)
            if not tf_path.exists():
                raise FileNotFoundError(f"TFRecord ausente: {tf_path}")
            if not idx_path.exists():
                raise FileNotFoundError(f"Arquivo .idx ausente: {idx_path}")
            offsets = np.loadtxt(str(idx_path), dtype=np.int64, usecols=0, ndmin=1)
            for offset in np.atleast_1d(offsets).astype(np.int64).tolist():
                self._entries.append((tf_path, int(offset)))

        if not self._entries:
            raise RuntimeError("Nenhuma entrada encontrada no conjunto de TFRecords/idx.")

    def __len__(self):
        return len(self._entries)

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

    def _open(self, path: Path):
        key = str(path)
        fh = self._file_handles.get(key)
        if fh is None:
            fh = open(path, "rb")
            self._file_handles[key] = fh
        return fh

    def _read_record_at(self, tf_path: Path, offset: int) -> bytes:
        fh = self._open(tf_path)
        fh.seek(offset)
        length_bytes = fh.read(8)
        if len(length_bytes) != 8:
            raise EOFError(f"Offset inválido: {offset}")
        length = struct.unpack("<Q", length_bytes)[0]
        _ = fh.read(4)
        data = fh.read(length)
        _ = fh.read(4)
        return data

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

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalize_mode == "preprocess":
            mean = np.asarray(self.norm_mean, dtype=np.float32)
            std = np.asarray(self.norm_std, dtype=np.float32)
            return (image - mean) / std
        if self.normalize_mode == "unit":
            return image / 255.0
        return image

    def __getitem__(self, idx: int):
        tf_path, offset = self._entries[idx]
        data = self._read_record_at(tf_path, offset)
        example = example_pb2.Example()
        example.ParseFromString(data)
        feats = example.features.feature
        if ("imagem" not in feats) or ("retinopatia" not in feats):
            raise KeyError("TFRecord não contém as chaves esperadas 'imagem'/'retinopatia'.")

        img_bytes = feats["imagem"].bytes_list.value[0]
        label = float(feats["retinopatia"].int64_list.value[0])

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = self._center_crop_fundus(img)
        if self.img_size > 0 and img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = self._apply_augmentations(img)

        x = np.asarray(img, dtype=np.float32)
        x = self._normalize(x)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


class RetinaCSVDataset(Dataset):
    """Dataset JPEG/PNG guiado por CSV para compartilhar o mesmo protocolo entre pipelines."""

    def __init__(
        self,
        csv_path: str,
        *,
        image_col: str,
        label_col: str,
        img_size: int,
        normalize_mode: str,
        augment: bool,
        fundus_crop_ratio: float,
        model_name: str,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV ausente: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_columns = {image_col, label_col}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"CSV {self.csv_path} sem colunas obrigatórias: {sorted(missing_columns)}")

        self.image_paths = [Path(path) for path in df[image_col].astype(str).tolist()]
        self.labels = [float(value) for value in df[label_col].tolist()]
        missing_files = [str(path) for path in self.image_paths if not path.exists()]
        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise FileNotFoundError(f"{len(missing_files)} imagens ausentes. Exemplo: {preview}")

        self.img_size = int(img_size)
        self.normalize_mode = str(normalize_mode)
        self.augment = bool(augment)
        self.fundus_crop_ratio = fundus_crop_ratio if 0.0 < float(fundus_crop_ratio) <= 1.0 else 1.0
        self.norm_mean, self.norm_std = get_normalization_for_backbone(model_name)

    def __len__(self):
        return len(self.image_paths)

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

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalize_mode == "preprocess":
            mean = np.asarray(self.norm_mean, dtype=np.float32)
            std = np.asarray(self.norm_std, dtype=np.float32)
            return (image - mean) / std
        if self.normalize_mode == "unit":
            return image / 255.0
        return image

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self._center_crop_fundus(img)
        if self.img_size > 0 and img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = self._apply_augmentations(img)

        x = np.asarray(img, dtype=np.float32)
        x = self._normalize(x)
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


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


def build_torch_loader(
    tfrec_files: list[str],
    *,
    batch_size: int,
    image_size: int,
    world_size: int,
    rank: int,
    normalize_mode: str,
    augment: bool,
    fundus_crop_ratio: float,
    model_name: str,
    use_cuda: bool,
    shuffle: bool,
):
    idx_files = [infer_index_file(f) for f in tfrec_files]
    dataset = RetinaTFRecordDataset(
        tfrecord_paths=tfrec_files,
        index_paths=idx_files,
        img_size=image_size,
        normalize_mode=normalize_mode,
        augment=augment,
        fundus_crop_ratio=fundus_crop_ratio,
        model_name=model_name,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if world_size > 1 else None
    num_workers = min(8, os.cpu_count() or 4)
    if num_workers < 1:
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    return loader, sampler, dataset, num_workers


def build_csv_loader(
    csv_path: str,
    *,
    image_col: str,
    label_col: str,
    batch_size: int,
    image_size: int,
    world_size: int,
    rank: int,
    normalize_mode: str,
    augment: bool,
    fundus_crop_ratio: float,
    model_name: str,
    use_cuda: bool,
    shuffle: bool,
):
    dataset = RetinaCSVDataset(
        csv_path=csv_path,
        image_col=image_col,
        label_col=label_col,
        img_size=image_size,
        normalize_mode=normalize_mode,
        augment=augment,
        fundus_crop_ratio=fundus_crop_ratio,
        model_name=model_name,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if world_size > 1 else None
    num_workers = min(8, os.cpu_count() or 4)
    if num_workers < 1:
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    return loader, sampler, dataset, num_workers


# ---------------------------
# Métricas auxiliares
# ---------------------------
SPECIFICITY_TARGET_SENSITIVITY = 0.95


def specificity_at_sensitivity(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    target_sensitivity: float = SPECIFICITY_TARGET_SENSITIVITY,
) -> float:
    y_true = y_true.reshape(-1).to(dtype=torch.float32)
    y_prob = y_prob.reshape(-1).to(device=y_true.device, dtype=torch.float32)
    finite = torch.isfinite(y_true) & torch.isfinite(y_prob)
    y_true = y_true[finite]
    y_prob = y_prob[finite].clamp(1e-7, 1.0 - 1e-7)
    if y_true.numel() == 0 or y_prob.numel() == 0:
        return float("nan")

    positives = y_true > 0.5
    n_pos = int(positives.sum().item())
    n_neg = int((~positives).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    target = float(target_sensitivity)
    if target <= 0.0:
        return 1.0
    if target > 1.0:
        return float("nan")

    order = torch.argsort(y_prob, descending=True, stable=True)
    sorted_scores = y_prob[order]
    sorted_true = positives[order]
    group_end = torch.ones(sorted_scores.numel(), device=sorted_scores.device, dtype=torch.bool)
    if sorted_scores.numel() > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_idxs = torch.nonzero(group_end, as_tuple=False).flatten()
    tp = torch.cumsum(sorted_true.to(torch.float64), dim=0)[threshold_idxs]
    fp = torch.cumsum((~sorted_true).to(torch.float64), dim=0)[threshold_idxs]
    tpr = torch.cat([torch.zeros(1, device=y_prob.device, dtype=torch.float64), tp / float(n_pos)])
    fpr = torch.cat([torch.zeros(1, device=y_prob.device, dtype=torch.float64), fp / float(n_neg)])

    target_tensor = torch.tensor(target, device=y_prob.device, dtype=torch.float64)
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


def compute_binary_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5):
    y_true = y_true.float()
    y_pred = (y_prob >= threshold).float()
    tp = torch.sum(y_true * y_pred)
    fn = torch.sum(y_true * (1.0 - y_pred))
    tn = torch.sum((1.0 - y_true) * (1.0 - y_pred))
    fp = torch.sum((1.0 - y_true) * y_pred)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = specificity_at_sensitivity(y_true, y_prob)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-7)
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "sensitivity": recall.item(),
        "specificity": specificity,
        "specificity_at_sens95": specificity,
        "f1": f1.item(),
    }


def build_terminal_final_summary(final_metrics, total_train_time_s, throughput_img_s=None, avg_gpu_memory_mb=None):
    def _fmt(value, precision):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(value):
            return None
        return f"{value:.{precision}f}"

    auc_str = _fmt(final_metrics.get("auc"), 4)
    precision_str = _fmt(final_metrics.get("precision"), 4)
    recall_str = _fmt(final_metrics.get("recall"), 4)
    sensitivity_str = _fmt(final_metrics.get("sensitivity"), 4)
    specificity_str = _fmt(final_metrics.get("specificity"), 4)
    f1_str = _fmt(final_metrics.get("f1"), 4)
    throughput_str = _fmt(throughput_img_s, 1)
    total_time_str = _fmt(total_train_time_s, 1)
    avg_mem_str = _fmt(avg_gpu_memory_mb, 1)

    lines = [
        "=" * 70,
        "FINAL SUMMARY",
        "=" * 70,
        f"AUC: {auc_str or 'n/a'}",
        f"Precision: {precision_str or 'n/a'}",
        f"Recall: {recall_str or 'n/a'}",
        f"Sensitivity: {sensitivity_str or 'n/a'}",
        f"Specificity: {specificity_str or 'n/a'}",
        f"F1: {f1_str or 'n/a'}",
        f"Throughput: {throughput_str} img/s" if throughput_str is not None else "Throughput: n/a",
        f"Total train time: {total_time_str}s" if total_time_str is not None else "Total train time: n/a",
        f"Avg GPU memory: {avg_mem_str} MB" if avg_mem_str is not None else "Avg GPU memory: n/a",
    ]
    return "\n".join(lines)


def build_terminal_epoch_summary(epoch, total_epochs, train_metrics, val_metrics, throughput_img_s=None, lr=None):
    def _fmt(value, precision):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(value):
            return None
        return f"{value:.{precision}f}"

    def _fmt_lr(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(value):
            return None
        return f"{value:.2e}"

    lines = [
        "=" * 70,
        f"Epoch {epoch + 1}/{total_epochs}",
        "=" * 70,
        (
            f"Train Loss: {_fmt(train_metrics.get('loss'), 4) or 'n/a'} | "
            f"AUC: {_fmt(train_metrics.get('auc'), 4) or 'n/a'} | "
            f"Precision: {_fmt(train_metrics.get('precision'), 4) or 'n/a'} | "
            f"Recall: {_fmt(train_metrics.get('recall'), 4) or 'n/a'} | "
            f"Spec: {_fmt(train_metrics.get('specificity'), 4) or 'n/a'} | "
            f"F1: {_fmt(train_metrics.get('f1'), 4) or 'n/a'}"
        ),
        (
            f"Val   Loss: {_fmt(val_metrics.get('loss'), 4) or 'n/a'} | "
            f"AUC: {_fmt(val_metrics.get('auc'), 4) or 'n/a'} | "
            f"Precision: {_fmt(val_metrics.get('precision'), 4) or 'n/a'} | "
            f"Recall: {_fmt(val_metrics.get('recall'), 4) or 'n/a'} | "
            f"Spec: {_fmt(val_metrics.get('specificity'), 4) or 'n/a'} | "
            f"F1: {_fmt(val_metrics.get('f1'), 4) or 'n/a'}"
        ),
        (
            f"Throughput: {_fmt(throughput_img_s, 1)} img/s"
            if _fmt(throughput_img_s, 1) is not None
            else "Throughput: n/a"
        ),
        f"LR: {_fmt_lr(lr)}" if _fmt_lr(lr) is not None else "LR: n/a",
    ]
    return "\n".join(lines)


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
def create_backbone(model_name: str, img_size: int, features_only: bool = False):
    """
    Cria backbone puro PyTorch usando timm com 1 logit de saída.

    Args:
        model_name: Nome do modelo (ex: "inception_v3")
        img_size: Tamanho da imagem
        features_only: Se True, retorna mapa de features [B, C, H, W] sem pooling.
                      Se False, retorna logit [B, 1] com global_pool="avg"
    """
    if features_only:
        # Para hybrids: retorna features espaciais sem pooling para token processing
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1,
            in_chans=3,
            features_only=True,  # Retorna lista de features em múltiplas escalas
        )
        return model
    else:
        # Para pytorch_opt: retorna logit direto com pooling
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


def resolve_adamw_impl(requested_impl: str, *, use_cuda: bool) -> list[str]:
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

    csv_mode = bool(args.train_csv.strip() or args.val_csv.strip())
    if csv_mode and not (args.train_csv.strip() and args.val_csv.strip()):
        raise SystemExit("Em modo CSV, informe --train_csv e --val_csv.")

    train_split_name = None
    valid_split_name = None
    test_split_name = None
    train_files: list[str] = []
    valid_files: list[str] = []
    test_files: list[str] = []
    if not csv_mode:
        train_files, train_split_name = resolve_split_files(args.tfrec_dir, args.train_split)
        valid_files, valid_split_name = resolve_split_files(
            args.tfrec_dir,
            args.val_split,
            default_validation_fallbacks(args.val_split),
        )
        test_files, test_split_name = resolve_split_files(
            args.tfrec_dir,
            "test",
            [valid_split_name or args.val_split],
        )
        if not test_files:
            test_files = valid_files
            test_split_name = valid_split_name
        if not train_files or not valid_files or not test_files:
            raise SystemExit(
                f"É necessário ter TFRecords de treino/validação em {args.tfrec_dir} "
                f"(train={args.train_split}, val={args.val_split})."
            )

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    runtime_profiler = RuntimeProfiler(results_dir, rank=rank, project_name="pytorch_opt")
    csv_path = results_dir / f"{MODEL_NAME}-{args.exec}.csv"
    csv_fields = COMMON_CSV_FIELDS
    csv_writer = None
    if rank == 0:
        csv_writer = pd.DataFrame(columns=csv_fields)

    use_dali = bool(getattr(args, "use_dali", True)) and not csv_mode
    use_cosine_scheduler = bool(getattr(args, "use_cosine_scheduler", True))
    amp_requested = bool(getattr(args, "use_amp", True))
    use_amp = use_cuda and amp_requested
    loss_finite_check_interval = max(0, int(getattr(args, "loss_finite_check_interval", 0)))
    train_sampler = None
    valid_sampler = None
    valid_dataset = None
    valid_num_workers = 0
    test_loader = None
    test_dataset = None
    test_num_workers = 0
    optimizer_impl_label = "default"

    dali_seed = args.seed + rank
    if rank == 0:
        print(
            f"[Ablation] dali={use_dali} amp={amp_requested} cosine={use_cosine_scheduler} "
            f"global_batch={GLOBAL_BATCH}"
        )
        if csv_mode:
            print(f"[CSV] train={args.train_csv} val={args.val_csv} label_col={args.csv_label_col}")
        else:
            print(f"[Splits] train={train_split_name} val={valid_split_name} test={test_split_name}")
        if use_dali:
            norm_mean, norm_std = get_normalization_for_backbone(MODEL_NAME)
            print(f"[DALI] Ativo: decode/resize/augment na GPU | shuffle=train | layout=NCHW")
            print(f"[DALI] Normalização para {MODEL_NAME}: mean={norm_mean}, std={norm_std}")
        else:
            print(f"[DALI] Desativado: usando DataLoader PyTorch para {MODEL_NAME}.")

    if csv_mode:
        train_loader, train_sampler, _, _ = build_csv_loader(
            args.train_csv,
            image_col=args.csv_image_col,
            label_col=args.csv_label_col,
            batch_size=LOCAL_BS,
            image_size=IMG,
            world_size=world_size,
            rank=rank,
            normalize_mode=args.normalize,
            augment=args.augment,
            fundus_crop_ratio=FUNDUS_CROP_RATIO,
            model_name=MODEL_NAME,
            use_cuda=use_cuda,
            shuffle=True,
        )
        valid_loader, valid_sampler, valid_dataset, valid_num_workers = build_csv_loader(
            args.val_csv,
            image_col=args.csv_image_col,
            label_col=args.csv_label_col,
            batch_size=LOCAL_BS,
            image_size=IMG,
            world_size=world_size,
            rank=rank,
            normalize_mode=args.normalize,
            augment=False,
            fundus_crop_ratio=FUNDUS_CROP_RATIO,
            model_name=MODEL_NAME,
            use_cuda=use_cuda,
            shuffle=False,
        )
        test_loader = valid_loader
        test_dataset = valid_dataset
        test_num_workers = valid_num_workers
        if rank == 0:
            print(f"[DataLoader] workers(valid)={valid_num_workers}")
    elif use_dali:
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
        test_loader = build_dali_loader(
            test_files,
            batch_size=LOCAL_BS,
            image_size=IMG,
            device_id=local_rank if use_cuda else 0,
            shard_id=rank,
            num_shards=world_size,
            augment=False,
            seed=dali_seed + 1999,
            fundus_crop_ratio=FUNDUS_CROP_RATIO,
            random_shuffle=False,
            model_name=MODEL_NAME,
        )
    else:
        train_loader, train_sampler, _, _ = build_torch_loader(
            train_files,
            batch_size=LOCAL_BS,
            image_size=IMG,
            world_size=world_size,
            rank=rank,
            normalize_mode=args.normalize,
            augment=args.augment,
            fundus_crop_ratio=FUNDUS_CROP_RATIO,
            model_name=MODEL_NAME,
            use_cuda=use_cuda,
            shuffle=True,
        )
        valid_loader, valid_sampler, valid_dataset, valid_num_workers = build_torch_loader(
            valid_files,
            batch_size=LOCAL_BS,
            image_size=IMG,
            world_size=world_size,
            rank=rank,
            normalize_mode=args.normalize,
            augment=False,
            fundus_crop_ratio=FUNDUS_CROP_RATIO,
            model_name=MODEL_NAME,
            use_cuda=use_cuda,
            shuffle=False,
        )
        test_loader, _, test_dataset, test_num_workers = build_torch_loader(
            test_files,
            batch_size=LOCAL_BS,
            image_size=IMG,
            world_size=world_size,
            rank=rank,
            normalize_mode=args.normalize,
            augment=False,
            fundus_crop_ratio=FUNDUS_CROP_RATIO,
            model_name=MODEL_NAME,
            use_cuda=use_cuda,
            shuffle=False,
        )
        if rank == 0:
            print(f"[DataLoader] workers(valid)={valid_num_workers}")

    model = create_backbone(MODEL_NAME, IMG)
    if torch.cuda.is_available() and os.environ.get("PT_DISABLE_COMPILE", "0") != "1":
        # NOTA: torch.compile() apenas devolve um wrapper preguiçoso aqui; a
        # compilação efetiva (TorchInductor -> kernels Triton) só acontece na 1a
        # passada. Falhas nesse momento (ex.: "libcuda.so cannot found!" quando o
        # container não tem o symlink libcuda.so -> libcuda.so.1) são suprimidas
        # pelo dynamo, que cai em EAGER silenciosamente. Para verificar se a
        # compilação realmente valeu, confira a AUSÊNCIA de "BackendCompilerFailed"
        # / "libcuda" no log. Use PT_DISABLE_COMPILE=1 para rodar baseline eager.
        try:
            torch._dynamo.config.suppress_errors = True  # não derruba o treino
            model = torch.compile(model, mode="reduce-overhead")
            if rank == 0:
                print("[torch.compile] solicitado (mode=reduce-overhead); "
                      "compilação efetiva ocorre na 1a passada.")
        except Exception as exc:
            if rank == 0:
                print(f"[WARN] torch.compile indisponível; seguindo em eager: {exc}")
    elif rank == 0 and os.environ.get("PT_DISABLE_COMPILE", "0") == "1":
        print("[torch.compile] desativado por PT_DISABLE_COMPILE=1 (eager mode).")

    ddp_model = model.to(device)
    core_model = ddp_model
    if world_size > 1:
        ddp_model = DDP(ddp_model, device_ids=[local_rank] if use_cuda else None, find_unused_parameters=True)
        core_model = ddp_model.module
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pos_weight_tensor = torch.tensor(POS_WEIGHT, device=device) if POS_WEIGHT > 0 else None
    if rank == 0:
        print(f"[AMP] requested={amp_requested} active={use_amp}")

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
        nonlocal optimizer_impl_label
        params = list(parameters_of_trainable(core_model))
        if len(params) == 0:
            raise RuntimeError("Nenhum parâmetro treinável encontrado ao criar o otimizador.")
        opt_name = str(getattr(args, "optimizer", "adamw")).lower()
        if opt_name in ("adamw", "adam_w"):
            last_error = None
            for impl_name in resolve_adamw_impl(args.adamw_impl, use_cuda=use_cuda):
                adamw_kwargs = dict(lr=lr, eps=1e-7, weight_decay=weight_decay)
                if impl_name == "foreach":
                    adamw_kwargs["foreach"] = True
                elif impl_name == "fused":
                    adamw_kwargs["fused"] = True
                try:
                    optimizer_impl_label = impl_name
                    return AdamW(params, **adamw_kwargs), params
                except Exception as exc:
                    last_error = exc
            optimizer_impl_label = "default"
            if last_error is not None and rank == 0:
                print(f"[WARN] AdamW impl='{args.adamw_impl}' falhou; usando default: {last_error}")
            return AdamW(params, lr=lr, eps=1e-7, weight_decay=weight_decay), params
        if opt_name == "adam":
            optimizer_impl_label = "default"
            return Adam(params, lr=lr, eps=1e-7, weight_decay=weight_decay), params
        if opt_name in ("sgd", "sgd_mom", "sgd-mom"):
            optimizer_impl_label = "default"
            return SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay), params
        if opt_name == "rmsprop":
            optimizer_impl_label = "default"
            return RMSprop(params, lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay), params
        if opt_name == "adadelta":
            optimizer_impl_label = "default"
            return Adadelta(params, lr=lr, rho=0.95, eps=1e-6, weight_decay=weight_decay), params
        optimizer_impl_label = "default"
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
        print(f"[Optimizer] name={args.optimizer} adamw_impl={optimizer_impl_label}")
    clip_norm = max(0.0, float(args.clip_grad_norm))
    
    # LR Scheduler com Warmup + Cosine Annealing
    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 3)))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    
    def create_scheduler(optimizer, base_lr, total_epochs, warmup_ep):
        """Cria scheduler com warmup linear + cosine annealing."""
        if not use_cosine_scheduler:
            return None
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
        if use_cosine_scheduler:
            print(f"[Scheduler] Warmup epochs={warmup_epochs}, min_lr={min_lr:.2e}, Cosine Annealing")
        else:
            print("[Scheduler] Desativado; learning rate constante.")
    def unpack_batch(batch):
        if use_dali:
            data = batch[0] if isinstance(batch, list) else batch
            xb = data["image"]
            yb = data["label"].squeeze(-1).float()
        else:
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
        return xb, yb

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
        ddp_model.train() if train else ddp_model.eval()

        track_memory = use_cuda and (device.type == "cuda")

        running_loss = torch.zeros((), device=device, dtype=torch.float64)
        n_samples = 0
        all_probs = []
        all_labels = []
        memory_samples_mb = []
        inference_time_s = 0.0
        inference_batches = 0
        inference_samples = 0
        t_start = time.time()
        loader_iter = iter(loader)
        step_idx = 0
        while True:
            try:
                batch = runtime_profiler.fetch_next(loader_iter, stage_name)
            except StopIteration:
                break

            with runtime_profiler.range(f"{stage_name}/batch"):
                with runtime_profiler.range(f"{stage_name}/batch_unpack"):
                    xb, yb_original = unpack_batch(batch)
                    yb = yb_original.clone()

                with runtime_profiler.range(f"{stage_name}/h2d"):
                    xb = xb.to(device, non_blocking=True)
                    yb_original = yb_original.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                if train:
                    with runtime_profiler.range(f"{stage_name}/augment"):
                        xb, yb = apply_mixup_cutmix(xb, yb, mixup_alpha, cutmix_alpha)

                if train:
                    with runtime_profiler.range(f"{stage_name}/optimizer_zero_grad"):
                        opt.zero_grad(set_to_none=True)
                    with runtime_profiler.range(f"{stage_name}/forward"):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            logits = ddp_model(xb).squeeze(1)
                            loss = compute_loss(logits, yb)
                    should_check_finite = (
                        (not use_amp)
                        and loss_finite_check_interval > 0
                        and ((step_idx + 1) % loss_finite_check_interval == 0)
                    )
                    if should_check_finite and not bool(torch.isfinite(loss.detach()).item()):
                        if rank == 0:
                            where = f" epoch={epoch_idx}" if epoch_idx is not None else ""
                            print(f"[WARN] loss não finita{where}; batch ignorado.")
                        runtime_profiler.step(stage_name)
                        step_idx += 1
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
                        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                            logits = ddp_model(xb).squeeze(1)
                            loss = compute_loss(logits, yb)
                    if track_memory:
                        torch.cuda.synchronize(device)
                    inference_time_s += time.perf_counter() - inference_start
                    inference_batches += 1
                    inference_samples += int(xb.shape[0])

                with runtime_profiler.range(f"{stage_name}/metrics"):
                    batch_size = xb.shape[0]
                    running_loss.add_(loss.detach().to(dtype=torch.float64), alpha=batch_size)
                    n_samples += batch_size
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        all_probs.append(probs.detach())
                        all_labels.append(yb_original.detach())

                if track_memory:
                    torch.cuda.synchronize(device)
                    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    _used_mb = _gpu_tele().memory_used_mb()
                    memory_samples_mb.append(
                        _used_mb if _used_mb is not None else max(allocated_mb, reserved_mb)
                    )

            runtime_profiler.step(stage_name)
            step_idx += 1

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
            auc_val = roc_auc_score(labels_all.cpu().numpy(), probs_all.cpu().numpy())
        except Exception:
            auc_val = float("nan")
        epoch_loss = (running_loss / max(n_samples, 1)).item()
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
        num_batches = step_idx
        throughput = global_samples / elapsed if elapsed > 0 else 0.0
        avg_batch_time_ms = (elapsed / max(num_batches, 1)) * 1000.0 if num_batches > 0 else 0.0
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
    best_epoch = 0
    best_state_dict = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        if current_stage == "freeze" and epoch >= FREEZE_EPOCHS:
            set_backbone_trainable(core_model, True)
            opt, trainable_params = rebuild_optimizer(FINE_TUNE_LR, FINE_TUNE_WEIGHT_DECAY)
            # Recria scheduler para fase de fine-tuning
            scheduler = create_scheduler(opt, FINE_TUNE_LR, EPOCHS - epoch, warmup_ep=warmup_epochs)
            current_stage = "finetune"
            if rank == 0:
                print(
                    f"[Stage] Liberando backbone no epoch {epoch} | lr={FINE_TUNE_LR:.2e} "
                    f"| adamw_impl={optimizer_impl_label}"
                )

        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        mixup_alpha_now = mixup_alpha_for_epoch(epoch)
        cutmix_alpha_now = CUTMIX_ALPHA

        _train_scope = EnergyScope(_gpu_tele())
        _train_scope.start()
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
            train_loader,
            train=True,
            epoch_idx=epoch,
            mixup_alpha=mixup_alpha_now,
            cutmix_alpha=cutmix_alpha_now,
        )
        _train_scope.stop()
        _val_scope = EnergyScope(_gpu_tele())
        _val_scope.start()
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
            valid_loader, train=False, epoch_idx=epoch, mixup_alpha=0.0, cutmix_alpha=0.0
        )
        _val_scope.stop()

        if scheduler is not None:
            scheduler.step()

        if rank == 0:
            print(
                f"[{current_stage} E{epoch}/{EPOCHS}] "
                f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
                f"train_precision={train_epoch_metrics['precision']:.4f} "
                f"train_sens={train_epoch_metrics['sensitivity']:.4f} "
                f"train_spec={train_epoch_metrics['specificity']:.4f} "
                f"train_f1={train_epoch_metrics['f1']:.4f} | "
                f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
                f"val_precision={val_epoch_metrics['precision']:.4f} "
                f"val_sens={val_epoch_metrics['sensitivity']:.4f} "
                f"val_spec={val_epoch_metrics['specificity']:.4f} "
                f"val_f1={val_epoch_metrics['f1']:.4f} | "
                f"thr={train_thpt:.1f} img/s lr={opt.param_groups[0]['lr']:.2e}"
            )
            if csv_writer is not None:
                new_row = {
                    "epoch": epoch,
                    "stage": current_stage,
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
                    "train_gpu_mem_avg_mb": (
                        _train_scope.avg_mem_mb if _train_scope.avg_mem_mb is not None else train_mem_avg
                    ),
                    "train_gpu_mem_peak_mb": _train_scope.peak_mem_mb,
                    "train_energy_j": _train_scope.energy_j,
                    "train_avg_power_w": _train_scope.avg_power_w,
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
                    "val_gpu_mem_avg_mb": (
                        _val_scope.avg_mem_mb if _val_scope.avg_mem_mb is not None else val_mem_avg
                    ),
                    "val_gpu_mem_peak_mb": _val_scope.peak_mem_mb,
                    "val_energy_j": _val_scope.energy_j,
                    "val_avg_power_w": _val_scope.avg_power_w,
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
                    "test_gpu_mem_peak_mb": np.nan,
                    "test_energy_j": np.nan,
                    "test_avg_power_w": np.nan,
                    "lr": opt.param_groups[0]["lr"],
                    "total_train_time_s": None,
                }
                csv_writer.loc[len(csv_writer)] = new_row
                csv_writer.to_csv(csv_path, index=False)

        if (rank == 0) and (val_auc > best_val_auc):
            best_val_auc = val_auc
            best_epoch = epoch
            to_save = unwrap_compiled_module(core_model)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in to_save.state_dict().items()}

    if rank == 0:
        eval_start = time.time()
        eval_num_batches = 0
        eval_loader = test_loader if test_loader is not None else valid_loader
        eval_memory_samples_mb = []
        eval_inference_time_s = 0.0
        eval_inference_samples = 0
        if use_dali:
            try:
                eval_loader.reset()
            except Exception:
                pass
        elif test_dataset is not None:
            eval_workers = max(1, test_num_workers // 2) if test_num_workers > 0 else 1
            eval_loader = DataLoader(
                test_dataset,
                batch_size=LOCAL_BS,
                shuffle=False,
                num_workers=eval_workers,
                pin_memory=use_cuda,
                drop_last=False,
                persistent_workers=(eval_workers > 0),
                prefetch_factor=(2 if eval_workers > 0 else None),
            )
        model_for_eval = unwrap_compiled_module(core_model)
        if best_state_dict is not None:
            try:
                model_for_eval.load_state_dict(best_state_dict, strict=False)
                print(f"[BEST] Restaurado checkpoint em memória: epoch={best_epoch} val_auc={best_val_auc:.6f}")
            except Exception as exc:
                print(f"[WARN] falha ao aplicar melhor estado em memória: {exc}")
        model_for_eval.eval()

        _test_scope = EnergyScope(_gpu_tele())
        _test_scope.start()
        y_true = []
        y_score = []
        with torch.no_grad():
            eval_iter = iter(eval_loader)
            while True:
                try:
                    batch = runtime_profiler.fetch_next(eval_iter, "eval")
                except StopIteration:
                    break
                with runtime_profiler.range("eval/batch"):
                    with runtime_profiler.range("eval/batch_unpack"):
                        xb, yb = unpack_batch(batch)
                    with runtime_profiler.range("eval/h2d"):
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                    with runtime_profiler.range("eval/tta_prepare"):
                        views = [xb]
                        if TTA_VIEWS > 1:
                            for view_idx in range(1, TTA_VIEWS):
                                mode = view_idx % 4
                                if mode == 1:
                                    views.append(torch.flip(xb, dims=[3]))
                                elif mode == 2:
                                    views.append(torch.flip(xb, dims=[2]))
                                elif mode == 3:
                                    views.append(torch.rot90(xb, k=1, dims=(2, 3)))
                                else:
                                    views.append(xb)
                    if use_cuda and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with runtime_profiler.range("eval/forward"):
                        probs_acc = None
                        for view in views:
                            logits = model_for_eval(view).squeeze(1)
                            probs_view = torch.sigmoid(logits)
                            probs_acc = probs_view if probs_acc is None else probs_acc + probs_view
                    if use_cuda and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    eval_inference_time_s += time.perf_counter() - inference_start
                    eval_inference_samples += int(xb.shape[0])
                    with runtime_profiler.range("eval/metrics"):
                        probs = (probs_acc / float(len(views))).cpu().numpy()
                        y_score.append(probs)
                        y_true.append(yb.cpu().numpy())
                        eval_num_batches += 1
                    if use_cuda and device.type == "cuda":
                        allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                        _used_mb = _gpu_tele().memory_used_mb()
                        eval_memory_samples_mb.append(
                            _used_mb if _used_mb is not None else max(allocated_mb, reserved_mb)
                        )
                runtime_profiler.step("eval")
        _test_scope.stop()
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        final_metrics = compute_binary_metrics(torch.from_numpy(y_true), torch.from_numpy(y_score))
        eval_elapsed = time.time() - eval_start
        val_throughput = (len(y_true) / eval_elapsed) if eval_elapsed > 0 else 0.0
        val_avg_batch_time_ms = (eval_elapsed / max(eval_num_batches, 1)) * 1000.0 if eval_num_batches > 0 else 0.0
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
        eval_mem_avg = (
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
        avg_gpu_memory_mb = eval_mem_avg
        if csv_writer is not None:
            for field in ("train_gpu_mem_avg_mb", "val_gpu_mem_avg_mb", "test_gpu_mem_avg_mb"):
                if field in csv_writer:
                    series = pd.to_numeric(csv_writer[field], errors="coerce").dropna()
                    if not series.empty:
                        field_mean = float(series.mean())
                        avg_gpu_memory_mb = field_mean if avg_gpu_memory_mb is None else float(
                            np.mean([avg_gpu_memory_mb, field_mean])
                        )
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
                "train_gpu_mem_avg_mb": None,
                "train_gpu_mem_peak_mb": np.nan,
                "train_energy_j": np.nan,
                "train_avg_power_w": np.nan,
                "val_loss": np.nan,
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
                "val_gpu_mem_peak_mb": np.nan,
                "val_energy_j": np.nan,
                "val_avg_power_w": np.nan,
                "test_loss": np.nan,
                "test_auc": auc_val,
                "test_precision": final_metrics["precision"],
                "test_recall": final_metrics["recall"],
                "test_f1": final_metrics["f1"],
                "test_sens": final_metrics["sensitivity"],
                "test_spec": final_metrics["specificity"],
                "test_spec_at_sens95": final_metrics["specificity_at_sens95"],
                "test_throughput_img_s": val_throughput,
                "test_elapsed_s": eval_elapsed,
                "test_avg_batch_time_ms": val_avg_batch_time_ms,
                "test_inference_latency_ms_img": test_inference_latency_ms_img,
                "test_inference_latency_ms_batch": test_inference_latency_ms_batch,
                "test_gpu_mem_avg_mb": (
                    _test_scope.avg_mem_mb if _test_scope.avg_mem_mb is not None else eval_mem_avg
                ),
                "test_gpu_mem_peak_mb": _test_scope.peak_mem_mb,
                "test_energy_j": _test_scope.energy_j,
                "test_avg_power_w": _test_scope.avg_power_w,
                "lr": opt.param_groups[0]["lr"],
                "total_train_time_s": final_elapsed,
            }
            csv_writer.loc[len(csv_writer)] = final_row
            csv_writer.to_csv(csv_path, index=False)
        final_summary_metrics = {"auc": auc_val, **final_metrics}
        print(build_terminal_final_summary(final_summary_metrics, final_elapsed, val_throughput, avg_gpu_memory_mb))

    runtime_profiler.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Versão básica do treinamento TensorFlow distribuído.
- Sem fases de fine-tuning, mixup, DALI ou EMA.
- Apenas TFRecord -> tf.data -> modelo keras.applications com Mirrored/MultiWorker.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tensorflow import keras
from tensorflow.keras import applications


class Sensitivity(tf.keras.metrics.Metric):
    """Sensitivity (Recall/True Positive Rate) at threshold 0.5."""

    def __init__(self, name="sensitivity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


def _compute_f1_from_precision_recall(precision, recall):
    if precision is None or recall is None:
        return None
    try:
        precision = float(precision)
        recall = float(recall)
    except Exception:
        return None
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return (2.0 * precision * recall) / (denom + 1e-8)


SPECIFICITY_TARGET_SENSITIVITY = 0.95


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = SPECIFICITY_TARGET_SENSITIVITY,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = np.clip(y_score[finite], 1e-7, 1.0 - 1e-7)
    if y_true.size == 0 or y_score.size == 0:
        return float("nan")

    positives = y_true == 1
    n_pos = int(np.sum(positives))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    target = float(target_sensitivity)
    if target <= 0.0:
        return 1.0
    if target > 1.0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = positives[order]
    group_end = np.ones(sorted_scores.shape[0], dtype=bool)
    if sorted_scores.shape[0] > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_idxs = np.flatnonzero(group_end)
    tp = np.cumsum(sorted_true, dtype=np.float64)[threshold_idxs]
    fp = np.cumsum(~sorted_true, dtype=np.float64)[threshold_idxs]
    tpr = np.r_[0.0, tp / float(n_pos)]
    fpr = np.r_[0.0, fp / float(n_neg)]

    idxs = np.flatnonzero(tpr >= target)
    if idxs.size == 0:
        return 0.0
    idx = int(idxs[0])
    if idx == 0:
        fpr_at_target = fpr[0]
    else:
        tpr_lo, tpr_hi = tpr[idx - 1], tpr[idx]
        fpr_lo, fpr_hi = fpr[idx - 1], fpr[idx]
        if abs(tpr_hi - tpr_lo) <= 1e-12:
            fpr_at_target = min(fpr_lo, fpr_hi)
        else:
            fraction = (target - tpr_lo) / (tpr_hi - tpr_lo)
            fpr_at_target = fpr_lo + fraction * (fpr_hi - fpr_lo)
    return float(np.clip(1.0 - fpr_at_target, 0.0, 1.0))


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    """Compute threshold metrics and specificity at 95% sensitivity."""
    preds = (y_score >= threshold).astype(np.float32)
    tp = np.sum((preds == 1) & (y_true == 1))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = specificity_at_sensitivity(y_true, y_score)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "specificity_at_sens95": float(specificity),
        "f1": float(f1),
    }


def compute_sens_spec(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    return metrics["sensitivity"], metrics["specificity"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento simples em GPU usando TFRecord + Keras com distribuição"
    )
    parser.add_argument("--tfrec_dir", type=str, default="/home/users/bmmorales/projects/hcpa/data/all-tfrec", help="Diretório com TFRecords")
    parser.add_argument("--dataset", type=str, default="all", help="Nome lógico do dataset")
    parser.add_argument("--results", type=str, default="./results/all", help="Diretório para salvar resultados")
    parser.add_argument("--exec", type=int, default=0, help="ID de execução")
    parser.add_argument("--img_sizes", type=int, default=299, help="Tamanho das imagens (quadradas)")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch por réplica/GPU (batch global = batch_size * réplicas)")
    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas")
    parser.add_argument("--lrate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd_mom", "rmsprop", "adadelta"],
        help="Otimizador a usar (padrão: adamw)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay usado pelo AdamW.")
    parser.add_argument("--clipnorm", type=float, default=1.0, help="Gradient clipping por norma; <=0 desativa.")
    parser.add_argument("--num_thresholds", type=int, default=200, help="(Mantido para compatibilidade)")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose do Keras")
    parser.add_argument("--model", type=str, default="InceptionV3", help="Backbone keras.applications")
    parser.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Habilita augmentações simples (flip + jitter de cor)",
    )
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Desabilita augmentações")
    parser.set_defaults(augment=True)
    parser.add_argument(
        "--normalize",
        type=str,
        default="preprocess",
        choices=["preprocess", "raw255", "unit"],
        help="Normalização aplicada após o decode",
    )
    parser.add_argument("--cores", type=int, default=0, help="Força número de threads de CPU (<=0 mantém padrão)")
    parser.add_argument(
        "--log-gpu-mem",
        dest="log_gpu_mem",
        action="store_true",
        help="Registra memória de GPU (peak/current) a cada época no CSV.",
    )
    parser.add_argument(
        "--no-log-gpu-mem",
        dest="log_gpu_mem",
        action="store_false",
        help="Desabilita o registro de memória de GPU.",
    )
    parser.set_defaults(log_gpu_mem=True)
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
        "--early_stop_patience",
        type=int,
        default=0,
        help="Paciência em épocas para early stopping por validação; 0 desativa.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Ganho mínimo exigido na métrica monitorada pelo early stopping.",
    )
    parser.add_argument(
        "--early_stop_monitor",
        type=str,
        default="val_AUC",
        help="Métrica usada pelo early stopping.",
    )
    parser.add_argument(
        "--exact-val-every-epoch",
        dest="exact_val_every_epoch",
        action="store_true",
        help="Recalcula métricas exatas da validação ao fim de cada época.",
    )
    parser.add_argument(
        "--no-exact-val-every-epoch",
        dest="exact_val_every_epoch",
        action="store_false",
        help="Usa métricas de validação do Keras durante treino e mantém avaliação exata no final_test.",
    )
    parser.set_defaults(exact_val_every_epoch=True)
    return parser.parse_args()


_PREPROCESS_MAP = {
    "InceptionV3": applications.inception_v3.preprocess_input,
    "InceptionResNetV2": applications.inception_resnet_v2.preprocess_input,
    "Xception": applications.xception.preprocess_input,
    "VGG16": applications.vgg16.preprocess_input,
    "VGG19": applications.vgg19.preprocess_input,
    "ResNet50": applications.resnet.preprocess_input if hasattr(applications, "resnet") else applications.resnet50.preprocess_input,
    "ResNet50V2": applications.resnet_v2.preprocess_input,
    "ResNet101": applications.resnet.preprocess_input,
    "ResNet101V2": applications.resnet_v2.preprocess_input,
    "ResNet152": applications.resnet.preprocess_input,
    "ResNet152V2": applications.resnet_v2.preprocess_input,
    "MobileNet": applications.mobilenet.preprocess_input,
    "MobileNetV2": applications.mobilenet_v2.preprocess_input,
    "DenseNet121": applications.densenet.preprocess_input,
    "DenseNet169": applications.densenet.preprocess_input,
    "DenseNet201": applications.densenet.preprocess_input,
    "NASNetMobile": applications.nasnet.preprocess_input,
    "NASNetLarge": applications.nasnet.preprocess_input,
    "EfficientNetB0": applications.efficientnet.preprocess_input,
    "EfficientNetB1": applications.efficientnet.preprocess_input,
    "EfficientNetB2": applications.efficientnet.preprocess_input,
    "EfficientNetB3": applications.efficientnet.preprocess_input,
    "EfficientNetB4": applications.efficientnet.preprocess_input,
    "EfficientNetB5": applications.efficientnet.preprocess_input,
    "EfficientNetB6": applications.efficientnet.preprocess_input,
    "EfficientNetB7": applications.efficientnet.preprocess_input,
    "ConvNeXtTiny": applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    "ConvNeXtSmall": applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    "ConvNeXtBase": applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    "ConvNeXtLarge": applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    "ConvNeXtXLarge": applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
}


def get_preprocess_fn(model_name: str) -> Callable:
    return _PREPROCESS_MAP.get(model_name, None)


def configure_hardware(cores: int):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("[Hardware] Execução requer GPU, mas nenhuma foi detectada.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"[Hardware] GPUs detectadas: {len(gpus)}")
    if cores > 0:
        os.environ["OMP_NUM_THREADS"] = str(cores)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(cores)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, cores // 2))
        except Exception as exc:
            print(f"[Hardware] Aviso ao configurar threads: {exc}")


def choose_strategy():
    tf_config_raw = os.environ.get("TF_CONFIG")
    if tf_config_raw:
        try:
            tf_config = json.loads(tf_config_raw)
            workers = tf_config.get("cluster", {}).get("worker", []) or []
            if len(workers) > 1:
                strategy = tf.distribute.MultiWorkerMirroredStrategy()
                print("[Distribuição] MultiWorkerMirroredStrategy ativa via TF_CONFIG.")
                return strategy
            os.environ.pop("TF_CONFIG", None)
            print("[Distribuição] TF_CONFIG single-worker ignorado; usando estratégia local.")
        except Exception:
            print("[Distribuição] TF_CONFIG inválido, usando fallback.")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("[Distribuição] Nenhuma GPU detectada; execução requer GPU.")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[Distribuição] MirroredStrategy em {len(gpus)} GPU(s).")
        return strategy
    strategy = tf.distribute.get_strategy()
    print("[Distribuição] Estratégia padrão (single GPU/CPU).")
    return strategy


def _read_gpu_current_memory_mb():
    """Retorna a memória GPU corrente em MB. Prefere o uso REAL via NVML."""
    _used = _gpu_tele().memory_used_mb()
    if _used is not None:
        return _used
    try:
        logical_gpus = tf.config.list_logical_devices("GPU")
        currents = []
        for idx, _ in enumerate(logical_gpus):
            try:
                info = tf.config.experimental.get_memory_info(f"GPU:{idx}")
            except Exception:
                continue
            currents.append(info.get("current", 0))
        if currents:
            to_mb = 1024 * 1024
            return max(currents) / to_mb
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        used_vals = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                used_vals.append(float(line))
            except Exception:
                continue
        if used_vals:
            return max(used_vals)
    except Exception:
        pass

    return None


class GPUMemoryLogger(keras.callbacks.Callback):
    """Amostra consumo corrente de memória GPU e grava média por época."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled
        self.train_samples = []
        self.val_samples = []
        self.power_samples = []
        self._energy_start_j = None
        self._tele = _gpu_tele()

    def on_epoch_begin(self, epoch, logs=None):
        if self.enabled:
            self.train_samples = []
            self.val_samples = []
            self.power_samples = []
            self._energy_start_j = self._tele.energy_j()

    def _sample(self, samples):
        current_mb = _read_gpu_current_memory_mb()
        if current_mb is not None:
            samples.append(float(current_mb))
        _pw = self._tele.power_w()
        if _pw is not None:
            self.power_samples.append(_pw)

    def on_train_batch_end(self, batch, logs=None):
        if self.enabled:
            self._sample(self.train_samples)

    def on_test_batch_end(self, batch, logs=None):
        if self.enabled:
            self._sample(self.val_samples)

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return
        logs = logs or {}
        if self.train_samples:
            logs["train_gpu_mem_avg_mb"] = float(np.mean(self.train_samples))
        if self.val_samples:
            logs["val_gpu_mem_avg_mb"] = float(np.mean(self.val_samples))
        all_mem = self.train_samples + self.val_samples
        if all_mem:
            logs["train_gpu_mem_peak_mb"] = float(np.max(all_mem))
        if self.power_samples:
            logs["train_avg_power_w"] = float(np.mean(self.power_samples))
        _e_end = self._tele.energy_j()
        if self._energy_start_j is not None and _e_end is not None:
            logs["train_energy_j"] = max(0.0, _e_end - self._energy_start_j)


class ExactEvalMetricsLogger(keras.callbacks.Callback):
    """Recalcula métricas de validação a partir de scores completos."""

    def __init__(self, dataset, prefix="val", steps=None, enabled=True):
        super().__init__()
        self.dataset = dataset
        self.prefix = str(prefix)
        self.steps = None if steps is None or steps < 0 else int(steps)
        self.enabled = bool(enabled)

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return
        logs = logs if logs is not None else {}
        labels = []
        scores = []
        for batch_idx, batch in enumerate(self.dataset):
            if self.steps is not None and batch_idx >= self.steps:
                break
            batch_images, batch_labels = batch[0], batch[1]
            preds = self.model(batch_images, training=False)
            labels.append(np.asarray(batch_labels.numpy()).reshape(-1))
            scores.append(np.asarray(preds.numpy()).reshape(-1))
        if not labels:
            return

        y_true = np.concatenate(labels).astype(np.int32)
        y_score = np.concatenate(scores).astype(np.float64)
        try:
            auc_value = roc_auc_score(y_true, y_score)
        except ValueError:
            auc_value = float("nan")
        metrics = compute_binary_metrics(y_true, y_score)
        prefix = f"{self.prefix}_"
        logs[f"{prefix}AUC"] = auc_value
        logs[f"{prefix}auc"] = auc_value
        logs[f"{prefix}precision"] = metrics["precision"]
        logs[f"{prefix}sensitivity"] = metrics["sensitivity"]
        logs[f"{prefix}recall"] = metrics["recall"]
        logs[f"{prefix}specificity"] = metrics["specificity"]
        logs[f"{prefix}specificity_at_sens95"] = metrics["specificity_at_sens95"]
        logs[f"{prefix}f1"] = metrics["f1"]


class BestWeightsTracker(keras.callbacks.Callback):
    """Guarda em memoria os melhores pesos para evitar reload HDF5 no final."""

    def __init__(self, monitor: str = "val_AUC", mode: str = "max", verbose: int = 0):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_value = -math.inf if mode == "max" else math.inf
        self.best_epoch = None
        self.best_weights = None

    def _is_better(self, value: float) -> bool:
        if self.best_epoch is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        raw_value = logs.get(self.monitor)
        try:
            value = float(raw_value)
        except Exception:
            return
        if not np.isfinite(value) or not self._is_better(value):
            return
        self.best_value = value
        self.best_epoch = int(epoch)
        self.best_weights = [weights.copy() for weights in self.model.get_weights()]
        if self.verbose:
            print(
                f"[INFO] Melhor checkpoint em memoria atualizado: "
                f"epoch={self.best_epoch + 1}, {self.monitor}={self.best_value:.6f}"
            )

    def restore(self, model) -> bool:
        if self.best_weights is None:
            return False
        model.set_weights(self.best_weights)
        return True



def _dataset_cardinality(dataset):
    try:
        n = tf.data.experimental.cardinality(dataset).numpy()
        return int(n) if n >= 0 else None
    except Exception:
        return None


import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SELF_DIR = Path(__file__).resolve().parent  # gpu_energy.py vive ao lado do script (robusto ao mount do container)
for _p in (str(_SELF_DIR), str(_PROJECT_ROOT)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
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


class EpochCsvLogger(keras.callbacks.Callback):
    """Escreve métricas por época em CSV com o layout do pytorch_opt."""

    _fields = COMMON_CSV_FIELDS

    def __init__(self, csv_path, stage, train_steps, val_steps, global_batch_size, append=False):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.stage = stage
        self.train_steps = train_steps if (train_steps is None or train_steps >= 0) else None
        self.val_steps = val_steps if (val_steps is None or val_steps >= 0) else None
        self.global_batch_size = max(1, int(global_batch_size))
        self.append = bool(append)
        self._file = None
        self._writer = None
        self._train_elapsed = 0.0
        self._val_elapsed = 0.0
        self._epoch_start = None
        self._train_end_time = None

    def on_train_begin(self, logs=None):
        mode = "a" if self.append and self.csv_path.exists() else "w"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
        if self._file.tell() == 0:
            self._writer.writeheader()
            self._file.flush()

    # OTIMIZAÇÃO: Removidos batch hooks para reduzir overhead (~5-10% throughput)
    # Agora medimos tempo no nível de epoch, que é mais eficiente
    
    def on_epoch_begin(self, epoch, logs=None):
        self._train_elapsed = 0.0
        self._val_elapsed = 0.0
        self._epoch_start = time.time()
        self._train_end_time = None
    
    def on_test_begin(self, logs=None):
        # Marca o fim do treino e início da validação
        if self._epoch_start is not None:
            self._train_end_time = time.time()
            self._train_elapsed = self._train_end_time - self._epoch_start
    
    def on_test_end(self, logs=None):
        # Calcula tempo de validação
        if self._train_end_time is not None:
            self._val_elapsed = time.time() - self._train_end_time

    def _resolve_lr(self):
        try:
            lr = getattr(self.model.optimizer, "lr", None)
            if lr is None:
                return None
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            return None

    def _resolve_auc(self, logs, prefix=""):
        keys = [k for k in logs.keys() if k.lower().startswith(f"{prefix}auc")]
        if keys:
            return logs.get(keys[0])
        return None

    def _resolve_metric(self, logs, name):
        if name in logs:
            return logs.get(name)
        for k, v in logs.items():
            if k.lower() == name.lower():
                return v
        return None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        train_auc = self._resolve_auc(logs, prefix="")
        train_precision = self._resolve_metric(logs, "precision")
        train_sens = self._resolve_metric(logs, "sensitivity")
        train_recall = train_sens
        train_f1 = _compute_f1_from_precision_recall(train_precision, train_recall)
        train_spec = self._resolve_metric(logs, "specificity")
        train_spec_at_sens95 = self._resolve_metric(logs, "specificity_at_sens95")
        if train_spec_at_sens95 is None:
            train_spec_at_sens95 = train_spec
        val_loss = logs.get("val_loss")
        val_auc = self._resolve_auc(logs, prefix="val_")
        val_precision = self._resolve_metric(logs, "val_precision")
        val_sens = self._resolve_metric(logs, "val_sensitivity")
        val_recall = val_sens
        val_f1 = _compute_f1_from_precision_recall(val_precision, val_recall)
        val_spec = self._resolve_metric(logs, "val_specificity")
        val_spec_at_sens95 = self._resolve_metric(logs, "val_specificity_at_sens95")
        if val_spec_at_sens95 is None:
            val_spec_at_sens95 = val_spec

        train_seen = None if self.train_steps is None else self.train_steps * self.global_batch_size
        val_seen = None if self.val_steps is None else self.val_steps * self.global_batch_size
        train_thpt = (train_seen / self._train_elapsed) if (train_seen and self._train_elapsed > 0) else None
        val_thpt = (val_seen / self._val_elapsed) if (val_seen and self._val_elapsed > 0) else None
        train_avg_batch_time_ms = (
            (self._train_elapsed / self.train_steps) * 1000.0
            if self.train_steps and self._train_elapsed > 0
            else None
        )
        val_avg_batch_time_ms = (
            (self._val_elapsed / self.val_steps) * 1000.0
            if self.val_steps and self._val_elapsed > 0
            else None
        )
        val_inference_latency_ms_img = (
            (self._val_elapsed / val_seen) * 1000.0
            if val_seen and self._val_elapsed > 0
            else None
        )
        val_inference_latency_ms_batch = val_avg_batch_time_ms
        train_mem_avg = logs.get("train_gpu_mem_avg_mb")
        val_mem_avg = logs.get("val_gpu_mem_avg_mb")
        train_mem_peak = logs.get("train_gpu_mem_peak_mb")
        train_energy_j = logs.get("train_energy_j")
        train_avg_power_w = logs.get("train_avg_power_w")

        row = {
            "epoch": int(epoch),
            "stage": self.stage,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "train_sens": train_sens,
            "train_spec": train_spec,
            "train_spec_at_sens95": train_spec_at_sens95,
            "train_throughput_img_s": train_thpt,
            "train_elapsed_s": self._train_elapsed if self._train_elapsed > 0 else None,
            "train_avg_batch_time_ms": train_avg_batch_time_ms,
            "train_inference_latency_ms_img": None,
            "train_inference_latency_ms_batch": None,
            "train_gpu_mem_avg_mb": train_mem_avg,
            "train_gpu_mem_peak_mb": train_mem_peak,
            "train_energy_j": train_energy_j,
            "train_avg_power_w": train_avg_power_w,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_sens": val_sens,
            "val_spec": val_spec,
            "val_spec_at_sens95": val_spec_at_sens95,
            "val_throughput_img_s": val_thpt,
            "val_elapsed_s": self._val_elapsed if self._val_elapsed > 0 else None,
            "val_avg_batch_time_ms": val_avg_batch_time_ms,
            "val_inference_latency_ms_img": val_inference_latency_ms_img,
            "val_inference_latency_ms_batch": val_inference_latency_ms_batch,
            "val_gpu_mem_avg_mb": val_mem_avg,
            "val_gpu_mem_peak_mb": None,
            "val_energy_j": None,
            "val_avg_power_w": None,
            "test_loss": None,
            "test_auc": None,
            "test_precision": None,
            "test_recall": None,
            "test_f1": None,
            "test_sens": None,
            "test_spec": None,
            "test_spec_at_sens95": None,
            "test_throughput_img_s": None,
            "test_elapsed_s": None,
            "test_avg_batch_time_ms": None,
            "test_inference_latency_ms_img": None,
            "test_inference_latency_ms_batch": None,
            "test_gpu_mem_avg_mb": None,
            "test_gpu_mem_peak_mb": None,
            "test_energy_j": None,
            "test_avg_power_w": None,
            "lr": self._resolve_lr(),
            "total_train_time_s": None,
        }
        if self._writer is not None:
            self._writer.writerow(row)
            self._file.flush()

    def on_train_end(self, logs=None):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None


def decode_example(example, img_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    feature_spec = {
        "imagem": tf.io.FixedLenFeature([], tf.string),
        "retinopatia": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example, feature_spec)
    image = tf.image.decode_jpeg(parsed["imagem"], channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    label = tf.cast(parsed["retinopatia"], tf.float32)
    return image, label


def create_lr_schedule(initial_lr: float, total_epochs: int, warmup_epochs: int, min_lr: float, steps_per_epoch: int):
    """
    Cria um learning rate schedule com warmup linear + cosine annealing.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def schedule(step):
        step = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(warmup_steps, tf.float32)
        total_steps_f = tf.cast(total_steps, tf.float32)
        
        # Warmup phase
        warmup_progress = step / tf.maximum(warmup_steps_f, 1.0)
        warmup_lr = initial_lr * warmup_progress
        
        # Cosine annealing phase
        cosine_step = step - warmup_steps_f
        cosine_total = total_steps_f - warmup_steps_f
        cosine_progress = cosine_step / tf.maximum(cosine_total, 1.0)
        cosine_progress = tf.minimum(tf.maximum(cosine_progress, 0.0), 1.0)
        cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + tf.cos(3.14159265 * cosine_progress))
        
        return tf.cond(step < warmup_steps_f, lambda: warmup_lr, lambda: cosine_lr)
    
    return tf.keras.optimizers.schedules.LearningRateSchedule.__class__(
        "WarmupCosineSchedule",
        (tf.keras.optimizers.schedules.LearningRateSchedule,),
        {"__init__": lambda self: None, "__call__": lambda self, step: schedule(step), "get_config": lambda self: {}}
    )()


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule com warmup linear + cosine annealing."""
    
    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int, min_lr: float):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(self.warmup_steps, tf.float32)
        total_steps_f = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_progress = step / tf.maximum(warmup_steps_f, 1.0)
        warmup_lr = self.initial_lr * warmup_progress
        
        # Cosine annealing phase
        cosine_step = step - warmup_steps_f
        cosine_total = total_steps_f - warmup_steps_f
        cosine_progress = cosine_step / tf.maximum(cosine_total, 1.0)
        cosine_progress = tf.minimum(tf.maximum(cosine_progress, 0.0), 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + tf.cos(3.14159265 * cosine_progress))
        
        return tf.cond(step < warmup_steps_f, lambda: warmup_lr, lambda: cosine_lr)
    
    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def normalize_image(image: tf.Tensor, mode: str, preprocess_fn: Callable):
    image = tf.cast(image, tf.float32)
    if mode == "preprocess" and preprocess_fn is not None:
        return preprocess_fn(image)
    if mode == "unit":
        return image / 255.0
    return image  # raw255 mantém 0..255


def augment_image(image: tf.Tensor, enable: bool):
    if not enable:
        return image
    image = tf.image.random_flip_left_right(image)
    image = tf.cond(
        tf.random.uniform([]) < 0.15, lambda: tf.image.flip_up_down(image), lambda: image
    )
    def _jitter(img, fn, low, high):
        factor = tf.random.uniform([], low, high)
        return fn(img, factor)
    image = tf.cond(
        tf.random.uniform([]) < 0.35,
        lambda: tf.clip_by_value(_jitter(image, lambda i, f: i * f, 0.85, 1.15), 0.0, 255.0),
        lambda: image,
    )
    image = tf.cond(
        tf.random.uniform([]) < 0.35,
        lambda: tf.clip_by_value(tf.image.adjust_contrast(image, tf.random.uniform([], 0.85, 1.15)), 0.0, 255.0),
        lambda: image,
    )
    image = tf.cond(
        tf.random.uniform([]) < 0.3,
        lambda: tf.clip_by_value(tf.image.adjust_saturation(image, tf.random.uniform([], 0.85, 1.15)), 0.0, 255.0),
        lambda: image,
    )
    return image


def build_dataset(
    files,
    batch_size: int,
    img_size: int,
    *,
    training: bool,
    normalize_mode: str,
    preprocess_fn: Callable,
    augment_flag: bool,
):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.with_options(options)
    if training:
        # Shuffle buffer maior para melhor randomização (8192 em vez de 2048)
        dataset = dataset.shuffle(8192, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda ex: decode_example(ex, img_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    if training:
        dataset = dataset.map(
            lambda img, label: (augment_image(img, augment_flag), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    dataset = dataset.map(
        lambda img, label: (normalize_image(img, normalize_mode, preprocess_fn), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size, drop_remainder=training)  # Drop last no treino para batches uniformes
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(model_name: str, input_shape):
    def _builder(app_fn):
        return lambda: app_fn(weights="imagenet", include_top=False, input_tensor=inputs)

    inputs = keras.Input(shape=input_shape)
    builders = {
        "Xception": applications.Xception,
        "VGG16": applications.VGG16,
        "VGG19": applications.VGG19,
        "ResNet50": applications.ResNet50,
        "ResNet50V2": applications.ResNet50V2,
        "ResNet101": applications.ResNet101,
        "ResNet101V2": applications.ResNet101V2,
        "ResNet152": applications.ResNet152,
        "ResNet152V2": applications.ResNet152V2,
        "InceptionV3": applications.InceptionV3,
        "InceptionResNetV2": applications.InceptionResNetV2,
        "MobileNet": applications.MobileNet,
        "MobileNetV2": applications.MobileNetV2,
        "DenseNet121": applications.DenseNet121,
        "DenseNet169": applications.DenseNet169,
        "DenseNet201": applications.DenseNet201,
        "NASNetMobile": applications.NASNetMobile,
        "NASNetLarge": applications.NASNetLarge,
        "EfficientNetB0": applications.EfficientNetB0,
        "EfficientNetB1": applications.EfficientNetB1,
        "EfficientNetB2": applications.EfficientNetB2,
        "EfficientNetB3": applications.EfficientNetB3,
        "EfficientNetB4": applications.EfficientNetB4,
        "EfficientNetB5": applications.EfficientNetB5,
        "EfficientNetB6": applications.EfficientNetB6,
        "EfficientNetB7": applications.EfficientNetB7,
        "ConvNeXtTiny": applications.ConvNeXtTiny,
        "ConvNeXtSmall": applications.ConvNeXtSmall,
        "ConvNeXtBase": applications.ConvNeXtBase,
        "ConvNeXtLarge": applications.ConvNeXtLarge,
        "ConvNeXtXLarge": applications.ConvNeXtXLarge,
    }
    if model_name not in builders:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    base = builders[model_name](weights="imagenet", include_top=False, input_tensor=inputs)
    for layer in base.layers:
        layer.trainable = True
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = keras.Model(inputs, outputs)
    model.base_model = base
    return model


def main():
    args = parse_args()

    tmp_hint = os.environ.get("TMPDIR") or "/tmp"
    try:
        os.makedirs(tmp_hint, exist_ok=True)
    except OSError:
        pass
    tempfile.tempdir = tmp_hint

    configure_hardware(max(0, int(args.cores)))
    strategy = choose_strategy()
    replicas = getattr(strategy, "num_replicas_in_sync", 1)

    IMG = int(args.img_sizes)
    IMAGE_SIZE = (IMG, IMG, 3)
    PER_REPLICA_BS = int(args.batch_size)
    GLOBAL_BATCH_SIZE = PER_REPLICA_BS * replicas  # batch global = batch por réplica * réplicas
    BATCH_SIZE = GLOBAL_BATCH_SIZE  # dataset.batch usa batch global para strategies Mirrored/DP
    EPOCHS = int(args.epochs)
    LR = float(args.lrate)
    VERBOSE = int(args.verbose)
    MODEL_NAME = args.model
    print(f"[INFO] Réplicas={replicas} | batch por réplica={PER_REPLICA_BS} | batch global={GLOBAL_BATCH_SIZE}")

    tfrec_dir = Path(args.tfrec_dir)
    train_files = sorted(tfrec_dir.glob("train*.tfrec"))
    val_files = sorted(tfrec_dir.glob("val*.tfrec")) + sorted(tfrec_dir.glob("valid*.tfrec"))
    test_files = sorted(tfrec_dir.glob("test*.tfrec"))
    valid_files = val_files or test_files
    test_files = test_files or valid_files
    if not train_files or not valid_files or not test_files:
        raise SystemExit("É necessário ao menos um TFRecord de treino e um de validação/teste.")
    print(
        f"Treino: {len(train_files)} arquivos | "
        f"Validação: {len(valid_files)} arquivos | Teste: {len(test_files)} arquivos"
    )

    preprocess_fn = get_preprocess_fn(MODEL_NAME) if args.normalize == "preprocess" else None
    if args.normalize == "preprocess" and preprocess_fn is None:
        print(f"[WARN] preprocess_input não disponível para '{MODEL_NAME}'. Usando 'raw255'.")
        args.normalize = "raw255"

    train_ds = build_dataset(
        [str(p) for p in train_files],
        batch_size=BATCH_SIZE,
        img_size=IMG,
        training=True,
        normalize_mode=args.normalize,
        preprocess_fn=preprocess_fn,
        augment_flag=args.augment,
    )
    valid_ds = build_dataset(
        [str(p) for p in valid_files],
        batch_size=BATCH_SIZE,
        img_size=IMG,
        training=False,
        normalize_mode=args.normalize,
        preprocess_fn=preprocess_fn,
        augment_flag=False,
    )

    results_path = Path(args.results)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / f"{MODEL_NAME}-{args.exec}.csv"
    raw_train_steps = _dataset_cardinality(train_ds)
    raw_val_steps = _dataset_cardinality(valid_ds)
    
    # LR Schedule com warmup + cosine annealing
    warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 3)))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    steps_per_epoch = raw_train_steps if raw_train_steps and raw_train_steps > 0 else 100
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    # Para cálculo de throughput, use cardinalidade real quando disponível; caso contrário, caia no steps_per_epoch.
    train_steps_for_log = raw_train_steps if raw_train_steps and raw_train_steps > 0 else steps_per_epoch
    val_steps_for_log = raw_val_steps if raw_val_steps and raw_val_steps > 0 else steps_per_epoch
    
    lr_schedule = WarmupCosineSchedule(
        initial_lr=LR,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr
    )
    print(f"[Scheduler] Warmup epochs={warmup_epochs}, min_lr={min_lr:.2e}, Cosine Annealing")

    # Label smoothing para regularização
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))
    if label_smoothing > 0:
        print(f"[Label Smoothing] Ativado com valor={label_smoothing:.3f}")
    weight_decay = max(0.0, float(getattr(args, "weight_decay", 1e-5)))
    clipnorm = float(getattr(args, "clipnorm", 1.0))
    optimizer_kwargs = {}
    if clipnorm > 0:
        optimizer_kwargs["clipnorm"] = clipnorm

    with strategy.scope():
        model = build_model(MODEL_NAME, IMAGE_SIZE)
        opt_name = str(getattr(args, "optimizer", "adam")).lower()
        if opt_name in ("adamw", "adam_w"):
            opt = keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                **optimizer_kwargs,
            )
        elif opt_name in ("sgd", "sgd_mom", "sgd-mom"):
            opt = keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=0.9,
                nesterov=True,
                **optimizer_kwargs,
            )
        elif opt_name == "rmsprop":
            opt = keras.optimizers.RMSprop(
                learning_rate=lr_schedule,
                rho=0.9,
                **optimizer_kwargs,
            )
        elif opt_name == "adadelta":
            opt = keras.optimizers.Adadelta(
                learning_rate=lr_schedule,
                rho=0.95,
                **optimizer_kwargs,
            )
        else:
            opt = keras.optimizers.Adam(learning_rate=lr_schedule, **optimizer_kwargs)
        print(
            "[Optimizer] "
            f"name={opt_name}, weight_decay={weight_decay:.1e}, "
            f"clipnorm={clipnorm if clipnorm > 0 else 'off'}"
        )
        model.compile(
            optimizer=opt,
            loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(num_thresholds=args.num_thresholds, name="AUC"),
                keras.metrics.Precision(name="precision"),
                Sensitivity(name="sensitivity"),
                keras.metrics.SpecificityAtSensitivity(
                    SPECIFICITY_TARGET_SENSITIVITY,
                    num_thresholds=args.num_thresholds,
                    name="specificity",
                ),
            ],
        )
        if VERBOSE:
            model.summary()

    csv_logger = EpochCsvLogger(
        csv_path,
        stage="train",
        train_steps=train_steps_for_log,
        val_steps=val_steps_for_log,
        global_batch_size=GLOBAL_BATCH_SIZE,
        append=False,
    )
    mem_logger = GPUMemoryLogger(enabled=args.log_gpu_mem)
    exact_val_logger = ExactEvalMetricsLogger(
        valid_ds,
        prefix="val",
        steps=val_steps_for_log,
        enabled=args.exact_val_every_epoch,
    )
    if not args.exact_val_every_epoch:
        print("[INFO] Validação exata por época desativada; final_test continua exato.")

    best_weights_tracker = BestWeightsTracker(
        monitor=args.early_stop_monitor,
        mode="max",
        verbose=VERBOSE,
    )
    print(
        "[INFO] Rastreador do melhor modelo ativo: "
        f"monitor={args.early_stop_monitor}, storage=memory"
    )

    callbacks = [mem_logger, exact_val_logger, best_weights_tracker]
    if args.early_stop_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=args.early_stop_monitor,
                mode="max",
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                restore_best_weights=True,
                verbose=1 if VERBOSE else 0,
            )
        )
        print(
            "[INFO] Early stopping ativo: "
            f"monitor={args.early_stop_monitor}, "
            f"patience={args.early_stop_patience}, "
            f"min_delta={args.early_stop_min_delta}, "
            "restore_best_weights=True"
        )
    callbacks.append(csv_logger)

    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=VERBOSE,
    )
    trained_epochs = len(history.epoch)
    monitored_values = history.history.get(args.early_stop_monitor)
    best_idx = None
    if monitored_values:
        best_idx = int(np.argmax(monitored_values))
        print(
            "[INFO] Treino finalizado: "
            f"epochs_run={trained_epochs}/{EPOCHS}, "
            f"best_epoch={best_idx + 1}, "
            f"best_{args.early_stop_monitor}={monitored_values[best_idx]:.6f}"
        )
    else:
        print(f"[INFO] Treino finalizado: epochs_run={trained_epochs}/{EPOCHS}")
    if best_weights_tracker.restore(model):
        tracker_epoch = (
            best_weights_tracker.best_epoch + 1
            if best_weights_tracker.best_epoch is not None
            else None
        )
        tracker_value = best_weights_tracker.best_value
        if tracker_epoch is not None and np.isfinite(tracker_value):
            print(
                "[INFO] Pesos restaurados do melhor checkpoint em memoria: "
                f"epoch={tracker_epoch}, "
                f"{args.early_stop_monitor}={tracker_value:.6f}"
            )
        elif best_idx is not None:
            print(
                "[INFO] Pesos restaurados do melhor checkpoint em memoria: "
                f"epoch={best_idx + 1}, "
                f"{args.early_stop_monitor}={monitored_values[best_idx]:.6f}"
            )
        else:
            print("[INFO] Pesos restaurados do melhor checkpoint em memoria para o final_test.")
    else:
        print("[WARN] Nenhum checkpoint em memoria encontrado para restaurar; usando pesos finais.")
    elapsed = round(time.time() - t_start, 1)

    valid_eval = build_dataset(
        [str(p) for p in test_files],
        batch_size=BATCH_SIZE,
        img_size=IMG,
        training=False,
        normalize_mode=args.normalize,
        preprocess_fn=preprocess_fn,
        augment_flag=False,
    )

    y_true = []
    y_score = []
    eval_predict_start = time.time()
    _test_scope = EnergyScope(_gpu_tele())
    _test_scope.start()
    eval_batches = 0
    eval_inference_time_s = 0.0
    eval_memory_samples_mb = []
    for batch_images, batch_labels in valid_eval:
        inference_start = time.perf_counter()
        preds = model(batch_images, training=False)
        preds_np = preds.numpy().ravel()
        eval_inference_time_s += time.perf_counter() - inference_start
        y_score.append(preds_np)
        y_true.append(batch_labels.numpy())
        eval_batches += 1
        current_mem = _read_gpu_current_memory_mb()
        if current_mem is not None:
            eval_memory_samples_mb.append(float(current_mem))
    _test_scope.stop()
    eval_predict_elapsed = time.time() - eval_predict_start
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    auc_val = roc_auc_score(y_true, y_score)
    final_metrics = compute_binary_metrics(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    test_inference_latency_ms_img = (
        eval_inference_time_s / len(y_true) * 1000.0 if len(y_true) > 0 else None
    )
    test_inference_latency_ms_batch = (
        eval_inference_time_s / eval_batches * 1000.0 if eval_batches > 0 else None
    )
    test_gpu_mem_avg_mb = (
        float(np.mean(eval_memory_samples_mb)) if eval_memory_samples_mb else None
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
    thresholds_path = results_path / f"{args.dataset}-{args.exec}-thresholds.csv"
    thresholds_df.to_csv(thresholds_path, index=False, encoding="utf-8")

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    pdf_path = results_path / f"{args.dataset}-{args.exec}.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    final_row = {
        "epoch": trained_epochs,
        "stage": "final_test",
        "train_loss": None,
        "train_auc": None,
        "train_precision": None,
        "train_recall": None,
        "train_f1": None,
        "train_sens": None,
        "train_spec": None,
        "train_spec_at_sens95": None,
        "train_throughput_img_s": None,
        "train_elapsed_s": None,
        "train_avg_batch_time_ms": None,
        "train_inference_latency_ms_img": None,
        "train_inference_latency_ms_batch": None,
        "train_gpu_mem_avg_mb": None,
        "train_gpu_mem_peak_mb": None,
        "train_energy_j": None,
        "train_avg_power_w": None,
        "val_loss": None,
        "val_auc": None,
        "val_precision": None,
        "val_recall": None,
        "val_f1": None,
        "val_sens": None,
        "val_spec": None,
        "val_spec_at_sens95": None,
        "val_throughput_img_s": None,
        "val_elapsed_s": None,
        "val_avg_batch_time_ms": None,
        "val_inference_latency_ms_img": None,
        "val_inference_latency_ms_batch": None,
        "val_gpu_mem_avg_mb": None,
        "val_gpu_mem_peak_mb": None,
        "val_energy_j": None,
        "val_avg_power_w": None,
        "test_loss": None,
        "test_auc": auc_val,
        "test_precision": final_metrics["precision"],
        "test_recall": final_metrics["recall"],
        "test_f1": final_metrics["f1"],
        "test_sens": final_metrics["sensitivity"],
        "test_spec": final_metrics["specificity"],
        "test_spec_at_sens95": final_metrics["specificity_at_sens95"],
        "test_throughput_img_s": (len(y_true) / eval_predict_elapsed) if eval_predict_elapsed > 0 else None,
        "test_elapsed_s": eval_predict_elapsed,
        "test_avg_batch_time_ms": (eval_predict_elapsed / max(eval_batches, 1)) * 1000.0 if eval_batches > 0 else None,
        "test_inference_latency_ms_img": test_inference_latency_ms_img,
        "test_inference_latency_ms_batch": test_inference_latency_ms_batch,
        "test_gpu_mem_avg_mb": (
            _test_scope.avg_mem_mb if _test_scope.avg_mem_mb is not None else test_gpu_mem_avg_mb
        ),
        "test_gpu_mem_peak_mb": _test_scope.peak_mem_mb,
        "test_energy_j": _test_scope.energy_j,
        "test_avg_power_w": _test_scope.avg_power_w,
        "lr": None,
        "total_train_time_s": elapsed,
    }
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EpochCsvLogger._fields)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(final_row)
    except Exception as exc:
        print(f"[WARN] Falha ao registrar linha final no CSV: {exc}")

    print(f"Test AUC (final): {auc_val:.4f}")
    print(f"Tempo total: {elapsed}s")
    print(f"{args.dataset},{args.exec},{auc_val:.6f},{elapsed}")


if __name__ == "__main__":
    main()

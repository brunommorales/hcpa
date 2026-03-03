# -*- coding: utf-8 -*-
"""
Versão básica do treinamento TensorFlow distribuído.
- Sem fases de fine-tuning, mixup, DALI ou EMA.
- Apenas TFRecord -> tf.data -> modelo keras.applications com Mirrored/MultiWorker.
"""
import argparse
import csv
import json
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


class Specificity(tf.keras.metrics.Metric):
    """Specificity (True Negative Rate) at threshold 0.5."""

    def __init__(self, name="specificity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)

    def result(self):
        return self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tn.assign(0.0)
        self.fp.assign(0.0)


def compute_sens_spec(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    """Compute sensitivity/specificity at a fixed threshold (default 0.5)."""
    preds = (y_score >= threshold).astype(np.float32)
    tp = np.sum((preds == 1) & (y_true == 1))
    fn = np.sum((preds == 0) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return sens, spec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento simples em GPU usando TFRecord + Keras com distribuição"
    )
    parser.add_argument("--tfrec_dir", type=str, default="./data/all", help="Diretório com TFRecords")
    parser.add_argument("--dataset", type=str, default="all", help="Nome lógico do dataset")
    parser.add_argument("--results", type=str, default="./results/all", help="Diretório para salvar resultados")
    parser.add_argument("--exec", type=int, default=0, help="ID de execução")
    parser.add_argument("--img_sizes", type=int, default=299, help="Tamanho das imagens (quadradas)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch por réplica/GPU (batch global = batch_size * réplicas)")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_thresholds", type=int, default=200, help="(Mantido para compatibilidade)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose do Keras")
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
            json.loads(tf_config_raw)
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print("[Distribuição] MultiWorkerMirroredStrategy ativa via TF_CONFIG.")
            return strategy
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


def _read_gpu_memory_mb():
    """Retorna (peak_mb, current_mb) considerando o uso corrente reportado pelo nvidia-smi."""
    # Preferimos o valor do nvidia-smi (consumo atual visível na placa)
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
            current = max(used_vals)
            return current, current
    except Exception:
        pass

    # Fallback: API do TensorFlow (current/peak por device)
    try:
        logical_gpus = tf.config.list_logical_devices("GPU")
        peaks = []
        currents = []
        for idx, _ in enumerate(logical_gpus):
            try:
                info = tf.config.experimental.get_memory_info(f"GPU:{idx}")
            except Exception:
                continue
            peaks.append(info.get("peak", 0))
            currents.append(info.get("current", 0))
        if not peaks:
            raise RuntimeError("get_memory_info retornou lista vazia")
        to_mb = 1024 * 1024
        return max(peaks) / to_mb, max(currents) / to_mb
    except Exception:
        return None, None


class GPUMemoryLogger(keras.callbacks.Callback):
    """Adiciona métricas de memória GPU ao dicionário de logs por época."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return
        logs = logs or {}
        peak_mb, current_mb = _read_gpu_memory_mb()
        if peak_mb is not None:
            logs["gpu_mem_peak_mb"] = peak_mb
        if current_mb is not None:
            logs["gpu_mem_current_mb"] = current_mb



def _dataset_cardinality(dataset):
    try:
        n = tf.data.experimental.cardinality(dataset).numpy()
        return int(n) if n >= 0 else None
    except Exception:
        return None


class EpochCsvLogger(keras.callbacks.Callback):
    """Escreve métricas por época em CSV com o layout do pytorch_opt."""

    _fields = [
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
        train_sens = self._resolve_metric(logs, "sensitivity")
        train_spec = self._resolve_metric(logs, "specificity")
        val_loss = logs.get("val_loss")
        val_auc = self._resolve_auc(logs, prefix="val_")
        val_sens = self._resolve_metric(logs, "val_sensitivity")
        val_spec = self._resolve_metric(logs, "val_specificity")

        train_seen = None if self.train_steps is None else self.train_steps * self.global_batch_size
        val_seen = None if self.val_steps is None else self.val_steps * self.global_batch_size
        train_thpt = (train_seen / self._train_elapsed) if (train_seen and self._train_elapsed > 0) else None
        val_thpt = (val_seen / self._val_elapsed) if (val_seen and self._val_elapsed > 0) else None

        mem_peak = logs.get("gpu_mem_peak_mb")
        mem_current = logs.get("gpu_mem_current_mb")
        # Usamos a alocação corrente como medida de consumo de memória.
        mem_alloc = mem_current if mem_current is not None else mem_peak
        mem_reserved = mem_alloc

        row = {
            "epoch": int(epoch),
            "stage": self.stage,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "train_sens": train_sens,
            "train_spec": train_spec,
            "train_throughput_img_s": train_thpt,
            "train_elapsed_s": self._train_elapsed if self._train_elapsed > 0 else None,
            "train_gpu_mem_alloc_mb": mem_alloc,
            "train_gpu_mem_reserved_mb": mem_reserved,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_sens": val_sens,
            "val_spec": val_spec,
            "val_throughput_img_s": val_thpt,
            "val_elapsed_s": self._val_elapsed if self._val_elapsed > 0 else None,
            "val_gpu_mem_alloc_mb": mem_alloc,
            "val_gpu_mem_reserved_mb": mem_reserved,
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
    valid_files = (
        sorted(tfrec_dir.glob("test*.tfrec"))
        + sorted(tfrec_dir.glob("val*.tfrec"))
        + sorted(tfrec_dir.glob("valid*.tfrec"))
    )
    if not train_files or not valid_files:
        raise SystemExit("É necessário ao menos um TFRecord de treino e um de validação.")
    print(f"Treino: {len(train_files)} arquivos | Validação: {len(valid_files)} arquivos")

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

    with strategy.scope():
        model = build_model(MODEL_NAME, IMAGE_SIZE)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(name="AUC"),
                Sensitivity(name="sensitivity"),
                Specificity(name="specificity"),
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

    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[mem_logger, csv_logger],
        verbose=VERBOSE,
    )
    elapsed = round(time.time() - t_start, 1)

    valid_eval = build_dataset(
        [str(p) for p in valid_files],
        batch_size=BATCH_SIZE,
        img_size=IMG,
        training=False,
        normalize_mode=args.normalize,
        preprocess_fn=preprocess_fn,
        augment_flag=False,
    )

    y_true = []
    y_score = []
    for batch_images, batch_labels in valid_eval:
        preds = model(batch_images, training=False)
        y_score.append(preds.numpy().ravel())
        y_true.append(batch_labels.numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    auc_val = roc_auc_score(y_true, y_score)
    sens_val, spec_val = compute_sens_spec(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

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
        "epoch": args.epochs,
        "stage": "final_eval",
        "train_loss": None,
        "train_auc": None,
        "train_sens": None,
        "train_spec": None,
        "train_throughput_img_s": None,
        "train_elapsed_s": None,
        "train_gpu_mem_alloc_mb": None,
        "train_gpu_mem_reserved_mb": None,
        "val_loss": None,
        "val_auc": auc_val,
        "val_sens": sens_val,
        "val_spec": spec_val,
        "val_throughput_img_s": None,
        "val_elapsed_s": None,
        "val_gpu_mem_alloc_mb": None,
        "val_gpu_mem_reserved_mb": None,
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

    print(f"Valid AUC (final): {auc_val:.4f}")
    print(f"Tempo total: {elapsed}s")
    print(f"{args.dataset},{args.exec},{auc_val:.6f},{elapsed}")


if __name__ == "__main__":
    main()

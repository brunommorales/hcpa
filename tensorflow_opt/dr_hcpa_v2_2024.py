# -*- coding: utf-8 -*-
import os
import tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
_tmp_hint = os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP")
if not _tmp_hint:
    _tmp_hint = "/tmp"
    os.environ.setdefault("TMPDIR", _tmp_hint)
try:
    os.makedirs(_tmp_hint, exist_ok=True)
except OSError:
    pass
_rank_hint = (
    os.environ.get("NODE_RANK")
    or os.environ.get("RANK")
    or os.environ.get("LOCAL_RANK")
    or os.environ.get("OMPI_COMM_WORLD_RANK")
    or "0"
)
_mpl_fallback = os.path.join(_tmp_hint, f"mplconfig_{_rank_hint}")
try:
    os.makedirs(_mpl_fallback, exist_ok=True)
except OSError:
    pass
os.environ.setdefault("MPLCONFIGDIR", _mpl_fallback)
tempfile.tempdir = _tmp_hint

import pandas as pd, numpy as np, math, json
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import re
from pathlib import Path
from shutil import rmtree
from os import makedirs, rename, listdir
from os.path import join, exists, isfile
import time
import argparse
from tensorflow.keras import applications
from tensorflow.keras import mixed_precision
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
try:
    from tensorflow.keras.utils import to_numpy_or_python_type  # TF >= 2.10
except ImportError:
    def to_numpy_or_python_type(value):
        if isinstance(value, tf.Variable):
            value = value.value()
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        return value
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import csv


CACHE_BASE_DIR = None
AUTO_SHARD_POLICY = tf.data.experimental.AutoShardPolicy.DATA
DISTRIBUTION_STRATEGY = None
USE_UINT8_DECODE = False
COMPUTE_DTYPE = tf.float32
SKIP_INTERNAL_CLEANUP = os.environ.get("HCPA_SKIP_INTERNAL_RESULTS_CLEANUP", "").strip().lower() in ("1", "true", "yes")


def to_compute_dtype(tensor):
    target = tf.as_dtype(COMPUTE_DTYPE)
    if tensor.dtype == target:
        return tensor
    return tf.cast(tensor, target)
DALI_AVAILABLE = False

try:
    from nvidia.dali import fn as dali_fn
    from nvidia.dali import types as dali_types
    from nvidia.dali import tfrecord as dali_tfrec
    from nvidia.dali.pipeline import pipeline_def as dali_pipeline_def
    from nvidia.dali.plugin.tf import DALIDataset
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy as DaliLastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    dali_fn = None
    dali_types = None
    dali_tfrec = None
    dali_pipeline_def = None
    DALIDataset = None

if not DALI_AVAILABLE:
    DaliLastBatchPolicy = None


def get_worker_rank():
    """Detect numeric worker rank from common environment variables."""
    for key in ("NODE_RANK", "RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        value = os.getenv(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    tf_config = os.getenv("TF_CONFIG")
    if tf_config:
        try:
            config = json.loads(tf_config)
            task = config.get("task", {})
            index = task.get("index")
            if index is not None:
                return int(index)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return 0


def is_primary_worker():
    """Return True only for the process that should manage shared resources."""
    return get_worker_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser(description='This script is used for training the hcpa model using dataset from tfrecord files. See more: python3 dr_hcpa_v2_2024.py -h')
    parser.add_argument('--tfrec_dir', type=str, default='./data/all', help='Directory containing TFRecord files')
    parser.add_argument('--dataset', type=str, default='all', help='Name of the dataset')
    parser.add_argument('--results', type=str, default='./results/all', help='Directory to save results')
    parser.add_argument('--exec', type=int, default=0, help='Execution number')
    parser.add_argument('--img_sizes', type=int, default=299, help='Image sizes')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch por réplica/GPU (batch global = batch_size * réplicas)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--lrate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--num_thresholds', type=int, default=200, help='Number of thresholds')
    parser.add_argument('--wait_epochs', type=int, default=30, help='Number of epochs to wait')
    parser.add_argument('--show_files', type=bool, default=False, help='Show files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose level for training')
    parser.add_argument('--model', type=str, default='InceptionV3', help='Model for training')
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='none',
        help="Directory to cache TF datasets; use 'none' (default) to disable caching",
    )
    parser.add_argument('--freeze_epochs', type=int, default=1, help='Number of initial epochs to keep the base model frozen')
    parser.add_argument('--fine_tune_lr_factor', type=float, default=0.1, help='Learning rate multiplier applied when unfreezing the base model')
    parser.add_argument('--fine_tune_at', type=int, default=-200, help='Layer index in the base model from which fine-tuning starts')
    parser.add_argument('--fine_tune_lr', type=float, default=2e-4, help='Absolute learning rate used during fine-tuning (overrides factor if > 0)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau'], help='Learning rate scheduler strategy')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs for cosine scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate enforced by the scheduler')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping norm (<=0 disables)')
    parser.add_argument('--mixup_alpha', type=float, default=0.3, help='Alpha parameter for mixup (<=0 disables)')
    parser.add_argument('--cutmix_alpha', type=float, default=0.0, help='Alpha parameter for CutMix (<=0 disables)')
    parser.add_argument('--label_smoothing', type=float, default=0.01, help='Label smoothing applied to the binary cross-entropy loss')
    parser.add_argument('--focal_gamma', type=float, default=0.0, help='Gamma for focal loss modulation (<=0 disables focal)')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Positive class weight for imbalance handling (>1 up-weights positives)')
    parser.add_argument('--fundus_crop_ratio', type=float, default=0.9, help='Central crop ratio to remove black borders (<=0 disables crop)')
    parser.add_argument('--augment', dest='augment', action='store_true', help='Enable data augmentations during training')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='Disable data augmentations during training')
    parser.set_defaults(augment=True)
    parser.add_argument('--freeze_bn', dest='freeze_bn', action='store_true', help='Keep BatchNorm layers frozen during fine-tuning')
    parser.add_argument('--no-freeze_bn', dest='freeze_bn', action='store_false', help='Allow BatchNorm layers to update statistics during fine-tuning')
    parser.set_defaults(freeze_bn=True)
    parser.add_argument('--fine_tune_schedule', type=str, default='', help='Optional schedule in the form epoch:layer[:bn] to progressively unfreeze layers')
    parser.add_argument('--normalize', type=str, default='preprocess', choices=['preprocess', 'raw255', 'unit'], help='Normalization scheme applied after decoding the image')
    parser.add_argument('--channels_last', action='store_true', help='Ensure Channels Last image data format')
    parser.add_argument('--h2d_uint8', action='store_true', help='Keep decoded images as uint8 until normalization')
    parser.add_argument('--tta_views', type=int, default=1, help='Number of TTA views used during final evaluation (>=1)')
    parser.add_argument('--cores', type=int, default=0,
                        help='Define OMP_NUM_THREADS e threads de CPU (<=0 mantém padrão)')
    parser.add_argument('--mixed_precision', dest='mixed_precision', action='store_true', help='Enable mixed precision (float16) when supported by the device')
    parser.add_argument('--no-mixed_precision', dest='mixed_precision', action='store_false', help='Disable mixed precision, force float32 execution')
    parser.set_defaults(mixed_precision=True)
    parser.add_argument('--use_dali', action='store_true', help='Use NVIDIA DALI for input pipeline')
    parser.add_argument('--dali_threads', type=int, default=4, help='Number of threads per DALI pipeline')
    parser.add_argument('--dali_layout', type=str, default='NHWC', choices=['NHWC', 'NCHW'], help='Image layout produced by DALI pipeline')
    parser.add_argument('--dali_seed', type=int, default=2024, help='Seed used by DALI pipeline (offset per worker)')
    parser.add_argument('--recompute_backbone', action='store_true', help='Habilita tf.recompute_grad no backbone para economizar memória (pode reduzir throughput)')
    parser.add_argument('--jit_compile', action='store_true', help='Habilita jit_compile no Keras/TensorFlow se suportado (pode acelerar, mas aumenta compilação)')
    parser.add_argument('--auc_target', type=float, default=0.95, help='Valor de AUC de validação para registrar tempo de chegada (não interrompe o treino)')
    parser.add_argument('--save_val_preds', type=str, default='', help='Se informado, salva logits e labels de validação em NPZ neste caminho')
    parser.add_argument('--calibrate', action='store_true', help='Executa calibração (temperature scaling) e busca limiar para sensibilidade-alvo')
    parser.add_argument('--sens_target', type=float, default=0.9, help='Sensibilidade-alvo para ajuste de limiar (0-1)')

    return parser.parse_args()

_PREPROCESS_MAP = {
    'InceptionV3': applications.inception_v3.preprocess_input,
    'InceptionResNetV2': applications.inception_resnet_v2.preprocess_input,
    'Xception': applications.xception.preprocess_input,
    'VGG16': applications.vgg16.preprocess_input,
    'VGG19': applications.vgg19.preprocess_input,
    'ResNet50': applications.resnet50.preprocess_input if hasattr(applications, "resnet50") else applications.resnet.preprocess_input,
    'ResNet50V2': applications.resnet_v2.preprocess_input,
    'ResNet101': applications.resnet.preprocess_input,
    'ResNet101V2': applications.resnet_v2.preprocess_input,
    'ResNet152': applications.resnet.preprocess_input,
    'ResNet152V2': applications.resnet_v2.preprocess_input,
    'MobileNet': applications.mobilenet.preprocess_input,
    'MobileNetV2': applications.mobilenet_v2.preprocess_input,
    'DenseNet121': applications.densenet.preprocess_input,
    'DenseNet169': applications.densenet.preprocess_input,
    'DenseNet201': applications.densenet.preprocess_input,
    'NASNetMobile': applications.nasnet.preprocess_input,
    'NASNetLarge': applications.nasnet.preprocess_input,
    'EfficientNetB0': applications.efficientnet.preprocess_input,
    'EfficientNetB1': applications.efficientnet.preprocess_input,
    'EfficientNetB2': applications.efficientnet.preprocess_input,
    'EfficientNetB3': applications.efficientnet.preprocess_input,
    'EfficientNetB4': applications.efficientnet.preprocess_input,
    'EfficientNetB5': applications.efficientnet.preprocess_input,
    'EfficientNetB6': applications.efficientnet.preprocess_input,
    'EfficientNetB7': applications.efficientnet.preprocess_input,
    'EfficientNetV2B0': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'EfficientNetV2B1': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'EfficientNetV2B2': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'EfficientNetV2B3': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'EfficientNetV2S': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'EfficientNetV2M': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'EfficientNetV2L': applications.efficientnet_v2.preprocess_input if hasattr(applications, "efficientnet_v2") else None,
    'ConvNeXtTiny': applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    'ConvNeXtSmall': applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    'ConvNeXtBase': applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    'ConvNeXtLarge': applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
    'ConvNeXtXLarge': applications.convnext.preprocess_input if hasattr(applications, "convnext") else None,
}


def get_preprocess_fn(model_name):
    fn = _PREPROCESS_MAP.get(model_name)
    return fn


if DALI_AVAILABLE:
    DALI_TFREC_FEATURES = {
        "imagem": dali_tfrec.FixedLenFeature((), dali_tfrec.string, ""),
        "retinopatia": dali_tfrec.FixedLenFeature([1], dali_tfrec.int64, 0),
    }

    def _dali_path_list(paths):
        return [str(Path(p)) for p in paths]

    def _dali_clamp(node, min_value: float, max_value: float):
        if hasattr(dali_fn, "clamp"):
            return dali_fn.clamp(node, min=min_value, max=max_value)
        return node

    @dali_pipeline_def(exec_dynamic=True)
    def _dali_retina_pipeline(
        tfrec_files,
        idx_files,
        *,
        shard_id: int,
        num_shards: int,
        reader_seed: int,
        image_size: int,
        enable_augment: bool,
        output_layout: str,
        fundus_crop_ratio: float,
    ):
        assert output_layout in ("NHWC", "NCHW")

        reader = dali_fn.readers.tfrecord(
            path=tfrec_files,
            index_path=idx_files,
            features=DALI_TFREC_FEATURES,
            random_shuffle=True,
            initial_fill=1024,
            read_ahead=True,
            prefetch_queue_depth=2,
            name="Reader",
            seed=reader_seed,
            skip_cached_images=False,
            shard_id=shard_id,
            num_shards=num_shards,
        )

        images = dali_fn.decoders.image(
            reader["imagem"],
            device="mixed",
            output_type=dali_types.RGB,
        )
        images = dali_fn.resize(images, resize_x=image_size, resize_y=image_size)
        if fundus_crop_ratio < 0.999:
            images = dali_fn.crop(
                images,
                crop=[fundus_crop_ratio, fundus_crop_ratio],
                crop_pos_x=0.5,
                crop_pos_y=0.5,
            )
            images = dali_fn.resize(images, resize_x=image_size, resize_y=image_size)
        images = dali_fn.cast(images, dtype=dali_types.FLOAT)

        if enable_augment:
            mirror_h = dali_fn.random.coin_flip(probability=0.5)
            mirror_v = dali_fn.random.coin_flip(probability=0.15)
            images = dali_fn.flip(images, horizontal=mirror_h, vertical=mirror_v)

            rotate_cond = dali_fn.random.coin_flip(probability=0.3)
            rotate_angle = dali_fn.random.uniform(range=(-15.0, 15.0)) * rotate_cond
            images = dali_fn.rotate(
                images,
                angle=rotate_angle,
                keep_size=True,
                fill_value=0.0,
            )

            contrast_random = dali_fn.random.uniform(range=(0.85, 1.15))
            contrast_cond = dali_fn.random.coin_flip(probability=0.35)
            contrast_factor = 1.0 + (contrast_random - 1.0) * contrast_cond

            brightness_random = dali_fn.random.uniform(range=(0.85, 1.15))
            brightness_cond = dali_fn.random.coin_flip(probability=0.35)
            brightness_factor = 1.0 + (brightness_random - 1.0) * brightness_cond

            images = dali_fn.brightness_contrast(
                images,
                contrast=contrast_factor,
                brightness=brightness_factor,
            )

            saturation_random = dali_fn.random.uniform(range=(0.85, 1.15))
            saturation_cond = dali_fn.random.coin_flip(probability=0.3)
            saturation_factor = 1.0 + (saturation_random - 1.0) * saturation_cond
            images = dali_fn.color_twist(images, saturation=saturation_factor)

            images = _dali_clamp(images, min_value=0.0, max_value=255.0)

        dali_layout = "HWC" if output_layout == "NHWC" else "CHW"
        images = dali_fn.crop_mirror_normalize(
            images,
            dtype=dali_types.FLOAT,
            output_layout=dali_layout,
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        )
        images = dali_fn.copy(images, device="cpu")

        labels = reader["retinopatia"]
        labels = dali_fn.cast(labels, dtype=dali_types.FLOAT)
        return images, labels

    def build_dali_tf_dataset(
        tfrec_files,
        idx_files,
        *,
        batch_size: int,
        image_size: int,
        augment: bool,
        layout: str,
        seed: int,
        threads: int,
        shard_id: int = 0,
        num_shards: int = 1,
        fundus_crop_ratio: float = 1.0,
    ):
        tfrec_files = _dali_path_list(tfrec_files)
        idx_files = _dali_path_list(idx_files)
        pipeline = _dali_retina_pipeline(
            tfrec_files=tfrec_files,
            idx_files=idx_files,
            batch_size=batch_size,
            num_threads=threads,
            device_id=shard_id,
            shard_id=shard_id,
            num_shards=num_shards,
            seed=seed,
            reader_seed=seed,
            image_size=image_size,
            enable_augment=augment,
            output_layout=layout,
            fundus_crop_ratio=fundus_crop_ratio,
        )
        if layout == "NHWC":
            image_shape = tf.TensorShape([batch_size, image_size, image_size, 3])
        else:
            image_shape = tf.TensorShape([batch_size, 3, image_size, image_size])
        label_shape = tf.TensorShape([batch_size, 1])
        dataset = DALIDataset(
            pipeline=pipeline,
            batch_size=batch_size,
            output_shapes=(image_shape, label_shape),
            output_dtypes=(tf.float32, tf.float32),
            device_id=shard_id,
            prefetch_queue_depth=2,
        )
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda images, labels: (to_compute_dtype(images), tf.cast(tf.reshape(labels, [batch_size]), tf.float32)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

def create_model(model_name, input_shape):
    def _builder(app_fn):
        return lambda: app_fn(weights='imagenet', include_top=False, input_shape=input_shape)

    builders = {
        'Xception': _builder(applications.Xception),
        'VGG16': _builder(applications.VGG16),
        'VGG19': _builder(applications.VGG19),
        'ResNet50': _builder(applications.ResNet50),
        'ResNet50V2': _builder(applications.ResNet50V2),
        'ResNet101': _builder(applications.ResNet101),
        'ResNet101V2': _builder(applications.ResNet101V2),
        'ResNet152': _builder(applications.ResNet152),
        'ResNet152V2': _builder(applications.ResNet152V2),
        'InceptionV3': _builder(applications.InceptionV3),
        'InceptionResNetV2': _builder(applications.InceptionResNetV2),
        'MobileNet': _builder(applications.MobileNet),
        'MobileNetV2': _builder(applications.MobileNetV2),
        'DenseNet121': _builder(applications.DenseNet121),
        'DenseNet169': _builder(applications.DenseNet169),
        'DenseNet201': _builder(applications.DenseNet201),
        'NASNetMobile': _builder(applications.NASNetMobile),
        'NASNetLarge': _builder(applications.NASNetLarge),
        'EfficientNetB0': _builder(applications.EfficientNetB0),
        'EfficientNetB1': _builder(applications.EfficientNetB1),
        'EfficientNetB2': _builder(applications.EfficientNetB2),
        'EfficientNetB3': _builder(applications.EfficientNetB3),
        'EfficientNetB4': _builder(applications.EfficientNetB4),
        'EfficientNetB5': _builder(applications.EfficientNetB5),
        'EfficientNetB6': _builder(applications.EfficientNetB6),
        'EfficientNetB7': _builder(applications.EfficientNetB7),
        'EfficientNetV2B0': _builder(applications.EfficientNetV2B0),
        'EfficientNetV2B1': _builder(applications.EfficientNetV2B1),
        'EfficientNetV2B2': _builder(applications.EfficientNetV2B2),
        'EfficientNetV2B3': _builder(applications.EfficientNetV2B3),
        'EfficientNetV2S': _builder(applications.EfficientNetV2S),
        'EfficientNetV2M': _builder(applications.EfficientNetV2M),
        'EfficientNetV2L': _builder(applications.EfficientNetV2L),
        'ConvNeXtTiny': _builder(applications.ConvNeXtTiny),
        'ConvNeXtSmall': _builder(applications.ConvNeXtSmall),
        'ConvNeXtBase': _builder(applications.ConvNeXtBase),
        'ConvNeXtLarge': _builder(applications.ConvNeXtLarge),
        'ConvNeXtXLarge': _builder(applications.ConvNeXtXLarge),
    }

    builder = builders.get(model_name)
    if builder is None:
        return None
    return builder()

class LogMetrics(Callback):
    pass


class EnsureScalarMetrics(Callback):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy

    def _resolve_value(self, value):
        if isinstance(value, tf.distribute.DistributedValues):
            value = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
        if isinstance(value, tf.Variable) and hasattr(value, 'values'):
            component_values = [to_numpy_or_python_type(v) for v in value.values]
            value = float(np.mean(component_values))
        try:
            return to_numpy_or_python_type(value)
        except (TypeError, NotImplementedError):
            return value

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        for key in list(logs.keys()):
            logs[key] = self._resolve_value(logs[key])

    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False


class GpuMemoryTracker(Callback):
    def __init__(self):
        super().__init__()
        self.device_names = []
        try:
            self.device_names = [dev.name for dev in tf.config.list_logical_devices('GPU')]
        except Exception:
            self.device_names = []
        self._get_memory_info = getattr(tf.config.experimental, "get_memory_info", None)
        self._reset_memory_stats = getattr(tf.config.experimental, "reset_memory_stats", None)
        self.enabled = bool(self.device_names) and callable(self._get_memory_info)
        self.max_peak_mb = 0.0

    def _read_nvidia_smi_usage(self):
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
                used_vals.append(float(line))
            if used_vals:
                return max(used_vals)
        except Exception:
            return None
        return None

    def _reset_stats(self):
        if not self.enabled or not callable(self._reset_memory_stats):
            return
        for name in self.device_names:
            try:
                self._reset_memory_stats(name)
            except Exception:
                continue

    def on_train_begin(self, logs=None):
        self._reset_stats()

    def on_epoch_begin(self, epoch, logs=None):
        self._reset_stats()

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled or logs is None:
            return
        current_mb = self._read_nvidia_smi_usage()
        peak_mb = current_mb
        if current_mb is None:
            max_peak = 0.0
            max_current = 0.0
            for name in self.device_names:
                try:
                    info = self._get_memory_info(name)
                except Exception:
                    continue
                if not isinstance(info, dict):
                    continue
                peak = info.get("peak") or 0
                current = info.get("current") or 0
                if peak > max_peak:
                    max_peak = peak
                if current > max_current:
                    max_current = current
            if max_peak <= 0 and max_current <= 0:
                return
            peak_mb = max_peak / (1024 ** 2)
            current_mb = max_current / (1024 ** 2)

        logs["gpu_mem_peak_mb"] = peak_mb
        logs["gpu_mem_current_mb"] = current_mb
        logs["gpu_mem_devices"] = len(self.device_names)
        self.max_peak_mb = max(self.max_peak_mb, peak_mb)

    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False


class TrainingSpeedLogger(Callback):
    """Mede tempo por época, throughput e registra tempo para atingir AUC alvo."""

    def __init__(self, steps_per_epoch, global_batch_size, target_auc, results_dir, primary_worker=True):
        super().__init__()
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.global_batch_size = max(1, int(global_batch_size))
        self.target_auc = float(target_auc)
        self.results_dir = Path(results_dir) if results_dir else None
        self.primary_worker = primary_worker
        self._epoch_start = None
        self._train_start = None
        self.target_hit_time = None

    def on_train_begin(self, logs=None):
        if self._train_start is None:
            self._train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_start is None:
            return
        duration = max(time.time() - self._epoch_start, 1e-6)
        imgs = self.steps_per_epoch * self.global_batch_size
        throughput = imgs / duration
        if logs is not None:
            logs["epoch_time_sec"] = duration
            logs["throughput_img_s"] = throughput
        val_auc = None
        if logs is not None:
            val_auc = logs.get("val_AUC")
        if self.target_hit_time is None and val_auc is not None and val_auc >= self.target_auc:
            self.target_hit_time = time.time() - (self._train_start or time.time())

    def on_train_end(self, logs=None):
        if not self.primary_worker or self.results_dir is None:
            return
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.results_dir / f"auc_target_{self.target_auc:.3f}.txt"
            if self.target_hit_time is None:
                out_path.write_text("not_reached\n", encoding="utf-8")
            else:
                out_path.write_text(f"{self.target_hit_time:.3f}\n", encoding="utf-8")
        except Exception:
            pass

    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False


class TuningFileCallback(Callback):
    """Aplica ajustes online vindos do autotuner lendo um arquivo JSON por época."""

    def __init__(self, tuning_file, optimizer, mixup_var=None, checkpoints_dir=None, best_path=None, last_path=None, verbose=False):
        super().__init__()
        self.tuning_file = Path(tuning_file) if tuning_file else None
        self.optimizer = optimizer
        self.mixup_var = mixup_var
        self.verbose = verbose
        self._last_payload = None
        self._rollback_abort_pending = False  # True quando abortamos época – aguardar on_epoch_begin
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else None
        self.best_path = Path(best_path) if best_path else (self.checkpoints_dir / "best.ckpt" if self.checkpoints_dir else None)
        self.last_path = Path(last_path) if last_path else (self.checkpoints_dir / "last.ckpt" if self.checkpoints_dir else None)

    # Keras 2.17 chama esses métodos para decidir se o callback implementa hooks por batch.
    # Habilitamos train_batch_hooks para podermos abortar a época em on_batch_end
    # quando o autotuner detecta loss: inf mid-epoch e escreve um payload de rollback.
    def _implements_train_batch_hooks(self):
        return True  # habilitado para suportar abort mid-epoch via on_batch_end

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False

    def on_batch_end(self, batch, logs=None):
        """Aborta a época imediatamente ao detectar pedido de rollback no tuning_file.

        Fluxo:
          Autotuner detecta loss:inf no stream → escreve payload com \"rollback\" em
          tuning_actions.json → on_batch_end lê o arquivo e seta stop_training=True
          → Keras para após o batch atual → on_epoch_begin da próxima época carrega
          o checkpoint e restaura a configuração → treino continua sem restart.

        O arquivo NÃO é deletado aqui — on_epoch_begin faz isso após carregar o ckpt.
        """
        if self._rollback_abort_pending:
            # Já pedimos abort nesta época; aguardar on_epoch_begin para restaurar.
            return
        payload = self._load_payload()
        if isinstance(payload, dict) and payload.get("rollback"):
            self._rollback_abort_pending = True
            try:
                self.model.stop_training = True
            except Exception:
                pass
            reason = payload["rollback"].get("reason", "?")
            print(
                f"[tuning] batch {batch}: rollback '{reason}' detectado — "
                "abortando época atual. "
                "Checkpoint será restaurado em on_epoch_begin."
            )

    def _load_payload(self):
        if not self.tuning_file or not self.tuning_file.exists():
            return None
        try:
            with self.tuning_file.open() as f:
                return json.load(f)
        except Exception:
            return None

    def on_epoch_begin(self, epoch, logs=None):
        # Resetar flag de abort da época anterior antes de processar o payload.
        # Se _rollback_abort_pending era True, é porque o on_batch_end abortou a
        # época anterior — agora vamos carregar o checkpoint e limpar o estado.
        self._rollback_abort_pending = False
        payload = self._load_payload()
        if not isinstance(payload, dict):
            return
        rb = payload.get("rollback")
        rollback_applied = False

        def _ckpt_prefix_exists(prefix: Path) -> bool:
            return (
                prefix.with_suffix(".index").exists()
                or any(prefix.parent.glob(prefix.name + ".data-*"))
            )

        if isinstance(rb, dict):
            mode = rb.get("mode", "best")
            path = rb.get("path")
            ckpt_path = None
            if path:
                ckpt_path = Path(path)
            elif mode == "last":
                ckpt_path = self.last_path
            else:
                ckpt_path = self.best_path or self.last_path
            # fallback se o path não existir
            if ckpt_path and not _ckpt_prefix_exists(ckpt_path) and self.last_path and _ckpt_prefix_exists(self.last_path):
                ckpt_path = self.last_path
            if ckpt_path and _ckpt_prefix_exists(ckpt_path):
                try:
                    self.model.load_weights(str(ckpt_path))
                    rollback_applied = True
                    if self.verbose:
                        print(f"[tuning] epoch {epoch} rollback -> {ckpt_path.name}")
                except Exception as exc:
                    if self.verbose:
                        print(f"[tuning] rollback falhou: {exc}")
            else:
                # sem checkpoint válido: aborta época para não seguir com NaN
                try:
                    self.model.stop_training = True
                except Exception:
                    pass
                if self.verbose:
                    print(f"[tuning] rollback abortado: checkpoint inexistente ({ckpt_path})")

        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        lr_new = cfg.get("lrate", cfg.get("learning_rate", None))
        if lr_new is not None:
            try:
                # Compatível com otimizadores Keras/TFA
                if hasattr(self.optimizer, "lr"):
                    self.optimizer.lr.assign(float(lr_new))
                elif hasattr(self.optimizer, "learning_rate"):
                    self.optimizer.learning_rate.assign(float(lr_new))
            except Exception:
                pass

        mix_new = cfg.get("mixup_alpha", None)
        if mix_new is not None and self.mixup_var is not None:
            try:
                self.mixup_var.assign(float(mix_new))
            except Exception:
                pass

        clip_new = cfg.get("clip_grad_norm", cfg.get("grad_clip_norm", None))
        if clip_new is not None and hasattr(self.optimizer, "clipnorm"):
            try:
                self.optimizer.clipnorm = float(clip_new)
            except Exception:
                pass

        if self.verbose:
            print(f"[tuning] epoch {epoch} aplicou payload: lr={lr_new} mixup={mix_new} clip={clip_new}")

        # Se houve rollback, reinicializa o otimizador para limpar estados possivelmente corrompidos
        if rollback_applied and hasattr(self.model, "optimizer") and self.model.optimizer is not None:
            try:
                opt_cfg = self.model.optimizer.get_config()
                opt_cls = self.model.optimizer.__class__
                new_opt = opt_cls.from_config(opt_cfg)
                if lr_new is not None:
                    if hasattr(new_opt, "lr"):
                        new_opt.lr = float(lr_new)
                    elif hasattr(new_opt, "learning_rate"):
                        new_opt.learning_rate = float(lr_new)
                self.model.compile(optimizer=new_opt, loss=self.model.loss, metrics=self.model.metrics)
            except Exception as exc:
                if self.verbose:
                    print(f"[tuning] falhou ao resetar optimizer pós-rollback: {exc}")
            # Resetar ReduceLROnPlateau para não re-reduzir imediatamente após rollback.
            # O ReduceLROnPlateau acumulou 'wait' contando épocas de plateau — ao restaurar
            # o modelo para um checkpoint bom, o plateau histórico não é mais válido.
            try:
                cb_container = getattr(self.model, "_callbacks", None)
                cb_list = (
                    cb_container.callbacks
                    if hasattr(cb_container, "callbacks")
                    else (cb_container if isinstance(cb_container, list) else [])
                )
                for _cb_rl in cb_list:
                    # ReduceLROnPlateau tem atributos: wait, cooldown_counter, best, monitor, mode
                    if (hasattr(_cb_rl, "wait") and hasattr(_cb_rl, "cooldown_counter")
                            and hasattr(_cb_rl, "best") and hasattr(_cb_rl, "monitor")):
                        _cb_rl.wait = 0
                        _cb_rl.cooldown_counter = 0
                        # Resetar 'best' para que ele não considere o melhor AUC anterior
                        # como referência (o rollback começa do 0 em relação ao scheduler)
                        _cb_rl.best = -math.inf if getattr(_cb_rl, "mode", "max") == "max" else math.inf
                        if self.verbose:
                            print(f"[tuning] {type(_cb_rl).__name__} reiniciado após rollback")
            except Exception as _exc_rl:
                if self.verbose:
                    print(f"[tuning] ReduceLROnPlateau reset falhou: {_exc_rl}")
        elif not rollback_applied and rb:
            # rollback pedido mas não aplicado: interrompe época para tentar na próxima
            try:
                self.model.stop_training = True
            except Exception:
                pass
            if self.verbose:
                print("[tuning] rollback não aplicado; interrompendo época para re-tentar na próxima.")

        self._last_payload = payload
        # Limpa arquivo para evitar reaplicar lixo antigo
        try:
            if self.tuning_file and self.tuning_file.exists():
                self.tuning_file.unlink()
        except Exception:
            pass


class SaveBestLastCallback(Callback):
    """Salva pesos 'last.ckpt' a cada época e 'best.ckpt' quando val_AUC melhora."""

    def __init__(self, checkpoints_dir: Path, monitor: str = "val_AUC"):
        super().__init__()
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.best_auc = -math.inf
        self.best_path = self.checkpoints_dir / "best.ckpt"
        self.last_path = self.checkpoints_dir / "last.ckpt"
        self.meta_path = self.checkpoints_dir / "best_meta.json"

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        val = logs.get(self.monitor)
        try:
            val_f = float(val)
        except Exception:
            # val_AUC não disponível (época abortada mid-epoch) → não salvar nada
            return
        if math.isnan(val_f) or math.isinf(val_f):
            # Não sobrescrever um checkpoint saudável com pesos de época degradada
            return
        # val_AUC ≤ 0.5001 = classificador aleatório / colapso total
        # (sensibilidade≈0, especificidade≈1.0 — como no exemplo do usuário).
        # Preservar last.ckpt anterior para que o rollback tenha algo útil.
        # best.ckpt só é sobrescrito quando val_f > self.best_auc (nunca em colapso).
        if val_f > 0.5001:
            try:
                self.model.save_weights(str(self.last_path))
            except Exception:
                pass
        if val_f > self.best_auc:
            self.best_auc = val_f
            try:
                self.model.save_weights(str(self.best_path))
                meta = {"epoch": int(epoch), "val_auc": val_f, "path": str(self.best_path)}
                self.meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

    # Keras <=2.17 checks these hooks; return False to avoid batch-level overhead
    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False


class EpochCsvLogger(Callback):
    """Escreve métricas por época em CSV com o mesmo layout usado no pytorch_opt."""

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
        self.train_steps = max(1, int(train_steps))
        self.val_steps = max(1, int(val_steps))
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
        # fallback: case-insensitive lookup
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

        train_seen = self.train_steps * self.global_batch_size
        val_seen = self.val_steps * self.global_batch_size
        train_thpt = train_seen / self._train_elapsed if self._train_elapsed > 0 else None
        val_thpt = val_seen / self._val_elapsed if self._val_elapsed > 0 else None

        mem_peak = logs.get("gpu_mem_peak_mb")
        mem_current = logs.get("gpu_mem_current_mb")
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

    def _implements_train_batch_hooks(self):
        return True

    def _implements_test_batch_hooks(self):
        return True

    def _implements_predict_batch_hooks(self):
        return False

class MixupScheduleCallback(Callback):
    def __init__(self, alpha_ref, base_alpha, total_epochs, freeze_epochs):
        super().__init__()
        self.alpha_ref = alpha_ref
        self.base_alpha = float(base_alpha)
        self.total_epochs = int(total_epochs)
        self.freeze_epochs = int(freeze_epochs)
        self.current_alpha = float(base_alpha)

    def _compute_alpha(self, epoch):
        if self.base_alpha <= 0.0:
            return 0.0
        eff_total = max(1, self.total_epochs - self.freeze_epochs)
        progress = (epoch - self.freeze_epochs) / eff_total
        if progress < 0.0:
            decay = 1.0
        elif progress < 0.6:
            decay = 1.0
        else:
            decay = max(0.0, 1.0 - (progress - 0.6) / 0.4)
        decay = min(max(decay, 0.0), 1.0)
        return self.base_alpha * decay

    def on_epoch_begin(self, epoch, logs=None):
        target_alpha = self._compute_alpha(epoch)
        if self.alpha_ref is not None:
            self.alpha_ref.assign(target_alpha)
        self.current_alpha = target_alpha
        if logs is not None:
            logs["mixup_alpha"] = target_alpha

    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False


class EMACallback(Callback):
    def __init__(self, decay=0.999, start_epoch=0):
        super().__init__()
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.shadow_weights = []
        self._backup = None
        self._current_epoch = 0

    def set_model(self, model):
        super().set_model(model)
        self.rebuild_shadow_variables()

    def rebuild_shadow_variables(self):
        if self.model is None:
            return
        self.shadow_weights = [tf.Variable(w, trainable=False) for w in self.model.trainable_variables]
        self._backup = None

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = int(epoch)

    def on_train_batch_end(self, batch, logs=None):
        if self.decay <= 0.0:
            return
        if self._current_epoch < self.start_epoch:
            return
        if not self.shadow_weights or len(self.shadow_weights) != len(self.model.trainable_variables):
            self.rebuild_shadow_variables()
        for shadow, weight in zip(self.shadow_weights, self.model.trainable_variables):
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * weight)

    def apply_ema_weights(self):
        if not self.shadow_weights:
            return False
        self._backup = [tf.identity(w) for w in self.model.trainable_variables]
        for weight, shadow in zip(self.model.trainable_variables, self.shadow_weights):
            weight.assign(shadow)
        return True

    def restore_original_weights(self):
        if self._backup is None:
            return
        for weight, backup in zip(self.model.trainable_variables, self._backup):
            weight.assign(backup)
        self._backup = None

    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False


def detect_hardware():
    """Detecção enxuta: MultiWorker via TF_CONFIG ou Mirrored/local/CPU."""
    import json

    def _parse_tf_config(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    tf_config = _parse_tf_config(os.environ.get("TF_CONFIG"))
    if tf_config:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print(f"[MWMS] workers={len(tf_config.get('cluster', {}).get('worker', []))} replicas={strategy.num_replicas_in_sync}")
        return strategy

    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[Hardware] MirroredStrategy em {len(gpus)} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print("[Hardware] 1 GPU disponível")
    else:
        strategy = tf.distribute.get_strategy()
        print("[Hardware] CPU (nenhuma GPU detectada)")

    print(f"[Hardware] num_replicas_in_sync = {strategy.num_replicas_in_sync}")
    return strategy


# not using metadata (only image, for now)


def cosine_lr(base_lr, min_lr, epoch_idx, total_epochs, warmup_epochs):
    if total_epochs <= 0:
        return base_lr
    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        scaled = base_lr * float(epoch_idx + 1) / float(max(1, warmup_epochs))
        return max(min_lr, scaled)
    if total_epochs <= warmup_epochs:
        return max(min_lr, base_lr)
    progress = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def parse_fine_tune_schedule(schedule):
    events = []
    if not schedule:
        return events
    for token in schedule.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) < 2:
            raise ValueError(f"Entrada inválida na agenda: '{token}' (esperado epoch:layer[:bn])")
        epoch = int(parts[0])
        layer_idx = int(parts[1])
        bn_flag = None
        if len(parts) >= 3:
            flag = parts[2].strip().lower()
            if flag in ("freeze_bn", "bn_freeze", "freeze"):
                bn_flag = True
            elif flag in ("unfreeze_bn", "bn_unfreeze", "unfreeze"):
                bn_flag = False
            elif flag in ("keep", "auto"):
                bn_flag = None
            else:
                raise ValueError(f"Flag BN desconhecida '{parts[2]}' em '{token}'")
        events.append((epoch, layer_idx, bn_flag))
    events.sort(key=lambda x: x[0])
    return events


def resolve_layer_index(total_layers, idx):
    if total_layers <= 0:
        return 0
    if idx < 0:
        return max(total_layers + idx, 0)
    return min(idx, total_layers)


def set_trainable_from(base_model, idx_start):
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= idx_start)


def set_all_trainable(base_model, flag):
    for layer in base_model.layers:
        layer.trainable = flag


def freeze_batchnorm_layers(base_model):
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def unfreeze_batchnorm_layers(base_model):
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def build_optimizer(learning_rate, weight_decay, clipnorm):
    clip_value = clipnorm if clipnorm and clipnorm > 0 else None
    return tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=weight_decay,
        clipnorm=clip_value
    )


def build_loss_fn(label_smoothing=0.0, focal_gamma=0.0, pos_weight=1.0):
    label_smoothing = float(label_smoothing)
    focal_gamma = float(focal_gamma)
    pos_weight = float(pos_weight)
    base_bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    def _weighted_bce(y_true, y_pred):
        weights = y_true * pos_weight + (1.0 - y_true)
        return base_bce(y_true, y_pred, sample_weight=weights)

    def _focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        weights = y_true * pos_weight + (1.0 - y_true)
        ce = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, label_smoothing=label_smoothing
        )
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        modulating = tf.pow(1.0 - p_t, focal_gamma)
        return tf.reduce_mean(modulating * ce * weights)

    if focal_gamma > 0.0:
        return _focal_loss
    if pos_weight != 1.0:
        return _weighted_bce
    return base_bce


def sample_beta(alpha):
    alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)
    def _default():
        return tf.constant(1.0, dtype=tf.float32)
    def _sample():
        gamma1 = tf.random.gamma([], alpha_tensor, 1.0)
        gamma2 = tf.random.gamma([], alpha_tensor, 1.0)
        return gamma1 / (gamma1 + gamma2)
    return tf.cond(alpha_tensor <= 0.0, _default, _sample)


def make_cosine_scheduler(base_lr, total_epochs, warmup_epochs, min_lr, offset=0):
    def schedule(epoch, _current_lr):
        phase_epoch = epoch - offset
        return cosine_lr(base_lr, min_lr, phase_epoch, total_epochs, warmup_epochs)
    return LearningRateScheduler(schedule, verbose=0)

def read_labeled_tfrecord(example, __return_only_label):
    LABELED_TFREC_FORMAT = {
        "imagem": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        # 'image_name': tf.io.FixedLenFeature([], tf.string),
        'retinopatia' : tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['imagem'])
    label = tf.cast(example['retinopatia'], tf.int32)
    # name = example['image_name']

    # return image, label, name
    return image, label

def read_unlabeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "imagem": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['imagem'])

    return image

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    if not USE_UINT8_DECODE:
        image = tf.cast(image, tf.float32)
    return image


def normalize_image(image, mode, preprocess_fn):
    image = tf.cast(image, tf.float32)
    if mode == 'preprocess' and preprocess_fn is not None:
        image = preprocess_fn(image)
    if mode == 'unit':
        image = image / 255.0
    return to_compute_dtype(image)


def maybe_fundus_crop(image, ratio):
    ratio_tensor = tf.convert_to_tensor(ratio, dtype=tf.float32)

    def _no_crop():
        return to_compute_dtype(image)

    def _do_crop():
        cropped = tf.image.central_crop(image, ratio_tensor)
        resized = tf.image.resize(cropped, IMAGE_SIZE, method='bilinear')
        return to_compute_dtype(resized)

    return tf.cond(ratio_tensor >= 0.999, _no_crop, _do_crop)


def augment_image(image, enable):
    if not enable:
        return image
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.cond(tf.random.uniform([]) < 0.15, lambda: tf.image.flip_up_down(image), lambda: image)
    def adjust_image(img, fn, low, high):
        factor = tf.random.uniform([], low, high)
        return fn(img, factor)
    image = tf.cond(
        tf.random.uniform([]) < 0.35,
        lambda: tf.clip_by_value(adjust_image(image, lambda img, f: img * f, 0.85, 1.15), 0.0, 255.0),
        lambda: image
    )
    image = tf.cond(
        tf.random.uniform([]) < 0.35,
        lambda: tf.clip_by_value(tf.image.adjust_contrast(image, tf.random.uniform([], 0.85, 1.15)), 0.0, 255.0),
        lambda: image
    )
    image = tf.cond(
        tf.random.uniform([]) < 0.3,
        lambda: tf.clip_by_value(tf.image.adjust_saturation(image, tf.random.uniform([], 0.85, 1.15)), 0.0, 255.0),
        lambda: image
    )
    return to_compute_dtype(image)


def apply_mixup(images, labels, alpha):
    alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)

    def _no_mix():
        return to_compute_dtype(images), tf.cast(labels, tf.float32)

    def _do_mix():
        batch_size = tf.shape(images)[0]
        shuffle_indices = tf.random.shuffle(tf.range(batch_size))
        base_images = to_compute_dtype(images)
        lam = tf.cast(sample_beta(alpha_tensor), COMPUTE_DTYPE)
        mixed_images = lam * base_images + (1.0 - lam) * tf.cast(tf.gather(base_images, shuffle_indices), COMPUTE_DTYPE)
        labels_float = tf.cast(labels, tf.float32)
        shuffled_labels = tf.gather(labels_float, shuffle_indices)
        lam_labels = tf.cast(lam, tf.float32)
        mixed_labels = lam_labels * labels_float + (1.0 - lam_labels) * shuffled_labels
        return mixed_images, mixed_labels

    return tf.cond(alpha_tensor <= 0.0, _no_mix, _do_mix)


def apply_cutmix(images, labels, alpha):
    alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)

    def _no_cutmix():
        return to_compute_dtype(images), tf.cast(labels, tf.float32)

    def _do_cutmix():
        images_f = to_compute_dtype(images)
        labels_f = tf.cast(labels, tf.float32)
        batch_size = tf.shape(images_f)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images_f, indices)
        shuffled_labels = tf.gather(labels_f, indices)

        lam = tf.cast(sample_beta(alpha_tensor), COMPUTE_DTYPE)
        img_height = tf.shape(images_f)[1]
        img_width = tf.shape(images_f)[2]
        # ensure geometry math happens in float32 to avoid dtype mismatches with float16 compute
        cut_ratio = tf.cast(tf.math.sqrt(tf.cast(1.0 - lam, tf.float32)), tf.float32)
        img_width_f = tf.cast(img_width, tf.float32)
        img_height_f = tf.cast(img_height, tf.float32)
        cut_w = tf.maximum(tf.cast(cut_ratio * img_width_f, tf.int32), 1)
        cut_h = tf.maximum(tf.cast(cut_ratio * img_height_f, tf.int32), 1)

        cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
        cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height)
        x2 = tf.clip_by_value(x1 + cut_w, 0, img_width)
        y2 = tf.clip_by_value(y1 + cut_h, 0, img_height)

        patch_shape = tf.stack([batch_size, y2 - y1, x2 - x1, tf.shape(images_f)[-1]])
        cutout = tf.ones(patch_shape, dtype=images_f.dtype)
        cutout = tf.pad(
            cutout,
            paddings=[
                [0, 0],
                [y1, img_height - y2],
                [x1, img_width - x2],
                [0, 0],
            ],
            constant_values=0.0,
        )

        mask = tf.ones_like(images_f) - cutout
        mixed_images = images_f * mask + shuffled_images * cutout
        area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
        lam_adjusted = 1.0 - (area / (img_width_f * img_height_f))
        mixed_labels = lam_adjusted * labels_f + (1.0 - lam_adjusted) * shuffled_labels
        return mixed_images, mixed_labels

    return tf.cond(alpha_tensor <= 0.0, _no_cutmix, _do_cutmix)


def apply_batch_augmentations(images, labels, mixup_alpha, cutmix_alpha):
    mix_alpha = tf.convert_to_tensor(mixup_alpha if mixup_alpha is not None else 0.0, dtype=tf.float32)
    cut_alpha = tf.convert_to_tensor(cutmix_alpha if cutmix_alpha is not None else 0.0, dtype=tf.float32)

    def _do_mix():
        return apply_mixup(images, labels, mix_alpha)

    def _do_cut():
        return apply_cutmix(images, labels, cut_alpha)

    def _pick_one():
        return tf.cond(tf.random.uniform([]) < 0.5, _do_cut, _do_mix)

    return tf.cond(
        cut_alpha > 0.0,
        lambda: tf.cond(mix_alpha > 0.0, _pick_one, _do_cut),
        lambda: tf.cond(mix_alpha > 0.0, _do_mix, lambda: (to_compute_dtype(images), tf.cast(labels, tf.float32))),
    )

# count # of images in files.. (embedded in file name)
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\."
                        ).search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def infer_idx_path(tfrec_path):
    path = Path(tfrec_path)
    cand1 = path.with_suffix(".idx")
    cand2 = Path(str(path) + ".idx")
    if cand2.exists():
        return str(cand2)
    if cand1.exists():
        return str(cand1)
    raise FileNotFoundError(f"Index file (.idx) not found for TFRecord: {tfrec_path}")


def build_idx_list(tfrec_paths):
    return [infer_idx_path(p) for p in tfrec_paths]

def load_dataset(filenames, labeled=True, ordered=False, return_only_label=False, dataset_name="train"):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False  # disable order, increase speed
    options.experimental_distribute.auto_shard_policy = AUTO_SHARD_POLICY

    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    )

    if CACHE_BASE_DIR:
        cache_dir = Path(CACHE_BASE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{dataset_name}.cache"
        dataset = dataset.cache(str(cache_path))

    dataset = dataset.with_options(options)
    dataset = dataset.map(
        lambda example: read_labeled_tfrecord(example, __return_only_label=return_only_label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # returns a dataset of (image, labels) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_training_dataset(
    filenames,
    _return_only_label=False,
    *,
    augment_flag=False,
    normalize_mode='raw255',
    preprocess_fn=None,
    mixup_alpha_ref=None,
    cutmix_alpha_ref=None,
    fundus_crop_ratio=1.0
):
    dataset = load_dataset(
        filenames,
        labeled=True,
        return_only_label=_return_only_label,
        dataset_name="train"
    )
    if fundus_crop_ratio < 0.999:
        dataset = dataset.map(
            lambda image, label: (maybe_fundus_crop(image, fundus_crop_ratio), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    if augment_flag:
        dataset = dataset.map(
            lambda image, label: (augment_image(image, augment_flag), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    if preprocess_fn is not None or normalize_mode != 'raw255':
        dataset = dataset.map(
            lambda image, label: (normalize_image(image, normalize_mode, preprocess_fn), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(GLOBAL_BATCH_SIZE)
    if mixup_alpha_ref is not None or cutmix_alpha_ref is not None:
        dataset = dataset.map(
            lambda images, labels: apply_batch_augmentations(images, labels, mixup_alpha_ref, cutmix_alpha_ref),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_valid_dataset(
    filenames,
    _return_only_label=False,
    dataset_name="valid",
    *,
    normalize_mode='raw255',
    preprocess_fn=None,
    fundus_crop_ratio=1.0
):
    dataset = load_dataset(
        filenames,
        labeled=True,
        ordered=True,
        return_only_label=_return_only_label,
        dataset_name=dataset_name
    )
    if fundus_crop_ratio < 0.999:
        dataset = dataset.map(
            lambda image, label: (maybe_fundus_crop(image, fundus_crop_ratio), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    if preprocess_fn is not None or normalize_mode != 'raw255':
        dataset = dataset.map(
            lambda image, label: (normalize_image(image, normalize_mode, preprocess_fn), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    dataset = dataset.batch(GLOBAL_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_test_dataset(
    filenames,
    *,
    normalize_mode='raw255',
    preprocess_fn=None,
    fundus_crop_ratio=1.0
):
    dataset = load_dataset(
        filenames,
        labeled=True,
        ordered=True,
        dataset_name="test"
    )
    if fundus_crop_ratio < 0.999:
        dataset = dataset.map(
            lambda image, label: (maybe_fundus_crop(image, fundus_crop_ratio), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    if preprocess_fn is not None or normalize_mode != 'raw255':
        dataset = dataset.map(
            lambda image, label: (normalize_image(image, normalize_mode, preprocess_fn), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    dataset = dataset.batch(GLOBAL_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def apply_tta_view(images, view_idx):
    if view_idx == 0:
        return images
    mode = view_idx % 4
    if mode == 1:
        return tf.image.flip_left_right(images)
    if mode == 2:
        return tf.image.flip_up_down(images)
    if mode == 3:
        return tf.image.rot90(images, k=1)
    return images


def predict_with_tta(model, dataset, tta_views):
    if tta_views <= 1:
        labels = np.concatenate([y.numpy() for _, y in dataset], axis=0)
        probs = model.predict(dataset, verbose=0).ravel()
        return labels, probs

    probs_chunks = []
    labels_chunks = []
    for batch_images, batch_labels in dataset:
        batch_acc = None
        for view_idx in range(tta_views):
            view_images = apply_tta_view(batch_images, view_idx)
            preds = model(view_images, training=False)
            preds = tf.cast(preds, tf.float32)
            if batch_acc is None:
                batch_acc = preds
            else:
                batch_acc += preds
        avg_preds = batch_acc / float(tta_views)
        probs_chunks.append(avg_preds.numpy())
        labels_chunks.append(batch_labels.numpy())
    labels = np.concatenate(labels_chunks, axis=0)
    probs = np.concatenate([chunk.ravel() for chunk in probs_chunks], axis=0)
    return labels, probs

class Sensitivity(tf.keras.metrics.Metric):
    """Sensibilidade (recall) com threshold fixo em 0.5."""

    def __init__(self, name="sensitivity", **kwargs):
        super().__init__(name=name, **kwargs)
        # Usa agregação em soma e sincronização on_write para evitar SyncOnRead em estratégias distribuídas.
        self.tp = self.add_weight(
            name="tp",
            initializer="zeros",
            aggregation=tf.VariableAggregation.SUM,
            synchronization=tf.VariableSynchronization.ON_WRITE,
        )
        self.fn = self.add_weight(
            name="fn",
            initializer="zeros",
            aggregation=tf.VariableAggregation.SUM,
            synchronization=tf.VariableSynchronization.ON_WRITE,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            tp = tf.multiply(tp, tf.reduce_mean(sample_weight))
            fn = tf.multiply(fn, tf.reduce_mean(sample_weight))
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        denom = self.tp + self.fn + tf.keras.backend.epsilon()
        return self.tp / denom

    def reset_states(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


class Specificity(tf.keras.metrics.Metric):
    """Especificidade (true negative rate) com threshold 0.5."""

    def __init__(self, name="specificity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = self.add_weight(
            name="tn",
            initializer="zeros",
            aggregation=tf.VariableAggregation.SUM,
            synchronization=tf.VariableSynchronization.ON_WRITE,
        )
        self.fp = self.add_weight(
            name="fp",
            initializer="zeros",
            aggregation=tf.VariableAggregation.SUM,
            synchronization=tf.VariableSynchronization.ON_WRITE,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            tn = tf.multiply(tn, tf.reduce_mean(sample_weight))
            fp = tf.multiply(fp, tf.reduce_mean(sample_weight))
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)

    def result(self):
        denom = self.tn + self.fp + tf.keras.backend.epsilon()
        return self.tn / denom

    def reset_states(self):
        self.tn.assign(0.0)
        self.fp.assign(0.0)



def generate_thresholds(num_thresholds, kepsilon=1e-7):
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds -1) for i in range(num_thresholds -2)
    ]
    return [0.0] + thresholds + [1.0]

def build_model(thresholds, dim=299, recompute_backbone=False):
    input_shape = (dim, dim, 3)
    inp = tf.keras.layers.Input(shape=input_shape)

    base = create_model(model_name, input_shape)
    if base is None:
        raise ValueError(f"Unknown model name: {model_name}")

    if recompute_backbone:
        def _recompute_layer(x):
            # Wrap recompute_grad in a Lambda to keep Keras symbolic tensors happy.
            return tf.recompute_grad(lambda y: base(y, training=True))(x)
        x = tf.keras.layers.Lambda(_recompute_layer, name="recompute_backbone")(inp)
    else:
        x = base(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=inp, outputs=x)
    for layer in base.layers:
        layer.trainable = False

    model.base_model = base

    return model


def compile_model(model, learning_rate, weight_decay, clipnorm, strategy=None, label_smoothing=0.0, jit_compile=False, focal_gamma=0.0, pos_weight=1.0):
    def _compile():
        opt = build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
        loss = build_loss_fn(label_smoothing=label_smoothing, focal_gamma=focal_gamma, pos_weight=pos_weight)

        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='AUC'),
            Sensitivity(name='sensitivity'),
            Specificity(name='specificity')
        ]

        compile_kwargs = dict(optimizer=opt, loss=loss, metrics=metrics)
        if jit_compile:
            try:
                compile_kwargs["jit_compile"] = True
            except Exception:
                pass
        model.compile(**compile_kwargs)

    if strategy is not None:
        with strategy.scope():
            _compile()
    else:
        _compile()


def configure_mixed_precision(strategy, enable=True):
    if not enable:
        print("[Mixed Precision] Desativada por configuração.")
        return None
    policy = None
    if isinstance(strategy, tf.distribute.TPUStrategy):
        policy = 'mixed_bfloat16'
    elif tf.config.list_physical_devices('GPU'):
        policy = 'mixed_float16'

    if policy:
        current_policy = mixed_precision.global_policy()
        if current_policy.name != policy:
            mixed_precision.set_global_policy(policy)
        print(f"[Mixed Precision] Policy '{policy}' habilitada.")
        return policy

    print("[Mixed Precision] Política padrão mantida.")
    return None


if __name__ == "__main__":
    print(f"tensorflow version: {tf.__version__}")
    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    except Exception:
        pass

    args = parse_args()
    TUNING_FILE = os.environ.get("HCPA_TUNING_FILE", "").strip()
    cores = max(0, int(getattr(args, "cores", 0)))
    if cores > 0:
        os.environ["OMP_NUM_THREADS"] = str(cores)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(cores)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, cores // 2 or 1))
        except Exception as exc:
            print(f"[Hardware] Aviso ao configurar threads: {exc}")

    gpus_detected = tf.config.list_physical_devices('GPU')
    if not gpus_detected:
        raise SystemExit("[Hardware] Execução requer GPU, mas nenhuma foi detectada.")
    print(f"[Hardware] GPUs visíveis: {len(gpus_detected)}")
    if cores > 0:
        print(f"[Hardware] OMP_NUM_THREADS configurado para {cores} via --cores.")
    try:
        for gpu in gpus_detected:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        if args.verbose:
            print(f"[WARN] memory_growth não pôde ser habilitado: {exc}")
    # Accessing arguments
    TFREC_DIR = args.tfrec_dir
    dataset = args.dataset
    results = args.results
    exec = args.exec
    IMG_SIZES = args.img_sizes
    # tune it, depende de imagem/tamanho/hardware
    PER_REPLICA_BS = int(args.batch_size)
    EPOCHS = args.epochs
    NUM_CLASSES = args.num_classes
    lrate = args.lrate
    num_thresholds = args.num_thresholds
    wait_epochs = args.wait_epochs
    freeze_epochs = max(0, min(args.freeze_epochs, EPOCHS))
    fine_tune_lr_factor = max(args.fine_tune_lr_factor, 1e-6)
    fine_tune_at = args.fine_tune_at
    fine_tune_lr_absolute = args.fine_tune_lr
    scheduler_mode = args.scheduler.lower()
    warmup_epochs = max(0, args.warmup_epochs)
    min_lr = max(1e-8, args.min_lr)
    grad_clip_norm = max(0.0, args.grad_clip_norm)
    mixup_alpha = max(0.0, args.mixup_alpha)
    cutmix_alpha = max(0.0, args.cutmix_alpha)
    label_smoothing = max(0.0, min(float(args.label_smoothing), 0.2))
    focal_gamma = max(0.0, float(args.focal_gamma))
    pos_weight = float(args.pos_weight)
    if pos_weight <= 0.0:
        pos_weight = 1.0
    fundus_crop_ratio = float(args.fundus_crop_ratio)
    if fundus_crop_ratio <= 0.0:
        fundus_crop_ratio = 1.0
    fundus_crop_ratio = min(fundus_crop_ratio, 1.0)
    freeze_bn_flag = args.freeze_bn
    model_name = args.model
    normalize_mode = args.normalize
    TTA_VIEWS = max(1, int(args.tta_views))
    use_dali = False
    if getattr(args, "use_dali", False):
        print("[INFO] suporte DALI desativado nesta versão otimizada; usando tf.data padrão.")
    dali_threads = 0
    dali_layout = "NHWC"
    dali_seed = 0
    recompute_backbone = False
    if getattr(args, "recompute_backbone", False):
        print("[INFO] recompute_grad desativado nesta versão otimizada.")
    jit_compile_flag = bool(args.jit_compile)
    auc_target = float(args.auc_target)
    if args.channels_last:
        tf.keras.backend.set_image_data_format('channels_last')
    USE_UINT8_DECODE = bool(args.h2d_uint8)
    preprocess_fn = get_preprocess_fn(model_name) if normalize_mode == 'preprocess' else None
    if normalize_mode == 'preprocess' and preprocess_fn is None:
        print(f"[WARN] preprocess_input não disponível para '{model_name}'. Usando 'raw255'.")
        normalize_mode = 'raw255'
    if use_dali:
        print("[INFO] Caminho DALI ignorado; use tf.data (normalize_mode será respeitado).")

    HEAD_WEIGHT_DECAY = 1e-4
    FINE_TUNE_WEIGHT_DECAY = 1e-5

    if mixup_alpha > 0 and (label_smoothing > 0 or focal_gamma > 0 or cutmix_alpha > 0):
        print("[WARN] mixup/cutmix com label_smoothing/focal podem interagir; revise se ganhos diminuírem.")

    IMAGE_SIZE = [IMG_SIZES, IMG_SIZES]
    kepsilon = 1e-7
    # constant to customize output
    SHOW_FILES = args.show_files
    VERBOSE = args.verbose
    '''
    Create folder for output.
    '''
    results_path = Path(results)
    target_dir = results_path.resolve() if results_path.is_symlink() else results_path
    ready_marker = target_dir / ".ready"
    primary_worker = is_primary_worker()

    if primary_worker:
        if ready_marker.exists():
            try:
                ready_marker.unlink()
            except OSError:
                pass
        if target_dir.exists():
            if not SKIP_INTERNAL_CLEANUP:
                if target_dir.is_dir():
                    rmtree(target_dir)
                else:
                    target_dir.unlink()
                target_dir.mkdir(parents=True, exist_ok=True)
            elif not target_dir.is_dir():
                raise RuntimeError(f"Resultados esperados como diretório: {target_dir}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
        results_dir = target_dir
    else:
        wait_start = time.time()
        timeout_seconds = 300
        while True:
            if target_dir.exists() and ready_marker.exists():
                break
            if time.time() - wait_start > timeout_seconds:
                break
            time.sleep(0.5)
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
        results_dir = target_dir

    rank_id = get_worker_rank()
    tmp_root = results_dir / "tmp"
    rank_tmp = tmp_root / f"worker_{rank_id}"
    rank_tmp.mkdir(parents=True, exist_ok=True)
    mpl_root = results_dir / "mplconfig"
    rank_mpl = mpl_root / f"worker_{rank_id}"
    rank_mpl.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    if primary_worker:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for env_name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[env_name] = str(rank_tmp)
    os.environ["MPLCONFIGDIR"] = str(rank_mpl)
    tempfile.tempdir = str(rank_tmp)

    if primary_worker:
        try:
            ready_marker.write_text("ready\n")
        except OSError:
            # Best effort: other workers already created it.
            ready_marker.touch(exist_ok=True)

    cache_dir_arg = args.cache_dir.strip()
    if cache_dir_arg.lower() == 'none':
        CACHE_BASE_DIR = None
    else:
        cache_root = Path(cache_dir_arg) if cache_dir_arg else Path(results) / 'tfdata_cache'
        cache_root.mkdir(parents=True, exist_ok=True)
        CACHE_BASE_DIR = str(cache_root)

    if CACHE_BASE_DIR:
        worker_id = None
        worker_type = None
        tf_config_raw = os.environ.get('TF_CONFIG')
        if tf_config_raw:
            try:
                tf_config = json.loads(tf_config_raw)
                task_cfg = tf_config.get('task', {}) or {}
                worker_id = task_cfg.get('index')
                worker_type = task_cfg.get('type')
            except Exception:
                worker_id = None
        if worker_id is None:
            proc = os.environ.get('SLURM_PROCID')
            if proc is not None:
                try:
                    worker_id = int(proc)
                except ValueError:
                    worker_id = None
        if worker_id is not None:
            subdir = f"worker_{worker_type}_{worker_id}" if worker_type else f"worker_{worker_id}"
            cache_root = Path(CACHE_BASE_DIR) / subdir
            cache_root.mkdir(parents=True, exist_ok=True)
            CACHE_BASE_DIR = str(cache_root)

    strategy = detect_hardware()
    REPLICAS = strategy.num_replicas_in_sync
    DISTRIBUTION_STRATEGY = strategy
    configure_mixed_precision(strategy, enable=args.mixed_precision)
    globals()["COMPUTE_DTYPE"] = tf.as_dtype(mixed_precision.global_policy().compute_dtype)

    print(f'REPLICAS: {REPLICAS}')
    GLOBAL_BATCH_SIZE = PER_REPLICA_BS * REPLICAS
    if GLOBAL_BATCH_SIZE <= 0:
        raise SystemExit("Batch size deve ser positivo.")
    BATCH_SIZE = PER_REPLICA_BS  # por réplica
    print(f'Batch global {GLOBAL_BATCH_SIZE} (por réplica: {BATCH_SIZE})')

    thresholds = generate_thresholds(num_thresholds, kepsilon)

    # for others investigations we store all the history
    histories = []

    # these will be split in folds
    num_total_train_files = len(tf.io.gfile.glob(TFREC_DIR + '/train*.tfrec'))
    num_total_valid_files = len(tf.io.gfile.glob(TFREC_DIR + '/test*.tfrec'))

    print('#### Image Size %i, batch_size global %i (por réplica %i)'%
        (IMG_SIZES, GLOBAL_BATCH_SIZE, BATCH_SIZE))
    print('#### Epochs: %i' %(EPOCHS))

    # CREATE TRAIN AND VALIDATION SUBSETS
    TRAINING_FILENAMES = tf.io.gfile.glob(TFREC_DIR + '/train*.tfrec')
    VALID_FILENAMES = tf.io.gfile.glob(TFREC_DIR + '/test*.tfrec')
    print('Train TFRecord files', len(TRAINING_FILENAMES))
    print('Train TFRecord files', len(VALID_FILENAMES))

    train_image_count = count_data_items(TRAINING_FILENAMES) if TRAINING_FILENAMES else 0
    valid_image_count = count_data_items(VALID_FILENAMES) if VALID_FILENAMES else 0
    if SHOW_FILES:
        print('Number of training images', train_image_count)
        print('Number of validation images', valid_image_count)
    if train_image_count <= 0 or valid_image_count <= 0:
        raise SystemExit("Nenhum arquivo TFRecord de treino/validação encontrado.")
    train_steps = max(1, math.ceil(train_image_count / GLOBAL_BATCH_SIZE))
    valid_steps = max(1, math.ceil(valid_image_count / GLOBAL_BATCH_SIZE))

    K.clear_session()

    print('#### ' + model_name  + ' in execution number ', exec)
    with strategy.scope():
        model = build_model(thresholds, IMG_SIZES, recompute_backbone=recompute_backbone)
        compile_model(
            model,
            lrate,
            HEAD_WEIGHT_DECAY,
            grad_clip_norm,
            strategy=None,
            label_smoothing=label_smoothing,
            jit_compile=jit_compile_flag,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight
        )
        model.summary()

    mixup_alpha_var = tf.Variable(mixup_alpha, dtype=tf.float32, trainable=False, name="mixup_alpha") if mixup_alpha > 0.0 else None
    cutmix_alpha_const = tf.constant(cutmix_alpha, dtype=tf.float32) if cutmix_alpha > 0.0 else None
    mixup_schedule_cb = MixupScheduleCallback(mixup_alpha_var, mixup_alpha, EPOCHS, freeze_epochs) if mixup_alpha_var is not None else None
    ema_start_epoch = max(0, int(0.6 * EPOCHS))
    ema_callback = EMACallback(decay=0.999, start_epoch=ema_start_epoch)
    ema_callback.set_model(model)

    if use_dali:
        train_idx_files = build_idx_list(TRAINING_FILENAMES)
        valid_idx_files = build_idx_list(VALID_FILENAMES)
        if VERBOSE:
            print(f"[DALI] threads={dali_threads} layout={dali_layout} seed={dali_seed}")
        train_dataset = build_dali_tf_dataset(
            TRAINING_FILENAMES,
            train_idx_files,
            batch_size=GLOBAL_BATCH_SIZE,
            image_size=IMG_SIZES,
            augment=args.augment,
            layout=dali_layout,
            seed=dali_seed,
            threads=dali_threads,
            shard_id=0,
            num_shards=1,
            fundus_crop_ratio=fundus_crop_ratio,
        )
        valid_dataset = build_dali_tf_dataset(
            VALID_FILENAMES,
            valid_idx_files,
            batch_size=GLOBAL_BATCH_SIZE,
            image_size=IMG_SIZES,
            augment=False,
            layout=dali_layout,
            seed=dali_seed,
            threads=max(1, dali_threads // 2),
            shard_id=0,
            num_shards=1,
            fundus_crop_ratio=fundus_crop_ratio,
        )
        if dali_layout == "NCHW":
            transpose_perm = [0, 2, 3, 1]
            train_dataset = train_dataset.map(
                lambda images, labels: (tf.transpose(images, perm=transpose_perm), labels),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            valid_dataset = valid_dataset.map(
                lambda images, labels: (tf.transpose(images, perm=transpose_perm), labels),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        if mixup_alpha_var is not None or cutmix_alpha_const is not None:
            train_dataset = train_dataset.map(
                lambda images, labels: apply_batch_augmentations(images, labels, mixup_alpha_var, cutmix_alpha_const),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        train_dataset = get_training_dataset(
            TRAINING_FILENAMES,
            augment_flag=args.augment,
            normalize_mode=normalize_mode,
            preprocess_fn=preprocess_fn,
            mixup_alpha_ref=mixup_alpha_var if mixup_alpha_var is not None else None,
            cutmix_alpha_ref=cutmix_alpha_const if cutmix_alpha_const is not None else None,
            fundus_crop_ratio=fundus_crop_ratio
        )
        valid_dataset = get_valid_dataset(
            VALID_FILENAMES,
            normalize_mode=normalize_mode,
            preprocess_fn=preprocess_fn,
            fundus_crop_ratio=fundus_crop_ratio
        )

    scalar_metrics_cb = EnsureScalarMetrics(strategy)
    gpu_memory_cb = GpuMemoryTracker()
    speed_logger = TrainingSpeedLogger(
        steps_per_epoch=train_steps,
        global_batch_size=GLOBAL_BATCH_SIZE,
        target_auc=auc_target,
        results_dir=results_dir,
        primary_worker=primary_worker
    )
    save_ckpt_cb = SaveBestLastCallback(checkpoints_dir) if primary_worker else None

    csv_logger_path = results + '/' + model_name + '-%i.csv' % exec
    csv_logger_stage1 = EpochCsvLogger(
        csv_logger_path,
        stage="freeze",
        train_steps=train_steps,
        val_steps=valid_steps,
        global_batch_size=GLOBAL_BATCH_SIZE,
        append=False,
    )

    tStart = time.time()
    history_stage1 = None

    def next_epoch_from_history(history_obj, fallback_epoch):
        if history_obj is not None:
            epochs_run = getattr(history_obj, "epoch", None)
            if epochs_run:
                try:
                    return max(epochs_run) + 1
                except Exception:
                    return fallback_epoch
        return fallback_epoch

    if freeze_epochs > 0:
        tuning_cb_stage1 = TuningFileCallback(TUNING_FILE, optimizer=model.optimizer, mixup_var=mixup_alpha_var, checkpoints_dir=checkpoints_dir, verbose=bool(args.verbose)) if TUNING_FILE else None
        callbacks_stage1 = [scalar_metrics_cb, gpu_memory_cb, csv_logger_stage1, ema_callback, speed_logger]
        if save_ckpt_cb is not None:
            callbacks_stage1.append(save_ckpt_cb)
        if tuning_cb_stage1 is not None:
            callbacks_stage1.append(tuning_cb_stage1)
        if mixup_schedule_cb is not None:
            callbacks_stage1.append(mixup_schedule_cb)
        if scheduler_mode == 'cosine':
            callbacks_stage1.append(
                make_cosine_scheduler(
                    base_lr=lrate,
                    total_epochs=freeze_epochs,
                    warmup_epochs=warmup_epochs,
                    min_lr=min_lr,
                    offset=0
                )
            )
        elif scheduler_mode == 'plateau':
            callbacks_stage1.append(
                ReduceLROnPlateau(
                    monitor='val_AUC',
                    mode='max',
                    factor=0.2,
                    patience=wait_epochs,
                    min_delta=1e-4,
                    cooldown=0,
                    min_lr=min_lr,
                    verbose=1 if VERBOSE else 0
                )
            )
        history_stage1 = model.fit(
            train_dataset,
            epochs=freeze_epochs,
            callbacks=callbacks_stage1,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            verbose=VERBOSE,
        )
        histories.append(history_stage1)

    current_epoch = next_epoch_from_history(history_stage1, freeze_epochs if freeze_epochs > 0 else 0)

    try:
        schedule_events = parse_fine_tune_schedule(args.fine_tune_schedule)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if schedule_events:
        filtered = [(e, l, b) for (e, l, b) in schedule_events if e <= EPOCHS]
        if len(filtered) != len(schedule_events) and VERBOSE:
            print(f"[fine-tune] Eventos além de {EPOCHS} épocas foram descartados.")
        schedule_events = filtered

    if current_epoch < EPOCHS:
        base_model = getattr(model, 'base_model', None)
        total_layers = len(base_model.layers) if base_model is not None else 0
        initial_start = resolve_layer_index(total_layers, fine_tune_at) if base_model is not None else 0
        ft_state = {
            "start": initial_start,
            "freeze_bn": freeze_bn_flag
        }

        def apply_finetune_settings(start_idx=None, bn_flag=None, announce=True):
            if base_model is None:
                return
            if start_idx is not None:
                resolved = resolve_layer_index(total_layers, start_idx)
                ft_state["start"] = resolved
                if resolved <= 0:
                    set_all_trainable(base_model, True)
                elif resolved >= total_layers:
                    set_all_trainable(base_model, False)
                else:
                    set_trainable_from(base_model, resolved)
            if bn_flag is not None:
                ft_state["freeze_bn"] = bool(bn_flag)
            if ft_state["freeze_bn"]:
                freeze_batchnorm_layers(base_model)
            else:
                unfreeze_batchnorm_layers(base_model)
            if announce and VERBOSE and base_model is not None:
                trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
                print(f"[fine-tune] lr will be set after compile | start_layer={ft_state['start']}  trainable_layers={trainable_layers}/{total_layers}  freeze_bn={ft_state['freeze_bn']}")

        apply_finetune_settings(start_idx=ft_state["start"], bn_flag=ft_state["freeze_bn"], announce=True)

        fine_tune_lr = fine_tune_lr_absolute if fine_tune_lr_absolute > 0 else max(lrate * fine_tune_lr_factor, 1e-6)
        compile_model(
            model,
            fine_tune_lr,
            FINE_TUNE_WEIGHT_DECAY,
            grad_clip_norm,
            strategy=getattr(model, 'distribute_strategy', None),
            label_smoothing=label_smoothing,
            jit_compile=jit_compile_flag,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight
        )
        ema_callback.rebuild_shadow_variables()
        if VERBOSE and base_model is not None:
            trainable_layers = sum(1 for layer in base_model.layers if layer.trainable)
            print(f"[fine-tune] lr={fine_tune_lr:.4e}  trainable_layers={trainable_layers}/{total_layers}  start_layer={ft_state['start']}  freeze_bn={ft_state['freeze_bn']}")


        csv_logger_stage2 = EpochCsvLogger(
            csv_logger_path,
            stage="finetune",
            train_steps=train_steps,
            val_steps=valid_steps,
            global_batch_size=GLOBAL_BATCH_SIZE,
            append=True,
        )
        if scheduler_mode == 'cosine':
            scheduler_cb = make_cosine_scheduler(
                base_lr=fine_tune_lr,
                total_epochs=max(EPOCHS - freeze_epochs, 1),
                warmup_epochs=warmup_epochs,
                min_lr=min_lr,
                offset=freeze_epochs
            )
        else:
            scheduler_cb = ReduceLROnPlateau(
                monitor='val_AUC',
                mode='max',
                factor=0.2,
                patience=wait_epochs,
                min_delta=1e-4,
                cooldown=0,
                min_lr=min_lr,
                verbose=1 if VERBOSE else 0
            )

        tuning_cb_stage2 = TuningFileCallback(TUNING_FILE, optimizer=model.optimizer, mixup_var=mixup_alpha_var, checkpoints_dir=checkpoints_dir, verbose=bool(args.verbose)) if TUNING_FILE else None
        stage2_callbacks_base = [scalar_metrics_cb, gpu_memory_cb, csv_logger_stage2, scheduler_cb, ema_callback, speed_logger]
        if save_ckpt_cb is not None:
            stage2_callbacks_base.append(save_ckpt_cb)
        if tuning_cb_stage2 is not None:
            stage2_callbacks_base.append(tuning_cb_stage2)
        if mixup_schedule_cb is not None:
            stage2_callbacks_base.append(mixup_schedule_cb)
        histories_stage2 = []

        while current_epoch < EPOCHS:
            while schedule_events and schedule_events[0][0] <= current_epoch:
                _, layer_idx, bn_flag = schedule_events.pop(0)
                apply_finetune_settings(start_idx=layer_idx, bn_flag=bn_flag, announce=True)
                compile_model(
                    model,
                    fine_tune_lr,
                    FINE_TUNE_WEIGHT_DECAY,
                    grad_clip_norm,
                    strategy=getattr(model, 'distribute_strategy', None),
                    label_smoothing=label_smoothing,
                    focal_gamma=focal_gamma,
                    pos_weight=pos_weight
                )
                ema_callback.rebuild_shadow_variables()
                if tuning_cb_stage2 is not None:
                    tuning_cb_stage2.optimizer = model.optimizer

            next_boundary = schedule_events[0][0] if schedule_events else EPOCHS
            target_epoch = min(next_boundary, EPOCHS)
            if target_epoch <= current_epoch:
                current_epoch += 1
                continue

            history_stage2_part = model.fit(
                train_dataset,
                epochs=target_epoch,
                initial_epoch=current_epoch,
                callbacks=list(stage2_callbacks_base),
                validation_data=valid_dataset,
                steps_per_epoch=train_steps,
                validation_steps=valid_steps,
                verbose=VERBOSE,
            )
            histories.append(history_stage2_part)
            histories_stage2.append(history_stage2_part)
            if history_stage2_part and getattr(history_stage2_part, "epoch", None):
                last_seen_epoch = max(history_stage2_part.epoch) + 1
            else:
                last_seen_epoch = current_epoch
            if last_seen_epoch < target_epoch and VERBOSE:
                print(f"[treino] fit terminou em {last_seen_epoch} < alvo {target_epoch}; retomando.")
            if last_seen_epoch <= current_epoch:
                raise RuntimeError("O treino não avançou de época; verifique o pipeline de dados.")
            current_epoch = last_seen_epoch

        if histories_stage2:
            history_stage2 = histories_stage2[-1]

    if current_epoch < EPOCHS:
        raise RuntimeError(f"Treinamento encerrou antes das {EPOCHS} épocas previstas (concluídas: {current_epoch}).")

    tElapsed = round(time.time() - tStart, 1)

    print(' ')
    print('Time (sec) elapsed: ', tElapsed)
    print('...')

    valid_eval_dataset = get_valid_dataset(
        VALID_FILENAMES,
        normalize_mode=normalize_mode,
        preprocess_fn=preprocess_fn,
        fundus_crop_ratio=fundus_crop_ratio
    )
    applied_ema = ema_callback.apply_ema_weights()
    labels, probabilities = predict_with_tta(model, valid_eval_dataset, TTA_VIEWS)
    probabilities = probabilities.ravel()

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, probabilities)
    auc_keras = auc(fpr_keras, tpr_keras)

    df = pd.DataFrame(thresholds_keras, columns=['thresholds'])
    df.insert(1, 'tpr', tpr_keras)
    df.insert(2, 'fpr', fpr_keras)
    df.insert(3, 'sens', tpr_keras)
    df['spec'] = 1-df['fpr']
    df.to_csv(results +'/'+ dataset + '-%i-thresholds.csv'%exec, encoding='utf-8', index=False)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.4f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(results +'/'+ dataset + '-%i.pdf' %exec, format="pdf", bbox_inches="tight")

    if applied_ema:
        ema_callback.restore_original_weights()

    train_eval_dataset = get_valid_dataset(
        TRAINING_FILENAMES,
        dataset_name="train_eval",
        normalize_mode=normalize_mode,
        preprocess_fn=preprocess_fn,
        fundus_crop_ratio=fundus_crop_ratio
    )
    valid_eval_dataset_for_eval = get_valid_dataset(
        VALID_FILENAMES,
        normalize_mode=normalize_mode,
        preprocess_fn=preprocess_fn,
        fundus_crop_ratio=fundus_crop_ratio
    )
    train_final_stats = model.evaluate(train_eval_dataset, verbose=VERBOSE)
    valid_final_stats = model.evaluate(valid_eval_dataset_for_eval, verbose=VERBOSE)

    print(f"Train final stats: {train_final_stats}")
    print(f"Valid final stats: {valid_final_stats}")
    # Padroniza o AUC reportado para o cálculo exato (roc_auc_score) usado também nos outros pipelines
    print(f"{dataset},{exec},{valid_final_stats[1]},{auc_keras},{tElapsed}")

    mem_peak = gpu_memory_cb.max_peak_mb if getattr(gpu_memory_cb, "max_peak_mb", 0) > 0 else None
    val_loss_final = valid_final_stats[0] if isinstance(valid_final_stats, (list, tuple)) and len(valid_final_stats) > 0 else None
    # Sempre persistir o AUC calculado com roc_auc_score (mais fiel)
    val_auc_final = auc_keras
    val_sens_final = None
    val_spec_final = None
    try:
        val_sens_final = valid_final_stats[3]
    except Exception:
        pass
    try:
        val_spec_final = valid_final_stats[4]
    except Exception:
        pass
    train_sens_final = None
    train_spec_final = None
    try:
        train_sens_final = train_final_stats[3]
        train_spec_final = train_final_stats[4]
    except Exception:
        pass
    final_row = {
        "epoch": current_epoch,
        "stage": "final_eval",
        "train_loss": None,
        "train_auc": None,
        "train_sens": train_sens_final,
        "train_spec": train_spec_final,
        "train_throughput_img_s": None,
        "train_elapsed_s": None,
        "train_gpu_mem_alloc_mb": mem_peak,
        "train_gpu_mem_reserved_mb": mem_peak,
        "val_loss": val_loss_final,
        "val_auc": val_auc_final,
        "val_sens": val_sens_final,
        "val_spec": val_spec_final,
        "val_throughput_img_s": None,
        "val_elapsed_s": None,
        "val_gpu_mem_alloc_mb": mem_peak,
        "val_gpu_mem_reserved_mb": mem_peak,
        "lr": None,
        "total_train_time_s": tElapsed,
    }
    try:
        with Path(csv_logger_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EpochCsvLogger._fields)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(final_row)
    except Exception as exc:
        print(f"[WARN] Falha ao gravar linha final no CSV: {exc}")

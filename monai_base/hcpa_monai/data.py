from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    EnsureTyped,
    Lambdad,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandFlipd,
    RandRotated,
    Resized,
    NormalizeIntensityd,
)

from .config import TrainConfig
from .utils import ensure_dir

try:  # optional; required only when reading real TFRecords
    from tfrecord.reader import tfrecord_loader
    from tfrecord.tools import tfrecord2idx
except Exception:  # pragma: no cover - optional runtime dep
    tfrecord_loader = None
    tfrecord2idx = None

# DALI explicit off: keep import guard but force-disabled to ensure pure MONAI path
try:  # pragma: no cover - GPU only
    from nvidia.dali import fn, types as dali_types, tfrecord as dali_tfrec  # type: ignore
    from nvidia.dali.pipeline import pipeline_def  # type: ignore
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy  # type: ignore
except Exception:  # pragma: no cover - CPU-only environments
    fn = dali_types = dali_tfrec = pipeline_def = DALIGenericIterator = LastBatchPolicy = None
# Never enable DALI in this variant
_DALI_AVAILABLE = False

TFREC_DESCRIPTION = {
    "imagem": "byte",
    "retinopatia": "int",
}


# ----------------------
# TFRecord helpers
# ----------------------

def ensure_index_file(tfrecord_path: Path) -> Path:
    if tfrecord2idx is None:
        raise ImportError("Missing dependency 'tfrecord'. Install requirements.txt to read TFRecords.")
    idx_path = tfrecord_path.with_suffix(tfrecord_path.suffix + ".idx")
    if idx_path.exists():
        return idx_path
    try:
        ensure_dir(idx_path.parent)
        tfrecord2idx.create_index(str(tfrecord_path), str(idx_path))
    except (PermissionError, OSError):  # fallback to tmp when read-only
        fallback = Path(os.environ.get("HCPA_IDX_DIR", "/tmp/hcpa_idx")) / idx_path.name
        ensure_dir(fallback.parent)
        if not fallback.exists():
            tfrecord2idx.create_index(str(tfrecord_path), str(fallback))
        return fallback
    return idx_path


def count_records(idx_path: Path) -> int:
    with idx_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def decode_image(image_bytes: bytes, image_size: int, fundus_crop_ratio: float) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes from TFRecord")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if fundus_crop_ratio < 0.999:
        h, w, _ = img.shape
        crop_size = int(min(h, w) * fundus_crop_ratio)
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        img = img[start_y : start_y + crop_size, start_x : start_x + crop_size]

    if (img.shape[0], img.shape[1]) != (image_size, image_size):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32)


def _label_to_int(raw) -> int:
    arr = np.asarray(raw).reshape(-1)
    val = arr[0]
    if isinstance(val, (bytes, bytearray)):
        try:
            return int(val.decode())
        except Exception:
            return int(val[0])
    return int(val)


# ----------------------
# Transforms
# ----------------------

def _norm_params(mode: str):
    mode = (mode or "none").lower()
    if mode == "inception":
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if mode == "imagenet":
        return np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return None, None


def _center_crop_ratio(x: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio >= 0.999:
        return x
    # x: C x H x W
    _, h, w = x.shape
    side = int(min(h, w) * ratio)
    top = max((h - side) // 2, 0)
    left = max((w - side) // 2, 0)
    return x[:, top : top + side, left : left + side]


def build_transforms(cfg: TrainConfig, training: bool) -> Compose:
    mean, std = _norm_params(cfg.normalize)
    ops = [
        Lambdad(keys="image", func=lambda x: x.astype(np.float32)),
        Lambdad(keys="image", func=lambda x: x / 255.0),
        # TFRecord decode returns HWC numpy arrays without MONAI metadata.
        # Force channel dimension to be the last axis to avoid meta inference errors.
        EnsureChannelFirstd(keys="image", channel_dim=-1),
        Lambdad(keys="image", func=lambda x: _center_crop_ratio(x, cfg.fundus_crop_ratio)),
        Resized(keys="image", spatial_size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False),
    ]

    if training and cfg.augment:
        ops.extend(
            [
                # For 2D images (C,H,W), spatial axes are 0 (H) and 1 (W).
                RandFlipd(keys="image", prob=0.5, spatial_axis=1),
                RandRotated(keys="image", prob=0.25, range_x=0.0, range_y=0.0, range_z=(-0.17, 0.17)),
                RandAdjustContrastd(keys="image", prob=0.25, gamma=(0.9, 1.1)),
                RandCoarseDropoutd(keys="image", prob=0.15, holes=1, spatial_size=32, fill_value=0.0),
            ]
        )

    if mean is not None and std is not None:
        ops.append(NormalizeIntensityd(keys="image", subtrahend=mean, divisor=std, channel_wise=True))

    ops.append(EnsureTyped(keys=("image", "label")))
    return Compose(ops)


# ----------------------
# Datasets
# ----------------------


class RetinaTFRecordIterableDataset(IterableDataset):
    def __init__(
        self,
        files: Sequence[Path],
        *,
        config: TrainConfig,
        training: bool,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        max_samples: int | None = None,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.cfg = config
        self.training = training
        self.shuffle = shuffle
        self.seed = seed
        self.transforms = transforms
        self.rank = rank
        self.world_size = world_size
        self.max_samples = max_samples
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _iter_file(self, tfrec_path: Path) -> Iterator[Dict[str, np.ndarray]]:
        if tfrecord_loader is None:
            raise ImportError("Missing dependency 'tfrecord'. Install requirements.txt to read TFRecords.")
        idx_path = ensure_index_file(tfrec_path)
        loader_kwargs = {}
        sig = inspect.signature(tfrecord_loader)
        if "shuffle_queue_size" in sig.parameters:
            loader_kwargs["shuffle_queue_size"] = 2048 if self.shuffle else 0
        loader = tfrecord_loader(str(tfrec_path), str(idx_path), TFREC_DESCRIPTION, **loader_kwargs)
        for record in loader:
            # use key existence, not truthiness, so label=0 is preserved
            image_bytes = record.get("imagem") if "imagem" in record else record.get(b"imagem")
            label_raw = record.get("retinopatia") if "retinopatia" in record else record.get(b"retinopatia")
            if image_bytes is None or label_raw is None:
                continue
            if isinstance(image_bytes, np.ndarray):
                image_bytes = image_bytes.tobytes()
            image = decode_image(image_bytes, self.cfg.image_size, self.cfg.fundus_crop_ratio)
            yield {
                "image": image,
                "label": np.array([_label_to_int(label_raw)], dtype=np.int64),
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        files = list(self.files)
        if self.shuffle:
            rng.shuffle(files)

        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        stride = max(1, num_workers * self.world_size)
        offset = worker_id + self.rank * num_workers
        emitted = 0

        for tfrec_path in files[offset::stride]:
            for sample in self._iter_file(tfrec_path):
                if self.transforms:
                    sample = self.transforms(sample)
                yield sample
                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return


class FakeRetinaDataset(IterableDataset):
    def __init__(self, *, num_samples: int, image_size: int, num_classes: int, seed: int) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            image = rng.random((3, self.image_size, self.image_size), dtype=np.float32)
            label = rng.integers(0, self.num_classes, dtype=np.int64)
            yield {
                "image": torch.tensor(image, dtype=torch.float32),
                "label": torch.tensor([label], dtype=torch.int64),
            }

    def set_epoch(self, epoch: int) -> None:  # pragma: no cover - for API parity
        _ = epoch


# ----------------------
# DALI pipeline
# ----------------------


def _dali_mean_std(mode: str):
    mode = (mode or "none").lower()
    if mode == "inception":
        return [127.5, 127.5, 127.5], [127.5, 127.5, 127.5]
    if mode == "imagenet":
        mean = [x * 255.0 for x in (0.485, 0.456, 0.406)]
        std = [x * 255.0 for x in (0.229, 0.224, 0.225)]
        return mean, std
    return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]


if _DALI_AVAILABLE:

    @pipeline_def
    def _retina_tfrecord_pipeline(
        tfrec_files: Sequence[str],
        idx_files: Sequence[str],
        *,
        shard_id: int,
        num_shards: int,
        reader_seed: int,
        image_size: int,
        enable_augment: bool,
        fundus_crop_ratio: float,
        mean: Sequence[float],
        std: Sequence[float],
        random_shuffle: bool,
    ):
        reader = fn.readers.tfrecord(
            path=tfrec_files,
            index_path=idx_files,
            features={
                "imagem": dali_tfrec.FixedLenFeature((), dali_tfrec.string, ""),
                "retinopatia": dali_tfrec.FixedLenFeature([1], dali_tfrec.int64, 0),
            },
            random_shuffle=random_shuffle,
            initial_fill=1024,
            read_ahead=True,
            prefetch_queue_depth=2,
            name="Reader",
            seed=reader_seed,
            skip_cached_images=False,
            shard_id=shard_id,
            num_shards=num_shards,
        )

        images = fn.decoders.image(reader["imagem"], device="mixed", output_type=dali_types.RGB)
        if fundus_crop_ratio < 0.999:
            images = fn.crop(images, crop=[fundus_crop_ratio, fundus_crop_ratio], crop_pos_x=0.5, crop_pos_y=0.5)
        images = fn.resize(
            images,
            resize_x=image_size,
            resize_y=image_size,
            interp_type=dali_types.INTERP_TRIANGULAR,
        )
        images = fn.cast(images, dtype=dali_types.FLOAT)

        if enable_augment:
            mirror_h = fn.random.coin_flip(probability=0.5)
            mirror_v = fn.random.coin_flip(probability=0.1)
            images = fn.flip(images, horizontal=mirror_h, vertical=mirror_v)
            angle = fn.random.uniform(range=(-10.0, 10.0))
            rotate_cond = fn.random.coin_flip(probability=0.25)
            images = fn.rotate(images, angle=angle * rotate_cond, keep_size=True, fill_value=128.0)
            contrast = fn.random.uniform(range=(0.9, 1.1))
            brightness = fn.random.uniform(range=(0.9, 1.1))
            images = fn.brightness_contrast(images, contrast=contrast, brightness=brightness)

        images = fn.crop_mirror_normalize(
            images,
            dtype=dali_types.FLOAT,
            output_layout="CHW",
            mean=mean,
            std=std,
        )

        labels = reader["retinopatia"]
        labels = fn.cast(labels, dtype=dali_types.FLOAT)
        labels = labels.gpu()
        return images, labels


class DaliWrapper:
    def __init__(self, iterator: DALIGenericIterator, steps: int) -> None:
        self.iterator = iterator
        self.steps = steps

    def __iter__(self):
        for data in self.iterator:
            batch = data[0]
            yield {
                "image": batch["images"],
                "label": batch["labels"].squeeze(-1),
            }

    def __len__(self):
        return self.steps

    def reset(self):
        self.iterator.reset()


# ----------------------
# Public API
# ----------------------

def find_tfrec_splits(tfrec_dir: Path) -> Tuple[List[Path], List[Path]]:
    train_files = sorted(tfrec_dir.glob("train*.tfrec"))
    eval_files = sorted(tfrec_dir.glob("test*.tfrec")) + sorted(tfrec_dir.glob("val*.tfrec")) + sorted(
        tfrec_dir.glob("valid*.tfrec")
    )
    return train_files, eval_files


def _make_loader(dataset: IterableDataset, cfg: TrainConfig, batch_size: int, *, drop_last: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=drop_last,
    )


def create_loaders(
    cfg: TrainConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Iterable, Iterable, Dict[str, int]]:
    """Build train and eval loaders plus dataset size metadata."""

    train_files, eval_files = find_tfrec_splits(cfg.tfrec_dir)
    meta: Dict[str, int] = {"train_items": 0, "eval_items": 0}
    # For this variant DALI is intentionally disabled.
    cfg.use_dali = False

    if cfg.use_fake_data or (not train_files and not eval_files):
        train_ds = FakeRetinaDataset(num_samples=cfg.fake_train_size, image_size=cfg.image_size, num_classes=cfg.num_classes, seed=cfg.seed)
        eval_ds = FakeRetinaDataset(num_samples=cfg.fake_eval_size, image_size=cfg.image_size, num_classes=cfg.num_classes, seed=cfg.seed + 1)
        train_loader = _make_loader(train_ds, cfg, cfg.batch_size, drop_last=cfg.drop_last)
        eval_loader = _make_loader(eval_ds, cfg, cfg.eval_batch_size, drop_last=False)
        meta["train_items"] = cfg.fake_train_size
        meta["eval_items"] = cfg.fake_eval_size
        return train_loader, eval_loader, meta

    if cfg.use_dali and _DALI_AVAILABLE:
        mean, std = _dali_mean_std(cfg.normalize)
        train_idx = [ensure_index_file(f) for f in train_files]
        eval_idx = [ensure_index_file(f) for f in eval_files]
        train_steps = sum(count_records(p) for p in train_idx) // cfg.batch_size
        eval_steps = max(1, sum(count_records(p) for p in eval_idx) // cfg.eval_batch_size)

        train_pipe = _retina_tfrecord_pipeline(
            batch_size=cfg.batch_size,
            num_threads=max(cfg.num_workers, 2),
            device_id=rank,
            tfrec_files=[str(p) for p in train_files],
            idx_files=[str(p) for p in train_idx],
            shard_id=rank,
            num_shards=world_size,
            reader_seed=cfg.seed,
            image_size=cfg.image_size,
            enable_augment=cfg.augment,
            fundus_crop_ratio=cfg.fundus_crop_ratio,
            mean=mean,
            std=std,
            random_shuffle=True,
        )
        eval_pipe = _retina_tfrecord_pipeline(
            batch_size=cfg.eval_batch_size,
            num_threads=max(cfg.num_workers, 2),
            device_id=rank,
            tfrec_files=[str(p) for p in eval_files],
            idx_files=[str(p) for p in eval_idx],
            shard_id=rank,
            num_shards=world_size,
            reader_seed=cfg.seed + 999,
            image_size=cfg.image_size,
            enable_augment=False,
            fundus_crop_ratio=cfg.fundus_crop_ratio,
            mean=mean,
            std=std,
            random_shuffle=False,
        )

        train_pipe.build()
        eval_pipe.build()

        train_iter = DALIGenericIterator([train_pipe], ["images", "labels"], reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP)
        eval_iter = DALIGenericIterator([eval_pipe], ["images", "labels"], reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

        meta["train_items"] = train_steps * cfg.batch_size
        meta["eval_items"] = eval_steps * cfg.eval_batch_size
        return DaliWrapper(train_iter, train_steps), DaliWrapper(eval_iter, eval_steps), meta

    # Python / MONAI path
    transforms_train = build_transforms(cfg, training=True)
    transforms_eval = build_transforms(cfg, training=False)

    train_counts = [count_records(ensure_index_file(f)) for f in train_files]
    eval_counts = [count_records(ensure_index_file(f)) for f in eval_files]
    meta["train_items"] = int(sum(train_counts))
    meta["eval_items"] = int(sum(eval_counts))

    train_ds = RetinaTFRecordIterableDataset(
        train_files,
        config=cfg,
        training=True,
        shuffle=True,
        seed=cfg.seed,
        rank=rank,
        world_size=world_size,
        transforms=transforms_train,
    )
    eval_ds = RetinaTFRecordIterableDataset(
        eval_files,
        config=cfg,
        training=False,
        shuffle=False,
        seed=cfg.seed + 123,
        rank=rank,
        world_size=world_size,
        transforms=transforms_eval,
        max_samples=meta["eval_items"] if meta["eval_items"] > 0 else None,
    )

    train_loader = _make_loader(train_ds, cfg, cfg.batch_size, drop_last=cfg.drop_last)
    eval_loader = _make_loader(eval_ds, cfg, cfg.eval_batch_size, drop_last=False)
    return train_loader, eval_loader, meta


__all__ = [
    "create_loaders",
    "find_tfrec_splits",
    "build_transforms",
    "RetinaTFRecordIterableDataset",
    "FakeRetinaDataset",
]

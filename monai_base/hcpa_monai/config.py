from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration shared by robust and optimized MONAI training flows."""

    # paths
    results_dir: Path
    tfrec_dir: Path

    # dataset
    image_size: int = 299
    num_classes: int = 2
    fundus_crop_ratio: float = 0.9
    normalize: str = "inception"  # inception|imagenet|none
    augment: bool = True
    color_jitter: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    label_smoothing: float = 0.0
    use_dali: bool = False
    smart_cache: bool = False
    cache_rate: float = 0.0
    use_fake_data: bool = False
    fake_train_size: int = 256
    fake_eval_size: int = 64

    # loader
    batch_size: int = 96
    eval_batch_size: Optional[int] = None
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True

    # model
    model_name: str = "inception_v3"
    pretrained: bool = True
    dropout: float = 0.2
    channels_last: bool = True

    # optimization
    epochs: int = 200
    learning_rate: float = 3e-4
    min_lr: float = 0.0
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "none"  # none
    grad_clip_norm: float = 1.0
    pos_weight: Optional[float] = None
    amp: bool = True
    compile: bool = False
    gradient_accumulation: int = 1
    ema_decay: float = 0.0
    ema_on_cpu: bool = False

    # evaluation / logging
    threshold: float = 0.5
    tta_views: int = 1
    patience: int = 0
    target_metric: str = "auc"
    log_every: int = 50
    save_every: int = 1
    seed: int = 42

    # data pipeline threading hints
    host_prefetch: int = 2
    device_prefetch: int = 2

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.tfrec_dir = Path(self.tfrec_dir)
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 for DR classification")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be >= 1")
        if self.mixup_alpha < 0 or self.cutmix_alpha < 0:
            raise ValueError("mixup/cutmix alpha must be >= 0")
        if self.fundus_crop_ratio <= 0:
            self.fundus_crop_ratio = 1.0
        # match pytorch_opt: eval batch = train batch unless explicit
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size


@dataclass
class VariantDefaults:
    """Wrapper that captures tuned defaults for a given variant."""

    name: str
    description: str
    config: TrainConfig


__all__ = ["TrainConfig", "VariantDefaults"]

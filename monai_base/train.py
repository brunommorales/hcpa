from __future__ import annotations

import argparse
from pathlib import Path

from hcpa_monai import train_and_evaluate
from hcpa_monai.config import TrainConfig


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Treino MONAI puro (PyTorch/MONAI) — sem DALI/TIMM/scheduler")
    p.add_argument("--results", type=Path, default=Path("./results/monai_puro"))
    p.add_argument("--tfrec_dir", type=Path, default=Path("./data/all-tfrec"))
    p.add_argument("--image_size", type=int, default=299)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--model", dest="model_name", type=str, default="inception_v3")
    p.add_argument("--normalize", type=str, default="inception", choices=["inception", "imagenet", "none"])
    p.add_argument("--fundus_crop_ratio", type=float, default=0.9)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--cutmix_alpha", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--channels_last", action="store_true", default=True)
    p.add_argument("--no_channels_last", dest="channels_last", action="store_false")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--scheduler", type=str, default="none", choices=["none"])
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--ema_on_cpu", action="store_true", default=False)
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--use_fake_data", action="store_true", default=False)

    args = p.parse_args()
    kwargs = vars(args)
    # Map CLI to TrainConfig fields
    kwargs["results_dir"] = kwargs.pop("results")
    return TrainConfig(**kwargs)


def main() -> None:
    cfg = parse_args()
    final = train_and_evaluate(cfg)
    print(final)


if __name__ == "__main__":
    main()

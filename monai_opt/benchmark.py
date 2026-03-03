from __future__ import annotations

import argparse
from pathlib import Path

from hcpa_monai_optimized import benchmark
from hcpa_monai_optimized.config import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark otimizado (PyTorch/MONAI)")
    p.add_argument("--results", type=Path, default=Path("./results/opt_bench"))
    p.add_argument("--tfrec_dir", type=Path, default=Path("./data/all-tfrec"))
    p.add_argument("--image_size", type=int, default=299)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--model", dest="model_name", type=str, default="inception_v3")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--use_dali", action="store_true", default=True)
    p.add_argument("--no_dali", dest="use_dali", action="store_false")
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--measure_steps", type=int, default=200)
    p.add_argument("--use_fake_data", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        results_dir=args.results,
        tfrec_dir=args.tfrec_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        model_name=args.model_name,
        amp=args.amp,
        use_dali=args.use_dali,
        use_fake_data=args.use_fake_data,
    )
    metrics = benchmark(cfg, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps)
    print(metrics)


if __name__ == "__main__":
    main()

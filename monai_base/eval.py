from __future__ import annotations

import argparse
from pathlib import Path

from hcpa_monai import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliar checkpoint (MONAI puro, sem DALI/TIMM)")
    p.add_argument("--results", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(args.results)
    print(metrics)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate 9 Matplotlib charts comparing TensorFlow vs PyTorch
for each batch size (96, 128, 160) and each metric.
"""

from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "tf_vs_pytorch_single_gpu.csv"
OUTPUT_DIR = ROOT / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS = [
    ("mean_auc", "Mean AUC"),
    ("mean_throughput_img_s", "Throughput (img/s)"),
    ("mean_train_time_s", "Training Time (s)"),
]

BATCHES = [96, 128, 160]

FRAMEWORK_COLORS = {
    "tensorflow": "#2a9d8f",
    "pytorch": "#f4a261",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {
        "framework", "gpus", "batch_size",
        "mean_auc", "mean_throughput_img_s", "mean_train_time_s"
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV {csv_path}: {sorted(missing)}")

    return df


def plot_single_batch(df: pd.DataFrame, batch_size: int, metric: str, label: str):
    """Create one plot comparing TF vs PT for a single batch and single metric."""
    subset = df[df["batch_size"] == batch_size]

    if subset.empty:
        print(f"⚠️ Warning: No data for batch size {batch_size}")
        return

    frameworks = ["tensorflow", "pytorch"]
    values = []

    for fw in frameworks:
        row = subset[subset["framework"] == fw]
        if row.empty:
            values.append(None)
        else:
            val = row.iloc[0][metric]
            values.append(val)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [0, 1]

    colors = [FRAMEWORK_COLORS.get(fw, "#4c72b0") for fw in frameworks]
    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=colors,
        edgecolor="#424242",
        linewidth=0.6,
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#f5f5f5")
    ax.set_xticks(x)
    ax.set_xticklabels(["TensorFlow", "PyTorch"], fontsize=11)
    ax.set_ylabel(label)
    ax.set_title(f"{label}\nBatch {batch_size}", fontsize=13, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add values above bars
    for bar, xpos, val in zip(bars, x, values):
        if isinstance(val, (int, float)) and math.isfinite(val):
            ax.text(
                xpos,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()

    out_path = OUTPUT_DIR / f"plot_{batch_size}_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")


def main() -> int:
    df = load_data(CSV_PATH)

    # filter only single GPU entries
    df = df[df["gpus"] == 1]

    if df.empty:
        raise SystemExit("CSV does not contain entries with gpus = 1.")

    # Create 9 plots
    for batch in BATCHES:
        for metric, label in METRICS:
            plot_single_batch(df, batch, metric, label)

    print(f"\nAll 9 charts saved under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate one multi-run result directory and generate job-specific plots.

This script is intentionally isolated from the existing plots_all pipeline so
that a single job can be analyzed without changing the global summaries.

Example:
    python3 plots_all/make_job_stats_plots.py \
        --result-dir pytorch_base/results/result772904_tupi_1xnvidia-geforce-rtx-4090_bs96
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "stats_plots"
TARGET_AUC = 0.95
TEXTURE_SEED = 12345
MAX_MEM_MB = 80_000

PROJECT_LABELS = {
    "pytorch_base": "PyTorch Base",
    "tensorflow_base": "TensorFlow Base",
    "monai_base": "MONAI Base",
    "pytorch_opt": "PyTorch Opt",
    "tensorflow_opt": "TensorFlow Opt",
    "monai_opt": "MONAI Opt",
}

PROJECT_COLORS = {
    "pytorch_base": "#0072B2",
    "tensorflow_base": "#009E73",
    "monai_base": "#56B4E9",
    "pytorch_opt": "#D55E00",
    "tensorflow_opt": "#CC79A7",
    "monai_opt": "#E69F00",
}

PROJECT_HATCHES = {
    "pytorch_base": "/",
    "tensorflow_base": "\\\\",
    "monai_base": "|",
    "pytorch_opt": "-",
    "tensorflow_opt": "+",
    "monai_opt": "x",
}

SUMMARY_METRICS: Sequence[Tuple[str, str, str, str, float]] = [
    ("auc", "auc", "AUC [ROC] (%)", "{:.1f}%", 100.0),
    ("spec", "spec", "Specificity @ Sens=0.95", "{:.3f}", 1.0),
    ("sens", "sens", "Sensitivity (threshold=0.5)", "{:.3f}", 1.0),
    ("throughput", "throughput", "Throughput [img/s]", "{:.0f}", 1.0),
    ("train_time", "train_time", "Training time [s]", "{:.0f}", 1.0),
    ("mem", "mem", "Average memory [MB]", "{:.0f}", 1.0),
    ("tta", "tta", "Time to AUC 0.95 [s]", "{:.1f}", 1.0),
]

CURVE_METRICS: Sequence[Tuple[str, str]] = [
    ("train_loss", "Train loss"),
    ("val_loss", "Validation loss"),
    ("train_auc", "Train AUC"),
    ("val_auc", "Validation AUC"),
    ("train_sens", "Train sensitivity"),
    ("val_sens", "Validation sensitivity"),
    ("train_spec", "Train specificity"),
    ("val_spec", "Validation specificity"),
    ("train_throughput_img_s", "Train throughput [img/s]"),
    ("val_throughput_img_s", "Validation throughput [img/s]"),
    ("train_gpu_mem_avg_mb", "Train memory [MB]"),
    ("val_gpu_mem_avg_mb", "Validation memory [MB]"),
    ("lr", "Learning rate"),
    ("epoch_elapsed_s", "Epoch elapsed [s]"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Directory containing env_manifest.txt and run_*/ subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root folder where the job-specific directory will be created (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        val = float(str(value).strip())
        return val if math.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def finite_values(values: Iterable[object]) -> np.ndarray:
    arr = np.array([safe_float(v) for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def mean_std(values: Iterable[object]) -> Tuple[float, float, int]:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def project_label(project: str) -> str:
    return PROJECT_LABELS.get(project, project.replace("_", " ").title())


def project_color(project: str) -> str:
    return PROJECT_COLORS.get(project, "#4C72B0")


def project_hatch(project: str) -> str:
    return PROJECT_HATCHES.get(project, "")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.facecolor": "#f7f7f7",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "grid.color": "#bcbcbc",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "hatch.color": "#00000014",
            "hatch.linewidth": 0.20,
        }
    )


def add_subtle_texture(ax, seed_offset: int = 0, alpha: float = 0.06) -> None:
    size = 128
    rng = np.random.default_rng(TEXTURE_SEED + seed_offset)
    vgrad = np.linspace(0.54, 0.58, size).reshape(-1, 1)
    base = np.broadcast_to(vgrad, (size, size))
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2) / center
    radial = 0.015 * (1 - np.clip(radius, 0, 1))
    noise = rng.normal(loc=0.0, scale=0.004, size=(size, size))
    texture = np.clip(base + radial + noise, 0.5, 0.6)
    ax.imshow(
        texture,
        cmap="Greys",
        interpolation="bilinear",
        extent=(0, 1, 0, 1),
        transform=ax.transAxes,
        origin="lower",
        alpha=alpha,
        zorder=0.05,
        aspect="auto",
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_manifest(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def normalize_gpu_label(result_dir: Path) -> str:
    name = result_dir.name.lower()
    if "4090" in name:
        return "RTX4090"
    if "l40s" in name:
        return "L40S"
    if "h200" in name or "gh200" in name:
        return "H200"
    if "a100" in name:
        return "A100"
    token = re.sub(r"^result[0-9]+_[^_]+_", "", result_dir.name)
    token = re.sub(r"_bs[0-9]+$", "", token)
    token = token.replace("1x", "").replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", token).strip().upper()


def pick_epoch_csv(run_dir: Path) -> Path:
    for candidate in sorted(run_dir.glob("*.csv")):
        try:
            with candidate.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
        except Exception:
            continue
        lowered = [col.strip().lower() for col in header]
        if "epoch" in lowered:
            return candidate
    raise FileNotFoundError(f"No training CSV with epoch column found in {run_dir}")


def pick_thresholds_csv(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*thresholds.csv"))
    if not matches:
        raise FileNotFoundError(f"No thresholds CSV found in {run_dir}")
    return matches[0]


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def last_finite(series: pd.Series) -> float:
    finite = pd.to_numeric(series, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return float(finite.iloc[-1]) if not finite.empty else float("nan")


def extract_total_train_time(train_df: pd.DataFrame, full_df: pd.DataFrame) -> float:
    total_series = numeric_column(full_df, "total_train_time_s")
    total_series = total_series[np.isfinite(total_series) & (total_series > 0)]
    if not total_series.empty:
        return float(total_series.max())
    epoch_elapsed = numeric_column(train_df, "train_elapsed_s").fillna(0.0) + numeric_column(
        train_df, "val_elapsed_s"
    ).fillna(0.0)
    finite = epoch_elapsed[np.isfinite(epoch_elapsed)]
    return float(finite.sum()) if not finite.empty else float("nan")


def extract_avg_mem(full_df: pd.DataFrame) -> float:
    columns = [
        "train_gpu_mem_avg_mb",
        "val_gpu_mem_avg_mb",
        "test_gpu_mem_avg_mb",
    ]
    values: List[float] = []
    for column in columns:
        if column not in full_df.columns:
            continue
        col = numeric_column(full_df, column)
        col = col[np.isfinite(col) & (col > 0) & (col < MAX_MEM_MB)]
        values.extend(col.tolist())
    return float(np.mean(values)) if values else float("nan")


def time_to_target_auc(train_df: pd.DataFrame, target_auc: float) -> float:
    work = train_df.copy()
    work["val_auc"] = numeric_column(work, "val_auc")
    work["epoch_elapsed_s"] = numeric_column(work, "train_elapsed_s").fillna(0.0) + numeric_column(
        work, "val_elapsed_s"
    ).fillna(0.0)
    work = work.sort_values("epoch").reset_index(drop=True)
    hit = work[np.isfinite(work["val_auc"]) & (work["val_auc"] >= target_auc)]
    if hit.empty:
        return float("nan")
    hit_index = int(hit.index[0])
    return float(work.loc[:hit_index, "epoch_elapsed_s"].sum())


def spec_at_target_sens(thresholds_df: pd.DataFrame, target_sens: float) -> float:
    if not {"sens", "spec"}.issubset(thresholds_df.columns):
        return float("nan")
    work = thresholds_df[["sens", "spec"]].copy()
    work["sens"] = pd.to_numeric(work["sens"], errors="coerce")
    work["spec"] = pd.to_numeric(work["spec"], errors="coerce")
    work = work[np.isfinite(work["sens"]) & np.isfinite(work["spec"])]
    if work.empty:
        return float("nan")
    work = work.sort_values("sens").drop_duplicates(subset=["sens"], keep="last")
    sens = work["sens"].to_numpy(dtype=float)
    spec = work["spec"].to_numpy(dtype=float)
    idx = np.searchsorted(sens, target_sens, side="left")
    if idx == 0 or idx >= len(sens):
        return float("nan")
    sens0, sens1 = sens[idx - 1], sens[idx]
    spec0, spec1 = spec[idx - 1], spec[idx]
    if sens1 == sens0:
        return float(spec1)
    weight = (target_sens - sens0) / (sens1 - sens0)
    return float(spec0 + weight * (spec1 - spec0))


def interpolate_roc(thresholds_df: pd.DataFrame, common_fpr: np.ndarray) -> pd.DataFrame:
    if not {"fpr", "tpr"}.issubset(thresholds_df.columns):
        raise ValueError("Thresholds CSV does not contain fpr/tpr columns")
    work = thresholds_df[["fpr", "tpr"]].copy()
    work["fpr"] = pd.to_numeric(work["fpr"], errors="coerce")
    work["tpr"] = pd.to_numeric(work["tpr"], errors="coerce")
    work = work[np.isfinite(work["fpr"]) & np.isfinite(work["tpr"])]
    if work.empty:
        raise ValueError("ROC data is empty after removing non-finite rows")
    work = work.sort_values("fpr").groupby("fpr", as_index=False)["tpr"].max()
    fpr = work["fpr"].to_numpy(dtype=float)
    tpr = work["tpr"].to_numpy(dtype=float)
    interp_tpr = np.interp(common_fpr, fpr, tpr, left=tpr[0], right=tpr[-1])
    return pd.DataFrame({"fpr": common_fpr, "tpr": interp_tpr})


def load_run(run_dir: Path, common_fpr: np.ndarray) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    run_id = int(run_dir.name.split("_")[-1])
    csv_path = pick_epoch_csv(run_dir)
    thresholds_path = pick_thresholds_csv(run_dir)
    df = pd.read_csv(csv_path)
    thresholds_df = pd.read_csv(thresholds_path)

    stage_series = df["stage"] if "stage" in df.columns else pd.Series("", index=df.index)
    df["stage"] = stage_series.astype(str).str.lower()
    train_df = df[df["stage"].isin(["freeze", "finetune"])].copy()
    train_df["epoch"] = pd.to_numeric(train_df["epoch"], errors="coerce").astype("Int64")
    train_df = train_df[train_df["epoch"].notna()].copy()
    train_df["epoch"] = train_df["epoch"].astype(int)
    train_df["epoch_elapsed_s"] = numeric_column(train_df, "train_elapsed_s").fillna(0.0) + numeric_column(
        train_df, "val_elapsed_s"
    ).fillna(0.0)

    final_auc = last_finite(df.get("val_auc", pd.Series(dtype=float)))
    final_spec = last_finite(df.get("val_spec", pd.Series(dtype=float)))
    final_sens = last_finite(df.get("val_sens", pd.Series(dtype=float)))

    throughput_mean = mean_std(numeric_column(train_df, "train_throughput_img_s"))[0]
    total_train_time = extract_total_train_time(train_df, df)
    avg_mem = extract_avg_mem(df)
    tta = time_to_target_auc(train_df, TARGET_AUC)
    spec95 = spec_at_target_sens(thresholds_df, TARGET_AUC)

    run_record: Dict[str, object] = {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "thresholds_path": str(thresholds_path),
        "epochs": int(train_df["epoch"].max() + 1) if not train_df.empty else 0,
        "auc_final": final_auc,
        "spec_final": final_spec,
        "sens_final": final_sens,
        "spec_at_sens95": spec95,
        "throughput_img_s": throughput_mean,
        "train_time_s": total_train_time,
        "avg_gpu_mem_mb": avg_mem,
        "time_to_auc_0_95_s": tta,
    }

    epoch_cols = ["run_id", "epoch", "stage"] + [metric for metric, _ in CURVE_METRICS]
    epoch_df = train_df.copy()
    epoch_df["run_id"] = run_id
    for metric, _ in CURVE_METRICS:
        if metric not in epoch_df.columns:
            epoch_df[metric] = np.nan
        epoch_df[metric] = pd.to_numeric(epoch_df[metric], errors="coerce")
    epoch_df = epoch_df[epoch_cols].copy()

    roc_df = interpolate_roc(thresholds_df, common_fpr)
    roc_df["run_id"] = run_id
    return run_record, epoch_df, roc_df


def build_job_summary(
    runs_df: pd.DataFrame,
    result_dir: Path,
    project: str,
    gpu: str,
    batch_size: Optional[int],
) -> pd.DataFrame:
    auc_mean, auc_std, _ = mean_std(runs_df["auc_final"])
    spec_mean, spec_std, _ = mean_std(runs_df["spec_at_sens95"])
    sens_mean, sens_std, _ = mean_std(runs_df["sens_final"])
    throughput_mean, throughput_std, _ = mean_std(runs_df["throughput_img_s"])
    train_time_mean, train_time_std, _ = mean_std(runs_df["train_time_s"])
    mem_mean, mem_std, _ = mean_std(runs_df["avg_gpu_mem_mb"])
    tta_mean, tta_std, tta_n = mean_std(runs_df["time_to_auc_0_95_s"])

    return pd.DataFrame(
        [
            {
                "result_dir": result_dir.name,
                "project": project,
                "project_label": project_label(project),
                "gpu": gpu,
                "batch_size": batch_size,
                "runs": int(len(runs_df)),
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "spec_mean": spec_mean,
                "spec_std": spec_std,
                "sens_mean": sens_mean,
                "sens_std": sens_std,
                "throughput_mean": throughput_mean,
                "throughput_std": throughput_std,
                "train_time_mean": train_time_mean,
                "train_time_std": train_time_std,
                "mem_mean": mem_mean,
                "mem_std": mem_std,
                "tta_mean": tta_mean,
                "tta_std": tta_std,
                "tta_n": tta_n,
            }
        ]
    )


def build_epoch_summary(epoch_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for epoch, sub in epoch_df.groupby("epoch", sort=True):
        row: Dict[str, object] = {
            "epoch": int(epoch),
            "stage": sub["stage"].mode(dropna=True).iloc[0] if not sub["stage"].mode(dropna=True).empty else "",
            "runs": int(sub["run_id"].nunique()),
        }
        for metric, _ in CURVE_METRICS:
            mean_value, std_value, count = mean_std(sub[metric])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_n"] = count
        records.append(row)
    return pd.DataFrame(records).sort_values("epoch").reset_index(drop=True)


def build_roc_summary(roc_runs_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for fpr, sub in roc_runs_df.groupby("fpr", sort=True):
        mean_value, std_value, count = mean_std(sub["tpr"])
        records.append(
            {
                "fpr": float(fpr),
                "tpr_mean": mean_value,
                "tpr_std": std_value,
                "runs": count,
            }
        )
    return pd.DataFrame(records).sort_values("fpr").reset_index(drop=True)


def metric_axis_limit(mean_value: float, std_value: float, scale: float) -> float:
    if not math.isfinite(mean_value):
        return 1.0
    value = mean_value * scale
    error = std_value * scale if math.isfinite(std_value) else 0.0
    if value <= 0:
        return 1.0
    return max(value + error, value * 1.18) + max(abs(value) * 0.06, 0.05)


def format_with_std(mean_value: float, std_value: float, fmt: str) -> str:
    if not math.isfinite(mean_value):
        return "N/A"
    if math.isfinite(std_value):
        return f"{fmt.format(mean_value)} ± {fmt.format(std_value)}"
    return fmt.format(mean_value)


def plot_summary_facets(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary = summary_df.iloc[0]
    metrics = list(SUMMARY_METRICS)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9.6, len(metrics) * 2.1), squeeze=False)
    axes = axes[:, 0]

    color = project_color(str(summary["project"]))
    hatch = project_hatch(str(summary["project"]))
    label = str(summary["project_label"])

    for idx, (metric_key, stem, ylabel, fmt, scale) in enumerate(metrics):
        ax = axes[idx]
        add_subtle_texture(ax, seed_offset=idx)
        mean_value = safe_float(summary[f"{stem}_mean"])
        std_value = safe_float(summary[f"{stem}_std"])
        display_mean = mean_value * scale if math.isfinite(mean_value) else float("nan")
        display_std = std_value * scale if math.isfinite(std_value) else float("nan")

        ax.set_ylabel(ylabel)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)

        if math.isfinite(display_mean):
            bar = ax.barh(
                [0],
                [display_mean],
                xerr=[display_std if math.isfinite(display_std) else 0.0],
                color=[color],
                edgecolor="black",
                height=0.58,
                capsize=4,
                linewidth=0.9,
                error_kw={"ecolor": "#333333", "lw": 1.0},
            )[0]
            bar.set_hatch(hatch)
            x_limit = metric_axis_limit(mean_value, std_value, scale)
            ax.set_xlim(0, x_limit)
            ax.set_yticks([0])
            ax.set_yticklabels([label])
            annotation = format_with_std(display_mean, display_std, fmt)
            ax.text(
                display_mean + x_limit * 0.015,
                0,
                annotation,
                va="center",
                ha="left",
                fontsize=10,
                color="#111111",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.35},
            )
        else:
            ax.set_xlim(0, 1)
            ax.set_yticks([0])
            ax.set_yticklabels([label])
            ax.text(
                0.5,
                0,
                "N/A",
                va="center",
                ha="center",
                fontsize=10,
                color="#444444",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.35},
            )

    title = f"{summary['project_label']} | {summary['gpu']} | {summary['result_dir']} | {int(summary['runs'])} runs"
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.995)
    fig.tight_layout(h_pad=1.35, rect=[0, 0, 1, 0.985])
    save_figure(fig, output_dir / "stats_summary_facets")


def plot_metric_bars(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary = summary_df.iloc[0]
    project = str(summary["project"])
    label = str(summary["project_label"])
    color = project_color(project)
    hatch = project_hatch(project)
    slug = str(summary["result_dir"]).replace(" ", "_")

    for idx, (metric_key, stem, ylabel, fmt, scale) in enumerate(SUMMARY_METRICS):
        mean_value = safe_float(summary[f"{stem}_mean"])
        std_value = safe_float(summary[f"{stem}_std"])
        display_mean = mean_value * scale if math.isfinite(mean_value) else float("nan")
        display_std = std_value * scale if math.isfinite(std_value) else float("nan")

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        add_subtle_texture(ax, seed_offset=50 + idx)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        if math.isfinite(display_mean):
            bar = ax.bar(
                [0],
                [display_mean],
                yerr=[display_std if math.isfinite(display_std) else 0.0],
                color=[color],
                edgecolor="black",
                width=0.58,
                capsize=4,
                linewidth=0.9,
                error_kw={"ecolor": "#333333", "lw": 1.0},
            )[0]
            bar.set_hatch(hatch)
            y_limit = metric_axis_limit(mean_value, std_value, scale)
            ax.set_ylim(0, y_limit)
            ax.text(
                0,
                display_mean + y_limit * 0.02,
                format_with_std(display_mean, display_std, fmt),
                ha="center",
                va="bottom",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.35},
            )
        else:
            ax.set_ylim(0, 1)
            ax.text(
                0,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.35},
            )

        ax.set_xticks([0])
        ax.set_xticklabels([label], rotation=18, ha="right")
        ax.set_title(f"{ylabel}\n{summary['gpu']}")

        metric_dir = output_dir / metric_key
        save_figure(fig, metric_dir / f"stats_{metric_key}_{slug}")


def plot_epoch_behavior(
    epoch_df: pd.DataFrame,
    epoch_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary = summary_df.iloc[0]
    project = str(summary["project"])
    color = project_color(project)
    freeze_epochs = epoch_summary_df.loc[epoch_summary_df["stage"] == "freeze", "epoch"]
    freeze_end = int(freeze_epochs.max()) if not freeze_epochs.empty else None

    cols = 4
    rows = math.ceil(len(CURVE_METRICS) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.6), squeeze=False)
    axes_flat = axes.ravel()

    for idx, (metric, title) in enumerate(CURVE_METRICS):
        ax = axes_flat[idx]
        add_subtle_texture(ax, seed_offset=100 + idx, alpha=0.05)
        raw = epoch_df[["run_id", "epoch", metric]].copy()
        raw = raw[np.isfinite(raw[metric])]
        for _, run_sub in raw.groupby("run_id"):
            ax.plot(run_sub["epoch"], run_sub[metric], color="#808080", alpha=0.16, linewidth=1.0, zorder=1)

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        x = epoch_summary_df["epoch"].to_numpy(dtype=float)
        mean_vals = epoch_summary_df[mean_col].to_numpy(dtype=float)
        std_vals = epoch_summary_df[std_col].to_numpy(dtype=float)
        lower = np.where(np.isfinite(mean_vals) & np.isfinite(std_vals), mean_vals - std_vals, np.nan)
        upper = np.where(np.isfinite(mean_vals) & np.isfinite(std_vals), mean_vals + std_vals, np.nan)

        if freeze_end is not None:
            ax.axvspan(-0.5, freeze_end + 0.5, color="#7f7f7f", alpha=0.10, zorder=0)
            ax.axvline(freeze_end + 0.5, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8, zorder=2)

        ax.fill_between(x, lower, upper, color=color, alpha=0.18, zorder=2)
        ax.plot(x, mean_vals, color=color, linewidth=2.2, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)

        if metric in {"train_auc", "val_auc", "train_sens", "val_sens", "train_spec", "val_spec"}:
            ax.set_ylim(0, 1.05)
        if metric == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    for ax in axes_flat[len(CURVE_METRICS) :]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.2, label="Mean"),
        Patch(facecolor=color, alpha=0.18, edgecolor="none", label="Mean ± 1 std"),
        Line2D([0], [0], color="#808080", lw=1.0, alpha=0.5, label="Individual runs"),
    ]
    if freeze_end is not None:
        legend_handles.append(Patch(facecolor="#7f7f7f", alpha=0.10, edgecolor="none", label="Freeze stage"))

    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"Epoch behavior | {summary['project_label']} | {summary['gpu']} | {summary['result_dir']}",
        fontsize=14,
        fontweight="semibold",
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.15, w_pad=1.0)
    save_figure(fig, output_dir / "behavior" / "epoch_behavior_facets")


def plot_roc_mean_std(
    roc_runs_df: pd.DataFrame,
    roc_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary = summary_df.iloc[0]
    color = project_color(str(summary["project"]))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    add_subtle_texture(ax, seed_offset=250, alpha=0.04)

    for _, sub in roc_runs_df.groupby("run_id"):
        ax.plot(sub["fpr"], sub["tpr"], color="#808080", alpha=0.18, linewidth=1.0, zorder=1)

    x = roc_summary_df["fpr"].to_numpy(dtype=float)
    mean_vals = roc_summary_df["tpr_mean"].to_numpy(dtype=float)
    std_vals = roc_summary_df["tpr_std"].to_numpy(dtype=float)
    lower = np.clip(mean_vals - std_vals, 0.0, 1.0)
    upper = np.clip(mean_vals + std_vals, 0.0, 1.0)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.7, zorder=0)
    ax.fill_between(x, lower, upper, color=color, alpha=0.18, zorder=2, label="Mean ± 1 std")
    auc_mean = safe_float(summary["auc_mean"]) * 100.0
    auc_std = safe_float(summary["auc_std"]) * 100.0
    ax.plot(
        x,
        mean_vals,
        color=color,
        linewidth=2.4,
        zorder=3,
        label=f"Mean ROC | AUC = {auc_mean:.1f}% ± {auc_std:.1f}%",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC mean/std | {summary['project_label']} | {summary['gpu']}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    save_figure(fig, output_dir / "roc" / "roc_mean_std")


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Result directory not found: {result_dir}")

    manifest = parse_manifest(result_dir / "env_manifest.txt")
    project = result_dir.parents[1].name
    gpu = normalize_gpu_label(result_dir)
    batch_size = int(manifest["global_batch_size"]) if manifest.get("global_batch_size", "").isdigit() else None
    output_dir = args.output_root.resolve() / result_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()

    common_fpr = np.linspace(0.0, 1.0, 401)
    run_records: List[Dict[str, object]] = []
    epoch_frames: List[pd.DataFrame] = []
    roc_frames: List[pd.DataFrame] = []

    for run_dir in sorted(result_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        run_record, epoch_df, roc_df = load_run(run_dir, common_fpr)
        run_record["project"] = project
        run_record["project_label"] = project_label(project)
        run_record["gpu"] = gpu
        run_record["result_dir"] = result_dir.name
        run_records.append(run_record)
        epoch_frames.append(epoch_df)
        roc_frames.append(roc_df)

    if not run_records:
        raise SystemExit(f"No run_* directories with valid metrics found in {result_dir}")

    runs_df = pd.DataFrame(run_records).sort_values("run_id").reset_index(drop=True)
    epoch_df = pd.concat(epoch_frames, ignore_index=True).sort_values(["run_id", "epoch"]).reset_index(drop=True)
    roc_runs_df = pd.concat(roc_frames, ignore_index=True).sort_values(["run_id", "fpr"]).reset_index(drop=True)

    summary_df = build_job_summary(runs_df, result_dir=result_dir, project=project, gpu=gpu, batch_size=batch_size)
    epoch_summary_df = build_epoch_summary(epoch_df)
    roc_summary_df = build_roc_summary(roc_runs_df)

    runs_df.to_csv(output_dir / "per_run_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "job_summary.csv", index=False)
    epoch_df.to_csv(output_dir / "epoch_metrics_long.csv", index=False)
    epoch_summary_df.to_csv(output_dir / "epoch_metrics_mean_std.csv", index=False)
    roc_runs_df.to_csv(output_dir / "roc_runs_interpolated.csv", index=False)
    roc_summary_df.to_csv(output_dir / "roc_mean_std.csv", index=False)

    plot_summary_facets(summary_df, output_dir)
    plot_metric_bars(summary_df, output_dir)
    plot_epoch_behavior(epoch_df, epoch_summary_df, summary_df, output_dir)
    plot_roc_mean_std(roc_runs_df, roc_summary_df, summary_df, output_dir)

    print(f"Wrote analysis to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

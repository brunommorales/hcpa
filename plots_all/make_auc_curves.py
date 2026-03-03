#!/usr/bin/env python3
"""
Gera gráficos de evolução de AUC (val_auc) por época, com média e desvio padrão
entre runs, para cada GPU. Saídas: plots_all/all_auc_curve_<GPU>.png.

Executar de dentro de projects/hcpa:
    python3 plots_all/make_auc_curves.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RUNS_CSV = REPO_ROOT / "single_gpu_runs.csv"

# Abordagens (nomes de diretório -> rótulo/cores)
APPROACHES: Dict[str, str] = {
    "tensorflow_opt": "TensorFlow Opt",
    "pytorch_opt": "PyTorch Opt",
    "monai_opt": "MONAI Opt",
    "tensorflow_base": "TensorFlow Base",
    "pytorch_base": "PyTorch Base",
    "monai_base": "MONAI Base",
}

COLORS: Dict[str, str] = {
    "tensorflow_opt": "#1f77b4",
    "pytorch_opt": "#ff7f0e",
    "monai_opt": "#2ca02c",
    "tensorflow_base": "#1f77b4",
    "pytorch_base": "#ff7f0e",
    "monai_base": "#9467bd",
}


@dataclass
class CurveStats:
    epochs: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def _pick_metrics_csv(run_dir: Path) -> Path | None:
    """Escolhe um CSV dentro de run_* que contenha epoch e val_auc."""
    candidates = sorted(run_dir.glob("*.csv"))
    for path in candidates:
        try:
            cols = pd.read_csv(path, nrows=0).columns.str.lower()
        except Exception:
            continue
        if {"epoch", "val_auc"}.issubset(set(cols)):
            return path
    return None


def load_auc_curves_from_runs_table(project: str) -> Dict[str, List[pd.DataFrame]]:
    """Usa single_gpu_runs.csv para localizar runs e agrupar por GPU."""
    out: Dict[str, List[pd.DataFrame]] = {}
    if not RUNS_CSV.exists():
        return out
    runs = pd.read_csv(RUNS_CSV)
    runs = runs[runs["project"] == project]
    for row in runs.itertuples():
        gpu = str(row.gpu)
        run_id = int(row.run_id)
        result_dir = getattr(row, "result_dir", None)
        if not result_dir or result_dir != result_dir:  # NaN check
            continue
        run_path = REPO_ROOT / project / "results" / result_dir / f"run_{run_id}"
        metrics = _pick_metrics_csv(run_path)
        if metrics is None:
            continue
        df = pd.read_csv(metrics, usecols=["epoch", "val_auc"])
        if df.empty:
            continue
        out.setdefault(gpu, []).append(df)
    return out


def compute_stats(run_dfs: List[pd.DataFrame]) -> CurveStats:
    """Alinha por época (inner join) e calcula média/DP das curvas."""
    # Interseção das épocas presentes em todos os runs para evitar NaN
    common_epochs = set(run_dfs[0]["epoch"].unique())
    for df in run_dfs[1:]:
        common_epochs &= set(df["epoch"].unique())
    if not common_epochs:
        return CurveStats(np.array([]), np.array([]), np.array([]))

    epochs = np.array(sorted(common_epochs))
    stacked = []
    for df in run_dfs:
        vals = df.set_index("epoch").loc[epochs, "val_auc"].to_numpy()
        stacked.append(vals)
    arr = np.vstack(stacked)  # shape: runs x epochs
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return CurveStats(epochs, mean, std)


def plot_gpu(gpu: str, stats_per_project: Dict[str, CurveStats]) -> None:
    plt.figure(figsize=(9, 5))
    for project, stats in stats_per_project.items():
        if stats.epochs.size == 0:
            continue
        x = stats.epochs
        y = stats.mean * 100  # porcentagem
        y_std = stats.std * 100
        label = APPROACHES[project]
        color = COLORS[project]
        plt.plot(x, y, label=label, color=color, linewidth=2)
        if y_std.any():
            plt.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.15, linewidth=0)

    plt.title(f"Evolução da AUC (val) — GPU {gpu}")
    plt.xlabel("Época")
    plt.ylabel("AUC (ROC) [%]")
    plt.ylim(70, 100)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=2, frameon=False)
    out_path = ROOT / f"all_auc_curve_{gpu.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✔ salvo: {out_path}")


def main() -> None:
    stats_by_gpu: Dict[str, Dict[str, CurveStats]] = {}
    for project in APPROACHES:
        curves = load_auc_curves_from_runs_table(project)
        for gpu, run_dfs in curves.items():
            stats = compute_stats(run_dfs)
            if gpu not in stats_by_gpu:
                stats_by_gpu[gpu] = {}
            stats_by_gpu[gpu][project] = stats

    if not stats_by_gpu:
        raise SystemExit("Nenhum metrics.csv encontrado para gerar curvas.")

    for gpu, proj_stats in stats_by_gpu.items():
        plot_gpu(gpu, proj_stats)


if __name__ == "__main__":
    main()

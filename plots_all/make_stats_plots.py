#!/usr/bin/env python3
"""Gera gráficos estatísticos facetados e tabela de recomendação.

Nova política (menos poluição):
- Mantém stats_summary_facets.png (referência global), stats_summary_<gpu>.png
  (uma figura por GPU) e recommendation_table.{csv,md,png}.
- Para cada métrica, gera um arquivo por GPU dentro de uma subpasta
  dedicada: stats_plots/<metrica>/stats_<metrica>_<gpu>.png.
- Pasta stats_plots é sobrescrita a cada execução.

Executar de dentro de ``projects/hcpa``:
    python3 plots_all/make_stats_plots.py
"""

from __future__ import annotations

import math
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "stats_plots"
OUT_DIR.mkdir(exist_ok=True)
TEXTURE_SEED = 12345

# Ordem e agrupamento informados pelo usuário
APPROACH_ORDER: List[str] = [
    "pytorch_base",
    "tensorflow_base",
    "monai_base",
    "pytorch_opt",
    "tensorflow_opt",
    "monai_opt",
]
BASIC_PROJECTS = ["pytorch_base", "tensorflow_base", "monai_base"]
OPT_PROJECTS = ["pytorch_opt", "tensorflow_opt", "monai_opt"]

APPROACH_LABELS: Dict[str, str] = {
    "pytorch_base": "PyTorch Base",
    "tensorflow_base": "TensorFlow Base",
    "monai_base": "MONAI Base",
    "pytorch_opt": "PyTorch Opt",
    "tensorflow_opt": "TensorFlow Opt",
    "monai_opt": "MONAI Opt",
}

# Paleta daltônica e consistente (Okabe-Ito)
APPROACH_COLORS: Dict[str, str] = {
    "pytorch_base": "#0072B2",
    "tensorflow_base": "#009E73",
    "monai_base": "#56B4E9",
    "pytorch_opt": "#D55E00",
    "tensorflow_opt": "#CC79A7",
    "monai_opt": "#E69F00",
}

WARNED_SCIPY = False
SCIPY_STATS = None

_T_FALLBACK = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

METRICS: Sequence[Tuple[str, str, str, str, str]] = [
    ("auc", "auc_mean", "auc_std", "AUC (ROC) (%)", "{:.1f}%"),
    ("spec", "spec_mean", "spec_std", "Especificidade @Sens=0.95", "{:.3f}"),
    ("sens", "sens_mean", "sens_std", "Sensibilidade (alvo=0.95)", "{:.3f}"),
    ("throughput", "throughput_mean", "throughput_std", "Throughput (img/s)", "{:.0f}"),
    ("train_time", "train_time_mean", "train_time_std", "Tempo de treino (s)", "{:.0f}"),
    ("mem", "mem_mean", "mem_std", "Memória de pico (MB)", "{:.0f}"),
    ("tta", "tta_mean", "tta_std", "Tempo p/ AUC 0.95 (s)", "{:.1f}"),
]


def reset_out_dir() -> None:
    """Remove arquivos/pastas antigos e recria stats_plots limpa."""

    if OUT_DIR.exists():
        for path in OUT_DIR.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    OUT_DIR.mkdir(exist_ok=True)


def configure_style() -> None:
    """Ajustes visuais para gráficos com aparência de paper."""

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.facecolor": "#f7f7f7",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "grid.color": "#bbbbbb",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )


def add_subtle_texture(ax, seed_offset: int = 0, alpha: float = 0.06) -> None:
    """Textura bem lisa e quase imperceptível para evitar fundo chapado.

    Estrutura: gradiente suave (vertical + radial leve) + ruído de baixa
    amplitude. Mantém opacidade baixa para não competir com as barras.
    """

    size = 128
    rng = np.random.default_rng(TEXTURE_SEED + seed_offset)

    # Gradiente vertical suave
    vgrad = np.linspace(0.54, 0.58, size).reshape(-1, 1)
    base = np.broadcast_to(vgrad, (size, size))

    # Radial leve para evitar bandas perceptíveis
    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2) / center
    radial = 0.015 * (1 - np.clip(radius, 0, 1))

    # Ruído muito fraco para dar granulação mínima
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
    )


def _t_critical(df: int, alpha: float = 0.05) -> float:
    """Valor crítico t bilateral para IC (default 95%)."""

    global WARNED_SCIPY, SCIPY_STATS
    if df <= 0:
        return float("nan")

    if SCIPY_STATS is None:
        try:
            from scipy import stats  # type: ignore

            SCIPY_STATS = stats
        except Exception:
            SCIPY_STATS = False
            if not WARNED_SCIPY:
                warnings.warn("scipy indisponível; usando tabela de fallback.", RuntimeWarning)
                WARNED_SCIPY = True

    if SCIPY_STATS:
        return float(SCIPY_STATS.t.ppf(1 - alpha / 2, df))
    return _T_FALLBACK.get(df, 1.96)


def _ci_from_mean_std(mean: float, std: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n is None or n <= 1 or math.isnan(mean) or math.isnan(std):
        return float("nan"), float("nan")
    t_val = _t_critical(n - 1, alpha)
    margin = t_val * std / math.sqrt(n)
    return mean - margin, mean + margin


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["project"].isin(APPROACH_ORDER)]
    df["project"] = pd.Categorical(df["project"], categories=APPROACH_ORDER, ordered=True)
    return df.sort_values(["gpu", "project"]).reset_index(drop=True)


def load_data():
    summary = pd.read_csv(ROOT / "hcpa_all_summary.csv")

    # Alinha com regra usada em make_all_plots: substitui spec/auc pelas
    # estatísticas recalculadas em spec_at_sens95_runs (Spec@Sens=0.95).
    spec_runs_path = ROOT / "spec_at_sens95_runs.csv"
    if spec_runs_path.exists():
        spec_runs = pd.read_csv(spec_runs_path)
        agg = (
            spec_runs.groupby(["gpu", "project"])
            .agg(
                spec_mean=("spec_at_sens95", "mean"),
                spec_std=("spec_at_sens95", "std"),
                auc_mean=("val_auc_final", "mean"),
                auc_std=("val_auc_final", "std"),
                runs=("spec_at_sens95", "count"),
            )
            .reset_index()
        )
        summary = summary.merge(agg, on=["gpu", "project"], how="left", suffixes=("", "_specfile"))
        for col in ["spec_mean", "spec_std", "auc_mean", "auc_std", "runs"]:
            summary[col] = summary[f"{col}_specfile"].combine_first(summary.get(col))
            summary.drop(columns=[f"{col}_specfile"], inplace=True)

    runs_path = ROOT.parent / "single_gpu_runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else None
    return _ordered(summary), None, runs


def plot_summary_facets(summary: pd.DataFrame) -> None:
    """Figura de referência com todas as métricas (métricas x GPUs)."""

    metrics = [m for m in METRICS if m[1] in summary.columns]
    if not metrics:
        return

    gpus = list(summary["gpu"].unique())
    fig, axes = plt.subplots(len(metrics), len(gpus), figsize=(len(gpus) * 4.4, len(metrics) * 2.8), squeeze=False)

    for i, (key, mean_col, std_col, ylabel, fmt) in enumerate(metrics):
        for j, gpu in enumerate(gpus):
            ax = axes[i][j]
            sub = summary[(summary["gpu"] == gpu) & np.isfinite(summary[mean_col])]
            if sub.empty:
                ax.axis("off")
                continue
            sub = sub.sort_values("project")
            add_subtle_texture(ax, seed_offset=i * len(gpus) + j)
            y = np.arange(len(sub))
            scale = 100.0 if key == "auc" else 1.0
            means = sub[mean_col].to_numpy() * scale
            stds = (sub[std_col].to_numpy() if std_col in sub.columns else np.zeros_like(means)) * scale
            colors = [APPROACH_COLORS.get(p, "#777777") for p in sub["project"]]
            labels = [APPROACH_LABELS.get(p, p) for p in sub["project"]]

            ax.barh(
                y,
                means,
                xerr=stds,
                color=colors,
                alpha=0.9,
                edgecolor="black",
                height=0.55,
                error_kw={"ecolor": "0.25", "capsize": 4, "lw": 1.1},
            )
            for idx, m in enumerate(means):
                ax.text(
                    m,
                    y[idx],
                    f" {( '{:.1f}%'.format(m) if key == 'auc' else fmt.format(m))}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#111111",
                    clip_on=False,
                )
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticklabels([])
            ax.margins(x=0.1)
            ax.grid(True, axis="x", linestyle="--", alpha=0.35)
            ax.set_xlabel(f"GPU {gpu}")

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    fig.savefig(OUT_DIR / "stats_summary_facets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_by_gpu(summary: pd.DataFrame) -> None:
    """Um arquivo por GPU contendo todas as métricas."""

    metrics = [m for m in METRICS if m[1] in summary.columns]
    if not metrics:
        return

    for gpu in summary["gpu"].unique():
        fig, axes = plt.subplots(len(metrics), 1, figsize=(5.2, len(metrics) * 2.8), squeeze=False)
        axes = axes[:, 0]

        for i, (key, mean_col, std_col, ylabel, fmt) in enumerate(metrics):
            ax = axes[i]
            sub = summary[(summary["gpu"] == gpu) & np.isfinite(summary[mean_col])]
            if sub.empty:
                ax.axis("off")
                continue
            sub = sub.sort_values("project")
            add_subtle_texture(ax, seed_offset=i)
            y = np.arange(len(sub))
            scale = 100.0 if key == "auc" else 1.0
            means = sub[mean_col].to_numpy() * scale
            stds = (sub[std_col].to_numpy() if std_col in sub.columns else np.zeros_like(means)) * scale
            colors = [APPROACH_COLORS.get(p, "#777777") for p in sub["project"]]
            labels = [APPROACH_LABELS.get(p, p) for p in sub["project"]]

            ax.barh(
                y,
                means,
                xerr=stds,
                color=colors,
                alpha=0.9,
                edgecolor="black",
                height=0.55,
                error_kw={"ecolor": "0.25", "capsize": 4, "lw": 1.1},
            )
            for idx, m in enumerate(means):
                ax.text(
                    m,
                    y[idx],
                    f" {( '{:.1f}%'.format(m) if key == 'auc' else fmt.format(m))}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#111111",
                    clip_on=False,
                )
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.margins(x=0.1)
            ax.grid(True, axis="x", linestyle="--", alpha=0.35)

        fig.tight_layout(h_pad=0.9)
        safe_name = gpu.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"stats_summary_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_per_metric(summary: pd.DataFrame) -> None:
    """Um arquivo por métrica *por GPU*, salvo em subpastas dedicadas."""

    gpus = list(summary["gpu"].unique())
    for key, mean_col, std_col, ylabel, fmt in METRICS:
        if mean_col not in summary.columns:
            continue

        metric_dir = OUT_DIR / key
        metric_dir.mkdir(exist_ok=True)

        for gpu_idx, gpu in enumerate(gpus):
            sub = summary[(summary["gpu"] == gpu) & np.isfinite(summary[mean_col])]
            if sub.empty:
                continue

            sub = sub.sort_values("project")
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            add_subtle_texture(ax, seed_offset=gpu_idx)

            x = np.arange(len(sub))
            scale = 100.0 if key == "auc" else 1.0
            means = sub[mean_col].to_numpy() * scale
            stds = (sub[std_col].to_numpy() if std_col in sub.columns else np.zeros_like(means)) * scale
            colors = [APPROACH_COLORS.get(p, "#777777") for p in sub["project"]]
            labels = [APPROACH_LABELS.get(p, p) for p in sub["project"]]

            ax.bar(
                x,
                means,
                yerr=stds,
                color=colors,
                alpha=0.9,
                edgecolor="black",
                capsize=4,
                error_kw={"ecolor": "0.25", "lw": 1.0},
            )
            for xi, m in zip(x, means):
                label = "{:.1f}%".format(m) if key == "auc" else fmt.format(m)
                ax.text(xi, m, label, ha="center", va="bottom", fontsize=9, color="#111111", clip_on=False)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=18, ha="right")
            ax.set_ylabel(ylabel)
            ax.margins(y=0.18)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

            safe_gpu = gpu.lower().replace(" ", "_")
            fig.tight_layout()
            fig.savefig(metric_dir / f"stats_{key}_{safe_gpu}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def build_recommendations(summary: pd.DataFrame) -> None:
    """Tabela comparativa das 6 abordagens: melhor e pior métrica."""

    if summary.empty:
        return

    # Define direção (True = maior melhor)
    metric_direction = {
        "auc_mean": True,
        "spec_mean": True,
        "sens_mean": True,
        "throughput_mean": True,
        "mem_mean": False,
        "train_time_mean": False,
        "tta_mean": False,
    }

    agg = summary.groupby("project").mean(numeric_only=True).reset_index()
    # Remove abordagens sem métricas válidas
    metric_cols = list(metric_direction.keys())
    agg = agg.dropna(subset=metric_cols, how="all")
    if agg.empty:
        return

    # Normaliza cada métrica para [0,1] na direção correta
    norm_cols = {}
    for col, higher_better in metric_direction.items():
        if col not in agg.columns:
            continue
        series = agg[col]
        if series.nunique(dropna=True) <= 1:
            norm = pd.Series(0.5, index=series.index)
        else:
            norm = (series - series.min()) / (series.max() - series.min())
            if not higher_better:
                norm = 1 - norm
        norm = norm.fillna(0.5)
        norm_cols[col] = norm
        agg[f"{col}_norm"] = norm

    rows = []
    for _, row in agg.iterrows():
        proj = row["project"]
        label = APPROACH_LABELS.get(proj, proj)

        best_metric = None
        best_score = -1
        worst_metric = None
        worst_score = 2

        for col, _norm_series in norm_cols.items():
            val = row[col]
            if pd.isna(val):
                continue
            score = row[f"{col}_norm"]
            if pd.isna(score):
                continue
            if score > best_score:
                best_score = score
                best_metric = col
            if score < worst_score:
                worst_score = score
                worst_metric = col

        # Se melhor e pior ficaram iguais, escolha o segundo pior (exceto a melhor)
        if best_metric == worst_metric:
            alt_worst = None
            alt_score = 2
            for col in norm_cols:
                if col == best_metric:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                score = row[f"{col}_norm"]
                if pd.isna(score):
                    continue
                if score < alt_score:
                    alt_score = score
                    alt_worst = col
            if alt_worst is not None:
                worst_metric = alt_worst

        def fmt_metric(metric: str) -> str:
            if not metric or metric not in metric_direction:
                return metric
            val = row[metric]
            if pd.isna(val):
                return "n/a"
            unit = ""
            if "throughput" in metric:
                unit = " img/s"
            if "mem" in metric:
                unit = " MB"
            if "time" in metric or "tta" in metric:
                unit = " s"
            return f"{metric.replace('_mean','').upper()} = {val:.3f}{unit}" if isinstance(val, float) else str(val)

        resumo = (
            f"AUC {row.get('auc_mean', float('nan')):.3f}; "
            f"Spec {row.get('spec_mean', float('nan')):.3f}; "
            f"Thr {row.get('throughput_mean', float('nan')):.0f} img/s; "
            f"Mem {row.get('mem_mean', float('nan')):.0f} MB; "
            f"Train {row.get('train_time_mean', float('nan')):.0f}s"
        )

        rows.append(
            {
                "abordagem": label,
                "melhor_metrica": fmt_metric(best_metric) if best_metric else "",
                "pior_metrica": fmt_metric(worst_metric if worst_metric != best_metric else None) if worst_metric else "",
                "resumo": resumo,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(OUT_DIR / "recommendation_table.csv", index=False)
        md_path = OUT_DIR / "recommendation_table.md"
        cols = list(table.columns)
        lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in table.iterrows():
            lines.append("|" + "|".join(str(row[c]) for c in cols) + "|")
        md_path.write_text("\n".join(lines))
        render_recommendation_png(table, OUT_DIR / "recommendation_table.png")


def render_recommendation_png(df: pd.DataFrame, out_path: Path) -> None:
    cols = ["abordagem", "melhor_metrica", "pior_metrica", "resumo"]
    df = df[cols].copy()
    df = df.sort_values("abordagem")

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(11.5, max(3.0, 0.5 * n_rows)))
    ax.axis("off")
    table = Table(ax, bbox=[0, 0, 1, 1])
    col_widths = [0.18, 0.22, 0.22, 0.38]
    headers = ["Abordagem", "Melhor métrica", "Pior métrica", "Resumo"]
    for col_idx, (header, width) in enumerate(zip(headers, col_widths)):
        table.add_cell(-1, col_idx, width, 0.08, text=header, loc="center", facecolor="#d9d9d9", fontproperties={"weight": "bold"})
    for row_idx, (_, row) in enumerate(df.iterrows()):
        for col_idx, (col, width) in enumerate(zip(cols, col_widths)):
            table.add_cell(row_idx, col_idx, width, 0.07, text=str(row[col]), loc="left", facecolor="#fdfdfd")
    table.set_fontsize(10.5)
    table.scale(1, 1.08)
    ax.add_table(table)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    reset_out_dir()
    configure_style()
    summary, spec_runs, runs = load_data()
    plot_summary_facets(summary)
    plot_summary_by_gpu(summary)
    plot_per_metric(summary)
    build_recommendations(summary)


if __name__ == "__main__":
    main()

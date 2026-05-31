#!/usr/bin/env python3
"""Reconstrói PR-AUC a partir dos thresholds ROC e plota 6 abordagens.

Os artefatos existentes em ``plots_all`` salvam ROC por run, mas não salvam a
curva precisão-recall. Como os arquivos de thresholds têm ``TPR``/``FPR`` e o
split de validação é conhecido, a precisão pode ser reconstruída exatamente por:

    precision = pi * TPR / (pi * TPR + (1 - pi) * FPR)

onde ``pi`` é a prevalência da classe positiva no split de validação.

Saídas:
    - plots_all/pr_auc_runs.csv
    - plots_all/pr_auc_summary_by_gpu.csv
    - plots_all/pr_auc_summary_overall.csv
    - plots_all/pr_auc_6approaches.{png,pdf}

Executar de dentro de ``projects/hcpa``:
    python3 plots_all/plot_pr_auc_approaches.py
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RUNS_PATH = REPO_ROOT / "single_gpu_runs.csv"

RUNS_OUT_PATH = ROOT / "pr_auc_runs.csv"
SUMMARY_GPU_OUT_PATH = ROOT / "pr_auc_summary_by_gpu.csv"
SUMMARY_OUT_PATH = ROOT / "pr_auc_summary_overall.csv"
PLOT_PNG_PATH = ROOT / "pr_auc_6approaches.png"
PLOT_PDF_PATH = ROOT / "pr_auc_6approaches.pdf"

APPROACH_ORDER: List[str] = [
    "pytorch_base",
    "tensorflow_base",
    "monai_base",
    "pytorch_opt",
    "tensorflow_opt",
    "monai_opt",
]

APPROACH_LABELS: Dict[str, str] = {
    "pytorch_base": "PyTorch Base",
    "tensorflow_base": "TensorFlow Base",
    "monai_base": "MONAI Base",
    "pytorch_opt": "PyTorch Opt",
    "tensorflow_opt": "TensorFlow Opt",
    "monai_opt": "MONAI Opt",
}

# Paleta consistente com os gráficos de stats.
APPROACH_COLORS: Dict[str, str] = {
    "pytorch_base": "#0072B2",
    "tensorflow_base": "#009E73",
    "monai_base": "#56B4E9",
    "pytorch_opt": "#D55E00",
    "tensorflow_opt": "#CC79A7",
    "monai_opt": "#E69F00",
}

APPROACH_HATCHES: Dict[str, str] = {
    "pytorch_base": "/",
    "tensorflow_base": "\\\\",
    "monai_base": "|",
    "pytorch_opt": "-",
    "tensorflow_opt": "+",
    "monai_opt": "x",
}

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

_WARNED_SCIPY = False


def _t_critical(df: int, alpha: float = 0.05) -> float:
    global _WARNED_SCIPY
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception as exc:  # pragma: no cover
        t_val = _T_FALLBACK.get(df, 1.96)
        if not _WARNED_SCIPY:
            warnings.warn(
                f"scipy indisponível para t crítico ({exc}); usando fallback tabelado.",
                RuntimeWarning,
            )
            _WARNED_SCIPY = True
        return t_val


def _ci_from_values(values: Iterable[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean
    std = float(arr.std(ddof=1))
    margin = _t_critical(int(arr.size) - 1, alpha) * std / math.sqrt(int(arr.size))
    return mean - margin, mean + margin


def _find_label_column(df: pd.DataFrame) -> str:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for key in ("retinopatia", "label", "target", "y_true", "class"):
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Nenhuma coluna de label reconhecida em {list(df.columns)}")


def _load_prevalence(project: str, cache: Dict[str, float]) -> float:
    if project in cache:
        return cache[project]

    valid_csv = REPO_ROOT / project / "data" / "csv" / "valid.csv"
    if not valid_csv.exists():
        raise FileNotFoundError(f"Split de validação não encontrado: {valid_csv}")

    df = pd.read_csv(valid_csv)
    label_col = _find_label_column(df)
    labels = pd.to_numeric(df[label_col], errors="coerce").dropna().to_numpy(dtype=float)
    if labels.size == 0:
        raise ValueError(f"Split vazio ou sem labels numéricas em {valid_csv}")

    prevalence = float((labels > 0).mean())
    cache[project] = prevalence
    return prevalence


def _threshold_candidates(run_root: Path, run_id: int) -> Sequence[Path]:
    return [
        run_root / f"all-{run_id}-thresholds.csv",
        run_root / "all-0-thresholds.csv",
        run_root / "val_thresholds.csv",
        run_root / "autotuner_output" / "results" / f"all-{run_id}-thresholds.csv",
        run_root / "autotuner_output" / "results" / "all-0-thresholds.csv",
        run_root / "autotuner_output" / "results" / "val_thresholds.csv",
    ]


def _candidate_run_roots(project: str, result_dir: str, run_id: int) -> Sequence[Path]:
    results_root = REPO_ROOT / project / "results"
    raw = Path(str(result_dir))

    candidates = [
        results_root / raw / f"run_{run_id}",
        results_root / raw.name / f"run_{run_id}",
    ]

    if len(raw.parts) > 1:
        candidates.append(results_root / raw.parts[0] / f"run_{run_id}")
        candidates.append(results_root / raw.parts[-1] / f"run_{run_id}")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        unique_candidates.append(candidate)
    return unique_candidates


def _resolve_threshold_path(run_root: Path, run_id: int) -> Optional[Path]:
    for candidate in _threshold_candidates(run_root, run_id):
        if candidate.exists():
            return candidate
    return None


def _compute_pr_auc_from_thresholds(threshold_path: Path, prevalence: float) -> Tuple[float, int]:
    df = pd.read_csv(threshold_path)

    if {"fpr", "tpr"}.issubset(df.columns):
        fpr = pd.to_numeric(df["fpr"], errors="coerce").to_numpy(dtype=float)
        tpr = pd.to_numeric(df["tpr"], errors="coerce").to_numpy(dtype=float)
    elif {"spec", "sens"}.issubset(df.columns):
        spec = pd.to_numeric(df["spec"], errors="coerce").to_numpy(dtype=float)
        sens = pd.to_numeric(df["sens"], errors="coerce").to_numpy(dtype=float)
        fpr = 1.0 - spec
        tpr = sens
    else:
        raise ValueError(f"Threshold CSV sem colunas ROC reconhecidas: {threshold_path}")

    mask = np.isfinite(fpr) & np.isfinite(tpr)
    fpr = np.clip(fpr[mask], 0.0, 1.0)
    tpr = np.clip(tpr[mask], 0.0, 1.0)
    if fpr.size == 0:
        raise ValueError(f"Threshold CSV vazio após limpeza: {threshold_path}")

    order = np.lexsort((fpr, tpr))
    fpr = fpr[order]
    tpr = tpr[order]

    denom = prevalence * tpr + (1.0 - prevalence) * fpr
    precision = np.divide(
        prevalence * tpr,
        denom,
        out=np.ones_like(tpr),
        where=denom > 0,
    )
    recall = tpr
    trapz = getattr(np, "trapezoid", np.trapz)
    pr_auc = float(trapz(precision, recall))
    return pr_auc, int(recall.size)


def build_runs_table() -> pd.DataFrame:
    if not RUNS_PATH.exists():
        raise SystemExit(f"Arquivo não encontrado: {RUNS_PATH}")

    runs = pd.read_csv(RUNS_PATH)
    prevalence_cache: Dict[str, float] = {}
    rows = []
    missing = []

    for row in runs.itertuples():
        project = str(row.project)
        gpu = str(row.gpu)
        run_id = int(row.run_id)
        result_dir = getattr(row, "result_dir", None)
        if result_dir is None or pd.isna(result_dir):
            missing.append(f"{project}/{gpu}/run_{run_id}: result_dir ausente")
            continue

        threshold_path = None
        for run_root in _candidate_run_roots(project, str(result_dir), run_id):
            threshold_path = _resolve_threshold_path(run_root, run_id)
            if threshold_path is not None:
                break
        if threshold_path is None:
            missing.append(f"{project}/{gpu}/run_{run_id}: threshold CSV ausente")
            continue

        prevalence = _load_prevalence(project, prevalence_cache)
        pr_auc, n_points = _compute_pr_auc_from_thresholds(threshold_path, prevalence)

        rows.append(
            {
                "project": project,
                "gpu": gpu,
                "run_id": run_id,
                "result_dir": str(result_dir),
                "positive_prevalence": prevalence,
                "pr_auc": pr_auc,
                "curve_points": n_points,
                "threshold_source": threshold_path.relative_to(REPO_ROOT).as_posix(),
            }
        )

    if missing:
        preview = "; ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" ... (+{len(missing) - 8} runs)"
        warnings.warn(f"Runs ignoradas sem thresholds: {preview}{suffix}", RuntimeWarning)

    if not rows:
        raise SystemExit("Nenhuma run com thresholds ROC foi encontrada para reconstruir PR-AUC.")

    df = pd.DataFrame(rows)
    df["project"] = pd.Categorical(df["project"], categories=APPROACH_ORDER, ordered=True)
    df = df.sort_values(["project", "gpu", "run_id"]).reset_index(drop=True)
    df.to_csv(RUNS_OUT_PATH, index=False)
    return df


def summarize_runs(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), observed=True)["pr_auc"]
    summary = (
        grouped.agg(pr_auc_mean="mean", pr_auc_std="std", runs="count")
        .reset_index()
    )

    ci_rows = []
    for key, values in grouped:
        record = {}
        if isinstance(key, tuple):
            for col, val in zip(group_cols, key):
                record[col] = val
        else:
            record[group_cols[0]] = key
        low, high = _ci_from_values(values.to_numpy(dtype=float))
        record["pr_auc_ci95_low"] = low
        record["pr_auc_ci95_high"] = high
        ci_rows.append(record)

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on=list(group_cols), how="left")
    summary["pr_auc_std"] = summary["pr_auc_std"].fillna(0.0)
    summary["pr_auc_ci95_halfwidth"] = (
        summary["pr_auc_ci95_high"] - summary["pr_auc_mean"]
    ).clip(lower=0.0)
    summary["project_label"] = summary["project"].map(APPROACH_LABELS)
    summary["project"] = pd.Categorical(summary["project"], categories=APPROACH_ORDER, ordered=True)
    summary = summary.sort_values(list(group_cols)).reset_index(drop=True)
    return summary


def plot_overall(summary: pd.DataFrame) -> None:
    summary = summary.copy()
    summary = summary[summary["project"].isin(APPROACH_ORDER)]
    summary["project"] = pd.Categorical(summary["project"], categories=APPROACH_ORDER, ordered=True)
    summary = summary.sort_values("project").reset_index(drop=True)

    means = summary["pr_auc_mean"].to_numpy(dtype=float) * 100.0
    ci_half = summary["pr_auc_ci95_halfwidth"].to_numpy(dtype=float) * 100.0
    yerr = np.vstack([ci_half, ci_half])

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
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "hatch.color": "#00000014",
            "hatch.linewidth": 0.20,
        }
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=5,
        width=0.72,
        edgecolor="#222222",
        linewidth=0.8,
        color=[APPROACH_COLORS[str(project)] for project in summary["project"]],
        error_kw={"elinewidth": 1.2},
    )

    for bar, project in zip(bars, summary["project"]):
        bar.set_hatch(APPROACH_HATCHES[str(project)])

    for bar, mean_val, ci_val in zip(bars, means, ci_half):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean_val + ci_val + 0.25,
            f"{mean_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    xtick_labels = [
        f"{APPROACH_LABELS[str(project)]}\n(n={int(runs)})"
        for project, runs in zip(summary["project"], summary["runs"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=18, ha="right")
    ax.set_ylabel("PR-AUC (%)")
    ax.set_title("PR-AUC por abordagem (todas as execuções)")
    ax.text(
        0.99,
        0.98,
        "Erro = IC95% entre runs",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
    )
    ax.set_axisbelow(True)

    ymin = max(0.0, float(np.floor((np.nanmin(means - ci_half) - 1.0) / 2.0) * 2.0))
    ymax = min(100.0, float(np.ceil((np.nanmax(means + ci_half) + 1.2) / 2.0) * 2.0))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin >= ymax:
        ymin, ymax = 0.0, 100.0
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(PLOT_PNG_PATH, bbox_inches="tight")
    fig.savefig(PLOT_PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs_df = build_runs_table()
    summary_by_gpu = summarize_runs(runs_df, ["gpu", "project"])
    summary_overall = summarize_runs(runs_df, ["project"])

    summary_by_gpu.to_csv(SUMMARY_GPU_OUT_PATH, index=False)
    summary_overall.to_csv(SUMMARY_OUT_PATH, index=False)
    plot_overall(summary_overall)

    printable = summary_overall.copy()
    printable["pr_auc_mean_pct"] = printable["pr_auc_mean"] * 100.0
    printable["pr_auc_ci95_low_pct"] = printable["pr_auc_ci95_low"] * 100.0
    printable["pr_auc_ci95_high_pct"] = printable["pr_auc_ci95_high"] * 100.0
    print(printable[["project", "runs", "pr_auc_mean_pct", "pr_auc_ci95_low_pct", "pr_auc_ci95_high_pct"]].to_string(index=False))
    print(f"Saved: {RUNS_OUT_PATH}")
    print(f"Saved: {SUMMARY_GPU_OUT_PATH}")
    print(f"Saved: {SUMMARY_OUT_PATH}")
    print(f"Saved: {PLOT_PNG_PATH}")
    print(f"Saved: {PLOT_PDF_PATH}")


if __name__ == "__main__":
    main()

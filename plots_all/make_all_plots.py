#!/usr/bin/env python3
"""Gera gráficos all_*.png com subplots por GPU e legenda única.

Regras atualizadas:
- Especificidade vem do ponto em que a sensibilidade atinge 0.95 (spec_at_sens95).
- Sensibilidade é mostrada fixa em 0.95 (sem desvio) apenas como referência visual.
- Tabelas:
    * Compacta: GPU/Projeto + médias±DP; adiciona IC95% teste somente se disponível.
    * Suplementar: inclui IC entre runs, IC teste, contagens e unidade.
    * Entre runs: média ± t*std/√n (t crítico de scipy ou fallback tabelado).
    * No teste: bootstrap estratificado por paciente (se patient_id), senão por amostra.

Executar de dentro de projects/hcpa:
    python3 plots_all/make_all_plots.py
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hcpa_all_summary.csv"
SPEC95_SUMMARY_PATH = ROOT / "spec_at_sens95_summary.csv"
SPEC95_RUNS_PATH = ROOT / "spec_at_sens95_runs.csv"
RUNS_PATH = ROOT.parent / "single_gpu_runs.csv"
SENS_TARGET = 0.95
RULE_DESC = (
    "Spec@Sens=0.95: especificidade no limiar interpolado (ROC do VAL) que atinge sens=0.95; "
    "limiar definido no VAL e aplicado no TESTE por run; sensibilidade alvo fixa em 0.95."
)


def recompute_spec_runs() -> None:
    """Reconstroi spec_at_sens95_runs.csv com colunas explícitas de regra/limiar."""
    if not RUNS_PATH.exists():
        return

    runs = pd.read_csv(RUNS_PATH)
    rows = []
    for row in runs.itertuples():
        project = row.project
        gpu = row.gpu
        run_id = int(row.run_id)
        result_dir = getattr(row, "result_dir", None)
        if not result_dir:
            continue
        run_root = ROOT.parent / project / "results" / result_dir / f"run_{run_id}"
        thr_path = run_root / f"all-{run_id}-thresholds.csv"
        if not thr_path.exists():
            alt_thr = run_root / "val_thresholds.csv"
            thr_path = alt_thr if alt_thr.exists() else None
        if thr_path is None or not thr_path.exists():
            continue
        thr_df = pd.read_csv(thr_path)
        if not {"sens", "spec", "thresholds"}.issubset(thr_df.columns):
            continue
        fpr_arr = thr_df["fpr"].to_numpy() if "fpr" in thr_df.columns else 1 - thr_df["spec"].to_numpy()
        tpr_arr = thr_df["tpr"].to_numpy() if "tpr" in thr_df.columns else thr_df["sens"].to_numpy()

        spec_at_sens95, thr_at_sens95, sens_interp = _interp_spec_at_sens(
            fpr_arr, tpr_arr, thr_df["thresholds"].to_numpy(), SENS_TARGET
        )

        # sens_at_spec95 (para retrocompatibilidade)
        try:
            spec_target = 0.95
            order = np.argsort(thr_df["spec"].to_numpy())
            spec_sorted = thr_df["spec"].to_numpy()[order]
            sens_sorted = thr_df["sens"].to_numpy()[order]
            thr_sorted = thr_df["thresholds"].to_numpy()[order]
            above = np.searchsorted(spec_sorted, spec_target, side="left")
            if above == 0 or above >= len(spec_sorted):
                sens_at_spec95 = float("nan")
                thr_at_spec95 = float("nan")
            else:
                s0, s1 = spec_sorted[above - 1], spec_sorted[above]
                t0, t1 = thr_sorted[above - 1], thr_sorted[above]
                sen0, sen1 = sens_sorted[above - 1], sens_sorted[above]
                w = (spec_target - s0) / (s1 - s0)
                sens_at_spec95 = sen0 + w * (sen1 - sen0)
                thr_at_spec95 = t0 + w * (t1 - t0)
        except Exception:
            sens_at_spec95 = float("nan")
            thr_at_spec95 = float("nan")

        rows.append(
            {
                "project": project,
                "gpu": gpu,
                "run_id": run_id,
                "spec_at_sens95": spec_at_sens95,
                "thr_at_sens95": thr_at_sens95,
                "achieved_sens": sens_interp,
                "achieved_spec": spec_at_sens95,
                "rule": "interpolated_sens_target_0.95",
                "split_used_for_threshold": "val",
                "sens_at_spec95": sens_at_spec95,
                "thr_at_spec95": thr_at_spec95,
                "val_auc_final": getattr(row, "val_auc_final", float("nan")),
            }
        )

    if rows:
        new_df = pd.DataFrame(rows)
        if SPEC95_RUNS_PATH.exists():
            old_df = pd.read_csv(SPEC95_RUNS_PATH)
            key_cols = ["project", "gpu", "run_id"]
            mask = ~old_df[key_cols].apply(tuple, axis=1).isin(new_df[key_cols].apply(tuple, axis=1))
            old_remaining = old_df[mask]
            new_df = pd.concat([new_df, old_remaining], ignore_index=True)
        new_df.to_csv(SPEC95_RUNS_PATH, index=False)
OUT_DIR = ROOT

# Ordem fixa para cores/legenda.
APPROACH_ORDER: List[str] = [
    "tensorflow_opt",
    "pytorch_opt",
    "monai_base",
    "tensorflow_base",
    "pytorch_base",
    "monai_opt",
]

APPROACH_LABELS = {
    "tensorflow_opt": "TensorFlow Opt",
    "pytorch_opt": "PyTorch Opt",
    "monai_opt": "MONAI Opt",
    "tensorflow_base": "TensorFlow Base",
    "pytorch_base": "PyTorch Base",
    "monai_base": "MONAI Base",
}

# Paleta simples, legível e consistente.
APPROACH_COLORS = {
    # Mesma cor para variantes do mesmo framework
    "tensorflow_opt": "#1f77b4",        # TensorFlow
    "tensorflow_base": "#1f77b4",
    "pytorch_opt": "#ff7f0e",         # PyTorch
    "pytorch_base": "#ff7f0e",
    "monai_base": "#9467bd",            # MONAI (base)
    "monai_opt": "#2ca02c",  # MONAI
}

# (base_name, mean_col, std_col, eixo_y, formato)
METRICS: List[Tuple[str, str, str, str, str]] = [
    ("auc", "auc_mean", "auc_std", "AUC (ROC) (%)", "{:.1f}%"),
    ("spec", "spec_mean", "spec_std", "Especificidade @Sens=0.95", "{:.3f}"),
    ("sens", "sens_mean", "sens_std", "Sensibilidade (alvo=0.95)", "{:.3f}"),
    ("throughput", "throughput_mean", "throughput_std", "Throughput (img/s)", "{:.0f}"),
    ("train_time", "train_time_mean", "train_time_std", "Tempo de treino (s)", "{:.0f}"),
    ("mem", "mem_mean", "mem_std", "Memória de pico (MB)", "{:.0f}"),
    ("tta", "tta_mean", "tta_std", "Tempo para AUC 0.95 (s)", "{:.1f}"),
]

# Grupos opcionais para gerar comparações separadas
GROUPS = [
    ("", None),  # todos
    ("_core", ["tensorflow_opt", "pytorch_opt", "monai_opt"]),
    ("_clean", ["tensorflow_base", "pytorch_base"]),
]


def _pad_ylim(current_max: float) -> float:
    return current_max * 1.12 if current_max > 0 else 1.0


def _ci_runs(mean: float, std: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Intervalo de confiança da média entre runs (t-Student)."""
    if n <= 1 or math.isnan(mean) or math.isnan(std):
        return float("nan"), float("nan")
    t_val = _t_critical(n - 1, alpha)
    margin = t_val * std / math.sqrt(n)
    return mean - margin, mean + margin


def _format_mean_std_ci(mean: float, std: float, n: int) -> str:
    if math.isnan(mean) or math.isnan(std):
        return "n/a"
    return f"{mean:.3f} ± {std:.3f} (n={n})"


def _format_ci(lower: float, upper: float) -> str:
    if math.isnan(lower) or math.isnan(upper):
        return "n/a"
    return f"{lower:.3f}-{upper:.3f}"


def _format_mean_std_ci_percent(mean: float, std: float, n: int, decimals: int = 1) -> str:
    """Formata média/DP convertendo para porcentagem."""
    if math.isnan(mean) or math.isnan(std):
        return "n/a"
    mean_pct = mean * 100
    std_pct = std * 100
    return f"{mean_pct:.{decimals}f}% ± {std_pct:.{decimals}f}% (n={n})"


def _format_ci_percent(lower: float, upper: float, decimals: int = 1) -> str:
    """Formata intervalo de confiança em porcentagem."""
    if math.isnan(lower) or math.isnan(upper):
        return "n/a"
    lower_pct = lower * 100
    upper_pct = upper * 100
    return f"{lower_pct:.{decimals}f}-{upper_pct:.{decimals}f}%"


def _drop_na_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas em que todos os valores são 'n/a' ou NaN."""
    keep_cols = []
    for col in df.columns:
        series = df[col]
        if series.isna().all():
            continue
        if (series.astype(str) == "n/a").all():
            continue
        keep_cols.append(col)
    return df[keep_cols]


def _interp_spec_at_sens(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, target: float):
    """Retorna (spec, threshold, sens_interp) no ponto interpolado de sens=target.
    Regra: usar interpolação linear entre os dois pontos de tpr que cercam o alvo.
    """
    order = np.argsort(tpr)
    tpr_sorted = tpr[order]
    fpr_sorted = fpr[order]
    thr_sorted = thresholds[order]
    above = np.searchsorted(tpr_sorted, target, side="left")
    if above == 0 or above >= len(tpr_sorted):
        return float("nan"), float("nan"), float("nan")
    tpr0, tpr1 = tpr_sorted[above - 1], tpr_sorted[above]
    fpr0, fpr1 = fpr_sorted[above - 1], fpr_sorted[above]
    thr0, thr1 = thr_sorted[above - 1], thr_sorted[above]
    if tpr1 == tpr0:
        w = 0.0
    else:
        w = (target - tpr0) / (tpr1 - tpr0)
    fpr_interp = fpr0 + w * (fpr1 - fpr0)
    thr_interp = thr0 + w * (thr1 - thr0)
    sens_interp = tpr0 + w * (tpr1 - tpr0)
    spec_interp = 1 - fpr_interp
    return spec_interp, thr_interp, sens_interp


def _compute_roc_metrics(y_true, y_score, target_sens: float = 0.95):
    """Calcula AUC e Spec@Sens alvo a partir de scores."""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"sklearn não disponível para CI de teste ({exc})")
        return math.nan, math.nan

    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec, thr = _interp_spec_at_sens(fpr, tpr, thresholds, target_sens)
    return auc, spec


def _bootstrap_test_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 0,
    target_sens: float = SENS_TARGET,
) -> Tuple[Tuple[float, float], Tuple[float, float], int, int, str]:
    """Bootstrap estratificado (cluster por paciente quando disponível)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids)
        if len(patient_ids) != len(y_true):
            patient_ids = None  # fallback para não quebrar

    # separa grupos para bootstrap estratificado
    if patient_ids is not None:
        unit = "paciente"
        pos_pat = np.unique(patient_ids[y_true == 1])
        neg_pat = np.unique(patient_ids[y_true == 0])
        n_pos, n_neg = len(pos_pat), len(neg_pat)
    else:
        unit = "amostra"
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return (math.nan, math.nan), (math.nan, math.nan), n_pos, n_neg, unit

    auc_samples = []
    spec_samples = []
    for _ in range(n_boot):
        if patient_ids is not None:
            sampled_pos = rng.choice(pos_pat, size=n_pos, replace=True)
            sampled_neg = rng.choice(neg_pat, size=n_neg, replace=True)
            keep_pats = np.concatenate([sampled_pos, sampled_neg])
            mask = np.isin(patient_ids, keep_pats)
            idx = np.where(mask)[0]
        else:
            sampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            sampled_neg = rng.choice(neg_idx, size=n_neg, replace=True)
            idx = np.concatenate([sampled_pos, sampled_neg])

        auc_b, spec_b = _compute_roc_metrics(y_true[idx], y_score[idx], target_sens)
        if not math.isnan(auc_b):
            auc_samples.append(auc_b)
        if not math.isnan(spec_b):
            spec_samples.append(spec_b)

    def _ci_from_samples(samples: List[float]) -> Tuple[float, float]:
        if len(samples) == 0:
            return float("nan"), float("nan")
        low, high = np.percentile(samples, [2.5, 97.5])
        return float(low), float(high)

    auc_ci = _ci_from_samples(auc_samples)
    spec_ci = _ci_from_samples(spec_samples)
    return auc_ci, spec_ci, n_pos, n_neg, unit


def _load_test_preds(run_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Procura arquivos de predição de teste no run_*:
    aceita colunas y_true / label e y_score / prob / logit; patient_id opcional.
    """
    if not run_path.exists():
        return None

    candidates = sorted(run_path.glob("*.csv"))
    # prioriza arquivos com 'pred' ou 'test' no nome
    candidates = sorted(
        candidates,
        key=lambda p: (
            "pred" not in p.stem.lower() and "test" not in p.stem.lower(),
            p.name,
        ),
    )
    for csv_path in candidates:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        y_col = cols.get("y_true") or cols.get("label") or cols.get("target")
        score_col = (
            cols.get("y_score")
            or cols.get("prob")
            or cols.get("score")
            or cols.get("logit")
            or cols.get("pred")
        )
        if y_col and score_col:
            pid_col = cols.get("patient_id") or cols.get("subject_id")
            y_true = df[y_col].to_numpy()
            y_score = df[score_col].to_numpy()
            patient_ids = df[pid_col].to_numpy() if pid_col else None
            return y_true, y_score, patient_ids
    return None

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


def _t_critical(df: int, alpha: float = 0.05) -> float:
    """Valor crítico t bilateral para o nível desejado (default 95%)."""
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(1 - alpha / 2, df))
    except Exception as exc:  # pragma: no cover - fallback para ambientes sem scipy
        if df in _T_FALLBACK:
            t_val = _T_FALLBACK[df]
        else:
            t_val = 1.96  # aproximação normal para df>30
        warnings.warn(
            f"scipy não disponível para t crítico ({exc}); usando tabela/fallback ({t_val})",
            RuntimeWarning,
        )
        return t_val


def plot_metric(
    df: pd.DataFrame,
    base: str,
    mean_col: str,
    std_col: str,
    ylabel: str,
    fmt: str,
    project_filter: Optional[List[str]] = None,
    suffix: str = "",
) -> None:
    # AUC em porcentagem (0-100) com 1 casa; demais métricas seguem escala original.
    scale = 100.0 if mean_col == "auc_mean" else 1.0
    fmt_local = "{:.1f}%" if mean_col == "auc_mean" else fmt
    ylabel_local = ylabel

    gpus = ["RTX4090", "L40S", "H200"]
    fig, axes = plt.subplots(1, len(gpus), figsize=(11.5, 3.6), sharey=True)

    metric_max = (df[mean_col].dropna() * scale).max()
    upper_ylim = _pad_ylim(metric_max)

    approaches = [p for p in APPROACH_ORDER if (project_filter is None or p in project_filter)]

    for ax, gpu in zip(axes, gpus):
        subset = df[df["gpu"] == gpu]
        xtick_labels: List[Tuple[int, str]] = []

        for idx, project in enumerate(approaches):
            row = subset[subset["project"] == project]
            if row.empty or pd.isna(row.iloc[0][mean_col]):
                continue

            mean_val = float(row.iloc[0][mean_col]) * scale
            std_val = float(row.iloc[0][std_col]) * scale if not pd.isna(row.iloc[0][std_col]) else 0.0

            bar_container = ax.bar(
                idx,
                mean_val,
                yerr=std_val,
                color=APPROACH_COLORS[project],
                edgecolor="0.2",
                linewidth=0.8,
                capsize=4,
                width=0.75,
            )
            ax.bar_label(
                bar_container,
                labels=[fmt_local.format(mean_val)],
                padding=3,
                fontsize=8,
            )
            xtick_labels.append((idx, APPROACH_LABELS[project]))

        ax.set_xticks([pos for pos, _ in xtick_labels])
        ax.set_xticklabels([lab for _, lab in xtick_labels], rotation=25, ha="right")
        ax.set_title(gpu, fontsize=11, pad=10)
        ax.set_ylabel(ylabel_local)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(0, upper_ylim)
        ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=APPROACH_COLORS[p], edgecolor="0.2", label=APPROACH_LABELS[p])
        for p in approaches
    ]
    fig.legend(
        handles=legend_handles,
        ncol=2,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=False,
        title="Abordagem",
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    outfile = OUT_DIR / f"all_{base}{suffix}.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"✔ salvo: {outfile}")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(f"Arquivo não encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Se existir o arquivo de runs, recalcula médias e desvios amostrais
    # para AUC e especificidade no ponto sens=0.95 (mais preciso).
    if SPEC95_RUNS_PATH.exists():
        runs = pd.read_csv(SPEC95_RUNS_PATH)
        agg = (
            runs.groupby(["gpu", "project"])
            .agg(
                spec_mean=("spec_at_sens95", "mean"),
                spec_std=("spec_at_sens95", "std"),  # desvio amostral
                auc_mean=("val_auc_final", "mean"),
                auc_std=("val_auc_final", "std"),
                runs=("spec_at_sens95", "count"),
            )
            .reset_index()
        )
        df = df.merge(agg, on=["gpu", "project"], how="left", suffixes=("", "_runs"))
        for col in ["spec_mean", "spec_std", "auc_mean", "auc_std"]:
            df[col] = df[f"{col}_runs"].combine_first(df[col])
            df.drop(columns=[f"{col}_runs"], inplace=True)
    elif SPEC95_SUMMARY_PATH.exists():
        # fallback: usa o summary pronto
        spec95 = pd.read_csv(SPEC95_SUMMARY_PATH)[["gpu", "project", "spec95_mean", "spec95_std"]]
        df = df.merge(spec95, on=["gpu", "project"], how="left")
        df["spec_mean"] = df["spec95_mean"].combine_first(df.get("spec_mean"))
        df["spec_std"] = df["spec95_std"].combine_first(df.get("spec_std"))
        df.drop(columns=["spec95_mean", "spec95_std"], inplace=True)

    # Sensibilidade fixada no alvo 0.95 para todos, sem desvio.
    df["sens_mean"] = 0.95
    df["sens_std"] = 0.0
    return df


def write_table_from_runs() -> None:
    """Gera tabelas compacta e suplementar com CIs entre runs e de teste."""
    if not SPEC95_RUNS_PATH.exists():
        print("[tabela] spec_at_sens95_runs.csv ausente; mantendo tabela anterior.")
        return

    runs = pd.read_csv(SPEC95_RUNS_PATH)
    run_meta = pd.read_csv(RUNS_PATH) if RUNS_PATH.exists() else pd.DataFrame()
    run_lookup = {
        (row.project, row.gpu, int(row.run_id)): row.result_dir
        for row in run_meta.itertuples()
        if hasattr(row, "result_dir")
    }

    rows = []
    any_test_auc = False
    any_test_spec = False
    gpu_order = ["L40S", "RTX4090", "H200", "A100", "H100"]
    for (gpu, project), grp in runs.groupby(["gpu", "project"]):
        n = len(grp)

        # Estatística entre runs (média/DP + IC da média)
        auc_mean = grp["val_auc_final"].mean()
        auc_std = grp["val_auc_final"].std(ddof=1)
        auc_ci_runs = _ci_runs(auc_mean, auc_std, n)

        spec_mean = grp["spec_at_sens95"].mean()
        spec_std = grp["spec_at_sens95"].std(ddof=1)
        spec_ci_runs = _ci_runs(spec_mean, spec_std, n)

        # IC de teste (bootstrap/DeLong) – usa o primeiro run que tiver preds disponíveis
        test_auc_ci = (float("nan"), float("nan"))
        test_spec_ci = (float("nan"), float("nan"))
        n_pos = n_neg = 0
        unit = "n/a"
        for run in grp.itertuples():
            key = (run.project, run.gpu, int(run.run_id))
            result_dir = run_lookup.get(key)
            if not result_dir:
                continue
            run_path = ROOT.parent / project / "results" / result_dir / f"run_{run.run_id}"
            preds = _load_test_preds(run_path)
            if preds is None:
                continue
            y_true, y_score, patient_ids = preds
            auc_ci, spec_ci, n_pos, n_neg, unit = _bootstrap_test_ci(
                y_true, y_score, patient_ids, n_boot=2000, seed=0, target_sens=0.95
            )
            test_auc_ci = auc_ci
            test_spec_ci = spec_ci
            if not any(math.isnan(x) for x in auc_ci):
                any_test_auc = True
            if not any(math.isnan(x) for x in spec_ci):
                any_test_spec = True
            break  # usa o primeiro run com dados de teste

        rows.append(
            {
                "GPU": gpu,
                "Projeto": project,
                "Runs": n,
                "AUC_mean": auc_mean,
                "AUC_std": auc_std,
                "AUC_ci_runs": auc_ci_runs,
                "AUC_ci_test": test_auc_ci,
                "Spec_mean": spec_mean,
                "Spec_std": spec_std,
                "Spec_ci_runs": spec_ci_runs,
                "Spec_ci_test": test_spec_ci,
                "Sens (fixo=0.95)": "0.950 (fixo)",
                "N_pos (teste)": n_pos if n_pos else "n/a",
                "N_neg (teste)": n_neg if n_neg else "n/a",
                "Unidade (teste)": unit,
            }
        )

    full_df = pd.DataFrame(rows)
    full_df["GPU"] = pd.Categorical(full_df["GPU"], categories=gpu_order, ordered=True)
    full_df = full_df.sort_values(["GPU", "Projeto"])

    # ---- Tabela suplementar (completa, CSV)
    supp_df = pd.DataFrame(
        {
            "GPU": full_df["GPU"],
            "Projeto": full_df["Projeto"],
            "Runs": full_df["Runs"],
            "AUC (média ± DP, n=runs)": [
                _format_mean_std_ci_percent(m, s, n) for m, s, n in zip(full_df["AUC_mean"], full_df["AUC_std"], full_df["Runs"])
            ],
            "AUC (IC95% entre runs)": [_format_ci_percent(*ci) for ci in full_df["AUC_ci_runs"]],
            "AUC (IC95% teste, bootstrap)": [_format_ci_percent(*ci) for ci in full_df["AUC_ci_test"]],
            "Spec@Sens=0.95 (média ± DP, n=runs)": [
                _format_mean_std_ci(m, s, n) for m, s, n in zip(full_df["Spec_mean"], full_df["Spec_std"], full_df["Runs"])
            ],
            "Spec@Sens=0.95 (IC95% entre runs)": [_format_ci(*ci) for ci in full_df["Spec_ci_runs"]],
            "Spec@Sens=0.95 (IC95% teste, bootstrap)": [_format_ci(*ci) for ci in full_df["Spec_ci_test"]],
            "Sens (fixo=0.95)": full_df["Sens (fixo=0.95)"],
            "N_pos (teste)": full_df["N_pos (teste)"],
            "N_neg (teste)": full_df["N_neg (teste)"],
            "Unidade (teste)": full_df["Unidade (teste)"],
            "Regra Spec@Sens=0.95": RULE_DESC,
            "Split limiar": "val",
        }
    )
    supp_csv = OUT_DIR / "spec_sens_table_supplement.csv"
    supp_df.to_csv(supp_csv, index=False)
    print(f"✔ salvo: {supp_csv}")

    # ---- Tabela compacta (paper)
    compact_columns = {
        "GPU": full_df["GPU"],
        "Projeto": full_df["Projeto"],
        "AUC (média ± DP, n=runs)": [
            _format_mean_std_ci_percent(m, s, n) for m, s, n in zip(full_df["AUC_mean"], full_df["AUC_std"], full_df["Runs"])
        ],
        "Spec@Sens=0.95 (média ± DP, n=runs)": [
            _format_mean_std_ci(m, s, n) for m, s, n in zip(full_df["Spec_mean"], full_df["Spec_std"], full_df["Runs"])
        ],
    }
    if any_test_auc:
        compact_columns["AUC (IC95% teste, bootstrap)"] = [_format_ci_percent(*ci) for ci in full_df["AUC_ci_test"]]
    if any_test_spec:
        compact_columns["Spec@Sens=0.95 (IC95% teste, bootstrap)"] = [_format_ci(*ci) for ci in full_df["Spec_ci_test"]]

    compact_df = pd.DataFrame(compact_columns)
    compact_df = _drop_na_columns(compact_df)

    # Rodapé para a figura (condicional aos dados disponíveis)
    footnotes = [
        "média ± DP: variação entre runs (seeds/treinos).",
        "AUC (média ± DP) calculada no split de validação (val_auc_final).",
        "AUC nas tabelas está em porcentagem (0–100%).",
        RULE_DESC,
    ]
    if any_test_auc or any_test_spec:
        footnotes.append(
            "IC95% teste: bootstrap estratificado (por paciente se houver patient_id, senão por amostra)."
        )
    unique_runs = full_df["Runs"].unique()
    if len(unique_runs) == 1:
        footnotes.append(f"Runs por GPU/projeto: {unique_runs[0]}.")

    compact_csv = OUT_DIR / "spec_sens_table_paper_compact.csv"
    compact_df.to_csv(compact_csv, index=False)
    print(f"✔ salvo: {compact_csv}")

    # PNG da tabela compacta com rodapé
    fig_width = max(9, 2 + 2.0 * len(compact_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.axis("off")
    table = ax.table(
        cellText=compact_df.values,
        colLabels=compact_df.columns,
        loc="center",
        cellLoc="center",
    )
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#444")
        if key[0] == 0:
            cell.set_fontsize(9)
    footer_text = " ".join(footnotes)
    fig.text(0.01, 0.02, footer_text, ha="left", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    compact_png = OUT_DIR / "spec_sens_table_paper_compact.png"
    fig.savefig(compact_png, dpi=300)
    plt.close(fig)
    print(f"✔ salvo: {compact_png}")


def main() -> int:
    # Recalcula spec_at_sens95_runs com metadados de regra/limiar.
    recompute_spec_runs()

    df = load_data()
    for base, mean_col, std_col, ylabel, fmt in METRICS:
        if mean_col not in df.columns or std_col not in df.columns:
            print(f"[ignorado] colunas ausentes para {base}")
            continue
        for suffix, projects in GROUPS:
            plot_metric(df, base, mean_col, std_col, ylabel, fmt, project_filter=projects, suffix=suffix)

    write_table_from_runs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

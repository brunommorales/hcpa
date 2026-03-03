#!/usr/bin/env python3
"""
Gera gráficos prontos para publicação (PDF/SVG + PNG) usando os resumos já
calculados pelo pipeline do projeto.

Características visuais (padrão acadêmico):
- Barras com desvios padrão (yerr) para cada métrica.
- Paleta consistente e compatível com preto-e-branco (cores + hachuras).
- Rótulos numéricos com valor±DP posicionados acima das barras com margem
  automática para evitar colisões.
- Títulos curtos, eixos legíveis e fonte adequada para impressão.

Entradas (já existentes no repositório):
- plots/hcpa_summary.csv               → médias/DP principais
- plots_all/spec_at_sens95_summary.csv → métricas @95% (sens/spec)

Saídas (criadas em paper_plots/):
- Um PDF e um PNG para cada métrica listada em METRICS.

Como executar (de dentro de projects/hcpa):
    python3 make_publication_plots.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuração visual
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 400,  # alta resolução para impressão
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Cores + hachuras para funcionar em PB.
PROJECT_ORDER = ["tensorflow_opt", "pytorch_opt"]
PALETTE: Dict[str, str] = {
    "tensorflow_opt": "#4c78a8",  # azul escuro
    "pytorch_opt": "#f58518",  # laranja
}
HATCHES: Dict[str, str] = {
    "tensorflow_opt": "",
    "pytorch_opt": "//",
}


# ---------------------------------------------------------------------------
# Estrutura de métricas
# ---------------------------------------------------------------------------
@dataclass
class MetricDef:
    key: str
    std_key: str
    label: str
    fmt: str
    unit: str | None = None
    ylim_pad_frac: float = 0.08  # espaço extra no topo


METRICS: List[MetricDef] = [
    MetricDef("auc_mean", "auc_std", "AUC (ROC)", "{:.3f}"),
    MetricDef("spec_mean", "spec_std", "Especificidade @Sens=0.95", "{:.3f}"),
    MetricDef("sens_mean", "sens_std", "Sensibilidade @Spec=0.95", "{:.3f}"),
    MetricDef("throughput_mean", "throughput_std", "Throughput", "{:.0f}", unit="img/s"),
    MetricDef("train_time_mean", "train_time_std", "Tempo de treino", "{:.0f}", unit="s"),
    MetricDef("mem_mean", "mem_std", "Memória de pico", "{:.0f}", unit="MB"),
    MetricDef("tta_mean", "tta_std", "Tempo p/ AUC 0.95", "{:.1f}", unit="s"),
]


# ---------------------------------------------------------------------------
# Leitura e preparação dos dados
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "plots" / "hcpa_summary.csv"
SPECSENS_PATH = ROOT / "plots_all" / "spec_at_sens95_summary.csv"
OUT_DIR = ROOT / "paper_plots"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Arquivo não encontrado: {SUMMARY_PATH}")
    if not SPECSENS_PATH.exists():
        raise SystemExit(f"Arquivo não encontrado: {SPECSENS_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    spec = pd.read_csv(SPECSENS_PATH).rename(
        columns={
            "spec95_mean": "spec_mean",
            "spec95_std": "spec_std",
            "sens_at_spec95_mean": "sens_mean",
            "sens_at_spec95_std": "sens_std",
            "auc_mean": "auc_mean_spec",
            "auc_std": "auc_std_spec",
        }
    )

    # Mantém apenas colunas relevantes e mescla pelas chaves.
    spec = spec[["gpu", "project", "spec_mean", "spec_std", "sens_mean", "sens_std"]]
    merged = df.merge(spec, on=["gpu", "project"], how="left", suffixes=("", "_specfile"))

    # Se houver valores de spec/sens do arquivo de thresholds, sobrepõe.
    for col in ["spec_mean", "spec_std", "sens_mean", "sens_std"]:
        merged[col] = merged[f"{col}_specfile"].combine_first(merged[col])
        merged.drop(columns=[f"{col}_specfile"], inplace=True)

    # Ordena projetos para consistência visual.
    merged["project"] = pd.Categorical(merged["project"], categories=PROJECT_ORDER, ordered=True)
    merged = merged.sort_values(["gpu", "project"])
    return merged


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def format_value(val: float | None, fmt: str) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "NA"
    return fmt.format(val)


def add_value_labels(ax, bars, means, stds, fmt: str):
    """
    Escreve valor±dp acima das barras com offset proporcional ao DP
    e pequena margem adicional para evitar colisões.
    """
    for bar, mean, std in zip(bars, means, stds):
        if mean is None or not math.isfinite(mean):
            continue
        std_val = std if std is not None and math.isfinite(std) else 0.0
        offset = std_val * 1.1 + abs(bar.get_height()) * 0.02
        y = bar.get_height() + offset
        text = f"{format_value(mean, fmt)}±{format_value(std_val, fmt)}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            ha="center",
            va="bottom",
            s=text,
            fontsize=8,
        )


def plot_metric(df: pd.DataFrame, metric: MetricDef):
    gpus = df["gpu"].unique().tolist()
    x = np.arange(len(gpus))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for idx, project in enumerate(PROJECT_ORDER):
        subset = df[df["project"] == project]
        means = subset[metric.key].to_numpy()
        stds = subset[metric.std_key].fillna(0).to_numpy()
        positions = x + (idx - 0.5) * width

        bars = ax.bar(
            positions,
            means,
            width,
            label=project.replace("hcpa_", "").replace("_", " "),
            color=PALETTE.get(project, "#777777"),
            hatch=HATCHES.get(project, ""),
            edgecolor="black",
            linewidth=0.8,
            yerr=stds,
            capsize=4,
        )
        add_value_labels(ax, bars, means, stds, metric.fmt)

    ax.set_xticks(x)
    ax.set_xticklabels(gpus)
    ylabel = metric.label + (f" ({metric.unit})" if metric.unit else "")
    ax.set_ylabel(ylabel)
    ax.set_title(metric.label)
    ax.legend(frameon=False, ncol=len(PROJECT_ORDER))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Ajusta limites para deixar espaço para os rótulos.
    y_min, y_max = ax.get_ylim()
    max_height = df[metric.key].max() if len(df) else 1
    pad = max(metric.ylim_pad_frac * abs(max_height), 0.05 * abs(y_max - y_min))
    ax.set_ylim(y_min, y_max + pad)

    outfile_base = OUT_DIR / f"{metric.key}"
    for ext, dpi in (("png", 400), ("pdf", 400)):
        fig.savefig(outfile_base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    df = load_data()

    # Subconjunto focado nos projetos core e GPUs existentes.
    df_core = df[df["project"].isin(PROJECT_ORDER)].copy()
    if df_core.empty:
        raise SystemExit("Resumo vazio para os projetos core.")

    for metric in METRICS:
        if metric.key not in df_core.columns:
            print(f"[WARN] Métrica {metric.key} ausente, pulando.")
            continue
        plot_metric(df_core, metric)

    print(f"Plots salvos em: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

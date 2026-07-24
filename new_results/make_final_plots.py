#!/usr/bin/env python3
"""
make_final_plots.py — figuras FINAIS de comparacao cross-GPU (GH200 vs A100).

Cada figura tem DOIS paineis lado a lado (GH200 | A100), NAO misturados, para
comparacao direta. Reaproveita a camada de dados e o estilo do make_plots.py
(inclusive o calculo CORRETO de spec@95sens via thresholds.csv). Sem os textos
de descricao no rodape. Saida em new_results/final/.

Excluidas (a pedido): fig_energy_early_stopping, fig_cec_efficiency.
Energia = fig_energy_wasted_after_peak.

Uso: cd new_results && python3 make_final_plots.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

import make_plots as mp  # reutiliza data layer + rcParams + constantes

# Fontes MAIORES para publicacao: estas figuras vao para o artigo e serao
# reduzidas para a largura da coluna/pagina, entao precisam de fonte generosa.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15.5,
    "axes.titleweight": "regular",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12.5,
    # fundo do plot levemente cinza (em vez de branco) + gridlines brancas
    "axes.facecolor": "#ECECEA",
    "grid.color": "white",
    "grid.linewidth": 1.1,
    "grid.alpha": 1.0,
    "hatch.linewidth": 0.35,
})

# texturas por abordagem (para leitura em tons de cinza / impressao).
# Padroes de densidade SIMPLES (1 caractere) + hatch.linewidth fino: a textura
# deve ser um canal secundario discreto, nunca competir com a cor da barra.
HATCH = {
    "tensorflow_base": "/",
    "tensorflow_opt": "\\",
    "pytorch_base": "x",
    "pytorch_opt": ".",
    "hybrid_simple": "o",
    "hybrid_token_reduction": "-",
    "hybrid_token_reduction_opt": "+",
    "retfound_green": "*",
}

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "final")
os.makedirs(OUT, exist_ok=True)

GPUS = [
    ("nvidia-gh200-480gb_g5k_hydra", "NVIDIA GH200"),
    ("nvidia-a100-sxm4-40gb_g5k_chuc", "NVIDIA A100"),
]

colors = mp.APPROACH_COLORS
BASELINE = mp.BASELINE


def disp(a):
    return mp.APPROACH_LABEL.get(a, a)


# ---------------------------------------------------------------- carga de dados
def _load(gpu_dir):
    path = mp.build_all_runs_csv(os.path.join(BASE, gpu_dir))
    return mp.load_summary_from_csv(path)


DATA = {tag: _load(tag) for tag, _ in GPUS}
# ordem canonica, so' abordagens presentes nos dois GPUs
plot_order = [a for a in mp.APPROACH_ORDER
              if all(a in DATA[t] for t, _ in GPUS) and a not in mp.EXCLUDE_APPROACHES]


def per_epoch_series(gpu_dir, approach, col, scale=1.0):
    """Serie por-epoca (NAO acumulada): media +/- desvio entre os 10 runs."""
    ad = os.path.join(BASE, gpu_dir, approach)
    if not os.path.isdir(ad):
        return None
    series = []
    for rid in range(10):
        p = mp._run_csv_path(ad, approach, rid)
        if p is None:
            continue
        rows = list(csv.DictReader(open(p)))
        vals = [(mp.fv(r.get(col)) or 0.0) * scale
                for r in rows if r.get("stage") not in ("final_test", "test")]
        if vals:
            series.append(vals)
    if not series:
        return None
    L = min(len(s) for s in series)
    arr = np.array([s[:L] for s in series])
    return arr.mean(axis=0), arr.std(axis=0)


def _style_xticks(ax):
    ax.set_xticks(np.arange(len(plot_order)))
    ax.set_xticklabels([disp(a) for a in plot_order], rotation=30, ha="right",
                       rotation_mode="anchor")
    for tick, a in zip(ax.get_xticklabels(), plot_order):
        if a == BASELINE:
            tick.set_fontweight("bold")
    ax.tick_params(axis="x", length=0)


def _grid(ax, axis="y"):
    getattr(ax, f"{axis}axis").grid(True, linestyle="-", linewidth=1.1,
                                    alpha=1.0, color="white")
    ax.set_axisbelow(True)


def _save(fig, fname):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    salvo: final/{fname}.pdf / .png")


def _edge(a):
    return ("#111111" if a == BASELINE else "#2A2A2A"), (1.6 if a == BASELINE else 0.5)


# ============================================================ barras (metrica)
def two_panel_bar(key, ylabel, fname, scale=1.0, dec=2, zoom=False, unit="",
                  ylim=None, ystep=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    tops, bots = [], []   # topo (media+dp) e base (media-dp) de todas as barras
    for ax, (tag, gpu) in zip(axes, GPUS):
        d = DATA[tag]
        for xi, a in enumerate(plot_order):
            m, s = mp.mean_std(d[a][key])
            if m is None:
                continue
            m *= scale; s = (s or 0) * scale
            ec, lw = _edge(a)
            ax.bar(xi, m, width=0.72, yerr=s, capsize=3.0, facecolor=colors[a],
                   edgecolor=ec, linewidth=lw, hatch=HATCH.get(a),
                   error_kw=dict(ecolor="#444444", lw=1.0), zorder=3)
            tops.append(m + s); bots.append(m - s)
        ax.set_title(gpu, pad=9)
        _style_xticks(ax)
        _grid(ax, "y")

    # limites do eixo Y
    hi = max(tops)
    if ylim is not None:                       # faixa fixa (ex.: spec@95 = 85..100)
        y0, y1 = ylim
    elif zoom and bots:
        lo = min(bots)
        span = max(hi - lo, 1e-9)
        y0, y1 = lo - span * 0.14, hi + span * 0.22   # padding aditivo (topo maior p/ rotulo)
    else:
        y0, y1 = 0.0, hi * 1.14
    axes[0].set_ylim(y0, y1)
    if ystep is not None:                      # ticks de N em N (ex.: 5 em 5)
        axes[0].yaxis.set_major_locator(MultipleLocator(ystep))

    # rotulo de valor no topo de cada barra (fonte legivel p/ paper)
    for ax, (tag, gpu) in zip(axes, GPUS):
        d = DATA[tag]
        for xi, a in enumerate(plot_order):
            m, s = mp.mean_std(d[a][key])
            if m is None:
                continue
            m *= scale; s = (s or 0) * scale
            ax.annotate(f"{m:.{dec}f}", xy=(xi, m + s), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color="#222222")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    _save(fig, fname)


# ============================================================ traintime breakdown
def two_panel_breakdown():
    SEG = [("warmup", "Compilation & autotuning (first epoch)", "#E8A33D", "."),
           ("train", "Training (steady state)", "#3B6B8F", ""),
           ("val", "Validation", "#5E9C7E", "/"),
           ("overhead", "Checkpoint, exact-metrics eval, I/O", "#C9C9C9", "x")]
    # sharey=False: o A100 chega a ~4400 s e, com eixo compartilhado, esmagaria
    # as barras do GH200 (~700-2000 s). Cada painel usa a propria escala; a
    # comparacao absoluta entre GPUs continua legivel pelos rotulos no topo.
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=False)
    for ax, (tag, gpu) in zip(axes, GPUS):
        gpu_dir = os.path.join(BASE, tag)
        bkd = {a: mp.load_time_breakdown(gpu_dir, a) for a in plot_order}
        x = np.arange(len(plot_order))
        bottom = np.zeros(len(plot_order))
        for key, _lbl, col, htch in SEG:
            hs = np.array([bkd[a][key][0] if bkd[a] else 0.0 for a in plot_order])
            ax.bar(x, hs, width=0.72, bottom=bottom, facecolor=col, hatch=htch,
                   edgecolor="#5A5A5A", linewidth=0.5, zorder=3)
            bottom += hs
        tops = []
        for xi, a in zip(x, plot_order):
            if not bkd[a]:
                continue
            tot = sum(bkd[a][k][0] for k, *_ in SEG)
            ax.annotate(f"{tot:.0f}", xy=(xi, tot), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color="#222222")
            tops.append(tot)
        ax.set_title(gpu, pad=9)
        ax.set_ylim(0, max(tops) * 1.10)   # escala propria por painel
        ax.set_ylabel("Total training time (s)")
        _style_xticks(ax)
        _grid(ax, "y")
    handles = [mpatches.Patch(facecolor=c, hatch=htch, edgecolor="#5A5A5A", label=l)
               for _k, l, c, htch in SEG]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _save(fig, "fig_traintime_breakdown")


# ============================================================ pareto spec95 x energia
def two_panel_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    for ax, (tag, gpu) in zip(axes, GPUS):
        d = DATA[tag]
        pts = []
        for a in plot_order:
            em, es = mp.mean_std(d[a]["total_energy_kj"])
            mm, ms = mp.mean_std(d[a]["test_spec_at_sens95"])
            if em is None or mm is None:
                continue
            pts.append((em, mm * 100, es, ms * 100, a))
        front = []
        for ex, ay, *_ in sorted(pts):
            if not front or ay > front[-1][1]:
                front.append((ex, ay))
        if len(front) > 1:
            fx, fy = zip(*front)
            ax.plot(fx, fy, color="#B0B0B0", lw=1.2, ls="--", zorder=1)
        for ex, ay, exs, ays, a in pts:
            ax.errorbar(ex, ay, xerr=exs, yerr=ays, fmt="none", ecolor=colors[a],
                        elinewidth=0.9, capsize=2, alpha=0.6, zorder=2)
            ec, lw = _edge(a)
            ax.scatter(ex, ay, s=70, color=colors[a], edgecolor=ec, linewidth=lw, zorder=3)
        ax.set_title(gpu, fontsize=12, pad=8)
        ax.set_xlabel("GPU energy, training phase (kJ)   ← lower is better")
        ax.grid(True, linestyle="-", linewidth=1.1, alpha=1.0, color="white")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Specificity @ 95% sensitivity (%)\nhigher is better →")
    handles = [mpatches.Patch(facecolor=colors[a], label=disp(a)) for a in plot_order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, labelspacing=0.5)
    fig.tight_layout()
    _save(fig, "fig_pareto_energy_spec95")


# ============================================================ speed vs energy (Q1)
def two_panel_speed_energy():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for ax, (tag, gpu) in zip(axes, GPUS):
        d = DATA[tag]
        for a in plot_order:
            tm, ts = mp.mean_std(d[a]["total_time_s"])
            em, es = mp.mean_std(d[a]["total_energy_kj"])
            if tm is None or em is None:
                continue
            tmin = tm / 60.0; tmins = (ts or 0) / 60.0
            ec, lw = _edge(a)
            ax.errorbar(tmin, em, xerr=tmins, yerr=es, fmt="none", ecolor=colors[a],
                        elinewidth=0.9, capsize=2, alpha=0.6, zorder=2)
            ax.scatter(tmin, em, s=70, color=colors[a], edgecolor=ec, linewidth=lw, zorder=3)
        ax.set_title(gpu, fontsize=12, pad=8)
        ax.set_xlabel("Total training time (min)")
        ax.grid(True, linestyle="-", linewidth=1.1, alpha=1.0, color="white")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("GPU energy, training phase (kJ)")
    handles = [mpatches.Patch(facecolor=colors[a], label=disp(a)) for a in plot_order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, labelspacing=0.5)
    fig.tight_layout()
    _save(fig, "fig_speed_vs_energy")


# ============================================================ energy wasted after peak
def two_panel_wasted():
    TOTAL = 200
    C_USE, C_WASTE = "#3B6B8F", "#D9534F"
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
    for ax, (tag, gpu) in zip(axes, GPUS):
        d = DATA[tag]
        yy = np.arange(len(plot_order))
        for i, a in enumerate(plot_order):
            pk, pk_sd = mp.mean_std(d[a]["best_auc_epoch"])
            eu = mp.mean_std(d[a]["energy_to_best_kj"])[0]
            ew = mp.mean_std(d[a]["energy_wasted_kj"])[0]
            pc = mp.mean_std(d[a]["energy_wasted_pct"])[0]
            if pk is None:
                continue
            pk = max(0.0, min(TOTAL, pk))
            ax.barh(yy[i], pk, height=0.62, facecolor=C_USE, edgecolor="white",
                    linewidth=1.2, zorder=3)
            ax.barh(yy[i], TOTAL - pk, left=pk, height=0.62, facecolor=C_WASTE,
                    hatch="/", edgecolor="white", linewidth=1.2, zorder=3)
            # azul (solido, escuro): texto BRANCO contrasta bem.
            fs_u = 9.5 if pk > TOTAL * 0.09 else 8.5
            ax.text(pk / 2, yy[i], f"{eu:.0f} kJ", ha="center", va="center",
                    fontsize=fs_u, color="white", zorder=5)
            # vermelho hachurado de branco: texto branco sumia no meio da hachura,
            # entao vai em PRETO (contrasta tanto com o vermelho quanto com a hachura).
            ww = TOTAL - pk
            fs_w = 9.5 if ww > TOTAL * 0.14 else 8.5
            ax.text(pk + ww / 2, yy[i], f"{ew:.0f} kJ ({pc:.0f}%)", ha="center",
                    va="center", fontsize=fs_w, color="#111111", zorder=5)
            ha = "left" if pk < TOTAL * 0.12 else "center"
            ax.annotate(f"peak @ ep {pk:.0f}±{(pk_sd or 0):.0f}", xy=(pk, yy[i]), xytext=(0, 10),
                        textcoords="offset points", ha=ha, va="bottom",
                        fontsize=7.4, color="#222222", zorder=6)
        ax.set_yticks(yy)
        ax.set_yticklabels([disp(a) for a in plot_order])
        ax.invert_yaxis()
        ax.set_xlim(0, TOTAL)
        ax.set_title(gpu, fontsize=12, pad=8)
        ax.set_xlabel("Epoch   (every run trains the full 200 epochs)")
        ax.xaxis.grid(True, ls="-", lw=1.1, alpha=1.0, color="white")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    h_use = mpatches.Patch(facecolor=C_USE, label="Energy to reach the AUC peak (useful)")
    h_waste = mpatches.Patch(facecolor=C_WASTE, hatch="/",
                             label="Energy spent training past the peak (wasted)")
    fig.legend(handles=[h_use, h_waste], loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _save(fig, "fig_energy_wasted_after_peak")


# ============================================================ per-epoch (energia/tempo)
def two_panel_per_epoch(col, ylabel, fname, scale=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    steady = []
    for ax, (tag, gpu) in zip(axes, GPUS):
        for a in plot_order:
            res = per_epoch_series(tag, a, col, scale)
            if res is None:
                continue
            v, sd = res
            ep = np.arange(len(v))
            ax.fill_between(ep, v - sd, v + sd, color=colors[a], alpha=0.16,
                            linewidth=0, zorder=2)
            ax.plot(ep, v, color=colors[a], lw=1.4, zorder=3)
            steady.extend((v + sd)[5:])
        ax.set_title(gpu, fontsize=12, pad=8)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 200)
        ax.grid(True, linestyle="-", linewidth=1.1, alpha=1.0, color="white")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    if steady:
        axes[0].set_ylim(0, max(steady) * 1.06)  # corta o pico de compilacao
    handles = [mpatches.Patch(facecolor=colors[a], label=disp(a)) for a in plot_order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.3)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _save(fig, fname)


def main():
    print("Gerando figuras FINAIS cross-GPU (GH200 | A100) em final/ ...")
    print(f"  abordagens: {plot_order}")
    two_panel_bar("test_auc", "AUC-ROC (%)", "fig_auc", scale=100, dec=2, zoom=False)
    two_panel_bar("test_spec_at_sens95", "Specificity @ 95% Sens. (%)", "fig_spec95",
                  scale=100, dec=1, ylim=(80, 100), ystep=10)
    two_panel_bar("total_energy_kj", "GPU energy, training phase (kJ)", "fig_energy_total",
                  scale=1, dec=0)
    two_panel_bar("gpu_mem_peak_mb", "Peak GPU memory (GB)", "fig_memory",
                  scale=1 / 1024, dec=2)
    two_panel_bar("mem_util_pct", "GPU memory-bandwidth utilization (%)", "fig_mem_util",
                  scale=1, dec=1)
    two_panel_breakdown()
    two_panel_pareto()
    two_panel_speed_energy()
    two_panel_wasted()
    two_panel_per_epoch("train_energy_j", "Energy per epoch (kJ)", "fig_energy_per_epoch",
                        scale=1 / 1000)
    two_panel_per_epoch("train_elapsed_s", "Training time per epoch (s)", "fig_time_per_epoch",
                        scale=1)
    print("Concluido.")


if __name__ == "__main__":
    main()

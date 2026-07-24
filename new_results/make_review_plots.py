#!/usr/bin/env python3
"""
Two figures added in response to the review, both from data already on disk.

  fig_patience_sweep.pdf  energy recovered against validation-AUC given up, for
                          patience in {10,20,30,40,50}. Replaces the single
                          patience-20 operating point with the whole curve.
  fig_breakeven.pdf       total cost of ownership against deployment volume,
                          under an explicit exchange rate between training
                          energy and one unnecessary referral.

Usage: cd new_results && python3 make_review_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "final")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15.5,
    "axes.titleweight": "regular",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.facecolor": "#ECECEA",
    "grid.color": "white",
    "grid.linewidth": 1.1,
    "grid.alpha": 1.0,
    "figure.dpi": 130,
})

LABEL = {
    "pytorch_opt": "PyTorch (opt.)",
    "tensorflow_opt": "TensorFlow (opt.)",
    "hybrid_token_reduction_opt": "Hybrid (TR, opt.)",
    "retfound_green": "RETFound-Green",
    "pytorch_base": "PyTorch (base)",
    "tensorflow_base": "TensorFlow (base)",
    "hybrid_simple": "Hybrid (plain)",
    "hybrid_token_reduction": "Hybrid (TR)",
}
COLORS = {
    "tensorflow_base": "#1F6F8B", "tensorflow_opt": "#5FA8C4",
    "pytorch_base": "#B23A2E", "pytorch_opt": "#E08258",
    "hybrid_simple": "#2E7D32", "hybrid_token_reduction": "#66A85B",
    "hybrid_token_reduction_opt": "#A6CE96", "retfound_green": "#6A4C93",
}
MARKERS = {
    "tensorflow_base": "s", "tensorflow_opt": "s", "pytorch_base": "o",
    "pytorch_opt": "o", "hybrid_simple": "^", "hybrid_token_reduction": "^",
    "hybrid_token_reduction_opt": "^", "retfound_green": "D",
}
ORDER = ["pytorch_opt", "tensorflow_opt", "hybrid_token_reduction_opt",
         "retfound_green", "pytorch_base", "tensorflow_base",
         "hybrid_simple", "hybrid_token_reduction"]
GPUS = ["GH200", "A100"]

res = json.load(open(os.path.join(BASE, "analysis", "reanalysis.json")))


def style(ax):
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- patience
def patience_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True, sharey=True)
    ps = [10, 20, 30, 40, 50]
    for ax, gpu in zip(axes, GPUS):
        for a in ORDER:
            e = res["patience"][f"{gpu}/{a}"]
            saved = [e["patience"][str(p)]["saved_pct"] for p in ps]
            pen = [e["patience"][str(p)]["val_penalty_pp"] for p in ps]
            ax.plot(pen, saved, "-", color=COLORS[a], marker=MARKERS[a],
                    markersize=7, linewidth=1.9, label=LABEL[a], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.7)
        ax.set_title(f"NVIDIA {gpu}")
        ax.set_xlabel("validation AUC given up (percentage points)")
        style(ax)
    axes[0].set_ylabel("training energy recovered (%)")
    axes[0].annotate("patience 10\n(top of each curve)", xy=(0.60, 0.06),
                     xycoords="axes fraction", fontsize=10.5, color="#444444")
    axes[1].annotate("patience 50\n(bottom of each curve)", xy=(0.58, 0.06),
                     xycoords="axes fraction", fontsize=10.5, color="#444444")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_patience_sweep.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_patience_sweep")


# --------------------------------------------------------------- break-even
def breakeven_figure():
    """Total cost C(N) = E_train + N * FRR * kappa, in kJ-equivalent.

    kappa is the training energy a centre would trade to avoid one unnecessary
    referral. Plotted for a deliberately tiny kappa (1 kJ) to show that even at
    that exchange rate the clinical term takes over almost immediately.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    N = np.logspace(2, 5.6, 400)
    kappa = 1.0
    d = res["descriptives"]
    for ax, gpu in zip(axes, GPUS):
        cost = {a: (d[f"{gpu}/{a}"]["e200"]["mean"]
                    + N * (1 - d[f"{gpu}/{a}"]["spec95"]["mean"] / 100) * kappa)
                for a in ORDER}
        best = np.min(np.vstack([cost[a] for a in ORDER]), axis=0)
        for a in ORDER:
            ax.plot(N, cost[a] / best, color=COLORS[a], linewidth=2.0,
                    label=LABEL[a], zorder=3)
        # where the identity of the cheapest configuration changes
        win = np.array([ORDER[i] for i in
                        np.argmin(np.vstack([cost[a] for a in ORDER]), axis=0)])
        switches = np.where(win[1:] != win[:-1])[0]
        for s in switches:
            ax.axvline(N[s], color="#555555", linestyle=":", linewidth=1.5, zorder=2)
            ax.annotate(f"N$\\approx${N[s]:,.0f}", xy=(N[s] * 1.1, 1.62),
                        fontsize=10.5, color="#333333", rotation=90)
        ax.set_xscale("log")
        ax.set_ylim(0.98, 1.75)
        ax.set_title(f"NVIDIA {gpu}")
        ax.set_xlabel("patients screened before the next re-training")
        style(ax)
    axes[0].set_ylabel("total cost relative to the best\nconfiguration at that volume")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_breakeven.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_breakeven")


if __name__ == "__main__":
    patience_figure()
    breakeven_figure()

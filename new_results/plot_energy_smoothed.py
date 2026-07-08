#!/usr/bin/env python3
"""Energia por epoca (media dos 10 runs) + suavizacao, GH200.
Marca: epoch 1 (recompilacao XLA freeze->finetune) e epoch 120 (inicio do EMA).
Gera fig_energy_per_epoch_smoothed.png em graficos/."""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "nvidia-gh200-480gb_g5k_hydra")
PATT = {"tensorflow_opt": "InceptionV3-*.csv",
        "tensorflow_base": "InceptionV3-*.csv",
        "pytorch_opt": "inception_v3-*.csv"}

def load(approach):
    files = sorted(glob.glob(os.path.join(ROOT, approach, "run_*", PATT[approach])))
    per_run = []
    for fn in files:
        d = {}
        with open(fn) as f:
            for r in csv.DictReader(f):
                try: e = int(r["epoch"])
                except: continue
                def g(k):
                    try: return float(r[k])
                    except: return np.nan
                d[e] = (g("train_energy_j"), g("train_avg_power_w"))
        if d: per_run.append(d)
    epochs = sorted(set().union(*[set(d) for d in per_run]))
    E = np.full((len(per_run), len(epochs)), np.nan)
    P = np.full((len(per_run), len(epochs)), np.nan)
    for i, d in enumerate(per_run):
        for j, e in enumerate(epochs):
            if e in d:
                E[i, j], P[i, j] = d[e]
    return np.array(epochs), E, P

def smooth(y, w=5):
    y = np.asarray(y, float); out = np.copy(y)
    half = w // 2
    for i in range(len(y)):
        lo, hi = max(0, i-half), min(len(y), i+half+1)
        seg = y[lo:hi]; seg = seg[~np.isnan(seg)]
        if seg.size: out[i] = seg.mean()
    return out

ep, E, P = load("tensorflow_opt")
e_mean, e_std = np.nanmean(E, 0), np.nanstd(E, 0)
p_mean = np.nanmean(P, 0)
EMA = 120

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), height_ratios=[1.15, 1])

# ---- (a) energia + potencia do tensorflow_opt ----
ax1.fill_between(ep, e_mean-e_std, e_mean+e_std, color="#4C72B0", alpha=.18, label="±1σ (10 runs)")
ax1.plot(ep, e_mean, color="#4C72B0", alpha=.30, lw=1, label="média bruta")
ax1.plot(ep, smooth(e_mean, 7), color="#1f3c88", lw=2.4, label="média suavizada (janela 7)")
ax1.axvline(EMA, color="#C44E52", ls="--", lw=1.6)
ax1.axvline(1, color="#8c8c8c", ls=":", lw=1.4)
ax1.annotate("epoch 1: recompilação XLA\n(freeze→finetune)", xy=(1, e_mean[1]),
             xytext=(14, e_mean[1]*.9), color="#555",
             arrowprops=dict(arrowstyle="->", color="#555"))
ax1.annotate("epoch 120 = int(0.6·200):\nEMA liga (update eager/batch)",
             xy=(EMA, np.nanmean(e_mean[122:130])), xytext=(128, e_mean[1]*.62),
             color="#C44E52", arrowprops=dict(arrowstyle="->", color="#C44E52"))
ax1.set_ylabel("energia por época (J)")
ax1.set_title("tensorflow_opt — energia por época (GH200, média de 10 runs)")
ax1.legend(loc="upper right", framealpha=.9)
axp = ax1.twinx(); axp.grid(False)
axp.plot(ep, smooth(p_mean, 7), color="#DD8452", lw=1.8, ls="-.", label="potência média (W)")
axp.set_ylabel("potência média (W)", color="#DD8452")
axp.tick_params(axis="y", colors="#DD8452")
axp.legend(loc="center right", framealpha=.9)

# ---- (b) comparacao: so o opt salta ----
for name, col in [("tensorflow_opt", "#4C72B0"), ("tensorflow_base", "#55A868"),
                  ("pytorch_opt", "#C44E52")]:
    e, En, _ = load(name)
    ax2.plot(e, smooth(np.nanmean(En, 0), 7), color=col, lw=2, label=name)
ax2.axvline(EMA, color="#C44E52", ls="--", lw=1.4)
ax2.set_yscale("log")
ax2.set_xlabel("época"); ax2.set_ylabel("energia/época (J, log)")
ax2.set_title("Só o tensorflow_opt salta no epoch 120 (base e pytorch_opt seguem planos → não é térmico do nó)")
ax2.legend(loc="center right", framealpha=.9)

fig.tight_layout()
out = os.path.join(ROOT, "graficos", "fig_energy_per_epoch_smoothed.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("salvo:", out)

# resumo numerico
def reg(a, lo, hi):
    m = (ep >= lo) & (ep <= hi)
    return np.nanmean(a[m])
print(f"energia  estável(2-119)={reg(e_mean,2,119):.0f}J  pós-EMA(120-199)={reg(e_mean,120,199):.0f}J "
      f"(+{reg(e_mean,120,199)/reg(e_mean,2,119)-1:.0%})")
print(f"potência estável={reg(p_mean,2,119):.0f}W  pós-EMA={reg(p_mean,120,199):.0f}W "
      f"({reg(p_mean,120,199)/reg(p_mean,2,119)-1:.0%})")

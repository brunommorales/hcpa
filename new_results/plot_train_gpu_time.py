#!/usr/bin/env python3
"""Gráfico de train time SÓ com o tempo de GPU (fase de treino).
Usa train_compute_s = soma dos train_elapsed_s por época (exclui validação,
exact-eval, checkpoint e I/O — o overhead de CPU/host). Substitui o antigo
fig_cpu_gpu_time. Gera fig_train_gpu_time.png em graficos/.

NOTA: train_compute_s é wall-clock da fase de treino. Exclui o overhead não-GPU,
mas ainda inclui as lacunas de dispatch DENTRO do step (a GPU roda a 268-303W,
faminta). O tempo GPU-busy ESTRITO = train_compute_s x (util%/100) exige amostrar
nvmlDeviceGetUtilizationRates — a ser coletado no re-run instrumentado."""
import csv, os, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "nvidia-gh200-480gb_g5k_hydra")
rows = defaultdict(list)
with open(os.path.join(ROOT, "gh200_all_runs.csv")) as f:
    for r in csv.DictReader(f):
        rows[r["approach"]].append(r)

def m(rs, k):
    v = [float(x[k]) for x in rs if x.get(k) not in (None, "", "nan")]
    return st.mean(v) if v else float("nan")
def sd(rs, k):
    v = [float(x[k]) for x in rs if x.get(k) not in (None, "", "nan")]
    return st.pstdev(v) if len(v) > 1 else 0.0

order = sorted(rows, key=lambda a: m(rows[a], "train_compute_s"))
gpu_t = np.array([m(rows[a], "train_compute_s") for a in order])
gpu_e = np.array([sd(rows[a], "train_compute_s") for a in order])

plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(order))
colors = ["#C44E52" if a == "tensorflow_opt" else "#3B6B8F" for a in order]
ax.barh(y, gpu_t, xerr=gpu_e, color=colors, edgecolor="white",
        error_kw=dict(ecolor="#555", lw=1))
for i, t in enumerate(gpu_t):
    ax.text(t + gpu_t.max()*0.01, i, f"{t:.0f}s", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set_xlabel("tempo de treino NA GPU — train_compute_s (s)")
ax.set_title("Train time = tempo de uso da GPU na fase de treino\n"
             "(exclui validação / exact-eval / checkpoint / I/O)")
ax.set_xlim(0, gpu_t.max()*1.16)
fig.tight_layout()
out = os.path.join(ROOT, "graficos", "fig_train_gpu_time.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("salvo:", out)
for a, t in zip(order, gpu_t):
    print(f"  {a:<28} train_compute_s={t:.0f}s")

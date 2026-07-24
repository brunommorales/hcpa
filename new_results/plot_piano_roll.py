#!/usr/bin/env python3
"""Piano roll: onde a GPU calculou e onde ela esperou o host.

Lê os JSON produzidos por gpu_kernel_profile.py (CUPTI) e desenha, para uma
janela de tempo DENTRO de uma época:

  linha por stream CUDA:  barras preenchidas = kernel rodando
  faixa "GPU parada":     as lacunas da união de TODOS os streams
  linha de cópias:        H2D/D2H (a lacuna entre kernels muitas vezes É uma cópia)

Diferente de fig_gpu_util_timeline (granularidade de ÉPOCA, estimativa NVML),
aqui a resolução é de KERNEL e o tempo ocupado é a UNIÃO dos intervalos — exato.

Uso:
    python plot_piano_roll.py <kprof_dir> [--epoch N] [--window-ms 40] [--start-ms auto]

Sem argumentos, procura em nvidia-gh200-480gb_g5k_hydra/*/run_0/kprof-0/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_kernel_profile import union_duration, union_gaps  # noqa: E402

INK, MUTED = "#222222", "#666666"
C_KERNEL = "#3B6B8F"   # GPU calculando
C_COPY = "#7FA88F"     # transferencia host<->device (H2D/D2H)
C_IDLE = "#D9534F"     # GPU parada (esperando o host)

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 9,
    "legend.frameon": False, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(path):
    d = json.loads(Path(path).read_text())
    cps = d["copies"]
    # formato novo: [inicio, fim, stream, tipo]; antigo: [inicio, fim, stream]
    if cps and len(cps[0]) == 4:
        transfers = [c[:3] for c in cps if c[3] in ("h2d", "d2h")]
        devwork = [c[:3] for c in cps if c[3] in ("d2d", "memset")]
    else:
        transfers, devwork = cps, []          # trace antigo: nao da para separar
    return d["summary"], d["kernels"] + devwork, transfers


def pick_window(kernels, window_us, start_us=None, transfers=None):
    """Janela representativa: ancorada na TRANSFERÊNCIA H2D de um batch, um pouco
    antes dela, de modo a conter um step inteiro (upload -> compute -> espera).

    Sem a âncora, a janela caía em qualquer ponto: no `pytorch_opt` o batch dura
    73 ms e há UMA transferência grande por batch, então uma janela de 40 ms podia
    não conter nenhuma — e a linha de transferências saía vazia, dando a impressão
    falsa de que o PyTorch não copia do host. (Ele copia: 114 H2D de ~4,7 ms por
    época, contra 115 de ~2,2 ms do TensorFlow.)
    """
    if not kernels:
        return 0.0, window_us
    t_min = min(k[0] for k in kernels)
    t_max = max(k[1] for k in kernels)
    if start_us is not None:
        s = t_min + start_us
        return s, s + window_us

    alvo = t_min + 0.2 * (t_max - t_min)   # evita o transiente da 1a iteração
    # transferências "de batch": as grandes (> 0.5 ms). As pequenas são d2d/memset.
    grandes = sorted((c[0] for c in (transfers or []) if (c[1] - c[0]) > 500.0))
    cand = [t for t in grandes if t >= alvo]
    if cand:
        # começa 10% da janela ANTES da transferência, para mostrar a espera que a precede
        s = max(t_min, cand[0] - 0.10 * window_us)
    else:
        after = [k[0] for k in kernels if k[0] >= alvo]
        s = min(after) if after else t_min
    return s, s + window_us


def clip(evs, t0, t1):
    out = []
    for s, e, st in evs:
        if e <= t0 or s >= t1:
            continue
        out.append((max(s, t0), min(e, t1), st))
    return out


def plot_epoch(json_path, out_path, window_ms=40.0, start_ms=None):
    summary, kernels, copies = load(json_path)
    win_us = window_ms * 1000.0
    t0, t1 = pick_window(kernels, win_us,
                         None if start_ms is None else start_ms * 1000.0,
                         transfers=copies)

    kw = clip(kernels, t0, t1)
    cw = clip(copies, t0, t1)
    if not kw:
        print(f"aviso: nenhum kernel na janela de {json_path}")
        return

    streams = sorted({st for _, _, st in kw})
    srow = {st: i for i, st in enumerate(streams)}
    n_rows = len(streams) + 2          # + linha de cópias + faixa de ociosidade

    fig, ax = plt.subplots(figsize=(11.4, 1.05 * n_rows + 2.1))

    # --- faixa de ociosidade (união de kernels + cópias)
    gaps = union_gaps([(s, e) for s, e, _ in kw] + [(s, e) for s, e, _ in cw], t0, t1)
    y_idle = n_rows - 1
    for g0, g1 in gaps:
        ax.barh(y_idle, (g1 - g0) / 1000.0, left=(g0 - t0) / 1000.0, height=0.55,
                color=C_IDLE, alpha=0.85, zorder=3)

    # --- cópias
    y_copy = n_rows - 2
    for s, e, _ in cw:
        ax.barh(y_copy, (e - s) / 1000.0, left=(s - t0) / 1000.0, height=0.55,
                color=C_COPY, zorder=3)

    # --- kernels por stream
    for s, e, st in kw:
        ax.barh(srow[st], (e - s) / 1000.0, left=(s - t0) / 1000.0, height=0.55,
                color=C_KERNEL, zorder=3)

    labels = [f"stream {st}" for st in streams] + ["transfer. H2D/D2H", "GPU PARADA"]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Tempo dentro da época (ms) — janela de {window_ms:.0f} ms")
    ax.set_xlim(0, window_ms)
    ax.xaxis.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    busy_win = union_duration([(s, e) for s, e, _ in kw]) / 1000.0
    idle_win = sum(g1 - g0 for g0, g1 in gaps) / 1000.0
    ax.set_title(
        f"{summary['approach']} · época {summary['epoch']} · "
        f"nesta janela: GPU calculando {busy_win:.1f} ms  |  parada {idle_win:.1f} ms",
        fontsize=10.5, color=INK, loc="left", pad=10)

    ax.legend(handles=[Patch(facecolor=C_KERNEL, label="GPU calculando (kernel, D2D, memset)"),
                       Patch(facecolor=C_COPY, label="transferência host↔device"),
                       Patch(facecolor=C_IDLE, alpha=0.85, label="GPU parada (esperando o host)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3,
              handlelength=1.3, columnspacing=1.4)

    fig.text(0.5, -0.02 - 0.01 * n_rows,
             f"Época inteira: GPU calculando {summary['busy_time_s']:.1f}s de "
             f"{summary['wall_time_s']:.1f}s ({summary['gpu_busy_pct']:.0f}%); "
             f"esperando o host {summary['cpu_wait_time_s']:.1f}s.\n"
             f"'Calculando' = UNIÃO dos intervalos de kernel (a SOMA daria "
             f"{summary['kernel_sum_time_s']:.1f}s: {summary['overlap_factor']:.2f}x maior, "
             f"porque streams se sobrepõem). Medido com CUPTI, não com util% do NVML.\n"
             f"ATENÇÃO: o profiler adiciona overhead — esta época NÃO deve ser usada "
             f"nas comparações de tempo/energia.",
             ha="center", va="top", fontsize=8.2, color=MUTED, linespacing=1.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out_path}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kprof_dir", nargs="?", default=None)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--window-ms", type=float, default=40.0)
    ap.add_argument("--start-ms", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(__file__).resolve().parent
    if a.kprof_dir is None:
        cands = sorted(root.glob("nvidia-*/*/run_0/kprof-0"))
        if not cands:
            print("nenhum kprof-* encontrado. Rode um treino com HCPA_PROFILE_EPOCHS=...")
            return 1
        a.kprof_dir = cands[0]

    files = sorted(Path(a.kprof_dir).glob("epoch_*.json"))
    if a.epoch is not None:
        files = [f for f in files if f.stem == f"epoch_{a.epoch}"]
    if not files:
        print(f"nenhum epoch_*.json em {a.kprof_dir}")
        return 1

    outdir = Path(a.out) if a.out else root / "graficos"
    outdir.mkdir(parents=True, exist_ok=True)
    for f in files:
        summary, _, _ = load(f)
        name = f"fig_piano_roll_{summary['approach']}_ep{summary['epoch']}"
        plot_epoch(f, outdir / name, a.window_ms, a.start_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())

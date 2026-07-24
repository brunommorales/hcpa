#!/usr/bin/env python3
"""Duas figuras sobre ociosidade da GPU (dados novos, instrumentados):

  fig_time_busy_idle    - decomposicao do tempo TOTAL, com o valor de CADA fase,
                          separando o treino em GPU-ativa vs GPU-ociosa.
  fig_gpu_util_timeline - util% por epoca: ONDE a GPU ficou ociosa e onde voltou
                          a calcular (granularidade de EPOCA, ver ressalva).

RESSALVA DE GRANULARIDADE: guardamos a MEDIA de util por epoca, nao a serie
temporal dentro da epoca. Entao o timeline mostra em QUE EPOCA a GPU ociou mais,
nao o instante (ms) exato em que um kernel parou/voltou. Para isso seria preciso
salvar a serie crua do sampler (ou CUPTI/Nsight).

Paleta validada (deltaE2000 sob deuteranopia/protanopia):
  segmentos: adjacentes OK; idle vs overhead = 11.3 -> exige encoding secundario
             => o segmento "GPU ociosa" leva HACHURA.
  abordagens: pior par 19.2 (OK).
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "nvidia-gh200-480gb_g5k_hydra")
OUT = os.path.join(ROOT, "graficos")

APPROACHES = [("tensorflow_opt", "InceptionV3-*.csv", "TensorFlow (opt.)", "#5FA8C4"),
              ("pytorch_base",   "inception_v3-*.csv", "PyTorch (base)",   "#B23A2E"),
              ("pytorch_opt",    "inception_v3-*.csv", "PyTorch (opt.)",   "#E08258")]

# Segmentos: hue = fase; hachura = "tempo desperdicado" (encoding secundario).
SEG = [("busy",     "Treino — GPU ativa",            "#3B6B8F", None),
       ("idle",     "Treino — GPU ociosa (esp. CPU)", "#9DBBD0", "///"),
       ("val",      "Validação",                      "#7FA88F", None),
       ("overhead", "Checkpoint, exact-eval, I/O",    "#C9C9C9", None)]

INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.labelsize": 11, "legend.fontsize": 9,
    "legend.frameon": False, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def fv(r, k):
    try: return float(r[k])
    except (TypeError, ValueError, KeyError): return None


def decompose(approach, pat):
    """Media, entre runs, do tempo de cada fase (s)."""
    acc = {k: [] for k, *_ in SEG}
    for fn in sorted(glob.glob(os.path.join(ROOT, approach, "run_*", pat))):
        rows = list(csv.DictReader(open(fn)))
        tr = [r for r in rows if r.get("stage") not in ("final_test", "test")]
        fin = [r for r in rows if r.get("stage") in ("final_test", "test")]
        if not fin:
            continue
        elapsed = sum(fv(r, "train_elapsed_s") or 0 for r in tr)
        busy = sum(fv(r, "train_busy_time_s") or 0 for r in tr)
        val = sum(fv(r, "val_elapsed_s") or 0 for r in tr)
        total = fv(fin[-1], "total_train_time_s") or 0
        acc["busy"].append(busy)
        acc["idle"].append(elapsed - busy)
        acc["val"].append(val)
        acc["overhead"].append(max(0.0, total - elapsed - val))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def util_series(approach, pat):
    """util% por epoca, media +/- dp entre os runs."""
    series = []
    for fn in sorted(glob.glob(os.path.join(ROOT, approach, "run_*", pat))):
        rows = list(csv.DictReader(open(fn)))
        tr = [r for r in rows if r.get("stage") not in ("final_test", "test")]
        u = [fv(r, "train_gpu_util_pct") for r in tr]
        u = [x for x in u if x is not None]
        if u:
            series.append(u)
    if not series:
        return None, None
    L = min(len(s) for s in series)
    arr = np.array([s[:L] for s in series])
    return arr.mean(axis=0), arr.std(axis=0)


# ---------------------------------------------------------------- figura 1
def fig_busy_idle():
    data = {a: decompose(a, p) for a, p, _, _ in APPROACHES}
    labels = [lbl for _, _, lbl, _ in APPROACHES]
    y = np.arange(len(APPROACHES))

    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    left = np.zeros(len(APPROACHES))
    for key, seg_label, color, hatch in SEG:
        w = np.array([data[a][key] for a, *_ in APPROACHES])
        ax.barh(y, w, left=left, height=0.62, facecolor=color, hatch=hatch,
                edgecolor="white", linewidth=1.6, label=seg_label, zorder=3)
        left += w

    totals = np.array([sum(data[a].values()) for a, *_ in APPROACHES])

    # rotulo de CADA fase; segmentos finos (<5%) vao acima da barra com guia
    for i, (a, *_ ) in enumerate(APPROACHES):
        acc = 0.0
        for key, _, color, _ in SEG:
            w = data[a][key]
            frac = w / totals[i]
            xc = acc + w / 2
            if frac >= 0.05:
                dark = key in ("busy",)
                ax.text(xc, y[i], f"{w:.0f}s", ha="center", va="center",
                        fontsize=9, color=("white" if dark else INK), zorder=5)
            else:
                ax.annotate(f"{w:.0f}s", xy=(xc, y[i] + 0.31),
                            xytext=(xc, y[i] + 0.46), ha="center", va="bottom",
                            fontsize=8, color=MUTED, zorder=5,
                            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.7))
            acc += w
        ax.text(totals[i] + totals.max() * 0.012, y[i], f"total {totals[i]:.0f}s",
                ha="left", va="center", fontsize=9.5, color=INK, fontweight="semibold")

    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Tempo (s)")
    ax.set_xlim(0, totals.max() * 1.16)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              handlelength=1.3, columnspacing=1.4)
    fig.text(0.5, -0.10,
             "A hachura marca o tempo em que a GPU esteve OCIOSA dentro do laço de treino "
             "(esperando dados/CPU).\n'GPU ativa' = tempo de treino × util% (NVML).",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_time_busy_idle.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("salvo: fig_time_busy_idle")
    return data, totals


# ---------------------------------------------------------------- figura 2
def fig_util_timeline():
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ends = []
    for a, pat, lbl, color in APPROACHES:
        m, s = util_series(a, pat)
        if m is None:
            continue
        ep = np.arange(len(m))
        ax.fill_between(ep, m - s, m + s, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(ep, m, color=color, lw=2.0, label=lbl, zorder=3)
        ends.append((len(m) - 1, m[-1], lbl, color))

    # rotulo direto na ponta (identidade nao depende so da cor)
    for x_, y_, lbl, color in ends:
        ax.annotate(lbl, xy=(x_, y_), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK)

    # evento do EMA no tensorflow_opt (util cai: mais tempo ocioso por step)
    ax.axvline(120, color="#999999", ls="--", lw=1.0, zorder=1)
    ax.annotate("epoch 120: EMA liga\n(GPU ocia mais por step)",
                xy=(120, 44), xytext=(128, 30), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

    ax.set_xlabel("Época")
    ax.set_ylabel("Utilização de compute da GPU (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 232)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.3, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.set_title("Onde a GPU ficou ociosa ao longo do treino  ·  util alto = calculando; "
                 "util baixo = esperando o host", fontsize=10.5, color=INK, pad=10)
    ax.legend(loc="lower left", ncol=3, handlelength=1.3, columnspacing=1.2)
    fig.text(0.5, -0.09,
             "Média de 10 runs (faixa = ±1 desvio-padrão). Granularidade de ÉPOCA: mostra em que época a GPU "
             "ociou mais,\nnão o instante exato (ms) em que um kernel parou/voltou — para isso seria preciso a série "
             "crua do sampler ou CUPTI.",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_gpu_util_timeline.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("salvo: fig_gpu_util_timeline")


if __name__ == "__main__":
    # BLOQUEADO como resultado. Estas duas figuras derivam do util% do NVML, e a
    # amostragem que as alimentou era ASSIMETRICA entre frameworks: o TF media
    # ~4 chamadas NVML por batch (460/epoca) dentro do laco de treino, enquanto
    # o PyTorch usava thread de fundo (~56/epoca). Isso inflava justamente a
    # ociosidade medida no TF. A amostragem ja foi uniformizada (thread de fundo
    # em todos), mas os dados AQUI sao os antigos: so voltam a valer depois de
    # (a) re-rodar com a instrumentacao nova e (b) validar contra o busy_time
    # exato do CUPTI (gpu_kernel_profile.py / plot_piano_roll.py).
    if os.environ.get("HCPA_UNVALIDATED_FIGS") != "1":
        print("fig_time_busy_idle / fig_gpu_util_timeline: BLOQUEADAS.")
        print("  Motivo: util% do NVML medido com amostragem assimetrica entre frameworks.")
        print("  Use o piano roll (CUPTI) em vez delas, ou HCPA_UNVALIDATED_FIGS=1 para inspecionar.")
        raise SystemExit(0)

    data, totals = fig_busy_idle()
    fig_util_timeline()
    print("\n--- valores (s) ---")
    print(f"{'approach':<16}" + "".join(f"{k:>12}" for k, *_ in SEG) + f"{'total':>10}")
    for i, (a, *_ ) in enumerate(APPROACHES):
        print(f"{a:<16}" + "".join(f"{data[a][k]:>12.0f}" for k, *_ in SEG)
              + f"{totals[i]:>10.0f}")

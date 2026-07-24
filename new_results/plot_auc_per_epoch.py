#!/usr/bin/env python3
"""AUC de validação por época: comportamento, pico e pós-pico.

Três painéis:
  A) visão completa          -> a subida rápida nas primeiras épocas
  B) zoom no platô           -> o que acontece DEPOIS do pico (eixo y truncado, sinalizado)
  C) época do pico por run   -> mostra que o pico é instável entre execuções

Usa val_auc (sem augmentation) — é a métrica clínica limpa por época.
train_auc reflete os lotes com mixup/cutmix e não serve para essa leitura.

Paleta das abordagens validada (ΔE2000 sob deuteranopia/protanopia: pior par 19.2).
"""
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "nvidia-gh200-480gb_g5k_hydra")
OUT = os.path.join(ROOT, "graficos")

# (dir, glob, rótulo, cor, desloc. do rótulo final, desloc. da anotação do pico)
# Os deslocamentos evitam colisão: o pico do pytorch_base (ep.162) fica perto da
# borda direita, então sua anotação vai para a esquerda/abaixo.
APPROACHES = [("tensorflow_opt", "InceptionV3-*.csv", "TensorFlow (opt.)", "#5FA8C4",   0, (9, 9)),
              ("pytorch_base",   "inception_v3-*.csv", "PyTorch (base)",   "#B23A2E", -16, (-118, -20)),
              ("pytorch_opt",    "inception_v3-*.csv", "PyTorch (opt.)",   "#E08258",  12, (9, 9))]

INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 9,
    "legend.frameon": False, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def fv(r, k):
    try: return float(r[k])
    except (TypeError, ValueError, KeyError): return None


def curves(approach, pat):
    series = []
    for fn in sorted(glob.glob(os.path.join(ROOT, approach, "run_*", pat))):
        rows = list(csv.DictReader(open(fn)))
        tr = [r for r in rows if r.get("stage") not in ("final_test", "test")]
        v = [fv(r, "val_auc") for r in tr]
        v = [x for x in v if x is not None]
        if v:
            series.append(v)
    if not series:
        return None, None, None
    L = min(len(s) for s in series)
    arr = np.array([s[:L] for s in series])
    peaks = [int(np.argmax(s)) for s in arr]
    return arr.mean(axis=0), arr.std(axis=0), peaks


def main():
    data = {a: curves(a, p) for a, p, _, _, _ in APPROACHES}

    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(10.6, 10.4), gridspec_kw={"height_ratios": [1, 1.35, 0.42]})

    # ---------------- A: visão completa ----------------
    for a, _, lbl, color, _ in APPROACHES:
        m, s, _ = data[a]
        ep = np.arange(len(m))
        axA.fill_between(ep, m - s, m + s, color=color, alpha=0.13, lw=0, zorder=2)
        axA.plot(ep, m, color=color, lw=2.0, label=lbl, zorder=3)
    axA.set_ylabel("AUC de validação")
    axA.set_xlim(0, 199); axA.set_ylim(0.72, 1.0)
    axA.yaxis.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
    axA.set_axisbelow(True)
    axA.set_title("A) Visão completa — a AUC sobe rápido e satura já nas primeiras épocas",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    axA.legend(loc="lower right", ncol=3, handlelength=1.3, columnspacing=1.2)

    # ---------------- B: zoom no platô ----------------
    for a, _, lbl, color, dy in APPROACHES:
        m, s, _ = data[a]
        ep = np.arange(len(m))
        axB.fill_between(ep, m - s, m + s, color=color, alpha=0.13, lw=0, zorder=2)
        axB.plot(ep, m, color=color, lw=2.0, zorder=3)

        pk = int(np.argmax(m))
        axB.plot(pk, m[pk], marker="o", ms=9, mfc=color, mec="white", mew=1.8, zorder=5)
        axB.annotate(f"pico ep.{pk}  ({m[pk]:.4f})", xy=(pk, m[pk]),
                     xytext=(9, 9), textcoords="offset points",
                     fontsize=8.5, color=INK, zorder=6)
        # rótulo direto na ponta, com deslocamento p/ evitar colisão
        axB.annotate(f"{lbl} — final {m[-1]:.4f}  (−{m[pk]-m[-1]:.4f})",
                     xy=(len(m) - 1, m[-1]), xytext=(8, dy), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK)

    axB.set_ylabel("AUC de validação")
    axB.set_xlim(0, 245); axB.set_ylim(0.944, 0.990)
    axB.yaxis.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
    axB.set_axisbelow(True)
    axB.set_title("B) Zoom no platô — o que acontece DEPOIS do pico  "
                  "(eixo y truncado em 0,944–0,990)",
                  fontsize=10.5, color=INK, loc="left", pad=8)

    # ---------------- C: época do pico por run ----------------
    for i, (a, _, lbl, color, _) in enumerate(APPROACHES):
        _, _, peaks = data[a]
        y = len(APPROACHES) - 1 - i
        axC.scatter(peaks, [y] * len(peaks), s=42, color=color, alpha=0.8,
                    edgecolor="white", linewidth=0.8, zorder=3)
        med = int(np.median(peaks))
        axC.plot([med], [y], marker="|", ms=16, color=INK, mew=1.8, zorder=4)
        axC.annotate(f"mediana {med}   ·   faixa {min(peaks)}–{max(peaks)}",
                     xy=(max(peaks) + 5, y), va="center", fontsize=8.2, color=MUTED)

    axC.set_yticks(range(len(APPROACHES)))
    axC.set_yticklabels([lbl for _, _, lbl, _, _ in reversed(APPROACHES)], fontsize=9)
    axC.set_xlabel("Época")
    axC.set_xlim(0, 245)
    axC.set_ylim(-0.6, len(APPROACHES) - 0.4)
    axC.xaxis.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
    axC.set_axisbelow(True)
    axC.tick_params(axis="y", length=0)
    axC.spines["left"].set_visible(False)
    axC.set_title("C) Época do pico em cada um dos 10 runs — o pico é MUITO instável "
                  "(por isso o best-checkpoint importa)",
                  fontsize=10.5, color=INK, loc="left", pad=8)

    fig.text(0.5, -0.015,
             "Média de 10 runs (faixa = ±1 desvio-padrão). Usa val_auc (sem augmentation); "
             "train_auc reflete lotes com mixup/cutmix e não serve para esta leitura.\n"
             "Não há colapso pós-pico: a queda média do pico até o fim é de apenas 0,0015 a 0,0072 de AUC "
             "— é oscilação em torno do platô, não degradação.",
             ha="center", va="top", fontsize=8.2, color=MUTED, linespacing=1.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_auc_per_epoch.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("salvo: fig_auc_per_epoch")

    print(f"\n{'approach':<16}{'pico_ep':>9}{'auc_pico':>10}{'final':>9}{'queda':>9}"
          f"{'pico_min':>10}{'pico_max':>10}{'mediana':>9}")
    for a, _, lbl, _, _ in APPROACHES:
        m, s, peaks = data[a]
        pk = int(np.argmax(m))
        print(f"{a:<16}{pk:>9}{m[pk]:>10.4f}{m[-1]:>9.4f}{m[pk]-m[-1]:>9.4f}"
              f"{min(peaks):>10}{max(peaks):>10}{int(np.median(peaks)):>9}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_plots.py — Gera as figuras-resumo da bateria de energia (G5K).

Fluxo:
  1) Para cada GPU em new_results/<gpu_dir>/, varre <approach>/run_N/*.csv e
     escreve um CSV consolidado por-run: <gpu_short>_all_runs.csv
     (uma linha por run, com TODAS as métricas clínicas e computacionais).
  2) O próprio script lê esse all_runs.csv e faz a sumarização (média ± desvio
     padrão) por abordagem — toda a estatística sai do CSV, não dos arquivos crus.
  3) Gera as figuras a partir da sumarização.

Saída: new_results/<gpu_dir>/graficos/
  - clinical_metrics.(png|pdf)        AUC-ROC + Especificidade@95% Sensibilidade
  - computational_metrics.(png|pdf)   Throughput, Memory, Power, Time
  - energy_metrics.(png|pdf)          Energia total por run + energia acumulada por época

Um conjunto de gráficos é gerado por GPU detectada.
"""
import csv
import sys
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

CSV_PREFIX = {
    "pytorch_base":               "inception_v3-",
    "pytorch_opt":                "inception_v3-",
    "tensorflow_base":            "InceptionV3-",
    "tensorflow_opt":             "InceptionV3-",
    "hybrid_simple":              "metrics_exec",
    "hybrid_token_reduction":     "metrics_exec",
    "hybrid_token_reduction_opt": "metrics_exec",
    "vit_pure":                   "metrics_exec",
    "retfound_green":             "metrics_exec",
}

# Rótulos por FRAMEWORK/abordagem, SEM o nome do modelo/arquitetura (vai no caption:
# "todas as variantes CNN usam InceptionV3"). As 4 primeiras variam framework×otimização.
APPROACH_LABEL = {
    "tensorflow_base":            "TensorFlow (base)",
    "tensorflow_opt":             "TensorFlow (opt.)",
    "pytorch_base":               "PyTorch (base)",
    "pytorch_opt":                "PyTorch (opt.)",
    "hybrid_simple":              "Hybrid CNN–ViT",
    "hybrid_token_reduction":     "Hybrid CNN–ViT (TR)",
    "hybrid_token_reduction_opt": "Hybrid CNN–ViT (TR, opt.)",
    "retfound_green":             "RETFound-Green",
    "vit_pure":                   "ViT-B/16",
}

# Ordem canônica de exibição (baseline primeiro)
APPROACH_ORDER = [
    "tensorflow_base", "tensorflow_opt", "pytorch_base", "pytorch_opt",
    "hybrid_simple", "hybrid_token_reduction", "hybrid_token_reduction_opt",
    "retfound_green", "vit_pure",
]

GPU_SHORT = {
    "nvidia-gh200-480gb_g5k_hydra":       "GH200",
    "nvidia-a100-sxm4-40gb_g5k_chuc":     "A100",
    "nvidia-h200-nvl-141gb_g5k_chicoree": "H200",
}

# Cor por FAMÍLIA de abordagem (não arco-íris): a cor codifica o framework,
# tons dentro da mesma família distinguem base vs. otimizado.
APPROACH_COLORS = {
    "tensorflow_base":            "#1F6F8B",  # teal escuro
    "tensorflow_opt":             "#5FA8C4",  # teal claro
    "pytorch_base":               "#B23A2E",  # vermelho-tijolo escuro
    "pytorch_opt":                "#E08258",  # laranja claro
    "hybrid_simple":              "#2E7D32",  # verde escuro
    "hybrid_token_reduction":     "#66A85B",  # verde médio
    "hybrid_token_reduction_opt": "#A6CE96",  # verde claro
    "retfound_green":             "#6A4C93",  # roxo
    "vit_pure":                   "#8C8C8C",  # cinza
}

# Colunas do all_runs.csv (clínicas + computacionais)
CLINICAL_COLS = ["test_auc", "test_precision", "test_f1",
                 "test_sens", "test_spec", "test_spec_at_sens95"]
COMPUTE_COLS  = ["throughput_img_s", "gpu_mem_peak_mb", "avg_power_w",
                 "avg_power_derived_w", "train_compute_s", "steady_epoch_time_s",
                 "best_auc_epoch", "total_time_s", "total_energy_kj",
                 "gpu_util_pct", "mem_util_pct", "busy_time_s",
                 # Q2 — limite de ORÁCULO: energia até o best-epoch e energia
                 # queimada depois dele. NÃO é realizável: só se sabe qual foi o
                 # best-epoch depois de treinar as 200. É um limite inferior.
                 "energy_to_best_kj", "energy_wasted_kj", "energy_wasted_pct",
                 # Q2 — REALIZÁVEL: o que um early stopping com paciência p teria
                 # feito, usando só a informação disponível até cada época.
                 # É este o número defensável.
                 "es_epoch", "es_best_epoch", "energy_es_kj",
                 "energy_saved_kj", "energy_saved_pct", "es_auc_penalty",
                 # Energia do JOB = treino + validação de cada época. A validação
                 # custa 12-23 % nas abordagens PyTorch e um treino real a paga.
                 # Fica separada de total_energy_kj (treino puro) porque o TF só
                 # passou a medi-la agora: nos dados antigos val_energy_j é 0.
                 "val_energy_kj", "energy_job_kj"]
ALL_COLS = CLINICAL_COLS + COMPUTE_COLS

BASELINE = "tensorflow_base"

# Abordagens omitidas das figuras E dos CSVs consolidados (all_runs/summary).
# vit_pure: ViT do zero é instável e distorce as escalas.
# As demais (hybrid_*, retfound_green, tensorflow_base) ainda não foram re-rodadas
# com a instrumentação nova (util%/mem_util%/busy_time_s/spec corrigido) — seus
# diretórios simplesmente não existem em new_results/ por enquanto, então são
# puladas automaticamente (ver build_all_runs_csv). Ao re-rodá-las, basta copiar
# os dados novos para cá.
# Extras podem ser adicionados via env: HCPA_EXCLUDE="a,b,c"
# Ex.: gerar só as abordagens com instrumentação nova (util%/mem_util%/busy_time):
#   HCPA_EXCLUDE="tensorflow_base,hybrid_simple,hybrid_token_reduction,\
#   hybrid_token_reduction_opt,retfound_green" python3 make_plots.py 480gb_g5k_hydra
_env_excl = os.environ.get("HCPA_EXCLUDE", "")
EXCLUDE_APPROACHES = {"vit_pure"} | {a.strip() for a in _env_excl.split(",") if a.strip()}

# Estilo de publicação: fundo branco, spines mínimos, texto vetorial embutido.
plt.rcParams.update({
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    0.8,
    "axes.grid":         False,
    "axes.axisbelow":    True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "regular",
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "legend.frameon":    False,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "xtick.color":       "#222222",
    "ytick.color":       "#222222",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.labelcolor":   "#222222",
    "text.color":        "#222222",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans"],
    "pdf.fonttype":      42,   # TrueType embutido (editável no PDF)
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})


def fv(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Etapa 1: consolida runs crus -> <gpu_short>_all_runs.csv
# ----------------------------------------------------------------------------
def _run_csv_path(approach_dir, approach, rid):
    pfx = CSV_PREFIX.get(approach)
    if pfx is None:
        return None
    p = os.path.join(approach_dir, f"run_{rid}", f"{pfx}{rid}.csv")
    return p if os.path.exists(p) else None


def _spec_at_sens95_from_thresholds(run_dir, target=0.95):
    """Deriva a especificidade no ponto de 95% de sensibilidade a partir da curva
    ROC salva do teste (run_dir/*-thresholds.csv: colunas sens, spec). É o "filtro
    no gráfico": entre os pontos com sens>=target, a maior especificidade (ponto de
    cruzamento). A coleta guarda só `spec` (threshold 0.5); o spec@95 vem daqui."""
    cand = glob.glob(os.path.join(run_dir, "*thresholds*.csv"))
    if not cand:
        return None
    best = None
    with open(cand[0]) as f:
        for r in csv.DictReader(f):
            s, sp = fv(r.get("sens")), fv(r.get("spec"))
            if s is not None and sp is not None and s >= target:
                best = sp if best is None else max(best, sp)
    return best


# Paciência do early stopping simulado (épocas sem melhora de val_auc).
ES_PATIENCE = int(os.environ.get("HCPA_ES_PATIENCE", "20"))


def early_stop_epoch(aucs, patience=ES_PATIENCE, min_delta=0.0):
    """Simula um early stopping REALIZÁVEL sobre a curva de val_auc.

    Só olha para o passado: na época i, decide com base nas épocas 0..i. É a
    diferença crucial em relação a "energia gasta depois do best-epoch", que
    exige saber o futuro (qual foi o pico) e portanto NÃO é executável na prática.

    Retorna (época_de_parada, época_do_melhor_checkpoint_até_ali, melhor_auc_até_ali).
    O modelo entregue é o best-checkpoint conhecido no momento da parada.
    """
    best, best_ep, bad = float("-inf"), 0, 0
    for i, a in enumerate(aucs):
        if a > best + min_delta:
            best, best_ep, bad = a, i, 0
        else:
            bad += 1
            if bad >= patience:
                return i, best_ep, best
    return len(aucs) - 1, best_ep, best


def _metrics_from_run(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    test = next((r for r in reversed(rows) if fv(r.get("test_auc")) is not None), None)
    if test is None:
        return None
    steady = rows[5:] if len(rows) > 5 else rows
    thrs = [fv(r.get("train_throughput_img_s")) for r in steady if fv(r.get("train_throughput_img_s"))]
    mems = [fv(r.get("train_gpu_mem_peak_mb"))  for r in steady if fv(r.get("train_gpu_mem_peak_mb"))]
    pwrs = [fv(r.get("train_avg_power_w"))       for r in steady if fv(r.get("train_avg_power_w"))]
    engs = [fv(r.get("train_energy_j"))          for r in rows   if fv(r.get("train_energy_j"))]
    # Utilização (NVML, só treino) e tempo GPU-ativa — novas métricas GPU-focused.
    guts = [fv(r.get("train_gpu_util_pct")) for r in steady if fv(r.get("train_gpu_util_pct")) is not None]
    muts = [fv(r.get("train_mem_util_pct")) for r in steady if fv(r.get("train_mem_util_pct")) is not None]
    busy = [fv(r.get("train_busy_time_s"))  for r in rows   if fv(r.get("train_busy_time_s")) is not None]
    # Tempo de treino COMPUTACIONAL: soma do forward+backward por época (train_elapsed).
    # Não inclui validação, checkpoint, I/O nem ocioso — apples-to-apples entre frameworks.
    telp = [fv(r.get("train_elapsed_s"))         for r in rows   if fv(r.get("train_elapsed_s"))]
    # Duração de UMA época em regime estável (mediana, época ≥5 — pula compilação/autotuning).
    steady_t = [fv(r.get("train_elapsed_s"))     for r in steady if fv(r.get("train_elapsed_s"))]
    # Época em que a AUC de validação foi máxima (curva de convergência por-run).
    _tr = [r for r in rows if r.get("stage") not in ("final_test", "test")]
    _va = [(int(float(r.get("epoch", i))), fv(r.get("val_auc")))
           for i, r in enumerate(_tr) if fv(r.get("val_auc")) is not None]
    _best_auc_epoch = max(_va, key=lambda t: t[1])[0] if _va else None
    _total_time_s = fv(test.get("total_train_time_s"))

    # Energia até o best-epoch (inclusive) vs energia queimada depois dele.
    # É o desperdício direto: o teste usa o best-checkpoint, então tudo o que foi
    # gasto após essa época não melhorou o resultado clínico entregue.
    _e_total = sum(engs) if engs else None
    _e_to_best = None
    if _best_auc_epoch is not None and engs:
        _per_ep = [fv(r.get("train_energy_j")) for r in _tr]
        _per_ep = [v for v in _per_ep if v is not None]
        if len(_per_ep) > _best_auc_epoch:
            _e_to_best = sum(_per_ep[:_best_auc_epoch + 1])
    _e_wasted = (_e_total - _e_to_best) if (_e_total and _e_to_best is not None) else None
    _e_val = sum(fv(r.get("val_energy_j")) or 0.0 for r in _tr)

    # --- Q2 realizável: early stopping com paciência ES_PATIENCE sobre val_auc.
    _auc_curve = [a for _, a in _va]
    _es_ep = _es_best = _e_es = _e_saved = _es_pen = None
    if _auc_curve:
        _es_ep, _es_best, _es_auc = early_stop_epoch(_auc_curve)
        _es_pen = max(_auc_curve) - _es_auc     # AUC perdida por parar cedo
        _per_ep = [fv(r.get("train_energy_j")) for r in _tr]
        _per_ep = [v for v in _per_ep if v is not None]
        if _e_total and len(_per_ep) > _es_ep:
            _e_es = sum(_per_ep[:_es_ep + 1])
            _e_saved = _e_total - _e_es

    return {
        "test_auc":            fv(test.get("test_auc")),
        "test_precision":      fv(test.get("test_precision")),
        "test_f1":             fv(test.get("test_f1")),
        "test_sens":           fv(test.get("test_sens")),
        "test_spec":           fv(test.get("test_spec")),
        "test_spec_at_sens95": _spec_at_sens95_from_thresholds(os.path.dirname(path)),
        "throughput_img_s":    float(np.mean(thrs)) if thrs else None,
        "gpu_mem_peak_mb":     float(np.mean(mems)) if mems else None,
        "avg_power_w":         float(np.mean(pwrs)) if pwrs else None,
        # Potência derivada do contador de energia exato (energia total ÷ tempo
        # total): garante potência×tempo = energia por construção. Preferida à
        # avg_power_w amostrada (200ms), que subestima os picos.
        "avg_power_derived_w": (sum(engs) / _total_time_s)
                               if (engs and _total_time_s) else None,
        "train_compute_s":     sum(telp) if telp else None,
        "steady_epoch_time_s": float(np.median(steady_t)) if steady_t else None,
        "best_auc_epoch":      _best_auc_epoch,
        "total_time_s":        _total_time_s,
        "total_energy_kj":     (sum(engs) / 1000.0) if engs else None,
        "gpu_util_pct":        float(np.mean(guts)) if guts else None,
        "mem_util_pct":        float(np.mean(muts)) if muts else None,
        "busy_time_s":         sum(busy) if busy else None,
        "energy_to_best_kj":   (_e_to_best / 1000.0) if _e_to_best is not None else None,
        "energy_wasted_kj":    (_e_wasted / 1000.0) if _e_wasted is not None else None,
        "energy_wasted_pct":   (100.0 * _e_wasted / _e_total)
                               if (_e_wasted is not None and _e_total) else None,
        "es_epoch":            _es_ep,
        "es_best_epoch":       _es_best,
        "energy_es_kj":        (_e_es / 1000.0) if _e_es is not None else None,
        "energy_saved_kj":     (_e_saved / 1000.0) if _e_saved is not None else None,
        "energy_saved_pct":    (100.0 * _e_saved / _e_total)
                               if (_e_saved is not None and _e_total) else None,
        "es_auc_penalty":      _es_pen,
        "val_energy_kj":       (_e_val / 1000.0) if _e_val else None,
        "energy_job_kj":       ((_e_total + _e_val) / 1000.0)
                               if (_e_total and _e_val is not None) else None,
    }


def build_all_runs_csv(gpu_dir):
    """Varre os runs crus e (re)escreve <gpu_short>_all_runs.csv. Retorna o caminho."""
    gpu_tag   = os.path.basename(gpu_dir)
    gpu_short = GPU_SHORT.get(gpu_tag, gpu_tag).lower()
    out_path  = os.path.join(gpu_dir, f"{gpu_short}_all_runs.csv")

    rows_out = []
    for approach in APPROACH_ORDER:
        if approach in EXCLUDE_APPROACHES:   # fora dos CSVs consolidados também
            continue
        approach_dir = os.path.join(gpu_dir, approach)
        if not os.path.isdir(approach_dir):
            continue
        for rid in range(10):
            p = _run_csv_path(approach_dir, approach, rid)
            if p is None:
                continue
            m = _metrics_from_run(p)
            if m is None:
                continue
            row = {"approach": approach, "run": rid}
            for k in ALL_COLS:
                v = m.get(k)
                row[k] = round(v, 6) if v is not None else ""
            rows_out.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["approach", "run"] + ALL_COLS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"    all_runs: {os.path.basename(out_path)} ({len(rows_out)} runs)")
    return out_path


# ----------------------------------------------------------------------------
# Etapa 2: lê o all_runs.csv e sumariza (média ± desvio padrão) por abordagem
# ----------------------------------------------------------------------------
def load_summary_from_csv(all_runs_path):
    """Lê o all_runs.csv e devolve {approach: {col: [valores por run]}}."""
    data = {}
    with open(all_runs_path) as f:
        for r in csv.DictReader(f):
            a = r["approach"]
            d = data.setdefault(a, {c: [] for c in ALL_COLS})
            for c in ALL_COLS:
                v = fv(r.get(c))
                if v is not None:
                    d[c].append(v)
    return data


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


# ----------------------------------------------------------------------------
# Etapa 3 (auxiliar): energia acumulada por época (precisa do dado por-época cru)
# ----------------------------------------------------------------------------
def load_energy_per_epoch(gpu_dir, approach):
    approach_dir = os.path.join(gpu_dir, approach)
    if not os.path.isdir(approach_dir):
        return None
    series = []
    for rid in range(10):
        p = _run_csv_path(approach_dir, approach, rid)
        if p is None:
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        has_final = any(r.get("stage") in ("final_test", "test") for r in rows)
        train = [r for r in rows if r.get("stage") not in ("final_test", "test")]
        e = [(fv(r.get("train_energy_j")) or 0.0) / 1000.0 for r in train]
        if has_final and any(v > 0 for v in e):
            series.append(np.cumsum(e))
    if not series:
        return None
    L = min(len(s) for s in series)
    arr = np.array([s[:L] for s in series])
    return arr.mean(axis=0), arr.std(axis=0)


# ----------------------------------------------------------------------------
# Decomposição do TEMPO TOTAL de treino (relógio de parede) por run:
#   compile   = excesso das primeiras épocas sobre o regime estável
#               (torch.compile/XLA/tf.function tracing/autotuning cuDNN)
#   train     = soma de train_elapsed_s menos o 'compile'
#   val       = soma de val_elapsed_s (validação nativa do fit)
#   overhead  = total_train_time_s - train - val - test  → é trabalho de treino
#               REAL: checkpoint + passada de métricas exatas + I/O (inclui o
#               teste final, 2-12s). NÃO é tempo alheio ao treino.
# ----------------------------------------------------------------------------
WARMUP_EPOCHS = 5  # só as primeiras épocas podem contar como compilação

def load_time_breakdown(gpu_dir, approach):
    approach_dir = os.path.join(gpu_dir, approach)
    if not os.path.isdir(approach_dir):
        return None
    comps = {"warmup": [], "train": [], "val": [], "overhead": []}
    for rid in range(10):
        p = _run_csv_path(approach_dir, approach, rid)
        if p is None:
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        train_rows = [r for r in rows if r.get("stage") not in ("final_test", "test")]
        final = [r for r in rows if r.get("stage") in ("final_test", "test")]
        if not train_rows or not final:
            continue
        t = [fv(r.get("train_elapsed_s")) or 0.0 for r in train_rows]
        v = [fv(r.get("val_elapsed_s")) or 0.0 for r in train_rows]
        total = fv(final[-1].get("total_train_time_s"))
        test_s = fv(final[-1].get("test_elapsed_s")) or 0.0
        if total is None or len(t) <= WARMUP_EPOCHS * 2:
            continue
        steady = float(np.median(t[WARMUP_EPOCHS:]))
        warm = sum(max(0.0, ti - steady) for ti in t[:WARMUP_EPOCHS])
        tr_sum, vl_sum = sum(t), sum(v)
        comps["warmup"].append(warm)
        comps["train"].append(tr_sum - warm)
        comps["val"].append(vl_sum)
        comps["overhead"].append(max(0.0, total - tr_sum - vl_sum - test_s) + test_s)
    if not comps["train"]:
        return None
    return {k: (float(np.mean(vs)), float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0)
            for k, vs in comps.items()}


# Deslocamentos candidatos (em pontos) e alinhamento, do mais desejável ao menos.
_LABEL_OFFSETS = [(9, 5, "left"), (9, -13, "left"), (-9, 5, "right"),
                  (-9, -13, "right"), (0, 14, "center"), (0, -20, "center"),
                  (9, 16, "left"), (-9, 16, "right"), (9, -24, "left"),
                  (-9, -24, "right")]


def _place_labels(fig, ax, points, fontsize=8.6, color="#222222"):
    """Anota (x, y, texto) escolhendo o deslocamento que não colide.

    Mede as caixas de texto de verdade (via renderer), em vez de chutar larguras:
    os rótulos deste estudo têm comprimentos muito diferentes ("PyTorch (opt.)"
    vs "Hybrid CNN-ViT (token reduction)") e um chute erra feio.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []                       # caixas já ocupadas (display coords)

    # pontos também são obstáculos: não escrever por cima dos marcadores
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        placed.append((px - 7, py - 7, px + 7, py + 7))

    def overlaps(b):
        return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3])
                   for o in placed)

    for x, y, text in sorted(points, key=lambda p: -p[1]):
        chosen = None
        for dx, dy, ha in _LABEL_OFFSETS:
            t = ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                            textcoords="offset points", fontsize=fontsize,
                            color=color, ha=ha, va="center", zorder=6)
            bb = t.get_window_extent(renderer=renderer)
            box = (bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)
            if not overlaps(box):
                chosen = box
                break
            t.remove()
        if chosen is None:            # nenhum candidato livre: usa o primeiro mesmo
            dx, dy, ha = _LABEL_OFFSETS[0]
            t = ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                            textcoords="offset points", fontsize=fontsize,
                            color=color, ha=ha, va="center", zorder=6)
            bb = t.get_window_extent(renderer=renderer)
            chosen = (bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)
        placed.append(chosen)


# ----------------------------------------------------------------------------
# Geração de gráficos para uma GPU
# ----------------------------------------------------------------------------
def _save(fig, out, fname, gpu_label=None):
    # Selo de proveniência (qual GPU gerou os dados), centrado sob a figura —
    # abaixo dos rótulos de eixo-x rotacionados, para não colidir com eles (o
    # bbox_inches="tight" absorve a margem extra automaticamente). Em preto e
    # com peso "semibold": é informação de leitura obrigatória, não rodapé.
    if gpu_label:
        fig.text(0.5, -0.16, gpu_label, ha="center", va="top",
                 fontsize=8.5, color="black", fontweight="semibold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{fname}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    salvo: {fname}.pdf / .png")


def generate_for_gpu(gpu_dir):
    gpu_tag   = os.path.basename(gpu_dir)
    gpu_short = GPU_SHORT.get(gpu_tag, gpu_tag)
    # Rótulo de proveniência estampado em toda figura (curto: "GPU: NVIDIA-GH200").
    gpu_label = (f"GPU: NVIDIA-{gpu_short}" if gpu_tag in GPU_SHORT
                else "GPU: " + gpu_tag.replace("_", " ").upper())
    OUT       = os.path.join(gpu_dir, "graficos")
    os.makedirs(OUT, exist_ok=True)

    all_runs_path = build_all_runs_csv(gpu_dir)
    data = load_summary_from_csv(all_runs_path)
    if not data:
        print(f"  [{gpu_short}] nenhuma abordagem encontrada, pulando.")
        return

    plot_order = [a for a in APPROACH_ORDER if a in data and a not in EXCLUDE_APPROACHES]
    colors  = {a: APPROACH_COLORS.get(a, "#666666") for a in plot_order}
    labels  = {a: APPROACH_LABEL.get(a, a) for a in plot_order}
    x = np.arange(len(plot_order))
    print(f"  [{gpu_short}] abordagens: {plot_order}")

    # A validação custa energia real. Se alguma abordagem não a mede, os totais de
    # energia NÃO são comparáveis entre si — e somar val silenciosamente faria a
    # abordagem que não mede parecer mais econômica. Avisa em vez de esconder.
    _sem_val = [a for a in plot_order
                if not data[a].get("val_energy_kj") or max(data[a]["val_energy_kj"]) <= 0]
    if _sem_val:
        print(f"  [AVISO] sem energia de validação: {', '.join(_sem_val)}")
        print( "          -> as figuras usam energia de TREINO apenas (comparável).")
        print( "          -> energy_job_kj (treino+val) só é válido após a re-rodada.")

    # Rótulo de exibição com marca de baseline (referência do estudo).
    def disp(a, inline=False):
        if a == BASELINE:
            return labels[a] + (" (baseline)" if inline else "\n(baseline)")
        return labels[a]

    def agg(key, scale):
        """Média e desvio absolutos (na unidade real) por abordagem."""
        mean, std = {}, {}
        for a in plot_order:
            m, s = mean_std(data[a][key])
            mean[a] = (m * scale) if m is not None else None
            std[a]  = (s * scale) if m is not None else 0.0
        return mean, std

    # ------------------------------------------------------------------
    # Uma métrica -> uma figura de barras, em unidade absoluta (sem
    # normalização). Cor por família; rótulo de valor discreto acima da barra.
    # ------------------------------------------------------------------
    def single_bar(mt):
        scale, dec, suffix = mt["scale"], mt["dec"], mt.get("suffix", "")
        absmean, absstd = agg(mt["key"], scale)

        heights = [absmean[a] if absmean[a] is not None else 0.0 for a in plot_order]
        errs    = [absstd[a]  for a in plot_order]

        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        for xi, a, h, e in zip(x, plot_order, heights, errs):
            if absmean[a] is None:
                continue
            is_base = (a == BASELINE)
            ax.bar(xi, h, width=0.72, yerr=e, capsize=2.5,
                   facecolor=colors[a],
                   edgecolor=("#111111" if is_base else "#2A2A2A"),
                   linewidth=(1.8 if is_base else 0.5),
                   error_kw=dict(ecolor="#444444", lw=0.9), zorder=3)
            # runs individuais como traços horizontais (barcode): mostra a
            # distribuição real sem as "bolinhas"; traços sobrepostos = densidade
            pts = [v * scale for v in data[a][mt["key"]] if v is not None]
            if pts:
                ax.hlines(pts, xi - 0.30, xi + 0.30, colors="#1A1A1A",
                          linewidth=0.8, alpha=0.55, zorder=4)

        vals = [h + e for h, e, a in zip(heights, errs, plot_order) if absmean[a] is not None]
        pt_max = [v * scale for a in plot_order for v in data[a][mt["key"]] if v is not None]
        hmax = max(vals + pt_max) if (vals or pt_max) else 1.0
        ax.set_ylim(0, hmax * 1.16)

        for xi, a, h, e in zip(x, plot_order, heights, errs):
            if absmean[a] is None:
                ax.annotate("N/A", xy=(xi, 0), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color="#999999",
                            fontstyle="italic")
                continue
            ax.annotate(f"{absmean[a]:.{dec}f}{suffix}", xy=(xi, h + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="#333333")

        ax.set_ylabel(mt["ylabel"])
        ax.set_xticks(x)
        ax.set_xticklabels([disp(a) for a in plot_order],
                           rotation=30, ha="right", rotation_mode="anchor")
        for tick, a in zip(ax.get_xticklabels(), plot_order):
            if a == BASELINE:
                tick.set_fontweight("bold")
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
        ax.set_axisbelow(True)
        _save(fig, OUT, mt["fname"], gpu_label)

    # Auditoria (2026-07). Removidos por redundância:
    #   fig_throughput      = imagens / steady_epoch_time  (é ~1/tempo)
    #   fig_power           = total_energy / total_time    (é energia/tempo)
    #   fig_time_per_epoch  = já aparece no fig_traintime_breakdown
    #   fig_energy_overview = duplica energy_total + energy_per_epoch
    #   fig_best_auc_epoch  = virou o painel C do fig_auc_per_epoch
    # Mantidos como DIAGNÓSTICO (não são resultado): fig_memory, fig_mem_util.
    # HCPA_EXTRA_FIGS=1 traz de volta os removidos, para inspeção.
    metrics = [
        {"key": "test_auc",            "fname": "fig_auc",
         "ylabel": "AUC-ROC (%)",          "scale": 100,    "dec": 2, "suffix": ""},
        {"key": "test_spec_at_sens95", "fname": "fig_spec95",
         "ylabel": "Specificity @ 95% Sens. (%)", "scale": 100, "dec": 1, "suffix": ""},
        {"key": "total_energy_kj",     "fname": "fig_energy_total",
         "ylabel": "GPU energy, training phase (kJ)", "scale": 1, "dec": 0, "suffix": ""},
        {"key": "gpu_mem_peak_mb",     "fname": "fig_memory",
         "ylabel": "Peak GPU memory (GB)",  "scale": 1/1024, "dec": 2, "suffix": ""},
        {"key": "mem_util_pct",        "fname": "fig_mem_util",
         "ylabel": "GPU memory-bandwidth utilization (%)", "scale": 1, "dec": 1, "suffix": ""},
    ]
    if os.environ.get("HCPA_EXTRA_FIGS") == "1":
        metrics += [
            {"key": "throughput_img_s",    "fname": "fig_throughput",
             "ylabel": "Throughput (images/s)", "scale": 1,     "dec": 0, "suffix": ""},
            {"key": "avg_power_derived_w", "fname": "fig_power",
             "ylabel": "Average power over training (W)", "scale": 1, "dec": 0, "suffix": ""},
            {"key": "steady_epoch_time_s", "fname": "fig_steady_epoch_time_bar",
             "ylabel": "Steady-state time per epoch (s)", "scale": 1, "dec": 1, "suffix": ""},
        ]
    # BLOQUEADOS: fig_gpu_util e fig_busy_time dependem de util% do NVML, cuja
    # medição era assimétrica entre frameworks (amostragem por-batch no TF vs
    # thread de fundo no PyTorch). A amostragem foi uniformizada, mas estas
    # figuras só voltam depois de validadas contra o busy_time exato do CUPTI
    # (gpu_kernel_profile.py). HCPA_UNVALIDATED_FIGS=1 para inspecioná-las.
    if os.environ.get("HCPA_UNVALIDATED_FIGS") == "1":
        metrics += [
            {"key": "gpu_util_pct",  "fname": "fig_gpu_util",
             "ylabel": "GPU compute utilization (%)", "scale": 1, "dec": 1, "suffix": ""},
            {"key": "busy_time_s",   "fname": "fig_busy_time",
             "ylabel": "GPU-active time, training phase (s)", "scale": 1, "dec": 0, "suffix": ""},
        ]
    for mt in metrics:
        single_bar(mt)

    # ------------------------------------------------------------------
    # Q1: o modelo mais rápido é o que gasta menos energia?
    # Dispersão tempo x energia. Se a resposta fosse "sim" sempre, os pontos
    # cairiam sobre UMA reta pela origem (potência média constante). O desvio
    # dessa reta é exatamente onde velocidade e energia se descolam.
    # ------------------------------------------------------------------
    t_e = [(a, mean_std(data[a]["total_time_s"]), mean_std(data[a]["total_energy_kj"]))
           for a in plot_order
           if data[a]["total_time_s"] and data[a]["total_energy_kj"]]
    if len(t_e) >= 2:
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        tmax = max(t[1][0] for t in t_e) * 1.18
        emax = max(t[2][0] for t in t_e) * 1.18

        # isolinhas de potência média (kJ = W x s / 1000)
        for w in (100, 200, 300, 400, 500, 600, 700):
            ax.plot([0, tmax], [0, w * tmax / 1000.0], color="#DDDDDD",
                    lw=0.8, ls="-", zorder=1)
            y_end = w * tmax / 1000.0
            if y_end <= emax:
                ax.annotate(f"{w} W", xy=(tmax, y_end), xytext=(3, 0),
                            textcoords="offset points", fontsize=7.5,
                            color="#AAAAAA", va="center")

        for a, (ts, tsd), (es, esd) in t_e:
            ax.errorbar(ts, es, xerr=tsd, yerr=esd, fmt="o", ms=9,
                        color=colors[a], ecolor="#888888", elinewidth=0.9,
                        capsize=2.5, mec="white", mew=1.4, zorder=4)

        ax.set_xlim(0, tmax * 1.10)
        ax.set_ylim(0, emax)
        ax.set_xlabel("Total training time (s)")
        ax.set_ylabel("GPU energy, training phase (kJ)")
        ax.set_title("Q1 — Is the fastest model the one that uses least energy?",
                     fontsize=10.5, color="#222222", loc="left", pad=10)
        ax.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
        ax.set_axisbelow(True)

        # rótulos sem sobreposição (os pontos ficam agrupados no canto superior)
        _place_labels(fig, ax, [(ts, es, f"{disp(a)}  ({1000*es/ts:.0f} W)")
                                for a, (ts, _), (es, _) in t_e])

        # contraexemplo explícito: o par mais lento-porém-mais-econômico
        pairs = [(a1, t1, e1, a2, t2, e2)
                 for a1, (t1, _), (e1, _) in t_e
                 for a2, (t2, _), (e2, _) in t_e
                 if t1 > t2 and e1 < e2]
        note = ""
        if pairs:
            a1, t1, e1, a2, t2, e2 = max(pairs, key=lambda p: (p[5] - p[2]))
            ax.annotate("", xy=(t2, e2), xytext=(t1, e1),
                        arrowprops=dict(arrowstyle="<->", color="#B23A2E",
                                        lw=1.2, shrinkA=7, shrinkB=7), zorder=5)
            note = (f"\nContra-exemplo: {disp(a1, inline=True)} é {t1-t2:.0f}s MAIS LENTO "
                    f"que {disp(a2, inline=True)} e ainda assim gasta {e2-e1:.0f} kJ a MENOS. "
                    "Logo: rápido ≠ econômico.")

        missing = [a for a in plot_order if not data[a]["total_energy_kj"]]
        miss_note = ("\nSem energia por época (dado perdido): "
                     + ", ".join(disp(m, inline=True) for m in missing)) if missing else ""

        fig.text(0.5, -0.02,
                 "Diagonal grey lines = constant average power. A point BELOW a line "
                 "draws less power than that line;\nif every approach sat on one line, "
                 "time and energy would be interchangeable." + note + miss_note,
                 ha="center", va="top", fontsize=8.2, color="#666666", linespacing=1.5)
        _save(fig, OUT, "fig_speed_vs_energy", gpu_label)

    # ------------------------------------------------------------------
    # Q2: quanta energia cada abordagem gasta a mais do que precisaria?
    #
    # Duas leituras, e a diferença entre elas é o ponto metodológico:
    #
    #  (a) ORÁCULO — "tudo depois do best-epoch foi desperdício". É um LIMITE
    #      INFERIOR do gasto necessário, mas NÃO é executável: para parar no pico
    #      seria preciso saber, na hora, que aquele é o pico. Só se sabe no fim.
    #
    #  (b) EARLY STOPPING (paciência p) — o que um treino real teria feito,
    #      decidindo a cada época com a informação disponível até ali. É o número
    #      defensável, e vem com um custo clínico explícito (a AUC que se perde
    #      por parar antes do pico verdadeiro).
    #
    # A figura mostra (b), com (a) marcado como referência.
    # ------------------------------------------------------------------
    es = [(a, mean_std(data[a]["energy_es_kj"]),
           mean_std(data[a]["energy_saved_kj"]),
           mean_std(data[a]["energy_saved_pct"]),
           mean_std(data[a]["es_epoch"]),
           mean_std(data[a]["es_auc_penalty"]),
           mean_std(data[a]["energy_to_best_kj"]))
          for a in plot_order if data[a]["energy_saved_kj"]]
    if es:
        fig, ax = plt.subplots(figsize=(10.4, 4.8))
        yy = np.arange(len(es))
        spent = np.array([e[1][0] for e in es])
        saved = np.array([e[2][0] for e in es])

        ax.barh(yy, spent, height=0.62, facecolor="#3B6B8F",
                edgecolor="white", linewidth=1.4, zorder=3,
                label=f"Energy actually needed (early stopping, patience {ES_PATIENCE})")
        ax.barh(yy, saved, left=spent, height=0.62, facecolor="#D9534F",
                hatch="//", edgecolor="white", linewidth=1.4, zorder=3,
                label="Energy early stopping would have SAVED")

        total_max = (spent + saved).max()
        for i, (a, (sp, _), (sv, _), (pc, _), (ep, _), (pen, _), (orc, _)) in enumerate(es):
            if sp > total_max * 0.12:
                ax.text(sp / 2, yy[i], f"{sp:.0f} kJ", ha="center", va="center",
                        fontsize=8.5, color="white", zorder=5)
            # marca do oráculo: onde o pico REAL da AUC ficou (limite inferior)
            ax.plot([orc, orc], [yy[i] - 0.31, yy[i] + 0.31], color="#222222",
                    lw=1.6, zorder=6)
            ax.annotate(f"−{sv:.0f} kJ ({pc:.0f}%)  ·  stops at ep. {ep:.0f}  ·  "
                        f"ΔAUC cost = {pen:.4f}",
                        xy=(sp + sv, yy[i]), xytext=(8, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8.3, color="#333333", zorder=5)

        ax.plot([], [], color="#222222", lw=1.6,
                label="Energy up to the TRUE best epoch (knowable only in hindsight)")

        ax.set_yticks(yy)
        ax.set_yticklabels([disp(e[0]) for e in es])
        ax.invert_yaxis()
        ax.set_xlabel("GPU energy, training phase (kJ)")
        ax.set_xlim(0, total_max * 1.62)
        ax.xaxis.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
                  handlelength=1.3, columnspacing=1.4)
        fig.text(0.5, -0.06,
                 f"Early stopping decides at each epoch using ONLY the epochs seen so far "
                 f"(patience = {ES_PATIENCE} epochs without val_auc improvement), so this is "
                 "energy a real run could have saved.\n"
                 "ΔAUC is what stopping early costs: the gap between the best AUC ever reached "
                 "and the best AUC known at the stopping epoch.\n"
                 "Black tick = energy to reach the TRUE best epoch. Tick to the RIGHT of the blue "
                 "bar: early stopping quit before the real peak, and pays for it in ΔAUC.\n"
                 "Tick to the LEFT: the peak came early and early stopping kept training past it, "
                 "paying in energy instead (ΔAUC ≈ 0).",
                 ha="center", va="top", fontsize=8.2, color="#666666", linespacing=1.5)
        _save(fig, OUT, "fig_energy_early_stopping", gpu_label)

    # ------------------------------------------------------------------
    # Energia ÚTIL × DESPERDIÇADA ao longo das 200 épocas, com corte no
    # PICO REAL da AUC (oráculo). Eixo X = época; a barra vai de 0 a 200 e
    # é partida na época do pico de val_auc: 0→pico = energia que levou ao
    # pico (útil); pico→200 = energia gasta sem ganho de AUC (desperdício).
    # É o LIMITE-ORÁCULO (só se sabe o pico depois de treinar tudo).
    # ------------------------------------------------------------------
    TOTAL_EPOCHS = 200
    wp = [(a, mean_std(data[a]["best_auc_epoch"]),
              mean_std(data[a]["energy_to_best_kj"]),
              mean_std(data[a]["energy_wasted_kj"]),
              mean_std(data[a]["energy_wasted_pct"]))
          for a in plot_order
          if data[a].get("best_auc_epoch") and data[a].get("energy_to_best_kj")]
    if wp:
        fig, ax = plt.subplots(figsize=(10.6, 4.8))
        yy = np.arange(len(wp))
        C_USE, C_WASTE = "#3B6B8F", "#D9534F"
        for i, (a, (pk, pks), (eu, _), (ew, _), (pc, _)) in enumerate(wp):
            pk = max(0.0, min(TOTAL_EPOCHS, pk))
            ax.barh(yy[i], pk, height=0.62, facecolor=C_USE,
                    edgecolor="white", linewidth=1.2, zorder=3)
            ax.barh(yy[i], TOTAL_EPOCHS - pk, left=pk, height=0.62, facecolor=C_WASTE,
                    hatch="//", edgecolor="white", linewidth=1.2, zorder=3)
            # segmento azul: dentro da barra se couber; senao, FORA (a direita da
            # divisao, sobre a barra vermelha) em texto escuro com halo branco —
            # nunca escondido, so' muda de posicao (bug: RETFound com pico muito
            # cedo tinha o segmento azul curto demais e o numero sumia).
            # mesmo estilo em todas as barras (texto branco, centralizado no
            # proprio segmento, sem caixa) — em segmentos estreitos so' encolhe a
            # fonte o suficiente para caber dentro da largura do segmento.
            useful_lbl = f"{eu:.0f} kJ"
            fs_use = 8.3 if pk > TOTAL_EPOCHS * 0.09 else 7.4
            ax.text(pk / 2, yy[i], useful_lbl, ha="center", va="center",
                    fontsize=fs_use, color="white", zorder=5)
            wasted_lbl = f"{ew:.0f} kJ  ({pc:.0f}%)"
            wasted_w = TOTAL_EPOCHS - pk
            fs_waste = 8.3 if wasted_w > TOTAL_EPOCHS * 0.12 else 6.6
            ax.text(pk + wasted_w / 2, yy[i], wasted_lbl, ha="center", va="center",
                    fontsize=fs_waste, color="white", zorder=5)
            # epoca do pico, acima da barra — deslocamento em PONTOS (gap fixo em
            # pixels, independente da escala de dados) para nunca encostar na barra;
            # ha dinamico evita que o texto "vaze" para a esquerda do eixo quando o
            # pico e' bem cedo (ex.: RETFound, pico ~ep 11).
            ha = "left" if pk < TOTAL_EPOCHS * 0.12 else "center"
            ax.annotate(f"peak AUC @ ep {pk:.0f}", xy=(pk, yy[i]),
                        xytext=(0, 10), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=7.8, color="#222222", zorder=6)
        h_use = mpatches.Patch(facecolor=C_USE,
                               label="Energy to reach the AUC peak (useful)")
        h_waste = mpatches.Patch(facecolor=C_WASTE, hatch="//",
                                 label="Energy spent training past the peak (wasted)")
        ax.legend(handles=[h_use, h_waste], loc="lower center",
                  bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.4,
                  columnspacing=1.4)
        ax.set_yticks(yy)
        ax.set_yticklabels([disp(e[0]) for e in wp])
        ax.invert_yaxis()
        ax.set_xlim(0, TOTAL_EPOCHS)
        ax.set_xlabel("Epoch   (every run trains the full 200 epochs)")
        ax.xaxis.grid(True, ls="-", lw=0.5, alpha=0.3, color="#BBBBBB")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        fig.text(0.5, -0.05,
                 "Bars span all 200 trained epochs; the split is the epoch of maximum validation "
                 "AUC (mean over 10 runs). Energy left of the split bought that peak; energy right "
                 "of it trained past the peak with no AUC gain.\nThe peak epoch is knowable only in "
                 "hindsight — this is the ORACLE upper bound of achievable savings; a realizable "
                 "early-stopping run saves less (see fig_energy_early_stopping).",
                 ha="center", va="top", fontsize=8.0, color="#666666", linespacing=1.5)
        _save(fig, OUT, "fig_energy_wasted_after_peak", gpu_label)

    # ------------------------------------------------------------------
    # Época da melhor AUC de validação — só o valor (mediana entre runs).
    # REDUNDANTE: virou o painel C do fig_auc_per_epoch, que ainda mostra a
    # dispersão entre runs. Só sai com HCPA_EXTRA_FIGS=1.
    # ------------------------------------------------------------------
    ep = {a: [v for v in data[a].get("best_auc_epoch", []) if v is not None] for a in plot_order}
    if any(ep.values()) and os.environ.get("HCPA_EXTRA_FIGS") == "1":
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        for xi, a in zip(x, plot_order):
            vals = ep[a]
            if not vals:
                continue
            med = float(np.median(vals))
            ax.bar(xi, med, width=0.72, facecolor=colors[a],
                   edgecolor=("#111111" if a == BASELINE else "#2A2A2A"),
                   linewidth=(1.8 if a == BASELINE else 0.5), zorder=3)
            ax.annotate(f"{med:.0f}", xy=(xi, med), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color="#333333")
        ax.set_ylim(0, 205)
        ax.set_ylabel("Epoch of best validation AUC")
        ax.set_xticks(x)
        ax.set_xticklabels([disp(a) for a in plot_order],
                           rotation=30, ha="right", rotation_mode="anchor")
        for tick, a in zip(ax.get_xticklabels(), plot_order):
            if a == BASELINE:
                tick.set_fontweight("bold")
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
        ax.set_axisbelow(True)
        _save(fig, OUT, "fig_best_auc_epoch", gpu_label)

    # ------------------------------------------------------------------
    # Tempo TOTAL de treino, decomposto (barra empilhada). Mostra o custo real
    # end-to-end de treinar, quebrado por componente. NÃO é linha do tempo — a
    # posição vertical é só ordem de empilhamento; a altura é o total por segmento.
    # ------------------------------------------------------------------
    SEG = [("warmup",   "Compilation & autotuning (first epoch)",   "#E0A458"),
           ("train",    "Training (steady state)",                  "#3B6B8F"),
           ("val",      "Validation",                               "#7FA88F"),
           ("overhead", "Checkpoint, exact-metrics eval, I/O",      "#C9C9C9")]
    bkd = {a: load_time_breakdown(gpu_dir, a) for a in plot_order}
    if any(bkd.values()):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        bottom = np.zeros(len(plot_order))
        for key, seg_label, seg_color in SEG:
            hs = np.array([bkd[a][key][0] if bkd[a] else 0.0 for a in plot_order])
            ax.bar(x, hs, width=0.72, bottom=bottom, facecolor=seg_color,
                   edgecolor="white", linewidth=0.4, label=seg_label, zorder=3)
            # valor de CADA fase, dentro do segmento (só se couber; senão fica só o total)
            totals_all = np.array([sum(bkd[a][k][0] for k, *_ in SEG) if bkd[a] else 1.0
                                   for a in plot_order])
            for xi, h, b, tot in zip(x, hs, bottom, totals_all):
                if tot > 0 and h / tot >= 0.06:
                    dark = seg_color in ("#3B6B8F",)
                    ax.text(xi, b + h / 2, f"{h:.0f}", ha="center", va="center",
                            fontsize=7.5, color=("white" if dark else "#222222"),
                            zorder=5)
            bottom += hs
        # total no topo de cada barra
        for xi, a in zip(x, plot_order):
            if not bkd[a]:
                continue
            tot = sum(bkd[a][k][0] for k, *_ in SEG)
            # whisker de ±1 desvio-padrão do tempo TOTAL entre os 10 runs
            _tstd = mean_std(data[a]["total_time_s"])[1] if data[a].get("total_time_s") else None
            if _tstd:
                ax.errorbar(xi, tot, yerr=_tstd, fmt="none", ecolor="#444444",
                            elinewidth=1.0, capsize=3, zorder=6)
            ax.annotate(f"{tot:.0f}", xy=(xi, tot + (_tstd or 0)), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="#333333")
        ax.set_ylim(0, bottom.max() * 1.16)
        ax.set_ylabel("Total training time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels([disp(a) for a in plot_order],
                           rotation=30, ha="right", rotation_mode="anchor")
        for tick, a in zip(ax.get_xticklabels(), plot_order):
            if a == BASELINE:
                tick.set_fontweight("bold")
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
        ax.set_axisbelow(True)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  handlelength=1.2, labelspacing=0.35, columnspacing=1.2)
        # Nota: (1) o 1º segmento é ESTIMADO (excesso das 1ªs épocas), não medido;
        # (2) 'overhead' é trabalho de treino real (checkpoint + métricas exatas + I/O).
        note = ("'Compilation' is estimated as the first epochs' time in excess of the "
                "steady-state rate (torch.compile / XLA / tf.function tracing / cuDNN "
                "autotuning); eager models without JIT show ≈ 0.\n'Checkpoint, exact-metrics "
                "eval, I/O' is real training-loop work: model checkpointing and a per-epoch "
                "exact-metrics validation pass (in PyTorch this is counted within validation).")
        fig.text(0.5, -0.14, note, ha="center", va="top", fontsize=7.2,
                 color="#666666", linespacing=1.4)
        # Selo de GPU em preto, abaixo da nota explicativa (2 linhas) — posição
        # própria para não colidir com o texto cinza acima.
        if gpu_label:
            fig.text(0.5, -0.27, gpu_label, ha="center", va="top",
                     fontsize=8.5, color="black", fontweight="semibold")
        _save(fig, OUT, "fig_traintime_breakdown")

    # ------------------------------------------------------------------
    # Fronteira de Pareto: métrica clínica × energia da fase de treino.
    # Ponto ideal = canto superior-esquerdo (métrica alta, baixa energia).
    # Geradas duas variantes: AUC e Especificidade@95%Sensibilidade.
    # ------------------------------------------------------------------
    def pareto(metric_key, ylabel, fname):
        mm, ms = agg(metric_key, 100)
        enm, ens = agg("total_energy_kj", 1)
        pts = [(enm[a], mm[a], ens[a], ms[a], a)
               for a in plot_order if enm[a] is not None and mm[a] is not None]

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        # fronteira: pontos não-dominados (menor energia, maior métrica)
        front = []
        for ex, ay, *_ in sorted(pts):
            if not front or ay > front[-1][1]:
                front.append((ex, ay))
        if len(front) > 1:
            fx, fy = zip(*front)
            ax.plot(fx, fy, color="#B0B0B0", lw=1.2, ls="--", zorder=1)

        for ex, ay, exs, ays, a in pts:
            ax.errorbar(ex, ay, xerr=exs, yerr=ays, fmt="none",
                        ecolor=colors[a], elinewidth=0.9, capsize=2, alpha=0.6, zorder=2)
            ax.scatter(ex, ay, s=70, color=colors[a],
                       edgecolor=("#111111" if a == BASELINE else "#2A2A2A"),
                       linewidth=(1.6 if a == BASELINE else 0.6),
                       zorder=3, label=disp(a, inline=True))

        ax.set_xlabel("GPU energy, training phase (kJ)   ← lower is better")
        ax.set_ylabel(f"{ylabel}\nhigher is better →")
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
        ax.set_axisbelow(True)
        # scatter: fecha a caixa (reativa as bordas de cima/direita que o estilo remove)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        # legenda FORA da caixa (à direita) para não cobrir nenhum ponto
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
                  handletextpad=0.4, labelspacing=0.5, frameon=False)
        _save(fig, OUT, fname, gpu_label)

    pareto("test_spec_at_sens95", "Specificity @ 95% sensitivity (%)", "fig_pareto_energy_spec95")

    # ------------------------------------------------------------------
    # Energia POR ÉPOCA (não acumulada), de um único run representativo —
    # mostra o comportamento instantâneo: pico de warm-up na época 0,
    # regime estável ~plano, e o degrau de fine-tuning (TF-opt, ép. ~120).
    # (a média de 10 runs + acumulação suavizaria justamente isso.)
    # ------------------------------------------------------------------
    def per_epoch_energy(gpu_dir, approach):
        """Energia por época: média ± desvio-padrão entre os 10 runs (kJ).
        Antes usava só o run_0; agora agrega, o que permite a banda de incerteza."""
        approach_dir = os.path.join(gpu_dir, approach)
        if not os.path.isdir(approach_dir):
            return None
        series = []
        for rid in range(10):
            p = _run_csv_path(approach_dir, approach, rid)
            if p is None:
                continue
            with open(p) as f:
                rows = list(csv.DictReader(f))
            e = [(fv(r.get("train_energy_j")) or 0.0) / 1000.0
                 for r in rows if r.get("stage") not in ("final_test", "test")]
            if e:
                series.append(e)
        if not series:
            return None
        L = min(len(s) for s in series)
        arr = np.array([s[:L] for s in series])
        return arr.mean(axis=0), arr.std(axis=0)

    def per_epoch_time(gpu_dir, approach):
        """Tempo de treino por época (s): média ± desvio-padrão entre os 10 runs.
        Espelha per_epoch_energy, mas lê train_elapsed_s."""
        approach_dir = os.path.join(gpu_dir, approach)
        if not os.path.isdir(approach_dir):
            return None
        series = []
        for rid in range(10):
            p = _run_csv_path(approach_dir, approach, rid)
            if p is None:
                continue
            with open(p) as f:
                rows = list(csv.DictReader(f))
            t = [(fv(r.get("train_elapsed_s")) or 0.0)
                 for r in rows if r.get("stage") not in ("final_test", "test")]
            if t:
                series.append(t)
        if not series:
            return None
        L = min(len(s) for s in series)
        arr = np.array([s[:L] for s in series])
        return arr.mean(axis=0), arr.std(axis=0)

    def per_epoch_perf(gpu_dir, approach, metric="val_auc"):
        """Média entre runs, por época: (métrica% best-so-far, energia acumulada kWh).
        best-so-far = envelope monotônico (o resultado do best-checkpoint com aquele
        orçamento de energia). Retorna None se faltar dado."""
        approach_dir = os.path.join(gpu_dir, approach)
        if not os.path.isdir(approach_dir):
            return None
        perfs, engs = [], []
        for rid in range(10):
            p = _run_csv_path(approach_dir, approach, rid)
            if p is None:
                continue
            rows = [r for r in csv.DictReader(open(p))
                    if r.get("stage") not in ("final_test", "test")]
            a = [fv(r.get(metric)) for r in rows]
            e = [fv(r.get("train_energy_j")) or 0.0 for r in rows]
            if not a or any(v is None for v in a) or not e:
                continue
            perfs.append(np.maximum.accumulate(np.array(a)))   # best-so-far
            engs.append(np.cumsum(e) / 3.6e6)                  # J -> kWh
        if not perfs:
            return None
        L = min(min(len(x) for x in perfs), min(len(x) for x in engs))
        P = np.array([x[:L] for x in perfs]) * 100.0
        E = np.array([x[:L] for x in engs])
        return P.mean(axis=0), E.mean(axis=0)

    # ------------------------------------------------------------------
    # Visão combinada lado a lado: (esq) BARRA de energia total acumulada ·
    # (dir) energia POR ÉPOCA oscilando (regime estável, épocas ≥ 5).
    # ------------------------------------------------------------------
    # (mpatches importado no topo do módulo)
    fig, (ax_tot, ax_ep) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    # esquerda: barra do total acumulado por abordagem
    tot_m, tot_s = agg("total_energy_kj", 1)
    xb = np.arange(len(plot_order))
    for xi, a in zip(xb, plot_order):
        if tot_m[a] is None:
            continue
        ax_tot.bar(xi, tot_m[a], width=0.72, yerr=tot_s[a], capsize=2.5,
                   facecolor=colors[a],
                   edgecolor=("#111111" if a == BASELINE else "#2A2A2A"),
                   linewidth=(1.8 if a == BASELINE else 0.5),
                   error_kw=dict(ecolor="#444444", lw=0.9), zorder=3)
        ax_tot.annotate(f"{tot_m[a]:.0f}", xy=(xi, tot_m[a] + (tot_s[a] or 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color="#333333")
    ax_tot.set_xticks([])
    ax_tot.set_title("Total accumulated energy", fontsize=11.5, pad=8)
    ax_tot.set_ylabel("GPU energy, training phase (kJ)")
    ax_tot.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
    ax_tot.set_axisbelow(True)
    # direita: energia por época (oscilação), regime estável (épocas ≥ 5)
    for a in plot_order:
        res = per_epoch_energy(gpu_dir, a)
        if res is not None:
            e, sd = res
            ep_ = np.arange(len(e))
            ax_ep.fill_between(ep_, e - sd, e + sd, color=colors[a], alpha=0.16,
                               linewidth=0, zorder=2)
            ax_ep.plot(ep_, e, color=colors[a], lw=1.3, zorder=3)
    ep_steady = [per_epoch_energy(gpu_dir, a) for a in plot_order]
    smax = max(((e + sd)[5:].max() for r in ep_steady if r is not None
                for e, sd in [r]), default=1.0)
    ax_ep.set_xlim(5, 200)
    ax_ep.set_ylim(0, smax * 1.08)
    ax_ep.set_title("Per-epoch energy (fluctuation, epochs ≥ 5)", fontsize=11.5, pad=8)
    ax_ep.set_xlabel("Epoch")
    ax_ep.set_ylabel("Energy per epoch (kJ)")
    ax_ep.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
    ax_ep.set_axisbelow(True)
    # legenda compartilhada (identidade por cor, serve aos dois painéis)
    handles = [mpatches.Patch(facecolor=colors[a],
                              edgecolor=("#111111" if a == BASELINE else "#2A2A2A"),
                              linewidth=(1.4 if a == BASELINE else 0.5),
                              label=disp(a, inline=True)) for a in plot_order]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
               handlelength=1.3, columnspacing=1.2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    # REDUNDANTE: duplica fig_energy_total + fig_energy_per_epoch.
    if os.environ.get("HCPA_EXTRA_FIGS") == "1":
        _save(fig, OUT, "fig_energy_overview", gpu_label)
    else:
        plt.close(fig)

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    steady_vals = []
    for a in plot_order:
        res = per_epoch_energy(gpu_dir, a)
        if res is None:
            continue
        e, sd = res
        ep = np.arange(len(e))
        # banda de ±1 desvio-padrão entre os 10 runs (o "borrão" de incerteza)
        for ax_ in (ax_full, ax_zoom):
            ax_.fill_between(ep, e - sd, e + sd, color=colors[a], alpha=0.16,
                             linewidth=0, zorder=2)
        ax_full.plot(ep, e, color=colors[a], lw=1.4, label=labels[a], zorder=3)
        ax_zoom.plot(ep, e, color=colors[a], lw=1.4, zorder=3)
        steady_vals.extend((e + sd)[5:])
    # painel esquerdo: visão completa (mostra os picos de warm-up na época 0)
    ax_full.set_title("Full trace — warm-up spikes at epoch 0", fontsize=11, pad=8)
    ax_full.set_xlabel("Epoch"); ax_full.set_ylabel("Energy per epoch (kJ)")
    ax_full.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
    ax_full.set_axisbelow(True)
    ax_full.legend(loc="upper right", ncol=2, fontsize=7.5, handlelength=1.2,
                   labelspacing=0.3, columnspacing=0.9)
    # painel direito: zoom no regime estável (mostra a oscilação real, pequena)
    if steady_vals:
        ax_zoom.set_ylim(0, max(steady_vals) * 1.08)
    ax_zoom.set_title("Steady-state zoom — per-epoch fluctuation is <1–2%",
                      fontsize=11, pad=8)
    ax_zoom.set_xlabel("Epoch"); ax_zoom.set_ylabel("Energy per epoch (kJ)")
    ax_zoom.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
    ax_zoom.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, OUT, "fig_energy_per_epoch", gpu_label)

    # ------------------------------------------------------------------
    # Tempo de treino POR ÉPOCA (s): média ± desvio-padrão entre os 10 runs.
    # Espelha fig_energy_per_epoch: painel completo (pico de compilação/warm-up
    # na época 0) + zoom no regime estável.
    # ------------------------------------------------------------------
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    steady_vals = []
    for a in plot_order:
        res = per_epoch_time(gpu_dir, a)
        if res is None:
            continue
        t, sd = res
        ep = np.arange(len(t))
        for ax_ in (ax_full, ax_zoom):
            ax_.fill_between(ep, t - sd, t + sd, color=colors[a], alpha=0.16,
                             linewidth=0, zorder=2)
        ax_full.plot(ep, t, color=colors[a], lw=1.4, label=labels[a], zorder=3)
        ax_zoom.plot(ep, t, color=colors[a], lw=1.4, zorder=3)
        steady_vals.extend((t + sd)[5:])
    # painel esquerdo: visão completa (mostra o pico de compilação na época 0)
    ax_full.set_title("Full trace — compilation/warm-up spike at epoch 0",
                      fontsize=11, pad=8)
    ax_full.set_xlabel("Epoch"); ax_full.set_ylabel("Training time per epoch (s)")
    ax_full.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
    ax_full.set_axisbelow(True)
    ax_full.legend(loc="upper right", ncol=2, fontsize=7.5, handlelength=1.2,
                   labelspacing=0.3, columnspacing=0.9)
    # painel direito: zoom no regime estável (o pico da época 0 fica clipado)
    if steady_vals:
        ax_zoom.set_ylim(0, max(steady_vals) * 1.08)
    ax_zoom.set_title("Steady-state zoom — per-epoch training time",
                      fontsize=11, pad=8)
    ax_zoom.set_xlabel("Epoch"); ax_zoom.set_ylabel("Training time per epoch (s)")
    ax_zoom.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
    ax_zoom.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, OUT, "fig_time_per_epoch", gpu_label)

    # ------------------------------------------------------------------
    # Curva CEC (Cost–Efficiency of Convergence): AUC (best-so-far) vs energia
    # ACUMULADA por época. O "joelho" = época CEC-ótima: a partir dela gasta-se
    # muita energia para ganho de AUC desprezível. Mostra que o resultado após
    # 200 épocas não compensa vs. parar no joelho.
    # ------------------------------------------------------------------
    # Métrica UNIFICADA com o fig_best_auc_epoch: por RUN calcula o pico
    # (argmax val_auc) e a 1ª época a atingir 99% do pico daquele run; agrega
    # por MEDIANA. Assim peak_ep == best_auc_epoch (mesma métrica nas 2 figuras).
    # Época → % de energia via curva média de energia acumulada (~determinística).
    rows_cec = []
    for a in plot_order:
        adir = os.path.join(gpu_dir, a)
        if not os.path.isdir(adir):
            continue
        peaks, ep99s, ens = [], [], []
        for rid in range(10):
            p = _run_csv_path(adir, a, rid)
            if p is None:
                continue
            rws = [r for r in csv.DictReader(open(p))
                   if r.get("stage") not in ("final_test", "test")]
            auc = [fv(r.get("val_auc")) for r in rws]
            e = [fv(r.get("train_energy_j")) or 0.0 for r in rws]
            if not auc or any(v is None for v in auc) or not e:
                continue
            auc = np.array(auc)
            bsf = np.maximum.accumulate(auc)
            peaks.append(int(np.argmax(auc)))
            ep99s.append(int(np.argmax(bsf >= 0.99 * auc.max())))
            ens.append(np.cumsum(e) / 3.6e6)
        if not peaks:
            continue
        L = min(len(x) for x in ens)
        mean_e = np.array([x[:L] for x in ens]).mean(axis=0)
        total = mean_e[-1]
        peak_ep = int(np.median(peaks))
        ep99 = int(np.median(ep99s))
        rows_cec.append((a, 100.0 * mean_e[min(ep99, L - 1)] / total, ep99,
                         100.0 * mean_e[min(peak_ep, L - 1)] / total, peak_ep, total))
    if rows_cec:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        yb = np.arange(len(rows_cec))[::-1]  # baseline (1º) no topo
        for (a, frac99, idx99, frac_pk, pk, e_tot), yi in zip(rows_cec, yb):
            is_base = (a == BASELINE)
            ax.barh(yi, frac99, height=0.62, color=colors[a],
                    edgecolor=("#111111" if is_base else "#2A2A2A"),
                    linewidth=(1.6 if is_base else 0.5), zorder=3)
            ax.barh(yi, 100 - frac99, left=frac99, height=0.62, color="#EAEAEA",
                    edgecolor="#D0D0D0", linewidth=0.4, zorder=2)
            ax.annotate(f"{frac99:.0f}% · ep{idx99}", xy=(frac99, yi), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8, color="#333333")
            # marca do pico absoluto (energia onde a AUC máxima foi atingida)
            ax.plot([frac_pk, frac_pk], [yi - 0.31, yi + 0.31],
                    color="#111111", lw=1.8, zorder=6)
            ax.annotate(f"peak ep{pk}", xy=(frac_pk, yi + 0.31), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.8, color="#111111")
            ax.annotate(f"{e_tot * 3600:.0f} kJ", xy=(100, yi), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7.5, color="#999999")
        ax.set_yticks(yb)
        ax.set_yticklabels([disp(a) for a, *_ in rows_cec])
        for tick, (a, *_ ) in zip(ax.get_yticklabels(), rows_cec):
            if a == BASELINE:
                tick.set_fontweight("bold")
        ax.set_ylim(-0.6, len(rows_cec) - 0.4)
        ax.set_xlim(0, 122)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("% of total training energy")
        ax.set_title("Energy to 99% of best AUC (colored) vs wasted (grey)  ·  ▏= absolute-peak epoch",
                     fontsize=10.5, pad=10)
        ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.35, color="#BBBBBB")
        ax.set_axisbelow(True)
        _save(fig, OUT, "fig_cec_efficiency", gpu_label)

    print(f"    figuras em: {OUT}")


# ----------------------------------------------------------------------------
# Entry point — gera para cada GPU presente em new_results/
# (sob guard __main__ para permitir `import make_plots` sem efeitos colaterais,
#  reutilizando as funcoes de carga de dados no make_final_plots.py)
# ----------------------------------------------------------------------------
def _main():
    gpu_dirs = sorted(d for d in glob.glob(os.path.join(BASE, "nvidia-*")) if os.path.isdir(d))
    # Filtro opcional por substring do diretório: `python3 make_plots.py 480gb_g5k_hydra`
    # regenera só o GH200 (evita mexer nos CSVs do a100/ddr).
    if len(sys.argv) > 1:
        _f = sys.argv[1]
        _exact = [d for d in gpu_dirs if os.path.basename(d) == _f]
        gpu_dirs = _exact if _exact else [d for d in gpu_dirs if _f in os.path.basename(d)]
    if not gpu_dirs:
        print("Nenhuma pasta nvidia-* encontrada em new_results/.")
    else:
        for gpu_dir in gpu_dirs:
            gpu_short = GPU_SHORT.get(os.path.basename(gpu_dir), os.path.basename(gpu_dir))
            print(f"\nGerando gráficos para {gpu_short} ({os.path.basename(gpu_dir)})...")
            generate_for_gpu(gpu_dir)


if __name__ == "__main__":
    _main()

# -*- coding: utf-8 -*-
"""
Comparacao: MESMA abordagem (tensorflow_opt / InceptionV3) na MESMA GPU (GH200),
variando o DATASET -> HCPA (Brasil) vs DDR (China). 10 runs cada.
Gera um grafico com painel clinico + painel computacional (media +- desvio).
"""
import csv, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.expanduser('~/projects/hcpa')
SETS = {
    'HCPA (Brazil)':  f'{BASE}/tensorflow_opt/results/nvidia-gh200-480gb_g5k_hydra',
    'DDR (China)':    f'{BASE}/new_results/nvidia-gh200-480gb-ddr_g5k_hydra',
}

def load_run(csv_path):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    ft = [r for r in rows if r['stage'] == 'final_test'][0]
    bestval = max(float(r['val_auc']) for r in rows if r['val_auc'])
    # throughput de regime: media das epocas finetune exceto as 2 primeiras (warmup/XLA)
    fine = [r for r in rows if r['stage'] == 'finetune' and r['train_throughput_img_s']]
    thr = st.mean(float(r['train_throughput_img_s']) for r in fine[2:]) if len(fine) > 2 else float('nan')
    return {
        'test_auc': float(ft['test_auc']),
        'sens': float(ft['test_sens']),
        'spec': float(ft['test_spec']),
        'prec': float(ft['test_precision']),
        'f1': float(ft['test_f1']),
        'throughput': thr,
        'latency': float(ft['test_inference_latency_ms_img']),
    }

def collect(folder):
    runs = []
    for n in range(10):
        for cand in (f'{folder}/run_{n}/InceptionV3-{n}.csv',):
            if os.path.exists(cand):
                runs.append(load_run(cand)); break
    return runs

data = {name: collect(f) for name, f in SETS.items()}
for name, runs in data.items():
    print(f"{name}: {len(runs)} runs")

def stat(name, key):
    v = [r[key] for r in data[name]]
    return st.mean(v), st.pstdev(v)

colors = {'HCPA (Brazil)': '#2c7fb8', 'DDR (China)': '#de2d26'}
names = list(SETS.keys())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- Painel 1: clinico ----
clin = [('test_AUC', 'test_auc'), ('Sensitivity\n@0.5', 'sens'),
        ('Specificity\n@0.5', 'spec'), ('Precision\n@0.5', 'prec'), ('F1\n@0.5', 'f1')]
x = np.arange(len(clin)); w = 0.38
ax = axes[0]
for i, name in enumerate(names):
    means = [stat(name, k)[0] for _, k in clin]
    errs  = [stat(name, k)[1] for _, k in clin]
    bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=4,
                  label=name, color=colors[name], alpha=0.9)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.012, f'{m:.3f}',
                ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in clin])
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title('Clinical metrics (test set)', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)

# ---- Painel 2: computacional ----
ax = axes[1]
comp = [('Train throughput\n(img/s)', 'throughput', 1),
        ('Inference latency\n(ms/img)', 'latency', 1)]
# eixos separados por escala -> usar dois subgrupos com twin
labels = [c[0] for c in comp]
xc = np.arange(len(comp))
# normaliza visualmente: barras com valor anotado, eixo log nao; usamos texto
for i, name in enumerate(names):
    thr_m, thr_s = stat(name, 'throughput')
    lat_m, lat_s = stat(name, 'latency')
    vals = [thr_m, lat_m]; errs = [thr_s, lat_s]
# desenho com dois eixos y
ax2 = ax
axb = ax.twinx()
wc = 0.38
for i, name in enumerate(names):
    thr_m, thr_s = stat(name, 'throughput')
    ax2.bar(0 + (i-0.5)*wc, thr_m, wc, yerr=thr_s, capsize=4, color=colors[name], alpha=0.9)
    ax2.text((i-0.5)*wc, thr_m+30, f'{thr_m:.0f}', ha='center', fontsize=8)
for i, name in enumerate(names):
    lat_m, lat_s = stat(name, 'latency')
    axb.bar(1 + (i-0.5)*wc, lat_m, wc, yerr=lat_s, capsize=4, color=colors[name], alpha=0.9,
            hatch='//')
    axb.text(1+(i-0.5)*wc, lat_m+0.03, f'{lat_m:.2f}', ha='center', fontsize=8)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels)
ax2.set_ylabel('Throughput (img/s)'); axb.set_ylabel('Latency (ms/img)')
ax2.set_ylim(0, 4600); axb.set_ylim(0, 4)
ax2.set_title('Computational metrics (GH200) — invariant to dataset', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('tensorflow_opt (InceptionV3) on NVIDIA GH200 — HCPA (Brazil) vs DDR (China)\n'
             'Same approach, same GPU, different dataset · mean ± std over 10 runs',
             fontweight='bold', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f'{BASE}/new_results/comparison_tfopt_gh200_HCPA_vs_DDR.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('salvo:', out)

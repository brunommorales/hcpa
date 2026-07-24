#!/usr/bin/env python3
"""Valida o resultado de um smoke-test: o CSV e o profiling de kernel.

Checa exatamente as coisas que mudamos, e falha alto se alguma regrediu.
Não olha convergência — o teste é de instrumentação.

Uso:
    python3 tools/smoke_check.py <dir_do_run>          # ex: .../run_0
    python3 tools/smoke_check.py <dir> --approach tensorflow_opt
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

EXPECTED_COLS = [
    "epoch", "stage",
    "train_loss", "train_throughput_img_s", "train_elapsed_s",
    "train_gpu_mem_peak_mb", "train_energy_j", "train_avg_power_w",
    "train_gpu_util_pct", "train_mem_util_pct", "train_busy_time_s",
    "val_loss", "val_auc", "val_precision", "val_f1", "val_sens", "val_spec",
    "val_spec_at_sens95", "val_elapsed_s", "val_gpu_mem_peak_mb", "val_energy_j",
    "test_auc", "test_precision", "test_f1", "test_sens", "test_spec",
    "test_spec_at_sens95", "test_throughput_img_s", "test_elapsed_s",
    "test_gpu_mem_peak_mb", "test_energy_j",
    "lr", "total_train_time_s",
]

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, status, detail=""):
    results.append((status, name, detail))


def fv(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--approach", default="?")
    a = ap.parse_args()

    csvs = [p for p in glob.glob(os.path.join(a.run_dir, "*.csv"))
            if not p.endswith("-thresholds.csv")]
    if not csvs:
        print(f"FAIL: nenhum CSV em {a.run_dir}")
        return 2
    rows = list(csv.DictReader(open(csvs[0])))
    cols = list(rows[0].keys()) if rows else []
    tr = [r for r in rows if r.get("stage") not in ("final_test", "test")]
    fin = [r for r in rows if r.get("stage") in ("final_test", "test")]

    # 1) schema exato
    check("schema = 33 colunas na ordem canônica",
          PASS if cols == EXPECTED_COLS else FAIL,
          "" if cols == EXPECTED_COLS
          else f"extra={set(cols)-set(EXPECTED_COLS)} faltando={set(EXPECTED_COLS)-set(cols)}")

    # 2) épocas gravadas
    check("épocas de treino gravadas", PASS if tr else FAIL, f"{len(tr)} linhas")
    check("linha de teste final presente", PASS if fin else FAIL)

    # 3) energia de TREINO por época > 0 em todas
    e = [fv(r.get("train_energy_j")) for r in tr]
    ok = all(v is not None and v > 0 for v in e)
    check("train_energy_j > 0 em todas as épocas", PASS if ok else FAIL,
          f"min={min((x for x in e if x is not None), default=None)}")

    # 4) energia de VALIDAÇÃO por época > 0  <-- era 0 no TensorFlow
    ev = [fv(r.get("val_energy_j")) for r in tr]
    okv = all(v is not None and v > 0 for v in ev)
    check("val_energy_j > 0 em todas as épocas (TF gravava None)",
          PASS if okv else FAIL,
          f"nao-nulos={sum(1 for v in ev if v)} / {len(ev)}")

    # 5) val_spec_at_sens95 preenchido e DIFERENTE de val_spec
    s95 = [fv(r.get("val_spec_at_sens95")) for r in tr]
    sp = [fv(r.get("val_spec")) for r in tr]
    check("val_spec_at_sens95 preenchido por época",
          PASS if all(v is not None for v in s95) else FAIL)
    diff = sum(1 for x, y in zip(s95, sp) if x is not None and y is not None and abs(x - y) > 1e-6)
    check("val_spec != val_spec_at_sens95 (métricas distintas)",
          PASS if diff > 0 else WARN,
          f"diferem em {diff}/{len(tr)} épocas")

    # 6) util% e mem_util% presentes (telemetria em thread de fundo)
    for col in ("train_gpu_util_pct", "train_mem_util_pct", "train_busy_time_s",
                "train_gpu_mem_peak_mb", "train_avg_power_w"):
        vals = [fv(r.get(col)) for r in tr]
        check(f"{col} preenchido",
              PASS if all(v is not None for v in vals) else FAIL)

    # 7) coerência física: busy_time <= elapsed ; energia/tempo dentro do TDP
    bad = [(fv(r.get("train_busy_time_s")), fv(r.get("train_elapsed_s"))) for r in tr]
    okb = all(b is not None and el is not None and b <= el * 1.01 for b, el in bad)
    check("train_busy_time_s <= train_elapsed_s", PASS if okb else FAIL)

    pw = [(fv(r.get("train_energy_j")), fv(r.get("train_elapsed_s"))) for r in tr]
    watts = [e_ / t for e_, t in pw if e_ and t]
    okw = all(0 < w < 1000 for w in watts)
    check("potência derivada (E/t) plausível (0-1000 W)", PASS if okw else FAIL,
          f"min={min(watts):.0f}W max={max(watts):.0f}W" if watts else "")

    # 8) métricas clínicas de TREINO removidas
    leftovers = [c for c in cols if c.startswith("train_") and
                 c.split("train_")[1] in ("auc", "precision", "f1", "sens", "spec")]
    check("métricas clínicas de treino removidas", PASS if not leftovers else FAIL,
          str(leftovers))

    # 9) profiling de kernel (CUPTI), se foi pedido
    kdirs = glob.glob(os.path.join(a.run_dir, "kprof-*"))
    kfiles = [f for d in kdirs for f in glob.glob(os.path.join(d, "epoch_*.json"))]
    if kfiles:
        k = json.loads(open(kfiles[0]).read())["summary"]
        check("CUPTI: JSON de kernel gerado", PASS, os.path.basename(kfiles[0]))
        check("CUPTI: n_kernels > 0", PASS if k["n_kernels"] > 0 else FAIL,
              f"{k['n_kernels']} kernels")
        check("CUPTI: busy_time <= wall_time", PASS if k["busy_time_s"] <= k["wall_time_s"] * 1.01 else FAIL,
              f"busy={k['busy_time_s']:.2f}s wall={k['wall_time_s']:.2f}s")
        check("CUPTI: soma >= união (streams sobrepostos)",
              PASS if k["kernel_sum_time_s"] >= k["busy_time_s"] * 0.999 else FAIL,
              f"overlap={k['overlap_factor']:.2f}x")
        check("CUPTI: cpu_wait_time >= 0", PASS if k["cpu_wait_time_s"] >= 0 else FAIL,
              f"cpu_wait={k['cpu_wait_time_s']:.2f}s ({100*k['cpu_wait_time_s']/k['wall_time_s']:.0f}% do wall)")
        if "n_h2d" in k:
            # H2D deve ser da ORDEM de 1-4 por batch (imagens + rotulos). Centenas
            # por batch significa que d2d/memset internos vazaram para o balde.
            check("CUPTI: H2D separado de D2D/memset", PASS,
                  f"h2d={k['n_h2d']} d2h={k['n_d2h']} d2d={k['n_d2d']} memset={k['n_memset']}")
            check("CUPTI: transfer_time < device_work_time",
                  PASS if k["transfer_time_s"] < k["device_work_time_s"] else WARN,
                  f"transfer={k['transfer_time_s']:.3f}s device_work={k['device_work_time_s']:.3f}s")
        else:
            check("CUPTI: tipos de copia (h2d/d2h/d2d/memset)", WARN,
                  "trace no formato antigo; regravar para auditar")
    else:
        check("CUPTI: JSON de kernel", WARN, "nenhum (HCPA_PROFILE_EPOCHS nao setado?)")

    # ---------------- relatório
    print(f"\n=== smoke_check: {a.approach} ({os.path.basename(a.run_dir)}) ===")
    w = max(len(n) for _, n, _ in results)
    for st, n, d in results:
        mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn "}[st]
        print(f"[{mark}] {n:<{w}}  {d}")
    nf = sum(1 for s, _, _ in results if s == FAIL)
    nw = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n  {len(results)-nf-nw} ok, {nw} avisos, {nf} FALHAS")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())

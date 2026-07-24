#!/usr/bin/env python3
"""Resume a mini-varredura de LR: pico de val_auc por (abordagem, LR).

Lê <approach>/results/lrsweep_<lr>_g5k_hydra/run_0/*.csv, tanto no repo local
quanto num diretório baixado do Grid5000.

Uso:
    python3 tools/lr_sweep_report.py [raiz_do_projeto]
"""
import csv
import re
import sys
from pathlib import Path

APPROACHES = ["tensorflow_opt", "pytorch_opt"]


def fv(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def read_run(run_dir: Path):
    csvs = [p for p in run_dir.glob("*.csv") if not p.name.endswith("-thresholds.csv")]
    if not csvs:
        return None
    rows = list(csv.DictReader(csvs[0].open()))
    train = [r for r in rows if r.get("stage") not in ("final_test", "test")]
    aucs = [(i, fv(r, "val_auc")) for i, r in enumerate(train)]
    aucs = [(i, a) for i, a in aucs if a is not None]
    if not aucs:
        return None
    best_ep, best_auc = max(aucs, key=lambda t: t[1])

    spec = [fv(r, "val_spec_at_sens95") for r in train]
    spec_at_best = spec[best_ep] if best_ep < len(spec) else None

    energy = sum(fv(r, "train_energy_j") or 0.0 for r in train)
    return {
        "epochs": len(train),
        "best_epoch": best_ep,
        "best_val_auc": best_auc,
        "final_val_auc": aucs[-1][1],
        "spec95_at_best": spec_at_best,
        "energy_kj": energy / 1000.0,
    }


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    table = {}
    lrs = set()
    for ap in APPROACHES:
        for d in sorted((root / ap / "results").glob("lrsweep_*_g5k_hydra/run_0")):
            m = re.match(r"lrsweep_(.+)_g5k_hydra", d.parent.name)
            if not m:
                continue
            lr = m.group(1)
            r = read_run(d)
            if r:
                table[(ap, lr)] = r
                lrs.add(lr)

    if not table:
        print("nenhum resultado de varredura encontrado.")
        print("esperado: <approach>/results/lrsweep_<lr>_g5k_hydra/run_0/*.csv")
        return 1

    lrs = sorted(lrs, key=lambda s: float(s))
    print(f"{'abordagem':<17}{'LR':>8}{'ep':>5}{'best_val_auc':>14}"
          f"{'final':>9}{'spec@95':>10}{'energia(kJ)':>13}")
    print("-" * 76)
    for ap in APPROACHES:
        for lr in lrs:
            r = table.get((ap, lr))
            if not r:
                continue
            sp = f"{r['spec95_at_best']:.4f}" if r["spec95_at_best"] is not None else "n/a"
            print(f"{ap:<17}{lr:>8}{r['best_epoch']:>5}{r['best_val_auc']:>14.4f}"
                  f"{r['final_val_auc']:>9.4f}{sp:>10}{r['energy_kj']:>13.1f}")

    # LR que maximiza a PIOR AUC entre os frameworks -> o melhor compromisso comum
    print()
    best_lr, best_score = None, -1.0
    for lr in lrs:
        got = [table[(ap, lr)]["best_val_auc"] for ap in APPROACHES if (ap, lr) in table]
        if len(got) < len(APPROACHES):
            continue
        score = min(got)
        print(f"LR={lr:>7}  pior AUC entre os frameworks = {score:.4f}")
        if score > best_score:
            best_lr, best_score = lr, score
    if best_lr:
        print(f"\n=> LR comum recomendado: {best_lr}  (maximiza a pior AUC entre os dois)")
        print(f"   aplique com: HCPA_LRATE={best_lr}  (ou edite tools/common_recipe.sh)")
    else:
        print("=> nenhum LR tem resultado nos dois frameworks ainda.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

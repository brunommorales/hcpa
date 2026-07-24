#!/usr/bin/env python3
"""
Offline re-analysis of the 160 HCPA runs. No GPU, no re-training.

Everything here is recomputed from artefacts that already exist on disk:
  - <gpu>/<approach>/run_<i>/*.csv          per-epoch history (val_auc, train_energy_j)
  - <gpu>/<approach>/run_<i>/*-thresholds.csv   test ROC curve of the selected checkpoint
  - <gpu>/<gpu>_all_runs.csv                per-run summary

Produces analysis/reanalysis.json plus a readable dump on stdout.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "analysis")
os.makedirs(OUT, exist_ok=True)

GPUS = {
    "GH200": "nvidia-gh200-480gb_g5k_hydra",
    "A100": "nvidia-a100-sxm4-40gb_g5k_chuc",
}
# canonical order and paper labels
APPROACHES = [
    ("pytorch_opt", "PyTorch (opt.)"),
    ("tensorflow_opt", "TensorFlow (opt.)"),
    ("hybrid_token_reduction_opt", "Hybrid (TR, opt.)"),
    ("retfound_green", "RETFound-Green"),
    ("pytorch_base", "PyTorch (base)"),
    ("tensorflow_base", "TensorFlow (base)"),
    ("hybrid_simple", "Hybrid (plain)"),
    ("hybrid_token_reduction", "Hybrid (TR)"),
]
LABEL = dict(APPROACHES)
KEYS = [a for a, _ in APPROACHES]

N_POS, N_NEG = 1257, 3559  # test set composition, recovered from the ROC granularity
RNG = np.random.default_rng(20260723)
NBOOT = 10000


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_summary():
    frames = {}
    for gpu, d in GPUS.items():
        f = glob.glob(os.path.join(ROOT, d, "*_all_runs.csv"))[0]
        df = pd.read_csv(f)
        df["gpu"] = gpu
        frames[gpu] = df
    return frames


def epoch_files(gpu_dir, approach):
    """Per-epoch history files, ordered by run index."""
    out = []
    for i in range(10):
        pats = [
            f"{gpu_dir}/{approach}/run_{i}/inception_v3-{i}.csv",
            f"{gpu_dir}/{approach}/run_{i}/InceptionV3-{i}.csv",
            f"{gpu_dir}/{approach}/run_{i}/metrics_exec{i}.csv",
        ]
        hit = [p for p in pats if os.path.exists(p)]
        if hit:
            out.append(hit[0])
    return out


def threshold_files(gpu_dir, approach):
    out = []
    for i in range(10):
        p = f"{gpu_dir}/{approach}/run_{i}/all-tfrec-v2-{i}-thresholds.csv"
        if os.path.exists(p):
            out.append(p)
    return out


# --------------------------------------------------------------------------
# statistics helpers (pure, no side effects)
# --------------------------------------------------------------------------
def exact_mwu(a, b):
    """Two-sided Mann-Whitney with the exact null distribution (n=10 each)."""
    res = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return float(res.pvalue)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def boot_ci_diff(a, b, nboot=NBOOT, rng=RNG):
    """Percentile bootstrap CI for mean(a) - mean(b), resampling seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = rng.integers(0, len(a), (nboot, len(a)))
    ib = rng.integers(0, len(b), (nboot, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------------------
# A. descriptives with dispersion
# --------------------------------------------------------------------------
METRICS = {
    "auc": ("test_auc", 100.0),
    "spec95": ("test_spec_at_sens95", 100.0),
    "e200": ("total_energy_kj", 1.0),
    "ees": ("energy_es_kj", 1.0),
    "mem": ("gpu_mem_peak_mb", 1 / 1024.0),
    "time": ("train_compute_s", 1.0),
    "thr": ("throughput_img_s", 1.0),
    "util": ("gpu_util_pct", 1.0),
    "power": ("avg_power_derived_w", 1.0),
}


def descriptives(frames):
    rows = {}
    for gpu, df in frames.items():
        for a in KEYS:
            sub = df[df.approach == a]
            rec = {}
            for name, (col, scale) in METRICS.items():
                v = sub[col].values.astype(float) * scale
                rec[name] = dict(
                    mean=float(v.mean()), sd=float(v.std(ddof=1)),
                    med=float(np.median(v)),
                    q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                    vals=v.tolist(),
                )
            # CEC per run, then mean (this is what the paper does)
            spec = sub.test_spec_at_sens95.values.astype(float)
            for tag, col in (("cec200", "total_energy_kj"), ("cec_es", "energy_es_kj")):
                c = (1.0 - spec) * sub[col].values.astype(float)
                rec[tag] = dict(mean=float(c.mean()), sd=float(c.std(ddof=1)),
                                med=float(np.median(c)), vals=c.tolist())
            rows[(gpu, a)] = rec
    return rows


# --------------------------------------------------------------------------
# B. pairwise tests, exact + Holm within family
# --------------------------------------------------------------------------
def pairwise(frames, desc):
    out = []
    for gpu in GPUS:
        for name in ("auc", "spec95", "e200", "ees", "mem", "cec200", "cec_es"):
            fam = []
            for i in range(len(KEYS)):
                for j in range(i + 1, len(KEYS)):
                    a, b = KEYS[i], KEYS[j]
                    va = np.array(desc[(gpu, a)][name]["vals"])
                    vb = np.array(desc[(gpu, b)][name]["vals"])
                    p = exact_mwu(va, vb)
                    d, lo, hi = boot_ci_diff(va, vb)
                    fam.append(dict(gpu=gpu, metric=name, a=a, b=b, p_raw=p,
                                    delta=d, ci_lo=lo, ci_hi=hi,
                                    cliff=cliffs_delta(va, vb)))
            adj = holm([f["p_raw"] for f in fam])
            for f, q in zip(fam, adj):
                f["p_holm"] = float(q)
            out.extend(fam)
    return out


# --------------------------------------------------------------------------
# C. Spearman between GPUs, bootstrapped over seeds
# --------------------------------------------------------------------------
def spearman_boot(frames, nboot=NBOOT):
    """Paired bootstrap: one seed resample per (gpu, approach) per replicate, reused
    for both protocols, so that the difference between the two rho values is formed
    on the same resampled runs rather than on independent draws."""
    cec = {}  # (gpu, approach, protocol) -> per-run CEC
    for tag, col in (("fixed", "total_energy_kj"), ("es", "energy_es_kj")):
        for gpu, df in frames.items():
            for a in KEYS:
                s = df[df.approach == a]
                cec[(gpu, a, tag)] = ((1 - s.test_spec_at_sens95.values) * s[col].values)

    def rho_of(sel, tag):
        g = [cec[("GH200", a, tag)][sel[("GH200", a)]].mean() for a in KEYS]
        h = [cec[("A100", a, tag)][sel[("A100", a)]].mean() for a in KEYS]
        return spearmanr(g, h).statistic

    full = {(gpu, a): np.arange(10) for gpu in GPUS for a in KEYS}
    obs = {t: float(rho_of(full, t)) for t in ("fixed", "es")}

    draws = {t: np.empty(nboot) for t in ("fixed", "es")}
    for k in range(nboot):
        sel = {(gpu, a): RNG.integers(0, 10, 10) for gpu in GPUS for a in KEYS}
        for t in ("fixed", "es"):
            draws[t][k] = rho_of(sel, t)

    res = {t: dict(rho=obs[t],
                   lo=float(np.percentile(draws[t], 2.5)),
                   hi=float(np.percentile(draws[t], 97.5)))
           for t in ("fixed", "es")}
    diff = draws["es"] - draws["fixed"]
    res["diff"] = dict(delta=obs["es"] - obs["fixed"],
                       lo=float(np.percentile(diff, 2.5)),
                       hi=float(np.percentile(diff, 97.5)),
                       p_two_sided=float(2 * min((diff <= 0).mean(), (diff >= 0).mean())))
    return res


# --------------------------------------------------------------------------
# D. patience sweep, replayed from the per-epoch histories
# --------------------------------------------------------------------------
def patience_sweep(patiences=(10, 20, 30, 40, 50)):
    out = {}
    for gpu, d in GPUS.items():
        gdir = os.path.join(ROOT, d)
        for a in KEYS:
            recs = {p: dict(stop=[], best=[], energy=[], pen=[]) for p in patiences}
            full_e, peak_e, peak_pen = [], [], []
            for f in epoch_files(gdir, a):
                df = pd.read_csv(f)
                df = df[df.stage != "final_test"]
                v = df.val_auc.values.astype(float)
                e = df.train_energy_j.values.astype(float) / 1000.0
                cum = np.cumsum(e)
                full_e.append(cum[-1])
                gbest = int(np.argmax(v))
                peak_e.append(cum[gbest])
                peak_pen.append(0.0)
                for P in patiences:
                    best, bi = -np.inf, 0
                    stop = len(v) - 1
                    for i, x in enumerate(v):
                        if x > best:
                            best, bi = x, i
                        elif i - bi >= P:
                            stop = i
                            break
                    recs[P]["stop"].append(stop)
                    recs[P]["best"].append(bi)
                    recs[P]["energy"].append(cum[stop])
                    recs[P]["pen"].append(v.max() - v[bi])
            entry = dict(full_energy=float(np.mean(full_e)),
                         peak_energy=float(np.mean(peak_e)),
                         peak_epoch=float(np.mean([0])),  # filled below
                         patience={})
            for P in patiences:
                r = recs[P]
                entry["patience"][P] = dict(
                    stop_epoch=float(np.mean(r["stop"])), stop_sd=float(np.std(r["stop"], ddof=1)),
                    best_epoch=float(np.mean(r["best"])), best_sd=float(np.std(r["best"], ddof=1)),
                    energy=float(np.mean(r["energy"])), energy_sd=float(np.std(r["energy"], ddof=1)),
                    saved_pct=float(100 * (1 - np.mean(r["energy"]) / np.mean(full_e))),
                    val_penalty_pp=float(100 * np.mean(r["pen"])),
                    val_penalty_max_pp=float(100 * np.max(r["pen"])),
                )
            out[f"{gpu}/{a}"] = entry
    return out


def peak_epoch_stats():
    """Dispersion of the epoch at which validation AUC peaks."""
    out = {}
    for gpu, d in GPUS.items():
        gdir = os.path.join(ROOT, d)
        for a in KEYS:
            eps = []
            for f in epoch_files(gdir, a):
                df = pd.read_csv(f)
                df = df[df.stage != "final_test"]
                eps.append(int(np.argmax(df.val_auc.values)))
            out[f"{gpu}/{a}"] = dict(mean=float(np.mean(eps)), sd=float(np.std(eps, ddof=1)),
                                     lo=int(np.min(eps)), hi=int(np.max(eps)), vals=eps)
    return out


# --------------------------------------------------------------------------
# E. threshold-selection optimism, from the reconstructed test ROC
# --------------------------------------------------------------------------
def roc_to_labels(path):
    """Recover the ranked label vector of the test set from a saved ROC curve.

    The curve stores (tpr, fpr) at every distinct score. Multiplying by the known
    class counts gives cumulative TP/FP, and consecutive differences give how many
    positives and negatives share each score. That is all a bootstrap needs.
    """
    df = pd.read_csv(path)
    tp = np.rint(df.tpr.values * N_POS).astype(int)
    fp = np.rint(df.fpr.values * N_NEG).astype(int)
    dtp = np.diff(np.concatenate([[0], tp]))
    dfp = np.diff(np.concatenate([[0], fp]))
    lab = []
    for p, n in zip(dtp, dfp):
        lab.extend([1] * max(p, 0))
        lab.extend([0] * max(n, 0))
    lab = np.array(lab, dtype=np.int8)
    assert lab.sum() == N_POS and len(lab) == N_POS + N_NEG, (lab.sum(), len(lab))
    return lab  # ordered from highest score to lowest


def spec_at_sens(labels, target=0.95):
    """In-sample spec@95: largest specificity among operating points with sens>=target."""
    npos, nneg = labels.sum(), len(labels) - labels.sum()
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    sens = tp / npos
    spec = 1 - fp / nneg
    ok = sens >= target
    return float(spec[ok].max()) if ok.any() else 0.0


def split_half_optimism(labels, nrep=400, rng=RNG):
    """Pick the operating point on one half, read sens/spec on the other."""
    n = len(labels)
    npos = labels.sum()
    gaps, sens_out, spec_out = [], [], []
    for _ in range(nrep):
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
        la, lb = labels[np.sort(A)], labels[np.sort(B)]
        # in-sample point on A -> index k = number of top-ranked samples kept
        tp = np.cumsum(la); fp = np.cumsum(1 - la)
        se = tp / la.sum(); sp = 1 - fp / (len(la) - la.sum())
        ok = se >= 0.95
        if not ok.any():
            continue
        k = int(np.argmax(np.where(ok, sp, -1)))
        frac = (k + 1) / len(la)           # transfer the operating point by rank fraction
        kb = min(len(lb) - 1, int(round(frac * len(lb))) - 1)
        tpb = np.cumsum(lb); fpb = np.cumsum(1 - lb)
        seb = tpb / lb.sum(); spb = 1 - fpb / (len(lb) - lb.sum())
        gaps.append(sp[k] - spb[kb])
        sens_out.append(seb[kb])
        spec_out.append(spb[kb])
    return (float(np.mean(gaps)), float(np.std(gaps, ddof=1)),
            float(np.mean(sens_out)), float(np.mean(spec_out)))


def optimism():
    out = {}
    for gpu, d in GPUS.items():
        gdir = os.path.join(ROOT, d)
        for a in KEYS:
            g, s_out, sp_out, ins = [], [], [], []
            for f in threshold_files(gdir, a):
                lab = roc_to_labels(f)
                ins.append(spec_at_sens(lab))
                gap, _, se, sp = split_half_optimism(lab)
                g.append(gap); s_out.append(se); sp_out.append(sp)
            out[f"{gpu}/{a}"] = dict(
                in_sample_spec95=float(100 * np.mean(ins)),
                optimism_pp=float(100 * np.mean(g)), optimism_sd_pp=float(100 * np.std(g, ddof=1)),
                transferred_sens=float(100 * np.mean(s_out)),
                transferred_spec=float(100 * np.mean(sp_out)),
                gaps=[float(100 * x) for x in g],
            )
    return out


# --------------------------------------------------------------------------
# F. total-cost formulation with an explicit deployment volume
# --------------------------------------------------------------------------
def breakeven(desc):
    """C_k(N) = E_k + N * (1-Spec_k) * kappa.

    kappa is the exchange rate: kJ of training energy a centre would spend to
    avoid one unnecessary referral. Report the kappa at which two configurations
    tie, for a given N, and the N at which they tie for a given kappa.
    """
    out = {}
    for gpu in GPUS:
        E = {a: desc[(gpu, a)]["e200"]["mean"] for a in KEYS}
        Ees = {a: desc[(gpu, a)]["ees"]["mean"] for a in KEYS}
        F = {a: 1 - desc[(gpu, a)]["spec95"]["mean"] / 100 for a in KEYS}
        pairs = {}
        for i in range(len(KEYS)):
            for j in range(len(KEYS)):
                a, b = KEYS[i], KEYS[j]
                if a == b or not (E[a] > E[b] and F[a] < F[b]):
                    continue  # only pairs where the pricier one is clinically better
                dE, dF = E[a] - E[b], F[b] - F[a]
                pairs[f"{a}_vs_{b}"] = dict(
                    dE_kj=dE, dFRR=dF,
                    kappa_at_N=  {N: dE / (N * dF) for N in (1000, 10000, 20000, 100000)},
                    kappa_at_N_es={N: (Ees[a]-Ees[b]) / (N * dF) for N in (1000, 10000, 20000, 100000)},
                )
        out[gpu] = pairs
    return out


# --------------------------------------------------------------------------
def main():
    frames = load_summary()
    desc = descriptives(frames)
    res = {}

    print("=" * 78)
    print("A. DESCRIPTIVES (mean +/- sd  [median])")
    print("=" * 78)
    for gpu in GPUS:
        print(f"\n--- {gpu} ---")
        hdr = f"{'config':22s} {'AUC%':>14s} {'Spec95%':>14s} {'E200 kJ':>13s} {'Ees kJ':>13s} {'Mem GB':>12s} {'time s':>12s}"
        print(hdr)
        for a in KEYS:
            r = desc[(gpu, a)]
            def f(k, n=2):
                return f"{r[k]['mean']:.{n}f}+-{r[k]['sd']:.{n}f}"
            print(f"{LABEL[a]:22s} {f('auc'):>14s} {f('spec95'):>14s} "
                  f"{f('e200',0):>13s} {f('ees',0):>13s} {f('mem',1):>12s} {f('time',0):>12s}")
    res["descriptives"] = {f"{g}/{a}": {k: {kk: vv for kk, vv in v.items() if kk != 'vals'}
                                        for k, v in desc[(g, a)].items()}
                           for g in GPUS for a in KEYS}

    print("\n" + "=" * 78)
    print("B. NORMALITY (Shapiro-Wilk, alpha=0.05)")
    print("=" * 78)
    rej = 0; tot = 0
    for gpu in GPUS:
        for a in KEYS:
            for m in ("auc", "spec95", "e200", "mem"):
                v = np.array(desc[(gpu, a)][m]["vals"])
                tot += 1
                if shapiro(v).pvalue < 0.05:
                    rej += 1
    print(f"rejected in {rej} of {tot} groups")
    res["shapiro"] = dict(rejected=rej, total=tot)

    print("\n" + "=" * 78)
    print("C. PAIRWISE TESTS (exact Mann-Whitney, Holm within GPU x metric)")
    print("=" * 78)
    pw = pairwise(frames, desc)
    res["pairwise"] = pw
    for gpu in GPUS:
        for m in ("auc", "spec95", "cec_es"):
            fam = [x for x in pw if x["gpu"] == gpu and x["metric"] == m]
            raw = sum(1 for x in fam if x["p_raw"] < 0.05)
            adj = sum(1 for x in fam if x["p_holm"] < 0.05)
            print(f"{gpu:6s} {m:8s}: {raw:2d}/28 significant raw -> {adj:2d}/28 after Holm")

    print("\n  key comparisons (delta = first minus second, 95% bootstrap CI):")
    for gpu in GPUS:
        for a, b in (("hybrid_token_reduction_opt", "retfound_green"),
                     ("hybrid_token_reduction_opt", "pytorch_opt")):
            for m in ("auc", "spec95"):
                x = [k for k in pw if k["gpu"] == gpu and k["metric"] == m
                     and {k["a"], k["b"]} == {a, b}][0]
                sgn = 1 if x["a"] == a else -1
                print(f"  {gpu:6s} {LABEL[a]:18s} vs {LABEL[b]:18s} {m:7s}: "
                      f"d={sgn*x['delta']:+.3f} CI[{sgn*x['ci_hi'] if sgn<0 else x['ci_lo']:+.3f},"
                      f"{sgn*x['ci_lo'] if sgn<0 else x['ci_hi']:+.3f}] "
                      f"p_raw={x['p_raw']:.4f} p_holm={x['p_holm']:.4f}")

    print("\n" + "=" * 78)
    print("D. CROSS-GPU RANKING AGREEMENT (Spearman, bootstrap over seeds)")
    print("=" * 78)
    sp = spearman_boot(frames)
    res["spearman"] = sp
    print(f"  fixed budget : rho={sp['fixed']['rho']:.3f}  95% CI [{sp['fixed']['lo']:.3f}, {sp['fixed']['hi']:.3f}]")
    print(f"  stopping rule: rho={sp['es']['rho']:.3f}  95% CI [{sp['es']['lo']:.3f}, {sp['es']['hi']:.3f}]")
    print(f"  difference   : {sp['diff']['delta']:+.3f} 95% CI [{sp['diff']['lo']:+.3f}, {sp['diff']['hi']:+.3f}] "
          f"p={sp['diff']['p_two_sided']:.4f}")

    print("\n" + "=" * 78)
    print("E. PEAK-EPOCH DISPERSION")
    print("=" * 78)
    pe = peak_epoch_stats()
    res["peak_epoch"] = pe
    for k, v in pe.items():
        print(f"  {k:45s} mean={v['mean']:6.1f} sd={v['sd']:6.1f} range=[{v['lo']},{v['hi']}]")

    print("\n" + "=" * 78)
    print("F. PATIENCE SWEEP (val-AUC penalty; test AUC at the stop is NOT available)")
    print("=" * 78)
    ps = patience_sweep()
    res["patience"] = ps
    for k, v in ps.items():
        row = " | ".join(f"P={P}: stop={d['stop_epoch']:5.1f} E={d['energy']:6.1f}kJ "
                         f"save={d['saved_pct']:4.1f}% pen={d['val_penalty_pp']:.2f}pp"
                         for P, d in v["patience"].items() if P in (10, 20, 30, 40))
        print(f"  {k:45s} full={v['full_energy']:7.1f}kJ")
        print(f"      {row}")

    print("\n" + "=" * 78)
    print("G. THRESHOLD-SELECTION OPTIMISM (split-half on the reconstructed test ROC)")
    print("=" * 78)
    op = optimism()
    res["optimism"] = op
    for k, v in op.items():
        print(f"  {k:45s} in-sample spec95={v['in_sample_spec95']:5.2f}%  "
              f"optimism={v['optimism_pp']:+5.2f}pp (sd {v['optimism_sd_pp']:.2f})  "
              f"transferred sens={v['transferred_sens']:5.2f}%")
    for gpu in GPUS:
        g = [op[f"{gpu}/{a}"]["optimism_pp"] for a in KEYS]
        print(f"  --> {gpu}: optimism across the eight spans {min(g):.2f} to {max(g):.2f} pp")

    print("\n" + "=" * 78)
    print("H. TOTAL COST WITH AN EXPLICIT DEPLOYMENT VOLUME")
    print("=" * 78)
    be = breakeven(desc)
    res["breakeven"] = be
    for gpu, pairs in be.items():
        print(f"\n--- {gpu} ---")
        for name, d in pairs.items():
            a, b = name.split("_vs_")
            print(f"  {LABEL[a]} costs {d['dE_kj']:.0f} kJ more than {LABEL[b]} "
                  f"and avoids {100*d['dFRR']:.2f} pp of unnecessary referrals")
            for N, kap in d["kappa_at_N"].items():
                print(f"      N={N:6d} patients -> tie at kappa={kap:8.3f} kJ per avoided referral")

    with open(os.path.join(OUT, "reanalysis.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nwrote {os.path.join(OUT, 'reanalysis.json')}")


if __name__ == "__main__":
    main()

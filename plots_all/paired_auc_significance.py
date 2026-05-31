#!/usr/bin/env python3
"""
Executa testes pareados de significancia para val_auc_final em single_gpu_runs.csv.

Regra:
- parear runs por framework, GPU e run_id
- aplicar Shapiro-Wilk nas diferencas (opt - base)
- usar paired t-test se p > 0.05; caso contrario, Wilcoxon signed-rank

Uso:
    /home/users/bmmorales/projects/hcpa/env/bin/python plots_all/paired_auc_significance.py
    /home/users/bmmorales/projects/hcpa/env/bin/python plots_all/paired_auc_significance.py --gpu H200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from scipy import stats

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RUNS_CSV = REPO_ROOT / "single_gpu_runs.csv"
DEFAULT_OUTPUT = ROOT / "paired_auc_significance.csv"

FRAMEWORK_VARIANTS = {
    "monai": ("base", "optimized"),
    "pytorch": ("clean", "original"),
    "tensorflow": ("clean", "original"),
}

CSV_FIELDS = [
    "gpu",
    "framework",
    "base_variant",
    "opt_variant",
    "n_pairs",
    "base_mean_auc",
    "opt_mean_auc",
    "mean_diff_auc",
    "shapiro_w",
    "shapiro_p",
    "selected_test",
    "test_statistic",
    "p_value",
    "significant_p_lt_0_05",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Filtra uma GPU especifica, ex.: H200")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV de saida (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def load_rows() -> list[dict[str, str]]:
    with RUNS_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect_auc_pairs(
    rows: list[dict[str, str]],
    framework: str,
    gpu: str,
    base_variant: str,
    opt_variant: str,
) -> tuple[list[float], list[float]]:
    base_by_run = {
        int(row["run_id"]): float(row["val_auc_final"])
        for row in rows
        if row["framework"] == framework
        and row["gpu"] == gpu
        and row["variant"] == base_variant
    }
    opt_by_run = {
        int(row["run_id"]): float(row["val_auc_final"])
        for row in rows
        if row["framework"] == framework
        and row["gpu"] == gpu
        and row["variant"] == opt_variant
    }

    common_run_ids = sorted(set(base_by_run) & set(opt_by_run))
    base_auc = [base_by_run[run_id] for run_id in common_run_ids]
    opt_auc = [opt_by_run[run_id] for run_id in common_run_ids]
    return base_auc, opt_auc


def run_significance_test(base_auc: list[float], opt_auc: list[float]) -> dict[str, float | str | bool]:
    diffs = [opt - base for base, opt in zip(base_auc, opt_auc)]
    shapiro = stats.shapiro(diffs)

    if shapiro.pvalue > 0.05:
        selected_test = "paired_t_test"
        test_result = stats.ttest_rel(opt_auc, base_auc)
    else:
        selected_test = "wilcoxon_signed_rank"
        test_result = stats.wilcoxon(
            opt_auc,
            base_auc,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )

    p_value = float(test_result.pvalue)
    return {
        "base_mean_auc": mean(base_auc),
        "opt_mean_auc": mean(opt_auc),
        "mean_diff_auc": mean(diffs),
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "selected_test": selected_test,
        "test_statistic": float(test_result.statistic),
        "p_value": p_value,
        "significant_p_lt_0_05": p_value < 0.05,
    }


def build_results(rows: list[dict[str, str]], gpu_filter: str | None) -> list[dict[str, object]]:
    gpus = sorted({row["gpu"] for row in rows})
    if gpu_filter:
        gpus = [gpu for gpu in gpus if gpu == gpu_filter]

    results: list[dict[str, object]] = []
    for gpu in gpus:
        for framework, (base_variant, opt_variant) in FRAMEWORK_VARIANTS.items():
            base_auc, opt_auc = collect_auc_pairs(rows, framework, gpu, base_variant, opt_variant)
            if len(base_auc) < 2 or len(base_auc) != len(opt_auc):
                continue

            result = {
                "gpu": gpu,
                "framework": framework,
                "base_variant": base_variant,
                "opt_variant": opt_variant,
                "n_pairs": len(base_auc),
            }
            result.update(run_significance_test(base_auc, opt_auc))
            results.append(result)

    return results


def write_results(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict[str, object]]) -> None:
    for row in results:
        print(
            f"{row['gpu']:>7} | {row['framework']:<10} | "
            f"{row['base_mean_auc']:.4f} -> {row['opt_mean_auc']:.4f} | "
            f"test={row['selected_test']} | p={row['p_value']:.6g} | "
            f"significant={row['significant_p_lt_0_05']}"
        )


def main() -> None:
    args = parse_args()
    rows = load_rows()
    results = build_results(rows, args.gpu)
    if not results:
        raise SystemExit("Nenhum par base/opt encontrado para os filtros informados.")

    write_results(results, args.output)
    print_summary(results)
    print(f"\nCSV salvo em: {args.output}")


if __name__ == "__main__":
    main()

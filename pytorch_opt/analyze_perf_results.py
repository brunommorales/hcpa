#!/usr/bin/env python3
"""
Aggregate training performance metrics from perf_results/logs_* directories.

The script parses run_<NGPU>gpu_<N>.log files produced by run_hcpa_perf.slurm,
extracts per-epoch throughput, learning-rate and validation AUC, and generates
summaries including averages, percentiles, and scalability indicators.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
EPOCH_RE = re.compile(r"\[E(\d+)/(\d+)\]\s+([^\s:]+):")
THROUGHPUT_RE = re.compile(r"thr=(\d+(?:\.\d+)?) img/s")
LR_RE = re.compile(r"lr=([0-9.eE+-]+)")
VALAUC_RE = re.compile(r"valAUC=([0-9.]+)")
ELAPSED_RE = re.compile(r"elapsed=(\d+(?:\.\d+)?)s")
VALID_AUC_RE = re.compile(r"Valid AUC:\s*(\d+(?:\.\d+)?)")
FINAL_AUC_RE = re.compile(r"AUC=([0-9.]+)")
LOG_NAME_RE = re.compile(r"run_(\d+)gpu_\d+\.log$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_vals[int(pos)])
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    fraction = pos - lower
    return float(lower_val + (upper_val - lower_val) * fraction)


def safe_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return math.nan
    return statistics.mean(finite)


def safe_stdev(values: Sequence[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) <= 1:
        return 0.0 if len(finite) == 1 else math.nan
    return statistics.stdev(finite)


def describe(values: Sequence[float]) -> tuple[float, float]:
    return safe_mean(values), safe_stdev(values)


def compute_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"mean": None, "stdev": None}
    if len(finite) == 1:
        return {"mean": finite[0], "stdev": 0.0}
    return {"mean": statistics.mean(finite), "stdev": statistics.stdev(finite)}


def format_stat(mean: Optional[float], stdev: Optional[float], value_fmt: str = "{:.2f}") -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    mean_txt = value_fmt.format(mean)
    if stdev is None or (isinstance(stdev, float) and math.isnan(stdev)):
        return mean_txt
    return f"{mean_txt} ± {value_fmt.format(stdev)}"


def format_value(value: Optional[float], value_fmt: str = "{:.2f}") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return value_fmt.format(value)


def pick_first_available(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None and math.isfinite(value):
            return value
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    epoch: int
    total_epochs: int
    phase: str
    lr: Optional[float]
    throughput_per_gpu: Optional[float]
    throughput_global: Optional[float]
    val_auc: Optional[float]


@dataclass
class RunMetrics:
    log: Path
    gpus: int
    elapsed_s: Optional[float]
    avg_throughput_img_s: Optional[float]
    final_valid_auc: Optional[float]
    final_summary_auc: Optional[float]
    best_val_auc: Optional[float]
    best_val_epoch: Optional[int]
    best_val_phase: Optional[str]
    lr_min: Optional[float]
    lr_max: Optional[float]
    lr_mean: Optional[float]
    throughput_percentiles: Dict[str, float]
    phase_mean_throughput: Dict[str, float]
    phase_epoch_counts: Dict[str, int]
    epoch_records: List[EpochRecord] = field(default_factory=list)
    throughput_global_values: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log(path: Path, gpus: int) -> RunMetrics:
    throughput_global_values: List[float] = []
    lr_values: List[float] = []
    phase_throughputs: Dict[str, List[float]] = defaultdict(list)
    phase_epoch_counts: Counter[str] = Counter()
    epoch_records: List[EpochRecord] = []

    elapsed: Optional[float] = None
    final_valid_auc: Optional[float] = None
    final_summary_auc: Optional[float] = None
    best_val_auc: Optional[float] = None
    best_val_epoch: Optional[int] = None
    best_val_phase: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                phase = epoch_match.group(3).strip().rstrip(":").lower()
                phase_epoch_counts[phase] += 1

                thr_match = THROUGHPUT_RE.search(line)
                per_gpu_thr = float(thr_match.group(1)) if thr_match else None
                global_thr = per_gpu_thr * gpus if per_gpu_thr is not None else None
                if global_thr is not None:
                    throughput_global_values.append(global_thr)
                    phase_throughputs[phase].append(global_thr)

                lr_match = LR_RE.search(line)
                lr_val = float(lr_match.group(1)) if lr_match else None
                if lr_val is not None:
                    lr_values.append(lr_val)

                val_auc_match = VALAUC_RE.search(line)
                val_auc = float(val_auc_match.group(1)) if val_auc_match else None
                if val_auc is not None and (best_val_auc is None or val_auc > best_val_auc):
                    best_val_auc = val_auc
                    best_val_epoch = epoch
                    best_val_phase = phase

                epoch_records.append(
                    EpochRecord(
                        epoch=epoch,
                        total_epochs=total_epochs,
                        phase=phase,
                        lr=lr_val,
                        throughput_per_gpu=per_gpu_thr,
                        throughput_global=global_thr,
                        val_auc=val_auc,
                    )
                )
                continue

            if "elapsed=" in line:
                elapsed_match = ELAPSED_RE.search(line)
                if elapsed_match:
                    elapsed = float(elapsed_match.group(1))

            valid_match = VALID_AUC_RE.search(line)
            if valid_match:
                final_valid_auc = float(valid_match.group(1))

            final_match = FINAL_AUC_RE.search(line)
            if final_match:
                final_summary_auc = float(final_match.group(1))

    avg_throughput = safe_mean(throughput_global_values)

    throughput_percentiles = {
        "p50": percentile(throughput_global_values, 50),
        "p90": percentile(throughput_global_values, 90),
        "p95": percentile(throughput_global_values, 95),
    }
    phase_mean_throughput = {
        phase: safe_mean(values) for phase, values in phase_throughputs.items()
    }

    lr_min = min(lr_values) if lr_values else math.nan
    lr_max = max(lr_values) if lr_values else math.nan
    lr_mean = statistics.mean(lr_values) if lr_values else math.nan

    return RunMetrics(
        log=path,
        gpus=gpus,
        elapsed_s=elapsed,
        avg_throughput_img_s=avg_throughput,
        final_valid_auc=final_valid_auc,
        final_summary_auc=final_summary_auc,
        best_val_auc=best_val_auc,
        best_val_epoch=best_val_epoch,
        best_val_phase=best_val_phase,
        lr_min=lr_min,
        lr_max=lr_max,
        lr_mean=lr_mean,
        throughput_percentiles=throughput_percentiles,
        phase_mean_throughput=phase_mean_throughput,
        phase_epoch_counts=dict(phase_epoch_counts),
        epoch_records=epoch_records,
        throughput_global_values=throughput_global_values,
    )


def aggregate_runs(logs_dir: Path) -> Dict[int, List[RunMetrics]]:
    aggregates: Dict[int, List[RunMetrics]] = defaultdict(list)
    for log_path in sorted(logs_dir.glob("run_*gpu_*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if not match:
            continue
        gpus = int(match.group(1))
        aggregates[gpus].append(parse_log(log_path, gpus))
    return dict(aggregates)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def build_comparisons(aggregates: Dict[int, List[RunMetrics]]) -> List[Dict[str, float]]:
    comparisons: List[Dict[str, float]] = []
    if len(aggregates) < 2:
        return comparisons

    sorted_gpus = sorted(aggregates)
    base_gpus = sorted_gpus[0]
    base_runs = aggregates[base_gpus]
    base_time = safe_mean([run.elapsed_s for run in base_runs])
    base_thr = safe_mean([run.avg_throughput_img_s for run in base_runs])

    for target_gpus in sorted_gpus[1:]:
        runs = aggregates[target_gpus]
        target_time = safe_mean([run.elapsed_s for run in runs])
        target_thr = safe_mean([run.avg_throughput_img_s for run in runs])

        speedup_time = (
            base_time / target_time
            if math.isfinite(base_time)
            and math.isfinite(target_time)
            and target_time > 0
            else math.nan
        )
        speedup_thr = (
            target_thr / base_thr
            if math.isfinite(base_thr)
            and math.isfinite(target_thr)
            and base_thr > 0
            else math.nan
        )
        efficiency = (
            speedup_time / (target_gpus / base_gpus)
            if math.isfinite(speedup_time) and target_gpus > 0
            else math.nan
        )
        comparisons.append(
            {
                "base_gpus": base_gpus,
                "target_gpus": target_gpus,
                "speedup_time": speedup_time,
                "speedup_throughput": speedup_thr,
                "scalability_efficiency": efficiency,
            }
        )

    return comparisons


def gather_numeric(runs: Sequence[RunMetrics], attr: str) -> List[float]:
    values: List[float] = []
    for run in runs:
        value = getattr(run, attr)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def gather_from_dict(
    runs: Sequence[RunMetrics], attr: str, key: str
) -> List[float]:
    values: List[float] = []
    for run in runs:
        container = getattr(run, attr)
        if container is None:
            continue
        value = container.get(key)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def gather_percentile_values(runs: Sequence[RunMetrics], key: str) -> List[float]:
    return gather_from_dict(runs, "throughput_percentiles", key)


def gather_phase_values(
    runs: Sequence[RunMetrics], attr: str, phase: str
) -> List[float]:
    return gather_from_dict(runs, attr, phase)


def gather_epoch_counts(runs: Sequence[RunMetrics], phase: str) -> List[float]:
    values: List[float] = []
    for run in runs:
        count = run.phase_epoch_counts.get(phase)
        if count is not None:
            values.append(float(count))
    return values


def run_eval_auc(run: RunMetrics) -> Optional[float]:
    return pick_first_available(run.final_valid_auc, run.final_summary_auc, run.best_val_auc)


def format_summary(aggregates: Dict[int, List[RunMetrics]], comparisons: Iterable[Dict[str, float]]) -> str:
    lines: List[str] = []
    lines.append("=== Performance Summary ===")
    lines.append("")
    header = (
        f"{'GPUs':>4} | {'Runs':>4} | {'Time (s)':>21} | "
        f"{'Throughput (img/s)':>24} | {'Eval AUC':>15}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for gpus in sorted(aggregates):
        runs = aggregates[gpus]
        time_mean, time_std = describe(gather_numeric(runs, "elapsed_s"))
        thr_mean, thr_std = describe(gather_numeric(runs, "avg_throughput_img_s"))
        auc_mean, auc_std = describe([run_eval_auc(run) for run in runs])

        lines.append(
            f"{gpus:>4} | {len(runs):>4} | "
            f"{format_stat(time_mean, time_std, '{:.2f}'):>21} | "
            f"{format_stat(thr_mean, thr_std, '{:.1f}'):>24} | "
            f"{format_stat(auc_mean, auc_std, '{:.4f}'):>15}"
        )

    if comparisons:
        lines.append("")
        lines.append("=== Scalability ===")
        lines.append("")
        for comp in comparisons:
            base = comp["base_gpus"]
            target = comp["target_gpus"]
            lines.append(
                f"{base:>2}->{target:<2} GPUs | "
                f"speedup(time)={format_value(comp['speedup_time'], '{:.3f}')}x | "
                f"speedup(thr)={format_value(comp['speedup_throughput'], '{:.3f}')}x | "
                f"efficiency={format_value(comp['scalability_efficiency'], '{:.3f}')}"
            )

    lines.append("")
    lines.append("=== Additional Metrics ===")
    lines.append("")

    for gpus in sorted(aggregates):
        runs = aggregates[gpus]
        lines.append(f"{gpus} GPU(s):")

        # Best validation AUC and epoch
        best_auc_mean, best_auc_std = describe(gather_numeric(runs, "best_val_auc"))
        best_epoch_mean, best_epoch_std = describe(gather_numeric(runs, "best_val_epoch"))
        phase_counts = Counter(
            run.best_val_phase for run in runs if run.best_val_phase is not None
        )
        if phase_counts:
            phase_counts_text = ", ".join(f"{phase}={count}" for phase, count in sorted(phase_counts.items()))
        else:
            phase_counts_text = "n/a"

        lines.append(
            f"  • Best val AUC: {format_stat(best_auc_mean, best_auc_std, '{:.4f}')} "
            f"(epoch {format_stat(best_epoch_mean, best_epoch_std, '{:.1f}')})"
        )
        lines.append(f"    Phase of best AUC: {phase_counts_text}")

        # Learning rate statistics
        lr_min_mean, lr_min_std = describe(gather_numeric(runs, "lr_min"))
        lr_max_mean, lr_max_std = describe(gather_numeric(runs, "lr_max"))
        lr_mean_mean, lr_mean_std = describe(gather_numeric(runs, "lr_mean"))
        lines.append(
            "  • LR range:"
            f" min {format_stat(lr_min_mean, lr_min_std, '{:.2e}')}"
            f" | max {format_stat(lr_max_mean, lr_max_std, '{:.2e}')}"
            f" | mean {format_stat(lr_mean_mean, lr_mean_std, '{:.2e}')}"
        )

        # Throughput percentiles
        p50_mean, p50_std = describe(gather_percentile_values(runs, "p50"))
        p90_mean, p90_std = describe(gather_percentile_values(runs, "p90"))
        p95_mean, p95_std = describe(gather_percentile_values(runs, "p95"))
        lines.append(
            "  • Throughput percentiles (img/s): "
            f"P50 {format_stat(p50_mean, p50_std, '{:.1f}')}, "
            f"P90 {format_stat(p90_mean, p90_std, '{:.1f}')}, "
            f"P95 {format_stat(p95_mean, p95_std, '{:.1f}')}"
        )

        # Phase throughput and epoch counts
        phases = sorted({phase for run in runs for phase in run.phase_mean_throughput})
        if phases:
            phase_lines = []
            for phase in phases:
                thr_mean, thr_std = describe(gather_phase_values(runs, "phase_mean_throughput", phase))
                epoch_mean, epoch_std = describe(gather_epoch_counts(runs, phase))
                phase_lines.append(
                    f"{phase}: thr {format_stat(thr_mean, thr_std, '{:.1f}')} img/s,"
                    f" epochs {format_stat(epoch_mean, epoch_std, '{:.1f}')}"
                )
            lines.append("  • Phase stats: " + " | ".join(phase_lines))
        else:
            lines.append("  • Phase stats: n/a")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_json_ready(aggregates: Dict[int, List[RunMetrics]], comparisons: Iterable[Dict[str, float]]) -> Dict:
    result = {
        "per_gpu": {},
        "comparisons": list(comparisons),
    }

    for gpus, runs in aggregates.items():
        runs_payload = []
        for run in runs:
            runs_payload.append(
                {
                    "log": str(run.log),
                    "gpus": run.gpus,
                    "elapsed_s": run.elapsed_s,
                    "avg_throughput_img_s": run.avg_throughput_img_s,
                    "final_valid_auc": run.final_valid_auc,
                    "final_summary_auc": run.final_summary_auc,
                    "best_val_auc": run.best_val_auc,
                    "best_val_epoch": run.best_val_epoch,
                    "best_val_phase": run.best_val_phase,
                    "lr": {
                        "min": run.lr_min,
                        "max": run.lr_max,
                        "mean": run.lr_mean,
                    },
                    "throughput_percentiles": run.throughput_percentiles,
                    "phase_mean_throughput": run.phase_mean_throughput,
                    "phase_epoch_counts": run.phase_epoch_counts,
                }
            )

        phases = sorted({phase for run in runs for phase in run.phase_mean_throughput})
        percentiles_keys = sorted({key for run in runs for key in run.throughput_percentiles})

        stats_payload = {
            "elapsed": compute_stats(gather_numeric(runs, "elapsed_s")),
            "avg_throughput": compute_stats(gather_numeric(runs, "avg_throughput_img_s")),
            "evaluation_auc": compute_stats([run_eval_auc(run) for run in runs]),
            "best_val_auc": compute_stats(gather_numeric(runs, "best_val_auc")),
            "best_val_epoch": compute_stats(gather_numeric(runs, "best_val_epoch")),
            "lr": {
                "min": compute_stats(gather_numeric(runs, "lr_min")),
                "max": compute_stats(gather_numeric(runs, "lr_max")),
                "mean": compute_stats(gather_numeric(runs, "lr_mean")),
            },
            "throughput_percentiles": {
                key: compute_stats(gather_percentile_values(runs, key))
                for key in percentiles_keys
            },
            "phase_throughput": {
                phase: compute_stats(gather_phase_values(runs, "phase_mean_throughput", phase))
                for phase in phases
            },
            "phase_epoch_counts": {
                phase: compute_stats(gather_epoch_counts(runs, phase))
                for phase in phases
            },
        }

        result["per_gpu"][gpus] = {
            "runs": runs_payload,
            "stats": stats_payload,
        }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate training performance logs.")
    parser.add_argument(
        "logs_dir",
        type=Path,
        help="Directory containing run_<NGPU>gpu_<N>.log files (e.g., perf_results/logs_XXXXXX)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        help="Optional path to write human-readable summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir: Path = args.logs_dir
    if not logs_dir.is_dir():
        raise SystemExit(f"Logs directory '{logs_dir}' not found.")

    aggregates = aggregate_runs(logs_dir)
    if not aggregates:
        raise SystemExit(f"No run_*gpu_*.log files found inside '{logs_dir}'.")

    comparisons = build_comparisons(aggregates)
    summary_text = format_summary(aggregates, comparisons)

    print(summary_text, end="")

    if args.output_text:
        args.output_text.write_text(summary_text, encoding="utf-8")

    if args.output_json:
        summary_json = to_json_ready(aggregates, comparisons)
        args.output_json.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

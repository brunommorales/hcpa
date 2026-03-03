#!/usr/bin/env python3
"""
Analyze results produced by distributed_run.slurm runs.

The analyzer scans results/ for directories named like:
  result<JOBID>_<partition>_gpu<NGPUS>/run_<i>/
and reads the CSV written by dr_hcpa_v2_2024.py inside each run_*.

It computes per-GPU and global throughput, total elapsed time, and summarizes
speedup and scalability efficiency comparing against the smallest GPU count
found as baseline.

Usage examples:
  python projects/hcpa/pytorch_opt/analyze_distributed_results.py
  python projects/hcpa/pytorch_opt/analyze_distributed_results.py --by-partition
  python projects/hcpa/pytorch_opt/analyze_distributed_results.py --root results --save results/dist_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


RESULT_DIR_RE = re.compile(r"^(?:([0-9]+)x_)?([a-z0-9._-]+)\+([a-z0-9._-]+)$")


@dataclass
class RunRecord:
    dir: Path
    csv: Path
    partition: str
    cluster: Optional[str]
    batch: Optional[str]
    gpus: int
    elapsed_s: Optional[float]
    avg_thr_per_gpu: Optional[float]
    avg_thr_global: Optional[float]
    final_auc: Optional[float]
    peak_train_mem_alloc_mb: Optional[float]
    peak_train_mem_reserved_mb: Optional[float]
    peak_val_mem_alloc_mb: Optional[float]
    peak_val_mem_reserved_mb: Optional[float]


def safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def safe_mean(values: Sequence[Optional[float]]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return math.nan
    return statistics.mean(finite)


def safe_stdev(values: Sequence[Optional[float]]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(finite) <= 1:
        return 0.0 if len(finite) == 1 else math.nan
    return statistics.stdev(finite)


def describe(values: Sequence[Optional[float]]) -> Tuple[float, float]:
    return safe_mean(values), safe_stdev(values)


def parse_manifest(base_dir: Path) -> Dict[str, str]:
    mpath = base_dir / "env_manifest.txt"
    data: Dict[str, str] = {}
    if not mpath.exists():
        return data
    try:
        for line in mpath.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    except Exception:
        pass
    return data


def pick_first_csv(run_dir: Path) -> Optional[Path]:
    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        # sometimes results can be nested, try one level deeper
        for p in run_dir.iterdir():
            if p.is_dir():
                inner = sorted(p.glob("*.csv"))
                if inner:
                    return inner[0]
        return None
    return csvs[0]


def parse_run(run_dir: Path, gpus: int, partition: str, cluster: Optional[str], batch: Optional[str]) -> Optional[RunRecord]:
    csv_path = pick_first_csv(run_dir)
    if not csv_path:
        return None
    thr_values: List[float] = []  # per-GPU throughput values from CSV
    train_mem_alloc_values: List[float] = []
    train_mem_reserved_values: List[float] = []
    val_mem_alloc_values: List[float] = []
    val_mem_reserved_values: List[float] = []
    final_auc: Optional[float] = None
    final_elapsed: Optional[float] = None

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage = (row.get("stage") or "").strip().lower()
                # collect training throughput per GPU across epochs
                if stage in ("freeze", "finetune"):
                    v = safe_float(row.get("train_throughput_img_s"))
                    if v is not None:
                        thr_values.append(v)
                    mem_alloc = safe_float(row.get("train_gpu_mem_alloc_mb"))
                    if mem_alloc is not None:
                        train_mem_alloc_values.append(mem_alloc)
                    mem_reserved = safe_float(row.get("train_gpu_mem_reserved_mb"))
                    if mem_reserved is not None:
                        train_mem_reserved_values.append(mem_reserved)
                val_mem_alloc = safe_float(row.get("val_gpu_mem_alloc_mb"))
                if val_mem_alloc is not None:
                    val_mem_alloc_values.append(val_mem_alloc)
                val_mem_reserved = safe_float(row.get("val_gpu_mem_reserved_mb"))
                if val_mem_reserved is not None:
                    val_mem_reserved_values.append(val_mem_reserved)
                # final eval row carries total elapsed
                if stage == "final_eval":
                    final_elapsed = safe_float(row.get("val_elapsed_s"))
                    final_auc = safe_float(row.get("val_auc"))
    except Exception:
        return None

    avg_thr_per_gpu = statistics.mean(thr_values) if thr_values else None
    avg_thr_global = (avg_thr_per_gpu * gpus) if avg_thr_per_gpu is not None else None
    peak_train_mem_alloc = max(train_mem_alloc_values) if train_mem_alloc_values else None
    peak_train_mem_reserved = max(train_mem_reserved_values) if train_mem_reserved_values else None
    peak_val_mem_alloc = max(val_mem_alloc_values) if val_mem_alloc_values else None
    peak_val_mem_reserved = max(val_mem_reserved_values) if val_mem_reserved_values else None

    return RunRecord(
        dir=run_dir,
        csv=csv_path,
        partition=partition,
        cluster=cluster,
        batch=batch,
        gpus=gpus,
        elapsed_s=final_elapsed,
        avg_thr_per_gpu=avg_thr_per_gpu,
        avg_thr_global=avg_thr_global,
        final_auc=final_auc,
        peak_train_mem_alloc_mb=peak_train_mem_alloc,
        peak_train_mem_reserved_mb=peak_train_mem_reserved,
        peak_val_mem_alloc_mb=peak_val_mem_alloc,
        peak_val_mem_reserved_mb=peak_val_mem_reserved,
    )


def scan_results(root: Path) -> List[RunRecord]:
    records: List[RunRecord] = []

    for base in sorted(root.iterdir()):
        if not base.is_dir():
            continue
        m = RESULT_DIR_RE.match(base.name)
        if not m:
            continue

        prefix_gpus, gpu_tag, job_id = m.groups()
        manifest = parse_manifest(base)
        cluster = manifest.get("cluster") or manifest.get("cluster_name")
        partition = manifest.get("partition") or manifest.get("partition_tag") or "unknown"

        gpus: Optional[int] = None
        if prefix_gpus:
            try:
                gpus = int(prefix_gpus)
            except Exception:
                gpus = None
        if gpus is None:
            try:
                gpus = int(manifest.get("world_size", "") or manifest.get("total_gpus", ""))
            except Exception:
                gpus = None
        if gpus is None:
            gpus = 1

        batch = manifest.get("global_batch_size") or None
        run_dirs = sorted(base.glob("runs_*"))
        if not run_dirs:
            run_dirs = sorted(base.glob("run_*"))  # fallback legado

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            rec = parse_run(run_dir, gpus=gpus, partition=partition, cluster=cluster, batch=batch)
            if rec is not None:
                records.append(rec)

    return records


def group_by_gpu(records: Sequence[RunRecord]) -> Dict[int, List[RunRecord]]:
    out: Dict[int, List[RunRecord]] = {}
    for r in records:
        out.setdefault(r.gpus, []).append(r)
    return out


def group_by_partition_gpu(records: Sequence[RunRecord]) -> Dict[Tuple[str, int], List[RunRecord]]:
    out: Dict[Tuple[str, int], List[RunRecord]] = {}
    for r in records:
        key = (r.partition or "unknown", r.gpus)
        out.setdefault(key, []).append(r)
    return out


def summarize_groups(groups: Dict[int, List[RunRecord]]) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}
    for g, runs in sorted(groups.items()):
        times = [r.elapsed_s for r in runs]
        thr = [r.avg_thr_global for r in runs]
        train_mem_alloc = [r.peak_train_mem_alloc_mb for r in runs]
        train_mem_reserved = [r.peak_train_mem_reserved_mb for r in runs]
        val_mem_alloc = [r.peak_val_mem_alloc_mb for r in runs]
        val_mem_reserved = [r.peak_val_mem_reserved_mb for r in runs]
        t_mean, t_std = describe(times)
        th_mean, th_std = describe(thr)
        tr_mem_alloc_mean, tr_mem_alloc_std = describe(train_mem_alloc)
        tr_mem_res_mean, tr_mem_res_std = describe(train_mem_reserved)
        val_mem_alloc_mean, val_mem_alloc_std = describe(val_mem_alloc)
        val_mem_res_mean, val_mem_res_std = describe(val_mem_reserved)
        summary[g] = {
            "runs": len(runs),
            "elapsed_mean": t_mean,
            "elapsed_stdev": t_std,
            "throughput_mean": th_mean,
            "throughput_stdev": th_std,
            "train_mem_alloc_mean": tr_mem_alloc_mean,
            "train_mem_alloc_stdev": tr_mem_alloc_std,
            "train_mem_reserved_mean": tr_mem_res_mean,
            "train_mem_reserved_stdev": tr_mem_res_std,
            "val_mem_alloc_mean": val_mem_alloc_mean,
            "val_mem_alloc_stdev": val_mem_alloc_std,
            "val_mem_reserved_mean": val_mem_res_mean,
            "val_mem_reserved_stdev": val_mem_res_std,
        }
    return summary


def compute_scalability(groups: Dict[int, Dict[str, float]]) -> List[Dict[str, float]]:
    if not groups:
        return []
    gpu_counts = sorted(groups)
    base = gpu_counts[0]
    base_time = groups[base]["elapsed_mean"]
    base_thr = groups[base]["throughput_mean"]
    comps: List[Dict[str, float]] = []
    for g in gpu_counts[1:]:
        t_mean = groups[g]["elapsed_mean"]
        th_mean = groups[g]["throughput_mean"]
        speed_time = (
            base_time / t_mean
            if math.isfinite(base_time) and math.isfinite(t_mean) and t_mean > 0
            else math.nan
        )
        speed_thr = (
            th_mean / base_thr
            if math.isfinite(base_thr) and math.isfinite(th_mean) and base_thr > 0
            else math.nan
        )
        efficiency = (
            speed_time / (g / base)
            if math.isfinite(speed_time) and g > 0
            else math.nan
        )
        comps.append(
            {
                "base_gpus": base,
                "target_gpus": g,
                "speedup_time": speed_time,
                "speedup_throughput": speed_thr,
                "scalability_efficiency": efficiency,
            }
        )
    return comps


def format_stat(mean: float, stdev: float, fmt: str) -> str:
    if not math.isfinite(mean):
        return "n/a"
    if not math.isfinite(stdev):
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(stdev)}"


def build_text_summary(groups: Dict[int, Dict[str, float]], comps: Iterable[Dict[str, float]]) -> str:
    lines: List[str] = []
    lines.append("=== Distributed Performance Summary ===")
    lines.append("")
    hdr = f"{'GPUs':>4} | {'Runs':>4} | {'Time (s)':>21} | {'Throughput (img/s)':>24}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for g in sorted(groups):
        s = groups[g]
        lines.append(
            f"{g:>4} | {s['runs']:>4} | "
            f"{format_stat(s['elapsed_mean'], s['elapsed_stdev'], '{:.2f}'):>21} | "
            f"{format_stat(s['throughput_mean'], s['throughput_stdev'], '{:.1f}'):>24}"
        )
    comps = list(comps)
    if comps:
        lines.append("")
        lines.append("=== Scalability ===")
        for c in comps:
            lines.append(
                f"{c['base_gpus']:>2}->{c['target_gpus']:<2} GPUs | "
                f"speedup(time)={c['speedup_time']:.3f}x | "
                f"speedup(thr)={c['speedup_throughput']:.3f}x | "
                f"efficiency={c['scalability_efficiency']:.3f}"
            )
    lines.append("")
    lines.append("=== GPU Memory (peak MB) ===")
    lines.append("")
    mem_hdr = (
        f"{'GPUs':>4} | {'Train alloc':>18} | {'Train reserved':>18} | "
        f"{'Val alloc':>18} | {'Val reserved':>18}"
    )
    lines.append(mem_hdr)
    lines.append("-" * len(mem_hdr))
    for g in sorted(groups):
        s = groups[g]
        lines.append(
            f"{g:>4} | "
            f"{format_stat(s['train_mem_alloc_mean'], s['train_mem_alloc_stdev'], '{:.1f}'):>18} | "
            f"{format_stat(s['train_mem_reserved_mean'], s['train_mem_reserved_stdev'], '{:.1f}'):>18} | "
            f"{format_stat(s['val_mem_alloc_mean'], s['val_mem_alloc_stdev'], '{:.1f}'):>18} | "
            f"{format_stat(s['val_mem_reserved_mean'], s['val_mem_reserved_stdev'], '{:.1f}'):>18}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze distributed training results")
    ap.add_argument("--root", default="results", help="Root results directory")
    ap.add_argument("--by-partition", action="store_true", help="Report per partition as well")
    ap.add_argument("--save", default=None, help="Path to save JSON summary")
    ap.add_argument("--text-out", default=None, help="Path to save text summary")
    args = ap.parse_args()

    root = Path(args.root)
    records = scan_results(root)
    if not records:
        print("No runs found under", root)
        return

    by_gpu = group_by_gpu(records)
    groups = summarize_groups(by_gpu)
    comps = compute_scalability(groups)

    text = build_text_summary(groups, comps)
    print(text)

    payload = {
        "groups": groups,
        "scalability": list(comps),
        "runs": [
            {
                "dir": str(r.dir),
                "csv": str(r.csv),
                "partition": r.partition,
                "cluster": r.cluster,
                "batch": r.batch,
                "gpus": r.gpus,
                "elapsed_s": r.elapsed_s,
                "avg_throughput_per_gpu": r.avg_thr_per_gpu,
                "avg_throughput_global": r.avg_thr_global,
                "final_auc": r.final_auc,
                "peak_train_mem_alloc_mb": r.peak_train_mem_alloc_mb,
                "peak_train_mem_reserved_mb": r.peak_train_mem_reserved_mb,
                "peak_val_mem_alloc_mb": r.peak_val_mem_alloc_mb,
                "peak_val_mem_reserved_mb": r.peak_val_mem_reserved_mb,
            }
            for r in records
        ],
    }

    if args.by_partition:
        part_groups: Dict[str, Dict[int, Dict[str, float]]] = {}
        for (part, g), runs in group_by_partition_gpu(records).items():
            part_groups.setdefault(part, {})[g] = summarize_groups({g: runs})[g]
        payload["by_partition"] = part_groups

    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        # default JSON alongside root
        out_json = root / "distributed_summary.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.text_out:
        Path(args.text_out).write_text(text, encoding="utf-8")
    else:
        out_txt = root / "distributed_summary.txt"
        out_txt.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

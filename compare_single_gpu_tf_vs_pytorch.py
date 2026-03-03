#!/usr/bin/env python3
"""
Aggregate TensorFlow x PyTorch runs (original and clean variants) and generate
comparison CSVs with average AUC, throughput (img/s) and training time.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

TARGET_BATCHES_SINGLE = (96, 128, 160)
TARGET_BATCHES_MULTI = (96, 192, 224)
MAX_EPOCH_SANE = 100_000  # ignore corrupted epoch values above this
TARGET_AUC = 0.95
MAX_MEM_MB = 80_000  # memória acima disso é tratada como inválida (valores corrompidos)
MAX_MEM_MB_LOG = 500_000  # margem extra para parsing de logs
VARIANT_PROJECTS = {
    "original": {
        "tensorflow": "tensorflow_opt",
        "pytorch": "pytorch_opt",
    },
    "clean": {
        "tensorflow": "tensorflow_base",
        "pytorch": "pytorch_base",
    },
}
CSV_FIELDNAMES = [
    "framework",
    "variant",
    "gpus",
    "batch_size",
    "runs",
    "mean_auc",
    "std_auc",
    "mean_throughput_img_s",
    "std_throughput_img_s",
    "mean_train_time_s",
    "std_train_time_s",
    "mean_peak_gpu_mem_mb",
    "std_peak_gpu_mem_mb",
]
CSV_FIELDNAMES_EXTENDED = CSV_FIELDNAMES + [
    "mean_specificity",
    "std_specificity",
    "mean_sensitivity",
    "std_sensitivity",
]
GPU2_BS96_RUN_FIELDS = [
    "pair",
    "project",
    "result_dir",
    "framework",
    "variant",
    "gpus",
    "batch_size",
    "run_id",
    "epochs",
    "val_auc_final",
    "val_auc_best",
    "train_time_s",
    "throughput_img_s",
    "peak_gpu_mem_mb",
]
GPU2_BS96_SUMMARY_FIELDS = [
    "pair",
    "project",
    "result_dir",
    "framework",
    "variant",
    "gpus",
    "batch_size",
    "runs",
    "mean_val_auc_final",
    "std_val_auc_final",
    "mean_val_auc_best",
    "std_val_auc_best",
    "mean_throughput_img_s",
    "std_throughput_img_s",
    "mean_train_time_s",
    "std_train_time_s",
    "mean_peak_gpu_mem_mb",
    "std_peak_gpu_mem_mb",
]
BATCH96_RUN_FIELDS = [
    "project",
    "framework",
    "variant",
    "gpus",
    "batch_size",
    "run_id",
    "epochs",
    "val_auc_final",
    "val_auc_best",
    "train_time_s",
    "throughput_img_s",
    "peak_gpu_mem_mb",
    "time_to_auc_0_95_s",
]
OUTPUT_FIELDS_BATCH96_GPU2 = [
    "project",
    "framework",
    "gpus",
    "batch_size",
    "runs",
    "mean_auc",
    "std_auc",
    "mean_throughput_img_s",
    "std_throughput_img_s",
    "mean_train_time_s",
    "std_train_time_s",
    "mean_peak_gpu_mem_mb",
    "std_peak_gpu_mem_mb",
]

SINGLE_GPU_RUN_FIELDS = [
    "project",
    "framework",
    "variant",
    "result_dir",
    "gpus",
    "batch_size",
    "run_id",
    "epochs",
    "val_auc_final",
    "val_auc_best",
    "val_spec_final",
    "val_spec_best",
    "val_sens_final",
    "val_sens_best",
    "train_time_s",
    "throughput_img_s",
    "peak_gpu_mem_mb",
    "time_to_auc_0_95_s",
]

SINGLE_GPU_SUMMARY_FIELDS = [
    "project",
    "framework",
    "variant",
    "gpus",
    "batch_size",
    "runs",
    "mean_auc",
    "std_auc",
    "mean_specificity",
    "std_specificity",
    "mean_sensitivity",
    "std_sensitivity",
    "mean_throughput_img_s",
    "std_throughput_img_s",
    "mean_train_time_s",
    "std_train_time_s",
    "mean_peak_gpu_mem_mb",
    "std_peak_gpu_mem_mb",
]


@dataclass
class RunMetrics:
    run_id: int
    auc: Optional[float]
    throughput_img_s: Optional[float]
    train_time_s: Optional[float]
    gpu_mem_mb: Optional[float] = None
    auc_best: Optional[float] = None
    spec: Optional[float] = None
    spec_best: Optional[float] = None
    sens: Optional[float] = None
    sens_best: Optional[float] = None
    epochs: Optional[int] = None
    time_to_target_auc_s: Optional[float] = None


SummaryKey = Tuple[str, str, int, int]
RESULT_DIR_RE = re.compile(r"^result(\d+)_([a-z0-9._-]+)_gpu(\d+)_bs(\d+)$")


def parse_manifest(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def safe_int(value: Optional[str]) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value.strip())
    except Exception:
        return None


def extract_result_dir_info(result_dir: Path) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "job_id": None,
        "partition": None,
        "gpus": None,
        "batch_size": None,
    }
    match = RESULT_DIR_RE.match(result_dir.name)
    if match:
        info["job_id"] = match.group(1)
        info["partition"] = match.group(2)
        info["gpus"] = match.group(3)
        info["batch_size"] = match.group(4)
    return info


def parse_result_metadata(result_dir: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    manifest = parse_manifest(result_dir / "env_manifest.txt")
    info = extract_result_dir_info(result_dir)
    gpus = safe_int(manifest.get("world_size")) or safe_int(info.get("gpus"))
    batch_size = safe_int(manifest.get("global_batch_size")) or safe_int(info.get("batch_size"))
    partition = manifest.get("partition") or info.get("partition")
    return gpus, batch_size, partition


def extract_job_id(result_dir: Path) -> int:
    manifest = parse_manifest(result_dir / "env_manifest.txt")
    job_id = safe_int(manifest.get("job_id"))
    if job_id is not None:
        return job_id
    info = extract_result_dir_info(result_dir)
    name_job = safe_int(info.get("job_id"))
    return name_job or 0


def extract_peak_mem_from_logs(result_dir: Path) -> Optional[float]:
    """
    Procura pico de memória em logs (nvidia-smi) associados ao job.
    """
    job_id = extract_job_id(result_dir)
    if not job_id:
        return None
    logs_root = result_dir.parents[1] / "logs"
    log_paths = list(logs_root.glob(f"*{job_id}*.out"))
    if not log_paths:
        return None
    numbers: List[int] = []
    for p in log_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in re.finditer(r"([0-9]+)MiB", text):
            val = int(match.group(1))
            if val <= 0:
                continue
            if val >= MAX_MEM_MB_LOG:
                continue
            # descarta valores que são claramente o total da placa (ex.: 46068MiB)
            if val >= 45000 and val <= 48000:
                continue
            numbers.append(val)
    if not numbers:
        return None
    return float(max(numbers))


def extract_train_throughput_from_csv(csv_path: Path) -> Optional[float]:
    """
    Usa a média da coluna train_throughput_img_s, se existir.
    """
    values: List[float] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = _safe_float(_get_row_value_case_insensitive(row, "train_throughput_img_s"))
                if val is not None and val > 0:
                    values.append(val)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not values:
        return None
    return float(sum(values) / len(values))


def extract_throughput_from_logs(result_dir: Path) -> Optional[float]:
    """
    Extrai média de throughput (throughput_img_s) dos logs do job.
    """
    job_id = extract_job_id(result_dir)
    if not job_id:
        return None
    logs_root = result_dir.parents[1] / "logs"
    log_paths = list(logs_root.glob(f"*{job_id}*.out"))
    if not log_paths:
        return None
    values: List[float] = []
    for p in log_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in re.finditer(r"throughput_img_s:\s*([0-9]+\.?[0-9]*)", text):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
    if not values:
        return None
    return float(sum(values) / len(values))


def find_result_dirs(results_root: Path, gpu_count: int, target_batches: Iterable[int]) -> Dict[int, Path]:
    """Pick the most recent (largest job_id) runs per batch for the given GPU count."""
    selected: Dict[int, Tuple[int, Path]] = {}
    for base in sorted(results_root.glob(f"result*_*_gpu{gpu_count}_bs*")):
        manifest = parse_manifest(base / "env_manifest.txt")
        world_size = safe_int(manifest.get("world_size"))
        if world_size not in (None, gpu_count):
            continue
        batch = manifest.get("global_batch_size")
        if batch is None:
            m = re.search(r"_bs(\d+)", base.name)
            batch = m.group(1) if m else None
        batch_size = safe_int(batch)
        if batch_size not in target_batches:
            continue
        job_id = safe_int(manifest.get("job_id"))
        if job_id is None:
            m = re.match(r"result(\d+)_", base.name)
            if m:
                job_id = int(m.group(1))
        if job_id is None:
            job_id = 0
        prev = selected.get(batch_size)
        if prev is None or job_id > prev[0]:
            selected[batch_size] = (job_id, base)
    return {batch: path for batch, (_, path) in selected.items()}


def find_result_dirs_exact(results_root: Path, gpu_count: int, batch_size: int) -> List[Path]:
    """Return all result directories matching the exact GPU count and batch size."""
    matches: List[Path] = []
    for base in sorted(results_root.glob(f"result*_*_bs{batch_size}")):
        if not base.is_dir():
            continue
        manifest = parse_manifest(base / "env_manifest.txt")
        world_size = safe_int(manifest.get("world_size"))
        info = extract_result_dir_info(base)
        name_gpus = safe_int(info.get("gpus"))
        # Admite casos em que a pasta usa gpu0 para 1 GPU.
        gpus_ok = False
        for candidate in (world_size, name_gpus):
            if candidate is None:
                continue
            if candidate == gpu_count:
                gpus_ok = True
                break
            if gpu_count == 1 and candidate == 0:
                gpus_ok = True
                break
        if not gpus_ok:
            continue

        manifest_batch = safe_int(manifest.get("global_batch_size"))
        if manifest_batch not in (None, batch_size, batch_size * (world_size or gpu_count)):
            # Ainda aceita se o nome da pasta carrega o batch correto.
            name_batch = safe_int(info.get("batch_size"))
            if name_batch not in (batch_size, batch_size * (world_size or gpu_count)):
                continue
        matches.append(base)
    return matches


def select_preferred_result_dir(
    results_root: Path,
    gpu_count: int,
    batch_size: int,
    prefer_partition: Optional[str] = "grace",
) -> Optional[Path]:
    matches = find_result_dirs_exact(results_root, gpu_count=gpu_count, batch_size=batch_size)
    if not matches:
        return None
    latest_overall = max(matches, key=extract_job_id)
    if prefer_partition:
        preferred = [m for m in matches if f"_{prefer_partition}_" in m.name]
        if preferred:
            latest_preferred = max(preferred, key=extract_job_id)
            # If the preferred partition is also the newest, keep it; otherwise
            # fall back to the most recent run regardless of partition.
            if extract_job_id(latest_preferred) >= extract_job_id(latest_overall):
                return latest_preferred
    return latest_overall


def count_train_images(data_dir: Path) -> int:
    tfrec_glob = sorted(data_dir.glob("train*.tfrec"))
    pattern = re.compile(r"-([0-9]+)\.")
    total = 0
    for tfrec in tfrec_glob:
        match = pattern.search(tfrec.name)
        if match:
            total += int(match.group(1))
    if total <= 0:
        raise RuntimeError(f"Nenhum train*.tfrec encontrado em {data_dir}")
    return total


def _read_csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            return []
        return [h.strip().lower() for h in header]
    except Exception:
        return []


def pick_first_csv(run_dir: Path) -> Optional[Path]:
    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        return None
    for candidate in csvs:
        header = _read_csv_header(candidate)
        if "epoch" in header:
            return candidate
    return csvs[0]


def infer_epochs_from_csv(csv_path: Path) -> int:
    max_epoch = None
    row_count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            stage = (row.get("stage") or "").strip().lower()
            if stage == "final_eval":
                continue
            epoch = row.get("epoch")
            if epoch is None or not epoch.strip():
                continue
            try:
                value = int(float(epoch))
            except Exception:
                continue
            if value > MAX_EPOCH_SANE:
                continue
            max_epoch = value if max_epoch is None else max(max_epoch, value)
    if max_epoch is None:
        raise RuntimeError(f"Não foi possível inferir epochs a partir de {csv_path}")
    epochs = max_epoch + 1
    if row_count > 0:
        epochs = min(epochs, row_count)
    return epochs


def time_to_target_auc(csv_path: Path, target_auc: float, total_time_s: Optional[float], epochs: Optional[int]) -> Optional[float]:
    """
    Calcula o tempo até atingir a AUC alvo. Usa, se disponível, colunas de tempo por epoch
    (train_elapsed_s/val_elapsed_s ou epoch_time_sec). Caso não exista tempo por epoch,
    aproxima usando tempo total dividido pelo número de épocas.
    """
    per_epoch_times: List[float] = []
    auc_hits_epoch: Optional[int] = None
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                epoch = _safe_float(row.get("epoch"))
                val_auc = _safe_float(row.get("val_auc")) or _safe_float(row.get("val_AUC"))
                train_elapsed = _safe_float(row.get("train_elapsed_s"))
                val_elapsed = _safe_float(row.get("val_elapsed_s"))
                epoch_time = _safe_float(row.get("epoch_time_sec"))
                if epoch_time is not None:
                    per_epoch_times.append(epoch_time)
                else:
                    elapsed_sum = 0.0
                    if train_elapsed is not None:
                        elapsed_sum += train_elapsed
                    if val_elapsed is not None:
                        elapsed_sum += val_elapsed
                    if elapsed_sum > 0:
                        per_epoch_times.append(elapsed_sum)
                if val_auc is not None and val_auc >= target_auc and auc_hits_epoch is None:
                    # epoch numerada a partir de zero
                    auc_hits_epoch = int(epoch) if epoch is not None else idx
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if auc_hits_epoch is None:
        return None

    # Caso tenhamos tempos por época, somamos até o epoch alvo (inclusive)
    if per_epoch_times and len(per_epoch_times) > auc_hits_epoch:
        return float(sum(per_epoch_times[: auc_hits_epoch + 1]))

    # Fallback: usa tempo total / epochs
    if total_time_s is not None and epochs:
        mean_epoch_time = total_time_s / max(epochs, 1)
        return mean_epoch_time * (auc_hits_epoch + 1)
    return None


def _list_run_dirs(result_dir: Path) -> List[Path]:
    runs = sorted([p for p in result_dir.glob("runs_*") if p.is_dir()])
    if runs:
        return runs
    return sorted([p for p in result_dir.glob("run_*") if p.is_dir()])


def _run_dir_for_id(result_dir: Path, run_id: int) -> Path:
    candidate = result_dir / f"runs_{run_id}"
    if candidate.is_dir() or candidate.parent.exists():
        return candidate
    return result_dir / f"run_{run_id}"


def existing_run_ids(result_dir: Path) -> List[int]:
    run_ids: List[int] = []
    for run_dir in _list_run_dirs(result_dir):
        name = run_dir.name
        if name.startswith("runs_"):
            suffix = name.split("_", 1)[-1]
        elif name.startswith("run_"):
            suffix = name.split("_", 1)[-1]
        else:
            suffix = name
        try:
            run_ids.append(int(suffix))
        except ValueError:
            continue
    return sorted(run_ids)


def _find_val_auc_key(fieldnames: Optional[Iterable[str]]) -> Optional[str]:
    if not fieldnames:
        return None
    for name in fieldnames:
        if name and name.strip().lower() == "val_auc":
            return name
    return None


def extract_final_val_auc(csv_path: Path) -> Optional[float]:
    last_val_auc: Optional[float] = None
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            key = _find_val_auc_key(reader.fieldnames)
            if key is None:
                return None
            for row in reader:
                val = _safe_float(row.get(key))
                if val is not None:
                    last_val_auc = val
    except Exception:
        return None
    return last_val_auc


def parse_tensorflow_log_text(log_text: str) -> Dict[int, Tuple[float, float]]:
    results: Dict[int, Tuple[float, float]] = {}
    for line in log_text.splitlines():
        if "all," not in line:
            continue
        idx = line.find("all,")
        if idx == -1:
            continue
        cleaned = line[idx:].strip()
        parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
        if len(parts) < 4 or parts[0] != "all":
            continue
        run_id = safe_int(parts[1])
        if run_id is None:
            continue
        auc = _safe_float(parts[-2])
        elapsed = _safe_float(parts[-1])
        if auc is None or elapsed is None:
            auc = _safe_float(parts[2]) if len(parts) > 2 else None
            elapsed = _safe_float(parts[3]) if len(parts) > 3 else None
        if auc is None or elapsed is None:
            continue
        if run_id not in results:
            results[run_id] = (auc, elapsed)
    return results


def _get_row_value_case_insensitive(row: Dict[str, str], key_lower: str) -> Optional[str]:
    for key, value in row.items():
        if key and key.strip().lower() == key_lower:
            return value
    return None


def parse_val_metrics(
    csv_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Retorna (auc_final, auc_best, spec_final, spec_best, sens_final, sens_best)
    a partir das colunas val_auc, val_spec, val_sens (case-insensitive).
    """
    last_auc: Optional[float] = None
    best_auc: Optional[float] = None
    final_auc: Optional[float] = None

    last_spec: Optional[float] = None
    best_spec: Optional[float] = None
    final_spec: Optional[float] = None

    last_sens: Optional[float] = None
    best_sens: Optional[float] = None
    final_sens: Optional[float] = None

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage = (_get_row_value_case_insensitive(row, "stage") or "").strip().lower()

                val_auc = _safe_float(_get_row_value_case_insensitive(row, "val_auc"))
                if val_auc is not None and 0.0 <= val_auc <= 1.0:
                    last_auc = val_auc
                    best_auc = val_auc if best_auc is None else max(best_auc, val_auc)

                val_spec = _safe_float(_get_row_value_case_insensitive(row, "val_spec"))
                if val_spec is not None and 0.0 <= val_spec <= 1.0:
                    last_spec = val_spec
                    best_spec = val_spec if best_spec is None else max(best_spec, val_spec)

                val_sens = _safe_float(_get_row_value_case_insensitive(row, "val_sens"))
                if val_sens is not None and 0.0 <= val_sens <= 1.0:
                    last_sens = val_sens
                    best_sens = val_sens if best_sens is None else max(best_sens, val_sens)

                if stage == "final_eval":
                    final_auc = val_auc if val_auc is not None else final_auc
                    final_spec = val_spec if val_spec is not None else final_spec
                    final_sens = val_sens if val_sens is not None else final_sens
    except Exception:
        return None, None, None, None, None, None

    final_auc = final_auc if final_auc is not None else last_auc
    final_spec = final_spec if final_spec is not None else last_spec
    final_sens = final_sens if final_sens is not None else last_sens

    return final_auc, best_auc, final_spec, best_spec, final_sens, best_sens


def parse_pytorch_elapsed(csv_path: Path) -> Optional[float]:
    return extract_total_train_time(csv_path)


def analyze_tensorflow_result(result_dir: Path, use_logs: bool) -> Tuple[int, List[RunMetrics]]:
    manifest = parse_manifest(result_dir / "env_manifest.txt")
    batch_size = safe_int(manifest.get("global_batch_size"))
    if batch_size is None:
        raise RuntimeError(f"Batch size não encontrado em {result_dir}")
    job_id = manifest.get("job_id")
    if not job_id:
        raise RuntimeError(f"job_id ausente em {result_dir}")
    job_name = manifest.get("job_name") or "tensorflow_distributed"
    project_root = result_dir.parents[1]
    logs_root = project_root / "logs"
    data_root = project_root / "data" / "all-tfrec"
    train_images = count_train_images(data_root)
    run_ids = existing_run_ids(result_dir)
    if not run_ids:
        raise RuntimeError(f"Nenhum run encontrado em {result_dir}")
    log_stats: Dict[int, Tuple[float, float]] = {}
    if use_logs:
        log_patterns = [f"{job_name}_{job_id}_*.out"]
        if job_name != "tensorflow_distributed":
            log_patterns.append(f"tensorflow_distributed_{job_id}_*.out")
        log_patterns.append(f"*{job_id}*.out")
        log_paths: List[Path] = []
        for pattern in log_patterns:
            log_paths = sorted(logs_root.glob(pattern))
            if log_paths:
                break
        if log_paths:
            log_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in log_paths)
            log_stats = parse_tensorflow_log_text(log_text)
        else:
            patterns = ", ".join(log_patterns)
            print(f"[WARN] Logs {patterns} nao encontrados em {logs_root}. Usando apenas CSVs.", file=sys.stderr)

    metrics: List[RunMetrics] = []
    for rid in run_ids:
        run_dir = _run_dir_for_id(result_dir, rid)
        csv_path = pick_first_csv(run_dir)
        epochs: Optional[int] = None
        auc_final: Optional[float] = None
        auc_best: Optional[float] = None
        if csv_path is not None:
            try:
                epochs = infer_epochs_from_csv(csv_path)
            except Exception:
                epochs = None
        (
            auc_final,
            auc_best,
            spec_final,
            spec_best,
            sens_final,
            sens_best,
        ) = parse_val_metrics(csv_path)
        elapsed = None
        log_auc = None
        log_elapsed = None
        if rid in log_stats:
            log_auc, log_elapsed = log_stats[rid]
        if log_auc is not None:
            auc_final = log_auc
        elapsed_from_csv = sum_elapsed_seconds_from_csv(csv_path) if csv_path is not None else None
        # Logs já trazem o tempo total; preferimos esse valor, caindo para soma por época via CSV quando ausente.
        elapsed = log_elapsed if log_elapsed is not None else elapsed_from_csv
        # Throughput: se temos tempo total e épocas, derivamos imgs/s para ser consistente com o elapsed.
        throughput = None
        if elapsed is not None and epochs is not None and elapsed > 0:
            throughput = (train_images * epochs) / elapsed
        if throughput is None and csv_path is not None:
            throughput = extract_train_throughput_from_csv(csv_path)
        if throughput is None:
            throughput = extract_throughput_from_logs(result_dir)
        # Se throughput veio de logs/CSV e ainda não temos elapsed, podemos recalcular tempo pelo throughput.
        if elapsed is None and throughput is not None and epochs is not None and throughput > 0:
            elapsed = (train_images * epochs) / throughput
        mem = extract_peak_gpu_mem_mb(csv_path) if csv_path is not None else None
        if mem is None:
            mem = extract_peak_mem_from_logs(result_dir)
        time_to_target = time_to_target_auc(csv_path, TARGET_AUC, total_time_s=elapsed, epochs=epochs)
        metrics.append(
            RunMetrics(
                run_id=rid,
                auc=auc_best or auc_final,
                throughput_img_s=throughput,
                train_time_s=elapsed,
                gpu_mem_mb=mem,
                auc_best=auc_best,
                spec=spec_best or spec_final,
                spec_best=spec_best,
                sens=sens_best or sens_final,
                sens_best=sens_best,
                epochs=epochs,
                time_to_target_auc_s=time_to_target,
            )
        )
    if not metrics:
        raise RuntimeError(f"Nenhuma métrica encontrada em {result_dir}")
    return batch_size, sorted(metrics, key=lambda m: m.run_id)


def analyze_pytorch_result(result_dir: Path) -> Tuple[int, List[RunMetrics]]:
    manifest = parse_manifest(result_dir / "env_manifest.txt")
    batch_size = safe_int(manifest.get("global_batch_size"))
    if batch_size is None:
        raise RuntimeError(f"Batch size não encontrado em {result_dir}")
    project_root = result_dir.parents[1]
    data_root = project_root / "data" / "all-tfrec"
    train_images = count_train_images(data_root)
    metrics: List[RunMetrics] = []
    for run_dir in _list_run_dirs(result_dir):
        csv_path = pick_first_csv(run_dir)
        if csv_path is None:
            continue
        try:
            epochs = infer_epochs_from_csv(csv_path)
        except Exception:
            epochs = None
        (
            val_auc_final,
            val_auc_best,
            val_spec_final,
            val_spec_best,
            val_sens_final,
            val_sens_best,
        ) = parse_val_metrics(csv_path)
        elapsed = parse_pytorch_elapsed(csv_path)
        if elapsed is None:
            elapsed = sum_elapsed_seconds_from_csv(csv_path)
        name = run_dir.name
        if name.startswith("run_"):
            name = name.split("_", 1)[-1]
        try:
            run_id = int(name)
        except ValueError:
            continue
        elapsed_from_csv = sum_elapsed_seconds_from_csv(csv_path)
        if elapsed is not None and elapsed_from_csv is not None:
            elapsed = max(elapsed, elapsed_from_csv)
        elif elapsed is None:
            elapsed = elapsed_from_csv
        throughput = extract_train_throughput_from_csv(csv_path)
        if throughput is None and elapsed is not None and elapsed > 0:
            throughput = (train_images * epochs) / elapsed
        if elapsed is None and throughput is not None and epochs is not None and throughput > 0:
            elapsed = (train_images * epochs) / throughput
        gpu_mem = extract_peak_gpu_mem_mb(csv_path)
        if gpu_mem is None:
            gpu_mem = extract_peak_mem_from_logs(result_dir)
        time_to_target = time_to_target_auc(csv_path, TARGET_AUC, total_time_s=elapsed, epochs=epochs)
        metrics.append(
            RunMetrics(
                run_id=run_id,
                auc=val_auc_best or val_auc_final,
                throughput_img_s=throughput,
                train_time_s=elapsed,
                gpu_mem_mb=gpu_mem,
                auc_best=val_auc_best,
                spec=val_spec_best or val_spec_final,
                spec_best=val_spec_best,
                sens=val_sens_best or val_sens_final,
                sens_best=val_sens_best,
                epochs=epochs,
                time_to_target_auc_s=time_to_target,
            )
        )
    if not metrics:
        raise RuntimeError(f"Nenhuma métrica encontrada em {result_dir}")
    return batch_size, sorted(metrics, key=lambda m: m.run_id)


def _safe_float(value: Optional[str]) -> Optional[float]:
    try:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        val = float(value)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None


def extract_total_train_time(csv_path: Path) -> Optional[float]:
    """
    Extrai tempo total de treino preferindo colunas dedicadas, depois linha final_eval,
    e por último somando tempos por época.
    """
    explicit_total: Optional[float] = None
    final_eval_elapsed: Optional[float] = None
    total = 0.0
    found_partial = False
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage = (_get_row_value_case_insensitive(row, "stage") or "").strip().lower()
                total_from_row = _safe_float(_get_row_value_case_insensitive(row, "total_train_time_s"))
                if total_from_row is None:
                    total_from_row = _safe_float(_get_row_value_case_insensitive(row, "total_elapsed_s"))
                if total_from_row is not None and total_from_row > 0:
                    explicit_total = total_from_row if explicit_total is None else max(explicit_total, total_from_row)
                if stage == "final_eval":
                    final_eval_elapsed = (
                        _safe_float(_get_row_value_case_insensitive(row, "val_elapsed_s"))
                        or _safe_float(_get_row_value_case_insensitive(row, "train_elapsed_s"))
                        or final_eval_elapsed
                    )
                    continue
                epoch_time = _safe_float(_get_row_value_case_insensitive(row, "epoch_time_sec"))
                train_elapsed = _safe_float(_get_row_value_case_insensitive(row, "train_elapsed_s"))
                val_elapsed = _safe_float(_get_row_value_case_insensitive(row, "val_elapsed_s"))
                if epoch_time is not None:
                    total += epoch_time
                    found_partial = True
                    continue
                if train_elapsed is not None:
                    total += train_elapsed
                    found_partial = True
                if val_elapsed is not None:
                    total += val_elapsed
                    found_partial = True
    except Exception:
        return None
    if explicit_total is not None:
        return explicit_total
    if final_eval_elapsed is not None:
        return final_eval_elapsed
    if found_partial:
        return total
    return None


def sum_elapsed_seconds_from_csv(csv_path: Path) -> Optional[float]:
    return extract_total_train_time(csv_path)


def extract_peak_gpu_mem_mb(csv_path: Path) -> Optional[float]:
    preferred_columns = [
        "train_gpu_mem_alloc_mb",
        "val_gpu_mem_alloc_mb",
        "gpu_mem_peak_mb",
        "gpu_mem_current_mb",
        "train_gpu_mem_reserved_mb",
        "val_gpu_mem_reserved_mb",
    ]
    max_val: Optional[float] = None
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in preferred_columns:
                    val = _safe_float(row.get(key))
                    if val is None:
                        continue
                    if val <= 0 or val > MAX_MEM_MB:
                        continue
                    max_val = val if max_val is None else max(max_val, val)
    except (FileNotFoundError, Exception):
        return None
    return max_val


def _mean_and_stdev(values: Iterable[Optional[float]]) -> Tuple[float, float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    mean_value = statistics.mean(finite)
    stdev_value = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return mean_value, stdev_value


def summarize_metrics(variant: str, framework: str, batch_size: int, gpus: int, runs: List[RunMetrics]) -> Dict[str, float]:
    aucs = [r.auc for r in runs]
    thr = [r.throughput_img_s for r in runs]
    times = [r.train_time_s for r in runs]
    mem = [r.gpu_mem_mb for r in runs if r.gpu_mem_mb is not None]
    specs = [r.spec for r in runs]
    sens = [r.sens for r in runs]
    mean_auc, std_auc = _mean_and_stdev(aucs)
    mean_thr, std_thr = _mean_and_stdev(thr)
    mean_times, std_times = _mean_and_stdev(times)
    mean_spec, std_spec = _mean_and_stdev(specs)
    mean_sens, std_sens = _mean_and_stdev(sens)
    if mem:
        mean_mem, std_mem = _mean_and_stdev(mem)
    else:
        mean_mem = float("nan")
        std_mem = float("nan")
    return {
        "project": VARIANT_PROJECTS[variant].get(framework, ""),
        "framework": framework,
        "variant": variant,
        "gpus": gpus,
        "batch_size": batch_size,
        "runs": len(runs),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_specificity": mean_spec,
        "std_specificity": std_spec,
        "mean_sensitivity": mean_sens,
        "std_sensitivity": std_sens,
        "mean_throughput_img_s": mean_thr,
        "std_throughput_img_s": std_thr,
        "mean_train_time_s": mean_times,
        "std_train_time_s": std_times,
        "mean_peak_gpu_mem_mb": mean_mem,
        "std_peak_gpu_mem_mb": std_mem,
    }


def _framework_label(name: str) -> str:
    lname = name.lower()
    if lname == "tensorflow":
        return "TensorFlow"
    if lname == "pytorch":
        return "PyTorch"
    return name


def _framework_gpu_label(name: str, gpus: int) -> str:
    gpu_label = "GPU" if gpus == 1 else "GPUs"
    return f"{_framework_label(name)} {gpus} {gpu_label}"


def aggregate_results(
    variant: str,
    framework: str,
    gpus: int,
    target_batches: Iterable[int],
    result_dirs: Dict[int, Path],
    analyzer: Callable[[Path], Tuple[int, List[RunMetrics]]],
    summary_store: Dict[SummaryKey, Dict[str, float]],
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    gpu_word = "GPU" if gpus == 1 else "GPUs"
    framework_label = _framework_label(framework)
    for batch in target_batches:
        result_dir = result_dirs.get(batch)
        if result_dir is None:
            print(
                f"[WARN] Nenhum resultado encontrado para {framework_label} ({variant}, {gpus} {gpu_word}) batch {batch}",
                file=sys.stderr,
            )
            continue
        try:
            _, metrics = analyzer(result_dir)
        except Exception as exc:
            print(f"[WARN] Falha ao analisar {result_dir}: {exc}", file=sys.stderr)
            continue
        summary = summarize_metrics(variant, framework, batch, gpus=gpus, runs=metrics)
        records.append(summary)
        summary_store[(variant, framework, gpus, batch)] = summary
    return records


def collect_gpu2_bs96_runs(
    variant_roots: Dict[str, Dict[str, Path]],
    analyzers: Dict[str, Callable[[Path], Tuple[int, List[RunMetrics]]]],
    gpus: int = 2,
    batch_size: int = 96,
    prefer_partition: Optional[str] = "grace",
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for variant, roots in variant_roots.items():
        for framework, results_root in roots.items():
            analyzer = analyzers.get(framework)
            if analyzer is None:
                print(f"[WARN] Nenhum analisador registrado para {framework}.", file=sys.stderr)
                continue
            result_dir = select_preferred_result_dir(
                results_root, gpu_count=gpus, batch_size=batch_size, prefer_partition=prefer_partition
            )
            if result_dir is None:
                print(
                    f"[WARN] Nenhum resultado encontrado para {_framework_label(framework)} "
                    f"({variant}, {gpus} GPUs) batch {batch_size}",
                    file=sys.stderr,
                )
                continue
            project = results_root.parent.name
            try:
                batch_found, run_metrics = analyzer(result_dir)
            except Exception as exc:
                print(f"[WARN] Falha ao analisar {result_dir} ({framework}): {exc}", file=sys.stderr)
                continue
            gpus_value, batch_value, _ = parse_result_metadata(result_dir)
            effective_batch = batch_found or batch_value or batch_size
            for run in run_metrics:
                records.append(
                    {
                        "pair": variant,
                        "project": project,
                        "result_dir": result_dir.name,
                        "framework": framework,
                        "variant": variant,
                        "gpus": gpus_value or gpus,
                        "batch_size": effective_batch,
                        "run_id": run.run_id,
                        "epochs": run.epochs,
                        "val_auc_final": run.auc,
                        "val_auc_best": run.auc_best,
                        "train_time_s": run.train_time_s,
                        "throughput_img_s": run.throughput_img_s,
                        "peak_gpu_mem_mb": run.gpu_mem_mb,
                        "time_to_auc_0_95_s": run.time_to_target_auc_s,
                    }
                )
    return records


def collect_batch96_all_runs(
    variant_roots: Dict[str, Dict[str, Path]],
    analyzers: Dict[str, Callable[[Path], Tuple[int, List[RunMetrics]]]],
    gpus_options: Iterable[int],
    batch_size: int = 96,
    prefer_partition: Optional[str] = "grace",
) -> List[Dict[str, float]]:
    """Coleta todos os runs de batch 96 para múltiplos contagens de GPU."""
    records: List[Dict[str, float]] = []
    for variant, roots in variant_roots.items():
        for framework, results_root in roots.items():
            analyzer = analyzers.get(framework)
            if analyzer is None:
                continue
            for gpus in gpus_options:
                result_dir = select_preferred_result_dir(
                    results_root, gpu_count=gpus, batch_size=batch_size, prefer_partition=prefer_partition
                )
                if result_dir is None:
                    continue
                project = results_root.parent.name
                try:
                    _, run_metrics = analyzer(result_dir)
                except Exception:
                    continue
                gpus_value, batch_value, _ = parse_result_metadata(result_dir)
                effective_batch = batch_value or batch_size
                for run in run_metrics:
                    records.append(
                        {
                            "project": project,
                            "framework": framework,
                            "variant": variant,
                            "gpus": gpus_value or gpus,
                            "batch_size": effective_batch,
                            "result_dir": result_dir.name,
                            "run_id": run.run_id,
                            "epochs": run.epochs,
                            "val_auc_final": run.auc,
                            "val_auc_best": run.auc_best,
                            "val_spec_final": run.spec,
                            "val_spec_best": run.spec_best,
                            "val_sens_final": run.sens,
                            "val_sens_best": run.sens_best,
                            "train_time_s": run.train_time_s,
                            "throughput_img_s": run.throughput_img_s,
                            "peak_gpu_mem_mb": run.gpu_mem_mb,
                            "time_to_auc_0_95_s": run.time_to_target_auc_s,
                        }
                    )
    return records


def summarize_gpu2_bs96_runs(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for row in rows:
        key = (row["pair"], row["project"], row["result_dir"])
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, float]] = []
    for (pair, project, result_dir), items in grouped.items():
        framework = items[0]["framework"]
        variant = items[0]["variant"]
        gpus = items[0]["gpus"]
        batch_size = items[0]["batch_size"]
        mean_val_auc_final, std_val_auc_final = _mean_and_stdev([i.get("val_auc_final") for i in items])
        mean_val_auc_best, std_val_auc_best = _mean_and_stdev([i.get("val_auc_best") for i in items])
        mean_thr, std_thr = _mean_and_stdev([i.get("throughput_img_s") for i in items])
        mean_time, std_time = _mean_and_stdev([i.get("train_time_s") for i in items])
        mean_mem, std_mem = _mean_and_stdev([i.get("peak_gpu_mem_mb") for i in items])
        summaries.append(
            {
                "pair": pair,
                "project": project,
                "result_dir": result_dir,
                "framework": framework,
                "variant": variant,
                "gpus": gpus,
                "batch_size": batch_size,
                "runs": len(items),
                "mean_val_auc_final": mean_val_auc_final,
                "std_val_auc_final": std_val_auc_final,
                "mean_val_auc_best": mean_val_auc_best,
                "std_val_auc_best": std_val_auc_best,
                "mean_throughput_img_s": mean_thr,
                "std_throughput_img_s": std_thr,
                "mean_train_time_s": mean_time,
                "std_train_time_s": std_time,
                "mean_peak_gpu_mem_mb": mean_mem,
                "std_peak_gpu_mem_mb": std_mem,
            }
        )
    return summaries


def summarize_single_gpu_runs(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Agrega todas as execuções (run_*), agrupando por projeto/framework/variant/gpus/batch_size.
    Usa os valores \"best\" quando disponíveis (AUC, sens, spec).
    """
    grouped: Dict[Tuple[str, str, str, int, int], List[Dict[str, float]]] = {}
    for row in rows:
        key = (row["project"], row["framework"], row["variant"], row["gpus"], row["batch_size"])
        grouped.setdefault(key, []).append(row)

    summaries: List[Dict[str, float]] = []
    for (project, framework, variant, gpus, batch_size), items in grouped.items():
        mean_auc, std_auc = _mean_and_stdev([i.get("val_auc_best") or i.get("val_auc_final") for i in items])
        mean_spec, std_spec = _mean_and_stdev([i.get("val_spec_best") or i.get("val_spec_final") for i in items])
        mean_sens, std_sens = _mean_and_stdev([i.get("val_sens_best") or i.get("val_sens_final") for i in items])
        mean_thr, std_thr = _mean_and_stdev([i.get("throughput_img_s") for i in items])
        mean_time, std_time = _mean_and_stdev([i.get("train_time_s") for i in items])
        mean_mem, std_mem = _mean_and_stdev([i.get("peak_gpu_mem_mb") for i in items])

        summaries.append(
            {
                "project": project,
                "framework": framework,
                "variant": variant,
                "gpus": gpus,
                "batch_size": batch_size,
                "runs": len(items),
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_specificity": mean_spec,
                "std_specificity": std_spec,
                "mean_sensitivity": mean_sens,
                "std_sensitivity": std_sens,
                "mean_throughput_img_s": mean_thr,
                "std_throughput_img_s": std_thr,
                "mean_train_time_s": mean_time,
                "std_train_time_s": std_time,
                "mean_peak_gpu_mem_mb": mean_mem,
                "std_peak_gpu_mem_mb": std_mem,
            }
        )

    return summaries


def write_summary_csv(path: Path, records: List[Dict[str, float]], fieldnames: Optional[List[str]] = None) -> None:
    fnames = fieldnames or CSV_FIELDNAMES
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r.get("variant", ""), r.get("framework", ""), r.get("gpus", 0), r.get("batch_size", 0))):
            writer.writerow(rec)


def write_gpu2_bs96_runs_csv(path: Path, records: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GPU2_BS96_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["pair"], r["project"], r["result_dir"], r["run_id"])):
            writer.writerow(rec)


def write_gpu2_bs96_summary_csv(path: Path, records: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GPU2_BS96_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["pair"], r["project"], r["result_dir"])):
            writer.writerow(rec)


def write_single_gpu_runs_csv(path: Path, records: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SINGLE_GPU_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted(
            records,
            key=lambda r: (
                r.get("project", ""),
                r.get("framework", ""),
                r.get("variant", ""),
                r.get("gpus", 0),
                r.get("batch_size", 0),
                r.get("run_id", 0),
            ),
        ):
            writer.writerow(rec)


def write_single_gpu_summary_csv(path: Path, records: List[Dict[str, float]]) -> None:
    write_summary_csv(path, records, fieldnames=SINGLE_GPU_SUMMARY_FIELDS)


def print_framework_comparison(
    summary_store: Dict[SummaryKey, Dict[str, float]],
    variant: str,
    gpus: int,
    target_batches: Iterable[int],
) -> None:
    for batch in target_batches:
        tf_summary = summary_store.get((variant, "tensorflow", gpus, batch))
        torch_summary = summary_store.get((variant, "pytorch", gpus, batch))
        if not tf_summary or not torch_summary:
            continue
        tf_time = tf_summary["mean_train_time_s"]
        torch_time = torch_summary["mean_train_time_s"]
        if not (math.isfinite(tf_time) and math.isfinite(torch_time)):
            continue
        if tf_time <= 0 or torch_time <= 0:
            continue
        if tf_time < torch_time:
            faster_framework = "tensorflow"
            faster_time = tf_time
            slower_time = torch_time
        else:
            faster_framework = "pytorch"
            faster_time = torch_time
            slower_time = tf_time
        speed_pct = (slower_time - faster_time) / slower_time * 100.0
        faster_label = _framework_gpu_label(faster_framework, gpus)
        print(
            f"[{variant}] Batch {batch}: {faster_label} foi {speed_pct:.2f}% mais rápido "
            f"(tempo médio {faster_time:.1f}s vs {slower_time:.1f}s)."
        )


def print_intra_framework_speedup(
    summary_store: Dict[SummaryKey, Dict[str, float]],
    variant: str,
    framework: str,
    from_gpus: int,
    to_gpus: int,
    target_batches: Iterable[int],
) -> None:
    batches = sorted(set(target_batches))
    for batch in batches:
        base = summary_store.get((variant, framework, from_gpus, batch))
        new = summary_store.get((variant, framework, to_gpus, batch))
        if not base or not new:
            continue
        base_time = base["mean_train_time_s"]
        new_time = new["mean_train_time_s"]
        if not (math.isfinite(base_time) and math.isfinite(new_time)):
            continue
        if base_time <= 0 or new_time <= 0:
            continue
        speed = (base_time - new_time) / base_time * 100.0
        print(
            f"[{variant}] [{_framework_label(framework)}] Batch {batch}: {to_gpus} GPUs foram "
            f"{speed:.2f}% mais rápidos que {from_gpus} GPU(s) "
            f"({new_time:.1f}s vs {base_time:.1f}s)."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara tensorflow_opt e pytorch_opt em 1 e 2 GPUs (batch 96) gerando um único CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "batch96_gpu_compare.csv",
        help="Caminho do CSV de saída (padrão: batch96_gpu_compare.csv no diretório do script).",
    )
    parser.add_argument(
        "--single-gpu-runs-output",
        type=Path,
        default=Path(__file__).resolve().parent / "single_gpu_runs.csv",
        help="CSV com todas as execuções (run_*) em 1 GPU (padrão: single_gpu_runs.csv).",
    )
    parser.add_argument(
        "--single-gpu-summary-output",
        type=Path,
        default=Path(__file__).resolve().parent / "single_gpu_summary.csv",
        help="CSV com médias/DP das execuções em 1 GPU (padrão: single_gpu_summary.csv).",
    )
    parser.add_argument(
        "--single-gpu-batch-size",
        type=int,
        default=96,
        help="Batch size alvo para o relatório de 1 GPU (padrão: 96).",
    )
    parser.add_argument(
        "--single-gpu-gpus",
        type=int,
        default=1,
        help="Quantidade de GPUs para o relatório single-GPU (padrão: 1).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    use_logs = True
    tf_analyzer = partial(analyze_tensorflow_result, use_logs=use_logs)
    analyzers = {
        "tensorflow": tf_analyzer,
        "pytorch": analyze_pytorch_result,
    }

    frameworks = {
        "tensorflow": "tensorflow_opt",
        "pytorch": "pytorch_opt",
    }

    records: List[Dict[str, float]] = []
    gpus_options = (1, 2)

    for framework, project_name in frameworks.items():
        results_root = root / project_name / "results"
        for gpus in gpus_options:
            result_dir = select_preferred_result_dir(
                results_root, gpu_count=gpus, batch_size=96, prefer_partition="grace"
            )
            if result_dir is None:
                print(
                    f"[WARN] Nenhum resultado encontrado para {framework} ({gpus} GPUs, batch 96).",
                    file=sys.stderr,
                )
                continue
            analyzer = analyzers[framework]
            try:
                _, run_metrics = analyzer(result_dir)
            except Exception as exc:
                print(f"[WARN] Falha ao analisar {result_dir} ({framework}): {exc}", file=sys.stderr)
                continue
            summary = summarize_metrics("original", framework, batch_size=96, gpus=gpus, runs=run_metrics)
            summary["project"] = project_name
            records.append(summary)

    if not records:
        raise SystemExit("Nenhum registro encontrado para 1/2 GPUs, batch 96.")

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS_BATCH96_GPU2, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r.get("gpus", 0), r.get("framework", ""))):
            writer.writerow(rec)

    print(f"Resumo 1×/2×GPU batch 96 salvo em: {args.output}")

    # --- Novo relatório: todas as variantes (original + clean) em 1 GPU ---
    variant_roots: Dict[str, Dict[str, Path]] = {
        variant: {fw: (root / proj / "results") for fw, proj in projects.items()}
        for variant, projects in VARIANT_PROJECTS.items()
    }

    single_gpu_runs = collect_batch96_all_runs(
        variant_roots,
        analyzers,
        gpus_options=(args.single_gpu_gpus,),
        batch_size=args.single_gpu_batch_size,
        prefer_partition="grace",
    )

    if single_gpu_runs:
        write_single_gpu_runs_csv(args.single_gpu_runs_output, single_gpu_runs)
        single_gpu_summary = summarize_single_gpu_runs(single_gpu_runs)
        write_single_gpu_summary_csv(args.single_gpu_summary_output, single_gpu_summary)
        print(f"Execuções 1×GPU salvas em: {args.single_gpu_runs_output}")
        print(f"Médias 1×GPU salvas em:   {args.single_gpu_summary_output}")
    else:
        print("[WARN] Nenhuma execução encontrada para o relatório 1×GPU.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

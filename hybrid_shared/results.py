"""
Shared helpers for richer training result artifacts.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from hybrid_shared.metrics import compute_auc

try:
    from sklearn.metrics import roc_curve as sklearn_roc_curve
except Exception:
    sklearn_roc_curve = None


# Schema do CSV por época. O writer projeta as linhas sobre esta lista, então
# retirar um nome daqui basta para parar de gravá-lo.
#
# Removidos na auditoria (2026-07):
#   train_auc/precision/f1/sens/spec  -> calculados sobre lotes com augmentation
#                                        (mixup/cutmix): enganosos como métrica clínica.
#   val_gpu_util_pct, val_busy_time_s -> a validação não entra em Q1/Q2.
#   *_avg_power_w (val/test)          -> redundante: energia / tempo.
#   *_avg_batch_time_ms (val/test)    -> redundante com elapsed / nº de batches.
#   *_inference_latency_*             -> não usado; é ~1/throughput.
#   *_gpu_mem_avg_mb                  -> o que importa é o PICO (footprint).
#
# spec@95sens NAO e' coletado no CSV: e' uma metrica DERIVADA (exige a curva ROC
# completa). Coletamos apenas spec (spec@0.5) e sens; o make_plots.py deriva o
# spec@95sens a partir do *-thresholds.csv. (Antes existia val/test_spec_at_sens95
# aqui, mas caiam num fallback SILENCIOSO para spec@0.5 -> valor inflado. Removidas.)
METRICS_CSV_FIELDS = [
    "epoch",
    "stage",
    # --- treino: só custo computacional (as métricas clínicas do treino são enganosas)
    "train_loss",
    "train_throughput_img_s",
    "train_elapsed_s",
    "train_gpu_mem_peak_mb",
    "train_energy_j",
    "train_avg_power_w",
    "train_gpu_util_pct",
    "train_mem_util_pct",
    "train_busy_time_s",
    # --- validação: trajetória clínica limpa (sem augmentation) + custo
    "val_loss",
    "val_auc",
    "val_precision",
    "val_f1",
    "val_sens",
    "val_spec",
    "val_elapsed_s",
    "val_gpu_mem_peak_mb",
    "val_energy_j",
    # --- teste final (linha stage=final_test)
    "test_auc",
    "test_precision",
    "test_f1",
    "test_sens",
    "test_spec",
    "test_throughput_img_s",
    "test_elapsed_s",
    "test_gpu_mem_peak_mb",
    "test_energy_j",
    "lr",
    "total_train_time_s",
]


def _scalar_or_nan(value: Any) -> Any:
    if value is None:
        return math.nan
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _split_metrics(prefix: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {
            f"{prefix}_loss": math.nan,
            f"{prefix}_auc": math.nan,
            f"{prefix}_precision": math.nan,
            f"{prefix}_recall": math.nan,
            f"{prefix}_f1": math.nan,
            f"{prefix}_sens": math.nan,
            f"{prefix}_spec": math.nan,
            f"{prefix}_throughput_img_s": math.nan,
            f"{prefix}_elapsed_s": math.nan,
            f"{prefix}_avg_batch_time_ms": math.nan,
            f"{prefix}_inference_latency_ms_img": math.nan,
            f"{prefix}_inference_latency_ms_batch": math.nan,
            f"{prefix}_gpu_mem_avg_mb": math.nan,
            f"{prefix}_gpu_mem_peak_mb": math.nan,
            f"{prefix}_energy_j": math.nan,
            f"{prefix}_avg_power_w": math.nan,
            f"{prefix}_gpu_util_pct": math.nan,
            f"{prefix}_mem_util_pct": math.nan,
            f"{prefix}_busy_time_s": math.nan,
        }

    # spec@95sens NAO e' coletado aqui: e' uma metrica DERIVADA (precisa da curva
    # ROC completa) e o make_plots.py a calcula a partir do *-thresholds.csv.
    # Coletamos apenas spec (spec@0.5) e sens brutos.
    return {
        f"{prefix}_loss": _scalar_or_nan(metrics.get("loss")),
        f"{prefix}_auc": _scalar_or_nan(metrics.get("auc")),
        f"{prefix}_precision": _scalar_or_nan(metrics.get("precision")),
        f"{prefix}_recall": _scalar_or_nan(metrics.get("recall")),
        f"{prefix}_f1": _scalar_or_nan(metrics.get("f1")),
        f"{prefix}_sens": _scalar_or_nan(metrics.get("sensitivity")),
        f"{prefix}_spec": _scalar_or_nan(metrics.get("specificity")),
        f"{prefix}_throughput_img_s": _scalar_or_nan(metrics.get("throughput")),
        f"{prefix}_elapsed_s": _scalar_or_nan(metrics.get("epoch_time")),
        f"{prefix}_avg_batch_time_ms": _scalar_or_nan(metrics.get("avg_batch_time_ms")),
        f"{prefix}_inference_latency_ms_img": _scalar_or_nan(metrics.get("inference_latency_ms_img")),
        f"{prefix}_inference_latency_ms_batch": _scalar_or_nan(metrics.get("inference_latency_ms_batch")),
        f"{prefix}_gpu_mem_avg_mb": _scalar_or_nan(metrics.get("memory_avg_mb")),
        f"{prefix}_gpu_mem_peak_mb": _scalar_or_nan(metrics.get("memory_peak_mb")),
        f"{prefix}_energy_j": _scalar_or_nan(metrics.get("energy_j")),
        f"{prefix}_avg_power_w": _scalar_or_nan(metrics.get("avg_power_w")),
        f"{prefix}_gpu_util_pct": _scalar_or_nan(metrics.get("gpu_util_pct")),
        f"{prefix}_mem_util_pct": _scalar_or_nan(metrics.get("mem_util_pct")),
        f"{prefix}_busy_time_s": _scalar_or_nan(metrics.get("busy_time_s")),
    }


def make_metrics_row(
    epoch: int,
    stage: str,
    lr: float,
    train_metrics: dict[str, Any] | None = None,
    val_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
    total_train_time_s: float | None = None,
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "stage": stage,
        "lr": _scalar_or_nan(lr),
        "total_train_time_s": _scalar_or_nan(total_train_time_s),
    }
    row.update(_split_metrics("train", train_metrics))
    row.update(_split_metrics("val", val_metrics))
    row.update(_split_metrics("test", test_metrics))
    return row


def save_metrics_history_csv(csv_path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_csv_value(value: Any) -> Any:
        value = _scalar_or_nan(value)
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
            return ""
        return value

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _serialize_csv_value(row.get(field, math.nan))
                    for field in METRICS_CSV_FIELDS
                }
            )


def _to_numpy_1d(values: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return np.asarray(array).reshape(-1)


def _compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sklearn_roc_curve is not None:
        return sklearn_roc_curve(y_true, y_score)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[distinct_indices, y_true_sorted.size - 1]

    true_positives = np.cumsum(y_true_sorted)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives

    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_indices]]

    positives = max(1, int(np.sum(y_true)))
    negatives = max(1, int(y_true.shape[0] - np.sum(y_true)))
    tpr = true_positives / positives
    fpr = false_positives / negatives
    return fpr, tpr, thresholds


def save_roc_curve_artifacts(
    y_true: torch.Tensor | np.ndarray | Sequence[float],
    y_score: torch.Tensor | np.ndarray | Sequence[float],
    results_dir: str | Path,
    dataset_name: str,
    exec_id: int,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true_np = _to_numpy_1d(y_true).astype(np.int64)
    y_score_np = np.clip(_to_numpy_1d(y_score).astype(np.float64), 1e-7, 1 - 1e-7)
    finite_mask = np.isfinite(y_true_np) & np.isfinite(y_score_np)
    if not np.all(finite_mask):
        y_true_np = y_true_np[finite_mask]
        y_score_np = y_score_np[finite_mask]

    if y_true_np.size == 0:
        return {
            "auc": math.nan,
            "thresholds_csv_path": None,
            "roc_pdf_path": None,
        }

    fpr, tpr, thresholds = _compute_roc_curve(y_true_np, y_score_np)
    auc_value = compute_auc(torch.from_numpy(y_true_np), torch.from_numpy(y_score_np.astype(np.float32)))

    thresholds_path = results_dir / f"{dataset_name}-{exec_id}-thresholds.csv"
    with thresholds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["thresholds", "tpr", "fpr", "sens", "spec"])
        for threshold, tpr_value, fpr_value in zip(thresholds, tpr, fpr):
            writer.writerow([threshold, tpr_value, fpr_value, tpr_value, 1.0 - fpr_value])

    roc_pdf_path = results_dir / f"{dataset_name}-{exec_id}.pdf"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="best")
        plt.savefig(roc_pdf_path, format="pdf", bbox_inches="tight")
        plt.close()
    except Exception:
        roc_pdf_path = None

    return {
        "auc": float(auc_value),
        "thresholds_csv_path": str(thresholds_path),
        "roc_pdf_path": str(roc_pdf_path) if roc_pdf_path is not None else None,
    }


def build_training_summary(
    model_name: str,
    best_epoch: int,
    best_val_auc: float,
    eval_split_name: str,
    eval_metrics: dict[str, Any],
    profiler,
    total_train_time_s: float | None,
    metadata: Iterable[tuple[str, Any]] | None = None,
    artifacts: Iterable[tuple[str, Any]] | None = None,
) -> str:
    profiler_summary = profiler.get_summary() if profiler is not None else {}
    efficiency = profiler.efficiency_score(eval_metrics["auc"]) if profiler is not None else {}
    split_label = eval_split_name.capitalize()

    lines = []
    for key, value in metadata or ():
        lines.append(f"{key}: {value}")

    if metadata:
        lines.append("")

    lines.extend(
        [
            f"Model: {model_name}",
            f"Best epoch: {best_epoch}",
            f"Best val AUC: {best_val_auc:.4f}",
            f"{split_label} AUC: {eval_metrics['auc']:.4f}",
            f"{split_label} Precision: {eval_metrics['precision']:.4f}",
            f"{split_label} Recall: {eval_metrics['recall']:.4f}",
            f"{split_label} Sensitivity: {eval_metrics['sensitivity']:.4f}",
            f"{split_label} Specificity @ sensitivity 95%: {eval_metrics['specificity']:.4f}",
            f"{split_label} F1: {eval_metrics['f1']:.4f}",
        ]
    )

    if "throughput" in eval_metrics:
        lines.append(f"{split_label} Throughput: {eval_metrics['throughput']:.1f} img/s")
    if "optimal_threshold" in eval_metrics:
        lines.append(f"{split_label} Optimal threshold (F1): {eval_metrics['optimal_threshold']:.4f}")
    if "optimal_f1" in eval_metrics:
        lines.append(f"{split_label} Optimal F1: {eval_metrics['optimal_f1']:.4f}")

    if total_train_time_s is not None:
        lines.append(f"Total train time: {total_train_time_s:.1f}s")

    if profiler_summary:
        lines.extend(
            [
                f"Avg train epoch time: {profiler_summary['avg_epoch_time_s']:.2f}s",
                (
                    "Train epoch time range: "
                    f"{profiler_summary['min_epoch_time_s']:.2f}s - {profiler_summary['max_epoch_time_s']:.2f}s"
                ),
                f"Avg train batch time: {profiler_summary['avg_batch_time_ms']:.2f}ms",
                f"Avg GPU memory: {profiler_summary['avg_memory_mb']:.2f} MB",
            ]
        )

    if efficiency:
        lines.extend(
            [
                f"AUC / second: {efficiency['auc_per_second']:.6f}",
                f"AUC / MB: {efficiency['auc_per_mb']:.6f}",
                f"GPU-seconds proxy: {efficiency['total_gpu_memory_seconds']:.0f}",
            ]
        )

    artifact_lines = [(label, value) for label, value in (artifacts or ()) if value]
    if artifact_lines:
        lines.append("")
        for label, value in artifact_lines:
            lines.append(f"{label}: {value}")

    return "\n".join(lines) + "\n"


def _is_valid_number(value: Any) -> bool:
    value = _scalar_or_nan(value)
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        return not math.isnan(float(value))
    return isinstance(value, (int, np.integer))


def compute_avg_gpu_memory_mb(
    metrics_rows: Sequence[dict[str, Any]] | None = None,
    eval_metrics: dict[str, Any] | None = None,
    profiler: Any | None = None,
) -> float | None:
    candidates: list[float] = []

    for row in metrics_rows or ():
        for field in (
            "train_gpu_mem_avg_mb",
            "val_gpu_mem_avg_mb",
            "test_gpu_mem_avg_mb",
        ):
            value = row.get(field)
            if _is_valid_number(value):
                candidates.append(float(_scalar_or_nan(value)))

    for field in ("memory_avg_mb",):
        if eval_metrics and _is_valid_number(eval_metrics.get(field)):
            candidates.append(float(_scalar_or_nan(eval_metrics.get(field))))

    if profiler is not None:
        try:
            profiler_summary = profiler.get_summary()
        except Exception:
            profiler_summary = {}
        if _is_valid_number(profiler_summary.get("avg_memory_mb")):
            candidates.append(float(_scalar_or_nan(profiler_summary["avg_memory_mb"])))

    return float(np.mean(candidates)) if candidates else None


def build_terminal_final_summary(
    eval_metrics: dict[str, Any],
    total_train_time_s: float | None,
    avg_gpu_memory_mb: float | None = None,
) -> str:
    throughput = eval_metrics.get("throughput")
    lines = [
        "=" * 70,
        "FINAL SUMMARY",
        "=" * 70,
        f"AUC: {eval_metrics['auc']:.4f}",
        f"Precision: {eval_metrics['precision']:.4f}",
        f"Recall: {eval_metrics['recall']:.4f}",
        f"Sensitivity: {eval_metrics['sensitivity']:.4f}",
        f"Specificity @ sensitivity 95%: {eval_metrics['specificity']:.4f}",
        f"F1: {eval_metrics['f1']:.4f}",
    ]

    if _is_valid_number(throughput):
        lines.append(f"Throughput: {float(_scalar_or_nan(throughput)):.1f} img/s")
    else:
        lines.append("Throughput: n/a")

    if _is_valid_number(total_train_time_s):
        lines.append(f"Total train time: {float(_scalar_or_nan(total_train_time_s)):.1f}s")
    else:
        lines.append("Total train time: n/a")

    if _is_valid_number(avg_gpu_memory_mb):
        lines.append(f"Avg GPU memory: {float(_scalar_or_nan(avg_gpu_memory_mb)):.1f} MB")
    else:
        lines.append("Avg GPU memory: n/a")

    return "\n".join(lines) + "\n"


def build_terminal_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    throughput_img_s: float | None = None,
    lr: float | None = None,
) -> str:
    def _fmt(value: Any, precision: int) -> str:
        if not _is_valid_number(value):
            return "n/a"
        return f"{float(_scalar_or_nan(value)):.{precision}f}"

    def _fmt_lr(value: Any) -> str:
        if not _is_valid_number(value):
            return "n/a"
        return f"{float(_scalar_or_nan(value)):.2e}"

    if throughput_img_s is None:
        throughput_img_s = train_metrics.get("throughput")

    lines = [
        "=" * 70,
        f"Epoch {epoch + 1}/{total_epochs}",
        "=" * 70,
        (
            f"Train Loss: {_fmt(train_metrics.get('loss'), 4)} | "
            f"AUC: {_fmt(train_metrics.get('auc'), 4)} | "
            f"Precision: {_fmt(train_metrics.get('precision'), 4)} | "
            f"Recall: {_fmt(train_metrics.get('recall'), 4)} | "
            f"Spec@Sens95: {_fmt(train_metrics.get('specificity'), 4)} | "
            f"F1: {_fmt(train_metrics.get('f1'), 4)}"
        ),
        (
            f"Val   Loss: {_fmt(val_metrics.get('loss'), 4)} | "
            f"AUC: {_fmt(val_metrics.get('auc'), 4)} | "
            f"Precision: {_fmt(val_metrics.get('precision'), 4)} | "
            f"Recall: {_fmt(val_metrics.get('recall'), 4)} | "
            f"Spec@Sens95: {_fmt(val_metrics.get('specificity'), 4)} | "
            f"F1: {_fmt(val_metrics.get('f1'), 4)}"
        ),
        (
            f"Throughput: {_fmt(throughput_img_s, 1)} img/s"
            if _is_valid_number(throughput_img_s)
            else "Throughput: n/a"
        ),
        f"LR: {_fmt_lr(lr)}" if _is_valid_number(lr) else "LR: n/a",
    ]
    return "\n".join(lines) + "\n"

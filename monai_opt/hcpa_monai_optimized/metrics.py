from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


SPECIFICITY_TARGET_SENSITIVITY = 0.95


@dataclass
class EpochStats:
    losses: List[float] = field(default_factory=list)
    probs: List[np.ndarray] = field(default_factory=list)
    labels: List[np.ndarray] = field(default_factory=list)

    def update(self, loss: float, probs_batch: np.ndarray, labels_batch: np.ndarray) -> None:
        self.losses.append(float(loss))
        self.probs.append(np.asarray(probs_batch).reshape(-1))
        self.labels.append(np.asarray(labels_batch).reshape(-1))

    def aggregate(self, threshold: float = 0.5) -> Dict[str, float]:
        if not self.losses:
            return {
                k: float("nan")
                for k in (
                    "loss",
                    "auc",
                    "precision",
                    "recall",
                    "sensitivity",
                    "specificity",
                    "specificity_at_sens95",
                    "f1",
                    "accuracy",
                )
            }
        losses = float(np.mean(self.losses))
        probs, labels = self.stack()
        metrics = compute_binary_metrics(probs, labels, threshold=threshold)
        metrics["loss"] = losses
        return metrics

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.probs:
            return np.array([]), np.array([])
        probs = np.concatenate(self.probs, axis=0)
        labels = np.concatenate(self.labels, axis=0).astype(np.int32)
        return probs, labels


def specificity_at_sensitivity(
    labels: Sequence[int],
    probs: Sequence[float],
    target_sensitivity: float = SPECIFICITY_TARGET_SENSITIVITY,
) -> float:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    finite = np.isfinite(labels_arr) & np.isfinite(probs_arr)
    labels_arr = labels_arr[finite]
    probs_arr = probs_arr[finite]
    if labels_arr.size == 0 or probs_arr.size == 0:
        return float("nan")

    positives = labels_arr == 1
    n_pos = int(np.sum(positives))
    n_neg = int(labels_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    target = float(target_sensitivity)
    if target <= 0.0:
        return 1.0
    if target > 1.0:
        return float("nan")

    order = np.argsort(-probs_arr, kind="mergesort")
    sorted_scores = probs_arr[order]
    sorted_true = positives[order]
    group_end = np.ones(sorted_scores.shape[0], dtype=bool)
    if sorted_scores.shape[0] > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    threshold_idxs = np.flatnonzero(group_end)
    tp = np.cumsum(sorted_true, dtype=np.float64)[threshold_idxs]
    fp = np.cumsum(~sorted_true, dtype=np.float64)[threshold_idxs]
    tpr = np.r_[0.0, tp / float(n_pos)]
    fpr = np.r_[0.0, fp / float(n_neg)]

    idxs = np.flatnonzero(tpr >= target)
    if idxs.size == 0:
        return 0.0
    idx = int(idxs[0])
    if idx == 0:
        fpr_at_target = fpr[0]
    else:
        tpr_lo, tpr_hi = tpr[idx - 1], tpr[idx]
        fpr_lo, fpr_hi = fpr[idx - 1], fpr[idx]
        if abs(tpr_hi - tpr_lo) <= 1e-12:
            fpr_at_target = min(fpr_lo, fpr_hi)
        else:
            fraction = (target - tpr_lo) / (tpr_hi - tpr_lo)
            fpr_at_target = fpr_lo + fraction * (fpr_hi - fpr_lo)
    return float(np.clip(1.0 - fpr_at_target, 0.0, 1.0))


def compute_binary_metrics(probs: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> Dict[str, float]:
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    if probs_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError("probs and labels must have same length")

    try:
        auc = roc_auc_score(labels_arr, probs_arr)
    except ValueError:
        auc = float("nan")

    preds = (probs_arr >= threshold).astype(np.int32)
    acc = accuracy_score(labels_arr, preds)
    tp = float(np.sum((preds == 1) & (labels_arr == 1)))
    tn = float(np.sum((preds == 0) & (labels_arr == 0)))
    fp = float(np.sum((preds == 1) & (labels_arr == 0)))
    fn = float(np.sum((preds == 0) & (labels_arr == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = specificity_at_sensitivity(labels_arr, probs_arr)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else float("nan")

    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "specificity_at_sens95": float(specificity),
        "f1": float(f1),
        "accuracy": float(acc),
    }


__all__ = ["EpochStats", "compute_binary_metrics"]

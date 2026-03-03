from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


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
            return {k: float("nan") for k in ("loss", "auc", "sensitivity", "specificity", "accuracy")}
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

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(acc),
    }


__all__ = ["EpochStats", "compute_binary_metrics"]

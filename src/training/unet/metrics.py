# training/unet/metrics.py

from __future__ import annotations
import numpy as np
import os
import yaml
from typing import Dict, Any

def confusion(
    preds: np.ndarray,
    refs: np.ndarray,
    ignore_index: int = 255,
) -> tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN for the WATER class (label=1), ignoring 255.
    preds, refs: numpy arrays of shape (N, H, W) or (N,).
    """
    # flatten
    preds_f = preds.reshape(-1)
    refs_f = refs.reshape(-1)

    # ignore 255
    mask = refs_f != ignore_index
    preds_f = preds_f[mask]
    refs_f = refs_f[mask]

    # water = 1, land = 0
    tp = np.sum((preds_f == 1) & (refs_f == 1))
    fp = np.sum((preds_f == 1) & (refs_f == 0))
    fn = np.sum((preds_f == 0) & (refs_f == 1))
    tn = np.sum((preds_f == 0) & (refs_f == 0))
    return int(tp), int(fp), int(fn), int(tn)


def derived_stats(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    omission = fn / (tp + fn + eps)
    commission = fp / (tp + fp + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "omission": float(omission),
        "commission": float(commission),
    }

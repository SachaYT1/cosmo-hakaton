"""Pooled AF metrics and deterministic acquisition-date folds (statement p. 13)."""
from __future__ import annotations

import numpy as np


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def counts(prob: np.ndarray, truth: np.ndarray, threshold: float,
           missed: int = 0) -> np.ndarray:
    """Count every pixel: fire pixels rejected by the candidate filter are FN.

    Rejected background pixels are TN and do not enter the F1 formula.
    """
    pred, truth = np.asarray(prob) >= threshold, np.asarray(truth, dtype=bool)
    return np.array([(pred & truth).sum(), (pred & ~truth).sum(),
                     (~pred & truth).sum() + missed], dtype=np.int64)


def metrics(c: np.ndarray) -> dict:
    tp, fp, fn = map(int, c)
    den = 2*tp + fp + fn
    return {"TP": tp, "FP": fp, "FN": fn,
            "F1_af": 2*tp/den if den else 1.0,
            "precision": tp/(tp+fp) if tp+fp else 1.0,
            "recall": tp/(tp+fn) if tp+fn else 1.0}


def tune_threshold(prob: np.ndarray, truth: np.ndarray, missed: int = 0,
                   grid: np.ndarray = THRESHOLDS) -> tuple[float, dict]:
    """Choose on calibration/OOF data; report untouched holdout separately.

    Ties favour a threshold closest to 0.5, then the smaller threshold.
    """
    results = [(float(t), metrics(counts(prob, truth, float(t), missed))) for t in grid]
    return max(results, key=lambda r: (r[1]["F1_af"], -abs(r[0]-0.5), -r[0]))


def date_folds(dates: np.ndarray, folds: int = 5, seed: int = 42) -> np.ndarray:
    """All chips from a date stay together; return one fold per input chip."""
    dates = np.asarray(dates, dtype=str)
    groups = np.unique(dates)
    if not 2 <= folds <= len(groups):
        raise ValueError(f"Need 2 <= folds <= number of acquisition dates ({len(groups)})")
    assignment = dict(zip(groups, np.random.default_rng(seed).permutation(len(groups)) % folds))
    return np.array([assignment[d] for d in dates], dtype=np.int16)

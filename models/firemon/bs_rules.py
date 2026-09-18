"""Rule-based burn severity: 3x3 median-filtered dNBR thresholded per WorldCover class.

EDA showed the reference severity classes are (almost) hard thresholds of median-smoothed dNBR
whose cut-offs depend on land cover (USGS-like for tree, lower for grass/crop). Pixels whose
post (or pre) SCL is nodata / cloud shadow / high cloud are never labelled burned.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# SCL classes where the reference is always 0 (nodata, cloud shadow, high-probability cloud)
INVALID_SCL = (0, 3, 9)


def smooth_dnbr(dnbr: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(np.nan_to_num(dnbr).astype(np.float32), 3)


def invalid_mask(scl_pre: np.ndarray, scl_post: np.ndarray) -> np.ndarray:
    return np.isin(scl_post, INVALID_SCL) | np.isin(scl_pre, INVALID_SCL)


class Thresholds:
    """Per-landcover (t_burn, t_12, t_23) on median dNBR; 'default' used for unseen classes."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = {k: tuple(v) for k, v in table.items()}

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        return cls(json.load(open(path)))

    def save(self, path: str | Path) -> None:
        json.dump({k: list(v) for k, v in self.table.items()}, open(path, "w"), indent=1)

    def maps(self, lc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.table["default"]
        t = [np.full(lc.shape, d[i], np.float32) for i in range(3)]
        for k, v in self.table.items():
            if k == "default":
                continue
            sel = lc == int(k)
            for i in range(3):
                t[i][sel] = v[i]
        return t[0], t[1], t[2]

    def severity(self, med: np.ndarray, lc: np.ndarray) -> np.ndarray:
        """Severity 1..3 for every pixel (caller decides which pixels are burned)."""
        _, t1, t2 = self.maps(lc)
        return (1 + (med >= t1) + (med >= t2)).astype(np.uint8)

    def burn_candidates(self, med: np.ndarray, lc: np.ndarray) -> np.ndarray:
        t0, _, _ = self.maps(lc)
        return med >= t0


def rule_predict(med: np.ndarray, lc: np.ndarray, invalid: np.ndarray, th: Thresholds,
                 min_size: int = 0) -> np.ndarray:
    burn = th.burn_candidates(med, lc) & ~invalid
    if min_size:
        burn = remove_small(burn, min_size)
    return np.where(burn, th.severity(med, lc), 0).astype(np.uint8)


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False
    return keep[lab]

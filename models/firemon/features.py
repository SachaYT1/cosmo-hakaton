"""Spectral indices for Sentinel-2 L2A chips (reflectance scaled 0..10000)."""
from __future__ import annotations

import numpy as np

from .io import S2_BANDS

_I = {b: i for i, b in enumerate(S2_BANDS)}


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, (a - b) / s, 0.0).astype(np.float32)


def nbr(s2: np.ndarray) -> np.ndarray:
    """NBR = (B8A - B12) / (B8A + B12)."""
    return _nd(s2[..., _I["B8A"]], s2[..., _I["B12"]])


def nbr2(s2: np.ndarray) -> np.ndarray:
    return _nd(s2[..., _I["B11"]], s2[..., _I["B12"]])


def ndvi(s2: np.ndarray) -> np.ndarray:
    return _nd(s2[..., _I["B8A"]], s2[..., _I["B4"]])


def ndwi(s2: np.ndarray) -> np.ndarray:
    return _nd(s2[..., _I["B3"]], s2[..., _I["B8A"]])


def bai(s2: np.ndarray) -> np.ndarray:
    """Burned Area Index on reflectance 0..1."""
    r = s2[..., _I["B4"]].astype(np.float32) / 1e4
    n = s2[..., _I["B8A"]].astype(np.float32) / 1e4
    return (1.0 / ((0.1 - r) ** 2 + (0.06 - n) ** 2 + 1e-6)).astype(np.float32)


def rgb(s2: np.ndarray, scale: float = 3000.0) -> np.ndarray:
    return np.clip(s2[..., [_I["B4"], _I["B3"], _I["B2"]]].astype(np.float32) / scale, 0, 1)

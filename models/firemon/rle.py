"""RLE as specified in the case statement: pixels numbered row-major, 1-based;
"start length" pairs in increasing order, runs neither overlap nor touch."""
from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    if not flat.any():
        return ""
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])  # 0-based starts / ends (exclusive)
    starts, ends = edges[0::2], edges[1::2]
    return " ".join(f"{s + 1} {e - s}" for s, e in zip(starts, ends))


def decode(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros(h * w, dtype=bool)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or not str(rle).strip():
        return out.reshape(h, w)
    nums = np.asarray(str(rle).split(), dtype=np.int64)
    if nums.size % 2:
        raise ValueError("RLE must contain an even number of values")
    starts, lengths = nums[0::2] - 1, nums[1::2]
    if (starts < 0).any() or (lengths <= 0).any() or (starts + lengths > h * w).any():
        raise ValueError("RLE run out of chip bounds")
    if (np.diff(starts) <= 0).any() or (starts[1:] <= starts[:-1] + lengths[:-1]).any():
        raise ValueError("RLE runs must be increasing and must not overlap or touch")
    for s, n in zip(starts, lengths):
        out[s:s + n] = True
    return out.reshape(h, w)

"""Кодирование/декодирование масок в формат RLE, требуемый submission.csv.

Нумерация пикселей построчная, слева направо и сверху вниз, начиная с 1
(см. постановку кейса, раздел «Формат submission.csv»).
"""
from __future__ import annotations

import numpy as np


def mask_to_rle(mask: np.ndarray) -> str:
    """Бинарная маска (H, W) -> строка "start len start len ...", 1-индексация."""
    flat = mask.astype(np.uint8).ravel(order="C")
    if flat.sum() == 0:
        return ""
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1  # 1-индексация
    ends = np.where(diff == -1)[0] + 1
    lengths = ends - starts
    pairs = np.empty(starts.size * 2, dtype=np.int64)
    pairs[0::2] = starts
    pairs[1::2] = lengths
    return " ".join(map(str, pairs))


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """Обратное преобразование: строка RLE -> маска (H, W) из 0/1."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle:
        return mask.reshape(height, width)
    values = list(map(int, rle.split()))
    for start, length in zip(values[0::2], values[1::2]):
        mask[start - 1 : start - 1 + length] = 1
    return mask.reshape(height, width)

"""Спектральные индексы для оценки гарей: NBR, dNBR, RdNBR."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def nbr(b8a: np.ndarray, b12: np.ndarray) -> np.ndarray:
    return (b8a - b12) / (b8a + b12 + EPS)


def dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    return nbr_pre - nbr_post


def rdnbr(dnbr_arr: np.ndarray, nbr_pre: np.ndarray) -> np.ndarray:
    """dNBR / sqrt(|NBR(до)|) — снижает зависимость от исходной биомассы."""
    return dnbr_arr / np.sqrt(np.abs(nbr_pre) + EPS)

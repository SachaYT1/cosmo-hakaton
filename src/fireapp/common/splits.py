"""Групповой train/val split — чтобы чипы одной группы (пожар/гео-плитка)
не утекали между train и val одновременно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_split(
    meta: pd.DataFrame,
    group_cols: list[str],
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (train_idx, val_idx) — позиционные индексы в meta.

    Делит на train/val не отдельные строки, а целые группы по group_cols,
    чтобы одна и та же геометрия/пожар не оказались одновременно в обеих частях.
    """
    groups = meta[group_cols].astype(str).agg("_".join, axis=1)
    unique_groups = groups.unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_val_groups = max(1, int(len(unique_groups) * val_frac))
    val_groups = set(unique_groups[:n_val_groups])

    is_val = groups.isin(val_groups).to_numpy()
    val_idx = np.where(is_val)[0]
    train_idx = np.where(~is_val)[0]
    return train_idx, val_idx

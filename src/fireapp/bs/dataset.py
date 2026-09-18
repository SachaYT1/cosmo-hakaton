"""Датасет BS-чипов.

Реальная структура на диске (train/bs/):
    sentinel2_pre/{chip_id}_Sentinel-2_pre.tif    10 каналов: B2,B3,B4,B5,B6,B7,B8A,B11,B12,SCL
    sentinel2_post/{chip_id}_Sentinel-2_post.tif  те же 10 каналов, дата "после"
    sentinel1_pre/{chip_id}_Sentinel-1_pre.tif    2 канала: VV, VH (int16, x100 -> дБ)
    sentinel1_post/{chip_id}_Sentinel-1_post.tif  2 канала: VV, VH
    aux/{chip_id}_AUX.tif                         4 канала: dem, slope, aspect, landcover
    masks/{chip_id}_MASK.tif                      1 канал, 0..3 — только для train

Отдаёт pre (13, H, W), post (13, H, W) — S2(10)+S1(2)+NBR(1) на каждую дату —
и static (4, H, W), ровно под UNetSiamese(branch_channels=13, static_channels=4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fireapp.bs.indices import nbr
from fireapp.common.io_raster import read_chip

_S2_B8A_IDX = 6
_S2_B12_IDX = 8
_S2_SCL_IDX = 9

# SCL-коды, которые считаем невалидными для обучения/метрики
# (0 no data, 1 saturated, 3 cloud shadow, 8/9 cloud medium/high, 10 cirrus)
_SCL_INVALID = {0, 1, 3, 8, 9, 10}


class BSDataset(Dataset):
    def __init__(self, data_dir: str | Path, meta: pd.DataFrame, train: bool = True):
        self.data_dir = Path(data_dir)
        self.meta = meta.reset_index(drop=True)
        self.train = train

    def __len__(self) -> int:
        return len(self.meta)

    def _load_s2(self, chip_id: str, suffix: str) -> np.ndarray:
        return read_chip(self.data_dir / f"sentinel2_{suffix}" / f"{chip_id}_Sentinel-2_{suffix}.tif")

    def _load_s1(self, chip_id: str, suffix: str) -> np.ndarray:
        return read_chip(self.data_dir / f"sentinel1_{suffix}" / f"{chip_id}_Sentinel-1_{suffix}.tif")

    def _load_aux(self, chip_id: str) -> np.ndarray:
        return read_chip(self.data_dir / "aux" / f"{chip_id}_AUX.tif").astype(np.float32)

    def _load_mask(self, chip_id: str) -> np.ndarray:
        mask = read_chip(self.data_dir / "masks" / f"{chip_id}_MASK.tif")
        return mask[0].astype(np.int64)  # (H, W), значения 0..3

    def _build_branch(self, chip_id: str, suffix: str) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (branch(13,H,W), valid_mask(H,W)) для одной даты."""
        s2_raw = self._load_s2(chip_id, suffix)
        scl = s2_raw[_S2_SCL_IDX]
        valid = ~np.isin(scl, list(_SCL_INVALID))

        s2 = s2_raw.astype(np.float32)
        s2[:_S2_SCL_IDX] = s2[:_S2_SCL_IDX] / 10000.0  # отражение -> [0,1]
        s2[_S2_SCL_IDX] = scl / 11.0  # SCL как слабый категориальный признак

        s1 = self._load_s1(chip_id, suffix).astype(np.float32) / 100.0  # -> дБ

        b8a = s2[_S2_B8A_IDX]
        b12 = s2[_S2_B12_IDX]
        chip_nbr = nbr(b8a, b12)[np.newaxis, ...]

        branch = np.concatenate([s2, s1, chip_nbr], axis=0)  # 10 + 2 + 1 = 13
        return branch, valid

    @staticmethod
    def _normalize_static(aux: np.ndarray) -> np.ndarray:
        aux = aux.copy()
        aux[0] = aux[0] / 1000.0  # dem
        aux[1] = aux[1] / 90.0  # slope
        aux[2] = aux[2] / 360.0  # aspect
        # landcover (idx 3) — категориальный код, оставляем как есть
        return aux

    def _augment(self, pre, post, static, valid, y):
        if np.random.rand() < 0.5:
            pre, post, static = pre[:, :, ::-1].copy(), post[:, :, ::-1].copy(), static[:, :, ::-1].copy()
            valid = valid[:, ::-1].copy()
            if y is not None:
                y = y[:, ::-1].copy()
        if np.random.rand() < 0.5:
            pre, post, static = pre[:, ::-1, :].copy(), post[:, ::-1, :].copy(), static[:, ::-1, :].copy()
            valid = valid[::-1, :].copy()
            if y is not None:
                y = y[::-1, :].copy()
        k = np.random.randint(4)
        if k:
            pre = np.rot90(pre, k, axes=(1, 2)).copy()
            post = np.rot90(post, k, axes=(1, 2)).copy()
            static = np.rot90(static, k, axes=(1, 2)).copy()
            valid = np.rot90(valid, k, axes=(0, 1)).copy()
            if y is not None:
                y = np.rot90(y, k, axes=(0, 1)).copy()
        return pre, post, static, valid, y

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        chip_id = row["chip_id"]

        pre, valid_pre = self._build_branch(chip_id, "pre")
        post, valid_post = self._build_branch(chip_id, "post")
        static = self._normalize_static(self._load_aux(chip_id))
        valid = (valid_pre & valid_post).astype(np.float32)  # (H, W)

        y = self._load_mask(chip_id) if self.train else None

        if self.train:
            pre, post, static, valid, y = self._augment(pre, post, static, valid, y)

        item = {
            "chip_id": chip_id,
            "pre": torch.from_numpy(pre),
            "post": torch.from_numpy(post),
            "static": torch.from_numpy(static),
            "valid": torch.from_numpy(valid),
        }
        if y is not None:
            item["y"] = torch.from_numpy(y)  # (H, W), long
        return item

"""Датасет AF-чипов.

Реальная структура на диске (train/af/):
    viirs/{chip_id}_VIIRS_I1-I5.tif   5 каналов, I1..I5 (отражение/яркостная T)
    aux/{chip_id}_AUX.tif             8 каналов, порядок фиксирован (см. _AUX_ORDER)
    masks/{chip_id}_MASK.tif          1 канал, 0/1 — только для train

Итоговый тензор x: (13, H, W) = VIIRS(5) + AUX(8), ровно под UNetSmall(in_channels=13).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fireapp.common.io_raster import read_chip

# порядок каналов AUX как в постановке кейса (раздел "Каналы AF-чипа")
_AUX_ORDER = [
    "landcover",
    "dem",
    "sun_zenith",
    "sensor_zenith",
    "t2m",
    "rh2m",
    "wind",
    "valid_mask",
]

N_VIIRS_CHANNELS = 5
# индекс канала valid_mask в итоговом тензоре x (после конкатенации VIIRS+AUX)
VALID_MASK_CHANNEL = N_VIIRS_CHANNELS + _AUX_ORDER.index("valid_mask")


class AFDataset(Dataset):
    def __init__(self, data_dir: str | Path, meta: pd.DataFrame, train: bool = True):
        """
        data_dir — каталог train/af (или test/af), содержащий viirs/, aux/, masks/.
        meta — строки meta.csv для этого набора (kind == 'af').
        train — если True, дополнительно читает маску и включает аугментации.
        """
        self.data_dir = Path(data_dir)
        self.meta = meta.reset_index(drop=True)
        self.train = train

    def __len__(self) -> int:
        return len(self.meta)

    def _load_raw(self, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
        viirs = read_chip(self.data_dir / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif").astype(np.float32)
        aux = read_chip(self.data_dir / "aux" / f"{chip_id}_AUX.tif").astype(np.float32)
        return viirs, aux

    def _load_mask(self, chip_id: str) -> np.ndarray:
        mask = read_chip(self.data_dir / "masks" / f"{chip_id}_MASK.tif")
        return mask[0].astype(np.float32)  # (H, W), 0/1

    @staticmethod
    def _normalize(viirs: np.ndarray, aux: np.ndarray) -> np.ndarray:
        # I1-I3: отражение [0,1] — как есть. I4/I5: яркостная T в K — центрируем вокруг фона ~300K.
        viirs = viirs.copy()
        viirs[3] = (viirs[3] - 300.0) / 50.0  # I4
        viirs[4] = (viirs[4] - 300.0) / 30.0  # I5

        aux = aux.copy()
        idx = {name: i for i, name in enumerate(_AUX_ORDER)}
        aux[idx["dem"]] = aux[idx["dem"]] / 1000.0
        aux[idx["sun_zenith"]] = aux[idx["sun_zenith"]] / 90.0
        aux[idx["sensor_zenith"]] = aux[idx["sensor_zenith"]] / 90.0
        aux[idx["t2m"]] = (aux[idx["t2m"]] - 290.0) / 20.0
        aux[idx["rh2m"]] = aux[idx["rh2m"]] / 100.0
        aux[idx["wind"]] = aux[idx["wind"]] / 15.0
        # landcover — категориальный код; valid_mask уже в {0,1} — не трогаем
        return np.concatenate([viirs, aux], axis=0)

    def _augment(self, x: np.ndarray, y: np.ndarray | None):
        if np.random.rand() < 0.5:
            x, y = x[:, :, ::-1].copy(), (None if y is None else y[:, ::-1].copy())
        if np.random.rand() < 0.5:
            x, y = x[:, ::-1, :].copy(), (None if y is None else y[::-1, :].copy())
        k = np.random.randint(4)
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
            if y is not None:
                y = np.rot90(y, k, axes=(0, 1)).copy()
        return x, y

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        chip_id = row["chip_id"]

        viirs, aux = self._load_raw(chip_id)
        x = self._normalize(viirs, aux)

        y = self._load_mask(chip_id) if self.train else None

        if self.train:
            x, y = self._augment(x, y)

        item = {"chip_id": chip_id, "x": torch.from_numpy(x)}
        if y is not None:
            item["y"] = torch.from_numpy(y).unsqueeze(0)  # (1, H, W)
        return item

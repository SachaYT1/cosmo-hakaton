"""Чтение чипов (GeoTIFF) и вспомогательных слоёв в numpy-массивы."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def read_chip(path: str | Path) -> np.ndarray:
    """Читает GeoTIFF в массив формы (C, H, W)."""
    with rasterio.open(path) as src:
        return src.read()


def read_chip_meta(path: str | Path) -> dict:
    """Возвращает geotransform/crs/nodata чипа — нужно для сервиса (векторизация в реальных координатах)."""
    with rasterio.open(path) as src:
        return {
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
        }

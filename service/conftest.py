"""Синтетический мини-датасет в формате выданных данных (1 AF-чип + 1 BS-чип)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

AF_BOUNDS = (384000.0, 5280000.0, 384000.0 + 8 * 375.0, 5280000.0 + 8 * 375.0)
BS_BOUNDS = (491520.0, 5376000.0, 491520.0 + 16 * 20.0, 5376000.0 + 16 * 20.0)

AF_META = {
    "chip_id": "AF_ts_000001", "kind": "af", "fire_event_id": "", "region": "",
    "epsg": 32637.0,
    "x_min": AF_BOUNDS[0], "y_min": AF_BOUNDS[1], "x_max": AF_BOUNDS[2], "y_max": AF_BOUNDS[3],
    "width": 8, "height": 8, "gsd": 375,
    "acq_datetime": "2021-07-15T10:00:00+00:00", "satellite": "SNPP",
    "date_pre": "", "date_post": "", "s1_date_pre": "", "s1_date_post": "",
    "valid_frac": 1.0, "cloud_frac": "", "landcover_top": "",
    "n_fire_px": 3, "burn_area_ha": "", "sev1_px": "", "sev2_px": "", "sev3_px": "",
}

BS_META = {
    "chip_id": "BS_ts_000001", "kind": "bs", "fire_event_id": "FE00001", "region": "",
    "epsg": 32638.0,
    "x_min": BS_BOUNDS[0], "y_min": BS_BOUNDS[1], "x_max": BS_BOUNDS[2], "y_max": BS_BOUNDS[3],
    "width": 16, "height": 16, "gsd": 20,
    "acq_datetime": "", "satellite": "",
    "date_pre": "2021-07-01", "date_post": "2021-07-20",
    "s1_date_pre": "2021-06-30", "s1_date_post": "2021-07-21",
    "valid_frac": 1.0, "cloud_frac": 0.0, "landcover_top": "",
    "n_fire_px": "", "burn_area_ha": 0.64, "sev1_px": 10, "sev2_px": 5, "sev3_px": 1,
}


def af_mask() -> np.ndarray:
    m = np.zeros((8, 8), dtype=np.uint8)
    m[0, 0] = m[3, 4] = m[7, 7] = 1
    return m


def bs_mask() -> np.ndarray:
    m = np.zeros((16, 16), dtype=np.uint8)
    m[0:2, 0:5] = 1   # 10 px -> 0.4 га (слабая)
    m[5, 5:10] = 2    # 5 px  -> 0.2 га (средняя)
    m[10, 10] = 3     # 1 px  -> 0.04 га (сильная)
    return m


def _write_mask(path: Path, arr: np.ndarray, epsg: int, bounds: tuple) -> None:
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
        crs=f"EPSG:{epsg}", transform=from_bounds(*bounds, w, h),
    ) as dst:
        dst.write(arr, 1)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    (root / "af" / "masks").mkdir(parents=True)
    (root / "bs" / "masks").mkdir(parents=True)
    pd.DataFrame([AF_META]).to_csv(root / "af" / "meta.csv", index=False)
    pd.DataFrame([BS_META]).to_csv(root / "bs" / "meta.csv", index=False)
    _write_mask(root / "af" / "masks" / "AF_ts_000001_MASK.tif", af_mask(), 32637, AF_BOUNDS)
    _write_mask(root / "bs" / "masks" / "BS_ts_000001_MASK.tif", bs_mask(), 32638, BS_BOUNDS)
    return root


@pytest.fixture(scope="session")
def catalog_path(dataset_dir, tmp_path_factory) -> Path:
    from app.ingest import build_catalog

    out = tmp_path_factory.mktemp("catalog") / "catalog.gpkg"
    build_catalog(dataset_dir, out, source="gt")
    return out

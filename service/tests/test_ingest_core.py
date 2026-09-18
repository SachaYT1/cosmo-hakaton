import numpy as np
import pandas as pd
import pytest

from app.ingest_core import contours_gdf, grid_from_meta, hotspots_gdf

AF_ROW = pd.Series(
    {
        "chip_id": "AF_ts_000001",
        "epsg": 32637.0,
        "x_min": 384000.0, "y_min": 5280000.0,
        "x_max": 384000.0 + 8 * 375.0, "y_max": 5280000.0 + 8 * 375.0,
        "width": 8, "height": 8,
        "acq_datetime": "2021-07-15T10:00:00+00:00",
        "satellite": "SNPP",
    }
)

BS_ROW = pd.Series(
    {
        "chip_id": "BS_ts_000001",
        "fire_event_id": "FE00001",
        "epsg": 32638.0,
        "x_min": 491520.0, "y_min": 5376000.0,
        "x_max": 491520.0 + 16 * 20.0, "y_max": 5376000.0 + 16 * 20.0,
        "width": 16, "height": 16,
        "date_pre": "2021-07-01", "date_post": "2021-07-20",
    }
)


def _bs_mask():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[0:2, 0:5] = 1
    m[5, 5:10] = 2
    m[10, 10] = 3
    return m


def test_grid_from_meta_casts_types():
    grid = grid_from_meta(AF_ROW)
    assert grid.epsg == 32637 and grid.width == 8


def test_hotspots_gdf():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = mask[3, 4] = mask[7, 7] = 1
    gdf = hotspots_gdf(AF_ROW, mask)
    assert len(gdf) == 3
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf["chip_id"]) == {"AF_ts_000001"}
    assert set(gdf.columns) == {"chip_id", "acq_datetime", "satellite", "geometry"}


def test_hotspots_gdf_empty_mask():
    gdf = hotspots_gdf(AF_ROW, np.zeros((8, 8), dtype=np.uint8))
    assert len(gdf) == 0


def test_contours_gdf():
    gdf = contours_gdf(BS_ROW, _bs_mask())
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf.columns) == {
        "contour_id", "chip_id", "fire_event_id", "severity", "area_ha",
        "date_pre", "date_post", "epsg", "geometry",
    }
    assert gdf["contour_id"].is_unique
    assert gdf.groupby("severity")["area_ha"].sum().to_dict() == pytest.approx(
        {1: 0.4, 2: 0.2, 3: 0.04}
    )
    assert set(gdf["epsg"]) == {32638}

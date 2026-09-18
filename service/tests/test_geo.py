import numpy as np
import pytest
from pyproj import Transformer

from app.geo import ChipGrid, hotspot_points, severity_polygons

GRID = ChipGrid(
    epsg=32637,
    x_min=384000.0, y_min=5280000.0,
    x_max=384000.0 + 8 * 375.0, y_max=5280000.0 + 8 * 375.0,
    width=8, height=8,
)


def test_pixel_area_ha():
    assert GRID.pixel_area_ha == pytest.approx(375 * 375 / 10_000)  # 14.0625 га
    bs = ChipGrid(32638, 0, 0, 16 * 20.0, 16 * 20.0, 16, 16)
    assert bs.pixel_area_ha == pytest.approx(0.04)


def test_hotspot_points_centers():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1  # верхний левый пиксель: центр (x_min + 187.5, y_max - 187.5)
    pts = hotspot_points(mask, GRID)
    assert len(pts) == 1
    t = Transformer.from_crs("EPSG:32637", "EPSG:4326", always_xy=True)
    exp_lon, exp_lat = t.transform(GRID.x_min + 187.5, GRID.y_max - 187.5)
    assert pts[0][0] == pytest.approx(exp_lon, abs=1e-9)
    assert pts[0][1] == pytest.approx(exp_lat, abs=1e-9)


def test_hotspot_points_empty():
    assert hotspot_points(np.zeros((8, 8), dtype=np.uint8), GRID) == []


def test_severity_polygons_areas_in_utm():
    grid = ChipGrid(32638, 491520.0, 5376000.0, 491520.0 + 16 * 20.0, 5376000.0 + 16 * 20.0, 16, 16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:2, 0:5] = 1   # 10 px = 0.4 га
    mask[5, 5:10] = 2    # 5 px = 0.2 га
    mask[10, 10] = 3     # 1 px = 0.04 га
    polys = severity_polygons(mask, grid)
    by_sev = {}
    for p in polys:
        by_sev[p["severity"]] = by_sev.get(p["severity"], 0.0) + p["area_ha"]
    assert by_sev[1] == pytest.approx(0.4, abs=1e-6)
    assert by_sev[2] == pytest.approx(0.2, abs=1e-6)
    assert by_sev[3] == pytest.approx(0.04, abs=1e-6)
    # геометрия в WGS84: долгота в разумных пределах зоны 38N
    lon = polys[0]["geometry"].centroid.x
    assert 42.0 < lon < 48.0

import pandas as pd
import pytest
from shapely.geometry import box

from app.reports import build_report, clipped_area_by_severity
from app.store import Catalog

WIDE_BOX = box(30.0, 40.0, 60.0, 60.0)  # покрывает оба тестовых чипа
D0 = pd.Timestamp("2021-01-01")
D1 = pd.Timestamp("2021-12-31")


@pytest.fixture(scope="module")
def catalog(catalog_path) -> Catalog:
    return Catalog.load(catalog_path)


def test_query_hotspots_spatial_and_dates(catalog):
    assert len(catalog.query_hotspots(WIDE_BOX, D0, D1)) == 3
    # период без данных
    assert len(catalog.query_hotspots(WIDE_BOX, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))) == 0
    # полигон мимо данных
    assert len(catalog.query_hotspots(box(0, 0, 1, 1), D0, D1)) == 0


def test_query_contours(catalog):
    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    assert len(contours) >= 3
    assert contours["area_ha"].sum() == pytest.approx(0.64)
    # date_post=2021-07-20 вне интервала -> пусто
    assert len(catalog.query_contours(WIDE_BOX, D0, pd.Timestamp("2021-07-10"))) == 0


def test_clipped_area_full_cover(catalog):
    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    areas = clipped_area_by_severity(contours, WIDE_BOX)
    assert areas[1] == pytest.approx(0.4, abs=1e-3)
    assert areas[2] == pytest.approx(0.2, abs=1e-3)
    assert areas[3] == pytest.approx(0.04, abs=1e-3)


def test_clipped_area_partial_cover(catalog):
    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    sev1 = contours[contours["severity"] == 1]
    minx, miny, maxx, maxy = sev1.total_bounds
    west_half = box(minx, miny, (minx + maxx) / 2, maxy)
    areas = clipped_area_by_severity(contours, west_half)
    assert areas[1] == pytest.approx(0.2, abs=0.02)  # половина от 0.4 га

    clipped = catalog.query_contours(west_half, D0, D1)
    assert clipped["area_ha"].sum() < contours["area_ha"].sum()
    assert clipped.total_bounds[2] <= west_half.bounds[2] + 1e-8


def test_build_report(catalog):
    from datetime import date

    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    hotspots = catalog.query_hotspots(WIDE_BOX, D0, D1)
    report = build_report(contours, hotspots, WIDE_BOX, date(2021, 1, 1), date(2021, 12, 31))
    assert report["total_burned_area_ha"] == pytest.approx(0.64, abs=1e-2)
    assert report["hotspot_count"] == 3
    sev1 = next(r for r in report["by_severity"] if r["severity"] == 1)
    assert sev1["label"] == "слабая"
    assert sev1["share_pct"] == pytest.approx(62.5, abs=0.1)


def test_empty_report(catalog):
    from datetime import date

    empty_geom = box(0, 0, 1, 1)
    contours = catalog.query_contours(empty_geom, D0, D1)
    hotspots = catalog.query_hotspots(empty_geom, D0, D1)
    report = build_report(contours, hotspots, empty_geom, date(2021, 1, 1), date(2021, 12, 31))
    assert report["total_burned_area_ha"] == 0.0
    assert report["hotspot_count"] == 0


def test_meta(catalog):
    meta = catalog.meta()
    assert meta["hotspot_count"] == 3
    assert meta["date_min"] == "2021-07-15"
    assert meta["date_max"] == "2021-07-20"
    assert meta["demo"]["fire_event_id"] == "FE00001"
    assert len(meta["demo"]["bbox"]) == 4


def test_missing_catalog_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="catalog not found"):
        Catalog.load(tmp_path / "missing.gpkg")

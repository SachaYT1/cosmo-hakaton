import geopandas as gpd
import pytest

from app.ingest import build_catalog


def test_build_catalog_gt(dataset_dir, tmp_path):
    out = tmp_path / "catalog.gpkg"
    build_catalog(dataset_dir, out, source="gt")
    assert out.exists()

    hotspots = gpd.read_file(out, layer="hotspots")
    assert len(hotspots) == 3
    assert hotspots.crs.to_epsg() == 4326
    assert {"chip_id", "acq_datetime", "satellite"} <= set(hotspots.columns)

    contours = gpd.read_file(out, layer="burn_contours")
    assert contours.crs.to_epsg() == 4326
    sums = contours.groupby("severity")["area_ha"].sum().to_dict()
    assert sums == pytest.approx({1: 0.4, 2: 0.2, 3: 0.04})
    # согласованность с meta.csv: burn_area_ha = 0.64
    assert contours["area_ha"].sum() == pytest.approx(0.64)


def test_build_catalog_overwrites(dataset_dir, tmp_path):
    out = tmp_path / "catalog.gpkg"
    build_catalog(dataset_dir, out, source="gt")
    build_catalog(dataset_dir, out, source="gt")  # повторный запуск не падает и не дублирует
    hotspots = gpd.read_file(out, layer="hotspots")
    assert len(hotspots) == 3

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

QUERY = {"bbox": [30.0, 40.0, 60.0, 60.0], "date_from": "2021-01-01", "date_to": "2021-12-31"}
EMPTY_QUERY = {"bbox": [0.0, 0.0, 1.0, 1.0], "date_from": "2021-01-01", "date_to": "2021-12-31"}


@pytest.fixture()
def client(catalog_path) -> TestClient:
    from app.main import create_app

    return TestClient(create_app(catalog_path))


def test_export_geojson(client):
    r = client.post("/api/export/burned-areas.geojson", json=QUERY)
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    fc = json.loads(r.content)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 3


def test_export_geojson_empty_selection(client):
    r = client.post("/api/export/burned-areas.geojson", json=EMPTY_QUERY)
    assert r.status_code == 200
    assert json.loads(r.content)["features"] == []


def test_export_shapefile_zip(client):
    r = client.post("/api/export/burned-areas.shp.zip", json=QUERY)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"burned_areas.shp", "burned_areas.shx", "burned_areas.dbf", "burned_areas.prj"} <= names


def test_export_shapefile_empty_selection_404(client):
    r = client.post("/api/export/burned-areas.shp.zip", json=EMPTY_QUERY)
    assert r.status_code == 404


def test_export_report_json(client):
    r = client.post("/api/export/report.json", json=QUERY)
    assert r.status_code == 200
    rep = json.loads(r.content)
    assert rep["total_burned_area_ha"] == pytest.approx(0.64, abs=0.01)


def test_export_report_csv(client):
    r = client.post("/api/export/report.csv", json=QUERY)
    assert r.status_code == 200
    text = r.content.decode("utf-8-sig")
    assert text.splitlines()[0] == "severity,label,area_ha,share_pct"
    assert "total" in text

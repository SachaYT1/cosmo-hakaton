import pytest
from fastapi.testclient import TestClient

QUERY = {"bbox": [30.0, 40.0, 60.0, 60.0], "date_from": "2021-01-01", "date_to": "2021-12-31"}
POLYGON_QUERY = {
    "polygon": {
        "type": "Polygon",
        "coordinates": [[[30, 40], [60, 40], [60, 60], [30, 60], [30, 40]]],
    },
    "date_from": "2021-01-01",
    "date_to": "2021-12-31",
}


@pytest.fixture()
def client(catalog_path) -> TestClient:
    from app.main import create_app

    return TestClient(create_app(catalog_path))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_hotspots_bbox(client):
    r = client.post("/api/hotspots", json=QUERY)
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3
    props = fc["features"][0]["properties"]
    assert {"chip_id", "acq_datetime", "satellite"} <= set(props)


def test_hotspots_polygon(client):
    r = client.post("/api/hotspots", json=POLYGON_QUERY)
    assert len(r.json()["features"]) == 3


def test_hotspots_date_filter(client):
    r = client.post("/api/hotspots", json={**QUERY, "date_from": "2020-01-01", "date_to": "2020-12-31"})
    assert r.json()["features"] == []


def test_burned_areas(client):
    r = client.post("/api/burned-areas", json=QUERY)
    assert r.status_code == 200
    feats = r.json()["features"]
    assert len(feats) >= 3
    total = sum(f["properties"]["area_ha"] for f in feats)
    assert total == pytest.approx(0.64, abs=1e-3)
    assert {f["properties"]["severity"] for f in feats} == {1, 2, 3}


def test_report(client):
    r = client.post("/api/report", json=QUERY)
    assert r.status_code == 200
    rep = r.json()
    assert rep["total_burned_area_ha"] == pytest.approx(0.64, abs=0.01)
    assert rep["hotspot_count"] == 3
    assert len(rep["by_severity"]) == 3


def test_meta(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    m = r.json()
    assert m["hotspot_count"] == 3
    assert m["demo"]["fire_event_id"] == "FE00001"


def test_both_polygon_and_bbox_rejected(client):
    r = client.post("/api/hotspots", json={**QUERY, "polygon": POLYGON_QUERY["polygon"]})
    assert r.status_code == 422


def test_neither_polygon_nor_bbox_rejected(client):
    r = client.post("/api/hotspots", json={"date_from": "2021-01-01", "date_to": "2021-12-31"})
    assert r.status_code == 422


def test_invalid_geometry_rejected(client):
    r = client.post("/api/hotspots", json={**POLYGON_QUERY, "polygon": {"type": "Polygon", "coordinates": []}})
    assert r.status_code == 422


def test_dates_swapped_rejected(client):
    r = client.post("/api/hotspots", json={**QUERY, "date_from": "2021-12-31", "date_to": "2021-01-01"})
    assert r.status_code == 422


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="map"' in r.text

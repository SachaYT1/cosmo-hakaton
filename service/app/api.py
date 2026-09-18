"""REST API: термоточки, контуры гарей, аналитическая справка, выгрузки."""

import json

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from app.exports import (
    contours_geojson_bytes,
    contours_shapefile_zip,
    report_csv_bytes,
    report_json_bytes,
)
from app.reports import build_report
from app.schemas import SpatioTemporalQuery
from app.store import Catalog

router = APIRouter(prefix="/api")


def healthz() -> dict:
    return {"status": "ok"}


def _catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def resolve_geometry(q: SpatioTemporalQuery) -> BaseGeometry:
    if q.bbox is not None:
        geom = box(*q.bbox)
    else:
        try:
            geom = shape(q.polygon)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"невалидная геометрия: {exc}") from exc
    if geom.is_empty or not geom.is_valid:
        raise HTTPException(status_code=422, detail="невалидная или пустая геометрия запроса")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise HTTPException(status_code=422, detail="ожидается Polygon или MultiPolygon")
    return geom


def _bounds(q: SpatioTemporalQuery) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(q.date_from), pd.Timestamp(q.date_to)


def feature_collection(gdf: gpd.GeoDataFrame) -> dict:
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(gdf.to_json(drop_id=True))


@router.post("/hotspots")
def hotspots(q: SpatioTemporalQuery, request: Request) -> dict:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    return feature_collection(_catalog(request).query_hotspots(geom, d0, d1))


@router.post("/burned-areas")
def burned_areas(q: SpatioTemporalQuery, request: Request) -> dict:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    return feature_collection(_catalog(request).query_contours(geom, d0, d1))


@router.post("/report")
def report(q: SpatioTemporalQuery, request: Request) -> dict:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    cat = _catalog(request)
    contours = cat.query_contours(geom, d0, d1)
    hs = cat.query_hotspots(geom, d0, d1)
    return build_report(contours, hs, geom, q.date_from, q.date_to)


@router.get("/meta")
def meta(request: Request) -> dict:
    return _catalog(request).meta()


def _attachment(content: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/export/burned-areas.geojson")
def export_geojson(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    gdf = _catalog(request).query_contours(geom, d0, d1)
    return _attachment(contours_geojson_bytes(gdf), "application/geo+json", "burned_areas.geojson")


@router.post("/export/burned-areas.shp.zip")
def export_shapefile(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    gdf = _catalog(request).query_contours(geom, d0, d1)
    if gdf.empty:
        raise HTTPException(status_code=404, detail="в выборке нет контуров гарей — нечего выгружать")
    return _attachment(contours_shapefile_zip(gdf), "application/zip", "burned_areas_shp.zip")


@router.post("/export/report.json")
def export_report_json(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    cat = _catalog(request)
    rep = build_report(
        cat.query_contours(geom, d0, d1), cat.query_hotspots(geom, d0, d1), geom, q.date_from, q.date_to
    )
    return _attachment(report_json_bytes(rep), "application/json", "report.json")


@router.post("/export/report.csv")
def export_report_csv(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    cat = _catalog(request)
    rep = build_report(
        cat.query_contours(geom, d0, d1), cat.query_hotspots(geom, d0, d1), geom, q.date_from, q.date_to
    )
    return _attachment(report_csv_bytes(rep), "text/csv; charset=utf-8", "report.csv")

"""REST API: термоточки, контуры гарей, аналитическая справка, выгрузки."""

import json

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

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

"""GET /api/burns — векторные контуры гарей (GeoJSON) с выгрузкой в Shapefile."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fireapp.service.db import get_session

router = APIRouter(prefix="/api/burns", tags=["burns"])


@router.post("")
def get_burns(query: dict, session: Session = Depends(get_session)) -> dict:
    raise NotImplementedError("TODO: пространственно-временной запрос -> GeoJSON FeatureCollection (id, severity, area_ha)")


@router.post("/export")
def export_burns(query: dict, fmt: str = "geojson", session: Session = Depends(get_session)):
    raise NotImplementedError("TODO: fmt in {geojson, shapefile} -> файл на выгрузку")

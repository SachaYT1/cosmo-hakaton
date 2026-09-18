"""GET /api/hotspots — карта термоточек за период по полигону/bbox."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fireapp.service.db import get_session

router = APIRouter(prefix="/api/hotspots", tags=["hotspots"])


@router.post("")
def get_hotspots(query: dict, session: Session = Depends(get_session)) -> dict:
    raise NotImplementedError("TODO: ST_Intersects(geom, :aoi) AND acq_datetime BETWEEN :from AND :to -> GeoJSON FeatureCollection")

"""Pydantic-схемы запросов/ответов REST API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class AoiQuery(BaseModel):
    polygon_geojson: dict  # полигон или bbox в GeoJSON
    date_from: date
    date_to: date


class SeverityBreakdown(BaseModel):
    sev1_ha: float
    sev2_ha: float
    sev3_ha: float
    total_ha: float


class AnalyticsReport(BaseModel):
    query: AoiQuery
    burn_area: SeverityBreakdown
    hotspot_count: int

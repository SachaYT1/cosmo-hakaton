"""GET /api/report — аналитическая справка: площадь гари и распределение по степеням."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fireapp.service.db import get_session
from fireapp.service.schemas import AnalyticsReport, AoiQuery

router = APIRouter(prefix="/api/report", tags=["report"])


@router.post("", response_model=AnalyticsReport)
def get_report(query: AoiQuery, session: Session = Depends(get_session)) -> AnalyticsReport:
    raise NotImplementedError("TODO: агрегировать area_ha по severity внутри AOI и периода")

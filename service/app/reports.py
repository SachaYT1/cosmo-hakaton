"""Аналитическая справка: площади пересечения с полигоном запроса, счёт в UTM."""

from datetime import date

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from app.store import clip_contours

SEVERITY_LABELS = {1: "слабая", 2: "средняя", 3: "сильная"}


def clipped_area_by_severity(contours: gpd.GeoDataFrame, query_geom: BaseGeometry) -> dict[int, float]:
    """Площадь пересечения контуров с полигоном запроса, по степеням, в гектарах.

    Пересечение считается в EPSG:4326, затем каждая группа контуров
    репроецируется в родную UTM-зону чипа (атрибут epsg) — там площадь точна.
    """
    areas = {1: 0.0, 2: 0.0, 3: 0.0}
    if contours.empty:
        return areas
    clipped = clip_contours(contours, query_geom)
    for sev, group in clipped.groupby("severity"):
        areas[int(sev)] += float(group["area_ha"].sum())
    return areas


def build_report(
    contours: gpd.GeoDataFrame,
    hotspots: gpd.GeoDataFrame,
    query_geom: BaseGeometry,
    date_from: date,
    date_to: date,
) -> dict:
    areas = clipped_area_by_severity(contours, query_geom)
    total = sum(areas.values())
    by_severity = [
        {
            "severity": sev,
            "label": SEVERITY_LABELS[sev],
            "area_ha": round(area, 2),
            "share_pct": round(area / total * 100.0, 1) if total else 0.0,
        }
        for sev, area in sorted(areas.items())
    ]
    return {
        "period": {"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
        "total_burned_area_ha": round(total, 2),
        "by_severity": by_severity,
        "hotspot_count": int(len(hotspots)),
        "contour_count": int(len(contours)),
        "note": "Площади — пересечение контуров с полигоном запроса, расчёт в UTM-зоне чипа. "
                "Атрибут area_ha в выгрузках соответствует обрезанной геометрии.",
    }

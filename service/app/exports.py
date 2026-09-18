"""Формирование файловых выгрузок: GeoJSON, Shapefile (zip), справка JSON/CSV."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd

# Имена колонок Shapefile ограничены 10 символами
SHP_RENAME = {"fire_event_id": "fire_event"}


def contours_geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    if gdf.empty:
        return json.dumps({"type": "FeatureCollection", "features": []}).encode()
    return gdf.to_json(drop_id=True).encode()


def contours_shapefile_zip(gdf: gpd.GeoDataFrame) -> bytes:
    """ZIP с shp/shx/dbf/prj/cpg. Для пустой выборки не вызывается (см. api)."""
    out = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        shp_path = Path(tmp) / "burned_areas.shp"
        gdf.rename(columns=SHP_RENAME).to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(Path(tmp).iterdir()):
                zf.write(f, f.name)
    return out.getvalue()


def report_json_bytes(report: dict) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode()


def report_csv_bytes(report: dict) -> bytes:
    lines = ["severity,label,area_ha,share_pct"]
    for row in report["by_severity"]:
        lines.append(f"{row['severity']},{row['label']},{row['area_ha']},{row['share_pct']}")
    total_share = 100.0 if report["total_burned_area_ha"] else 0.0
    lines.append(f"total,итого,{report['total_burned_area_ha']},{total_share}")
    return ("\n".join(lines) + "\n").encode("utf-8-sig")

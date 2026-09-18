"""In-memory каталог: слои GeoPackage + пространственно-временные выборки."""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

CRS_WGS84 = "EPSG:4326"
HOTSPOT_SCHEMA = ["chip_id", "acq_datetime", "satellite"]
CONTOUR_SCHEMA = [
    "contour_id", "chip_id", "fire_event_id", "severity", "area_ha",
    "date_pre", "date_post", "epsg",
]


def _read_layer(path: Path, layer: str, columns: list[str]) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:
        logger.warning("layer %r not found in %s — using empty layer", layer, path)
        return gpd.GeoDataFrame({c: [] for c in columns}, geometry=[], crs=CRS_WGS84)


class Catalog:
    def __init__(self, hotspots: gpd.GeoDataFrame, contours: gpd.GeoDataFrame):
        self.hotspots = hotspots
        self.contours = contours
        self.hotspots["_date"] = (
            pd.to_datetime(self.hotspots["acq_datetime"], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        self.contours["_post_date"] = pd.to_datetime(
            self.contours["date_post"], errors="coerce"
        ).dt.normalize()

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        hotspots = _read_layer(path, "hotspots", HOTSPOT_SCHEMA)
        contours = _read_layer(path, "burn_contours", CONTOUR_SCHEMA)
        logger.info("catalog loaded: %d hotspots, %d contours", len(hotspots), len(contours))
        return cls(hotspots, contours)

    def query_hotspots(self, geom: BaseGeometry, date_from: pd.Timestamp, date_to: pd.Timestamp) -> gpd.GeoDataFrame:
        return self._query(self.hotspots, "_date", geom, date_from, date_to)

    def query_contours(self, geom: BaseGeometry, date_from: pd.Timestamp, date_to: pd.Timestamp) -> gpd.GeoDataFrame:
        """Контур попадает в выборку, если его date_post лежит в интервале."""
        return self._query(self.contours, "_post_date", geom, date_from, date_to)

    @staticmethod
    def _query(gdf: gpd.GeoDataFrame, date_col: str, geom, date_from, date_to) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.drop(columns=[date_col]).copy()
        idx = gdf.sindex.query(geom, predicate="intersects")
        sel = gdf.iloc[sorted(idx)]
        sel = sel[(sel[date_col] >= date_from) & (sel[date_col] <= date_to)]
        return sel.drop(columns=[date_col])

    def meta(self) -> dict:
        """Границы данных и демо-пример (крупнейший пожар) для UI."""
        dates = []
        if not self.hotspots.empty:
            dates += [self.hotspots["_date"].min(), self.hotspots["_date"].max()]
        if not self.contours.empty:
            dates += [self.contours["_post_date"].min(), self.contours["_post_date"].max()]
        date_min = min(dates).date().isoformat() if dates else "2019-04-01"
        date_max = max(dates).date().isoformat() if dates else "2025-10-31"

        demo = None
        if not self.contours.empty:
            event = self.contours.groupby("fire_event_id")["area_ha"].sum().idxmax()
            ev = self.contours[self.contours["fire_event_id"] == event]
            minx, miny, maxx, maxy = ev.total_bounds
            dx = (maxx - minx) * 0.2 or 0.05
            dy = (maxy - miny) * 0.2 or 0.05
            d0 = pd.to_datetime(ev["date_pre"], errors="coerce").min() - pd.Timedelta(days=14)
            d1 = ev["_post_date"].max() + pd.Timedelta(days=14)
            demo = {
                "fire_event_id": str(event),
                "bbox": [minx - dx, miny - dy, maxx + dx, maxy + dy],
                "date_from": d0.date().isoformat(),
                "date_to": d1.date().isoformat(),
            }
        return {
            "hotspot_count": int(len(self.hotspots)),
            "contour_count": int(len(self.contours)),
            "date_min": date_min,
            "date_max": date_max,
            "demo": demo,
        }

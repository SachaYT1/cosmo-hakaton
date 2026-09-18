"""Сборка GeoDataFrame термоточек и контуров гарей из маски одного чипа."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from app.geo import ChipGrid, hotspot_points, severity_polygons

CRS_WGS84 = "EPSG:4326"


def grid_from_meta(row: pd.Series) -> ChipGrid:
    return ChipGrid(
        epsg=int(float(row["epsg"])),
        x_min=float(row["x_min"]),
        y_min=float(row["y_min"]),
        x_max=float(row["x_max"]),
        y_max=float(row["y_max"]),
        width=int(row["width"]),
        height=int(row["height"]),
    )


def hotspots_gdf(row: pd.Series, mask: np.ndarray) -> gpd.GeoDataFrame:
    pts = hotspot_points(mask, grid_from_meta(row))
    data = {
        "chip_id": [str(row["chip_id"])] * len(pts),
        "acq_datetime": [str(row["acq_datetime"])] * len(pts),
        "satellite": [str(row["satellite"])] * len(pts),
    }
    geometry = [Point(lon, lat) for lon, lat in pts]
    return gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_WGS84)


def contours_gdf(row: pd.Series, mask: np.ndarray) -> gpd.GeoDataFrame:
    grid = grid_from_meta(row)
    polys = severity_polygons(mask, grid)
    data = {
        "contour_id": [f"{row['chip_id']}_{i:04d}" for i in range(len(polys))],
        "chip_id": [str(row["chip_id"])] * len(polys),
        "fire_event_id": [str(row["fire_event_id"])] * len(polys),
        "severity": [int(p["severity"]) for p in polys],
        "area_ha": [float(p["area_ha"]) for p in polys],
        "date_pre": [str(row["date_pre"])] * len(polys),
        "date_post": [str(row["date_post"])] * len(polys),
        "epsg": [grid.epsg] * len(polys),
    }
    geometry = [p["geometry"] for p in polys]
    return gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_WGS84)

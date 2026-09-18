"""Геопривязка чипов и конвертация масок в векторные объекты."""

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_bounds, xy
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


@dataclass(frozen=True)
class ChipGrid:
    """Сетка чипа: UTM-проекция и охват из meta.csv."""

    epsg: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: int
    height: int

    @property
    def transform(self):
        return from_bounds(self.x_min, self.y_min, self.x_max, self.y_max, self.width, self.height)

    @property
    def pixel_area_ha(self) -> float:
        px_w = (self.x_max - self.x_min) / self.width
        px_h = (self.y_max - self.y_min) / self.height
        return px_w * px_h / 10_000.0


def to_wgs84(epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def hotspot_points(mask: np.ndarray, grid: ChipGrid) -> list[tuple[float, float]]:
    """Центры пикселей mask==1 в координатах (lon, lat)."""
    rows, cols = np.nonzero(mask == 1)
    if len(rows) == 0:
        return []
    xs, ys = xy(grid.transform, rows, cols)
    lons, lats = to_wgs84(grid.epsg).transform(xs, ys)
    return list(zip(lons, lats))


def severity_polygons(mask: np.ndarray, grid: ChipGrid) -> list[dict]:
    """Полигонизация классов 1..3. Площадь считается в родном UTM (без искажений)."""
    out: list[dict] = []
    transformer = to_wgs84(grid.epsg)
    for sev in (1, 2, 3):
        class_mask = mask == sev
        if not class_mask.any():
            continue
        shapes = features.shapes(class_mask.astype(np.uint8), mask=class_mask, transform=grid.transform)
        for geom, _value in shapes:
            poly_utm = shape(geom)
            poly_wgs = shp_transform(transformer.transform, poly_utm)
            out.append(
                {
                    "severity": sev,
                    "geometry": poly_wgs,
                    "area_ha": round(poly_utm.area / 10_000.0, 4),
                }
            )
    return out

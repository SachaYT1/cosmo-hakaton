"""Pydantic-модели запросов."""

from datetime import date
import math

from pydantic import BaseModel, Field, model_validator


class SpatioTemporalQuery(BaseModel):
    """Полигон (GeoJSON geometry) ИЛИ bbox [minx, miny, maxx, maxy] в EPSG:4326 + интервал дат."""

    polygon: dict | None = None
    bbox: list[float] | None = None
    date_from: date
    date_to: date
    # Фильтр отрисовки: скрыть контуры мельче порога (только /api/burned-areas;
    # справка и выгрузки всегда считаются по всем контурам)
    min_area_ha: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "SpatioTemporalQuery":
        if (self.polygon is None) == (self.bbox is None):
            raise ValueError("передайте ровно одно из polygon или bbox")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox должен быть [minx, miny, maxx, maxy]")
            minx, miny, maxx, maxy = self.bbox
            if not all(math.isfinite(v) for v in self.bbox):
                raise ValueError("координаты bbox должны быть конечными числами")
            if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
                raise ValueError("bbox должен иметь возрастающие координаты в пределах EPSG:4326")
        if self.date_from > self.date_to:
            raise ValueError("date_from позже date_to")
        return self

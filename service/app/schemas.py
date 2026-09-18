"""Pydantic-модели запросов."""

from datetime import date

from pydantic import BaseModel, model_validator


class SpatioTemporalQuery(BaseModel):
    """Полигон (GeoJSON geometry) ИЛИ bbox [minx, miny, maxx, maxy] в EPSG:4326 + интервал дат."""

    polygon: dict | None = None
    bbox: list[float] | None = None
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def _check(self) -> "SpatioTemporalQuery":
        if (self.polygon is None) == (self.bbox is None):
            raise ValueError("передайте ровно одно из polygon или bbox")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox должен быть [minx, miny, maxx, maxy]")
        if self.date_from > self.date_to:
            raise ValueError("date_from позже date_to")
        return self

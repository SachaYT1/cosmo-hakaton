"""Fail-closed policy checks for external AF observations."""
from __future__ import annotations

from datetime import date

# Regions are deliberately far from southern/southwestern Russia. A prepared
# 256x256 tile must fit completely inside one box.
SAFE_REGIONS = {
    "australia": (-45.0, -10.0, 112.0, 155.0),
    "conus": (24.0, 50.0, -125.0, -66.0),
    "southern_africa": (-35.0, -10.0, 10.0, 40.0),
    "south_america": (-55.0, 12.0, -82.0, -34.0),
}


def safe_date(value: str) -> bool:
    """Allow public observations outside the case's April-October test season."""
    observed = date.fromisoformat(value[:10])
    return 2019 <= observed.year <= 2025 and observed.month in (11, 12, 1, 2, 3)


def containing_region(lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float) -> str | None:
    if not (-90 <= lat_min <= lat_max <= 90 and -180 <= lon_min <= lon_max <= 180):
        return None
    for name, (south, north, west, east) in SAFE_REGIONS.items():
        if south <= lat_min <= lat_max <= north and west <= lon_min <= lon_max <= east:
            return name
    return None


def validate_chip(item: dict) -> str:
    if not safe_date(item["acq_datetime"]):
        raise ValueError(f"External date overlaps or violates approved scope: {item['acq_datetime']}")
    region = containing_region(item["lat_min"], item["lat_max"],
                               item["lon_min"], item["lon_max"])
    if region is None or item.get("region") != region:
        raise ValueError(f"External chip outside approved regions: {item['chip_id']}")
    return region

"""Построение data/catalog.gpkg из GT-масок или предсказаний моделей (RLE)."""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from app.ingest_core import contours_gdf, hotspots_gdf
from app.rle import decode_rle

logger = logging.getLogger(__name__)

CRS_WGS84 = "EPSG:4326"


def read_gt_mask(masks_dir: Path, chip_id: str) -> np.ndarray:
    with rasterio.open(masks_dir / f"{chip_id}_MASK.tif") as src:
        return src.read(1)


def mask_from_submission(sub: pd.DataFrame, chip_id: str, width: int, height: int) -> np.ndarray:
    """Собирает маску 0..3 из строк submission.csv для одного чипа."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, r in sub[sub["chip_id"] == chip_id].iterrows():
        rle = "" if pd.isna(r["rle"]) else str(r["rle"])
        class_mask = decode_rle(rle, width, height)
        mask[class_mask == 1] = int(r["class_id"])
    return mask


def _chip_mask(row: pd.Series, masks_dir: Path, source: str, sub: pd.DataFrame | None) -> np.ndarray | None:
    if source == "gt":
        return read_gt_mask(masks_dir, str(row["chip_id"]))
    if str(row["chip_id"]) not in set(sub["chip_id"]):
        logger.warning("chip %s not in submission, skipped", row["chip_id"])
        return None
    return mask_from_submission(sub, str(row["chip_id"]), int(row["width"]), int(row["height"]))


def build_catalog(data_dir: Path, output: Path, source: str = "gt", submission: Path | None = None) -> None:
    """data_dir — каталог со структурой train/ (af/, bs/, meta.csv в каждом)."""
    if source == "predictions" and submission is None:
        raise ValueError("source=predictions требует путь к submission.csv")
    sub = pd.read_csv(submission) if source == "predictions" else None

    af_meta = pd.read_csv(data_dir / "af" / "meta.csv")
    bs_meta = pd.read_csv(data_dir / "bs" / "meta.csv")

    hotspot_parts = []
    for _, row in af_meta.iterrows():
        mask = _chip_mask(row, data_dir / "af" / "masks", source, sub)
        if mask is not None:
            hotspot_parts.append(hotspots_gdf(row, mask))

    contour_parts = []
    for _, row in bs_meta.iterrows():
        mask = _chip_mask(row, data_dir / "bs" / "masks", source, sub)
        if mask is not None:
            contour_parts.append(contours_gdf(row, mask))

    hotspots = gpd.GeoDataFrame(pd.concat(hotspot_parts, ignore_index=True), crs=CRS_WGS84)
    contours = gpd.GeoDataFrame(pd.concat(contour_parts, ignore_index=True), crs=CRS_WGS84)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    hotspots.to_file(output, layer="hotspots", driver="GPKG")
    contours.to_file(output, layer="burn_contours", driver="GPKG")
    logger.info("catalog written: %s (%d hotspots, %d contours)", output, len(hotspots), len(contours))

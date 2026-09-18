"""Chip loaders. Paths follow the layout of the organisers' archive:

    <root>/bs/sentinel2_{pre,post}/<id>_Sentinel-2_{pre,post}.tif   (H,W,10) uint16
    <root>/bs/sentinel1_{pre,post}/<id>_Sentinel-1_{pre,post}.tif   (H,W,2)  int16, dB*100
    <root>/bs/aux/<id>_AUX.tif                                      (H,W,3)  int16: dem, slope, landcover
    <root>/bs/masks/<id>_MASK.tif                                   (H,W)    uint8 0..3   (train only)
    <root>/af/viirs/<id>_VIIRS_I1-I5.tif                            (H,W,8)  float32
    <root>/af/aux/<id>_AUX.tif                                      (H,W,5)  float32
    <root>/af/masks/<id>_MASK.tif                                   (H,W)    uint8 0/1    (train only)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12", "SCL"]
S1_BANDS = ["VV", "VH"]
BS_AUX_BANDS = ["dem", "slope", "landcover"]
VIIRS_BANDS = ["I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith", "valid"]
AF_AUX_BANDS = ["landcover", "dem", "t2m", "rh2m", "wind_speed"]

# ESA WorldCover v200 codes
WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
              70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
# Sentinel-2 L2A scene classification
SCL = {0: "nodata", 1: "saturated", 2: "dark/shadow", 3: "cloud_shadow", 4: "vegetation",
       5: "bare", 6: "water", 7: "unclassified", 8: "cloud_med", 9: "cloud_high",
       10: "cirrus", 11: "snow"}


def _read(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def _data_dir(root: Path, kind: str, *candidates: str) -> Path:
    """First existing candidate sub-folder (the archive ships some legacy empty dirs)."""
    for c in candidates:
        p = root / kind / c
        if p.is_dir() and any(p.iterdir()):
            return p
    return root / kind / candidates[0]


@dataclass
class BSChip:
    chip_id: str
    s2_pre: np.ndarray    # (H,W,10) uint16
    s2_post: np.ndarray
    s1_pre: np.ndarray    # (H,W,2) int16
    s1_post: np.ndarray
    aux: np.ndarray       # (H,W,3) int16
    mask: np.ndarray | None

    def band(self, name: str, when: str) -> np.ndarray:
        arr = self.s2_pre if when == "pre" else self.s2_post
        return arr[..., S2_BANDS.index(name)]


@dataclass
class AFChip:
    chip_id: str
    viirs: np.ndarray     # (H,W,8) float32
    aux: np.ndarray       # (H,W,5) float32
    mask: np.ndarray | None

    def band(self, name: str) -> np.ndarray:
        return self.viirs[..., VIIRS_BANDS.index(name)]


def load_bs(root: str | Path, chip_id: str, with_mask: bool = True) -> BSChip:
    root = Path(root)
    s2pre = _data_dir(root, "bs", "sentinel2_pre", "pre")
    s2post = _data_dir(root, "bs", "sentinel2_post", "post")
    s1pre = _data_dir(root, "bs", "sentinel1_pre", "sar_pre")
    s1post = _data_dir(root, "bs", "sentinel1_post", "sar_post")
    mask_p = root / "bs" / "masks" / f"{chip_id}_MASK.tif"
    return BSChip(
        chip_id=chip_id,
        s2_pre=_read(s2pre / f"{chip_id}_Sentinel-2_pre.tif"),
        s2_post=_read(s2post / f"{chip_id}_Sentinel-2_post.tif"),
        s1_pre=_read(s1pre / f"{chip_id}_Sentinel-1_pre.tif"),
        s1_post=_read(s1post / f"{chip_id}_Sentinel-1_post.tif"),
        aux=_read(root / "bs" / "aux" / f"{chip_id}_AUX.tif"),
        mask=_read(mask_p) if with_mask and mask_p.exists() else None,
    )


def load_af(root: str | Path, chip_id: str, with_mask: bool = True) -> AFChip:
    root = Path(root)
    viirs_dir = _data_dir(root, "af", "viirs", "images")
    mask_p = root / "af" / "masks" / f"{chip_id}_MASK.tif"
    return AFChip(
        chip_id=chip_id,
        viirs=_read(viirs_dir / f"{chip_id}_VIIRS_I1-I5.tif"),
        aux=_read(root / "af" / "aux" / f"{chip_id}_AUX.tif"),
        mask=_read(mask_p) if with_mask and mask_p.exists() else None,
    )


def list_chips(root: str | Path, kind: str) -> list[str]:
    """Chip ids of a kind, taken from the image folder (works for train and test)."""
    root = Path(root)
    if kind == "bs":
        d = _data_dir(root, "bs", "sentinel2_pre", "pre")
        return sorted(p.name.replace("_Sentinel-2_pre.tif", "") for p in d.glob("*.tif"))
    d = _data_dir(root, "af", "viirs", "images")
    return sorted(p.name.replace("_VIIRS_I1-I5.tif", "") for p in d.glob("*.tif"))

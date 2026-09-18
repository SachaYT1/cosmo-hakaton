"""Active-fire (AF) module: contextual candidate selection + gradient-boosted pixel classifier.

VIIRS bands: I1..I5, solar_zenith, sensor_zenith, valid. I1-I3 are NaN at night.
Standard fire algorithms (VNP14IMG) compare a pixel's MIR brightness temperature (I4) and the
MIR-TIR difference (I4-I5) against statistics of the surrounding window; we compute the same
contextual statistics as features and let a classifier learn the decision, day and night.
"""
from __future__ import annotations

import cv2
import numpy as np

from .io import AFChip

DAY_SZA = 85.0
CONTEXT_WINDOWS = (7, 15, 31)

FEATURES = [
    "I1", "I2", "I3", "I4", "I5", "dT", "sza", "vza", "day",
    *[f"{b}_{s}{w}" for w in CONTEXT_WINDOWS for b in ("I4", "dT") for s in ("dmean", "z")],
    "I4_dmax3", "I4_rank3", "I5_dmean15",
    "lc", "dem", "t2m", "rh2m", "wind",
    "ndvi", "glint",
]

CONTEXT_FEATURES = [*FEATURES, "valid"]
for _band in ("I4", "dT", "I5"):
    for _window in (7, 15, 31):
        CONTEXT_FEATURES += [f"{_band}_ring_{s}{_window}" for s in ("delta", "z", "std")]
    CONTEXT_FEATURES += [f"{_band}_dmedian{w}" for w in (3, 5)]
    CONTEXT_FEATURES += [f"{_band}_dmean3", f"{_band}_std3"]
for _window in (3, 7, 15):
    CONTEXT_FEATURES += [f"{s}_frac{_window}" for s in
                         ("valid", "hot330", "dt20", "candidate", "lc10", "lc40", "lc50", "lc80")]
CONTEXT_FEATURES += ["I4_minus_t2m", "I5_minus_t2m", "I3_minus_I1", "I3_minus_I2", "I3_I2_index"]


def feature_names(feature_set: str = "legacy") -> list[str]:
    if feature_set == "legacy":
        return FEATURES
    if feature_set == "context":
        return CONTEXT_FEATURES
    raise ValueError(f"Unknown AF feature set: {feature_set}")


def extract_features(chip: AFChip, feature_set: str = "legacy") -> np.ndarray:
    feature_names(feature_set)  # reject an incompatible model/config early
    return enhanced_features(chip) if feature_set == "context" else features(chip)


def _mean_std(x: np.ndarray, valid: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over a w x w window counting only valid pixels (candidate excluded implicitly
    through the robust z-score; fires cover <0.1% of pixels so contamination is small)."""
    v = valid.astype(np.float32)
    xs = np.where(valid, x, 0).astype(np.float32)
    n = cv2.boxFilter(v, -1, (w, w), normalize=False, borderType=cv2.BORDER_REFLECT)
    s1 = cv2.boxFilter(xs, -1, (w, w), normalize=False, borderType=cv2.BORDER_REFLECT)
    s2 = cv2.boxFilter(xs * xs, -1, (w, w), normalize=False, borderType=cv2.BORDER_REFLECT)
    n = np.maximum(n, 1)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    return mean, std


def features(chip: AFChip) -> np.ndarray:
    """(H, W, len(FEATURES)) float32."""
    v = chip.viirs
    i1, i2, i3, i4, i5 = (v[..., k] for k in range(5))
    sza, vza, valid = v[..., 5], v[..., 6], v[..., 7] > 0
    valid = valid & np.isfinite(i4) & np.isfinite(i5)
    i4 = np.where(valid, i4, np.nan)
    i5 = np.where(valid, i5, np.nan)
    i4f = np.nan_to_num(i4, nan=np.nanmedian(i4) if np.isfinite(i4).any() else 280.0)
    i5f = np.nan_to_num(i5, nan=np.nanmedian(i5) if np.isfinite(i5).any() else 280.0)
    dT = i4f - i5f
    feats = {"I1": i1, "I2": i2, "I3": i3, "I4": i4f, "I5": i5f, "dT": dT, "sza": sza, "vza": vza,
             "day": (sza < DAY_SZA).astype(np.float32)}
    for w in CONTEXT_WINDOWS:
        for name, x in (("I4", i4f), ("dT", dT)):
            m, s = _mean_std(x, valid, w)
            feats[f"{name}_dmean{w}"] = x - m
            feats[f"{name}_z{w}"] = (x - m) / (s + 1.0)
    k3 = np.ones((3, 3), np.uint8)
    k3[1, 1] = 0
    nb_max = cv2.dilate(i4f.astype(np.float32), k3)
    feats["I4_dmax3"] = i4f - nb_max                       # >0: local maximum (fire core)
    feats["I4_rank3"] = (i4f[..., None] > np.stack(
        [np.roll(np.roll(i4f, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
         if dy or dx], -1)).sum(-1).astype(np.float32)
    m5, _ = _mean_std(i5f, valid, 15)
    feats["I5_dmean15"] = i5f - m5
    a = chip.aux
    feats.update(lc=a[..., 0], dem=a[..., 1], t2m=a[..., 2], rh2m=a[..., 3], wind=a[..., 4])
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["ndvi"] = (i2 - i1) / (i2 + i1)
    feats["glint"] = i3  # strong I3 reflectance indicates sun glint / reflective roofs
    out = np.stack([np.asarray(feats[f], np.float32) for f in FEATURES], -1)
    return out


def candidates(f: np.ndarray) -> np.ndarray:
    """Cheap pre-filter keeping ~all fire pixels: warm vs context or absolutely hot."""
    i4 = f[..., FEATURES.index("I4")]
    d = f[..., FEATURES.index("I4_dmean15")]
    ddt = f[..., FEATURES.index("dT_dmean15")]
    return (i4 > 320) | ((d > 3) & (ddt > 2))


def enhanced_features(chip: AFChip) -> np.ndarray:
    """Image-only contextual features; the first columns retain the v1 schema.

    Annuli exclude the central 3x3 neighbourhood from the background estimate.
    This reduces contamination by the hot source itself. No metadata, dates,
    coordinates or labels are used, so the same transform works on blind test.
    """
    base = features(chip)
    i4 = base[..., FEATURES.index("I4")]
    i5 = base[..., FEATURES.index("I5")]
    dt = i4 - i5
    valid = ((chip.viirs[..., 7] > 0) & np.isfinite(chip.viirs[..., 3])
             & np.isfinite(chip.viirs[..., 4]))
    extra = [valid.astype(np.float32)]
    v = valid.astype(np.float32)

    def box(x, w):
        return cv2.boxFilter(x, -1, (w, w), normalize=False, borderType=cv2.BORDER_REFLECT)

    for x in (i4, dt, i5):
        xv = np.where(valid, x, 0).astype(np.float32)
        for w in (7, 15, 31):
            n = np.maximum(box(v, w) - box(v, 3), 1)
            mean = (box(xv, w) - box(xv, 3)) / n
            variance = np.maximum((box(xv*xv, w)-box(xv*xv, 3))/n - mean*mean, 0)
            std = np.sqrt(variance)
            extra.extend([x-mean, (x-mean)/(std+1), std])
        for w in (3, 5):
            median = cv2.medianBlur(x.astype(np.float32), w)
            extra.append(x-median)
        mean, std = _mean_std(x, valid, 3)
        extra.extend([x-mean, std])

    # Neighbour support and land-cover context distinguish isolated reflectors,
    # water/urban edges and clusters without deleting small real fires by rule.
    for w in (3, 7, 15):
        denom = np.maximum(box(v, w), 1)
        extra.append(box(v, w)/(w*w))
        for hot in ((i4 > 330), (dt > 20), candidates(base)):
            extra.append(box((hot & valid).astype(np.float32), w)/denom)
        for cover in (10, 40, 50, 80):
            extra.append(box((chip.aux[..., 0] == cover).astype(np.float32), w)/(w*w))

    i1, i2, i3 = (chip.viirs[..., k] for k in (0, 1, 2))
    extra.extend([i4-chip.aux[..., 2], i5-chip.aux[..., 2], i3-i1, i3-i2])
    with np.errstate(divide="ignore", invalid="ignore"):
        extra.append((i3-i2)/(np.abs(i3)+np.abs(i2)+0.01))
    out = np.concatenate([base, np.stack(extra, axis=-1)], axis=-1).astype(np.float32)
    out[np.isinf(out)] = np.nan
    return out


def rule_predict(chip: AFChip, day_t: float = 330.0, night_t: float = 317.0,
                 night_dmean: float = 3.0) -> np.ndarray:
    """Baseline (fitted on train, F1 0.866): absolute I4 threshold by day; lower absolute
    threshold plus a contextual excess over the 15x15 background at night."""
    f = features(chip)
    i4 = f[..., FEATURES.index("I4")]
    day = f[..., FEATURES.index("day")] > 0
    d = f[..., FEATURES.index("I4_dmean15")]
    return np.where(day, i4 > day_t, (i4 > night_t) & (d > night_dmean)).astype(np.uint8)

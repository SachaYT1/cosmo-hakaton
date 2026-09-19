"""Convert every audited NOAA VIIRS granule into weakly labelled AF chips."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import tifffile

from firemon.external_af import containing_region, safe_date

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/noaa_viirs"
OUTPUT = ROOT / "data/external/noaa_af_chips"
SIZE = 256
REQUIRED = {*(f"VIIRS-I{i}-SDR" for i in range(1, 6)),
            "VIIRS-IMG-GEO-TC", "VIIRS-AF-Iband-EDR"}


def checked_files(granule: Path) -> tuple[dict[str, Path], str, str]:
    downloads = json.loads((granule / "downloads.json").read_text())
    files, dates, starts = {}, set(), set()
    for item in downloads:
        path = ROOT / item["file"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Invalid NOAA download: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        match = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", item["key"])
        start = re.search(r"_[ts](?:20\d{6})?(\d{7})_", item["key"])
        if not match or not start:
            raise ValueError(f"Date/time missing from NOAA key: {item['key']}")
        dates.add("-".join(match.groups()))
        starts.add(start.group(1))
        files[item["product"]] = path
    if set(files) != REQUIRED or len(dates) != 1 or len(starts) != 1:
        raise ValueError(f"Incomplete or mismatched NOAA granule: {granule}")
    observed = dates.pop()
    if not safe_date(observed):
        raise ValueError(f"NOAA date violates external-data policy: {observed}")
    return files, observed, starts.pop()


def unpack_band(path: Path, number: int, row: int, col: int) -> np.ndarray:
    with h5py.File(path) as source:
        group = source[f"All_Data/VIIRS-I{number}-SDR_All"]
        key = "Reflectance" if number <= 3 else "BrightnessTemperature"
        raw = group[key][row:row+SIZE, col:col+SIZE]
        scale, offset = group[key + "Factors"][:]
    values = raw.astype(np.float32) * scale + offset
    values[raw >= 65528] = np.nan
    return values


def prepare_granule(granule: Path, out: Path) -> tuple[list[dict], dict]:
    files, observed, start = checked_files(granule)
    with h5py.File(files["VIIRS-AF-Iband-EDR"]) as source:
        fire_mask = source["Fire Mask/fire_mask"][:]
        lines = source["Fire Pixels/FP_line"][:]
        samples = source["Fire Pixels/FP_sample"][:]
        categories = source["Fire Pixels/FP_PersistentAnomalyCategory"][:]
    detections = np.isin(fire_mask, [8, 9])
    labels = np.full(fire_mask.shape, 255, np.uint8)
    labels[fire_mask == 5] = 0
    labels[detections] = 1
    category_counts = {str(i): int((categories == i).sum()) for i in range(6)}
    for row, col, category in zip(lines, samples, categories):
        if category != 0 and row < labels.shape[0] and col < labels.shape[1]:
            labels[row, col] = 0 if category in (1, 3, 4) else 255
    records = []
    with h5py.File(files["VIIRS-IMG-GEO-TC"]) as geo:
        group = geo["All_Data/VIIRS-IMG-GEO-TC_All"]
        for row in range(0, fire_mask.shape[0], SIZE):
            for col in range(0, fire_mask.shape[1], SIZE):
                detected = detections[row:row+SIZE, col:col+SIZE]
                if detected.shape != (SIZE, SIZE) or not detected.any():
                    continue
                lat = group["Latitude"][row:row+SIZE, col:col+SIZE]
                lon = group["Longitude"][row:row+SIZE, col:col+SIZE]
                if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
                    continue
                bounds = (float(lat.min()), float(lat.max()),
                          float(lon.min()), float(lon.max()))
                region = containing_region(*bounds)
                if region is None:
                    continue
                bands = [unpack_band(files[f"VIIRS-I{i}-SDR"], i, row, col)
                         for i in range(1, 6)]
                solar = group["SolarZenithAngle"][row:row+SIZE, col:col+SIZE]
                sensor = group["SatelliteZenithAngle"][row:row+SIZE, col:col+SIZE]
                mask = fire_mask[row:row+SIZE, col:col+SIZE]
                valid = (np.isfinite(bands[3]) & np.isfinite(bands[4])
                         & np.isin(mask, [5, 7, 8, 9]))
                viirs = np.stack([*bands, solar, sensor, valid.astype(np.float32)], -1)
                tile_labels = labels[row:row+SIZE, col:col+SIZE]
                aux = np.full((SIZE, SIZE, 5), np.nan, np.float32)
                # GITCO height is terrain elevation in metres and matches the
                # competition AUX DEM semantics. Other AUX bands remain missing.
                elevation = group["Height"][row:row+SIZE, col:col+SIZE].astype(np.float32)
                elevation[~np.isfinite(elevation) | (elevation < -500)] = np.nan
                aux[..., 1] = elevation
                chip_id = (f"NOAA_{region.upper()}_{observed.replace('-', '')}_{start[:6]}"
                           f"_r{row:04d}_c{col:04d}")
                tifffile.imwrite(out / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif", viirs,
                                 compression="zlib")
                tifffile.imwrite(out / "aux" / f"{chip_id}_AUX.tif", aux, compression="zlib")
                tifffile.imwrite(out / "masks" / f"{chip_id}_MASK.tif", tile_labels,
                                 compression="zlib")
                records.append({"chip_id": chip_id,
                                "acq_datetime": f"{observed}T{start[:2]}:{start[2:4]}:{start[4:6]}Z",
                                "satellite": "SNPP", "region": region,
                                "lat_min": bounds[0], "lat_max": bounds[1],
                                "lon_min": bounds[2], "lon_max": bounds[3],
                                "weak_positive_pixels": int((tile_labels == 1).sum()),
                                "labelled_negative_pixels": int((tile_labels == 0).sum()),
                                "unknown_pixels": int((tile_labels == 255).sum())})
    source_info = {"directory": str(granule.relative_to(ROOT)), "date": observed,
                   "start": start, "persistent_categories": category_counts}
    return records, source_info


def main() -> None:
    granules = sorted(SOURCE.glob("*/*/downloads.json"))
    if not granules:
        raise FileNotFoundError(f"No NOAA download manifests below {SOURCE}")
    out = OUTPUT / "af"
    for part in ("viirs", "aux", "masks"):
        folder = out / part
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("NOAA_*.tif"):
            stale.unlink()
    records, sources = [], []
    for download_manifest in granules:
        new_records, source_info = prepare_granule(download_manifest.parent, out)
        records.extend(new_records)
        source_info["prepared_chips"] = len(new_records)
        sources.append(source_info)
    manifest = {
        "source": "NOAA JPSS NODD VIIRS SDR and Active Fire I-band EDR",
        "source_url": "https://registry.opendata.aws/noaa-jpss/",
        "policy": "winter-2019-2025-outside-eurasia-v1",
        "label_kind": "weak NOAA fire detection with persistent-anomaly negatives",
        "training_eligible": "optional weak supervision only",
        "unknown_mask_value": 255,
        "aux_channels": "DEM from GITCO Height; landcover/weather missing as NaN",
        "sources": sources, "chips": records,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    positives = sum(item["weak_positive_pixels"] for item in records)
    print(f"Wrote {len(records)} chips and {positives} weak positives from {len(sources)} granules")


if __name__ == "__main__":
    main()

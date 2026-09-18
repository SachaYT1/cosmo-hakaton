"""Turn an approved NOAA VIIRS granule into AF-format weakly labelled chips.

Only December 2022 Australian pixels are currently approved by this importer.
The NOAA active-fire mask is a detection product, not the competition's natural
versus industrial ground truth. Its output must be treated as weak supervision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
GRANULE = ROOT / "data/external/noaa_viirs/2022-12-15/t0429233"
OUTPUT = ROOT / "data/external/noaa_af_chips"
SIZE = 256


def checked_files() -> dict[str, Path]:
    downloads = json.loads((GRANULE / "downloads.json").read_text())
    files = {}
    for item in downloads:
        path = ROOT / item["file"]
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Invalid NOAA download: {path}")
        if "2022/12/15/" not in item["key"] or "_t0429233_" not in item["key"] and "_s202212150429233_" not in item["key"]:
            raise ValueError(f"Unapproved observation: {item['key']}")
        files[item["product"]] = path
    required = {*(f"VIIRS-I{i}-SDR" for i in range(1, 6)), "VIIRS-IMG-GEO-TC", "VIIRS-AF-Iband-EDR"}
    if set(files) != required:
        raise ValueError("Expected exactly five I-band SDRs, GEO, and AF EDR")
    return files


def unpack_band(path: Path, number: int, row: int, col: int) -> np.ndarray:
    with h5py.File(path) as f:
        group = f[f"All_Data/VIIRS-I{number}-SDR_All"]
        key = "Reflectance" if number <= 3 else "BrightnessTemperature"
        raw = group[key][row:row+SIZE, col:col+SIZE]
        scale, offset = group[key + "Factors"][:]
    values = raw.astype(np.float32) * scale + offset
    values[raw >= 65528] = np.nan  # VIIRS HDF5 reserved/fill codes
    return values


def main() -> None:
    files = checked_files()
    with h5py.File(files["VIIRS-AF-Iband-EDR"]) as f:
        fire_mask = f["Fire Mask/fire_mask"][:]
        lines = f["Fire Pixels/FP_line"][:]
        samples = f["Fire Pixels/FP_sample"][:]
        categories = f["Fire Pixels/FP_PersistentAnomalyCategory"][:]
    # 255 means NOAA did not establish a clear land/fire label there.
    labels = np.full(fire_mask.shape, 255, np.uint8)
    labels[fire_mask == 5] = 0
    labels[np.isin(fire_mask, [8, 9])] = 1
    for row, col, category in zip(lines, samples, categories):
        if category != 0 and row < labels.shape[0] and col < labels.shape[1]:
            labels[row, col] = 0 if category in (1, 3, 4) else 255
    OUT = OUTPUT / "af"
    for part in ("viirs", "aux", "masks"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(files["VIIRS-IMG-GEO-TC"]) as geo:
        group = geo["All_Data/VIIRS-IMG-GEO-TC_All"]
        for row in range(0, fire_mask.shape[0], SIZE):
            for col in range(0, fire_mask.shape[1], SIZE):
                tile_labels = labels[row:row+SIZE, col:col+SIZE]
                if tile_labels.shape != (SIZE, SIZE) or not (tile_labels == 1).any():
                    continue
                lat = group["Latitude"][row:row+SIZE, col:col+SIZE]
                lon = group["Longitude"][row:row+SIZE, col:col+SIZE]
                # Require the *entire* tile to lie far outside the case region.
                if not (np.isfinite(lat).all() and np.isfinite(lon).all()
                        and lat.min() >= -45 and lat.max() <= -10
                        and lon.min() >= 112 and lon.max() <= 155):
                    continue
                bands = [unpack_band(files[f"VIIRS-I{i}-SDR"], i, row, col)
                         for i in range(1, 6)]
                solar = group["SolarZenithAngle"][row:row+SIZE, col:col+SIZE]
                sensor = group["SatelliteZenithAngle"][row:row+SIZE, col:col+SIZE]
                valid = (np.isfinite(bands[3]) & np.isfinite(bands[4])
                         & np.isin(fire_mask[row:row+SIZE, col:col+SIZE], [5, 7, 8, 9]))
                viirs = np.stack([*bands, solar, sensor, valid.astype(np.float32)], -1)
                aux = np.full((SIZE, SIZE, 5), np.nan, np.float32)
                chip_id = f"NOAA_AUS_20221215_r{row:04d}_c{col:04d}"
                tifffile.imwrite(OUT / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif", viirs,
                                 compression="zlib")
                tifffile.imwrite(OUT / "aux" / f"{chip_id}_AUX.tif", aux,
                                 compression="zlib")
                tifffile.imwrite(OUT / "masks" / f"{chip_id}_MASK.tif", tile_labels,
                                 compression="zlib")
                records.append({"chip_id": chip_id, "acq_datetime": "2022-12-15T04:29:23Z",
                                "satellite": "SNPP", "lat_min": float(lat.min()),
                                "lat_max": float(lat.max()), "lon_min": float(lon.min()),
                                "lon_max": float(lon.max()), "weak_positive_pixels": int((tile_labels == 1).sum()),
                                "labelled_negative_pixels": int((tile_labels == 0).sum())})
    manifest = {"source": "NOAA JPSS NODD VIIRS SDR and Active Fire I-band EDR",
                "source_url": "https://registry.opendata.aws/noaa-jpss/",
                "observation_date": "2022-12-15", "region": "Australia",
                "label_kind": "weak NOAA fire detection, with persistent-anomaly flags",
                "training_eligible": "optional weak supervision only",
                "unknown_mask_value": 255, "aux_channels": "missing, stored as NaN",
                "chips": records}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records)} chips and {sum(r['weak_positive_pixels'] for r in records)} weak positives")


if __name__ == "__main__":
    main()

"""Prepare and audit every approved external AF source.

Run from any directory:
    python -m scripts.prepare_af_external
    python -m scripts.prepare_af_external --download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from firemon.io import list_chips, load_af

ROOT = Path(__file__).resolve().parents[1]


def run(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main(download: bool = False) -> None:
    if download:
        run("scripts.download_af_open_data")
    run("scripts.prepare_af_points")
    run("scripts.prepare_af_noaa")

    usfs_path = ROOT / "data/external/usfs_viirs_conus/manifest.json"
    noaa_path = ROOT / "data/external/noaa_af_chips/manifest.json"
    usfs = json.loads(usfs_path.read_text())
    noaa = json.loads(noaa_path.read_text())
    ids = list_chips(noaa_path.parent, "af")
    if ids != sorted(item["chip_id"] for item in noaa["chips"]):
        raise ValueError("NOAA manifest and prepared chips differ")
    positive = negative = unknown = 0
    for chip_id in ids:
        mask = load_af(noaa_path.parent, chip_id).mask
        positive += int((mask == 1).sum())
        negative += int((mask == 0).sum())
        unknown += int((mask == 255).sum())
    source_files = []
    download_manifests = sorted((ROOT / "data/external/noaa_viirs").glob("*/*/downloads.json"))
    for downloads in download_manifests:
        for item in json.loads(downloads.read_text()):
            path = ROOT / item["file"]
            sha = digest(path)
            if path.stat().st_size != item["bytes"] or sha != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            source_files.append({"file": item["file"], "bytes": item["bytes"],
                                 "sha256": sha})
    report = {
        "policy": {
            "test_scope_from_case": "southern/southwestern Russia, April-October 2019-2025",
            "external_scope": "CONUS 2017-2018 index; approved remote winter NOAA chips",
            "test_geolocation_recovery": False,
            "test_related_data_used": False,
        },
        "usfs_detection_index": {
            "records": usfs["records_total"],
            "training_eligible": usfs["training_eligible"],
            "manifest": str(usfs_path.relative_to(ROOT)),
        },
        "noaa_weak_chips": {
            "source_granules": len(download_manifests), "chips": len(ids),
            "positive_pixels": positive,
            "negative_pixels": negative, "unknown_pixels": unknown,
            "manifest": str(noaa_path.relative_to(ROOT)),
            "training_weight": 0.02,
        },
        "source_files": source_files,
    }
    destination = ROOT / "reports/af_external_data.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="download the fixed public allowlist before preparing")
    args = parser.parse_args()
    main(args.download)

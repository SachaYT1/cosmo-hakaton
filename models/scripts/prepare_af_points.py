"""Normalize public, historical USFS VIIRS detections for AF data preparation.

These are detection locations, not natural-fire labels. The output is a source
index for finding imagery and checking temporal persistence; it is never read
by inference or treated as a pixel mask.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/usfs_viirs_conus"
SOURCES = {
    2017: "https://fsapps.nwcg.gov/afm/data_viirs/fireptdata/viirs-af_fire_2017_365_conus_shapefile.zip",
    2018: "https://fsapps.nwcg.gov/afm/data_viirs_iband/fireptdata/viirs_iband_fire_2018_365_conus_shapefile.zip",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "historical detection index; labels are unverified",
                "training_eligible": False, "sources": []}
    counts = {}
    with gzip.open(OUT / "detections.csv.gz", "wt", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["source_year", "acq_date", "utc_hhmm", "latitude", "longitude",
                         "brightness_k", "frp_mw", "confidence", "satellite"])
        for year, url in SOURCES.items():
            archive = OUT / f"usfs{year}.zip"
            if not archive.exists():
                raise FileNotFoundError(f"Download {url} to {archive}")
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            with tempfile.TemporaryDirectory() as directory:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(directory)
                shp = next(Path(directory).rglob("*.shp"))
                reader = shapefile.Reader(str(shp))
                fields = [field[0] for field in reader.fields[1:]]
                count = 0
                for raw in reader.iterRecords():
                    rec = dict(zip(fields, raw))
                    lat, lon = float(rec["LAT"]), float(rec["LONG"])
                    date = rec["DATE"].isoformat()
                    if not (24 <= lat <= 50 and -125 <= lon <= -66 and
                            date.startswith(str(year)) and year < 2019):
                        continue
                    writer.writerow([year, date, rec["GMT"], lat, lon,
                                     rec.get("BT4TEMP", rec.get("TEMP")), rec["FRP"],
                                     rec["CONF"], rec["SAT_SRC"]])
                    count += 1
            counts[str(year)] = count
            manifest["sources"].append({"url": url, "file": str(archive.relative_to(ROOT)),
                                         "sha256": sha, "records_in_region": count})
    manifest["records_total"] = sum(counts.values())
    manifest["output"] = str((OUT / "detections.csv.gz").relative_to(ROOT))
    manifest["allowed_region"] = "CONUS: latitude 24..50, longitude -125..-66"
    manifest["allowed_observation_years"] = "before 2019"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts), "total", manifest["records_total"])


if __name__ == "__main__":
    main()

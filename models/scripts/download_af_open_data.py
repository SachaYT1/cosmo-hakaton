"""Download the audited, public USFS and NOAA AF source files without login.

Only the fixed historical CONUS archives and one Australian winter granule are
in scope. Expand this allowlist only after checking new dates and geography.
"""
from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://noaa-nesdis-snpp-pds.s3.amazonaws.com/"
USFS = {
    2017: "https://fsapps.nwcg.gov/afm/data_viirs/fireptdata/viirs-af_fire_2017_365_conus_shapefile.zip",
    2018: "https://fsapps.nwcg.gov/afm/data_viirs_iband/fireptdata/viirs_iband_fire_2018_365_conus_shapefile.zip",
}
PRODUCTS = [*(f"VIIRS-I{i}-SDR" for i in range(1, 6)),
            "VIIRS-IMG-GEO-TC", "VIIRS-AF-Iband-EDR"]


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=90) as response:
                response.raise_for_status()
                partial = path.with_suffix(path.suffix + ".part")
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                partial.replace(path)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2)


def object_key(product: str) -> str:
    prefix = f"{product}/2022/12/15/"
    response = requests.get(BASE, params={"list-type": 2, "prefix": prefix,
                                          "max-keys": 1000}, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    token = "_s202212150429233_" if product == "VIIRS-AF-Iband-EDR" else "_t0429233_"
    matches = [node.findtext("s:Key", namespaces=ns)
               for node in root.findall(".//s:Contents", ns)
               if token in node.findtext("s:Key", namespaces=ns)]
    if len(matches) != 1:
        raise ValueError(f"Expected one approved granule for {product}; found {len(matches)}")
    return matches[0]


def main() -> None:
    points = ROOT / "data/external/usfs_viirs_conus"
    for year, url in USFS.items():
        fetch(url, points / f"usfs{year}.zip")
    output = ROOT / "data/external/noaa_viirs/2022-12-15/t0429233"
    records = []
    for product in PRODUCTS:
        key = object_key(product)
        path = output / Path(key).name
        fetch(BASE + key, path)
        records.append({"product": product, "key": key,
                        "file": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(product, path.stat().st_size, flush=True)
    (output / "downloads.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

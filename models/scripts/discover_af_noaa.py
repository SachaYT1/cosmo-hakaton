"""Discover and download useful NOAA VIIRS winter granules without login.

The sparse AF EDR files are screened first. Only granules containing detections
inside explicitly approved, remote regions are followed by the much larger
I1-I5 and geolocation downloads. Raw data are later filtered tile by tile by
``prepare_af_noaa``; this script never accesses competition test metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import requests

from firemon.external_af import SAFE_REGIONS, safe_date

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://noaa-nesdis-snpp-pds.s3.amazonaws.com/"
AF = "VIIRS-AF-Iband-EDR"
PRODUCTS = [*(f"VIIRS-I{i}-SDR" for i in range(1, 6)), "VIIRS-IMG-GEO-TC"]
DEFAULT_DATES = [
    "2022-12-10", "2022-12-15", "2022-12-20", "2022-12-25", "2022-12-30",
    "2023-01-05", "2023-01-10", "2023-01-15", "2023-01-20", "2023-01-25",
    "2023-02-01", "2023-02-08", "2023-02-15", "2023-02-22",
]
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
SCREEN_CACHE = ROOT / "cache/af_noaa_screening.json"


def list_objects(session: requests.Session, prefix: str) -> list[tuple[str, int]]:
    result, token = [], None
    while True:
        params = {"list-type": 2, "prefix": prefix, "max-keys": 1000}
        if token:
            params["continuation-token"] = token
        response = session.get(BASE, params=params, timeout=45)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall(".//s:Contents", NS):
            result.append((item.findtext("s:Key", namespaces=NS),
                           int(item.findtext("s:Size", namespaces=NS))))
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            return result


def get(session: requests.Session, key: str) -> bytes:
    for attempt in range(5):
        try:
            response = session.get(BASE + key, timeout=90)
            response.raise_for_status()
            return response.content
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                raise
            if attempt == 4:
                raise
            time.sleep(2)
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def region_counts(content: bytes) -> tuple[dict[str, int], int]:
    with h5py.File(io.BytesIO(content)) as source:
        lat = source["Fire Pixels/FP_latitude"][:]
        lon = source["Fire Pixels/FP_longitude"][:]
        category = source["Fire Pixels/FP_PersistentAnomalyCategory"][:]
    counts = {}
    industrial = 0
    for name, (south, north, west, east) in SAFE_REGIONS.items():
        inside = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
        counts[name] = int(inside.sum())
        industrial += int((inside & np.isin(category, [1, 3, 4])).sum())
    return counts, industrial


def matching_key(objects: list[tuple[str, int]], token: str, product: str) -> tuple[str, int]:
    matches = [item for item in objects if token in item[0]]
    if len(matches) != 1:
        raise ValueError(f"Expected one {product} object for {token}, found {len(matches)}")
    return matches[0]


def existing_af_keys() -> set[str]:
    keys = set()
    for path in (ROOT / "data/external/noaa_viirs").glob("*/*/downloads.json"):
        for item in json.loads(path.read_text()):
            if item["product"] == AF:
                keys.add(item["key"])
    return keys


def read_screen_cache() -> dict[str, dict]:
    if not SCREEN_CACHE.exists():
        return {}
    try:
        return json.loads(SCREEN_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_screen_cache(cache: dict[str, dict]) -> None:
    SCREEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SCREEN_CACHE.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, separators=(",", ":")) + "\n")
    temporary.replace(SCREEN_CACHE)


def main(dates: list[str], max_granules: int, min_detections: int,
         scan_stride: int, max_gb: float) -> None:
    if any(not safe_date(value) for value in dates):
        raise ValueError("Every date must be in November-March of 2019-2025")
    if min(max_granules, min_detections, scan_stride) < 1 or max_gb <= 0:
        raise ValueError("Limits must be positive")
    session = requests.Session()
    already = existing_af_keys()
    screen_cache = read_screen_cache()
    candidates = []
    for observed in dates:
        year, month, day = observed.split("-")
        prefix = f"{AF}/{year}/{month}/{day}/"
        objects = list_objects(session, prefix)
        print(f"{observed}: screening {len(objects[::scan_stride])}/{len(objects)} AF files",
              flush=True)
        selected_objects = objects[::scan_stride]
        for index, (key, size) in enumerate(selected_objects, 1):
            if key in already:
                continue
            cached = screen_cache.get(key)
            if cached is None:
                try:
                    content = get(session, key)
                    counts, industrial = region_counts(content)
                    cached = {"counts": counts, "industrial": industrial}
                except (requests.RequestException, OSError, KeyError, ValueError) as error:
                    print(f"skip unavailable AF object: {key}", flush=True)
                    response = getattr(error, "response", None)
                    if response is not None and response.status_code == 404:
                        cached = {"error": "HTTP 404: stale S3 listing key"}
                    else:
                        continue  # retry transient network/HDF errors on the next run
                screen_cache[key] = cached
                if index % 20 == 0:
                    write_screen_cache(screen_cache)
            if "error" in cached:
                continue
            counts = cached["counts"]
            industrial = cached["industrial"]
            total = sum(counts.values())
            if total >= min_detections:
                candidates.append({"date": observed, "key": key, "size": size,
                                   "counts": counts, "industrial": industrial,
                                   "score": total + 20 * industrial})
        write_screen_cache(screen_cache)
        print(f"{observed}: checked {len(selected_objects)}, candidates so far "
              f"{len(candidates)}", flush=True)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    downloaded, used_bytes = [], 0
    object_cache: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for candidate in candidates:
        if len(downloaded) >= max_granules:
            break
        match = re.search(r"_s(20\d{6})(\d{7})_", candidate["key"])
        if not match:
            continue
        _, start = match.groups()
        observed = candidate["date"]
        y, m, d = observed.split("-")
        token = f"_t{start}_"
        selected = [(AF, candidate["key"], candidate["size"])]
        try:
            for product in PRODUCTS:
                cache_key = (product, observed)
                if cache_key not in object_cache:
                    object_cache[cache_key] = list_objects(
                        session, f"{product}/{y}/{m}/{d}/")
                key, size = matching_key(object_cache[cache_key], token, product)
                selected.append((product, key, size))
        except (requests.RequestException, ValueError) as error:
            print(f"skip {candidate['key']}: {error}", flush=True)
            continue
        total_size = sum(item[2] for item in selected)
        if used_bytes + total_size > max_gb * 1024**3:
            continue
        folder = ROOT / f"data/external/noaa_viirs/{observed}/t{start}"
        folder.mkdir(parents=True, exist_ok=True)
        records, payloads = [], []
        try:
            for product, key, expected_size in selected:
                payload = get(session, key)
                if len(payload) != expected_size:
                    raise ValueError(f"Size mismatch for {key}")
                payloads.append((product, key, payload))
        except (requests.RequestException, ValueError) as error:
            print(f"skip incomplete granule {candidate['key']}: {error}", flush=True)
            continue
        for product, key, payload in payloads:
            path = folder / Path(key).name
            path.write_bytes(payload)
            records.append({"product": product, "key": key,
                            "file": str(path.relative_to(ROOT)), "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
        (folder / "downloads.json").write_text(json.dumps(records, indent=2) + "\n")
        used_bytes += total_size
        downloaded.append({key: candidate[key] for key in
                           ("date", "key", "counts", "industrial", "score")})
        print(f"downloaded {observed} t{start}: {total_size/1024**2:.1f} MiB, "
              f"detections={sum(candidate['counts'].values())}", flush=True)
    report = {"dates": dates, "scan_stride": scan_stride,
              "min_detections": min_detections, "downloaded": downloaded,
              "bytes": used_bytes}
    path = ROOT / "reports/af_noaa_discovery.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected {len(downloaded)} granules, {used_bytes/1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--max-granules", type=int, default=24)
    parser.add_argument("--min-detections", type=int, default=10)
    parser.add_argument("--scan-stride", type=int, default=4,
                        help="screen every Nth orbit fragment to bound traffic")
    parser.add_argument("--max-gb", type=float, default=8.0)
    args = parser.parse_args()
    main(args.dates, args.max_granules, args.min_detections,
         args.scan_stride, args.max_gb)

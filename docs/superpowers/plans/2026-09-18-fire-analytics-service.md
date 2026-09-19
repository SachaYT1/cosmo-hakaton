# Fire Analytics Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI + Leaflet сервис: по полигону/bbox и интервалу дат возвращает термоточки, векторные контуры гарей и аналитическую справку (Раздел 4 критериев, 12 баллов).

**Architecture:** Оффлайн ingest конвертирует GT-маски train-чипов (или RLE-предсказания моделей) в GeoPackage (слои `hotspots`, `burn_contours`, EPSG:4326, площади посчитаны в UTM). FastAPI грузит каталог в память (GeoDataFrame + sindex), отдаёт REST API и статическую Leaflet-страницу.

**Tech Stack:** Python 3.12 (uv), FastAPI, uvicorn, geopandas/pyogrio, rasterio, shapely 2, pyproj, pytest + httpx (TestClient), Leaflet 1.9.4 + leaflet-draw 1.0.4 (CDN).

**Verified data facts** (не перепроверять, уже проверено):
- Маски: `train/af/masks/{chip_id}_MASK.tif` (0/1), `train/bs/masks/{chip_id}_MASK.tif` (0–3).
- `train/{af,bs}/meta.csv`: колонки `chip_id, epsg (32637.0/32638.0), x_min, y_min, x_max, y_max, width, height, gsd, acq_datetime (AF), satellite (AF), date_pre, date_post (BS), fire_event_id (BS), burn_area_ha, sev1_px, sev2_px, sev3_px`.
- AF: 420 чипов, 256×256 @375 м; BS: 224 чипа, 512×512 @20 м (пиксель = 0.04 га).
- AOI: `fire-aoi/fire_monitoring_aoi.geojson`, 3 фичи, у границы мониторинга `feature.id == "aoi"`.
- RLE (постановка): пиксели нумеруются построчно слева направо, сверху вниз, **с 1**; пары «старт длина» через пробел.

**Working directory:** все команды выполняются из `service/`; пути к данным задаются относительно корня репозитория.

---

### Task 1: Scaffold uv project

**Files:**
- Create: `service/pyproject.toml`
- Create: `service/.python-version`
- Create: `service/app/__init__.py`
- Create: `service/tests/__init__.py`
- Create: `service/conftest.py` (пустой пока)

- [ ] **Step 1: Create `service/pyproject.toml`**

```toml
[project]
name = "fire-analytics-service"
version = "0.1.0"
description = "Информационно-аналитический сервис мониторинга природных пожаров (КосмоХакатон 2026)"
requires-python = ">=3.11,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "geopandas>=1.0",
    "pyogrio>=0.9",
    "rasterio>=1.3",
    "shapely>=2.0",
    "pyproj>=3.6",
    "pandas>=2.2",
    "numpy>=1.26",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "httpx>=0.27",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package files**

`service/app/__init__.py` — пустой. `service/tests/__init__.py` — пустой. `service/conftest.py` — пока пустой (заполнится в Task 5). `service/.python-version` с содержимым `3.12`.

- [ ] **Step 3: Sync environment and verify**

Run (from `service/`): `uv sync`
Expected: создаётся `.venv` и `uv.lock`, все зависимости ставятся без ошибок.

Run: `uv run python -c "import geopandas, rasterio, fastapi; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add service/pyproject.toml service/uv.lock service/.python-version service/app/__init__.py service/tests/__init__.py service/conftest.py
git commit -m "chore(service): scaffold uv project for analytics service"
```

---

### Task 2: RLE codec

**Files:**
- Create: `service/app/rle.py`
- Test: `service/tests/test_rle.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_rle.py`:

```python
import numpy as np

from app.rle import decode_rle, encode_rle


def test_decode_known_pairs():
    # Пиксели нумеруются с 1, построчно: "1 3" -> пиксели 1..3, "10 2" -> 10..11.
    mask = decode_rle("1 3 10 2", width=4, height=4)
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[0, 0:3] = 1          # пиксели 1,2,3
    expected[2, 1:3] = 1          # пиксели 10,11 (строка 3, колонки 2-3)
    assert mask.dtype == np.uint8
    assert (mask == expected).all()


def test_decode_empty_string_gives_zeros():
    mask = decode_rle("", width=8, height=8)
    assert mask.shape == (8, 8)
    assert mask.sum() == 0


def test_encode_known_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0:3] = 1
    mask[2, 1:3] = 1
    assert encode_rle(mask) == "1 3 10 2"


def test_roundtrip_random():
    rng = np.random.default_rng(42)
    mask = (rng.random((16, 16)) > 0.7).astype(np.uint8)
    assert (decode_rle(encode_rle(mask), 16, 16) == mask).all()


def test_encode_empty_mask():
    assert encode_rle(np.zeros((4, 4), dtype=np.uint8)) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `service/`): `uv run pytest tests/test_rle.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'app.rle'`

- [ ] **Step 3: Implement `service/app/rle.py`**

```python
"""RLE-кодек по правилам постановки: нумерация пикселей построчно, с единицы."""

import numpy as np


def decode_rle(rle: str, width: int, height: int) -> np.ndarray:
    """Decode "start length ..." pairs (1-based, row-major) into a (height, width) mask."""
    mask = np.zeros(width * height, dtype=np.uint8)
    if rle and rle.strip():
        values = [int(v) for v in rle.split()]
        if len(values) % 2 != 0:
            raise ValueError("RLE must contain an even number of values")
        for start, length in zip(values[0::2], values[1::2]):
            if start < 1 or start - 1 + length > mask.size:
                raise ValueError(f"RLE run out of bounds: start={start} length={length}")
            mask[start - 1 : start - 1 + length] = 1
    return mask.reshape(height, width)


def encode_rle(mask: np.ndarray) -> str:
    """Encode a binary mask into "start length ..." pairs (1-based, row-major)."""
    flat = (np.asarray(mask).ravel() > 0).astype(np.int8)
    padded = np.concatenate([[0], flat, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    lengths = changes[1::2] - starts
    return " ".join(f"{s + 1} {l}" for s, l in zip(starts, lengths))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rle.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add service/app/rle.py service/tests/test_rle.py
git commit -m "feat(service): RLE codec per competition rules"
```

---

### Task 3: Geo helpers (chip grid, hotspot points, severity polygons)

**Files:**
- Create: `service/app/geo.py`
- Test: `service/tests/test_geo.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_geo.py`:

```python
import numpy as np
import pytest
from pyproj import Transformer

from app.geo import ChipGrid, hotspot_points, severity_polygons

GRID = ChipGrid(
    epsg=32637,
    x_min=384000.0, y_min=5280000.0,
    x_max=384000.0 + 8 * 375.0, y_max=5280000.0 + 8 * 375.0,
    width=8, height=8,
)


def test_pixel_area_ha():
    assert GRID.pixel_area_ha == pytest.approx(375 * 375 / 10_000)  # 14.0625 га
    bs = ChipGrid(32638, 0, 0, 16 * 20.0, 16 * 20.0, 16, 16)
    assert bs.pixel_area_ha == pytest.approx(0.04)


def test_hotspot_points_centers():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1  # верхний левый пиксель: центр (x_min + 187.5, y_max - 187.5)
    pts = hotspot_points(mask, GRID)
    assert len(pts) == 1
    t = Transformer.from_crs("EPSG:32637", "EPSG:4326", always_xy=True)
    exp_lon, exp_lat = t.transform(GRID.x_min + 187.5, GRID.y_max - 187.5)
    assert pts[0][0] == pytest.approx(exp_lon, abs=1e-9)
    assert pts[0][1] == pytest.approx(exp_lat, abs=1e-9)


def test_hotspot_points_empty():
    assert hotspot_points(np.zeros((8, 8), dtype=np.uint8), GRID) == []


def test_severity_polygons_areas_in_utm():
    grid = ChipGrid(32638, 491520.0, 5376000.0, 491520.0 + 16 * 20.0, 5376000.0 + 16 * 20.0, 16, 16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:2, 0:5] = 1   # 10 px = 0.4 га
    mask[5, 5:10] = 2    # 5 px = 0.2 га
    mask[10, 10] = 3     # 1 px = 0.04 га
    polys = severity_polygons(mask, grid)
    by_sev = {}
    for p in polys:
        by_sev[p["severity"]] = by_sev.get(p["severity"], 0.0) + p["area_ha"]
    assert by_sev[1] == pytest.approx(0.4, abs=1e-6)
    assert by_sev[2] == pytest.approx(0.2, abs=1e-6)
    assert by_sev[3] == pytest.approx(0.04, abs=1e-6)
    # геометрия в WGS84: долгота в разумных пределах зоны 38N
    lon = polys[0]["geometry"].centroid.x
    assert 42.0 < lon < 48.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_geo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.geo'`

- [ ] **Step 3: Implement `service/app/geo.py`**

```python
"""Геопривязка чипов и конвертация масок в векторные объекты."""

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_bounds, xy
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


@dataclass(frozen=True)
class ChipGrid:
    """Сетка чипа: UTM-проекция и охват из meta.csv."""

    epsg: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: int
    height: int

    @property
    def transform(self):
        return from_bounds(self.x_min, self.y_min, self.x_max, self.y_max, self.width, self.height)

    @property
    def pixel_area_ha(self) -> float:
        px_w = (self.x_max - self.x_min) / self.width
        px_h = (self.y_max - self.y_min) / self.height
        return px_w * px_h / 10_000.0


def to_wgs84(epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def hotspot_points(mask: np.ndarray, grid: ChipGrid) -> list[tuple[float, float]]:
    """Центры пикселей mask==1 в координатах (lon, lat)."""
    rows, cols = np.nonzero(mask == 1)
    if len(rows) == 0:
        return []
    xs, ys = xy(grid.transform, rows, cols)
    lons, lats = to_wgs84(grid.epsg).transform(xs, ys)
    return list(zip(lons, lats))


def severity_polygons(mask: np.ndarray, grid: ChipGrid) -> list[dict]:
    """Полигонизация классов 1..3. Площадь считается в родном UTM (без искажений)."""
    out: list[dict] = []
    transformer = to_wgs84(grid.epsg)
    for sev in (1, 2, 3):
        class_mask = mask == sev
        if not class_mask.any():
            continue
        shapes = features.shapes(class_mask.astype(np.uint8), mask=class_mask, transform=grid.transform)
        for geom, _value in shapes:
            poly_utm = shape(geom)
            poly_wgs = shp_transform(transformer.transform, poly_utm)
            out.append(
                {
                    "severity": sev,
                    "geometry": poly_wgs,
                    "area_ha": round(poly_utm.area / 10_000.0, 4),
                }
            )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_geo.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add service/app/geo.py service/tests/test_geo.py
git commit -m "feat(service): chip grid and mask vectorization helpers"
```

---

### Task 4: Ingest core (per-chip GeoDataFrames)

**Files:**
- Create: `service/app/ingest_core.py`
- Test: `service/tests/test_ingest_core.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_ingest_core.py`:

```python
import numpy as np
import pandas as pd
import pytest

from app.ingest_core import contours_gdf, grid_from_meta, hotspots_gdf

AF_ROW = pd.Series(
    {
        "chip_id": "AF_ts_000001",
        "epsg": 32637.0,
        "x_min": 384000.0, "y_min": 5280000.0,
        "x_max": 384000.0 + 8 * 375.0, "y_max": 5280000.0 + 8 * 375.0,
        "width": 8, "height": 8,
        "acq_datetime": "2021-07-15T10:00:00+00:00",
        "satellite": "SNPP",
    }
)

BS_ROW = pd.Series(
    {
        "chip_id": "BS_ts_000001",
        "fire_event_id": "FE00001",
        "epsg": 32638.0,
        "x_min": 491520.0, "y_min": 5376000.0,
        "x_max": 491520.0 + 16 * 20.0, "y_max": 5376000.0 + 16 * 20.0,
        "width": 16, "height": 16,
        "date_pre": "2021-07-01", "date_post": "2021-07-20",
    }
)


def _bs_mask():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[0:2, 0:5] = 1
    m[5, 5:10] = 2
    m[10, 10] = 3
    return m


def test_grid_from_meta_casts_types():
    grid = grid_from_meta(AF_ROW)
    assert grid.epsg == 32637 and grid.width == 8


def test_hotspots_gdf():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = mask[3, 4] = mask[7, 7] = 1
    gdf = hotspots_gdf(AF_ROW, mask)
    assert len(gdf) == 3
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf["chip_id"]) == {"AF_ts_000001"}
    assert set(gdf.columns) == {"chip_id", "acq_datetime", "satellite", "geometry"}


def test_hotspots_gdf_empty_mask():
    gdf = hotspots_gdf(AF_ROW, np.zeros((8, 8), dtype=np.uint8))
    assert len(gdf) == 0


def test_contours_gdf():
    gdf = contours_gdf(BS_ROW, _bs_mask())
    assert gdf.crs.to_epsg() == 4326
    assert set(gdf.columns) == {
        "contour_id", "chip_id", "fire_event_id", "severity", "area_ha",
        "date_pre", "date_post", "epsg", "geometry",
    }
    assert gdf["contour_id"].is_unique
    assert gdf.groupby("severity")["area_ha"].sum().to_dict() == pytest.approx(
        {1: 0.4, 2: 0.2, 3: 0.04}
    )
    assert set(gdf["epsg"]) == {32638}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest_core'`

- [ ] **Step 3: Implement `service/app/ingest_core.py`**

```python
"""Сборка GeoDataFrame термоточек и контуров гарей из маски одного чипа."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from app.geo import ChipGrid, hotspot_points, severity_polygons

CRS_WGS84 = "EPSG:4326"


def grid_from_meta(row: pd.Series) -> ChipGrid:
    return ChipGrid(
        epsg=int(float(row["epsg"])),
        x_min=float(row["x_min"]),
        y_min=float(row["y_min"]),
        x_max=float(row["x_max"]),
        y_max=float(row["y_max"]),
        width=int(row["width"]),
        height=int(row["height"]),
    )


def hotspots_gdf(row: pd.Series, mask: np.ndarray) -> gpd.GeoDataFrame:
    pts = hotspot_points(mask, grid_from_meta(row))
    data = {
        "chip_id": [str(row["chip_id"])] * len(pts),
        "acq_datetime": [str(row["acq_datetime"])] * len(pts),
        "satellite": [str(row["satellite"])] * len(pts),
    }
    geometry = [Point(lon, lat) for lon, lat in pts]
    return gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_WGS84)


def contours_gdf(row: pd.Series, mask: np.ndarray) -> gpd.GeoDataFrame:
    grid = grid_from_meta(row)
    polys = severity_polygons(mask, grid)
    data = {
        "contour_id": [f"{row['chip_id']}_{i:04d}" for i in range(len(polys))],
        "chip_id": [str(row["chip_id"])] * len(polys),
        "fire_event_id": [str(row["fire_event_id"])] * len(polys),
        "severity": [int(p["severity"]) for p in polys],
        "area_ha": [float(p["area_ha"]) for p in polys],
        "date_pre": [str(row["date_pre"])] * len(polys),
        "date_post": [str(row["date_post"])] * len(polys),
        "epsg": [grid.epsg] * len(polys),
    }
    geometry = [p["geometry"] for p in polys]
    return gpd.GeoDataFrame(data, geometry=geometry, crs=CRS_WGS84)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest_core.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add service/app/ingest_core.py service/tests/test_ingest_core.py
git commit -m "feat(service): per-chip hotspot and contour GeoDataFrames"
```

---

### Task 5: Catalog builder (GT mode) + CLI + test fixtures

**Files:**
- Create: `service/app/ingest.py`
- Create: `service/scripts/ingest.py`
- Modify: `service/conftest.py` (фикстуры синтетического датасета)
- Test: `service/tests/test_ingest.py`

- [ ] **Step 1: Fill `service/conftest.py` with synthetic dataset fixtures**

```python
"""Синтетический мини-датасет в формате выданных данных (1 AF-чип + 1 BS-чип)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

AF_BOUNDS = (384000.0, 5280000.0, 384000.0 + 8 * 375.0, 5280000.0 + 8 * 375.0)
BS_BOUNDS = (491520.0, 5376000.0, 491520.0 + 16 * 20.0, 5376000.0 + 16 * 20.0)

AF_META = {
    "chip_id": "AF_ts_000001", "kind": "af", "fire_event_id": "", "region": "",
    "epsg": 32637.0,
    "x_min": AF_BOUNDS[0], "y_min": AF_BOUNDS[1], "x_max": AF_BOUNDS[2], "y_max": AF_BOUNDS[3],
    "width": 8, "height": 8, "gsd": 375,
    "acq_datetime": "2021-07-15T10:00:00+00:00", "satellite": "SNPP",
    "date_pre": "", "date_post": "", "s1_date_pre": "", "s1_date_post": "",
    "valid_frac": 1.0, "cloud_frac": "", "landcover_top": "",
    "n_fire_px": 3, "burn_area_ha": "", "sev1_px": "", "sev2_px": "", "sev3_px": "",
}

BS_META = {
    "chip_id": "BS_ts_000001", "kind": "bs", "fire_event_id": "FE00001", "region": "",
    "epsg": 32638.0,
    "x_min": BS_BOUNDS[0], "y_min": BS_BOUNDS[1], "x_max": BS_BOUNDS[2], "y_max": BS_BOUNDS[3],
    "width": 16, "height": 16, "gsd": 20,
    "acq_datetime": "", "satellite": "",
    "date_pre": "2021-07-01", "date_post": "2021-07-20",
    "s1_date_pre": "2021-06-30", "s1_date_post": "2021-07-21",
    "valid_frac": 1.0, "cloud_frac": 0.0, "landcover_top": "",
    "n_fire_px": "", "burn_area_ha": 0.64, "sev1_px": 10, "sev2_px": 5, "sev3_px": 1,
}


def af_mask() -> np.ndarray:
    m = np.zeros((8, 8), dtype=np.uint8)
    m[0, 0] = m[3, 4] = m[7, 7] = 1
    return m


def bs_mask() -> np.ndarray:
    m = np.zeros((16, 16), dtype=np.uint8)
    m[0:2, 0:5] = 1   # 10 px -> 0.4 га (слабая)
    m[5, 5:10] = 2    # 5 px  -> 0.2 га (средняя)
    m[10, 10] = 3     # 1 px  -> 0.04 га (сильная)
    return m


def _write_mask(path: Path, arr: np.ndarray, epsg: int, bounds: tuple) -> None:
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8",
        crs=f"EPSG:{epsg}", transform=from_bounds(*bounds, w, h),
    ) as dst:
        dst.write(arr, 1)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    (root / "af" / "masks").mkdir(parents=True)
    (root / "bs" / "masks").mkdir(parents=True)
    pd.DataFrame([AF_META]).to_csv(root / "af" / "meta.csv", index=False)
    pd.DataFrame([BS_META]).to_csv(root / "bs" / "meta.csv", index=False)
    _write_mask(root / "af" / "masks" / "AF_ts_000001_MASK.tif", af_mask(), 32637, AF_BOUNDS)
    _write_mask(root / "bs" / "masks" / "BS_ts_000001_MASK.tif", bs_mask(), 32638, BS_BOUNDS)
    return root


@pytest.fixture(scope="session")
def catalog_path(dataset_dir, tmp_path_factory) -> Path:
    from app.ingest import build_catalog

    out = tmp_path_factory.mktemp("catalog") / "catalog.gpkg"
    build_catalog(dataset_dir, out, source="gt")
    return out
```

- [ ] **Step 2: Write the failing tests**

`service/tests/test_ingest.py`:

```python
import geopandas as gpd
import pytest

from app.ingest import build_catalog


def test_build_catalog_gt(dataset_dir, tmp_path):
    out = tmp_path / "catalog.gpkg"
    build_catalog(dataset_dir, out, source="gt")
    assert out.exists()

    hotspots = gpd.read_file(out, layer="hotspots")
    assert len(hotspots) == 3
    assert hotspots.crs.to_epsg() == 4326
    assert {"chip_id", "acq_datetime", "satellite"} <= set(hotspots.columns)

    contours = gpd.read_file(out, layer="burn_contours")
    assert contours.crs.to_epsg() == 4326
    sums = contours.groupby("severity")["area_ha"].sum().to_dict()
    assert sums == pytest.approx({1: 0.4, 2: 0.2, 3: 0.04})
    # согласованность с meta.csv: burn_area_ha = 0.64
    assert contours["area_ha"].sum() == pytest.approx(0.64)


def test_build_catalog_overwrites(dataset_dir, tmp_path):
    out = tmp_path / "catalog.gpkg"
    build_catalog(dataset_dir, out, source="gt")
    build_catalog(dataset_dir, out, source="gt")  # повторный запуск не падает и не дублирует
    hotspots = gpd.read_file(out, layer="hotspots")
    assert len(hotspots) == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 4: Implement `service/app/ingest.py`**

```python
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
```

- [ ] **Step 5: Implement CLI wrapper `service/scripts/ingest.py`**

```python
"""CLI: сборка каталога сервиса из GT-масок или предсказаний моделей.

Примеры:
    uv run python scripts/ingest.py --data-dir ../train
    uv run python scripts/ingest.py --data-dir ../train --source predictions --submission preds.csv
"""

import argparse
import logging
from pathlib import Path

from app.ingest import build_catalog


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build service catalog (GeoPackage)")
    parser.add_argument("--data-dir", type=Path, default=Path("../train"),
                        help="каталог с af/ и bs/ (структура train/)")
    parser.add_argument("--source", choices=["gt", "predictions"], default="gt")
    parser.add_argument("--submission", type=Path, default=None,
                        help="submission.csv с RLE-предсказаниями (для --source predictions)")
    parser.add_argument("--output", type=Path, default=Path("data/catalog.gpkg"))
    args = parser.parse_args()
    build_catalog(args.data_dir, args.output, source=args.source, submission=args.submission)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add service/app/ingest.py service/scripts/ingest.py service/conftest.py service/tests/test_ingest.py
git commit -m "feat(service): catalog builder with GT mode and CLI"
```

---

### Task 6: Predictions mode (RLE submission ingest)

**Files:**
- Modify: `service/tests/test_ingest.py` (добавить тест в конец файла)

- [ ] **Step 1: Write the failing test (append to `service/tests/test_ingest.py`)**

```python
def test_build_catalog_predictions_matches_gt(dataset_dir, tmp_path):
    """Ingest RLE-предсказаний, идентичных GT-маскам, даёт те же суммы площадей."""
    import numpy as np
    import pandas as pd

    from app.rle import encode_rle
    from conftest import af_mask, bs_mask

    rows = [{"chip_id": "AF_ts_000001", "class_id": 1, "rle": encode_rle(af_mask())}]
    bs = bs_mask()
    for cls in (1, 2, 3):
        rows.append({"chip_id": "BS_ts_000001", "class_id": cls,
                     "rle": encode_rle((bs == cls).astype(np.uint8))})
    sub_path = tmp_path / "submission.csv"
    pd.DataFrame(rows).to_csv(sub_path, index=False)

    out = tmp_path / "catalog_pred.gpkg"
    build_catalog(dataset_dir, out, source="predictions", submission=sub_path)

    import geopandas as gpd
    hotspots = gpd.read_file(out, layer="hotspots")
    contours = gpd.read_file(out, layer="burn_contours")
    assert len(hotspots) == 3
    assert contours.groupby("severity")["area_ha"].sum().to_dict() == pytest.approx(
        {1: 0.4, 2: 0.2, 3: 0.04}
    )
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 3 passed — режим predictions уже реализован в Task 5; если тест падает, чинить `mask_from_submission`/`_chip_mask`, а не тест.

- [ ] **Step 3: Commit**

```bash
git add service/tests/test_ingest.py
git commit -m "test(service): predictions-mode ingest matches GT areas"
```

---

### Task 7: Catalog store + report builder

**Files:**
- Create: `service/app/store.py`
- Create: `service/app/reports.py`
- Test: `service/tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_store.py`:

```python
import pandas as pd
import pytest
from shapely.geometry import box

from app.reports import build_report, clipped_area_by_severity
from app.store import Catalog

WIDE_BOX = box(30.0, 40.0, 60.0, 60.0)  # покрывает оба тестовых чипа
D0 = pd.Timestamp("2021-01-01")
D1 = pd.Timestamp("2021-12-31")


@pytest.fixture(scope="module")
def catalog(catalog_path) -> Catalog:
    return Catalog.load(catalog_path)


def test_query_hotspots_spatial_and_dates(catalog):
    assert len(catalog.query_hotspots(WIDE_BOX, D0, D1)) == 3
    # период без данных
    assert len(catalog.query_hotspots(WIDE_BOX, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))) == 0
    # полигон мимо данных
    assert len(catalog.query_hotspots(box(0, 0, 1, 1), D0, D1)) == 0


def test_query_contours(catalog):
    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    assert len(contours) >= 3
    assert contours["area_ha"].sum() == pytest.approx(0.64)
    # date_post=2021-07-20 вне интервала -> пусто
    assert len(catalog.query_contours(WIDE_BOX, D0, pd.Timestamp("2021-07-10"))) == 0


def test_clipped_area_full_cover(catalog):
    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    areas = clipped_area_by_severity(contours, WIDE_BOX)
    assert areas[1] == pytest.approx(0.4, abs=1e-3)
    assert areas[2] == pytest.approx(0.2, abs=1e-3)
    assert areas[3] == pytest.approx(0.04, abs=1e-3)


def test_clipped_area_partial_cover(catalog):
    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    sev1 = contours[contours["severity"] == 1]
    minx, miny, maxx, maxy = sev1.total_bounds
    west_half = box(minx, miny, (minx + maxx) / 2, maxy)
    areas = clipped_area_by_severity(contours, west_half)
    assert areas[1] == pytest.approx(0.2, abs=0.02)  # половина от 0.4 га


def test_build_report(catalog):
    from datetime import date

    contours = catalog.query_contours(WIDE_BOX, D0, D1)
    hotspots = catalog.query_hotspots(WIDE_BOX, D0, D1)
    report = build_report(contours, hotspots, WIDE_BOX, date(2021, 1, 1), date(2021, 12, 31))
    assert report["total_burned_area_ha"] == pytest.approx(0.64, abs=1e-2)
    assert report["hotspot_count"] == 3
    sev1 = next(r for r in report["by_severity"] if r["severity"] == 1)
    assert sev1["label"] == "слабая"
    assert sev1["share_pct"] == pytest.approx(62.5, abs=0.1)


def test_empty_report(catalog):
    from datetime import date

    empty_geom = box(0, 0, 1, 1)
    contours = catalog.query_contours(empty_geom, D0, D1)
    hotspots = catalog.query_hotspots(empty_geom, D0, D1)
    report = build_report(contours, hotspots, empty_geom, date(2021, 1, 1), date(2021, 12, 31))
    assert report["total_burned_area_ha"] == 0.0
    assert report["hotspot_count"] == 0


def test_meta(catalog):
    meta = catalog.meta()
    assert meta["hotspot_count"] == 3
    assert meta["date_min"] == "2021-07-15"
    assert meta["date_max"] == "2021-07-20"
    assert meta["demo"]["fire_event_id"] == "FE00001"
    assert len(meta["demo"]["bbox"]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.store'`

- [ ] **Step 3: Implement `service/app/store.py`**

```python
"""In-memory каталог: слои GeoPackage + пространственно-временные выборки."""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

CRS_WGS84 = "EPSG:4326"
HOTSPOT_SCHEMA = ["chip_id", "acq_datetime", "satellite"]
CONTOUR_SCHEMA = [
    "contour_id", "chip_id", "fire_event_id", "severity", "area_ha",
    "date_pre", "date_post", "epsg",
]


def _read_layer(path: Path, layer: str, columns: list[str]) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=layer)
    except Exception:
        logger.warning("layer %r not found in %s — using empty layer", layer, path)
        return gpd.GeoDataFrame({c: [] for c in columns}, geometry=[], crs=CRS_WGS84)


class Catalog:
    def __init__(self, hotspots: gpd.GeoDataFrame, contours: gpd.GeoDataFrame):
        self.hotspots = hotspots
        self.contours = contours
        self.hotspots["_date"] = (
            pd.to_datetime(self.hotspots["acq_datetime"], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        self.contours["_post_date"] = pd.to_datetime(
            self.contours["date_post"], errors="coerce"
        ).dt.normalize()

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        hotspots = _read_layer(path, "hotspots", HOTSPOT_SCHEMA)
        contours = _read_layer(path, "burn_contours", CONTOUR_SCHEMA)
        logger.info("catalog loaded: %d hotspots, %d contours", len(hotspots), len(contours))
        return cls(hotspots, contours)

    def query_hotspots(self, geom: BaseGeometry, date_from: pd.Timestamp, date_to: pd.Timestamp) -> gpd.GeoDataFrame:
        return self._query(self.hotspots, "_date", geom, date_from, date_to)

    def query_contours(self, geom: BaseGeometry, date_from: pd.Timestamp, date_to: pd.Timestamp) -> gpd.GeoDataFrame:
        """Контур попадает в выборку, если его date_post лежит в интервале."""
        return self._query(self.contours, "_post_date", geom, date_from, date_to)

    @staticmethod
    def _query(gdf: gpd.GeoDataFrame, date_col: str, geom, date_from, date_to) -> gpd.GeoDataFrame:
        if gdf.empty:
            return gdf.drop(columns=[date_col]).copy()
        idx = gdf.sindex.query(geom, predicate="intersects")
        sel = gdf.iloc[sorted(idx)]
        sel = sel[(sel[date_col] >= date_from) & (sel[date_col] <= date_to)]
        return sel.drop(columns=[date_col])

    def meta(self) -> dict:
        """Границы данных и демо-пример (крупнейший пожар) для UI."""
        dates = []
        if not self.hotspots.empty:
            dates += [self.hotspots["_date"].min(), self.hotspots["_date"].max()]
        if not self.contours.empty:
            dates += [self.contours["_post_date"].min(), self.contours["_post_date"].max()]
        date_min = min(dates).date().isoformat() if dates else "2019-04-01"
        date_max = max(dates).date().isoformat() if dates else "2025-10-31"

        demo = None
        if not self.contours.empty:
            event = self.contours.groupby("fire_event_id")["area_ha"].sum().idxmax()
            ev = self.contours[self.contours["fire_event_id"] == event]
            minx, miny, maxx, maxy = ev.total_bounds
            dx = (maxx - minx) * 0.2 or 0.05
            dy = (maxy - miny) * 0.2 or 0.05
            d0 = pd.to_datetime(ev["date_pre"], errors="coerce").min() - pd.Timedelta(days=14)
            d1 = ev["_post_date"].max() + pd.Timedelta(days=14)
            demo = {
                "fire_event_id": str(event),
                "bbox": [minx - dx, miny - dy, maxx + dx, maxy + dy],
                "date_from": d0.date().isoformat(),
                "date_to": d1.date().isoformat(),
            }
        return {
            "hotspot_count": int(len(self.hotspots)),
            "contour_count": int(len(self.contours)),
            "date_min": date_min,
            "date_max": date_max,
            "demo": demo,
        }
```

- [ ] **Step 4: Implement `service/app/reports.py`**

```python
"""Аналитическая справка: площади пересечения с полигоном запроса, счёт в UTM."""

from datetime import date

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

SEVERITY_LABELS = {1: "слабая", 2: "средняя", 3: "сильная"}


def clipped_area_by_severity(contours: gpd.GeoDataFrame, query_geom: BaseGeometry) -> dict[int, float]:
    """Площадь пересечения контуров с полигоном запроса, по степеням, в гектарах.

    Пересечение считается в EPSG:4326, затем каждая группа контуров
    репроецируется в родную UTM-зону чипа (атрибут epsg) — там площадь точна.
    """
    areas = {1: 0.0, 2: 0.0, 3: 0.0}
    if contours.empty:
        return areas
    clipped = contours.copy()
    clipped["geometry"] = clipped.geometry.intersection(query_geom)
    clipped = clipped[~clipped.geometry.is_empty]
    for epsg, group in clipped.groupby("epsg"):
        utm = group.to_crs(int(epsg))
        for sev, sev_group in utm.groupby("severity"):
            areas[int(sev)] += float(sev_group.geometry.area.sum()) / 10_000.0
    return areas


def build_report(
    contours: gpd.GeoDataFrame,
    hotspots: gpd.GeoDataFrame,
    query_geom: BaseGeometry,
    date_from: date,
    date_to: date,
) -> dict:
    areas = clipped_area_by_severity(contours, query_geom)
    total = sum(areas.values())
    by_severity = [
        {
            "severity": sev,
            "label": SEVERITY_LABELS[sev],
            "area_ha": round(area, 2),
            "share_pct": round(area / total * 100.0, 1) if total else 0.0,
        }
        for sev, area in sorted(areas.items())
    ]
    return {
        "period": {"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
        "total_burned_area_ha": round(total, 2),
        "by_severity": by_severity,
        "hotspot_count": int(len(hotspots)),
        "contour_count": int(len(contours)),
        "note": "Площади — пересечение контуров с полигоном запроса, расчёт в UTM-зоне чипа. "
                "Полная площадь каждого контура — атрибут area_ha в выгрузках.",
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add service/app/store.py service/app/reports.py service/tests/test_store.py
git commit -m "feat(service): in-memory catalog store and analytical report builder"
```

---

### Task 8: REST API (schemas, routes, app factory)

**Files:**
- Create: `service/app/schemas.py`
- Create: `service/app/api.py`
- Create: `service/app/main.py`
- Create: `service/app/static/index.html` (минимальная заглушка, полный UI — Task 10)
- Test: `service/tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

QUERY = {"bbox": [30.0, 40.0, 60.0, 60.0], "date_from": "2021-01-01", "date_to": "2021-12-31"}
POLYGON_QUERY = {
    "polygon": {
        "type": "Polygon",
        "coordinates": [[[30, 40], [60, 40], [60, 60], [30, 60], [30, 40]]],
    },
    "date_from": "2021-01-01",
    "date_to": "2021-12-31",
}


@pytest.fixture()
def client(catalog_path) -> TestClient:
    from app.main import create_app

    return TestClient(create_app(catalog_path))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_hotspots_bbox(client):
    r = client.post("/api/hotspots", json=QUERY)
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3
    props = fc["features"][0]["properties"]
    assert {"chip_id", "acq_datetime", "satellite"} <= set(props)


def test_hotspots_polygon(client):
    r = client.post("/api/hotspots", json=POLYGON_QUERY)
    assert len(r.json()["features"]) == 3


def test_hotspots_date_filter(client):
    r = client.post("/api/hotspots", json={**QUERY, "date_from": "2020-01-01", "date_to": "2020-12-31"})
    assert r.json()["features"] == []


def test_burned_areas(client):
    r = client.post("/api/burned-areas", json=QUERY)
    assert r.status_code == 200
    feats = r.json()["features"]
    assert len(feats) >= 3
    total = sum(f["properties"]["area_ha"] for f in feats)
    assert total == pytest.approx(0.64, abs=1e-3)
    assert {f["properties"]["severity"] for f in feats} == {1, 2, 3}


def test_report(client):
    r = client.post("/api/report", json=QUERY)
    assert r.status_code == 200
    rep = r.json()
    assert rep["total_burned_area_ha"] == pytest.approx(0.64, abs=0.01)
    assert rep["hotspot_count"] == 3
    assert len(rep["by_severity"]) == 3


def test_meta(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    m = r.json()
    assert m["hotspot_count"] == 3
    assert m["demo"]["fire_event_id"] == "FE00001"


def test_both_polygon_and_bbox_rejected(client):
    r = client.post("/api/hotspots", json={**QUERY, "polygon": POLYGON_QUERY["polygon"]})
    assert r.status_code == 422


def test_neither_polygon_nor_bbox_rejected(client):
    r = client.post("/api/hotspots", json={"date_from": "2021-01-01", "date_to": "2021-12-31"})
    assert r.status_code == 422


def test_invalid_geometry_rejected(client):
    r = client.post("/api/hotspots", json={**POLYGON_QUERY, "polygon": {"type": "Polygon", "coordinates": []}})
    assert r.status_code == 422


def test_dates_swapped_rejected(client):
    r = client.post("/api/hotspots", json={**QUERY, "date_from": "2021-12-31", "date_to": "2021-01-01"})
    assert r.status_code == 422


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="map"' in r.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement `service/app/schemas.py`**

```python
"""Pydantic-модели запросов."""

from datetime import date

from pydantic import BaseModel, model_validator


class SpatioTemporalQuery(BaseModel):
    """Полигон (GeoJSON geometry) ИЛИ bbox [minx, miny, maxx, maxy] в EPSG:4326 + интервал дат."""

    polygon: dict | None = None
    bbox: list[float] | None = None
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def _check(self) -> "SpatioTemporalQuery":
        if (self.polygon is None) == (self.bbox is None):
            raise ValueError("передайте ровно одно из polygon или bbox")
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox должен быть [minx, miny, maxx, maxy]")
        if self.date_from > self.date_to:
            raise ValueError("date_from позже date_to")
        return self
```

- [ ] **Step 4: Implement `service/app/api.py`**

```python
"""REST API: термоточки, контуры гарей, аналитическая справка, выгрузки."""

import json

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from app.reports import build_report
from app.schemas import SpatioTemporalQuery
from app.store import Catalog

router = APIRouter(prefix="/api")


def healthz() -> dict:
    return {"status": "ok"}


def _catalog(request: Request) -> Catalog:
    return request.app.state.catalog


def resolve_geometry(q: SpatioTemporalQuery) -> BaseGeometry:
    if q.bbox is not None:
        geom = box(*q.bbox)
    else:
        try:
            geom = shape(q.polygon)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"невалидная геометрия: {exc}") from exc
    if geom.is_empty or not geom.is_valid:
        raise HTTPException(status_code=422, detail="невалидная или пустая геометрия запроса")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise HTTPException(status_code=422, detail="ожидается Polygon или MultiPolygon")
    return geom


def _bounds(q: SpatioTemporalQuery) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(q.date_from), pd.Timestamp(q.date_to)


def feature_collection(gdf: gpd.GeoDataFrame) -> dict:
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(gdf.to_json(drop_id=True))


@router.post("/hotspots")
def hotspots(q: SpatioTemporalQuery, request: Request) -> dict:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    return feature_collection(_catalog(request).query_hotspots(geom, d0, d1))


@router.post("/burned-areas")
def burned_areas(q: SpatioTemporalQuery, request: Request) -> dict:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    return feature_collection(_catalog(request).query_contours(geom, d0, d1))


@router.post("/report")
def report(q: SpatioTemporalQuery, request: Request) -> dict:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    cat = _catalog(request)
    contours = cat.query_contours(geom, d0, d1)
    hs = cat.query_hotspots(geom, d0, d1)
    return build_report(contours, hs, geom, q.date_from, q.date_to)


@router.get("/meta")
def meta(request: Request) -> dict:
    return _catalog(request).meta()
```

- [ ] **Step 5: Implement `service/app/main.py`**

```python
"""Фабрика FastAPI-приложения: REST API + статический Leaflet-фронтенд."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import healthz, router
from app.store import Catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "data" / "catalog.gpkg"


def create_app(catalog_path: Path | None = None) -> FastAPI:
    path = Path(catalog_path or os.environ.get("CATALOG_PATH", DEFAULT_CATALOG))
    app = FastAPI(
        title="Мониторинг природных пожаров — информационно-аналитический сервис",
        version="0.1.0",
    )
    app.state.catalog = Catalog.load(path)
    app.include_router(router)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
```

- [ ] **Step 6: Create placeholder `service/app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>Мониторинг пожаров</title></head>
<body><div id="map"></div></body>
</html>
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 12 passed

Run: `uv run pytest -q`
Expected: все тесты проекта зелёные.

- [ ] **Step 8: Commit**

```bash
git add service/app/schemas.py service/app/api.py service/app/main.py service/app/static/index.html service/tests/test_api.py
git commit -m "feat(service): REST API with spatio-temporal queries and report"
```

---

### Task 9: Export endpoints (GeoJSON, Shapefile ZIP, report JSON/CSV)

**Files:**
- Create: `service/app/exports.py`
- Modify: `service/app/api.py` (добавить маршруты в конец файла)
- Test: `service/tests/test_exports.py`

- [ ] **Step 1: Write the failing tests**

`service/tests/test_exports.py`:

```python
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

QUERY = {"bbox": [30.0, 40.0, 60.0, 60.0], "date_from": "2021-01-01", "date_to": "2021-12-31"}
EMPTY_QUERY = {"bbox": [0.0, 0.0, 1.0, 1.0], "date_from": "2021-01-01", "date_to": "2021-12-31"}


@pytest.fixture()
def client(catalog_path) -> TestClient:
    from app.main import create_app

    return TestClient(create_app(catalog_path))


def test_export_geojson(client):
    r = client.post("/api/export/burned-areas.geojson", json=QUERY)
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    fc = json.loads(r.content)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 3


def test_export_geojson_empty_selection(client):
    r = client.post("/api/export/burned-areas.geojson", json=EMPTY_QUERY)
    assert r.status_code == 200
    assert json.loads(r.content)["features"] == []


def test_export_shapefile_zip(client):
    r = client.post("/api/export/burned-areas.shp.zip", json=QUERY)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"burned_areas.shp", "burned_areas.shx", "burned_areas.dbf", "burned_areas.prj"} <= names


def test_export_shapefile_empty_selection_404(client):
    r = client.post("/api/export/burned-areas.shp.zip", json=EMPTY_QUERY)
    assert r.status_code == 404


def test_export_report_json(client):
    r = client.post("/api/export/report.json", json=QUERY)
    assert r.status_code == 200
    rep = json.loads(r.content)
    assert rep["total_burned_area_ha"] == pytest.approx(0.64, abs=0.01)


def test_export_report_csv(client):
    r = client.post("/api/export/report.csv", json=QUERY)
    assert r.status_code == 200
    text = r.content.decode("utf-8-sig")
    assert text.splitlines()[0] == "severity,label,area_ha,share_pct"
    assert "total" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exports.py -v`
Expected: FAIL — 404 на маршрутах `/api/export/...` (роуты ещё не добавлены).

- [ ] **Step 3: Implement `service/app/exports.py`**

```python
"""Формирование файловых выгрузок: GeoJSON, Shapefile (zip), справка JSON/CSV."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd

# Имена колонок Shapefile ограничены 10 символами
SHP_RENAME = {"fire_event_id": "fire_event"}


def contours_geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    if gdf.empty:
        return json.dumps({"type": "FeatureCollection", "features": []}).encode()
    return gdf.to_json(drop_id=True).encode()


def contours_shapefile_zip(gdf: gpd.GeoDataFrame) -> bytes:
    """ZIP с shp/shx/dbf/prj/cpg. Для пустой выборки не вызывается (см. api)."""
    out = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        shp_path = Path(tmp) / "burned_areas.shp"
        gdf.rename(columns=SHP_RENAME).to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(Path(tmp).iterdir()):
                zf.write(f, f.name)
    return out.getvalue()


def report_json_bytes(report: dict) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode()


def report_csv_bytes(report: dict) -> bytes:
    lines = ["severity,label,area_ha,share_pct"]
    for row in report["by_severity"]:
        lines.append(f"{row['severity']},{row['label']},{row['area_ha']},{row['share_pct']}")
    total_share = 100.0 if report["total_burned_area_ha"] else 0.0
    lines.append(f"total,итого,{report['total_burned_area_ha']},{total_share}")
    return ("\n".join(lines) + "\n").encode("utf-8-sig")
```

- [ ] **Step 4: Append export routes to `service/app/api.py`**

Добавить в импорты файла: `from fastapi import Response` (расширить существующую строку импорта fastapi) и `from app.exports import contours_geojson_bytes, contours_shapefile_zip, report_csv_bytes, report_json_bytes`. В конец файла добавить:

```python
def _attachment(content: bytes, media_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/export/burned-areas.geojson")
def export_geojson(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    gdf = _catalog(request).query_contours(geom, d0, d1)
    return _attachment(contours_geojson_bytes(gdf), "application/geo+json", "burned_areas.geojson")


@router.post("/export/burned-areas.shp.zip")
def export_shapefile(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    gdf = _catalog(request).query_contours(geom, d0, d1)
    if gdf.empty:
        raise HTTPException(status_code=404, detail="в выборке нет контуров гарей — нечего выгружать")
    return _attachment(contours_shapefile_zip(gdf), "application/zip", "burned_areas_shp.zip")


@router.post("/export/report.json")
def export_report_json(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    cat = _catalog(request)
    rep = build_report(cat.query_contours(geom, d0, d1), cat.query_hotspots(geom, d0, d1), geom, q.date_from, q.date_to)
    return _attachment(report_json_bytes(rep), "application/json", "report.json")


@router.post("/export/report.csv")
def export_report_csv(q: SpatioTemporalQuery, request: Request) -> Response:
    geom = resolve_geometry(q)
    d0, d1 = _bounds(q)
    cat = _catalog(request)
    rep = build_report(cat.query_contours(geom, d0, d1), cat.query_hotspots(geom, d0, d1), geom, q.date_from, q.date_to)
    return _attachment(report_csv_bytes(rep), "text/csv; charset=utf-8", "report.csv")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_exports.py -v`
Expected: 6 passed

Run: `uv run pytest -q`
Expected: все зелёные.

- [ ] **Step 6: Commit**

```bash
git add service/app/exports.py service/app/api.py service/tests/test_exports.py
git commit -m "feat(service): GeoJSON/Shapefile/report exports"
```

---

### Task 10: Leaflet frontend

**Files:**
- Modify: `service/app/static/index.html` (полная версия вместо заглушки)
- Create: `service/app/static/style.css`
- Create: `service/app/static/app.js`
- Create: `service/app/static/aoi.geojson` (копия из fire-aoi)

- [ ] **Step 1: Copy AOI file**

Run (from repo root): `cp fire-aoi/fire_monitoring_aoi.geojson service/app/static/aoi.geojson`

- [ ] **Step 2: Write full `service/app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Мониторинг природных пожаров — аналитический сервис</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
<link rel="stylesheet" href="style.css">
</head>
<body>
<aside id="panel">
  <h1>🔥 Мониторинг пожаров</h1>
  <p class="hint">Нарисуйте полигон или прямоугольник на карте (инструменты слева сверху),
  задайте период и нажмите «Запросить».</p>
  <label>Дата с <input type="date" id="date-from"></label>
  <label>Дата по <input type="date" id="date-to"></label>
  <div class="row">
    <button id="btn-query" class="primary">Запросить</button>
    <button id="btn-demo">Показать пример</button>
  </div>
  <div id="error" class="error hidden"></div>
  <section id="report" class="hidden">
    <h2>Аналитическая справка</h2>
    <div id="report-body"></div>
    <h2>Выгрузки</h2>
    <div class="row wrap">
      <button data-export="burned-areas.geojson">GeoJSON</button>
      <button data-export="burned-areas.shp.zip">Shapefile</button>
      <button data-export="report.json">Справка JSON</button>
      <button data-export="report.csv">Справка CSV</button>
    </div>
  </section>
  <section>
    <h2>Легенда</h2>
    <div class="legend"><span class="swatch" style="background:#ffd23f"></span> 1 — слабая степень</div>
    <div class="legend"><span class="swatch" style="background:#ff8c1a"></span> 2 — средняя степень</div>
    <div class="legend"><span class="swatch" style="background:#d7191c"></span> 3 — сильная степень</div>
    <div class="legend"><span class="dot"></span> термоточка VIIRS</div>
  </section>
  <footer>REST API: <a href="/docs" target="_blank">/docs</a></footer>
</aside>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `service/app/static/style.css`**

```css
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
body { display: flex; }

#panel {
  width: 340px; min-width: 340px; height: 100%; overflow-y: auto;
  padding: 16px; background: #1c1f26; color: #e8eaed;
  display: flex; flex-direction: column; gap: 10px;
}
#panel h1 { font-size: 19px; margin: 0 0 4px; }
#panel h2 { font-size: 14px; margin: 12px 0 4px; color: #9aa0a6; text-transform: uppercase; letter-spacing: .04em; }
.hint { font-size: 13px; color: #9aa0a6; margin: 0; }
label { display: flex; justify-content: space-between; align-items: center; font-size: 14px; gap: 8px; }
input[type="date"] { background: #2a2e37; color: #e8eaed; border: 1px solid #3c4043; border-radius: 6px; padding: 6px 8px; }
.row { display: flex; gap: 8px; }
.row.wrap { flex-wrap: wrap; }
button {
  padding: 8px 12px; border: 1px solid #3c4043; border-radius: 6px;
  background: #2a2e37; color: #e8eaed; cursor: pointer; font-size: 13px;
}
button:hover { background: #353a45; }
button.primary { background: #d7191c; border-color: #d7191c; font-weight: 600; }
button.primary:hover { background: #b91518; }
button:disabled { opacity: .5; cursor: wait; }
.error { background: #5c1a1a; border: 1px solid #a33; border-radius: 6px; padding: 8px 10px; font-size: 13px; }
.hidden { display: none; }

#report-body table { width: 100%; border-collapse: collapse; font-size: 13px; }
#report-body td, #report-body th { padding: 4px 6px; border-bottom: 1px solid #3c4043; text-align: left; }
#report-body .total-row td { font-weight: 700; border-top: 2px solid #9aa0a6; }

.legend { display: flex; align-items: center; gap: 8px; font-size: 13px; margin: 2px 0; }
.swatch { width: 16px; height: 16px; border-radius: 3px; display: inline-block; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #ff3b30; border: 2px solid #7a0000; display: inline-block; }
footer { margin-top: auto; font-size: 12px; color: #9aa0a6; }
footer a { color: #8ab4f8; }

#map { flex: 1; height: 100%; }
```

- [ ] **Step 4: Write `service/app/static/app.js`**

```javascript
"use strict";

const SEVERITY_COLORS = { 1: "#ffd23f", 2: "#ff8c1a", 3: "#d7191c" };
const SEVERITY_LABELS = { 1: "слабая", 2: "средняя", 3: "сильная" };

const map = L.map("map").setView([48.0, 44.0], 6);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const drawnItems = new L.FeatureGroup().addTo(map);
const contoursLayer = L.layerGroup().addTo(map);
const hotspotsLayer = L.layerGroup().addTo(map);
let demo = null;

new L.Control.Draw({
  draw: {
    polygon: { allowIntersection: false, showArea: true },
    rectangle: {},
    polyline: false, circle: false, marker: false, circlemarker: false,
  },
  edit: { featureGroup: drawnItems, edit: false },
}).addTo(map);

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);
});

// Граница территории мониторинга (пунктир)
fetch("aoi.geojson")
  .then((r) => (r.ok ? r.json() : null))
  .then((geo) => {
    if (!geo) return;
    const aoi = (geo.features || []).filter((f) => f.id === "aoi");
    L.geoJSON(aoi.length ? { type: "FeatureCollection", features: aoi } : geo, {
      style: { color: "#666", weight: 1.5, dashArray: "6 4", fill: false },
    }).addTo(map);
  })
  .catch(() => {});

// Метаданные каталога: диапазон дат по умолчанию + демо-пример
fetch("/api/meta")
  .then((r) => r.json())
  .then((m) => {
    document.getElementById("date-from").value = m.date_min;
    document.getElementById("date-to").value = m.date_max;
    demo = m.demo;
  })
  .catch(() => {});

function currentGeometry() {
  const layers = drawnItems.getLayers();
  return layers.length ? layers[0].toGeoJSON().geometry : null;
}

function requestBody(geometry) {
  return {
    polygon: geometry,
    date_from: document.getElementById("date-from").value,
    date_to: document.getElementById("date-to").value,
  };
}

function showError(msg) {
  const el = document.getElementById("error");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function clearError() {
  document.getElementById("error").classList.add("hidden");
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(typeof d.detail === "string" ? d.detail : `Ошибка API (HTTP ${r.status})`);
  }
  return r.json();
}

function renderHotspots(fc) {
  hotspotsLayer.clearLayers();
  L.geoJSON(fc, {
    pointToLayer: (f, latlng) =>
      L.circleMarker(latlng, { radius: 4, color: "#7a0000", weight: 1, fillColor: "#ff3b30", fillOpacity: 0.9 }),
    onEachFeature: (f, layer) =>
      layer.bindPopup(
        `<b>Термоточка</b><br>Съёмка: ${f.properties.acq_datetime}<br>Спутник: ${f.properties.satellite}`
      ),
  }).addTo(hotspotsLayer);
}

function renderContours(fc) {
  contoursLayer.clearLayers();
  L.geoJSON(fc, {
    style: (f) => ({
      color: SEVERITY_COLORS[f.properties.severity] || "#999",
      weight: 1,
      fillColor: SEVERITY_COLORS[f.properties.severity] || "#999",
      fillOpacity: 0.45,
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(
        `<b>Гарь ${p.contour_id}</b><br>Степень: ${p.severity} (${SEVERITY_LABELS[p.severity]})` +
          `<br>Площадь: ${p.area_ha} га<br>Период: ${p.date_pre} — ${p.date_post}`
      );
    },
  }).addTo(contoursLayer);
}

function renderReport(rep) {
  const rows = rep.by_severity
    .map(
      (r) =>
        `<tr><td><span class="swatch" style="background:${SEVERITY_COLORS[r.severity]}"></span></td>` +
        `<td>${r.severity} — ${r.label}</td><td>${r.area_ha.toFixed(2)} га</td><td>${r.share_pct}%</td></tr>`
    )
    .join("");
  document.getElementById("report-body").innerHTML =
    `<table><tr><th></th><th>Степень</th><th>Площадь</th><th>Доля</th></tr>${rows}` +
    `<tr class="total-row"><td></td><td>Итого гарей</td><td>${rep.total_burned_area_ha.toFixed(2)} га</td><td></td></tr>` +
    `<tr><td></td><td>Термоточек</td><td colspan="2">${rep.hotspot_count}</td></tr>` +
    `<tr><td></td><td>Контуров</td><td colspan="2">${rep.contour_count}</td></tr></table>`;
  document.getElementById("report").classList.remove("hidden");
}

async function runQuery() {
  clearError();
  const geometry = currentGeometry();
  if (!geometry) {
    showError("Сначала нарисуйте полигон или прямоугольник на карте");
    return;
  }
  const body = requestBody(geometry);
  const btn = document.getElementById("btn-query");
  btn.disabled = true;
  try {
    const [hs, ba, rep] = await Promise.all([
      apiPost("/api/hotspots", body),
      apiPost("/api/burned-areas", body),
      apiPost("/api/report", body),
    ]);
    renderHotspots(hs);
    renderContours(ba);
    renderReport(rep);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("btn-query").addEventListener("click", runQuery);

document.getElementById("btn-demo").addEventListener("click", () => {
  if (!demo) {
    showError("Демо-пример недоступен: каталог пуст");
    return;
  }
  const [w, s, e, n] = demo.bbox;
  drawnItems.clearLayers();
  drawnItems.addLayer(L.rectangle([[s, w], [n, e]], { color: "#8ab4f8", weight: 2, fill: false }));
  document.getElementById("date-from").value = demo.date_from;
  document.getElementById("date-to").value = demo.date_to;
  map.fitBounds([[s, w], [n, e]], { padding: [30, 30] });
  runQuery();
});

document.querySelectorAll("[data-export]").forEach((btn) =>
  btn.addEventListener("click", async () => {
    clearError();
    const geometry = currentGeometry();
    if (!geometry) {
      showError("Нарисуйте область для выгрузки");
      return;
    }
    const name = btn.dataset.export;
    const r = await fetch(`/api/export/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(geometry)),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      showError(typeof d.detail === "string" ? d.detail : `Ошибка выгрузки (HTTP ${r.status})`);
      return;
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name.replace(".shp.zip", "_shp.zip");
    a.click();
    URL.revokeObjectURL(a.href);
  })
);
```

- [ ] **Step 5: Verify tests still pass and page is served**

Run: `uv run pytest -q`
Expected: все зелёные (тест `test_index_served` проверяет `id="map"`).

- [ ] **Step 6: Commit**

```bash
git add service/app/static/
git commit -m "feat(service): Leaflet SPA with draw tools, layers, report and exports"
```

---

### Task 11: Real-data ingest, smoke test, README

**Files:**
- Create: `service/README.md`
- Create: `service/data/catalog.gpkg` (генерируется, НЕ коммитится — data/ в .gitignore)

- [ ] **Step 1: Run ingest on real train data**

Run (from `service/`): `uv run python scripts/ingest.py --data-dir ../train --output data/catalog.gpkg`
Expected: завершение без ошибок, лог `catalog written: ... hotspots, ... contours` (примерно: десятки тысяч термоточек из 420 AF-чипов, тысячи контуров из 224 BS-чипов). Записать реальные числа — они пойдут в README и отчёт.

- [ ] **Step 2: Start the server and smoke-test with curl**

Run: `uv run uvicorn app.main:app --port 8000` (в фоне)

```bash
curl -s localhost:8000/healthz
# ожидается: {"status":"ok"}

curl -s localhost:8000/api/meta | python3 -m json.tool
# ожидается: hotspot_count/contour_count из шага 1, date_min ~2019-04, date_max ~2024-11, demo с fire_event FE12837

# запрос по демо-области (bbox взять из /api/meta -> demo.bbox, даты из demo)
curl -s -X POST localhost:8000/api/report -H 'Content-Type: application/json' \
  -d '{"bbox": [<demo bbox>], "date_from": "<demo date_from>", "date_to": "<demo date_to>"}' | python3 -m json.tool
# ожидается: total_burned_area_ha в районе 5800-5900 га (крупнейший пожар FE12837 = 5877.88 га по meta.csv)
```

Сверка: `total_burned_area_ha` по полному покрытию события должно совпадать с `burn_area_ha` из `train/bs/meta.csv` для чипов события с точностью до округления — это и есть «воспроизводимость чисел справки».

- [ ] **Step 3: Open UI in browser, click «Показать пример»**

Проверить: карта рисует прямоугольник, приходят контуры с раскраской по степеням, термоточки, справка заполняется, все 4 выгрузки скачиваются. GeoJSON открывается в QGIS (или проверить `python3 -c "import json; json.load(open(...))"`).

- [ ] **Step 4: Write `service/README.md`**

```markdown
# Информационно-аналитический сервис мониторинга природных пожаров

Веб-приложение + REST API: по полигону (или bbox) и интервалу дат возвращает
карту термоточек VIIRS, векторные контуры гарей со степенями поражения
и аналитическую справку с площадями.

## Требования

- [uv](https://docs.astral.sh/uv/) (Python 3.12 подтянется автоматически)
- Выданные данные соревнования в `../train` (папки `af/`, `bs/` с `meta.csv` и `masks/`)

## Запуск

```bash
cd service
uv sync                                                  # окружение
uv run python scripts/ingest.py --data-dir ../train      # каталог: data/catalog.gpkg (~1 мин)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000   # сервис
```

UI: http://localhost:8000 · Swagger: http://localhost:8000/docs

## Источник данных сервиса

Постановка разрешает работу «на заранее подготовленном ограниченном наборе сцен».
Сервис работает поверх georeferenced train-чипов (geопривязка и даты из meta.csv).
По умолчанию каталог собирается из эталонных масок; чтобы подключить предсказания
Модулей 1–2, выполните ingest из submission.csv по тем же чипам:

```bash
uv run python scripts/ingest.py --data-dir ../train --source predictions --submission preds.csv
```

## REST API

Все запросы — POST JSON: `polygon` (GeoJSON geometry, EPSG:4326) **или**
`bbox` `[minx, miny, maxx, maxy]`, плюс `date_from`, `date_to` (ISO-даты).

| Метод и путь | Ответ |
|---|---|
| `POST /api/hotspots` | GeoJSON FeatureCollection термоточек |
| `POST /api/burned-areas` | GeoJSON FeatureCollection контуров гарей |
| `POST /api/report` | JSON-справка: площади по степеням, термоточки |
| `POST /api/export/burned-areas.geojson` | выгрузка контуров GeoJSON |
| `POST /api/export/burned-areas.shp.zip` | выгрузка контуров Shapefile (zip) |
| `POST /api/export/report.json` / `report.csv` | машиночитаемая справка |
| `GET /api/meta` | границы данных + демо-пример |
| `GET /healthz` | статус |

Пример:

```bash
curl -s -X POST localhost:8000/api/report -H 'Content-Type: application/json' -d '{
  "bbox": [46.0, 49.0, 48.5, 50.5],
  "date_from": "2022-06-01",
  "date_to": "2022-08-01"
}'
```

## Семантика расчётов (важно для воспроизводимости)

- **Площади считаются в родной UTM-зоне чипа** (EPSG:32637/32638), пиксель BS = 20×20 м = 0.04 га.
- **Справка** учитывает площадь *пересечения* контуров с полигоном запроса;
  полная площадь каждого контура — атрибут `area_ha` в выгрузках.
- **Фильтр по датам:** термоточка — по дате пролёта `acq_datetime`;
  контур гари попадает в выборку, если его `date_post` лежит в интервале.
- Выгрузки — в EPSG:4326 (WGS84), у Shapefile колонка `fire_event_id`
  переименована в `fire_event` (ограничение формата в 10 символов).

## Тесты

```bash
uv run pytest
```
```

- [ ] **Step 5: Full verification**

Run: `uv run pytest -q`
Expected: все зелёные.

- [ ] **Step 6: Commit**

```bash
git add service/README.md
git commit -m "docs(service): README with launch instructions and API reference"
```

---

### Task 12 (polish, optional — только после Task 11): Dockerfile

**Files:**
- Create: `service/Dockerfile`
- Modify: `service/README.md` (раздел Docker)

- [ ] **Step 1: Create `service/Dockerfile`**

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /srv/service
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY scripts ./scripts
# каталог собирается заранее: uv run python scripts/ingest.py --data-dir ../train
COPY data ./data

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Build and smoke-test**

Run (from `service/`): `docker build -t fire-service . && docker run --rm -p 8000:8000 fire-service`
Expected: `curl localhost:8000/healthz` → `{"status":"ok"}`

- [ ] **Step 3: Append to `service/README.md`**

```markdown
## Docker

```bash
cd service
uv run python scripts/ingest.py --data-dir ../train   # каталог до сборки образа
docker build -t fire-service .
docker run --rm -p 8000:8000 fire-service
```
```

- [ ] **Step 4: Commit**

```bash
git add service/Dockerfile service/README.md
git commit -m "chore(service): Dockerfile for one-command deployment"
```

---

## Self-review notes

- **Spec coverage:** приём полигон/bbox+даты (Task 8), карта термоточек (Tasks 5, 8, 10), контуры GeoJSON/Shapefile с CRS и атрибутами id/степень/площадь (Tasks 4, 9), справка с площадью и распределением + машиночитаемая выгрузка (Tasks 7, 9), REST API без UI (Task 8), площади в UTM (Tasks 3, 7), demo-кнопка (Tasks 7, 10), README-инструкция запуска (Task 11), predictions-режим (Tasks 5, 6). Пропусков нет.
- **Консистентность имён:** `build_catalog(data_dir, output, source, submission)`, `Catalog.load/query_hotspots/query_contours/meta`, `clipped_area_by_severity`, `build_report(contours, hotspots, query_geom, date_from, date_to)`, `SpatioTemporalQuery`, слои `hotspots`/`burn_contours` — использованы одинаково во всех задачах.
- **Известные допущения:** пустая выборка Shapefile → 404 (документировано); фильтр контуров по `date_post`; предсказания ingest-ятся только для чипов, присутствующих в meta.csv с геопривязкой.

# Информационно-аналитический сервис мониторинга пожаров — Design Spec

Date: 2026-09-18
Status: approved (user), КосмоХакатон 2026, Красноярск
Scope owner: наша часть командного решения — Раздел 4 критериев (до 12 баллов)

## 1. Goal & success criteria

Web-приложение + обязательный REST API: по полигону (или bbox) и интервалу дат возвращает
карту термоточек, векторные контуры гарей и аналитическую справку.

Mapping to оценочные критерии (Раздел 4):

| Критерий | Баллы | Что закрываем |
|---|---|---|
| Работоспособность сервиса | 0 или 4 | Запуск одной командой из README, запрос полигон+даты, ответ за конечное время, REST API обязателен |
| Картографический вывод и векторные контуры | 0–4 | Термоточки + контуры поверх подложки; выгрузка GeoJSON/Shapefile, корректная CRS, атрибуты: id, класс степени, площадь |
| Аналитическая справка | 0–4 | Суммарная площадь гари (га) + распределение по 3 степеням, машиночитаемая выгрузка, площадь считается в UTM |

Постановка явно разрешает работу «на заранее подготовленном ограниченном наборе сцен» —
демонстрируем на georeferenced train-чипах.

## 2. Decisions (approved)

1. **Источник масок:** ingest-слой с единым форматом. Сейчас — ground-truth маски train-чипов
   (420 AF + 224 BS, у всех есть epsg/bounds/даты в meta.csv). Когда тиммейты дадут
   предсказания Модулей 1–2 — повторный ingest из submission.csv (RLE), сервис не меняется.
2. **Стек:** FastAPI (REST API + раздача статики) + Leaflet SPA без сборки фронта.
3. **Объём:** минимум по критериям + полировка; доп-фичи только после закрытия базы.
4. **Семантика площади:** справка считает площадь **пересечения** контуров с полигоном запроса
   (пересчёт в UTM-зоне чипа); каждая фича в выгрузке несёт полную `area_ha` контура.
   Обе величины документируются в README (критерий воспроизводимости чисел).

## 3. Architecture

```
train/{af,bs}/... + meta.csv          (GT-маски, геопривязка EPSG:32637/32638)
        │
        ▼  scripts/ingest.py  (оффлайн, однократно; режимы gt | predictions)
data/catalog.gpkg                     (GeoPackage, EPSG:4326: слои hotspots, burn_contours)
        │
        ▼  загрузка в память при старте (GeoDataFrame + STRtree)
FastAPI ── REST API /api/* ── Leaflet SPA (app/static)
```

Без PostGIS/БД: объём данных мал (сотни чипов, ~10k точек), in-memory запросы — миллисекунды.

## 4. Ingest — `scripts/ingest.py`

- **Hotspots (AF-чипы):** пиксели класса 1 из маски → центры пикселей в UTM (epsg+bounds из
  meta.csv) → EPSG:4326. Атрибуты: `chip_id`, `acq_datetime`, `satellite`.
- **Burn contours (BS-чипы):** полигонизация маски 0–3 по классам (`rasterio.features.shapes`),
  `area_ha` считается в родном UTM (пиксель 20 м = 0.04 га), затем репроекция в 4326.
  Атрибуты: `contour_id`, `chip_id`, `fire_event_id`, `severity` (1/2/3), `area_ha`,
  `date_pre`, `date_post`.
- **Режимы:** `--source gt` (default) и `--source predictions --submission path.csv`
  (декодирование RLE по тем же правилам, что в постановке: нумерация построчно с 1).
- Выход: `data/catalog.gpkg`, слои `hotspots`, `burn_contours`.

## 5. REST API (FastAPI, OpenAPI на /docs)

Общие параметры запроса (POST JSON): `polygon` (GeoJSON geometry) **или** `bbox`
[minx, miny, maxx, maxy] в EPSG:4326; `date_from`, `date_to` (ISO date).

| Endpoint | Ответ |
|---|---|
| `POST /api/hotspots` | GeoJSON FeatureCollection точек |
| `POST /api/burned-areas` | GeoJSON FeatureCollection полигонов по степеням |
| `POST /api/report` | JSON: `total_area_ha`, `by_severity {1,2,3}` (га и %), `hotspot_count`, период, эхо запроса |
| `POST /api/export/burned-areas.geojson` / `.shp.zip` | файл с корректной CRS и атрибутами |
| `POST /api/export/report.json` / `.csv` | машиночитаемая справка |
| `GET /healthz` | ok |

Ошибки: невалидная геометрия → 422 с сообщением; пустой результат → валидные пустые
коллекции и нули в справке (не ошибка); даты вне данных → пустой результат.
Для термоточек фильтр по `acq_datetime`; для гарей контур попадает в выборку, если
`date_post` внутри интервала (документируется).

## 6. Frontend (одна страница, Leaflet + leaflet-draw, русский UI)

- Карта: OSM-подложка, граница AOI из `fire-aoi/`, инструмент рисования полигона/bbox,
  выбор диапазона дат (default — период с данными), кнопка «Запросить».
- Слои: термоточки (точки), контуры гарей с раскраской по степени
  (1 — жёлтый, 2 — оранжевый, 3 — красный), легенда.
- Сайдбар: аналитическая справка + кнопки выгрузки (GeoJSON / Shapefile / JSON / CSV).
- Кнопка «Показать пример»: zoom на крупный fire_event с готовым полигоном и датами —
  защита не зависит от ручного поиска.

## 7. Project layout & run

```
service/
  pyproject.toml          # uv; fastapi, uvicorn, geopandas, rasterio, shapely, pyproj, pandas
  README.md               # ingest + запуск + примеры curl
  app/
    main.py               # создание приложения, mount static
    api.py                # маршруты
    store.py              # загрузка catalog.gpkg, пространственные запросы
    reports.py            # расчёт справки (площади в UTM)
    exports.py            # geojson / shapefile.zip / csv
    schemas.py            # pydantic-модели запросов/ответов
    static/               # index.html, app.js, style.css
  scripts/ingest.py
  tests/
    test_ingest.py        # known-answer: площадь = n_px * 0.04 га; RLE round-trip
    test_api.py           # контракт API, пустые/невалидные запросы
```

Запуск: `uv run python scripts/ingest.py` → `uv run uvicorn app.main:app --port 8000`.
Полировка (после базы): Dockerfile, кластеризация точек, графики динамики.

## 8. Testing & verification

- pytest: ingest на фикстуре из 2–3 чипов, сверка площадей с `burn_area_ha`/`sev*_px`
  из meta.csv; RLE encode/decode round-trip; API-контракт (валидные/пустые/битые запросы).
- Ручная проверка UI через Playwright/браузер: запрос по примеру, выгрузки открываются в QGIS.
- Числа справки воспроизводимы из выданных масок (критерий).

## 9. Out of scope

Обучение моделей, inference.py соревнования, submission.csv генерация — делают тиммейты.
Сервис лишь потребляет их маски через ingest.

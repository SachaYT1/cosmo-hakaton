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
uv run python scripts/ingest.py --data-dir ../train      # каталог: data/catalog.gpkg (~10 сек)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000   # сервис
```

UI: http://localhost:8000 · Swagger: http://localhost:8000/docs

На выданном train-наборе каталог содержит **9 725 термоточек** (совпадает
с числом пикселей горения из постановки) и **99 239 контуров гарей**.
В UI кнопка «Показать пример» строит запрос по крупнейшему пожару
(FE12837, 5 877.88 га, июнь–июль 2022).

## Источник данных сервиса

Постановка разрешает работу «на заранее подготовленном ограниченном наборе сцен».
Сервис работает поверх georeferenced train-чипов (геопривязка и даты из meta.csv).
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
  "bbox": [47.06, 48.04, 47.26, 48.15],
  "date_from": "2022-06-06",
  "date_to": "2022-07-22"
}'
```

Ответ для этого примера воспроизводит эталон: `total_burned_area_ha = 5877.88`
(равно `burn_area_ha` чипа BS_tr_000095 в `train/bs/meta.csv`), по степеням —
`sev*_px × 0.04 га`: 672.88 / 3220.72 / 1984.28.

## Семантика расчётов (важно для воспроизводимости)

- **Площади считаются в родной UTM-зоне чипа** (EPSG:32637/32638), пиксель BS = 20×20 м = 0.04 га.
- **Справка** учитывает площадь *пересечения* контуров с полигоном запроса;
  полная площадь каждого контура — атрибут `area_ha` в выгрузках.
- **Фильтр по датам:** термоточка — по дате пролёта `acq_datetime`;
  контур гари попадает в выборку, если его `date_post` лежит в интервале.
- Выгрузки — в EPSG:4326 (WGS84), у Shapefile колонка `fire_event_id`
  переименована в `fire_event` (ограничение формата в 10 символов).

## Docker

```bash
cd service
uv run python scripts/ingest.py --data-dir ../train   # каталог собирается до сборки образа
docker build -t fire-service .
docker run --rm -p 8000:8000 fire-service
```

## Тесты

```bash
uv run pytest
```

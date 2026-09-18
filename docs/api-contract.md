# Контракт REST API сервиса

Живая версия: `http://localhost:8000/docs` (Swagger). База: `app/api.py`, `app/schemas.py`.

## Общий формат запроса (все POST-ручки)

```json
{
  "polygon": {"type": "Polygon", "coordinates": [[[47.06, 48.04], [47.26, 48.04], [47.26, 48.15], [47.06, 48.15], [47.06, 48.04]]]},
  "bbox": null,
  "date_from": "2022-06-06",
  "date_to": "2022-07-22"
}
```

- `polygon` — GeoJSON geometry (Polygon/MultiPolygon), EPSG:4326, **или** `bbox` `[minx, miny, maxx, maxy]` в lon/lat — ровно одно из двух (оба/ни одного → 422).
- `date_from ≤ date_to`, ISO-даты, границы включительно.

## Ручки

| Ручка | Ответ |
|---|---|
| `POST /api/hotspots` | GeoJSON FC точек; properties: `chip_id`, `acq_datetime` (пролёт), `satellite`. Фильтр: `acq_datetime` в интервале |
| `POST /api/burned-areas` | GeoJSON FC полигонов; properties: `contour_id`, `chip_id`, `fire_event_id`, `severity` (1/2/3), `area_ha` (полная площадь контура, UTM), `date_pre`, `date_post`, `epsg`. Фильтр: `date_post` в интервале |
| `POST /api/report` | JSON: `period`, `total_burned_area_ha`, `by_severity[{severity,label,area_ha,share_pct}]`, `hotspot_count`, `contour_count`, `note`. Площади = пересечение контуров с полигоном запроса (счёт в UTM) |
| `POST /api/export/burned-areas.geojson` | файл GeoJSON (attachment) |
| `POST /api/export/burned-areas.shp.zip` | zip shp+shx+dbf+prj+cpg, WGS84; `fire_event_id`→`fire_event`; пустая выборка → 404 |
| `POST /api/export/report.json` / `report.csv` | справка файлом; CSV: `severity,label,area_ha,share_pct` + строка `total` |
| `GET /api/meta` | `hotspot_count`, `contour_count`, `date_min`, `date_max`, `demo{fire_event_id,bbox,date_from,date_to}` |
| `GET /healthz` | `{"status": "ok"}` |

## Пример ответа /api/report (демо-область FE12837)

```json
{
  "period": {"date_from": "2022-06-06", "date_to": "2022-07-22"},
  "total_burned_area_ha": 5877.88,
  "by_severity": [
    {"severity": 1, "label": "слабая",  "area_ha": 672.88,  "share_pct": 11.4},
    {"severity": 2, "label": "средняя", "area_ha": 3220.72, "share_pct": 54.8},
    {"severity": 3, "label": "сильная", "area_ha": 1984.28, "share_pct": 33.8}
  ],
  "hotspot_count": 0,
  "contour_count": 991
}
```

## Ошибки

- **422** — невалидный запрос; `detail` — строка (проверка геометрии) или список (pydantic).
- **404** — только shp.zip при пустой выборке.
- Пустой результат — не ошибка: 200 с `"features": []` / нулями.

## Пример curl

```bash
curl -s -X POST localhost:8000/api/report -H 'Content-Type: application/json' \
  -d '{"bbox": [47.06, 48.04, 47.26, 48.15], "date_from": "2022-06-06", "date_to": "2022-07-22"}'
```

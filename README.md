# Двухэтапный мониторинг природных пожаров — КосмоХакатон 2026

Решение кейса «двухэтапный мониторинг природных пожаров по данным VIIRS, Sentinel-2
и Sentinel-1»: два модуля детекции + информационно-аналитический сервис.

**Итоговая метрика (out-of-fold на train, через формат submission):**

| F1_af | IoU_burn | mIoU_sev (1 / 2 / 3) | **Score** |
|---|---|---|---|
| 0.953 | 0.434 | 0.462 (0.228 / 0.479 / 0.679) | **0.624** |

## Структура репозитория

```
models/     Модули 1–2: детекция активного горения (LightGBM) и картирование гарей
            (пороговая модель dNBR по типам покрова). inference.py, обучение, веса,
            EDA-отчёты. Подробности: models/PROGRESS.md
service/    Информационно-аналитический сервис: REST API + веб-карта (FastAPI + Leaflet).
            Подробности: service/README.md
docs/       Контракт API, спека и план сервиса
fire-aoi/   Граница территории мониторинга (Нижнее Поволжье и Подонье)
REPORT.md   Отчёт по решению
```

Данные соревнования кладутся рядом: `train/` и `test/` в корне репозитория
(в git не входят).

## Быстрый старт

### Инференс (submission.csv)

```bash
cd models
pip install -r requirements-af.txt     # macOS дополнительно: brew install libomp
python inference.py --data-dir ../test --output submission.csv --bs-model rules
python -m scripts.validate_submission submission.csv    # -> VALID
```

Тестовый набор (180 AF + 89 BS чипов) обрабатывается за ~9 секунд на CPU.
На macOS при проблемах с пулом процессов добавьте `--workers 1`.

### Информационно-аналитический сервис

```bash
cd service
uv sync
uv run python scripts/ingest.py --data-dir ../train      # геокаталог из эталонных масок
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

UI: http://localhost:8000 (кнопка «Показать пример» — крупнейший пожар набора),
Swagger: http://localhost:8000/docs. Сервис поверх предсказаний моделей:

```bash
uv run python scripts/ingest.py --data-dir ../train \
    --source predictions --submission preds_train.csv --output data/catalog_pred.gpkg
CATALOG_PATH=data/catalog_pred.gpkg uv run uvicorn app.main:app --port 8001
```

`service/preds_train.csv` (предсказания моделей по train-чипам) лежит в репозитории.

### Обучение

```bash
cd models
python -m scripts.train_af --data-dir /path/to/train --external-dir data/external/noaa_af_chips
python -m scripts.fit_bs_rules         # BS: пороги dNBR по покрову -> configs/bs_thresholds.json
python -m unittest discover -s tests -v
```

Случайные начальные значения зафиксированы (seed 42), разбиения фолдов
детерминированы (AF — по дате съёмки, BS — по чипу/пожару).

## Лицензии и данные

- Copernicus Sentinel-1/2, Copernicus DEM — открытая лицензия Copernicus.
- VIIRS (NASA/NOAA) — общественное достояние; ESA WorldCover — CC BY 4.0.
- Для слабого дообучения AF использованы исторические NOAA VIIRS SDR/AF EDR за зимние
  даты вне тестовой территории; они не входят в валидацию. Источники и лицензии:
  `models/configs/af_external_data.json`.
- Библиотеки: FastAPI, geopandas, rasterio, shapely, LightGBM, XGBoost, OpenCV, Leaflet.

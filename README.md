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
(в git не входят). Ожидаемая структура:

```text
train/
  af/{viirs,aux,masks}/ + af/meta.csv
  bs/{sentinel2_pre,sentinel2_post,sentinel1_pre,sentinel1_post,aux,masks}/ + bs/meta.csv
test/
  sample_submission.csv + meta.csv
  af/{viirs,aux}/
  bs/{sentinel2_pre,sentinel2_post,sentinel1_pre,sentinel1_post,aux}/
```

## Быстрый старт

### Инференс (submission.csv)

Проверенное окружение — Python 3.12. Финальный AF-вес
`models/weights/af_augmented_best_lgb.txt` и BS-пороги уже находятся в репозитории;
дополнительное скачивание весов не требуется.

```bash
cd models
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements-af.txt
# macOS при отсутствии OpenMP: brew install libomp
python inference.py --data-dir ../test --output submission.csv
python -m scripts.validate_submission submission.csv --data-dir ../test    # -> VALID
```

Ожидаемый результат — `submission.csv` с 447 строками и сообщение `VALID`.
Тестовый набор (180 AF + 89 BS чипов) обработан за 11,1 секунды на Apple Silicon CPU.
Два последовательных запуска дали побайтово одинаковые файлы с SHA-256
`d7c7008c0330956b659986f247881abd58bbb76a3690b9aacb811e14fec1567f`.
На macOS при проблемах с пулом процессов добавьте `--workers 1`.

### Информационно-аналитический сервис

```bash
cd service
uv sync
uv run python scripts/ingest.py --data-dir ../train      # каталог из preds_train.csv
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

UI: http://localhost:8000 (кнопка «Показать пример» — крупнейший пожар набора),
Swagger: http://localhost:8000/docs. По умолчанию сервис работает поверх предсказаний
`service/preds_train.csv`. Для контрольной проверки на эталонных масках:

```bash
uv run python scripts/ingest.py --data-dir ../train \
    --source gt --output data/catalog_gt.gpkg
CATALOG_PATH=data/catalog_gt.gpkg uv run uvicorn app.main:app --port 8001
```

`service/preds_train.csv` (предсказания моделей по train-чипам) лежит в репозитории.

### Обучение

Финальная AF-конфигурация использует три фолда по дате, seed 42 и внешние weak labels
с весом 0.02. Подготовка внешних данных требует сетевого доступа к открытым NOAA-источникам:

```bash
cd models
source .venv/bin/activate
python -m scripts.prepare_af_external --download
python -m scripts.train_af --data-dir ../train --folds 3 --threads 6 \
    --external-dir data/external/noaa_af_chips --external-weight 0.02
python -m scripts.fit_bs_rules --data-dir ../train --folds 5
python -m unittest discover -s tests -v
```

`train_af` сохраняет повторный эксперимент отдельно (`configs/af_augmented.json`,
`weights/af_augmented_lgb.txt`, `reports/af_augmented_training.json`) и не перезаписывает
проверенную финальную модель. Для сдаваемого инференса уже подключены
`configs/af.json` и `weights/af_augmented_best_lgb.txt`. Повторное обучение и инференс
детерминированы при одинаковом окружении и входных данных.

Случайные начальные значения зафиксированы (seed 42), разбиения фолдов
детерминированы (AF — по дате съёмки, BS — по чипу/пожару).

## Лицензии и данные

- Copernicus Sentinel-1/2, Copernicus DEM — открытая лицензия Copernicus.
- VIIRS (NASA/NOAA) — общественное достояние; ESA WorldCover — CC BY 4.0.
- Для слабого дообучения AF использованы исторические NOAA VIIRS SDR/AF EDR за зимние
  даты вне тестовой территории; они не входят в валидацию. Источники и лицензии:
  `models/configs/af_external_data.json`.
- Библиотеки: FastAPI, geopandas, rasterio, shapely, LightGBM, XGBoost, OpenCV, Leaflet.

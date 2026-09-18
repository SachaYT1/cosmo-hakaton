# Мониторинг природных пожаров — КосмоХакатон 2026

Двухэтапный мониторинг: детекция активного горения (AF, VIIRS) +
картирование гарей и степени поражения (BS, Sentinel-2/1).

## Структура репозитория

```
.
├── inference.py            # единая точка входа инференса (см. ниже)
├── train.py                # точка входа обучения (AF + BS)
├── configs/                # конфиги моделей и пайплайна (yaml)
├── src/fireapp/
│   ├── common/              # RLE-кодек, чтение чипов, общие утилиты
│   ├── af/                  # датасет/модель/train/infer для активного горения
│   ├── bs/                  # датасет/модель/train/infer для гарей и степени
│   ├── pipeline/            # сборка submission.csv, наполнение БД для сервиса
│   └── service/             # FastAPI-сервис + REST API + статика карты
├── metric/                  # официальный скрипт подсчёта Score
├── scripts/                 # вспомогательные скрипты (EDA, загрузка данных)
├── notebooks/               # ноутбуки EDA
├── report/                  # отчёт команды
├── weights/                 # веса моделей (не в git, см. .gitignore)
└── tests/                   # unit-тесты (RLE, метрика, форматы)
```

## Окружение

```bash
pip install -r requirements.txt
# или
docker compose build
```

## Ожидаемая структура входных данных

```
<data-dir>/
├── af/
│   ├── viirs/{chip_id}_VIIRS_I1-I5.tif
│   ├── aux/{chip_id}_AUX.tif
│   ├── masks/{chip_id}_MASK.tif        # только train
│   └── meta.csv
├── bs/
│   ├── sentinel2_pre/{chip_id}_Sentinel-2_pre.tif
│   ├── sentinel2_post/{chip_id}_Sentinel-2_post.tif
│   ├── sentinel1_pre/{chip_id}_Sentinel-1_pre.tif
│   ├── sentinel1_post/{chip_id}_Sentinel-1_post.tif
│   ├── aux/{chip_id}_AUX.tif
│   ├── masks/{chip_id}_MASK.tif        # только train
│   └── meta.csv
└── sample_submission.csv
```

## Запуск инференса

```bash
python inference.py --data-dir /path/to/test --output /path/to/submission.csv
```

`submission.csv` появится по указанному пути `--output`.

## Запуск обучения

```bash
python train.py --module af --config configs/af.yaml --data-dir /path/to/train/af
python train.py --module bs --config configs/bs.yaml --data-dir /path/to/train/bs
```

## Веса

TODO: ссылка на веса и куда их положить (по умолчанию — `weights/af.pt`, `weights/bs.pt`).

## Информационно-аналитический сервис

```bash
docker compose up service db
```

REST API поднимается на `http://localhost:8000` (`/docs` — OpenAPI).
Веб-карта — `http://localhost:8000/`.

Перед запуском сервиса нужно наполнить БД предвычисленными результатами:

```bash
python -m src.fireapp.pipeline.populate_db --data-dir /path/to/scenes
```

## Метрика

```bash
# gt.csv для локальной проверки собирается из train-масок:
python scripts/build_gt_csv.py --af-dir train/af --bs-dir train/bs --output gt.csv
python metric/score.py --pred submission.csv --gt gt.csv --meta train/af/meta.csv train/bs/meta.csv
```

`Score = 0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev`

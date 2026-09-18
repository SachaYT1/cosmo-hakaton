"""Прогоняет инференс AF/BS по заданному набору сцен и пишет результат в PostGIS
для информационно-аналитического сервиса (термоточки, контуры гарей, площади).

Запуск: python -m fireapp.pipeline.populate_db --data-dir /path/to/scenes
"""
from __future__ import annotations

from pathlib import Path


def populate(data_dir: str | Path) -> None:
    raise NotImplementedError(
        "TODO: run_af_inference + run_bs_inference -> векторизация масок (rasterio.features.shapes) "
        "-> запись в таблицы hotspots(point, ts) и burn_polygons(geom, severity, area_ha, chip_id)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    populate(args.data_dir)

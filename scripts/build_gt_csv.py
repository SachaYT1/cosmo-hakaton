"""Собирает gt.csv (тот же формат, что submission.csv) из локальных train-масок,
чтобы можно было прогнать metric/score.py на своём val-сплите до сдачи решения.

Запуск:
    python scripts/build_gt_csv.py --af-dir /path/to/train/af --bs-dir /path/to/train/bs --output gt.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fireapp.common.io_raster import read_chip
from fireapp.common.rle import mask_to_rle


def main(af_dir: str | None, bs_dir: str | None, output: str) -> None:
    rows = []

    if af_dir:
        af_dir = Path(af_dir)
        meta = pd.read_csv(af_dir / "meta.csv")
        for chip_id in meta["chip_id"]:
            mask = read_chip(af_dir / "masks" / f"{chip_id}_MASK.tif")[0]
            rows.append({"chip_id": chip_id, "class_id": 1, "rle": mask_to_rle(mask == 1)})

    if bs_dir:
        bs_dir = Path(bs_dir)
        meta = pd.read_csv(bs_dir / "meta.csv")
        for chip_id in meta["chip_id"]:
            mask = read_chip(bs_dir / "masks" / f"{chip_id}_MASK.tif")[0]
            for class_id in (1, 2, 3):
                rows.append({"chip_id": chip_id, "class_id": class_id, "rle": mask_to_rle(mask == class_id)})

    pd.DataFrame(rows).to_csv(output, index=False, quoting=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--af-dir", default=None, help="каталог train/af (опционально)")
    parser.add_argument("--bs-dir", default=None, help="каталог train/bs (опционально)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.af_dir and not args.bs_dir:
        raise SystemExit("укажи хотя бы один из --af-dir / --bs-dir")
    main(args.af_dir, args.bs_dir, args.output)

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

"""Единая точка входа обучения обоих модулей.

Запуск:
    python train.py --module af --config configs/af.yaml --data-dir /path/to/train/af
    python train.py --module bs --config configs/bs.yaml --data-dir /path/to/train/bs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main(module: str, config: str, data_dir: str) -> None:
    if module == "af":
        from fireapp.af.train import main as train_af

        train_af(config, data_dir)
    elif module == "bs":
        from fireapp.bs.train import main as train_bs

        train_bs(config, data_dir)
    else:
        raise ValueError(f"неизвестный модуль: {module}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["af", "bs"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True, help="каталог train/af или train/bs")
    args = parser.parse_args()
    main(args.module, args.config, args.data_dir)

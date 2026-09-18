"""Скрипт быстрого EDA по meta.csv (без открытия растров).

Запуск: python scripts/eda.py --data-dir /path/to/train
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main(data_dir: str) -> None:
    af_meta = pd.read_csv(Path(data_dir) / "af" / "meta.csv")
    bs_meta = pd.read_csv(Path(data_dir) / "bs" / "meta.csv")

    print("AF: чипов всего", len(af_meta))
    print("AF: с горением (n_fire_px > 0)", (af_meta["n_fire_px"] > 0).sum())
    print("AF: доля пикселей горения", af_meta["n_fire_px"].sum() / (af_meta["width"] * af_meta["height"]).sum())

    print("\nBS: чипов всего", len(bs_meta))
    print("BS: площадь гари по степеням (px)", bs_meta[["sev1_px", "sev2_px", "sev3_px"]].sum())
    print("BS: медианная облачность", bs_meta["cloud_frac"].median())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    main(args.data_dir)

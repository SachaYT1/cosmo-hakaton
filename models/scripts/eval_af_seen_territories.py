"""Evaluate 2024 AF chips whose exact territory appeared before 2024."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from firemon.af_eval import counts, metrics
from scripts.train_af import ROOT, build_external, predict_fit


def main(threads: int = 4, threshold: float = 0.44, external_weight: float = 0.02) -> None:
    cache = ROOT / "cache/af_experiments"
    X = np.load(cache / "X.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npz")
    y, chip, missed = labels["y"], labels["chip"], labels["missed"]
    meta = pd.read_csv(cache / "meta.csv")
    years = pd.to_datetime(meta.acq_datetime, utc=True).dt.year.to_numpy()
    territory_columns = ["epsg", "x_min", "y_min", "x_max", "y_max"]
    territory = meta[territory_columns].astype(str).agg("|".join, axis=1).to_numpy()
    development_chips = years < 2024
    holdout_chips = years == 2024
    seen_keys = set(territory[development_chips])
    seen_chips = holdout_chips & np.array([key in seen_keys for key in territory])
    unseen_chips = holdout_chips & ~seen_chips
    external = build_external(str(ROOT / "data/external/noaa_af_chips"), "context")
    members = json.loads((ROOT / "configs/af_train.json").read_text())["models"]
    probability = predict_fit(X, y, development_chips[chip], holdout_chips[chip],
                              members, threads, external, external_weight)
    holdout_chip_index = chip[holdout_chips[chip]]
    result = {"protocol": {
        "train_years": "2019-2023", "holdout_year": 2024,
        "seen_territory": "exact equality of EPSG,x_min,y_min,x_max,y_max",
        "threshold": threshold, "external_weight": external_weight,
    }}
    for name, selected_chips in (("all_2024", holdout_chips),
                                 ("seen_territories", seen_chips),
                                 ("unseen_territories", unseen_chips)):
        selected = selected_chips[holdout_chip_index]
        result[name] = {
            "chips": int(selected_chips.sum()),
            **metrics(counts(probability[selected], y[holdout_chips[chip]][selected],
                             threshold, int(missed[selected_chips].sum()))),
        }
    destination = ROOT / "reports/af_seen_territories_2024.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

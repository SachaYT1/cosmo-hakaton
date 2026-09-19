"""Compare external weak-label weights on the frozen 2019-2023 AF protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from firemon.af_eval import counts, metrics, tune_threshold
from scripts.train_af import ROOT, build_external, cross_validate


def main(weights: list[float], threads: int) -> None:
    cache = ROOT / "cache/af_experiments"
    X = np.load(cache / "X.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npz")
    y, chip, missed = labels["y"], labels["chip"], labels["missed"]
    meta = pd.read_csv(cache / "meta.csv")
    dates = meta.acq_datetime.str[:10].to_numpy()
    years = pd.to_datetime(meta.acq_datetime, utc=True).dt.year.to_numpy()
    selected = years < 2024
    spec = json.loads((ROOT / "configs/af_train.json").read_text())
    external = build_external(str(ROOT / "data/external/noaa_af_chips"), "context")
    results = []
    for weight in weights:
        prob, fold, _ = cross_validate(X, y, chip, dates, selected, spec["models"],
                                       3, threads, external, weight)
        use = fold >= 0
        threshold, score = tune_threshold(prob[use], y[use], int(missed[selected].sum()))
        fixed = metrics(counts(prob[use], y[use], 0.5, int(missed[selected].sum())))
        result = {"external_weight": weight, "threshold": threshold,
                  "oof": score, "oof_at_0.5": fixed}
        results.append(result)
        print(json.dumps(result), flush=True)
    destination = ROOT / "reports/af_external_weight_search.json"
    destination.write_text(json.dumps({"protocol": "3 date folds, 2019-2023 only",
                                       "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.weights):
        parser.error("weights must be in (0, 1]")
    main(args.weights, args.threads)

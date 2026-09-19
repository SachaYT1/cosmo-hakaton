"""Fit and activate the best audited external-data AF configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from firemon import af_models
from firemon.af import feature_names
from scripts.train_af import ROOT, build_external


def main(threads: int = 4) -> None:
    search_path = ROOT / "reports/af_external_weight_search.json"
    search = json.loads(search_path.read_text())
    best = max(search["results"], key=lambda item: item["oof"]["F1_af"])
    weight = float(best["external_weight"])
    X = np.load(ROOT / "cache/af_experiments/X.npy", mmap_mode="r")
    labels = np.load(ROOT / "cache/af_experiments/labels.npz")
    y = labels["y"]
    external_x, external_y = build_external(
        str(ROOT / "data/external/noaa_af_chips"), "context")
    train_x = np.concatenate([X, external_x])
    train_y = np.concatenate([y, external_y])
    sample_weight = np.concatenate([
        np.ones(len(y), np.float32),
        np.full(len(external_y), weight, np.float32),
    ])
    training = json.loads((ROOT / "configs/af_train.json").read_text())
    member = training["models"][0]
    params = {**member["params"], "n_jobs": threads, "device_type": "cpu"}
    model = af_models.create("lgb", params)
    model.fit(train_x, train_y, sample_weight=sample_weight)
    weights = ROOT / "weights/af_augmented_best_lgb.txt"
    af_models.save(model, "lgb", weights)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    runtime = {
        "threshold": best["threshold"], "threshold_rule": ">=",
        "cv_f1": best["oof"]["F1_af"],
        "cv_protocol": search["protocol"],
        "external_weight": weight,
        "models": [{
            "file": str(weights.relative_to(ROOT)), "kind": "lgb",
            "feature_set": "context", "feature_names": feature_names("context"),
            "weight": 1.0,
        }],
    }
    config = ROOT / "configs/af_augmented_best.json"
    config.write_text(json.dumps(runtime, indent=2) + "\n")
    # af_baseline.json remains an immutable copy of the previous winner.
    (ROOT / "configs/af.json").write_text(json.dumps(runtime, indent=2) + "\n")
    report = {
        "selected_external_weight": weight,
        "selection": best,
        "organizer_candidate_pixels": len(y),
        "external_candidate_pixels": len(external_y),
        "external_positive_pixels": int(external_y.sum()),
        "weights": str(weights.relative_to(ROOT)), "sha256": sha,
        "previous_baseline_cv_f1": 0.9527591400794194,
        "external_source_report": "reports/af_external_data.json",
    }
    (ROOT / "reports/af_augmented_best.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

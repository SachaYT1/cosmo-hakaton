"""Train CPU AF models with pooled-pixel F1 and acquisition-date grouped CV.

    python -m scripts.train_af --data-dir /path/to/train --threads 6

Hyperparameters and feature sets are read from configs/af_train.json. Thresholds
are calibrated on out-of-fold predictions. --validate-year additionally fits
only on earlier years and evaluates the requested year with a threshold chosen
on those earlier years. It does not tune on the held-out year.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from firemon.af import candidates, extract_features, feature_names, rule_predict
from firemon import af_models
from firemon.af_eval import counts, date_folds, metrics, tune_threshold
from firemon.io import VIIRS_BANDS, list_chips, load_af
from firemon.external_af import validate_chip

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=2, tree_method="hist",
              random_state=SEED, n_jobs=6, eval_metric="logloss")


def build(root: str, feature_set: str = "legacy"):
    """Features of candidates, plus all false negatives outside that filter."""
    X, Y, G, CH = [], [], [], []
    ids = list_chips(root, "af")
    meta = pd.read_csv(Path(root) / "af" / "meta.csv").set_index("chip_id")
    if not ids:
        raise ValueError(f"No AF chips found in {root}")
    missed = np.zeros(len(ids), np.int64)
    rule = np.zeros((len(ids), 3), np.int64)
    chip_sza = np.zeros(len(ids), np.float32)
    for i, cid in enumerate(ids):
        c = load_af(root, cid)
        if c.mask is None:
            raise ValueError(f"Missing training mask for {cid}")
        chip_sza[i] = np.nanmedian(c.viirs[..., VIIRS_BANDS.index("solar_zenith")])
        f = extract_features(c, feature_set)
        cand = candidates(f)
        y = c.mask > 0
        missed[i] = (y & ~cand).sum()
        p = rule_predict(c) > 0
        rule[i] = [(p & y).sum(), (p & ~y).sum(), (~p & y).sum()]
        X.append(f[cand]); Y.append(y[cand])
        G.append(np.full(cand.sum(), str(meta.loc[cid, "acq_datetime"])[:10]))
        CH.append(np.full(cand.sum(), i, np.int16))
    total_fire = int((rule[:, 0]+rule[:, 2]).sum())
    print(f"Candidate recall: {1-missed.sum()/total_fire if total_fire else 1:.6f}; "
          f"missed fire pixels: {missed.sum()}", flush=True)
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(CH),
            missed, rule, chip_sza, ids, meta)


def build_external(root: str, feature_set: str):
    """Load only approved weak labels; 255 is unknown and never becomes a target."""
    root_path = Path(root)
    manifest = json.loads((root_path / "manifest.json").read_text())
    if manifest.get("policy") != "winter-2019-2025-outside-eurasia-v1":
        raise ValueError("External source lacks the approved leakage policy")
    X, y = [], []
    for item in manifest["chips"]:
        validate_chip(item)
        chip = load_af(root_path, item["chip_id"])
        if chip.mask is None or chip.mask.shape != (256, 256):
            raise ValueError(f"Missing external mask: {item['chip_id']}")
        if not np.isin(chip.mask, [0, 1, 255]).all():
            raise ValueError(f"Unexpected external labels: {item['chip_id']}")
        feat = extract_features(chip, feature_set)
        selected = candidates(feat) & (chip.mask != 255)
        X.append(feat[selected]); y.append(chip.mask[selected])
    if not X or not any(len(part) for part in X):
        raise ValueError("No eligible external candidates")
    return np.concatenate(X), np.concatenate(y)


def predict_fit(X, y, tr, va, members, threads, external=None, external_weight=0.1):
    """Train each member sequentially to limit memory and average probabilities."""
    out = np.zeros(int(va.sum()), np.float32)
    total = sum(m.get("weight", 1.0) for m in members)
    for member in members:
        nf = len(feature_names(member["feature_set"]))
        kind = member.get("kind", "xgb")
        params = {**(PARAMS if kind == "xgb" else {}), **member["params"], "n_jobs": threads}
        params["device" if kind == "xgb" else "device_type"] = "cpu"
        model = af_models.create(kind, params)
        if external is None:
            model.fit(X[tr, :nf], y[tr])
        else:
            ex, ey = external
            train_x = np.concatenate([X[tr, :nf], ex[:, :nf]])
            train_y = np.concatenate([y[tr], ey])
            weights = np.concatenate([np.ones(tr.sum(), np.float32),
                                      np.full(len(ey), external_weight, np.float32)])
            model.fit(train_x, train_y, sample_weight=weights)
        out += (member.get("weight", 1.0)/total) * af_models.probability(model, X[va, :nf])
        del model
        gc.collect()
    return out


def cross_validate(X, y, chip, dates, selected, members, folds, threads,
                   external=None, external_weight=0.1):
    chip_fold = np.full(len(dates), -1, np.int16)
    chip_fold[selected] = date_folds(dates[selected], folds, SEED)
    fold = chip_fold[chip]
    prob = np.full(len(y), np.nan, np.float32)
    for f in range(folds):
        tic = time.monotonic()
        tr, va = (fold >= 0) & (fold != f), fold == f
        prob[va] = predict_fit(X, y, tr, va, members, threads, external, external_weight)
        print(f"Fold {f+1}/{folds}: {time.monotonic()-tic:.1f}s", flush=True)
    return prob, fold, chip_fold


def main(root: str, folds: int = 5, threads: int = 6,
         train_config: str | None = None, validate_year: int | None = None,
         validation_folds: int = 3, external_dir: str | None = None,
         external_weight: float = 0.02) -> None:
    tic = time.monotonic()
    config_path = Path(train_config) if train_config else ROOT/"configs/af_train.json"
    spec = json.loads(config_path.read_text()) if config_path.exists() else {
        "models": [{"feature_set": "legacy", "params": PARAMS, "weight": 1.0}]}
    members = spec["models"]
    if not members or any(m.get("weight", 1.0) < 0 for m in members) or sum(m.get("weight", 1.0) for m in members) <= 0:
        raise ValueError("Need model members with nonnegative weights and positive total weight")
    feature_set = "context" if any(m["feature_set"] == "context" for m in members) else "legacy"
    X, y, _, chip, missed, rule, chip_sza, ids, meta = build(root, feature_set)
    if not 0 < external_weight <= 1:
        raise ValueError("External weight must be in (0, 1]")
    external = build_external(external_dir, feature_set) if external_dir else None
    ordered = meta.loc[ids]
    dates = ordered.acq_datetime.str[:10].to_numpy()
    years = pd.to_datetime(ordered.acq_datetime, utc=True).dt.year.to_numpy()
    all_chips = np.ones(len(ids), bool)
    report = {"metric": "pooled pixel F1 = 2TP/(2TP+FP+FN); empty denominator = 1",
              "seed": SEED, "folds": folds, "chips": len(ids), "candidate_pixels": len(y),
              "missed_fire_pixels": int(missed.sum()), "models": members,
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__}}
    report["external_training"] = ({"source": external_dir, "candidate_pixels": len(external[1]),
        "positive_pixels": int(external[1].sum()), "weight": external_weight,
        "label_quality": "weak NOAA fire detection, not competition ground truth"}
        if external is not None else None)
    if any(m.get("kind") == "lgb" for m in members):
        import lightgbm
        report["versions"]["lightgbm"] = lightgbm.__version__
    if validate_year is not None:
        dev, holdout = years < validate_year, years == validate_year
        if not dev.any() or not holdout.any():
            raise ValueError("Validation year must have chips and some earlier training years")
        print(f"Temporal validation: train {dev.sum()} chips; year {validate_year}: {holdout.sum()} chips", flush=True)
        dev_prob, _, _ = cross_validate(X, y, chip, dates, dev, members, validation_folds,
                                        threads, external, external_weight)
        t, dev_metrics = tune_threshold(dev_prob[dev[chip]], y[dev[chip]], int(missed[dev].sum()))
        val_prob = predict_fit(X, y, dev[chip], holdout[chip], members, threads,
                               external, external_weight)
        val_metrics = metrics(counts(val_prob, y[holdout[chip]], t, int(missed[holdout].sum())))
        report["temporal_holdout"] = {"year": validate_year, "calibration_folds": validation_folds,
            "threshold_from_earlier_years": t,
            "dev_oof": dev_metrics, "holdout": val_metrics,
            "holdout_at_0.5": metrics(counts(val_prob, y[holdout[chip]], 0.5, int(missed[holdout].sum())))}
        print("Temporal holdout:", json.dumps(report["temporal_holdout"]), flush=True)
    prob, fold, chip_fold = cross_validate(X, y, chip, dates, all_chips, members, folds,
                                           threads, external, external_weight)
    threshold, pooled = tune_threshold(prob, y, int(missed.sum()))
    report["oof"] = {"threshold": threshold, **pooled,
                     "note": "Threshold selected on these OOF predictions; this is a calibration score."}
    report["oof_at_0.5"] = metrics(counts(prob, y, 0.5, int(missed.sum())))
    report["folds_at_selected_threshold"] = [
        metrics(counts(prob[fold == f], y[fold == f], threshold, int(missed[chip_fold == f].sum())))
        for f in range(folds)]
    report["day_night"] = {}
    for name, selected in (("day", chip_sza < 85), ("night", chip_sza >= 85)):
        report["day_night"][name] = metrics(counts(prob[selected[chip]], y[selected[chip]], threshold,
                                                  int(missed[selected].sum())))
    cache = ROOT/"cache"; cache.mkdir(exist_ok=True)
    # The selected submission baseline in configs/af.json is immutable here.
    stem = "af_augmented" if external is not None else "af_candidate"
    np.savez_compressed(cache/f"{stem}_oof.npz", prob=prob, y=y, chip=chip, fold=fold.astype(np.int8),
        chip_fold=chip_fold, missed=missed, rule=rule, chip_sza=chip_sza, ids=np.array(ids),
        satellite=ordered.satellite.to_numpy(str), acq_datetime=ordered.acq_datetime.to_numpy(str),
        threshold=threshold, threshold_rule=">=")
    (ROOT/"weights").mkdir(exist_ok=True)
    runtime_members = []
    for i, member in enumerate(members):
        kind = member.get("kind", "xgb")
        params = {**(PARAMS if kind == "xgb" else {}), **member["params"], "n_jobs": threads}
        params["device" if kind == "xgb" else "device_type"] = "cpu"
        nf = len(feature_names(member["feature_set"]))
        if external is None:
            model = af_models.create(kind, params).fit(X[:, :nf], y)
        else:
            ex, ey = external
            all_x = np.concatenate([X[:, :nf], ex[:, :nf]])
            all_y = np.concatenate([y, ey])
            weights = np.concatenate([np.ones(len(y), np.float32),
                                      np.full(len(ey), external_weight, np.float32)])
            model = af_models.create(kind, params).fit(all_x, all_y, sample_weight=weights)
        suffix = "json" if kind == "xgb" else "txt"
        name = f"{stem}_{kind}{'_' + str(i+1) if i else ''}.{suffix}"
        af_models.save(model, kind, ROOT/"weights"/name)
        runtime_members.append({"file": f"weights/{name}", "kind": kind, "feature_set": member["feature_set"],
                                "feature_names": feature_names(member["feature_set"]),
                                "weight": member.get("weight", 1.0)})
        report.setdefault("feature_importance", []).append(sorted(
            zip(feature_names(member["feature_set"]), map(float, af_models.importance(model, kind))),
            key=lambda x: x[1], reverse=True)[:20])
        del model; gc.collect()
    runtime = {"threshold": threshold, "threshold_rule": ">=", "cv_f1": pooled["F1_af"],
               "models": runtime_members, "training_config": str(config_path.relative_to(ROOT))
               if config_path.is_relative_to(ROOT) else str(config_path)}
    (ROOT/"configs").mkdir(exist_ok=True)
    (ROOT/f"configs/{stem}.json").write_text(json.dumps(runtime, indent=2)+"\n")
    report["elapsed_seconds"] = round(time.monotonic()-tic, 1)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mb"] = round(rss / (1024**2 if sys.platform == "darwin" else 1024), 1)
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/f"reports/{stem}_training.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Final OOF:", json.dumps(report["oof"]), flush=True)
    print(f"Saved models and reports in {ROOT}; {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--config", help="Training configuration; defaults to configs/af_train.json")
    ap.add_argument("--validate-year", type=int, help="Optional temporal holdout; train only on earlier years")
    ap.add_argument("--validation-folds", type=int, default=3, help="Earlier-year calibration folds for the temporal holdout")
    ap.add_argument("--external-dir", help="Approved AF chips with weak labels and provenance manifest")
    ap.add_argument("--external-weight", type=float, default=0.02)
    a = ap.parse_args()
    if a.threads < 1:
        ap.error("--threads must be positive")
    main(a.data_dir, a.folds, a.threads, a.config, a.validate_year,
         a.validation_folds, a.external_dir, a.external_weight)

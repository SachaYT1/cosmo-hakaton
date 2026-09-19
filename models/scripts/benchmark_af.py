"""Reproduce CPU AF trials on earlier years, reserving the last year.

    python -m scripts.benchmark_af --data-dir /path/to/train --seconds 1800

Completed trials are resumable. The parent enforces a wall-clock timeout, even
inside native model fitting. No model weights or production config are changed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from firemon import af
from firemon.af_eval import date_folds, tune_threshold
from firemon.io import load_af
from scripts.train_af import PARAMS, ROOT


def cache_data(root: Path, out: Path):
    digest = hashlib.sha256(Path(af.__file__).read_bytes())
    digest.update((root/"af/meta.csv").read_bytes())
    for folder in ("viirs", "aux", "masks"):
        for p in sorted((root/"af"/folder).glob("*.tif")):
            digest.update(f"{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
    signature = digest.hexdigest()
    manifest = out/"cache_signature.txt"
    if not manifest.exists() or manifest.read_text() != signature:
        m = pd.read_csv(root/"af/meta.csv")
        m = m.loc[m.kind.eq("af")].sort_values("chip_id").reset_index(drop=True)
        m["year"] = pd.to_datetime(m.acq_datetime, utc=True).dt.year
        xs, ys, chips, missed = [], [], [], []
        for i, cid in enumerate(m.chip_id):
            chip = load_af(root, cid)
            if chip.mask is None:
                raise ValueError(f"Missing training mask: {cid}")
            f = af.enhanced_features(chip); cand = af.candidates(f); truth = chip.mask > 0
            xs.append(f[cand]); ys.append(truth[cand])
            chips.append(np.full(int(cand.sum()), i, np.int16))
            missed.append(int((truth & ~cand).sum()))
        np.save(out/"X.npy", np.concatenate(xs))
        np.savez(out/"labels.npz", y=np.concatenate(ys), chip=np.concatenate(chips), missed=missed)
        m.to_csv(out/"meta.csv", index=False)
        manifest.write_text(signature)
        del xs, ys, chips; gc.collect()
    return (np.load(out/"X.npy", mmap_mode="r"), dict(np.load(out/"labels.npz")),
            pd.read_csv(out/"meta.csv"), signature)


def run(a):
    out = Path(a.cache_dir); out.mkdir(parents=True, exist_ok=True)
    X, z, meta, signature = cache_data(Path(a.data_dir), out)
    y, chip, missed = z["y"], z["chip"], z["missed"]
    year = a.holdout_year if a.holdout_year is not None else int(meta.year.max())
    dev = (meta.year < year).to_numpy()
    dates = meta.acq_datetime.str[:10].to_numpy()
    cf = np.full(len(meta), -1, np.int16)
    cf[dev] = date_folds(dates[dev], a.folds)
    fold = cf[chip]
    trials = json.loads(Path(a.config).read_text())["trials"]
    if a.trial:
        trials = [t for t in trials if t["name"] == a.trial]
        if not trials:
            raise ValueError(f"Unknown trial: {a.trial}")
    for trial in trials:
        path = out/(trial["name"]+".json")
        run_signature = hashlib.sha256(json.dumps(
            [signature, trial, a.folds, year, a.threads], sort_keys=True).encode()).hexdigest()
        if path.exists() and json.loads(path.read_text()).get("run_signature") == run_signature:
            print("Already completed:", trial["name"], flush=True)
            continue
        tic = time.monotonic()
        nf = len(af.feature_names(trial["feature_set"]))
        p = np.full(len(y), np.nan, np.float32)
        kind = trial.get("kind", "xgb")
        params = {**(PARAMS if kind == "xgb" else {}), **trial["params"], "n_jobs": a.threads}
        for f in range(a.folds):
            tr, va = (fold >= 0) & (fold != f), fold == f
            if kind == "xgb":
                model = xgb.XGBClassifier(**params)
            elif kind == "lgb":
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params)
            else:
                raise ValueError(f"Unknown model kind: {kind}")
            model.fit(X[tr, :nf], y[tr])
            p[va] = model.predict_proba(X[va, :nf])[:, 1]
            del model; gc.collect()
            print(trial["name"], "fold", f+1, "seconds", round(time.monotonic()-tic, 1), flush=True)
        threshold, best = tune_threshold(p[fold >= 0], y[fold >= 0], int(missed[dev].sum()))
        result = {**trial, "params": params, "run_signature": run_signature,
                  "holdout_year_reserved": year, "folds": a.folds,
                  "best": {"threshold": threshold, **best}, "seconds": round(time.monotonic()-tic, 1)}
        np.save(out/(trial["name"]+"_oof.npy"), p)
        path.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=str(ROOT/"configs/af_search.json"))
    ap.add_argument("--cache-dir", default=str(ROOT/"cache/af_benchmark"))
    ap.add_argument("--trial")
    ap.add_argument("--holdout-year", type=int)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.seconds <= 0 or a.threads <= 0:
        ap.error("--seconds and --threads must be positive")
    if a.worker:
        run(a)
    else:
        try:
            subprocess.run([sys.executable, "-m", "scripts.benchmark_af", *sys.argv[1:], "--worker"],
                           timeout=a.seconds, check=True)
        except subprocess.TimeoutExpired:
            print("Time budget exhausted. Completed trial results are retained.")
        except subprocess.CalledProcessError as e:
            raise SystemExit(e.returncode)

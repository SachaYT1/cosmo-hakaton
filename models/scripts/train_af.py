"""Train the AF classifier (XGBoost on contextual pixel features of candidate pixels).

CV: 5 folds grouped by acquisition date, so chips of the same overpass never leak across folds.
The decision threshold is chosen on out-of-fold predictions to maximise pooled F1.
Writes weights/af_xgb.json, configs/af.json and cache/af_oof.npz (per-candidate OOF probabilities
plus per-chip rule-baseline counts, used by scripts/eval_af.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from firemon.af import FEATURES, candidates, features, rule_predict
from firemon.io import VIIRS_BANDS, list_chips, load_af

SEED = 42
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=2, tree_method="hist",
              random_state=SEED, n_jobs=8, eval_metric="logloss")


def build(root: str):
    """Candidate-pixel features plus per-chip bookkeeping.

    missed[i] = fire pixels of chip i dropped by the candidate filter (always FN);
    rule[i] = (tp, fp, fn) of the rule baseline on chip i.
    """
    X, Y, G, CH = [], [], [], []
    ids = list_chips(root, "af")
    meta = pd.read_csv(Path(root) / "af" / "meta.csv").set_index("chip_id")
    missed = np.zeros(len(ids), np.int64)
    rule = np.zeros((len(ids), 3), np.int64)
    chip_sza = np.zeros(len(ids), np.float32)
    for i, cid in enumerate(ids):
        c = load_af(root, cid)
        chip_sza[i] = np.nanmedian(c.viirs[..., VIIRS_BANDS.index("solar_zenith")])
        f = features(c)
        cand = candidates(f)
        y = c.mask > 0
        missed[i] = (y & ~cand).sum()
        p = rule_predict(c) > 0
        rule[i] = [(p & y).sum(), (p & ~y).sum(), (~p & y).sum()]
        X.append(f[cand]); Y.append(y[cand])
        # group = acquisition date (string, deterministic; builtin hash() is salted per process)
        G.append(np.full(cand.sum(), meta.loc[cid, "acq_datetime"][:10]))
        CH.append(np.full(cand.sum(), i))
    print(f"candidates keep {1 - missed.sum() / (rule[:, 0] + rule[:, 2]).sum():.4f} of fire px")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(CH),
            missed, rule, chip_sza, ids, meta)


def main(root: str, folds: int) -> None:
    X, Y, G, CH, missed_chip, rule, chip_sza, ids, meta = build(root)
    missed = missed_chip.sum()
    print("candidate pixels", len(Y), "positives", Y.sum())
    groups = np.unique(G)
    fold_of_group = dict(zip(groups, np.random.default_rng(SEED).permutation(len(groups)) % folds))
    fold = np.array([fold_of_group[g] for g in G])
    oof = np.zeros(len(Y))
    for f in range(folds):
        tr, va = fold != f, fold == f
        m = xgb.XGBClassifier(**PARAMS).fit(X[tr], Y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    best = (0, 0.5)
    for t in np.arange(0.1, 0.9, 0.02):
        p = oof > t
        tp, fp = (p & Y).sum(), (p & ~Y).sum()
        fn = (~p & Y).sum() + missed
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(t))
    print(f"CV (grouped by date) F1_af={best[0]:.4f} at threshold {best[1]:.2f}")
    np.savez_compressed("cache/af_oof.npz", prob=oof.astype(np.float32), y=Y, chip=CH.astype(np.int32),
                        fold=fold.astype(np.int8), missed=missed_chip, rule=rule, chip_sza=chip_sza, ids=np.array(ids),
                        satellite=meta.loc[ids, "satellite"].to_numpy(str),
                        acq_datetime=meta.loc[ids, "acq_datetime"].to_numpy(str), threshold=best[1])
    model = xgb.XGBClassifier(**PARAMS).fit(X, Y)
    Path("weights").mkdir(exist_ok=True)
    model.save_model("weights/af_xgb.json")
    json.dump({"threshold": round(best[1], 2), "cv_f1": round(best[0], 4)}, open("configs/af.json", "w"), indent=1)
    imp = sorted(zip(model.feature_importances_, FEATURES), reverse=True)[:12]
    print("top features:", [(n, round(float(v), 3)) for v, n in imp])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.data_dir, a.folds)

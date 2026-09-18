"""Build a submission.csv for the TRAINING chips from out-of-fold predictions and score it
through the submission format (RLE encode -> CSV -> decode -> metric).

AF: XGBoost OOF probabilities from cache/af_oof.npz (scripts/train_af.py).
BS: rule model, same 5-fold split as scripts/fit_bs_rules.py; each chip is predicted with
    thresholds fitted on the other four folds.

As a consistency check the metric is also computed directly from the in-memory masks;
both numbers must match.

    python -m scripts.oof_submission   # -> outputs/oof_submission_train.csv, reports/oof_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from firemon.af import candidates, features
from firemon.bs_rules import invalid_mask, rule_predict, smooth_dnbr
from firemon.cache import bs_layers
from firemon.io import load_af
from firemon.metric import Accumulator, score_submission
from firemon.rle import encode
from inference import write_submission
from scripts.fit_bs_rules import fit
from scripts.validate_submission import validate

SEED = 42


def af_oof_masks(root: str, oof: str):
    z = dict(np.load(oof))
    t = float(z["threshold"])
    for i, cid in enumerate(z["ids"]):
        chip = load_af(root, cid)
        cand = candidates(features(chip))
        pred = np.zeros(cand.shape, bool)
        pred[cand] = z["prob"][z["chip"] == i] > t
        yield str(cid), pred.astype(np.uint8), chip.mask


def bs_oof_masks(root: str, folds: int):
    L = bs_layers(root)
    med = np.stack([smooth_dnbr(d.astype(np.float32)) for d in L["dnbr"]])
    lc, mask = L["lc"], L["mask"]
    invalid = np.stack([invalid_mask(a, b) for a, b in zip(L["scl_pre"], L["scl_post"])])
    fold_of = np.random.default_rng(SEED).permutation(len(mask)) % folds  # same split as fit_bs_rules
    for f in range(folds):
        tr, va = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        th, ms = fit(med, lc, mask, invalid, tr)
        print(f"BS fold {f}: fitted on {len(tr)} chips, predicting {len(va)}", flush=True)
        for i in va:
            yield str(L["ids"][i]), rule_predict(med[i], lc[i], invalid[i], th, ms), mask[i]


def main(root: str, oof: str, out: str, report: str, folds: int) -> None:
    rows, direct = [], Accumulator()
    for cid, pred, true in af_oof_masks(root, oof):
        rows.append((cid, 1, encode(pred == 1)))
        direct.add_af(pred, true)
    for cid, pred, true in bs_oof_masks(root, folds):
        rows += [(cid, k, encode(pred == k)) for k in (1, 2, 3)]
        direct.add_bs(pred, true)

    sub = pd.DataFrame(rows, columns=["chip_id", "class_id", "rle"]).sort_values(["chip_id", "class_id"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_submission(sub, out)

    # format check with the same validator as for the test submission
    meta = pd.concat([pd.read_csv(Path(root) / k / "meta.csv") for k in ("af", "bs")])
    Path("cache").mkdir(exist_ok=True)
    meta.to_csv("cache/train_meta_all.csv", index=False)
    sub[["chip_id", "class_id"]].assign(rle="").to_csv("cache/train_template.csv", index=False)
    errors = validate(out, "cache/train_template.csv", "cache/train_meta_all.csv")
    print("format:", "VALID" if not errors else errors[:10])

    via_csv = score_submission(out, root)
    ref = direct.result()
    same = all(abs(via_csv[k] - ref[k]) < 1e-12 for k in ref)
    n_af = int(sub.chip_id.str.startswith("AF").sum())
    print(f"\nrows {len(sub)}: AF {n_af} (1 per chip), BS {len(sub) - n_af} (3 per chip)")
    print("metric via submission.csv:", json.dumps({k: round(v, 4) for k, v in via_csv.items()}))
    print("metric from masks (check):", "identical" if same else json.dumps(ref))
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"via_submission": via_csv, "direct_equal": same, "format_errors": errors[:10],
               "af": "xgb OOF (5-fold grouped by date)", "bs": f"rule model OOF ({folds}-fold by chip)"},
              open(report, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--oof", default="cache/af_oof.npz")
    ap.add_argument("--out", default="outputs/oof_submission_train.csv")
    ap.add_argument("--report", default="reports/oof_metrics.json")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.data_dir, a.oof, a.out, a.report, a.folds)

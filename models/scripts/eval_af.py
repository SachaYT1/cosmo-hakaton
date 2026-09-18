"""Break down AF out-of-fold F1 by day / night (and satellite), for XGBoost and the rule baseline.

Needs cache/af_oof.npz from scripts/train_af.py. A chip is "night" if its median solar zenith
angle is >= 85 deg (same definition as the `day` feature). F1 is pooled over pixels (micro), as
in the competition metric.

The test set has a larger night share than train, so we also report F1 with train chip counts
reweighted to the test day/night mix (night/day shares measured on data/test the same way).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from firemon.af import DAY_SZA
from firemon.io import VIIRS_BANDS, list_chips, load_af


def f1(c: np.ndarray) -> float:
    tp, fp, fn = c
    return 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def chip_counts(z, threshold: float | np.ndarray) -> np.ndarray:
    """(n_chips, 3) tp/fp/fn of the XGB model at `threshold` (scalar or per-chip array)."""
    n = len(z["ids"])
    t = np.broadcast_to(threshold, (n,))[z["chip"]]
    inclusive = str(z.get("threshold_rule", ">")) == ">="
    p, y = (z["prob"] >= t if inclusive else z["prob"] > t), z["y"]
    c = np.zeros((n, 3), np.int64)
    np.add.at(c[:, 0], z["chip"], p & y)
    np.add.at(c[:, 1], z["chip"], p & ~y)
    np.add.at(c[:, 2], z["chip"], ~p & y)
    c[:, 2] += z["missed"]
    return c


def test_night_share(test_root: str) -> float:
    ids = list_chips(test_root, "af")
    sza = [np.nanmedian(load_af(test_root, c, with_mask=False).viirs[..., VIIRS_BANDS.index("solar_zenith")])
           for c in ids]
    return float(np.mean(np.array(sza) >= DAY_SZA))


def table(name: str, c: np.ndarray, night: np.ndarray, w: np.ndarray) -> dict:
    row = {"all": f1(c.sum(0)), "day": f1(c[~night].sum(0)), "night": f1(c[night].sum(0)),
           "test_mix": f1((c * w[:, None]).sum(0))}
    for k, sel in (("day", ~night), ("night", night)):
        tp, fp, fn = c[sel].sum(0)
        row[f"{k}_prec"] = tp / max(tp + fp, 1)
        row[f"{k}_rec"] = tp / max(tp + fn, 1)
        row[f"{k}_fire_px"] = int(tp + fn)
    print(f"{name:34s} all {row['all']:.4f} | day {row['day']:.4f} (P {row['day_prec']:.3f} R {row['day_rec']:.3f})"
          f" | night {row['night']:.4f} (P {row['night_prec']:.3f} R {row['night_rec']:.3f})"
          f" | test-mix {row['test_mix']:.4f}")
    return row


def main(oof_path: str, test_root: str, out: str) -> None:
    z = dict(np.load(oof_path))
    night = z["chip_sza"] >= DAY_SZA
    tr_night = night.mean()
    te_night = test_night_share(test_root)
    w = np.where(night, te_night / tr_night, (1 - te_night) / (1 - tr_night))
    print(f"night chips: train {tr_night:.3f} ({night.sum()}/{len(night)}), test {te_night:.3f}")
    print(f"fire px: day {int((z['rule'][~night, 0] + z['rule'][~night, 2]).sum())}, "
          f"night {int((z['rule'][night, 0] + z['rule'][night, 2]).sum())}")

    res = {"night_share_train": float(tr_night), "night_share_test": te_night}
    res["rule"] = table("rule baseline (fit on all train)", z["rule"], night, w)
    t = float(z["threshold"])
    res["xgb"] = table(f"xgb OOF, one threshold {t:.2f}", chip_counts(z, t), night, w)

    # Diagnostic cross-fitted thresholds, NOT a strict nested-CV estimate:
    # models behind the other OOF folds have seen this fold's training labels.
    # A temporal holdout with calibration restricted to earlier years is reported
    # separately by train_af.py --validate-year.
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    chip_fold = z.get("chip_fold", np.full(len(night), -1))
    chip_fold[z["chip"]] = z["fold"]
    per_chip_t = np.full(len(night), t)
    for f in np.unique(z["fold"]):
        other = chip_fold != f
        for sel in (~night, night):
            scores = [f1(chip_counts(z, g)[other & sel].sum(0)) for g in grid]
            per_chip_t[(chip_fold == f) & sel] = grid[int(np.argmax(scores))]
    res["xgb_daynight_t"] = table("xgb OOF, day/night thresholds", chip_counts(z, per_chip_t), night, w)
    full = {}
    for k, sel in (("day", ~night), ("night", night)):
        scores = [f1(chip_counts(z, g)[sel].sum(0)) for g in grid]
        full[k] = float(grid[int(np.argmax(scores))])
    res["daynight_thresholds_full"] = full
    print("day/night thresholds fitted on all OOF:", full)

    # F1 vs threshold curve (stability of the operating point)
    res["threshold_curve"] = {str(g): {k: f1(chip_counts(z, g)[sel].sum(0))
                                       for k, sel in (("all", np.ones_like(night)), ("day", ~night), ("night", night))}
                              for g in grid[::5]}
    for s in np.unique(z["satellite"]):
        sel = z["satellite"] == s
        c = chip_counts(z, t)
        print(f"  {s:7s} chips {sel.sum():3d}  F1 {f1(c[sel].sum(0)):.4f}  "
              f"(day {f1(c[sel & ~night].sum(0)):.4f}, night {f1(c[sel & night].sum(0)):.4f})")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="cache/af_oof.npz")
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--out", default="reports/af_daynight.json")
    a = ap.parse_args()
    main(a.oof, a.test_dir, a.out)

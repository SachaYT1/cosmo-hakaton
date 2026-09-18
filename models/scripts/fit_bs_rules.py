"""Fit per-landcover dNBR thresholds for the rule-based BS module and evaluate with 5-fold CV.

Severity cut-offs (t_12, t_23) are fitted on truly burned pixels (maximise accuracy);
the burn cut-off t_burn and the minimum component size maximise IoU_burn.
Writes configs/bs_thresholds.json (fit on all training chips) and prints CV metrics.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from firemon.bs_rules import Thresholds, invalid_mask, rule_predict, smooth_dnbr
from firemon.cache import bs_layers
from firemon.metric import Accumulator

GRID = np.round(np.arange(0.0, 1.0, 0.01), 2)
MIN_PIX = 20000  # landcover classes with fewer burned px fall back to the default


def fit_severity(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Exhaustive search of (t12, t23) maximising accuracy via cumulative histograms."""
    idx = np.clip(np.searchsorted(GRID, x, side="right") - 1, 0, len(GRID) - 1)
    c = [np.bincount(idx[y == k], minlength=len(GRID)).cumsum() for k in (1, 2, 3)]
    best, arg = -1, (0.2, 0.4)
    for i in range(1, len(GRID)):
        for j in range(i + 1, len(GRID)):
            # class1 below GRID[i], class2 in [GRID[i], GRID[j]), class3 at/above GRID[j]
            acc = c[0][i - 1] + (c[1][j - 1] - c[1][i - 1]) + (c[2][-1] - c[2][j - 1])
            if acc > best:
                best, arg = acc, (float(GRID[i]), float(GRID[j]))
    return arg


def fit(med, lc, mask, invalid, idx) -> tuple[Thresholds, int]:
    sel_b = mask[idx] > 0
    lcs, m, lcb = lc[idx], mask[idx], None
    table = {"default": [0.1, *fit_severity(med[idx][sel_b], m[sel_b])]}
    for l in np.unique(lcs):
        s = (lcs == l) & sel_b
        if s.sum() >= MIN_PIX:
            table[str(int(l))] = [0.1, *fit_severity(med[idx][s], m[s])]
    th = Thresholds(table)
    # burn threshold: one global offset (per-class offsets over-fit on rare classes) + min size
    best = (-1, 0.1, 0)
    for t0 in np.arange(0.04, 0.26, 0.02):
        for ms in (0, 10, 50, 200):
            for k in table:
                table[k][0] = float(t0)
            th = Thresholds(table)
            acc = Accumulator()
            for i in idx:
                acc.add_bs(rule_predict(med[i], lc[i], invalid[i], th, ms), mask[i])
            r = acc.result()
            if r["IoU_burn"] > best[0]:
                best = (r["IoU_burn"], float(t0), ms)
    for k in table:
        table[k][0] = round(best[1], 2)
    return Thresholds(table), best[2]


def main(root: str, out: str, folds: int) -> None:
    L = bs_layers(root)
    med = np.stack([smooth_dnbr(d.astype(np.float32)) for d in L["dnbr"]])
    lc, mask = L["lc"], L["mask"]
    invalid = np.stack([invalid_mask(a, b) for a, b in zip(L["scl_pre"], L["scl_post"])])
    n = len(mask)
    rng = np.random.default_rng(42)
    fold_of = rng.permutation(n) % folds  # one chip == one fire event, so chip-level folds are grouped
    acc = Accumulator()
    for f in range(folds):
        tr, va = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        th, ms = fit(med, lc, mask, invalid, tr)
        for i in va:
            acc.add_bs(rule_predict(med[i], lc[i], invalid[i], th, ms), mask[i])
    cv = acc.result()
    print("CV:", json.dumps({k: round(v, 4) for k, v in cv.items() if k != "F1_af"}))
    th, ms = fit(med, lc, mask, invalid, np.arange(n))
    th.save(out)
    json.dump({"min_size": ms}, open(out.replace(".json", "_post.json"), "w"))
    print("full-fit thresholds:", th.table, "min_size", ms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--out", default="configs/bs_thresholds.json")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.data_dir, a.out, a.folds)

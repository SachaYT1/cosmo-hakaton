"""Visualise out-of-fold AF predictions on random training chips.

For each chip one PNG with two columns on the same I4 background:
  left  - prediction vs truth: green = TP, red = FP, blue = FN; per-chip precision / recall / F1
  right - reference mask (yellow)
Top row: whole chip. Bottom row: zoom on the box around all fire / predicted pixels.
Thin VIIRS bow-tie gaps (NaN rows) are inpainted for display only; large no-data areas stay black.
Predictions are out-of-fold (cache/af_oof.npz from scripts/train_af.py), i.e. each chip is
predicted by the fold model that did not see it during training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Patch, Rectangle

from firemon.af import DAY_SZA, candidates, features
from firemon.io import VIIRS_BANDS, load_af

COLORS = {"TP": (0.10, 0.85, 0.20), "FP": (0.95, 0.15, 0.15), "FN": (0.15, 0.45, 1.00)}
GT_COLOR = (1.0, 0.85, 0.0)


def oof_mask(z: dict, i: int, chip) -> np.ndarray:
    """Re-place chip i's OOF candidate probabilities (stored in row-major candidate order)."""
    cand = candidates(features(chip))
    prob = np.zeros(cand.shape, np.float32)
    prob[cand] = z["prob"][z["chip"] == i]
    return prob > float(z["threshold"])


def background(chip) -> np.ndarray:
    i4 = chip.viirs[..., VIIRS_BANDS.index("I4")]
    gap = ~np.isfinite(i4)
    lo, hi = np.nanpercentile(i4, [2, 99.5]) if (~gap).any() else (0, 1)
    g8 = (np.clip((np.nan_to_num(i4, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    if gap.any() and (~gap).any():
        # only thin gaps (<= 2 px from valid data); wide no-data areas (off-swath) are left black
        thin = gap & (cv2.distanceTransform(gap.astype(np.uint8), cv2.DIST_L2, 3) <= 2)
        g8 = cv2.inpaint(g8, thin.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        g8[gap & ~thin] = 0
    g = g8.astype(np.float32) / 255 * 0.7  # keep the background dim so the colours stand out
    return np.repeat(g[..., None], 3, -1)


def zoom_box(m: np.ndarray, pad: int = 12, min_size: int = 40) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) square-ish window around all marked pixels (whole chip if none)."""
    h, w = m.shape
    if not m.any():
        return 0, h, 0, w
    r, c = np.nonzero(m)
    size = max(r.max() - r.min(), c.max() - c.min()) + 2 * pad
    size = min(max(size, min_size), max(h, w))
    cr, cc = (r.min() + r.max()) // 2, (c.min() + c.max()) // 2
    r0 = int(np.clip(cr - size // 2, 0, h - size))
    c0 = int(np.clip(cc - size // 2, 0, w - size))
    return r0, r0 + size, c0, c0 + size


def paint(bg: np.ndarray, layers: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
    img = bg.copy()
    for m, c in layers:
        img[m] = c
    return img


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main(root: str, oof: str, out: str, n: int, seed: int) -> None:
    z = dict(np.load(oof))
    ids = z["ids"]
    pick = np.sort(np.random.default_rng(seed).choice(len(ids), n, replace=False))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(pick):
        chip = load_af(root, ids[i])
        y = chip.mask > 0
        p = oof_mask(z, i, chip)
        tp, fp, fn = p & y, p & ~y, ~p & y
        ntp, nfp, nfn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        prec = ntp / (ntp + nfp) if ntp + nfp else None
        rec = ntp / (ntp + nfn) if ntp + nfn else None
        f1 = 2 * ntp / (2 * ntp + nfp + nfn) if ntp + nfp + nfn else None
        when = "night" if z["chip_sza"][i] >= DAY_SZA else "day"
        bg = background(chip)

        pred_img = paint(bg, [(tp, COLORS["TP"]), (fp, COLORS["FP"]), (fn, COLORS["FN"])])
        gt_img = paint(bg, [(y, GT_COLOR)])
        r0, r1, c0, c1 = zoom_box(p | y)

        fig, ax = plt.subplots(2, 2, figsize=(12, 13.5), layout="constrained")
        fig.suptitle(f"{ids[i]}  ·  {z['satellite'][i]}  ·  {when}  ·  {z['acq_datetime'][i][:16]} UTC\n"
                     f"background: I4 brightness temperature (bow-tie gaps filled for display)", fontsize=12)
        ax[0, 0].set_title(f"Prediction (out-of-fold)\nprecision {fmt(prec)}   recall {fmt(rec)}   F1 {fmt(f1)}\n"
                           f"TP {ntp}   FP {nfp}   FN {nfn}", fontsize=11)
        ax[0, 1].set_title(f"Reference mask\n\nfire pixels {int(y.sum())}", fontsize=11)
        for col, img in enumerate((pred_img, gt_img)):
            ax[0, col].imshow(img, interpolation="nearest")
            ax[1, col].imshow(img[r0:r1, c0:c1], interpolation="nearest",
                              extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
            ax[1, col].set_title(f"zoom: rows {r0}-{r1}, cols {c0}-{c1}", fontsize=10)
            ax[0, col].add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                                           fill=False, ec="white", lw=1, ls="--"))
            for a in ax[:, col]:
                a.set_xticks([])
                a.set_yticks([])
        ax[0, 0].legend(handles=[Patch(color=c, label=l) for l, c in COLORS.items()],
                        loc="lower right", fontsize=9, framealpha=0.85)
        ax[0, 1].legend(handles=[Patch(color=GT_COLOR, label="fire (reference)")],
                        loc="lower right", fontsize=9, framealpha=0.85)
        name = f"{k + 1:02d}_{ids[i]}.png"
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        print(f"{name}: {when:5s} TP {ntp:3d} FP {nfp:3d} FN {nfn:3d}  P {fmt(prec)} R {fmt(rec)} F1 {fmt(f1)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--oof", default="cache/af_oof.npz")
    ap.add_argument("--out", default="reports/af_samples")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data_dir, a.oof, a.out, a.n, a.seed)

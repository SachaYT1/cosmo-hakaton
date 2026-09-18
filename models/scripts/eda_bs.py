"""EDA for the burn-severity (BS) module.

Outputs (reports/):
  figures/bs_examples.png         RGB pre | RGB post | dNBR | landcover | SCL post | mask for a few chips
  figures/bs_dnbr_by_landcover.png  dNBR histograms by mask class, one panel per landcover
  bs_eda.json                     pixel counts / class shares / dNBR quantiles per landcover x class
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firemon.io import WORLDCOVER, SCL, list_chips, load_bs
from firemon.features import nbr, rgb

BINS = np.linspace(-1, 1.5, 251)


def main(root: str, out: str, n_examples: int) -> None:
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ids = list_chips(root, "bs")
    meta = pd.read_csv(Path(root) / "bs" / "meta.csv").set_index("chip_id")

    hist: dict[tuple[int, int], np.ndarray] = {}
    scl_vs_mask = np.zeros((12, 12, 4), np.int64)       # scl_pre, scl_post, mask
    lc_counts: dict[int, np.ndarray] = {}
    per_chip = []
    for cid in ids:
        c = load_bs(root, cid)
        d = nbr(c.s2_pre) - nbr(c.s2_post)
        lc = c.aux[..., 2].astype(int)
        m = c.mask.astype(int)
        sp, sq = c.s2_pre[..., 9].astype(int), c.s2_post[..., 9].astype(int)
        np.add.at(scl_vs_mask, (sp.ravel(), sq.ravel(), m.ravel()), 1)
        for l in np.unique(lc):
            sel = lc == l
            lc_counts.setdefault(l, np.zeros(4, np.int64))
            lc_counts[l] += np.bincount(m[sel], minlength=4)
            for k in range(4):
                s = sel & (m == k)
                if s.any():
                    h, _ = np.histogram(d[s], BINS)
                    hist[(l, k)] = hist.get((l, k), 0) + h
        per_chip.append({"chip_id": cid, "burn_frac": (m > 0).mean(), "cloud_frac": meta.loc[cid, "cloud_frac"],
                         "dnbr_burn_med": float(np.nanmedian(d[m > 0])) if (m > 0).any() else np.nan,
                         "dnbr_unburn_med": float(np.nanmedian(d[m == 0])),
                         "top_lc": int(np.bincount(lc.ravel()).argmax())})

    # --- histogram figure
    lcs = sorted(l for l in lc_counts if lc_counts[l].sum() > 1e4)
    fig, axes = plt.subplots(2, (len(lcs) + 1) // 2, figsize=(4 * ((len(lcs) + 1) // 2), 7))
    centers = (BINS[1:] + BINS[:-1]) / 2
    stats = {}
    for ax, l in zip(axes.ravel(), lcs):
        stats[WORLDCOVER.get(l, str(l))] = {"pixels": lc_counts[l].tolist()}
        for k, col in zip(range(4), ["#999", "#f2c14e", "#f07c28", "#b3172b"]):
            h = hist.get((l, k))
            if h is None:
                continue
            ax.plot(centers, h / max(h.sum(), 1), color=col, label=f"class {k}")
            cdf = np.cumsum(h) / h.sum()
            stats[WORLDCOVER.get(l, str(l))][f"class{k}_q"] = [
                round(float(centers[np.searchsorted(cdf, q)]), 3) for q in (0.01, 0.05, 0.5, 0.95, 0.99)]
        ax.set_title(f"{WORLDCOVER.get(l, l)} ({l}) n={lc_counts[l].sum():,}")
        ax.set_xlim(-0.4, 1.2)
        ax.set_xlabel("dNBR")
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "bs_dnbr_by_landcover.png", dpi=110)

    # --- examples
    pc = pd.DataFrame(per_chip)
    ex = pc.sort_values("burn_frac").iloc[np.linspace(10, len(pc) - 1, n_examples).astype(int)].chip_id
    fig, axes = plt.subplots(n_examples, 6, figsize=(21, 3.6 * n_examples))
    for row, cid in zip(axes, ex):
        c = load_bs(root, cid)
        d = nbr(c.s2_pre) - nbr(c.s2_post)
        panels = [(rgb(c.s2_pre), None, "RGB pre"), (rgb(c.s2_post), None, "RGB post"),
                  (d, dict(cmap="RdYlGn_r", vmin=-0.3, vmax=0.8), "dNBR"),
                  (c.aux[..., 2], dict(cmap="tab20", vmin=0, vmax=100, interpolation="nearest"), "landcover"),
                  (c.s2_post[..., 9], dict(cmap="tab20", vmin=0, vmax=11, interpolation="nearest"), "SCL post"),
                  (c.mask, dict(cmap="inferno", vmin=0, vmax=3, interpolation="nearest"), "mask")]
        for ax, (img, kw, t) in zip(row, panels):
            ax.imshow(img, **(kw or {}))
            ax.set_title(f"{cid} {t}" if t == "RGB pre" else t)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "figures" / "bs_examples.png", dpi=70)

    scl_mask = {f"post_{SCL[s]}": np.bincount(np.repeat(np.arange(4), scl_vs_mask[:, s].sum(0)), minlength=4).tolist()
                for s in range(12) if scl_vs_mask[:, s].sum()}
    json.dump({"per_landcover": stats, "scl_post_vs_mask": scl_mask,
               "chip_burn_frac": pc.burn_frac.describe().to_dict(),
               "corr_cloud_vs_burnfrac": float(pc[["cloud_frac", "burn_frac"]].corr().iloc[0, 1])},
              open(out / "bs_eda.json", "w"), indent=1)
    pc.to_csv(out / "bs_per_chip.csv", index=False)
    print(json.dumps(stats, indent=1))
    print(json.dumps(scl_mask, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--n-examples", type=int, default=5)
    a = ap.parse_args()
    main(a.data_dir, a.out, a.n_examples)

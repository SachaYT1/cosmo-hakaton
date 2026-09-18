"""Precompute U-Net inputs for all training BS chips into a float16 memmap (cache/bs_x.npy)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from firemon.bs_unet import N_CHANNELS, bs_input
from firemon.io import list_chips, load_bs


def main(root: str, out: str) -> None:
    ids = list_chips(root, "bs")
    Path(out).mkdir(exist_ok=True)
    x = np.lib.format.open_memmap(f"{out}/bs_x.npy", "w+", np.float16, (len(ids), N_CHANNELS, 512, 512))
    y = np.zeros((len(ids), 512, 512), np.uint8)
    for i, cid in enumerate(ids):
        c = load_bs(root, cid)
        x[i] = bs_input(c).astype(np.float16)
        y[i] = c.mask
    x.flush()
    np.save(f"{out}/bs_y.npy", y)
    np.save(f"{out}/bs_ids.npy", np.array(ids))
    print("done", x.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--out", default="cache")
    a = ap.parse_args()
    main(a.data_dir, a.out)

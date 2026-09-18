"""Produce submission.csv for all chips listed in <data-dir>/sample_submission.csv.

    python inference.py --data-dir /path/to/test --output /path/to/submission.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from firemon.io import load_af, load_bs
from firemon.rle import encode

_PRED = {}


def _init(bs_model: str) -> None:
    from firemon.pipeline import AFPredictor, BSRulePredictor
    _PRED["af"] = AFPredictor()
    if bs_model == "unet":
        from firemon.bs_unet import BSUNetPredictor
        _PRED["bs"] = BSUNetPredictor()
    else:
        _PRED["bs"] = BSRulePredictor()


def _predict(args: tuple[str, str]) -> list[tuple[str, int, str]]:
    root, chip_id = args
    try:
        if chip_id.startswith("AF"):
            mask = _PRED["af"](load_af(root, chip_id, with_mask=False))
            return [(chip_id, 1, encode(mask == 1))]
        mask = _PRED["bs"](load_bs(root, chip_id, with_mask=False))
        return [(chip_id, k, encode(mask == k)) for k in (1, 2, 3)]
    except Exception as e:  # never fail the whole submission because of one chip
        print(f"WARNING: {chip_id} failed ({e!r}); writing empty mask", file=sys.stderr)
        return [(chip_id, k, "") for k in ((1,) if chip_id.startswith("AF") else (1, 2, 3))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--bs-model", choices=["unet", "rules"], default=os.environ.get("BS_MODEL", "unet"))
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()

    t0 = time.time()
    root = Path(a.data_dir)
    template = pd.read_csv(root / "sample_submission.csv", keep_default_na=False)
    chips = list(dict.fromkeys(template.chip_id))
    if a.bs_model == "unet" or a.workers <= 1:
        # torch parallelises internally; keep a single process
        _init(a.bs_model)
        rows = [r for c in chips for r in _predict((str(root), c))]
    else:
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.bs_model,)) as ex:
            rows = [r for res in ex.map(_predict, [(str(root), c) for c in chips], chunksize=4) for r in res]

    pred = pd.DataFrame(rows, columns=["chip_id", "class_id", "rle"])
    sub = template[["chip_id", "class_id"]].merge(pred, on=["chip_id", "class_id"], how="left")
    sub["rle"] = sub["rle"].fillna("")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    write_submission(sub, a.output)
    print(f"wrote {a.output}: {len(sub)} rows in {time.time() - t0:.1f}s")


def write_submission(sub: pd.DataFrame, path: str) -> None:
    """Spec: UTF-8, comma-separated, rle always in double quotes (no pixels -> \"\")."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("chip_id,class_id,rle\n")
        for c, k, r in zip(sub.chip_id, sub.class_id, sub.rle):
            f.write(f'{c},{int(k)},"{r}"\n')


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

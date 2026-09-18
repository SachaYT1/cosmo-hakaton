"""Check a submission.csv against every format rule in the case statement."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from firemon.rle import decode


def validate(sub_path: str, template_path: str, meta_path: str) -> list[str]:
    errors = []
    sub = pd.read_csv(sub_path, keep_default_na=False, dtype={"rle": str})
    tpl = pd.read_csv(template_path, keep_default_na=False)
    meta = pd.read_csv(meta_path).set_index("chip_id")
    if list(sub.columns) != ["chip_id", "class_id", "rle"]:
        errors.append(f"bad header {list(sub.columns)}")
    if len(sub) != len(tpl):
        errors.append(f"{len(sub)} rows, template has {len(tpl)}")
    a = set(zip(sub.chip_id, sub.class_id.astype(int)))
    b = set(zip(tpl.chip_id, tpl.class_id.astype(int)))
    if a != b:
        errors.append(f"(chip_id,class_id) mismatch: {len(a - b)} extra, {len(b - a)} missing")
    if sub.duplicated(["chip_id", "class_id"]).any():
        errors.append("duplicate (chip_id,class_id)")
    for chip_id, g in sub.groupby("chip_id"):
        shape = (int(meta.loc[chip_id, "height"]), int(meta.loc[chip_id, "width"]))
        used = np.zeros(shape, bool)
        for rle in g.rle:
            try:
                m = decode(rle, shape)
            except ValueError as e:
                errors.append(f"{chip_id}: {e}")
                continue
            if (used & m).any():
                errors.append(f"{chip_id}: overlapping classes")
            used |= m
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--data-dir", default="data/test")
    a = ap.parse_args()
    errs = validate(a.submission, f"{a.data_dir}/sample_submission.csv", f"{a.data_dir}/meta.csv")
    print("VALID" if not errs else "\n".join(errs[:50]))
    sys.exit(1 if errs else 0)

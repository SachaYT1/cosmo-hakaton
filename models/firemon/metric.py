"""Competition metric, re-implemented from the case statement (micro-averaged over all pixels):

    Score = 0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev

A zero denominator (class absent in both truth and prediction) counts as 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _ratio(num: float, den: float) -> float:
    return 1.0 if den == 0 else num / den


@dataclass
class Accumulator:
    af: np.ndarray = field(default_factory=lambda: np.zeros(3, np.int64))        # tp fp fn
    burn: np.ndarray = field(default_factory=lambda: np.zeros(3, np.int64))
    sev: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), np.int64))  # per class 1..3

    @staticmethod
    def _counts(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        pred, true = pred.astype(bool), true.astype(bool)
        return np.array([(pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()])

    def add_af(self, pred: np.ndarray, true: np.ndarray) -> None:
        self.af += self._counts(pred > 0, true > 0)

    def add_bs(self, pred: np.ndarray, true: np.ndarray) -> None:
        self.burn += self._counts(pred > 0, true > 0)
        for k in (1, 2, 3):
            self.sev[k - 1] += self._counts(pred == k, true == k)

    def result(self) -> dict[str, float]:
        tp, fp, fn = self.af
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        iou_burn = _ratio(self.burn[0], self.burn.sum())
        ious = [_ratio(c[0], c.sum()) for c in self.sev]
        miou = float(np.mean(ious))
        return {"F1_af": f1, "IoU_burn": iou_burn, "mIoU_sev": miou,
                "IoU_sev1": ious[0], "IoU_sev2": ious[1], "IoU_sev3": ious[2],
                "Score": 0.35 * f1 + 0.35 * iou_burn + 0.30 * miou}


def score_submission(sub_csv: str, gt_root: str) -> dict[str, float]:
    """Score a submission.csv against a folder with train-style masks (<root>/{af,bs}/masks)."""
    import pandas as pd
    from pathlib import Path
    from .rle import decode
    import tifffile

    sub = pd.read_csv(sub_csv, keep_default_na=False)
    acc = Accumulator()
    for chip_id, g in sub.groupby("chip_id"):
        kind = "af" if chip_id.startswith("AF") else "bs"
        true = tifffile.imread(str(Path(gt_root) / kind / "masks" / f"{chip_id}_MASK.tif"))
        pred = np.zeros(true.shape, np.uint8)
        for cls, rle in zip(g.class_id, g.rle):
            pred[decode(rle, true.shape)] = int(cls)
        (acc.add_af if kind == "af" else acc.add_bs)(pred, true)
    return acc.result()

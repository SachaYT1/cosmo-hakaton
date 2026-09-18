"""Chip-level predictors shared by inference.py, the evaluation script and the service."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import af as afm
from .bs_rules import Thresholds, invalid_mask, rule_predict, smooth_dnbr
from .features import nbr
from .io import AFChip, BSChip

ROOT = Path(__file__).resolve().parent.parent


class AFPredictor:
    def __init__(self, weights: Path = ROOT / "weights" / "af_xgb.json",
                 config: Path = ROOT / "configs" / "af.json"):
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(weights))
        self.model.set_params(n_jobs=1)
        self.threshold = json.load(open(config))["threshold"]

    def __call__(self, chip: AFChip) -> np.ndarray:
        f = afm.features(chip)
        cand = afm.candidates(f)
        out = np.zeros(f.shape[:2], np.uint8)
        if cand.any():
            p = self.model.predict_proba(f[cand])[:, 1]
            out[cand] = p > self.threshold
        return out


class BSRulePredictor:
    def __init__(self, thresholds: Path = ROOT / "configs" / "bs_thresholds.json"):
        self.th = Thresholds.load(thresholds)
        post = Path(str(thresholds).replace(".json", "_post.json"))
        self.min_size = json.load(open(post))["min_size"] if post.exists() else 0

    def __call__(self, chip: BSChip) -> np.ndarray:
        med = smooth_dnbr(nbr(chip.s2_pre) - nbr(chip.s2_post))
        inv = invalid_mask(chip.s2_pre[..., 9], chip.s2_post[..., 9])
        return rule_predict(med, chip.aux[..., 2], inv, self.th, self.min_size)

"""Chip-level predictors shared by inference.py, the evaluation script and the service."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import af as afm
from . import af_models
from .bs_rules import Thresholds, invalid_mask, rule_predict, smooth_dnbr
from .features import nbr
from .io import AFChip, BSChip

ROOT = Path(__file__).resolve().parent.parent


class AFPredictor:
    def __init__(self, weights: Path | None = None,
                 config: Path = ROOT / "configs" / "af.json"):
        with open(config) as f:
            cfg = json.load(f)
        self.threshold = float(cfg["threshold"])
        self.inclusive = cfg.get("threshold_rule", ">") == ">="
        members = cfg.get("models", [{"file": "weights/af_xgb.json", "feature_set": "legacy", "weight": 1.0}])
        if weights is not None:
            if len(members) != 1:
                raise ValueError("A single weights override cannot replace an ensemble")
            members = [{**members[0], "file": str(Path(weights).resolve())}]
        self.models = []
        for member in members:
            path = Path(member["file"])
            if not path.is_absolute():
                path = ROOT / path
            model = af_models.load(member.get("kind", "xgb"), path)
            feature_set = member.get("feature_set", "legacy")
            if af_models.feature_count(model) != len(afm.feature_names(feature_set)):
                raise ValueError(f"AF model {path} does not match feature set {feature_set}")
            if "feature_names" in member and member["feature_names"] != afm.feature_names(feature_set):
                raise ValueError(f"AF model {path} feature order has changed")
            self.models.append((model, feature_set, float(member.get("weight", 1.0))))
        self.total_weight = sum(w for _, _, w in self.models)
        if not self.models or self.total_weight <= 0 or any(w < 0 for _, _, w in self.models):
            raise ValueError("AF ensemble needs nonnegative weights with a positive sum")
        self.model = self.models[0][0]

    def __call__(self, chip: AFChip) -> np.ndarray:
        feature_set = "context" if any(s == "context" for _, s, _ in self.models) else "legacy"
        f = afm.extract_features(chip, feature_set)
        cand = afm.candidates(f)
        out = np.zeros(f.shape[:2], np.uint8)
        if cand.any():
            x = f[cand]
            p = np.zeros(len(x), np.float32)
            for model, feature_set, weight in self.models:
                p += (weight / self.total_weight) * af_models.probability(
                    model, x[:, :len(afm.feature_names(feature_set))])
            out[cand] = p >= self.threshold if self.inclusive else p > self.threshold
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

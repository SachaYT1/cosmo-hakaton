"""Contract checks for the AF metric, features and deployable predictor."""
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xgboost as xgb

from firemon.af import candidates, extract_features, feature_names
from firemon.af_eval import counts, date_folds, metrics, tune_threshold
from firemon.io import AFChip
from firemon.metric import Accumulator
from firemon.pipeline import AFPredictor


def chip():
    rng = np.random.default_rng(7)
    v = np.zeros((16, 16, 8), np.float32)
    v[..., :3] = rng.uniform(0, 0.4, (16, 16, 3))
    v[..., 3:5] = rng.normal(300, 2, (16, 16, 2))
    v[7, 8, 3] = 345
    v[..., 5] = 40
    v[..., 6] = 20
    v[..., 7] = 1
    aux = np.zeros((16, 16, 5), np.float32)
    aux[..., 0] = 30
    aux[..., 2] = 290
    return AFChip("synthetic", v, aux, None)


class AFTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "Optional LightGBM dependency")
    def test_lightgbm_artifact_predictor(self):
        from lightgbm import LGBMClassifier
        c = chip(); f = extract_features(c, "context"); cand = candidates(f)
        rng = np.random.default_rng(23)
        X = rng.normal(size=(100, f.shape[-1])).astype(np.float32)
        model = LGBMClassifier(n_estimators=3, min_child_samples=5, n_jobs=1,
                              verbosity=-1).fit(X, X[:, 0] > 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.txt"; model.booster_.save_model(str(path))
            cfg = Path(tmp)/"af.json"
            cfg.write_text(json.dumps({"threshold": 0.5, "threshold_rule": ">=", "models": [
                {"file": str(path), "kind": "lgb", "feature_set": "context"}]}))
            pred = AFPredictor(config=cfg)(c)
            expected = model.booster_.predict(f[cand], num_threads=1).astype(np.float32) >= 0.5
            np.testing.assert_array_equal(pred[cand], expected)
            self.assertFalse(pred[~cand].any())

    def test_candidate_counts_equal_full_raster_metric(self):
        # Includes a positive excluded by the filter and a background-only chip.
        truth = np.array([[1, 0, 1], [0, 0, 0]], np.uint8)
        candidate = np.array([[True, True, False], [True, False, False]])
        prob = np.array([0.5, 0.6, 0.1])
        pred = np.zeros_like(truth)
        pred[candidate] = prob >= 0.5
        acc = Accumulator(); acc.add_af(pred, truth)
        c = counts(prob, truth[candidate], 0.5, missed=1)
        np.testing.assert_array_equal(c, acc.af)
        self.assertEqual(metrics(c)["F1_af"], acc.result()["F1_af"])
        self.assertEqual(metrics(np.zeros(3))["F1_af"], 1)
        self.assertEqual(metrics(np.array([0, 0, 1]))["F1_af"], 0)

    def test_acquisition_groups_never_cross_folds(self):
        dates = np.array(["a", "b", "a", "c", "d", "b", "e", "f"])
        folds = date_folds(dates, 3)
        np.testing.assert_array_equal(folds, date_folds(dates, 3))
        for d in np.unique(dates):
            self.assertEqual(len(np.unique(folds[dates == d])), 1)

    def test_threshold_and_missing_candidates(self):
        t, r = tune_threshold(np.array([0.9, 0.8]), np.array([True, False]),
                              missed=1, grid=np.array([0.5, 0.85]))
        self.assertEqual(t, 0.85)
        self.assertEqual((r["TP"], r["FP"], r["FN"]), (1, 0, 1))

    def test_schema_and_night_nodata(self):
        c = chip()
        base, context = extract_features(c), extract_features(c, "context")
        np.testing.assert_allclose(base, context[..., :len(feature_names())])
        self.assertEqual(context.shape[-1], len(feature_names("context")))
        self.assertEqual(len(set(feature_names("context"))), context.shape[-1])
        c.viirs[..., :3] = np.nan
        c.viirs[:3, :, 3:5] = np.nan
        c.viirs[:3, :, 7] = 0
        context = extract_features(c, "context")
        self.assertFalse(np.isinf(context).any())
        self.assertEqual(candidates(context).shape, (16, 16))

    def test_saved_ensemble_equals_direct_prediction(self):
        c = chip(); f = extract_features(c, "context"); cand = candidates(f)
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            members = []; direct = np.zeros(cand.sum(), np.float32)
            for i, feature_set in enumerate(("legacy", "context")):
                nf = len(feature_names(feature_set))
                X = rng.normal(size=(64, nf)).astype(np.float32)
                y = X[:, 0] > 0
                model = xgb.XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1).fit(X, y)
                path = Path(tmp)/f"model{i}.json"; model.save_model(path)
                members.append({"file": str(path), "feature_set": feature_set, "weight": i+1})
                direct += (i+1)/3 * model.predict_proba(f[cand, :nf])[:, 1]
            cfg = Path(tmp)/"af.json"
            cfg.write_text(json.dumps({"threshold": 0.5, "threshold_rule": ">=", "models": members}))
            pred = AFPredictor(config=cfg)(c)
            np.testing.assert_array_equal(pred[cand], direct >= 0.5)
            self.assertFalse(pred[~cand].any())
            self.assertEqual(pred.dtype, np.uint8)
            # The old one-model JSON and explicit weights argument still work.
            cfg.write_text(json.dumps({"threshold": 0.5}))
            old = AFPredictor(weights=Path(members[0]["file"]), config=cfg)
            old_expected = old.model.predict_proba(f[cand, :len(feature_names())])[:, 1] > 0.5
            np.testing.assert_array_equal(old(c)[cand], old_expected)


if __name__ == "__main__":
    unittest.main()

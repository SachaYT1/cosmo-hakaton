"""Small CPU model adapter; LightGBM is imported only for LightGBM artifacts."""
from __future__ import annotations

import numpy as np


def create(kind: str, params: dict):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if kind == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown AF model kind: {kind}")


def load(kind: str, path):
    if kind == "xgb":
        model = create(kind, {"n_jobs": 1})
        model.load_model(str(path))
        model.set_params(n_jobs=1)
        return model
    if kind == "lgb":
        from lightgbm import Booster
        return Booster(model_file=str(path))
    raise ValueError(f"Unknown AF model kind: {kind}")


def feature_count(model) -> int:
    return model.n_features_in_ if hasattr(model, "n_features_in_") else model.num_feature()


def probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X, num_threads=1)


def save(model, kind: str, path) -> None:
    if kind == "xgb":
        model.save_model(path)
    else:
        model.booster_.save_model(str(path))


def importance(model, kind: str) -> np.ndarray:
    if kind == "xgb":
        return model.feature_importances_
    gain = model.booster_.feature_importance(importance_type="gain")
    return gain/max(float(gain.sum()), 1.0)

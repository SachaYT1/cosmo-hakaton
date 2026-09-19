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
        from xgboost import Booster
        model = Booster(params={"nthread": 1})
        model.load_model(str(path))
        return model
    if kind == "lgb":
        from lightgbm import Booster
        return Booster(model_file=str(path))
    raise ValueError(f"Unknown AF model kind: {kind}")


def feature_count(model) -> int:
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    if hasattr(model, "num_feature"):
        return model.num_feature()
    return model.num_features()


def probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "inplace_predict"):
        return model.inplace_predict(X)
    return model.predict(X, num_threads=1)


def save(model, kind: str, path) -> None:
    if kind == "xgb":
        # XGBoost 2.1 combined with recent scikit-learn cannot always infer
        # `_estimator_type` in XGBClassifier.save_model. The native Booster
        # artifact is the same inference model and works across both 2.x/3.x.
        model.get_booster().save_model(str(path))
    else:
        model.booster_.save_model(str(path))


def importance(model, kind: str) -> np.ndarray:
    if kind == "xgb":
        return model.feature_importances_
    gain = model.booster_.feature_importance(importance_type="gain")
    return gain/max(float(gain.sum()), 1.0)

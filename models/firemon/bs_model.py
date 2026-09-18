"""Learned burn delineation using spectral change, SAR and spatial context.

Only raster inputs are used: no chip IDs, dates, coordinates or target-derived
metadata. Severity is decoded separately using training-fitted dNBR thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb

from .bs_rules import Thresholds, invalid_mask, remove_small, smooth_dnbr
from .features import nbr, ndvi, nbr2, ndwi
from .io import BSChip

ROOT = Path(__file__).resolve().parent.parent


def extract_features(chip: BSChip) -> tuple[np.ndarray, list[str]]:
    """Return H,W,C float32 features. Finite inputs and arbitrary image sizes supported."""
    values, names = [], []

    def add(name, a):
        names.append(name)
        values.append(np.nan_to_num(a, nan=0, posinf=0, neginf=0).astype(np.float32))

    pre = chip.s2_pre[..., :9].astype(np.float32) / 10000
    post = chip.s2_post[..., :9].astype(np.float32) / 10000
    bands = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12')
    for j, b in enumerate(bands):
        for label, a in [('pre', pre), ('post', post), ('delta', pre-post)]:
            add(f'{label}_{b}', a[..., j])
    context = {}
    for name, fn in [('nbr', nbr), ('ndvi', ndvi), ('nbr2', nbr2), ('ndwi', ndwi)]:
        a, b = fn(chip.s2_pre), fn(chip.s2_post)
        add(f'{name}_pre', a)
        add(f'{name}_post', b)
        add(f'd_{name}', a-b)
        context[f'd_{name}'] = a-b
    dnbr = context['d_nbr']
    add('rdnbr', dnbr / np.sqrt(np.maximum(np.abs(nbr(chip.s2_pre)), .02)))
    for size in (3, 5):
        add(f'dnbr_median{size}', cv2.medianBlur(dnbr, size))
    for size in (5, 15, 41):
        for name, a in context.items():
            mean = cv2.blur(a, (size, size))
            add(f'{name}_mean{size}', mean)
            if name == 'd_nbr':
                add(f'{name}_std{size}', np.sqrt(np.maximum(cv2.blur(a*a, (size,size))-mean*mean, 0)))
        for j in (2, 6, 8):
            add(f'd_{bands[j]}_mean{size}', cv2.blur(pre[..., j]-post[..., j], (size,size)))
    for j, name in enumerate(('VV','VH')):
        a, b = chip.s1_pre[..., j].astype(np.float32)/100, chip.s1_post[..., j].astype(np.float32)/100
        for label, v in [('pre', a), ('post', b), ('delta', b-a)]:
            add(f'{name}_{label}', v)
            for size in (5, 15):
                add(f'{name}_{label}_mean{size}', cv2.blur(v, (size,size)))
    for label, a in [('pre', chip.s1_pre), ('post', chip.s1_post)]:
        add(f'sar_ratio_{label}', (a[..., 0].astype(np.float32)-a[..., 1])/100)
    for j, name in enumerate(('dem','slope','landcover')):
        add(name, chip.aux[..., j])
    for label, a in [('pre',chip.s2_pre),('post',chip.s2_post)]:
        add(f'scl_{label}', a[..., 9])
        add(f'cloud_fraction_{label}', cv2.blur(np.isin(a[..., 9], (0,3,8,9,10)).astype(np.float32), (15,15)))
    return np.stack(values, axis=-1), names


def decode(prob: np.ndarray, chip: BSChip, thresholds: Thresholds,
           cutoff: float = .5, min_size: int = 0, smooth: int = 0) -> np.ndarray:
    if smooth:
        prob = cv2.blur(prob, (smooth, smooth))
    burn = prob >= cutoff
    burn &= ~invalid_mask(chip.s2_pre[..., 9], chip.s2_post[..., 9])
    if min_size:
        burn = remove_small(burn, min_size)
    severity = thresholds.severity(smooth_dnbr(nbr(chip.s2_pre)-nbr(chip.s2_post)), chip.aux[..., 2])
    return np.where(burn, severity, 0).astype(np.uint8)


class BSPredictor:
    def __init__(self, config: str | Path = ROOT / 'configs/bs_model.json', threads: int = 1):
        config = Path(config)
        self.cfg = json.loads(config.read_text())
        self.thresholds = Thresholds(self.cfg['severity_thresholds'])
        self.models = []
        for entry in self.cfg['models']:
            model = xgb.XGBClassifier(n_jobs=threads)
            path = Path(entry)
            model.load_model(str(path if path.is_absolute() else config.parent.parent/path))
            model.set_params(n_jobs=threads)
            self.models.append(model)
        if not self.models:
            raise ValueError('BS configuration has no models')

    def probability(self, chip: BSChip) -> np.ndarray:
        features, names = extract_features(chip)
        if names != self.cfg['feature_names']:
            raise ValueError('BS feature schema differs from trained model')
        h, w, c = features.shape
        x = features.reshape(-1, c)
        p = np.zeros(h*w, np.float32)
        for m in self.models:
            # Bound temporary prediction memory for production workers.
            for start in range(0, len(x), 65536):
                p[start:start+65536] += m.predict_proba(x[start:start+65536])[:, 1]/len(self.models)
        return p.reshape(h,w)

    def __call__(self, chip: BSChip) -> np.ndarray:
        return decode(self.probability(chip), chip, self.thresholds,
                      self.cfg['cutoff'], self.cfg.get('min_size', 0), self.cfg.get('smooth', 0))

"""Fit the already selected AF trial once on all competition training chips."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from firemon import af_models
from firemon.af import feature_names
from scripts.train_af import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trial', default='lgb_context31')
    ap.add_argument('--cache-dir', type=Path, default=ROOT/'cache/af_experiments')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    trial = json.loads((a.cache_dir/(a.trial+'.json')).read_text())
    kind = trial.get('kind', 'xgb')
    feature_set = 'context' if trial['n_features'] == 100 else 'legacy'
    member = {'kind': kind, 'feature_set': feature_set, 'params': trial['params'], 'weight': 1.0}
    params = {**trial['params'], 'n_jobs': a.threads}
    model = af_models.create(kind, params)
    X = np.load(a.cache_dir/'X.npy', mmap_mode='r')[:, :trial['n_features']]
    z = np.load(a.cache_dir/'labels.npz')
    tic = time.monotonic()
    print('Fitting frozen baseline:', a.trial, X.shape, flush=True)
    model.fit(X, z['y'])
    suffix = 'txt' if kind == 'lgb' else 'json'
    filename = f'af_baseline_{a.trial}.{suffix}'
    path = ROOT/'weights'/filename
    af_models.save(model, kind, path)
    cfg = {'threshold': trial['best']['threshold'], 'threshold_rule': '>=',
           'cv_f1': trial['best']['f1'],
           'cv_protocol': '3 acquisition-date folds on 2019-2023; threshold calibrated on OOF; 2024 excluded from selection',
           'baseline_trial': a.trial,
           'models': [{'file': f'weights/{filename}', 'kind': kind, 'feature_set': feature_set,
                       'feature_names': feature_names(feature_set), 'weight': 1.0}]}
    old = ROOT/'configs/af_original.json'
    if not old.exists(): shutil.copy2(ROOT/'configs/af.json', old)
    (ROOT/'configs/af_train.json').write_text(json.dumps({'models': [member], 'selected_trial': a.trial}, indent=2)+'\n')
    for name in ('af.json', 'af_baseline.json'):
        (ROOT/'configs'/name).write_text(json.dumps(cfg, indent=2)+'\n')
    report = {'selected': trial, 'training_samples': len(z['y']), 'full_training_chips': 420,
              'seconds_full_fit': round(time.monotonic()-tic, 1),
              'weights': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'external_training_used': False,
              'note': 'Selected by the common pre-2024 development protocol. This is not a private score.'}
    (ROOT/'reports/af_baseline.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()

"""Reproducible BS feature cache, held-event evaluation and full-data refit."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from firemon.bs_model import extract_features, decode
from firemon.bs_rules import Thresholds, smooth_dnbr
from firemon.features import nbr
from firemon.io import load_bs, list_chips
from firemon.metric import Accumulator
from scripts.fit_bs_rules import fit_severity

# Published in Figure 6 of the case statement, available before seeing test data.
PUBLISHED_THRESHOLDS = {'default':[.1,.27,.66], '10':[.1,.27,.66], '20':[.1,.27,.66],
                        '30':[.062,.204,.386], '40':[.07,.177,.38], '90':[.075,.331,.677]}


def fine_severity(x,y):
    grid=np.arange(0,1.001,.001)
    bins=np.clip(np.searchsorted(grid,x,side='right')-1,0,len(grid)-1)
    counts=np.stack([np.bincount(bins[y==k],minlength=len(grid)) for k in (1,2,3)])
    c=np.pad(counts.cumsum(1),((0,0),(1,0)))[:,:len(grid)]
    left=c[0]-c[1];right=c[1]-c[2]
    prefix=np.maximum.accumulate(left)
    scores=prefix[:-1]+right[1:]
    j=int(np.argmax(scores))+1;i=int(np.argmax(left[:j]))
    return float(grid[i]),float(grid[j])


def prepare(root, out, pixels, seed):
    ids = list_chips(root, 'bs')
    rng = np.random.default_rng(seed)
    xs, ys, groups = [], [], []
    severity_x, severity_y, severity_lc, severity_group = [], [], [], []
    t0 = time.time()
    for i, cid in enumerate(ids):
        chip = load_bs(root, cid)
        if chip.mask is None:
            raise ValueError(f'Missing training mask: {cid}')
        f, names = extract_features(chip)
        ix = rng.choice(chip.mask.size, min(pixels, chip.mask.size), replace=False)
        xs.append(f.reshape(-1, f.shape[-1])[ix])
        ys.append(chip.mask.ravel()[ix])
        groups.append(np.full(len(ix), i, np.int16))
        burned = np.flatnonzero(chip.mask.ravel() > 0)
        ix = rng.choice(burned, min(4096,len(burned)), replace=False)
        severity_x.append(smooth_dnbr(nbr(chip.s2_pre)-nbr(chip.s2_post)).ravel()[ix])
        severity_y.append(chip.mask.ravel()[ix])
        severity_lc.append(chip.aux[...,2].ravel()[ix])
        severity_group.append(np.full(len(ix),i,np.int16))
        if (i+1)%20 == 0: print(f'features {i+1}/{len(ids)} {time.time()-t0:.0f}s',flush=True)
    np.savez(out/'samples.npz', x=np.concatenate(xs), y=np.concatenate(ys), group=np.concatenate(groups),
             sx=np.concatenate(severity_x), sy=np.concatenate(severity_y), slc=np.concatenate(severity_lc),
             sg=np.concatenate(severity_group), ids=np.array(ids), names=np.array(names))
    (out/'cache_manifest.json').write_text(json.dumps({'data_dir':str(Path(root).resolve()),'pixels':pixels,'seed':seed},indent=2))


def severity_fit(data, train_idx):
    selected = np.isin(data['sg'], train_idx)
    x,y,lc = data['sx'][selected],data['sy'][selected],data['slc'][selected]
    table = {k:list(v) for k,v in PUBLISHED_THRESHOLDS.items()}
    for code in np.unique(lc):
        sel=lc==code
        if sel.sum()>=1000:
            table[str(int(code))]=[.12,*fine_severity(x[sel],y[sel])]
    return Thresholds(table)


def bs_result(acc):
    r=acc.result()
    return {**{k:v for k,v in r.items() if k not in ('Score','F1_af')},
            'BS_contribution':.35*r['IoU_burn']+.30*r['mIoU_sev'],
            'burn_counts':acc.burn.tolist(),'severity_counts':acc.sev.tolist()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--cache',default='cache/bs_experiments')
    ap.add_argument('--pixels',type=int,default=4096)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--fold',type=int,default=0,help='-1: all folds, -2: final full-data refit')
    ap.add_argument('--trees',type=int,default=350)
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--threads',type=int,default=6)
    ap.add_argument('--tag',default='d6')
    ap.add_argument('--split',choices=['events','time'],default='events')
    ap.add_argument('--evaluate-only',action='store_true')
    a=ap.parse_args()
    out=Path(a.cache);out.mkdir(parents=True,exist_ok=True)
    if not (out/'samples.npz').exists(): prepare(a.data_dir,out,a.pixels,a.seed)
    manifest=json.loads((out/'cache_manifest.json').read_text())
    if manifest != {'data_dir':str(Path(a.data_dir).resolve()),'pixels':a.pixels,'seed':a.seed}:
        raise ValueError('Cache belongs to different data/sampling settings; choose a new --cache')
    data=np.load(out/'samples.npz')
    ids=data['ids'];x=data['x'];y=data['y']>0;group=data['group']
    meta=pd.read_csv(Path(a.data_dir)/'bs/meta.csv').set_index('chip_id').loc[ids]
    # Group by fire event, even if a future release has multiple chips per event.
    groups=meta.fire_event_id.fillna(pd.Series(ids,index=meta.index)).astype(str).to_numpy()
    split=list(GroupKFold(a.folds,shuffle=True,random_state=a.seed).split(ids,groups=groups))
    if a.split=='time':
        years=meta.date_pre.str[:4].astype(int).to_numpy()
        split=[(np.flatnonzero(years<years.max()),np.flatnonzero(years==years.max()))]
    fold_of=np.full(len(ids),-1)
    for f,(_,va) in enumerate(split):fold_of[va]=f
    (out/f'folds_{a.split}.json').write_text(json.dumps(dict(zip(ids.tolist(),fold_of.tolist())),indent=2))
    folds=range(len(split)) if a.fold==-1 else [a.fold]
    params=dict(n_estimators=a.trees,max_depth=a.depth,learning_rate=.07,min_child_weight=20,
                subsample=.85,colsample_bytree=.9,reg_lambda=10,tree_method='hist',max_bin=128,
                n_jobs=a.threads,random_state=a.seed,objective='binary:logistic',eval_metric='logloss')
    for f in folds:
        t0=time.time()
        tr,va=split[f] if f>=0 else (np.arange(len(ids)),np.array([],int))
        model=xgb.XGBClassifier(**params)
        sel=np.isin(group,tr)
        weights=Path('weights')/f'bs_{a.tag}_f{f}.ubj';weights.parent.mkdir(exist_ok=True)
        if a.evaluate_only:
            model.load_model(weights);model.set_params(n_jobs=a.threads)
        else:
            model.fit(x[sel],y[sel])
            model.save_model(weights)
        thresholds=severity_fit(data,tr)
        print(f'trained fold {f} in {time.time()-t0:.1f}s',flush=True)
        cfg={'models':[str(weights)],'feature_names':data['names'].tolist(),
             'severity_thresholds':thresholds.table,'cutoff':.5,'min_size':0,'smooth':0,
             'params':params,'seed':a.seed}
        if f==-2:
            (Path('configs')/'bs_model.json').write_text(json.dumps(cfg,indent=2));continue
        pred_dir=out/f'{a.tag}_f{f}';pred_dir.mkdir(exist_ok=True)
        (pred_dir/'config.json').write_text(json.dumps(cfg,indent=2))
        accs={(round(float(t),2),smooth,ms):Accumulator() for t in np.arange(.1,.61,.05)
              for smooth in (0,3) for ms in (0,50)}
        from firemon.pipeline import BSRulePredictor
        baseline=Accumulator();rule=BSRulePredictor()
        for j,i in enumerate(va):
            c=load_bs(a.data_dir,str(ids[i]))
            cached=pred_dir/f'{ids[i]}.npy'
            if a.evaluate_only and cached.exists():
                p=np.load(cached)
            else:
                features,_=extract_features(c)
                p=model.predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(c.mask.shape)
                np.save(cached,p)
            baseline.add_bs(rule(c),c.mask)
            for (t,sm,ms),acc in accs.items():acc.add_bs(decode(p,c,thresholds,t,ms,sm),c.mask)
            if (j+1)%10==0:print(f'eval {j+1}/{len(va)} {time.time()-t0:.0f}s',flush=True)
        results=[{'cutoff':t,'smooth':sm,'min_size':ms,**bs_result(acc)} for (t,sm,ms),acc in accs.items()]
        results.sort(key=lambda r:r['BS_contribution'],reverse=True)
        report={'fold':f,'split':a.split,'train_chips':ids[tr].tolist(),'validation_chips':ids[va].tolist(),
                'baseline':bs_result(baseline),'ranking':results,'seconds':time.time()-t0}
        (out/f'{a.tag}_f{f}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'fold':f,'baseline':report['baseline'],'best':results[:3]},indent=2),flush=True)


if __name__=='__main__':main()

"""Train whole-scene U-Net on 128/256px BS chips; save full-resolution OOF maps."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from firemon.bs_unet import UNet, bs_input, N_CHANNELS
from firemon.io import list_chips, load_bs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--cache',default='cache/bs_spatial')
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--base',type=int,default=16)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tag',default='spatial128')
    p.add_argument('--init',default=None)
    a=p.parse_args();torch.set_num_threads(4)
    torch.manual_seed(a.seed);np.random.seed(a.seed);rng=np.random.default_rng(a.seed)
    out=Path(a.cache);out.mkdir(parents=True,exist_ok=True)
    ids=list_chips(a.data_dir,'bs');xp=out/f'x{a.size}.npy';yp=out/f'y{a.size}.npy'
    if not xp.exists():
        x=np.empty((len(ids),N_CHANNELS,a.size,a.size),np.float16)
        y=np.empty((len(ids),1,a.size,a.size),np.float32)
        for i,cid in enumerate(ids):
            c=load_bs(a.data_dir,cid);v=np.clip(bs_input(c),-5,5)
            x[i]=cv2.resize(v.transpose(1,2,0),(a.size,a.size),interpolation=cv2.INTER_AREA).transpose(2,0,1)
            y[i,0]=cv2.resize((c.mask>0).astype(np.float32),(a.size,a.size),interpolation=cv2.INTER_AREA)
            if (i+1)%40==0:print(f'prepared {i+1}/{len(ids)}',flush=True)
        np.save(xp,x);np.save(yp,y)
    x=np.load(xp);y=np.load(yp)
    fold_map=json.loads(Path('cache/bs_experiments/folds_events.json').read_text())
    va=np.array([i for i,cid in enumerate(ids) if fold_map[cid]==a.fold],int) if a.fold>=0 else np.array([],int)
    tr=np.array([i for i in range(len(ids)) if i not in va])
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model=UNet(cin=N_CHANNELS,cout=1,base=a.base,depth=3).to(device)
    if a.init:model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.00005)
    best=1e9;best_epoch=0;t0=time.time()
    path=Path('weights')/f'bs_{a.tag}_f{a.fold}.pt'
    print(f'{device}: {len(tr)} train, {len(va)} validation',flush=True)
    for epoch in range(a.epochs):
        model.train();losses=[]
        for start in range(0,len(tr),a.batch):
            if start==0:order=rng.permutation(tr)
            ix=order[start:start+a.batch]
            xb=x[ix].astype(np.float32);yb=y[ix].copy()
            for j in range(len(ix)):
                k=rng.integers(4);xb[j]=np.rot90(xb[j],k,(1,2));yb[j]=np.rot90(yb[j],k,(1,2))
                if rng.random()<.5:xb[j]=xb[j,:,::-1];yb[j]=yb[j,:,::-1]
            xb=torch.from_numpy(xb).to(device);yb=torch.from_numpy(yb).to(device)
            logits=model(xb);prob=logits.sigmoid()
            bce=F.binary_cross_entropy_with_logits(logits,yb)
            dice=1-(2*(prob*yb).sum()+1)/(prob.sum()+yb.sum()+1)
            loss=bce+dice
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            losses.append(loss.item())
        sched.step()
        if (epoch+1)%5==0 or epoch==a.epochs-1:
            model.eval();vl=[]
            with torch.no_grad():
                for start in range(0,len(va),a.batch):
                    ix=va[start:start+a.batch];xb=torch.from_numpy(x[ix].astype(np.float32)).to(device)
                    yb=torch.from_numpy(y[ix]).to(device);logits=model(xb);prob=logits.sigmoid()
                    vl.append((F.binary_cross_entropy_with_logits(logits,yb)+1-(2*(prob*yb).sum()+1)/(prob.sum()+yb.sum()+1)).item())
            value=float(np.mean(vl)) if vl else float(np.mean(losses))
            if value<best or not len(va):
                best=value;best_epoch=epoch+1;torch.save({k:v.cpu() for k,v in model.state_dict().items()},path)
            print(f'epoch {epoch+1}: train={np.mean(losses):.4f} val={value:.4f} best={best:.4f} {time.time()-t0:.0f}s',flush=True)
    cfg={'size':a.size,'base':a.base,'depth':3,'epochs':a.epochs,'best_epoch':best_epoch,'fold':a.fold,
         'seed':a.seed,'device':str(device),'seconds':time.time()-t0,'weight':str(path),'train_chips':[ids[i] for i in tr],
         'validation_chips':[ids[i] for i in va]}
    (out/f'{a.tag}_f{a.fold}.json').write_text(json.dumps(cfg,indent=2))
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval()
    pred_dir=out/f'{a.tag}_f{a.fold}';pred_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for i in va:
            xb=torch.from_numpy(x[i:i+1].astype(np.float32)).to(device)
            # Geometric TTA; probabilities are averaged, then resized to native grid.
            prob=(model(xb).sigmoid()+model(xb.flip(-1)).sigmoid().flip(-1)+model(xb.flip(-2)).sigmoid().flip(-2))/3
            pp=cv2.resize(prob[0,0].cpu().numpy(),(512,512),interpolation=cv2.INTER_LINEAR)
            np.save(pred_dir/f'{ids[i]}.npy',pp)


if __name__=='__main__':main()

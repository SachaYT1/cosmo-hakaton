"""Train the BS U-Net.

    python -m scripts.train_bs_unet --fold 0        # one CV fold, saves OOF predictions
    python -m scripts.train_bs_unet --fold -1       # all chips (final model)

Folds are the same chip-level 5-fold split as scripts/fit_bs_rules.py (seed 42); every chip is a
separate fire event, so this is a grouped split. Needs cache/ from scripts/prepare_bs.py.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from firemon.bs_unet import UNet

SEED = 42
CROP = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    """Each loader worker gets its own deterministic crop stream (otherwise all copies repeat)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def folds(n: int, k: int = 5) -> np.ndarray:
    return np.random.default_rng(SEED).permutation(n) % k


class Crops(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, idx: np.ndarray, n: int, seed: int):
        self.x, self.y, self.idx, self.n = x, y, idx, n
        self.rng = np.random.default_rng(seed)
        self.burn_px = {i: np.argwhere(y[i] > 0) for i in idx}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _):
        i = self.rng.choice(self.idx)
        bp = self.burn_px[i]
        if len(bp) and self.rng.random() < 0.6:  # oversample crops that contain burn
            cy, cx = bp[self.rng.integers(len(bp))]
            y0 = int(np.clip(cy - self.rng.integers(CROP), 0, 512 - CROP))
            x0 = int(np.clip(cx - self.rng.integers(CROP), 0, 512 - CROP))
        else:
            y0, x0 = self.rng.integers(0, 512 - CROP + 1, 2)
        xb = np.asarray(self.x[i, :, y0:y0 + CROP, x0:x0 + CROP], np.float32)
        yb = self.y[i, y0:y0 + CROP, x0:x0 + CROP].astype(np.int64)
        k = self.rng.integers(4)
        xb, yb = np.rot90(xb, k, (1, 2)), np.rot90(yb, k)
        if self.rng.random() < 0.5:
            xb, yb = xb[:, :, ::-1], yb[:, ::-1]
        return torch.from_numpy(xb.copy()), torch.from_numpy(yb.copy())


def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    burn = (y > 0).float()
    p = torch.sigmoid(out[:, 0])
    bce = F.binary_cross_entropy_with_logits(out[:, 0], burn)
    dice = 1 - (2 * (p * burn).sum() + 1) / (p.sum() + burn.sum() + 1)
    sev_t = (y - 1).clamp(min=0)
    ce = F.cross_entropy(out[:, 1:], sev_t, reduction="none")
    sel = (y > 0).float()
    sev = (ce * sel).sum() / sel.sum().clamp(min=1)
    return bce + dice + 0.5 * sev


@torch.no_grad()
def predict_full(model: torch.nn.Module, x: np.ndarray, device) -> np.ndarray:
    xb = torch.from_numpy(np.asarray(x, np.float32))[None].to(device)
    out = (model(xb) + model(xb.flip(-1)).flip(-1)) / 2
    return out[0].float().cpu().numpy()


def main(fold: int, epochs: int, iters: int, base: int, bs: int, lr: float, tag: str, workers: int) -> None:
    set_seed(SEED + max(fold, 0))
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    x = np.load("cache/bs_x.npy", mmap_mode="r")
    y = np.load("cache/bs_y.npy")
    f = folds(len(y))
    tr = np.flatnonzero(f != fold) if fold >= 0 else np.arange(len(y))
    va = np.flatnonzero(f == fold) if fold >= 0 else np.array([], int)

    model = UNet(base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * iters, pct_start=0.1)
    dl = torch.utils.data.DataLoader(Crops(x, y, tr, epochs * iters * bs, SEED + fold), batch_size=bs,
                                     num_workers=workers, worker_init_fn=_seed_worker,
                                     persistent_workers=workers > 0)
    model.train()
    t0, run = time.time(), 0.0
    for step, (xb, yb) in enumerate(dl):
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if (step + 1) % iters == 0:
            print(f"fold {fold} epoch {(step + 1) // iters}/{epochs} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)

    Path("weights").mkdir(exist_ok=True)
    name = f"bs_unet_{tag}_f{fold}.pt" if fold >= 0 else f"bs_unet_{tag}_full.pt"
    torch.save(model.state_dict(), Path("weights") / name)
    if len(va):
        model.eval()
        Path("cache/oof").mkdir(parents=True, exist_ok=True)
        for i in va:
            out = predict_full(model, x[i], device)
            np.save(f"cache/oof/{tag}_{i:03d}.npy", out.astype(np.float16))
    json.dump({"fold": fold, "epochs": epochs, "iters": iters, "base": base, "bs": bs, "lr": lr,
               "seed": SEED, "seconds": round(time.time() - t0)},
              open(Path("weights") / name.replace(".pt", ".json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag", default="b16")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    main(a.fold, a.epochs, a.iters, a.base, a.bs, a.lr, a.tag, a.workers)

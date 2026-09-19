"""U-Net for burn delineation + severity (BS module).

Head: 4 logits per pixel = [burn, sev1, sev2, sev3]. Burn is trained with BCE+Dice on all valid
pixels; severity with cross-entropy on truly burned pixels only. At inference the burn outline
comes from the network; severity comes either from the network head or from the dNBR rule
(configs/bs_unet.json: "severity": "net" | "rule"), whichever won in cross-validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bs_rules import Thresholds, invalid_mask, remove_small, smooth_dnbr
from .features import nbr, ndvi
from .io import BSChip

ROOT = Path(__file__).resolve().parent.parent
LC_CODES = (10, 20, 30, 40, 50, 60, 80, 90)
N_CHANNELS = 18 + 5 + 6 + 2 + len(LC_CODES) + 3

# Channel layout of bs_input(). The final model deliberately excludes Sentinel-1:
# the five-fold ablation showed a small but statistically significant degradation.
CHANNEL_GROUPS = {
    "s2": list(range(0, 23)) + [39, 40, 41],
    "s1": list(range(23, 29)),
    "aux": list(range(29, 39)),
}


def channel_index(groups: list[str]) -> list[int]:
    return sorted(i for group in groups for i in CHANNEL_GROUPS[group])


def bs_input(chip: BSChip) -> np.ndarray:
    """(C, H, W) float32 network input; every channel roughly in [-3, 3]."""
    pre, post = chip.s2_pre, chip.s2_post
    refl = lambda s: np.log1p(s[..., :9].astype(np.float32) / 1000.0).transpose(2, 0, 1)  # noqa: E731
    n_pre, n_post = nbr(pre), nbr(post)
    dnbr = n_pre - n_post
    idx = np.stack([n_pre, n_post, dnbr * 2, smooth_dnbr(dnbr) * 2, (ndvi(pre) - ndvi(post)) * 2])
    s1 = lambda s: (s.astype(np.float32) / 100.0 + 15.0) / 6.0  # dB*100 -> ~N(0,1)  # noqa: E731
    vv0, vh0 = s1(chip.s1_pre[..., 0]), s1(chip.s1_pre[..., 1])
    vv1, vh1 = s1(chip.s1_post[..., 0]), s1(chip.s1_post[..., 1])
    sar = np.stack([vv0, vh0, vv1, vh1, vv1 - vv0, vh1 - vh0])
    sar = np.clip(np.nan_to_num(sar), -5, 5)
    dem = chip.aux[..., 0].astype(np.float32)
    topo = np.stack([(dem - dem.mean()) / 50.0, chip.aux[..., 1].astype(np.float32) / 10.0])
    lc = chip.aux[..., 2]
    onehot = np.stack([(lc == c) for c in LC_CODES]).astype(np.float32)
    scl0, scl1 = pre[..., 9], post[..., 9]
    flags = np.stack([invalid_mask(scl0, scl1), np.isin(scl1, (8, 10)), np.isin(scl0, (8, 10))]).astype(np.float32)
    x = np.concatenate([refl(pre), refl(post), idx, sar, topo, onehot, flags])
    return np.nan_to_num(x).astype(np.float32)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin: int = N_CHANNELS, cout: int = 4, base: int = 32, depth: int = 4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.stem = _block(cin, ch[0])
        self.down = nn.ModuleList(_block(ch[i], ch[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth)))
        self.dec = nn.ModuleList(_block(ch[i] * 2, ch[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(ch[0], cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up, dec in zip(self.up, self.dec):
            y = dec(torch.cat([up(y), skips.pop()], 1))
        return self.head(y)


def decode_output(logits: np.ndarray, med: np.ndarray, lc: np.ndarray, invalid: np.ndarray,
                  cfg: dict, th: Thresholds | None) -> np.ndarray:
    """logits (4,H,W) -> mask 0..3."""
    burn = 1 / (1 + np.exp(-logits[0])) > cfg["burn_threshold"]
    if cfg.get("mask_invalid", True):
        burn &= ~invalid
    if cfg.get("min_size"):
        burn = remove_small(burn, cfg["min_size"])
    if cfg.get("min_dnbr") is not None:
        burn &= med >= cfg["min_dnbr"]
    if cfg["severity"] == "rule" and th is not None:
        sev = th.severity(med, lc)
    else:
        sev = 1 + logits[1:].argmax(0)
    return np.where(burn, sev, 0).astype(np.uint8)


class BSUNetPredictor:
    def __init__(self, weights_dir: Path = ROOT / "weights", config: Path = ROOT / "configs" / "bs_unet.json"):
        self.cfg = json.load(open(config))
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channels = channel_index(self.cfg.get("channels", ["s2", "s1", "aux"]))
        self.models = []
        for name in self.cfg["weights"]:
            m = UNet(cin=len(self.channels), base=self.cfg.get("base", 16))
            m.load_state_dict(torch.load(weights_dir / name, map_location="cpu", weights_only=True))
            self.models.append(m.eval().to(self.device))
        self.th = Thresholds.load(ROOT / "configs" / "bs_thresholds.json")

    @torch.no_grad()
    def logits(self, chip: BSChip) -> np.ndarray:
        x = torch.from_numpy(bs_input(chip)[self.channels])[None].to(self.device)
        out = 0
        for m in self.models:
            out = out + m(x)
            if self.cfg.get("tta"):
                out = out + m(x.flip(-1)).flip(-1)
        n = len(self.models) * (2 if self.cfg.get("tta") else 1)
        return (out / n)[0].float().cpu().numpy()

    def __call__(self, chip: BSChip) -> np.ndarray:
        lg = self.logits(chip)
        med = smooth_dnbr(nbr(chip.s2_pre) - nbr(chip.s2_post))
        inv = invalid_mask(chip.s2_pre[..., 9], chip.s2_post[..., 9])
        return decode_output(lg, med, chip.aux[..., 2], inv, self.cfg, self.th)

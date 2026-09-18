"""Compact in-memory cache of per-chip BS layers used by fitting scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import nbr
from .io import list_chips, load_bs

CACHE = Path("cache")


def bs_layers(root: str, refresh: bool = False) -> dict[str, np.ndarray]:
    """Stacks (N,H,W): dnbr, nbr_pre, landcover, scl_pre, scl_post, mask; plus chip ids."""
    p = CACHE / f"bs_layers_{Path(root).name}.npz"
    if p.exists() and not refresh:
        z = np.load(p)
        return {k: z[k] for k in z.files}
    ids = list_chips(root, "bs")
    out = {k: [] for k in ("dnbr", "nbr_pre", "lc", "scl_pre", "scl_post", "mask")}
    for cid in ids:
        c = load_bs(root, cid)
        pre = nbr(c.s2_pre)
        out["dnbr"].append((pre - nbr(c.s2_post)).astype(np.float16))
        out["nbr_pre"].append(pre.astype(np.float16))
        out["lc"].append(c.aux[..., 2].astype(np.uint8))
        out["scl_pre"].append(c.s2_pre[..., 9].astype(np.uint8))
        out["scl_post"].append(c.s2_post[..., 9].astype(np.uint8))
        out["mask"].append(c.mask if c.mask is not None else np.zeros(c.aux.shape[:2], np.uint8))
    res = {k: np.stack(v) for k, v in out.items()}
    res["ids"] = np.array(ids)
    CACHE.mkdir(exist_ok=True)
    np.savez(p, **res)
    return res

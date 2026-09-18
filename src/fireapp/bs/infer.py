"""Инференс модуля BS: каталог с BS-чипами -> {chip_id: маска классов 0-3}."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fireapp.bs.dataset import BSDataset
from fireapp.bs.model import UNetSiamese


def run_bs_inference(
    data_dir: str | Path,
    cfg: dict,
    batch_size: int = 8,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """cfg — загруженный configs/bs.yaml (нужны model.branch_channels, model.static_channels,
    model.n_classes, weights_path)."""
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "meta.csv")

    dataset = BSDataset(data_dir, meta, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNetSiamese(
        branch_channels=cfg["model"]["branch_channels"],
        static_channels=cfg["model"]["static_channels"],
        n_classes=cfg["model"]["n_classes"],
    )
    state = torch.load(cfg["weights_path"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    preds: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            static = batch["static"].to(device)
            logits = model(pre, post, static)
            classes = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            for chip_id, mask in zip(batch["chip_id"], classes):
                preds[chip_id] = mask
    return preds

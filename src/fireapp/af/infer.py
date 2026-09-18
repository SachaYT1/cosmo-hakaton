"""Инференс модуля AF: каталог с AF-чипами -> {chip_id: бинарная маска}.

Фильтрация техногенных термоаномалий по геопространственной повторяемости
(af/postprocess.py) здесь НЕ применяется — геопривязка тестовых чипов скрыта
организаторами (см. постановку кейса), поэтому сопоставить чип с историческим
архивом по координатам невозможно. Модель отделяет факелы от природного
горения по признакам самого чипа, выученным во время обучения.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fireapp.af.dataset import AFDataset
from fireapp.af.model import UNetSmall


def run_af_inference(
    data_dir: str | Path,
    cfg: dict,
    threshold: float = 0.5,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """cfg — загруженный configs/af.yaml (нужны model.in_channels, model.n_classes, weights_path)."""
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "meta.csv")

    dataset = AFDataset(data_dir, meta, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNetSmall(in_channels=cfg["model"]["in_channels"], n_classes=cfg["model"]["n_classes"])
    state = torch.load(cfg["weights_path"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    preds: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).squeeze(1).cpu().numpy().astype(np.uint8)
            for chip_id, mask in zip(batch["chip_id"], masks):
                preds[chip_id] = mask
    return preds

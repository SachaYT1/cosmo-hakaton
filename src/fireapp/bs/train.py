"""Обучение модуля BS.

Запуск: python train.py --module bs --config configs/bs.yaml --data-dir /path/to/train
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from fireapp.bs.dataset import BSDataset
from fireapp.bs.model import UNetSiamese
from fireapp.common.config import load_config
from fireapp.common.losses import BSLoss
from fireapp.common.splits import group_split


def main(config_path: str, data_dir: str) -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    data_root = Path(data_dir)

    meta = pd.read_csv(data_root / "meta.csv")
    train_idx, val_idx = group_split(
        meta,
        group_cols=train_cfg["split_group_cols"],
        val_frac=train_cfg["val_split"],
        seed=train_cfg["seed"],
    )

    full_dataset = BSDataset(data_root, meta, train=True)
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(BSDataset(data_root, meta, train=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_cfg["seed"])

    model = UNetSiamese(
        branch_channels=cfg["model"]["branch_channels"],
        static_channels=cfg["model"]["static_channels"],
        n_classes=cfg["model"]["n_classes"],
    ).to(device)
    loss_fn = BSLoss(
        class_weights=train_cfg["class_weights"],
        ce_weight=cfg["loss"]["ce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])

    best_val_loss = float("inf")
    weights_path = Path(cfg["weights_path"])
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            static = batch["static"].to(device)
            valid = batch["valid"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(pre, post, static)
            loss = loss_fn(logits, y, valid)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pre.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pre = batch["pre"].to(device)
                post = batch["post"].to(device)
                static = batch["static"].to(device)
                valid = batch["valid"].to(device)
                y = batch["y"].to(device)
                logits = model(pre, post, static)
                val_loss += loss_fn(logits, y, valid).item() * pre.size(0)
        val_loss /= max(len(val_ds), 1)

        print(f"[bs] epoch {epoch + 1}/{train_cfg['epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True, help="каталог train/bs")
    args = parser.parse_args()
    main(args.config, args.data_dir)

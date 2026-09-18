"""Официальный скрипт подсчёта Score (см. постановку кейса, раздел «Метрика оценки»).

Score = 0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev
Все компоненты — микро-усреднение (TP/FP/FN суммируются по всем чипам).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fireapp.common.rle import rle_to_mask


def _confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def compute_score(
    pred_path: str | Path,
    gt_path: str | Path,
    meta_path: str | Path | list[str | Path],
) -> dict:
    """meta_path — один combined meta.csv, либо список путей (train/af/meta.csv
    и train/bs/meta.csv по отдельности — они конкатенируются)."""
    pred = pd.read_csv(pred_path, dtype={"rle": str}).fillna({"rle": ""})
    gt = pd.read_csv(gt_path, dtype={"rle": str}).fillna({"rle": ""})

    meta_paths = meta_path if isinstance(meta_path, (list, tuple)) else [meta_path]
    meta = pd.concat([pd.read_csv(p) for p in meta_paths], ignore_index=True)

    # набор пар (chip_id, class_id) в pred и gt должен совпадать с шаблоном/друг другом
    key_cols = ["chip_id", "class_id"]
    mismatch = pd.merge(gt[key_cols], pred[key_cols], on=key_cols, how="outer", indicator=True)
    mismatch = mismatch[mismatch["_merge"] != "both"]
    if not mismatch.empty:
        raise ValueError(f"submission не совпадает с шаблоном по (chip_id, class_id): {len(mismatch)} расхождений")

    size_by_chip = {r.chip_id: (int(r.height), int(r.width)) for r in meta.itertuples(index=False)}
    kind_by_chip = {r.chip_id: r.kind for r in meta.itertuples(index=False)}

    pred_rle = {(r.chip_id, r.class_id): r.rle for r in pred.itertuples(index=False)}
    gt_rle = {(r.chip_id, r.class_id): r.rle for r in gt.itertuples(index=False)}

    af_tp = af_fp = af_fn = 0
    burn_tp = burn_fp = burn_fn = 0
    sev_tp = {1: 0, 2: 0, 3: 0}
    sev_fp = {1: 0, 2: 0, 3: 0}
    sev_fn = {1: 0, 2: 0, 3: 0}

    for chip_id, kind in kind_by_chip.items():
        h, w = size_by_chip[chip_id]

        if kind == "af":
            p_mask = rle_to_mask(pred_rle.get((chip_id, 1), ""), h, w)
            g_mask = rle_to_mask(gt_rle.get((chip_id, 1), ""), h, w)
            tp, fp, fn = _confusion(p_mask, g_mask)
            af_tp += tp
            af_fp += fp
            af_fn += fn

        elif kind == "bs":
            p_union = np.zeros((h, w), dtype=np.uint8)
            g_union = np.zeros((h, w), dtype=np.uint8)
            for k in (1, 2, 3):
                p_k = rle_to_mask(pred_rle.get((chip_id, k), ""), h, w)
                g_k = rle_to_mask(gt_rle.get((chip_id, k), ""), h, w)
                tp, fp, fn = _confusion(p_k, g_k)
                sev_tp[k] += tp
                sev_fp[k] += fp
                sev_fn[k] += fn
                p_union |= p_k
                g_union |= g_k

            tp, fp, fn = _confusion(p_union, g_union)
            burn_tp += tp
            burn_fp += fp
            burn_fn += fn
        else:
            raise ValueError(f"неизвестный kind={kind!r} для chip_id={chip_id!r}")

    f1_af = _f1(af_tp, af_fp, af_fn)
    iou_burn = _iou(burn_tp, burn_fp, burn_fn)
    iou_sev = {k: _iou(sev_tp[k], sev_fp[k], sev_fn[k]) for k in (1, 2, 3)}
    miou_sev = sum(iou_sev.values()) / 3

    score = 0.35 * f1_af + 0.35 * iou_burn + 0.30 * miou_sev

    return {
        "F1_af": f1_af,
        "IoU_burn": iou_burn,
        "IoU_sev1": iou_sev[1],
        "IoU_sev2": iou_sev[2],
        "IoU_sev3": iou_sev[3],
        "mIoU_sev": miou_sev,
        "Score": score,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--meta", required=True, nargs="+", help="один или несколько meta.csv (af и/или bs)")
    args = parser.parse_args()
    print(compute_score(args.pred, args.gt, args.meta))

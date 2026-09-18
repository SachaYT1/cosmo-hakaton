"""Сборка submission.csv из предсказаний AF и BS в формате RLE (см. постановку кейса)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fireapp.common.rle import mask_to_rle


def build_submission(
    af_preds: dict[str, np.ndarray],
    bs_preds: dict[str, np.ndarray],
    sample_submission_path: str | Path,
    output_path: str | Path,
) -> None:
    """af_preds: {chip_id: маска (H,W) 0/1}. bs_preds: {chip_id: маска (H,W) 0..3}."""
    template = pd.read_csv(sample_submission_path)
    rows = []

    for chip_id, mask in af_preds.items():
        rows.append({"chip_id": chip_id, "class_id": 1, "rle": mask_to_rle(mask)})

    for chip_id, mask in bs_preds.items():
        for class_id in (1, 2, 3):
            rows.append(
                {"chip_id": chip_id, "class_id": class_id, "rle": mask_to_rle(mask == class_id)}
            )

    submission = pd.DataFrame(rows)

    # проверка полного совпадения набора пар (chip_id, class_id) с шаблоном
    key_cols = ["chip_id", "class_id"]
    missing = template.merge(submission[key_cols], on=key_cols, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    if not missing.empty:
        raise ValueError(f"submission не покрывает {len(missing)} строк шаблона: {missing[key_cols].values}")

    submission = template[key_cols].merge(submission, on=key_cols, how="left")
    submission["rle"] = submission["rle"].fillna("")
    submission.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL -> rle всегда в кавычках

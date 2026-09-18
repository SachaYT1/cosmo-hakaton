import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fireapp.common.rle import mask_to_rle, rle_to_mask


def test_roundtrip_empty():
    mask = np.zeros((8, 8), dtype=np.uint8)
    rle = mask_to_rle(mask)
    assert rle == ""
    assert np.array_equal(rle_to_mask(rle, 8, 8), mask)


def test_roundtrip_pattern():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2, 3:5] = 1  # строка 3, столбцы 4-5 (1-индексация)
    mask[4, 1:4] = 1  # строка 5, столбцы 2-4

    rle = mask_to_rle(mask)
    assert rle == "20 2 34 3"
    assert np.array_equal(rle_to_mask(rle, 8, 8), mask)

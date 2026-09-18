import numpy as np

from app.rle import decode_rle, encode_rle


def test_decode_known_pairs():
    # Пиксели нумеруются с 1, построчно: "1 3" -> пиксели 1..3, "10 2" -> 10..11.
    mask = decode_rle("1 3 10 2", width=4, height=4)
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[0, 0:3] = 1          # пиксели 1,2,3
    expected[2, 1:3] = 1          # пиксели 10,11 (строка 3, колонки 2-3)
    assert mask.dtype == np.uint8
    assert (mask == expected).all()


def test_decode_empty_string_gives_zeros():
    mask = decode_rle("", width=8, height=8)
    assert mask.shape == (8, 8)
    assert mask.sum() == 0


def test_encode_known_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0:3] = 1
    mask[2, 1:3] = 1
    assert encode_rle(mask) == "1 3 10 2"


def test_roundtrip_random():
    rng = np.random.default_rng(42)
    mask = (rng.random((16, 16)) > 0.7).astype(np.uint8)
    assert (decode_rle(encode_rle(mask), 16, 16) == mask).all()


def test_encode_empty_mask():
    assert encode_rle(np.zeros((4, 4), dtype=np.uint8)) == ""

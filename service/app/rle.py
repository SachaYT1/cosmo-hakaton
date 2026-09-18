"""RLE-кодек по правилам постановки: нумерация пикселей построчно, с единицы."""

import numpy as np


def decode_rle(rle: str, width: int, height: int) -> np.ndarray:
    """Decode "start length ..." pairs (1-based, row-major) into a (height, width) mask."""
    mask = np.zeros(width * height, dtype=np.uint8)
    if rle and rle.strip():
        values = [int(v) for v in rle.split()]
        if len(values) % 2 != 0:
            raise ValueError("RLE must contain an even number of values")
        for start, length in zip(values[0::2], values[1::2]):
            if start < 1 or start - 1 + length > mask.size:
                raise ValueError(f"RLE run out of bounds: start={start} length={length}")
            mask[start - 1 : start - 1 + length] = 1
    return mask.reshape(height, width)


def encode_rle(mask: np.ndarray) -> str:
    """Encode a binary mask into "start length ..." pairs (1-based, row-major)."""
    flat = (np.asarray(mask).ravel() > 0).astype(np.int8)
    padded = np.concatenate([[0], flat, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    lengths = changes[1::2] - starts
    return " ".join(f"{s + 1} {l}" for s, l in zip(starts, lengths))

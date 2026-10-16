"""Generation sizes and how long they take.

Every size is about one megapixel in multiples of 64 (draw-things-cli rejects others). The
size closest to the requested aspect ratio (by log-ratio) is generated; the workflow then
cover-crops it to the exact output size, so the aspect ratio is never distorted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

SIZES: tuple[tuple[int, int], ...] = (
    (640, 1536),  # 0.42 (9:21)
    (704, 1408),  # 0.50
    (768, 1344),  # 0.57 (9:16)
    (832, 1280),  # 0.65
    (896, 1152),  # 0.78
    (1024, 1024),  # 1.00
    (1152, 896),  # 1.29
    (1280, 832),  # 1.54
    (1344, 768),  # 1.75 (16:9)
    (1408, 704),  # 2.00
    (1536, 640),  # 2.40 (21:9)
)

# Seconds per image (cold process, model load included), measured with draw-things-cli
# v26.1011.0 on the target Mac (M5, 32 GB) — per model, only what was actually run.
# 832x1280, 896x1152 and 1152x896 were run earlier with other models, not with klein.
MEASURED_SECONDS: Mapping[str, Mapping[tuple[int, int], float]] = {
    "flux_2_klein_9b_i8x.ckpt": {
        (1024, 1024): 25.9,
        (1344, 768): 30.4,
        (768, 1344): 26.0,
        (1536, 640): 35.1,
    },
    "flux_2_klein_4b_i8x.ckpt": {
        (1024, 1024): 14.0,
    },
}
VERIFIED_SIZES: Mapping[str, frozenset[tuple[int, int]]] = {
    model: frozenset(times) for model, times in MEASURED_SECONDS.items()
}
SQUARE = (1024, 1024)


def generation_size(aspect_ratio: float) -> tuple[int, int]:
    if aspect_ratio <= 0:
        raise ValueError("aspect ratio must be positive")
    return min(SIZES, key=lambda size: abs(math.log(size[0] / size[1] / aspect_ratio)))


def _skew(size: tuple[int, int]) -> float:
    return abs(math.log(size[0] / size[1]))


def size_factor(size: tuple[int, int], reference_model: str) -> float:
    """How much slower than 1024x1024 a size is, never under-predicting.

    Wide and tall sizes are slower than their pixel count suggests (1536x640 is 0.94 MP but
    took 1.36x the square's time). A measured size uses ``max(pixels, measured)``; an
    unmeasured one also takes the measured factor of the next size at least as far from
    square (or the most skewed measured one).
    """
    times = MEASURED_SECONDS[reference_model]
    base = times[SQUARE]
    pixels = size[0] * size[1] / (SQUARE[0] * SQUARE[1])
    if size in times:
        return max(pixels, times[size] / base)
    skew = _skew(size)
    further = [s for s in times if _skew(s) >= skew]
    if further:  # the closest skew at least as large; on a tie, the slower one
        nearest = min(further, key=lambda s: (_skew(s), -times[s]))
    else:
        nearest = max(times, key=_skew)
    return max(pixels, times[nearest] / base)


# Measured with the 9B model at 1024x1024: one reference 50.3 s, two 72.9 s (cold 25.9 s).
# (factors 1.94 and 2.81, rounded up). 3-4 are extrapolated with the larger step (+0.95).
REFERENCE_FACTORS = (1.0, 1.95, 2.85, 3.8, 4.75)


def reference_factor(references: int) -> float:
    return REFERENCE_FACTORS[min(max(references, 0), len(REFERENCE_FACTORS) - 1)]

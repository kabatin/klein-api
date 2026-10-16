"""Generation sizes: about one megapixel, multiples of 64 (draw-things-cli rejects others).

The size closest to the requested aspect ratio (by log-ratio) is generated; the workflow then
cover-crops it to the exact output size, so the aspect ratio is never distorted.
"""

from __future__ import annotations

import math

# (width, height). Measured on the target Mac (2026-10-16): 1024x1024, 1344x768, 1152x896,
# 896x1152, 832x1280. The others follow the same rule (~1 MP, multiples of 64) but have not
# been run yet.
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
VERIFIED_SIZES = frozenset({(1024, 1024), (1344, 768), (1152, 896), (896, 1152), (832, 1280)})


def generation_size(aspect_ratio: float) -> tuple[int, int]:
    if aspect_ratio <= 0:
        raise ValueError("aspect ratio must be positive")
    return min(SIZES, key=lambda size: abs(math.log(size[0] / size[1] / aspect_ratio)))

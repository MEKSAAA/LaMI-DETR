"""Utilities for 1000x1000 normalized coordinate conversions.

Doubao grounding uses coordinates in [0, 999] over a virtual 1000x1000 grid.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def clamp_1000(v: float) -> float:
    return max(0.0, min(999.0, float(v)))


def bbox_1000_to_pixel(bbox_1000: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    w, h = image_size
    x1, y1, x2, y2 = [clamp_1000(v) for v in bbox_1000]
    return [
        x1 * w / 1000.0,
        y1 * h / 1000.0,
        x2 * w / 1000.0,
        y2 * h / 1000.0,
    ]


def bbox_pixel_to_1000(bbox_px: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        clamp_1000(x1 * 1000.0 / w),
        clamp_1000(y1 * 1000.0 / h),
        clamp_1000(x2 * 1000.0 / w),
        clamp_1000(y2 * 1000.0 / h),
    ]


def points_1000_to_norm(points_1000: Iterable[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for p in points_1000:
        x, y = float(p[0]), float(p[1])
        out.append([clamp_1000(x) / 1000.0, clamp_1000(y) / 1000.0])
    return out


def points_norm_to_1000(points_norm: Iterable[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for p in points_norm:
        x, y = float(p[0]), float(p[1])
        out.append([clamp_1000(x * 1000.0), clamp_1000(y * 1000.0)])
    return out


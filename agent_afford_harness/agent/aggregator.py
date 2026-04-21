"""Evidence aggregation → benchmark-ready normalized points."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_afford_harness.agent import router as R


def _spread_points_in_rect(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    n: int,
    seed: int = 0,
) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    pts: List[Tuple[float, float]] = []
    if xmax <= xmin or ymax <= ymin:
        return pts
    for i in range(n):
        u = 0.25 + 0.5 * (i / max(n - 1, 1)) if n > 1 else 0.5
        vx = xmin + (xmax - xmin) * (0.35 + 0.3 * rng.random() + 0.1 * math.sin(i + 0.3))
        vy = ymin + (ymax - ymin) * (0.35 + 0.3 * rng.random() + 0.1 * math.cos(i + 0.1))
        pts.append((vx, vy))
    return pts


def aggregate_recognition(
    image_size: Tuple[int, int],
    lami_result: Dict[str, Any],
) -> List[Tuple[float, float]]:
    W, H = image_size
    boxes = lami_result.get("boxes") or []
    if not boxes:
        return [(W * 0.5, H * 0.5)]
    best = max(boxes, key=lambda b: b.get("score", 0.0))
    x1, y1, x2, y2 = best.get("bbox", [0, 0, W, H])
    return _spread_points_in_rect(x1, y1, x2, y2, 3, seed=1)


def aggregate_part(
    full_image_size: Tuple[int, int],
    crop_meta: Optional[Dict[str, Any]],
    code_result: Dict[str, Any],
    lami_result: Dict[str, Any],
) -> List[Tuple[float, float]]:
    """
    Prefer code tool points in *full image* coords if provided;
    else shrink-center heuristic inside LaMI box.
    """
    pts_norm_crop = code_result.get("points_normalized_crop") or []
    pts_full_from_code = code_result.get("points_full_image") or []

    if pts_full_from_code:
        return [(float(p[0]), float(p[1])) for p in pts_full_from_code]

    if pts_norm_crop and crop_meta:
        ox0, oy0 = crop_meta.get("origin_xy", (0.0, 0.0))
        cw = crop_meta.get("crop_width") or 1
        ch = crop_meta.get("crop_height") or 1
        Wf, Hf = full_image_size
        out = []
        for nx, ny in pts_norm_crop:
            xf = ox0 + nx * cw
            yf = oy0 + ny * ch
            out.append((xf, yf))
        return out

    # Fallback: center of best LaMI box
    boxes = lami_result.get("boxes") or []
    W, H = full_image_size
    if not boxes:
        return [(W * 0.5, H * 0.5)]
    best = max(boxes, key=lambda b: b.get("score", 0.0))
    x1, y1, x2, y2 = best.get("bbox", [0, 0, W, H])
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    return [(mx, my), (mx + 3, my), (mx - 3, my)]


def aggregate_spatial(
    image_size: Tuple[int, int],
    code_result: Dict[str, Any],
    lami_result: Dict[str, Any],
) -> List[Tuple[float, float]]:
    pts_full = code_result.get("points_full_image") or []
    if pts_full:
        return [(float(p[0]), float(p[1])) for p in pts_full]

    # Fallback: midpoint between top two boxes horizontally
    boxes = sorted((lami_result.get("boxes") or []), key=lambda b: -b.get("score", 0.0))[:2]
    W, H = image_size
    if len(boxes) < 2:
        return [(W * 0.5, H * 0.5)]
    b1 = boxes[0]["bbox"]
    b2 = boxes[1]["bbox"]
    cx1 = (b1[0] + b1[2]) / 2
    cx2 = (b2[0] + b2[2]) / 2
    mx = (cx1 + cx2) / 2
    my = (b1[1] + b2[3]) / 4 + (b2[1] + b2[3]) / 4
    pts = [(mx + i * 4, my + i * 2) for i in range(-2, 3)]
    return pts[:5]

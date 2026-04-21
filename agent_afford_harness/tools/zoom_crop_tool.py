"""Deterministic zoom / crop with full back-projection metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def tool_zoom_crop(
    image: Image.Image,
    bbox_xyxy: List[float],
    padding_ratio: float = 0.15,
    max_side: int = 1024,
) -> Dict[str, Any]:
    w, h = image.size
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * padding_ratio
    pad_h = bh * padding_ratio
    ox1 = max(0.0, x1 - pad_w)
    oy1 = max(0.0, y1 - pad_h)
    ox2 = min(float(w), x2 + pad_w)
    oy2 = min(float(h), y2 + pad_h)

    crop = image.crop((int(ox1), int(oy1), int(ox2), int(oy2)))
    cw, ch = crop.size
    if max(cw, ch) > max_side:
        scale = max_side / max(cw, ch)
        rcw, rch = int(cw * scale), int(ch * scale)
        crop = crop.resize((rcw, rch), Image.BICUBIC)
    else:
        scale = 1.0
        rcw, rch = cw, ch

    return {
        "crop": crop,
        "origin_xy": (float(ox1), float(oy1)),
        "crop_size_raw": (int(cw), int(ch)),
        "crop_width": float(rcw),
        "crop_height": float(rch),
        "scale": float(scale),
        "bbox_input_xyxy": [float(x) for x in bbox_xyxy],
        "full_image_size": (w, h),
        "inverse_map_note": "full_x = origin_x + norm_x * crop_width (after resize)",
    }


def crop_coords_to_full(
    xy_norm_crop: Tuple[float, float],
    crop_meta: Dict[str, Any],
) -> Tuple[float, float]:
    nx, ny = xy_norm_crop
    ox, oy = crop_meta["origin_xy"]
    cw = crop_meta["crop_width"]
    ch = crop_meta["crop_height"]
    return ox + nx * cw, oy + ny * ch

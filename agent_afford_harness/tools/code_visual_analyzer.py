"""
Template-based deterministic visual analysis (no free-form codegen).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from agent_afford_harness.tools.zoom_crop_tool import crop_coords_to_full


def _np_gray(rgb: Image.Image) -> np.ndarray:
    g = np.asarray(rgb.convert("L"))
    return g.astype(np.float32) / 255.0


def template_interior_region(gray: np.ndarray, n: int = 3) -> List[Tuple[float, float]]:
    """Central interior via threshold + distance transform peaks."""
    try:
        import cv2
    except Exception:
        h, w = gray.shape
        return [(0.5, 0.5)] * n

    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    inv = cv2.erode(inv, np.ones((5, 5), np.uint8), iterations=2)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        h, w = gray.shape
        return [(0.5, 0.5)] * n
    flat_idx = np.argsort(dist.ravel())[::-1][: max(50, n * 10)]
    h, w = gray.shape
    pts: List[Tuple[float, float]] = []
    ys, xs = np.unravel_index(flat_idx, dist.shape)
    margin = 0.08
    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        nx, ny = x / max(w - 1, 1), y / max(h - 1, 1)
        if margin < nx < 1 - margin and margin < ny < 1 - margin:
            pts.append((nx, ny))
        if len(pts) >= n:
            break
    while len(pts) < n:
        pts.append((0.5, 0.5))
    return pts[:n]


def template_elongated_grasp(gray: np.ndarray, n: int = 3) -> List[Tuple[float, float]]:
    try:
        import cv2
    except Exception:
        return [(0.35, 0.5), (0.5, 0.5), (0.65, 0.5)]

    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    e = cv2.Canny(u8, 60, 120)
    contours, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [(0.35, 0.5), (0.5, 0.5), (0.65, 0.5)]
    best = max(contours, key=lambda c: cv2.arcLength(c, False))
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    center = rect[0]
    xs, ys = box[:, 0], box[:, 1]
    mini, maxi = xs.min(), xs.max()
    pts = []
    for t in np.linspace(0.25, 0.75, n):
        x = (mini + (maxi - mini) * t) / gray.shape[1]
        y = center[1] / gray.shape[0]
        pts.append((float(x), float(y)))
    return pts[:n]


def template_free_space_between_boxes(
    crop_size: Tuple[int, int],
    boxes_xyxy: Sequence[Sequence[float]],
    n: int = 5,
) -> List[Tuple[float, float]]:
    """Sample gap between two boxes (crop pixel xyxy → normalized crop coords)."""
    w, h = crop_size
    if len(boxes_xyxy) < 2:
        return [(0.5, 0.5)] * n
    b0 = [float(x) for x in boxes_xyxy[0]]
    b1 = [float(x) for x in boxes_xyxy[1]]
    if b0[0] > b1[0]:
        b0, b1 = b1, b0
    # horizontal gap between two non-overlapping side-by-side boxes
    if b0[2] < b1[0]:
        x1, x2 = b0[2], b1[0]
    elif b1[2] < b0[0]:
        x1, x2 = b1[2], b0[0]
    else:
        x1 = (b0[0] + b0[2]) / 2
        x2 = (b1[0] + b1[2]) / 2
        if x1 > x2:
            x1, x2 = x2, x1
    y_low = max(b0[1], b1[1])
    y_high = min(b0[3], b1[3])
    if y_high <= y_low:
        y_mid = ((b0[3] + b1[3]) / 2)
        y_low, y_high = y_mid - 10, y_mid + 10
    xm0 = (x1 + x2) / 2
    ym0 = (y_low + y_high) / 2
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        x = x1 + (x2 - x1) * t
        y = ym0 + (i - n / 2) * 0.06 * (y_high - y_low + 1)
        pts.append((x / max(w, 1), y / max(h, 1)))
    return pts[:n]


def tool_code_visual_analyzer(
    image: Image.Image,
    analysis_type: str,
    *,
    candidate_boxes_xyxy: Optional[Sequence[Sequence[float]]] = None,
    crop_meta: Optional[Dict[str, Any]] = None,
    full_image_size: Optional[Tuple[int, int]] = None,
    n_points: int = 3,
) -> Dict[str, Any]:
    """
    Returns normalized crop points and optional projection to full image using crop_meta.
    """
    gray = _np_gray(image)
    cw, ch = image.size

    at = (analysis_type or "").lower().replace("-", "_")
    if at in ("free_space_between_boxes", "spatial_affordance_localization"):
        pts = template_free_space_between_boxes(
            (cw, ch),
            candidate_boxes_xyxy or [],
            n=max(5, n_points),
        )
    elif at in ("elongated_grasp_region",):
        pts = template_elongated_grasp(gray, n=max(3, n_points))
    else:
        pts = template_interior_region(gray, n=max(3, n_points))

    full: List[Tuple[float, float]] = []
    if crop_meta and full_image_size:
        for p in pts:
            full.append(crop_coords_to_full(p, crop_meta))
    elif full_image_size:
        fw, fh = full_image_size
        for p in pts:
            full.append((p[0] * fw, p[1] * fh))

    conf = 0.7 if candidate_boxes_xyxy and len(candidate_boxes_xyxy) >= 2 else 0.55
    return {
        "analysis_type": analysis_type,
        "points_normalized_crop": [[float(a), float(b)] for a, b in pts],
        "points_full_image": [[float(a), float(b)] for a, b in full] if full else [],
        "text": f"template={at} points={len(pts)}",
        "confidence": conf,
        "mask_debug": None,
    }
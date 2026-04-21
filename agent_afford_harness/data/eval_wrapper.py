"""RoboAfford-style point accuracy (normalized coords vs mask)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def parse_points_from_answer(answer_text: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    patterns = [
        r"\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)",
        r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]",
        r"(\d+\.?\d*)\s+(\d+\.?\d*)",
        r"(\d+\.?\d*),(\d+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer_text)
        if matches:
            for match in matches:
                try:
                    points.append((float(match[0]), float(match[1])))
                except (ValueError, IndexError):
                    continue
    return points


def calculate_accuracy(pred_points: Sequence[Sequence[float]], mask_path: str) -> float:
    pts = [tuple(map(float, p)) for p in pred_points]
    if not pts:
        return 0.0
    try:
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        height, width = mask.shape
        correct_count = 0
        is_normalized = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pts)
        for x, y in pts:
            if is_normalized:
                xi = int(round(x * width))
                yi = int(round(y * height))
            else:
                xi = int(round(x))
                yi = int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                if mask[yi, xi] > 0.5:
                    correct_count += 1
        return correct_count / len(pts)
    except Exception:
        return 0.0


def points_list_to_answer_string(points: List[List[float]]) -> str:
    inner = ", ".join(f"({p[0]:.4f}, {p[1]:.4f})" for p in points)
    return f"[{inner}]"


def evaluate_trace_points(
    final_points: List[List[float]],
    mask_path: str,
) -> Dict[str, Any]:
    acc = calculate_accuracy(final_points, mask_path)
    return {
        "accuracy": acc,
        "num_points": len(final_points),
        "mask_path": mask_path,
    }

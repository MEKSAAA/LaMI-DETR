"""Minimal task parsing helpers (reserved for future LLM-based parsing)."""

from __future__ import annotations

import re
from typing import Tuple


def strip_benchmark_format_instructions(question: str) -> str:
    """Keep semantic query; drop trailing coordinate-format boilerplate when present."""
    q = question.strip()
    # Common RoboAfford suffix pattern
    if "formatted as a list of tuples" in q.lower():
        q = re.split(r"(?:Your answer should be|Please provide|Coordinates should)", q, flags=re.I)[0].strip()
    return q


def normalize_category_label(category: str) -> str:
    c = category.lower().strip()
    if c == "object affordance":
        return "object affordance prediction"
    if c == "object reference":
        return "object affordance recognition"
    if c == "spatial affordance":
        return "spatial affordance localization"
    return c


def split_spatial_entities(question: str) -> Tuple[str, str]:
    """Very light heuristic: return (left-ish phrase, right-ish phrase) for 'between A and B'."""
    m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\.|$)", question, flags=re.I | re.S)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()

"""Load RoboAfford-Eval style samples (JSON annotations + image/mask paths)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_afford_harness.paths import harness_root


@dataclass
class RoboSample:
    sample_id: str
    image_path: str
    question: str
    category: str
    mask_path: str
    raw: Dict[str, Any]

    @property
    def img(self) -> str:
        return self.raw.get("img", "")


def default_annotations_path() -> Path:
    # .../LaMI-DETR/agent_afford_harness/data/ -> parents[2]=LaMI-DETR, parent->projects26
    return Path(__file__).resolve().parents[2].parent / "RoboAfford" / "annotations_normxy.json"


def load_annotations(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or default_annotations_path()
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def annotation_to_sample(
    entry: Dict[str, Any],
    idx: int,
    *,
    image_root: Path,
    mask_root: Path,
    sample_id_prefix: str = "ann",
) -> RoboSample:
    img_rel = entry.get("img", "")
    mask_rel = entry.get("mask", "")
    sid = f"{sample_id_prefix}_{idx:05d}_{img_rel.replace('/', '_')}"
    return RoboSample(
        sample_id=sid,
        image_path=str(image_root / img_rel),
        question=str(entry.get("question", "")),
        category=str(entry.get("category", "")),
        mask_path=str(mask_root / mask_rel),
        raw=dict(entry),
    )


def load_showcase_samples(
    subset_json: Optional[Path] = None,
    *,
    image_root: Optional[Path] = None,
    mask_root: Optional[Path] = None,
) -> List[RoboSample]:
    """Load showcase entries from agent_afford_harness/data/sample_subset.json."""
    pack = harness_root()
    sj = subset_json or (pack / "data" / "sample_subset.json")
    with open(sj, "r", encoding="utf-8") as f:
        meta = json.load(f)
    entries = meta.get("entries") or meta
    def _res(p: Optional[str], default_rel: str) -> Path:
        if not p:
            return (pack / default_rel).resolve()
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (pack / pp).resolve()

    img_root = Path(image_root) if image_root else _res(meta.get("image_root"), "../../../RoboAfford/images")
    m_root = Path(mask_root) if mask_root else _res(meta.get("mask_root"), "../../../RoboAfford/masks")
    if meta.get("annotations_path"):
        ap = Path(meta["annotations_path"])
        ann_path = ap if ap.is_absolute() else (pack / ap).resolve()
    else:
        ann_path = default_annotations_path()
    all_ann = load_annotations(ann_path)
    out: List[RoboSample] = []
    for item in entries:
        idx = int(item["annotation_index"])
        sid = item.get("sample_id") or f"showcase_{idx}"
        bucket = item.get("bucket", "")
        e = all_ann[idx]
        s = annotation_to_sample(e, idx, image_root=img_root, mask_root=m_root)
        s.sample_id = sid
        s.raw["_bucket"] = bucket
        out.append(s)
    return out


def resolve_path_maybe(path: str, bases: List[Path]) -> str:
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    for b in bases:
        cand = (b / path).resolve()
        if cand.is_file():
            return str(cand)
    return path

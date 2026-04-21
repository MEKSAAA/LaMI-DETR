"""
Six hand-written skills (declarative policies). No external side effects.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agent_afford_harness.agent import router as R
from agent_afford_harness.agent.task_parser import split_spatial_entities, strip_benchmark_format_instructions
from agent_afford_harness.config_load import load_prompt_hints


def skill_task_type_classification(question: str, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    hints = hints or load_prompt_hints()
    q = strip_benchmark_format_instructions(question).lower()
    spatial_kw = [k.lower() for k in (hints.get("task_type_keywords") or {}).get("spatial", [])]
    part_kw = [k.lower() for k in (hints.get("task_type_keywords") or {}).get("part", [])]

    score_spatial = sum(1 for k in spatial_kw if k in q)
    score_part = sum(1 for k in part_kw if k in q)

    # Strong spatial signals
    if "between" in q and "and" in q:
        score_spatial += 3
    if "free area" in q or "empty" in q or "free space" in q:
        score_spatial += 2

    chosen = R.OBJECT_AFFORDANCE_PREDICTION
    reason = "default object affordance (part) unless spatial/recognition cues dominate"
    conf = 0.55

    if score_spatial >= max(2, score_part + 1):
        chosen = R.SPATIAL_AFFORDANCE_LOCALIZATION
        reason = "spatial/between/free-space cues"
        conf = min(0.95, 0.6 + 0.05 * score_spatial)
    elif score_part >= 2 and score_spatial < 2:
        chosen = R.OBJECT_AFFORDANCE_PREDICTION
        reason = "part/functional-region cues"
        conf = min(0.93, 0.65 + 0.06 * score_part)
    elif score_part == 0 and score_spatial == 0:
        # recognition: pick object, highlight, left/right relations without "part" language
        if any(x in q for x in ("highlight", "select", "which object", "pick", "point")):
            chosen = R.OBJECT_AFFORDANCE_RECOGNITION
            reason = "object selection / reference language"
            conf = 0.78

    return {"task_type": chosen, "confidence": float(conf), "reason": reason}


def _expand_search_from_evidence(question: str, search_blob: Dict[str, Any]) -> str:
    """Flatten search concepts into a short suffix for prompts."""
    parts: List[str] = []
    for q in search_blob.get("queries", []) or []:
        res = (search_blob.get("results") or {}).get(q) or {}
        concepts = res.get("concepts") or []
        cues = res.get("visual_cues") or []
        parts.extend(concepts[:2])
        parts.extend(cues[:1])
    if not parts:
        return ""
    uniq = []
    for p in parts:
        if p and p not in uniq:
            uniq.append(p)
    return "; ".join(uniq[:5])


def skill_lami_prompt_engineering(
    question: str,
    task_info: Dict[str, Any],
    search_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task = task_info.get("task_type")
    q0 = strip_benchmark_format_instructions(question)
    q = re.sub(r"\s+", " ", q0).strip()
    search_hint = _expand_search_from_evidence(q, search_result) if search_result else ""

    object_prompt = q
    part_prompt = ""
    spatial_query = ""
    entity_names: List[str] = []
    negative_hint = "not background clutter"

    if task == R.SPATIAL_AFFORDANCE_LOCALIZATION:
        a, b = split_spatial_entities(q)
        if a and b:
            entity_names = ["entity_a", "entity_b"]
            object_prompt = f"{a}"
            part_prompt = f"{b}"
            spatial_query = q
        else:
            spatial_query = q
            entity_names = ["anchor_left", "anchor_right"]
    elif task == R.OBJECT_AFFORDANCE_PREDICTION:
        m = re.search(
            r"(?:what part|which part|part) (?:of (?:a |an |the )?)?(.+?)(?:\?|$)",
            q,
            flags=re.I,
        )
        obj_guess = m.group(1).strip() if m else "object"
        object_prompt = obj_guess[:120]
        part_prompt = re.sub(
            r"^.*?((?:handle|blade|cap|body|mouth|rim|edge|bottom|top|interior).*)$",
            r"\1",
            q,
            flags=re.I | re.S,
        )
        if part_prompt == q:
            part_prompt = f"functional region for: {q[:160]}"
        negative_hint = "not unrelated background"
    else:
        # recognition — compress to object + scene relation
        object_prompt = re.sub(
            r"(?i)^(highlight|mark|locate|select|pick|several points on)\s+",
            "",
            q,
        )
        object_prompt = object_prompt[:160]
        negative_hint = "wrong instance, duplicate object class in other region"

    if search_hint:
        object_prompt = f"{object_prompt}. Context: {search_hint}"

    return {
        "object_prompt": object_prompt[:240],
        "part_prompt": (part_prompt or "")[:240],
        "spatial_query": spatial_query[:240],
        "entity_names": entity_names,
        "negative_hint": negative_hint,
        "lami_classes": _build_lami_classes(task, object_prompt, part_prompt, spatial_query, entity_names),
    }


def _build_lami_classes(
    task: str,
    object_prompt: str,
    part_prompt: str,
    spatial_query: str,
    entity_names: List[str],
) -> Dict[str, List[str]]:
    """
    Build names + visual_descs fragments for LaMI (list of short descriptor strings per class).
    """
    if task == R.SPATIAL_AFFORDANCE_LOCALIZATION and len(entity_names) >= 2:
        # Two anchors for free-space
        left = spatial_query.split("between")[1].split("and")[0].strip() if "between" in spatial_query.lower() else object_prompt
        rest = spatial_query.split("and", 1)[1].strip() if "between" in spatial_query.lower() and "and" in spatial_query.lower() else part_prompt
        return {
            entity_names[0]: _split_prompt_to_descs(left or "left object"),
            entity_names[1]: _split_prompt_to_descs(rest or "right object"),
        }
    if task == R.OBJECT_AFFORDANCE_PREDICTION:
        return {
            "target_object": _split_prompt_to_descs(object_prompt),
            "functional_part": _split_prompt_to_descs(part_prompt or object_prompt),
        }
    # recognition
    return {"target": _split_prompt_to_descs(object_prompt)}


def _split_prompt_to_descs(text: str) -> List[str]:
    t = text.strip()
    if not t:
        return ["object"]
    chunks = re.split(r"[,;]|\sand\s", t)
    out = [c.strip() for c in chunks if c.strip()]
    if len(out) == 1:
        # split long sentence into pseudo-attributes
        words = out[0].split()
        if len(words) > 8:
            mid = len(words) // 2
            return [" ".join(words[:mid]), " ".join(words[mid:])]
    return out[:6] if out else ["object"]


def skill_search_triggering(question: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
    hints = load_prompt_hints()
    q = strip_benchmark_format_instructions(question).lower()
    extra = hints.get("search_trigger_extra") or {}
    attr_words = [a.lower() for a in extra.get("attributes", [])]

    triggered = False
    reasons: List[str] = []
    if any(a in q for a in attr_words):
        triggered = True
        reasons.append("attribute/long-tail phrase")
    if len(q.split()) > 28:
        triggered = True
        reasons.append("long question")
    if any(x in q for x in ("novel", "unusual", "specific type", "brand")):
        triggered = True
        reasons.append("explicit rarity")

    queries: List[str] = []
    if triggered:
        short = " ".join(q.split()[:20])
        queries.append(f"{short} visual features object recognition")
        queries.append(f"{short} common appearance cues")

    return {"need_search": triggered, "queries": queries[:3], "reasons": reasons}


def _box_area(xyxy: List[float]) -> float:
    return max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])


def skill_zoom_before_detail(
    task_type: str,
    question: str,
    lami_boxes: List[Dict[str, Any]],
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    q = strip_benchmark_format_instructions(question).lower()
    fine = any(
        k in q
        for k in (
            "handle",
            "blade",
            "cap",
            "interior",
            "edge",
            "between",
            "free",
            "part",
            "rim",
            "mouth",
        )
    )
    need_zoom = False
    padding = 0.15
    if task_type == R.OBJECT_AFFORDANCE_PREDICTION:
        need_zoom = True
    if task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION and fine:
        need_zoom = True

    # small best box => zoom more aggressive (pixel area ratio)
    if lami_boxes and image_size:
        W, H = image_size
        best = max(lami_boxes, key=lambda b: b.get("score", 0.0))
        bbox = best.get("bbox") or [0, 0, 0, 0]
        area_ratio = _box_area(bbox) / float(max(W * H, 1))
        if area_ratio < 0.03:
            need_zoom = True

    target_idx = 0
    if lami_boxes:
        # largest area box for cropping (coarse object)
        def area(b):
            bb = b.get("bbox") or [0, 0, 0, 0]
            return _box_area(bb)

        target_idx = max(range(len(lami_boxes)), key=lambda i: area(lami_boxes[i]))

    return {
        "need_zoom": bool(need_zoom),
        "crop_target_index": target_idx,
        "padding_ratio": padding,
        "resize_max_side": 1024,
        "reason": "part/spatial detail or small detection",
    }


def skill_code_when_geometry_matters(
    task_type: str,
    question: str,
    tool_outputs_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    q = strip_benchmark_format_instructions(question).lower()
    need = False
    analysis = "interior_region"

    if task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION:
        need = True
        analysis = "free_space_between_boxes"
    elif task_type == R.OBJECT_AFFORDANCE_PREDICTION:
        need = True
        if any(k in q for k in ("handle", "grip", "grasp", "held")):
            analysis = "elongated_grasp_region"
        elif any(k in q for k in ("interior", "liquid", "inside", "mouth", "cup")):
            analysis = "interior_region"
        else:
            analysis = "interior_region"
    else:
        need = False
        analysis = "none"

    return {"need_code_analysis": need, "analysis_type": analysis}


def skill_point_output_normalization(
    points_xy: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
    task_type: str,
) -> List[List[float]]:
    """
    Ensure list of [x,y] in [0,1]. Accepts either normalized or pixel inputs.
    """
    if not points_xy:
        return []

    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    norm_in = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points_xy)

    out: List[List[float]] = []
    for x, y in points_xy:
        if norm_in:
            nx = min(1.0, max(0.0, float(x)))
            ny = min(1.0, max(0.0, float(y)))
        else:
            nx = min(1.0, max(0.0, float(x) / max(image_width, 1)))
            ny = min(1.0, max(0.0, float(y) / max(image_height, 1)))
        out.append([nx, ny])

    # lightly avoid exact border (spec)
    eps = 1e-3
    out = [[min(1 - eps, max(eps, p[0])), min(1 - eps, max(eps, p[1]))] for p in out]

    # expected counts — pad by jitter from mean if short
    n_expect = 3 if task_type != R.SPATIAL_AFFORDANCE_LOCALIZATION else 5
    if len(out) < n_expect and len(out) > 0:
        mx = sum(p[0] for p in out) / len(out)
        my = sum(p[1] for p in out) / len(out)
        j = 0.002
        while len(out) < n_expect:
            k = len(out)
            out.append([min(1 - eps, max(eps, mx + j * ((k % 3) - 1))), min(1 - eps, max(eps, my + j * ((k % 2) - 0.5)))])
    return out

#!/usr/bin/env python3
"""Fixed pipeline: Doubao descriptions -> LaMI -> boxed image -> Doubao points."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from PIL import ImageDraw

from agent_afford_harness.paths import load_env_overrides, outputs_dir
from agent_afford_harness.tools.lami_detr_tool import tool_lami_detr_grounder


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _is_generic_key(k: str) -> bool:
    g = {
        "target_object",
        "functional_part",
        "object",
        "part",
        "obj",
        "region",
        "entity",
        "item",
    }
    return k.strip().lower() in g


def _safe_semantic_key(k: str, fallback: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (k or "").strip().lower()).strip("_")
    if not s:
        s = fallback
    if s and s[0].isdigit():
        s = f"{fallback}_{s}"
    return s


def _semanticize_lami_classes(lami_classes: Dict[str, Any], question: str) -> Dict[str, List[str]]:
    # Keep only list-of-string descriptions and convert key names to semantic snake_case.
    parsed: Dict[str, List[str]] = {}
    for k, v in (lami_classes or {}).items():
        if not isinstance(v, list):
            continue
        descs = [str(x).strip() for x in v if str(x).strip()]
        if not descs:
            continue
        key = _safe_semantic_key(str(k), "semantic_object")
        if _is_generic_key(key):
            key = "semantic_object"
        parsed[key] = descs

    if parsed and any(k != "semantic_object" for k in parsed):
        return parsed

    # Fallback: build two semantic keys from question intent.
    q = question.lower()
    base = "target_item"
    if "mug" in q:
        base = "mug"
    elif "knife" in q:
        base = "knife"
    elif "chair" in q:
        base = "chair"
    elif "bottle" in q:
        base = "bottle"
    elif "cup" in q:
        base = "cup"

    obj_key = f"{base}_body"
    part_key = f"{base}_affordance_part"
    if parsed and "semantic_object" in parsed:
        return {obj_key: parsed["semantic_object"], part_key: [f"functional area for {base}"]}

    return {
        obj_key: [question[:80]],
        part_key: [f"functional region for: {question[:80]}"],
    }


def _bbox_px_to_norm(bbox: List[float], image_size: tuple[int, int]) -> List[float]:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [
        _clip01(x1 / max(1.0, float(w))),
        _clip01(y1 / max(1.0, float(h))),
        _clip01(x2 / max(1.0, float(w))),
        _clip01(y2 / max(1.0, float(h))),
    ]


def _bbox_norm_to_px(bbox: List[float], image_size: tuple[int, int]) -> List[float]:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 * w, y1 * h, x2 * w, y2 * h]


def _normalize_lami_result(lami_result: Dict[str, Any], image_size: tuple[int, int]) -> Dict[str, Any]:
    boxes = lami_result.get("boxes") or []
    norm_boxes: List[Dict[str, Any]] = []
    for b in boxes:
        bb = b.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            continue
        norm_boxes.append(
            {
                "bbox": _bbox_px_to_norm([float(x) for x in bb[:4]], image_size),
                "score": float(b.get("score", 0.0)),
                "class_name": str(b.get("class_name", "unknown")),
            }
        )
    return {
        "prompt_summary": lami_result.get("prompt_summary", ""),
        "names": lami_result.get("names", []),
        "visual_descs": lami_result.get("visual_descs", {}),
        "boxes": norm_boxes,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "coord_space": "normalized_xyxy",
        "backend": lami_result.get("backend", "lami"),
    }


def _parse_points_text(text: str, image_size: tuple[int, int]) -> List[List[float]]:
    w, h = image_size
    out_px: List[List[float]] = []

    # <point>x y</point>
    for m in re.finditer(r"<point>\s*([0-9\.\s,]+)\s*</point>", text or "", flags=re.I):
        nums = re.findall(r"-?\d+\.?\d*", m.group(1))
        if len(nums) >= 2:
            out_px.append([float(nums[0]), float(nums[1])])

    # [(x,y), ...]
    if not out_px:
        pairs = re.findall(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", text or "")
        for x, y in pairs:
            out_px.append([float(x), float(y)])

    # JSON-like [[x,y],...]
    if not out_px:
        m = re.search(r"\[\s*\[.*\]\s*\]", text or "", flags=re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            out_px.append([float(item[0]), float(item[1])])
            except Exception:
                pass

    # fallback: first even-numbered sequence as xy pairs
    if not out_px:
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", text or "")]
        for i in range(0, len(nums) - 1, 2):
            out_px.append([nums[i], nums[i + 1]])

    # Convert to normalized [0,1]
    out_norm: List[List[float]] = []
    for x, y in out_px[:10]:
        nx = _clip01(x / max(1.0, float(w)))
        ny = _clip01(y / max(1.0, float(h)))
        out_norm.append([nx, ny])
    return out_norm


def _parse_grouped_points_payload(
    points_payload: Any,
    coord: str,
    image_size: tuple[int, int],
) -> List[List[List[float]]]:
    w, h = image_size
    groups: List[List[List[float]]] = []
    if not isinstance(points_payload, list):
        return groups
    for grp in points_payload:
        if not isinstance(grp, list):
            continue
        one_group: List[List[float]] = []
        for p in grp:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x, y = float(p[0]), float(p[1])
            if coord == "normalized":
                one_group.append([_clip01(x), _clip01(y)])
            else:
                one_group.append([_clip01(x / max(1.0, float(w))), _clip01(y / max(1.0, float(h)))])
        if one_group:
            groups.append(one_group)
    return groups


def _require_api_client():
    from openai import OpenAI

    api_key = (
        os.environ.get("ARK_API_KEY")
        or os.environ.get("AGENT_HARNESS_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY/AGENT_HARNESS_API_KEY/OPENAI_API_KEY")
    base_url = os.environ.get("AGENT_HARNESS_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.environ.get("AGENT_HARNESS_API_MODEL", "doubao-seed-2-0-lite-260215")
    return OpenAI(api_key=api_key, base_url=base_url), model, base_url


def _stage1_make_lami_classes(
    client: Any,
    model: str,
    image: Image.Image,
    question: str,
) -> Dict[str, Any]:
    sys_prompt = (
        "You generate LaMI-DETR visual descriptions.\n"
        "Output STRICT JSON only with keys:\n"
        "{\"lami_classes\":{\"semantic_key_1\":[\"...\"],\"semantic_key_2\":[\"...\"]},\"reason\":\"...\"}\n"
        "CRITICAL lami_classes RULES:\n"
        "- lami_classes MUST be tightly relevant to the question intent.\n"
        "- Each key/description should represent the answer target or highly relevant evidence for the answer.\n"
        "- Do NOT include unrelated objects, background-only regions, or generic scene content.\n"
        "- If uncertain, prefer fewer but high-relevance classes.\n"
        "- lami_classes keys MUST be semantic, object-specific snake_case names.\n"
        "- DO NOT use generic keys like target_object, functional_part, object, part.\n"
        "- Good keys: mug_body, mug_handle, knife_blade, chair_backrest.\n"
        "- Bad keys: target_object, functional_part, object1, region.\n"
        "- Each description item must be short, concrete, and visually detectable.\n"
        "- Avoid abstract language and task restatement.\n"
        "- Focus on basic type, typical color/material, shape/form, and notable parts.\n"
        "- Do not include coordinates in lami_classes.\n"
        "- Prefer 2-5 phrases per class.\n"
        "- Include at least one whole-object key and one affordance-part key.\n"
        "GOOD phrase examples: \"ceramic mug body\", \"curved handle loop\", \"metal blade edge\".\n"
        "BAD phrase examples: \"the thing used by humans\", \"region that should be selected\".\n"
        "- Output JSON only, no markdown, no extra text."
    )
    user_prompt = f"Question: {question}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            },
        ],
    )
    raw = resp.choices[0].message.content or ""
    parsed = _extract_first_json(raw) or {}
    raw_lami_classes = parsed.get("lami_classes")
    if not isinstance(raw_lami_classes, dict):
        raw_lami_classes = {}
    lami_classes = _semanticize_lami_classes(raw_lami_classes, question)
    return {
        "conversation": {
            "system": sys_prompt,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": "<data_url_omitted>"}}]},
                {"role": "assistant", "content": raw},
            ],
        },
        "response": {
            "id": getattr(resp, "id", ""),
            "model": getattr(resp, "model", ""),
            "raw_text": raw,
            "parsed": parsed,
        },
        "raw_lami_classes": raw_lami_classes,
        "lami_classes": lami_classes,
    }


def _topk_from_lami_classes(lami_classes: Dict[str, Any]) -> int:
    if not isinstance(lami_classes, dict):
        return 3
    n = len([k for k in lami_classes.keys() if str(k).strip()])
    return max(1, n)


def _render_lami_boxes(
    image: Image.Image,
    lami_result: Dict[str, Any],
    sample_id: str,
) -> Dict[str, Any]:
    palette = ["red", "lime", "cyan", "yellow", "magenta", "orange", "blue", "white"]
    boxes = lami_result.get("boxes") or []
    class_to_color: Dict[str, str] = {}
    ann = image.convert("RGB").copy()
    draw = ImageDraw.Draw(ann)
    for b in boxes:
        cname = str(b.get("class_name", "unknown"))
        if cname not in class_to_color:
            class_to_color[cname] = palette[len(class_to_color) % len(palette)]
        color = class_to_color[cname]
        x1, y1, x2, y2 = _bbox_norm_to_px([float(v) for v in b.get("bbox", [0, 0, 0, 0])], image.size)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{cname}", fill=color)
    out_path = outputs_dir("debug_images") / f"{sample_id}_lami_boxes.jpg"
    ann.save(out_path, format="JPEG", quality=92)
    legend = [{"class_name": k, "color": v} for k, v in class_to_color.items()]
    return {"annotated_image_path": str(out_path), "color_legend": legend}


def _render_final_points(
    image: Image.Image,
    sample_id: str,
    final_points: List[List[float]],
    final_point_groups: List[List[List[float]]],
) -> Dict[str, Any]:
    palette = ["red", "lime", "cyan", "yellow", "magenta", "orange", "blue", "white"]
    w, h = image.size
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    if final_point_groups:
        for gi, grp in enumerate(final_point_groups):
            color = palette[gi % len(palette)]
            for p in grp:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = float(p[0]) * w
                y = float(p[1]) * h
                r = 4
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    else:
        color = "red"
        for p in final_points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            x = float(p[0]) * w
            y = float(p[1]) * h
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)

    out_path = outputs_dir("debug_images") / f"{sample_id}_final_points.jpg"
    canvas.save(out_path, format="JPEG", quality=92)
    return {"final_points_image_path": str(out_path)}


def _stage3_points_from_annotated_image(
    client: Any,
    model: str,
    question: str,
    annotated_image: Image.Image,
    image_size: tuple[int, int],
    color_legend: List[Dict[str, str]],
    lami_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sys_prompt = (
        "You are an affordance point predictor.\n"
        "You are given an image with colored bounding boxes, a color legend, and box metadata.\n"
        "Use the legend + boxes to infer affordance points.\n"
        "Return STRICT JSON only:\n"
        "{\"points\":[[[x,y],...],[[x,y],...]],\"coord\":\"pixel|normalized\",\"reason\":\"...\"}\n"
        "IMPORTANT: points must be grouped by object instance.\n"
        "- one object instance -> one inner list of points\n"
        "- multiple objects -> multiple inner lists\n"
        "- each inner list should contain multiple points when possible\n"
        "No markdown, no extra text."
    )
    payload = {
        "question": question,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "color_legend": color_legend,
        "boxes": lami_boxes,
    }
    user_text = (
        "The attached image is already annotated with bounding boxes.\n"
        "Color legend and exact box metadata are below (JSON). "
        "All box coordinates are normalized [0,1] XYXY:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(annotated_image)}},
                ],
            },
        ],
    )
    raw = resp.choices[0].message.content or ""
    parsed = _extract_first_json(raw) or {}
    coord = str(parsed.get("coord", "")).lower()
    pts = parsed.get("points")
    grouped_norm = _parse_grouped_points_payload(pts, coord, image_size)
    points_norm: List[List[float]] = [p for g in grouped_norm for p in g]
    if not points_norm:
        points_norm = _parse_points_text(raw, image_size)
        if points_norm:
            grouped_norm = [points_norm]
    return {
        "conversation": {
            "system": sys_prompt,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": "<data_url_omitted>"}}]},
                {"role": "assistant", "content": raw},
            ],
        },
        "response": {
            "id": getattr(resp, "id", ""),
            "model": getattr(resp, "model", ""),
            "raw_text": raw,
            "parsed": parsed,
        },
        "final_point_groups": grouped_norm,
        "final_points": points_norm,
    }


def _run_two_turn_conversation(
    client: Any,
    model: str,
    question: str,
    turn1_raw: str,
    turn1_parsed: Dict[str, Any],
    turn1_response_id: str,
    annotated_image: Image.Image,
    image_size: tuple[int, int],
    color_legend: List[Dict[str, str]],
    lami_boxes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    system_prompt = (
        "You solve affordance localization in two turns.\n"
        "Turn-1: output STRICT JSON with lami_classes:\n"
        "{\"lami_classes\":{\"semantic_key_1\":[\"...\"],\"semantic_key_2\":[\"...\"]},\"reason\":\"...\"}\n"
        "Turn-1 lami_classes must be answer-relevant: classes should be the answer target or highly relevant evidence.\n"
        "Do not include unrelated/background classes.\n"
        "Turn-1 keys must be semantic snake_case (e.g., mug_body, mug_handle), not generic labels.\n"
        "Turn-2: after I provide boxed-image metadata, output STRICT JSON:\n"
        "{\"points\":[[[x,y],...],[[x,y],...]],\"coord\":\"pixel|normalized\",\"reason\":\"...\"}\n"
        "In Turn-2, points must be grouped by object instance (one inner list per object).\n"
        "Each object should contain multiple points when possible.\n"
        "Rules: JSON only, no markdown, no extra text."
    )
    user1 = f"Turn-1. Question: {question}. Provide lami_classes for LaMI-DETR."
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    log_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    # Reconstruct turn-1 context in same session message history.
    log_messages.append({"role": "user", "content": [{"type": "text", "text": user1}, {"type": "image_url", "image_url": {"url": "<data_url_omitted>"}}]})
    log_messages.append({"role": "assistant", "content": turn1_raw})
    messages.append({"role": "user", "content": user1})
    messages.append({"role": "assistant", "content": turn1_raw})

    payload2 = {
        "question": question,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "color_legend": color_legend,
        "boxes": lami_boxes,
    }
    user2 = (
        "Turn-2. I already ran LaMI and drew colored boxes on the attached image. "
        "Use this legend and boxes to output final points only. "
        "All box coordinates in JSON are normalized [0,1] XYXY.\n"
        f"{json.dumps(payload2, ensure_ascii=False)}"
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user2},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(annotated_image)}},
            ],
        }
    )
    log_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user2},
                {"type": "image_url", "image_url": {"url": "<data_url_omitted>"}},
            ],
        }
    )

    r2 = client.chat.completions.create(model=model, temperature=0.0, messages=messages)
    raw2 = r2.choices[0].message.content or ""
    parsed2 = _extract_first_json(raw2) or {}
    log_messages.append({"role": "assistant", "content": raw2})

    coord = str(parsed2.get("coord", "")).lower()
    pts = parsed2.get("points")
    grouped_norm = _parse_grouped_points_payload(pts, coord, image_size)
    points_norm: List[List[float]] = [p for g in grouped_norm for p in g]
    if not points_norm:
        points_norm = _parse_points_text(raw2, image_size)
        if points_norm:
            grouped_norm = [points_norm]

    return {
        "messages": log_messages,
        "turn1": {"raw_text": turn1_raw, "parsed": turn1_parsed, "response_id": turn1_response_id},
        "turn2": {"raw_text": raw2, "parsed": parsed2, "response_id": getattr(r2, "id", "")},
        "final_point_groups": grouped_norm,
        "final_points": points_norm,
    }


def main() -> None:
    load_env_overrides()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--sample-id", default="fixed_doubao_lami")
    ap.add_argument("--trace-out", default=None)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    image = Image.open(args.image).convert("RGB")
    client, model, base_url = _require_api_client()

    trace: Dict[str, Any] = {
        "run_id": uuid.uuid4().hex[:12],
        "sample_id": args.sample_id,
        "image_path": str(Path(args.image).resolve()),
        "question": args.question,
        "pipeline": "fixed_doubao_lami",
        "stages": {},
        "metadata": {
            "api_model": model,
            "api_base_url": base_url,
            "grounding_backend": "lami",
        },
    }

    # Turn-1 (in unified dialogue) is for lami_classes only.
    s1 = _stage1_make_lami_classes(client, model, image, args.question)
    lami_prompt_info = {"lami_classes": s1["lami_classes"]}
    dynamic_topk = _topk_from_lami_classes(s1["lami_classes"])
    raw_lami_result = tool_lami_detr_grounder(image, lami_prompt_info, topk=dynamic_topk, backend="lami")
    lami_result = _normalize_lami_result(raw_lami_result, image.size)
    trace["stages"]["lami_grounding"] = {"input": lami_prompt_info, "output": lami_result}
    trace["metadata"]["lami_topk_used"] = dynamic_topk

    render_pack = _render_lami_boxes(image, lami_result, args.sample_id)
    trace["stages"]["boxed_image"] = render_pack
    ann_image = Image.open(render_pack["annotated_image_path"]).convert("RGB")

    convo = _run_two_turn_conversation(
        client=client,
        model=model,
        question=args.question,
        turn1_raw=str((s1.get("response") or {}).get("raw_text", "")),
        turn1_parsed=dict((s1.get("response") or {}).get("parsed") or {}),
        turn1_response_id=str((s1.get("response") or {}).get("id", "")),
        annotated_image=ann_image,
        image_size=image.size,
        color_legend=render_pack["color_legend"],
        lami_boxes=lami_result.get("boxes") or [],
    )
    trace["stages"]["full_conversation"] = convo["messages"]
    trace["stages"]["turn1_lami_classes_response"] = convo["turn1"]
    trace["stages"]["turn2_points_response"] = convo["turn2"]
    trace["final_point_groups"] = convo.get("final_point_groups") or []
    trace["final_points"] = convo["final_points"]
    trace["stages"]["final_points_image"] = _render_final_points(
        image=image,
        sample_id=args.sample_id,
        final_points=trace["final_points"],
        final_point_groups=trace["final_point_groups"],
    )
    trace["elapsed_s"] = time.time() - t0

    tpath = args.trace_out or str(outputs_dir("traces") / f"{args.sample_id}.json")
    Path(tpath).parent.mkdir(parents=True, exist_ok=True)
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    print(json.dumps({"final_points": trace["final_points"], "trace": tpath}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


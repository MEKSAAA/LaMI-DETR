#!/usr/bin/env python3
"""Batch inference for fixed Doubao->LaMI->Doubao pipeline from annotations json."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_afford_harness.data.eval_wrapper import calculate_accuracy
from agent_afford_harness.harness.run_fixed_doubao_lami import (
    _normalize_lami_result,
    _render_final_points,
    _render_lami_boxes,
    _require_api_client,
    _run_two_turn_conversation,
    _stage1_make_lami_classes,
    _topk_from_lami_classes,
)
from agent_afford_harness.paths import load_env_overrides, outputs_dir
from agent_afford_harness.tools.lami_detr_tool import tool_lami_detr_grounder


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_sample_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:06d}"


def _resolve_data_path(data_root: Path, rel_path: str, kind: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return p
    direct = (data_root / p).resolve()
    if direct.is_file():
        return direct

    # Common RoboAfford layout fallback:
    # - images under <root>/images/*.jpg
    # - masks under <root>/masks/*.png
    name_only = p.name
    if kind == "img":
        alt = (data_root / "images" / name_only).resolve()
        if alt.is_file():
            return alt
    if kind == "mask":
        alt = (data_root / "masks" / name_only).resolve()
        if alt.is_file():
            return alt

    return direct


def run_batch(
    annotations_file: Path,
    data_root: Path,
    trace_dir: Path,
    output_summary: Path,
    start: int,
    limit: int,
    topk: int,
    sample_prefix: str,
    skip_existing: bool,
    with_eval: bool,
) -> Dict[str, Any]:
    anns = _read_json(annotations_file)
    if not isinstance(anns, list):
        raise RuntimeError(f"annotations file must be a list JSON: {annotations_file}")

    total = len(anns)
    end = total if limit <= 0 else min(total, start + limit)
    client, model, base_url = _require_api_client()

    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    ok_count = 0

    for idx in range(start, end):
        row = anns[idx]
        if not isinstance(row, dict):
            summary_rows.append({"index": idx, "ok": False, "error": "row is not dict"})
            continue
        question = str(row.get("question", "")).strip()
        img_rel = str(row.get("img", "")).strip()
        mask_rel = str(row.get("mask", "")).strip()
        category = str(row.get("category", "")).strip()
        sample_id = _safe_sample_id(sample_prefix, idx)
        trace_path = trace_dir / f"{sample_id}.json"

        if skip_existing and trace_path.is_file():
            summary_rows.append(
                {
                    "index": idx,
                    "sample_id": sample_id,
                    "ok": True,
                    "skipped": True,
                    "trace": str(trace_path),
                }
            )
            continue

        image_path = _resolve_data_path(data_root, img_rel, kind="img")
        mask_path = _resolve_data_path(data_root, mask_rel, kind="mask") if mask_rel else None
        if not question or not image_path.is_file():
            summary_rows.append(
                {
                    "index": idx,
                    "sample_id": sample_id,
                    "ok": False,
                    "error": f"invalid row or missing image: {image_path}",
                }
            )
            continue

        t0 = time.time()
        try:
            image = Image.open(image_path).convert("RGB")
            trace: Dict[str, Any] = {
                "run_id": uuid.uuid4().hex[:12],
                "sample_id": sample_id,
                "index": idx,
                "image_path": str(image_path),
                "question": question,
                "category": category,
                "pipeline": "fixed_doubao_lami_batch",
                "stages": {},
                "metadata": {
                    "api_model": model,
                    "api_base_url": base_url,
                    "grounding_backend": "lami",
                    "gt_mask_path": str(mask_path) if mask_path else "",
                },
            }

            s1 = _stage1_make_lami_classes(client, model, image, question)
            lami_prompt_info = {"lami_classes": s1["lami_classes"]}
            dynamic_topk = _topk_from_lami_classes(s1["lami_classes"])
            raw_lami_result = tool_lami_detr_grounder(image, lami_prompt_info, topk=dynamic_topk, backend="lami")
            lami_result = _normalize_lami_result(raw_lami_result, image.size)
            trace["stages"]["lami_grounding"] = {"input": lami_prompt_info, "output": lami_result}
            trace["metadata"]["lami_topk_used"] = dynamic_topk

            render_pack = _render_lami_boxes(image, lami_result, sample_id)
            trace["stages"]["boxed_image"] = render_pack
            ann_image = Image.open(render_pack["annotated_image_path"]).convert("RGB")

            convo = _run_two_turn_conversation(
                client=client,
                model=model,
                question=question,
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
                sample_id=sample_id,
                final_points=trace["final_points"],
                final_point_groups=trace["final_point_groups"],
            )
            trace["elapsed_s"] = time.time() - t0

            if with_eval and mask_path and mask_path.is_file():
                trace["reward"] = calculate_accuracy(trace["final_points"], str(mask_path))

            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

            one = {
                "index": idx,
                "sample_id": sample_id,
                "ok": True,
                "trace": str(trace_path),
                "num_points": len(trace["final_points"]),
                "elapsed_s": trace["elapsed_s"],
            }
            if "reward" in trace:
                one["reward"] = trace["reward"]
            summary_rows.append(one)
            ok_count += 1
        except Exception as e:
            summary_rows.append(
                {
                    "index": idx,
                    "sample_id": sample_id,
                    "ok": False,
                    "error": str(e),
                }
            )

    result = {
        "annotations_file": str(annotations_file),
        "data_root": str(data_root),
        "trace_dir": str(trace_dir),
        "start": start,
        "limit": limit,
        "processed": len(summary_rows),
        "ok": ok_count,
        "rows": summary_rows,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    load_env_overrides()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotations-file",
        default="/data9/data/miaojw/projects26/RoboAfford/annotations_absxy.json",
        help="RoboAfford annotations json",
    )
    ap.add_argument(
        "--data-root",
        default="/data9/data/miaojw/projects26/RoboAfford",
        help="Data root used to resolve img/mask relative paths",
    )
    ap.add_argument(
        "--trace-dir",
        default=str(outputs_dir("traces") / "batch_fixed_doubao_lami"),
        help="Directory to write per-sample trace files",
    )
    ap.add_argument(
        "--output-summary",
        default=str(outputs_dir("predictions") / "batch_fixed_doubao_lami_summary.json"),
        help="Summary json path",
    )
    ap.add_argument("--start", type=int, default=0, help="Start index in annotations")
    ap.add_argument("--limit", type=int, default=0, help="How many rows to run; <=0 means all")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--sample-prefix", default="batch_fixed")
    ap.add_argument("--no-skip-existing", action="store_true", help="Do not skip existing trace files")
    ap.add_argument("--no-eval", action="store_true", help="Disable per-trace reward calculation")
    args = ap.parse_args()

    result = run_batch(
        annotations_file=Path(args.annotations_file),
        data_root=Path(args.data_root),
        trace_dir=Path(args.trace_dir),
        output_summary=Path(args.output_summary),
        start=max(0, int(args.start)),
        limit=int(args.limit),
        topk=max(1, int(args.topk)),
        sample_prefix=str(args.sample_prefix),
        skip_existing=not bool(args.no_skip_existing),
        with_eval=not bool(args.no_eval),
    )
    print(
        json.dumps(
            {
                "summary": str(args.output_summary),
                "processed": result["processed"],
                "ok": result["ok"],
                "trace_dir": str(args.trace_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


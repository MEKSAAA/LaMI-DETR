"""Orchestrator: skill selection + tool planning + aggregation."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agent_afford_harness.agent import router as R
from agent_afford_harness.agent.aggregator import aggregate_part, aggregate_recognition, aggregate_spatial
from agent_afford_harness.agent.library import build_skill_library, build_tool_library
from agent_afford_harness.agent.llm_reasoner import LLMReasoner, load_reasoner_config
from agent_afford_harness.agent.state import HarnessTrace
from agent_afford_harness.agent.task_parser import normalize_category_label
from agent_afford_harness.config_load import load_pipeline_config
from agent_afford_harness.tools.lami_detr_tool import LaMIDetrGrounder
from agent_afford_harness.utils.coord_1000 import points_norm_to_1000


_REASONER: Optional[LLMReasoner] = None


def _get_reasoner(cfg: Dict[str, Any]) -> Optional[LLMReasoner]:
    global _REASONER
    rcfg = load_reasoner_config(cfg)
    if (rcfg.mode or "rules").lower() == "rules":
        return None
    if _REASONER is None or _REASONER.cfg.mode != rcfg.mode:
        _REASONER = LLMReasoner(rcfg)
    return _REASONER


def map_benchmark_category(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    c = normalize_category_label(category).lower()
    if "recognition" in c or "reference" in c:
        return R.OBJECT_AFFORDANCE_RECOGNITION
    if "spatial" in c:
        return R.SPATIAL_AFFORDANCE_LOCALIZATION
    return R.OBJECT_AFFORDANCE_PREDICTION


def _boxes_to_crop_space(
    boxes: List[Dict[str, Any]],
    crop_meta: Dict[str, Any],
) -> List[List[float]]:
    ox, oy = crop_meta["origin_xy"]
    cw = crop_meta["crop_width"]
    ch = crop_meta["crop_height"]
    out: List[List[float]] = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        out.append(
            [
                x1 - ox,
                y1 - oy,
                x2 - ox,
                y2 - oy,
            ]
        )
    return out


def run_pipeline(
    image: Image.Image,
    question: str,
    sample_id: str,
    *,
    image_path: str = "",
    benchmark_category: Optional[str] = None,
    grounder: Optional[LaMIDetrGrounder] = None,
    pipeline_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[float]], HarnessTrace]:
    cfg = pipeline_cfg or load_pipeline_config()
    orch = cfg.get("orchestrator") or {}
    nms_iou = float(orch.get("nms_iou", 0.7))
    score_th = float(orch.get("score_threshold", 0.25))
    topk = int(orch.get("lami_topk", 8))
    grounding_backend = str(
        os.environ.get("AGENT_HARNESS_GROUNDING_BACKEND", orch.get("grounding_backend", "lami"))
    )

    trace = HarnessTrace(sample_id=sample_id, image_path=image_path or "", question=question)
    trace.metadata["pipeline_cfg"] = {"grounding_backend": grounding_backend}
    skill_lib = build_skill_library()
    tool_lib = build_tool_library()
    trace.metadata["library"] = {
        "skills": skill_lib.names(),
        "tools": tool_lib.names(),
    }
    llm_plan: Dict[str, Any] = {}
    llm_error: Optional[str] = None
    reasoner = _get_reasoner(cfg)
    if reasoner is not None:
        try:
            llm_plan = reasoner.plan(question, image)
            trace.metadata["llm_plan"] = llm_plan
            trace.metadata["llm_dialog"] = reasoner.get_last_trace()
        except Exception as e:
            llm_error = str(e)
            trace.metadata["llm_plan_error"] = llm_error
            dialog = reasoner.get_last_trace()
            if dialog:
                dialog["error"] = llm_error
                trace.metadata["llm_dialog"] = dialog

    # Skill 1: task type
    mapped = map_benchmark_category(benchmark_category)
    if mapped:
        task_type = mapped
        task_info = {
            "task_type": task_type,
            "confidence": 0.99,
            "reason": "benchmark category",
        }
    else:
        llm_task = llm_plan.get("task_type")
        if llm_task in (
            R.OBJECT_AFFORDANCE_RECOGNITION,
            R.OBJECT_AFFORDANCE_PREDICTION,
            R.SPATIAL_AFFORDANCE_LOCALIZATION,
        ):
            task_type = llm_task
            task_info = {
                "task_type": llm_task,
                "confidence": float(llm_plan.get("confidence", 0.7)),
                "reason": llm_plan.get("reason", "llm planner"),
            }
            trace.reasoning_summary = str(llm_plan.get("reasoning_summary", ""))
        else:
            task_info = skill_lib.call("task_type_classification", question)
            task_type = task_info["task_type"]
    trace.task_type = task_type
    trace.add_skill("task_type_classification")
    trace.metadata["task_info"] = task_info
    trace.add_execution_step(phase="plan", action="task_type_classification", details=task_info)

    # Skill: search
    search_info = skill_lib.call("search_triggering", question, task_info)
    if isinstance(llm_plan.get("need_search"), bool):
        search_info["need_search"] = bool(llm_plan.get("need_search"))
        qs = llm_plan.get("search_queries")
        if isinstance(qs, list) and qs:
            search_info["queries"] = [str(x) for x in qs if str(x).strip()][:3]
    trace.add_skill("search_triggering")
    trace.add_execution_step(phase="plan", action="search_triggering", details=search_info)
    search_result = None
    if search_info.get("need_search"):
        t0 = time.time()
        search_result = tool_lib.call("web_search_tool", search_info.get("queries") or [])
        trace.add_tool_call("web_search_tool", {"queries": search_info.get("queries")}, search_result, time.time() - t0)
        trace.search_evidence = search_result

    # Skill: LaMI prompt
    prompt_info = skill_lib.call("lami_prompt_engineering", question, task_info, search_result)
    pov = llm_plan.get("prompt_overrides")
    if isinstance(pov, dict):
        for k in ("object_prompt", "part_prompt", "spatial_query", "negative_hint"):
            if pov.get(k):
                prompt_info[k] = str(pov[k])
        if isinstance(pov.get("lami_classes"), dict):
            prompt_info["lami_classes"] = pov["lami_classes"]
    trace.engineered_prompt = prompt_info
    trace.add_skill("lami_prompt_engineering")
    trace.add_execution_step(phase="plan", action="lami_prompt_engineering", details=prompt_info)

    W, H = image.size
    g = grounder or LaMIDetrGrounder()
    ctx: Dict[str, Any] = {
        "crop_image": image,
        "crop_meta": None,
        "lami_result": {},
        "zoom_info": {},
        "code_skill": {},
        "code_result": {},
    }

    pending_actions: List[str] = ["run_lami_grounding"]
    max_steps = int(orch.get("max_agent_steps", 8))
    steps = 0
    done = False
    while pending_actions and not done and steps < max_steps:
        action = pending_actions.pop(0)
        steps += 1
        trace.add_execution_step(phase="act", action=action, details={"step": steps})

        if action == "run_lami_grounding":
            t0 = time.time()
            lami_result = tool_lib.call(
                "lami_detr_grounder",
                image,
                prompt_info,
                nms_iou=nms_iou,
                score_threshold=score_th,
                topk=topk,
                backend=grounding_backend,
                grounder=g,
            )
            ctx["lami_result"] = lami_result
            trace.add_tool_call(
                "lami_detr_grounder",
                {"lami_classes": prompt_info.get("lami_classes"), "prompt_summary": lami_result.get("prompt_summary")},
                lami_result,
                time.time() - t0,
            )
            if task_type == R.OBJECT_AFFORDANCE_RECOGNITION:
                pending_actions.append("finalize_recognition")
            else:
                pending_actions.append("plan_zoom")
            continue

        if action == "plan_zoom":
            trace.add_skill("zoom_before_detail")
            zoom_info = skill_lib.call(
                "zoom_before_detail",
                task_type,
                question,
                ctx["lami_result"].get("boxes") or [],
                image_size=(W, H),
            )
            if isinstance(llm_plan.get("need_zoom"), bool):
                zoom_info["need_zoom"] = bool(llm_plan.get("need_zoom"))
            ctx["zoom_info"] = zoom_info
            trace.add_execution_step(phase="decide", action="plan_zoom", details=zoom_info)
            if zoom_info.get("need_zoom") and (ctx["lami_result"].get("boxes") or []):
                pending_actions.append("run_zoom")
            pending_actions.append("plan_code_analysis")
            continue

        if action == "run_zoom":
            idx = int(ctx["zoom_info"].get("crop_target_index", 0))
            idx = min(idx, len(ctx["lami_result"]["boxes"]) - 1)
            bbox = ctx["lami_result"]["boxes"][idx]["bbox"]
            t0 = time.time()
            crop_pack = tool_lib.call(
                "zoom_crop_tool",
                image,
                bbox,
                padding_ratio=float(ctx["zoom_info"].get("padding_ratio", 0.15)),
                max_side=int(ctx["zoom_info"].get("resize_max_side", 1024)),
            )
            ctx["crop_meta"] = {k: v for k, v in crop_pack.items() if k != "crop"}
            ctx["crop_image"] = crop_pack["crop"]
            trace.add_tool_call("zoom_crop_tool", {"bbox": bbox, **ctx["zoom_info"]}, crop_pack, time.time() - t0)
            continue

        if action == "plan_code_analysis":
            trace.add_skill("code_when_geometry_matters")
            code_skill = skill_lib.call("code_when_geometry_matters", task_type, question, {})
            if llm_plan.get("analysis_type"):
                code_skill["analysis_type"] = str(llm_plan["analysis_type"])
                code_skill["need_code_analysis"] = True
            if task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION:
                code_skill["need_code_analysis"] = True
                code_skill["analysis_type"] = "free_space_between_boxes"
            ctx["code_skill"] = code_skill
            trace.add_execution_step(phase="decide", action="plan_code_analysis", details=code_skill)
            if code_skill.get("need_code_analysis"):
                pending_actions.append("run_code_analysis")
            if task_type == R.OBJECT_AFFORDANCE_PREDICTION:
                pending_actions.append("finalize_part")
            else:
                pending_actions.append("finalize_spatial")
            continue

        if action == "run_code_analysis":
            lami_boxes = ctx["lami_result"].get("boxes") or []
            cand = None
            if ctx["crop_meta"] and lami_boxes:
                cand = _boxes_to_crop_space(lami_boxes[:2], ctx["crop_meta"])
            elif task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION:
                cand = [list(map(float, b["bbox"])) for b in lami_boxes[:2]]
                while len(cand) < 2:
                    cw, ch = ctx["crop_image"].size
                    cand.append([cw * 0.25, ch * 0.25, cw * 0.35, ch * 0.75])
            n_points = 5 if task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION else 3
            t0 = time.time()
            code_result = tool_lib.call(
                "code_visual_analyzer",
                ctx["crop_image"],
                ctx["code_skill"].get("analysis_type", "interior_region"),
                candidate_boxes_xyxy=cand,
                crop_meta=ctx["crop_meta"],
                full_image_size=(W, H),
                n_points=n_points,
            )
            ctx["code_result"] = code_result
            trace.add_tool_call("code_visual_analyzer", ctx["code_skill"], code_result, time.time() - t0)
            continue

        if action == "finalize_recognition":
            trace.add_skill("point_output_normalization")
            raw_pts = aggregate_recognition((W, H), ctx["lami_result"])
            final_list = skill_lib.call("point_output_normalization", raw_pts, W, H, task_type)
            trace.final_points = final_list
            trace.metadata["final_points_1000"] = points_norm_to_1000(final_list)
            trace.reasoning_summary = "recognition: spread points from top LaMI box"
            done = True
            continue

        if action == "finalize_part":
            trace.add_skill("point_output_normalization")
            raw_pts = aggregate_part((W, H), ctx["crop_meta"], ctx["code_result"], ctx["lami_result"])
            final_list = skill_lib.call("point_output_normalization", raw_pts, W, H, task_type)
            trace.final_points = final_list
            trace.metadata["final_points_1000"] = points_norm_to_1000(final_list)
            trace.reasoning_summary = "part: LaMI object + zoom + template code"
            done = True
            continue

        if action == "finalize_spatial":
            trace.add_skill("point_output_normalization")
            raw_pts = aggregate_spatial((W, H), ctx["code_result"], ctx["lami_result"])
            final_list = skill_lib.call("point_output_normalization", raw_pts, W, H, task_type)
            trace.final_points = final_list
            trace.metadata["final_points_1000"] = points_norm_to_1000(final_list)
            if not trace.reasoning_summary:
                trace.reasoning_summary = "spatial: free-space template between LaMI boxes"
            done = True
            continue

    if not done:
        trace.failure_tags.append("agent_step_limit_or_empty_plan")
        trace.add_execution_step(
            phase="decide",
            action="fallback_finalize",
            details={"steps": steps, "pending_actions": pending_actions},
        )
        raw_pts = [(W * 0.5, H * 0.5)]
        trace.final_points = skill_lib.call("point_output_normalization", raw_pts, W, H, task_type)
        trace.metadata["final_points_1000"] = points_norm_to_1000(trace.final_points)
        if not trace.reasoning_summary:
            trace.reasoning_summary = "fallback: default center point after incomplete action loop"

    if llm_error and not trace.reasoning_summary:
        trace.reasoning_summary = f"llm planner fallback: {llm_error}"
    return trace.final_points, trace

"""Orchestrator: LLM plan -> validation -> step execution -> fallback."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agent_afford_harness.agent import router as R
from agent_afford_harness.agent.execution_state import ExecutionState
from agent_afford_harness.agent.library import build_skill_library, build_tool_library
from agent_afford_harness.agent.library_context import build_library_context, tool_required_args_hint
from agent_afford_harness.agent.llm_reasoner import LLMReasoner, load_reasoner_config
from agent_afford_harness.agent.plan_executor import execute_plan
from agent_afford_harness.agent.plan_schema import Plan, plan_from_dict
from agent_afford_harness.agent.plan_validator import validate_plan
from agent_afford_harness.agent.skills_runtime import (
    skill_code_when_geometry_matters,
    skill_lami_prompt_engineering,
    skill_search_triggering,
    skill_task_type_classification,
    skill_zoom_before_detail,
)
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


def _build_rule_plan(question: str, benchmark_category: Optional[str], image_size: Tuple[int, int]) -> Dict[str, Any]:
    mapped = map_benchmark_category(benchmark_category)
    task_info = {"task_type": mapped, "confidence": 0.99, "reason": "benchmark category"} if mapped else skill_task_type_classification(question)
    task_type = task_info["task_type"]
    search_info = skill_search_triggering(question, task_info)
    prompt_info = skill_lami_prompt_engineering(question, task_info, None)
    zoom_info = skill_zoom_before_detail(task_type, question, [], image_size=image_size)
    code_info = skill_code_when_geometry_matters(task_type, question, {})
    if task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION:
        code_info["need_code_analysis"] = True
        code_info["analysis_type"] = "free_space_between_boxes"

    selected_skills = [
        {"name": "task_type_classification", "reason": "baseline task routing"},
        {"name": "lami_prompt_engineering", "reason": "build grounding prompt"},
        {"name": "point_output_normalization", "reason": "format benchmark output"},
    ]
    selected_tools = [{"name": "lami_detr_grounder", "reason": "obtain candidates"}]
    steps: List[Dict[str, Any]] = [
        {
            "step_id": 1,
            "kind": "tool",
            "name": "lami_detr_grounder",
            "args": {"prompt_info": prompt_info},
            "expected_observation": "candidate detection boxes",
        }
    ]
    sid = 2
    if search_info.get("need_search"):
        selected_skills.append({"name": "search_triggering", "reason": "long-tail knowledge"})
        selected_tools.append({"name": "web_search_tool", "reason": "fetch concept cues"})
        steps.insert(
            0,
            {
                "step_id": sid,
                "kind": "tool",
                "name": "web_search_tool",
                "args": {"queries": search_info.get("queries") or [question]},
                "expected_observation": "external visual cues",
            },
        )
        sid += 1
    if zoom_info.get("need_zoom"):
        selected_skills.append({"name": "zoom_before_detail", "reason": "small/fine target"})
        selected_tools.append({"name": "zoom_crop_tool", "reason": "focus on ROI"})
        steps.append(
            {
                "step_id": sid,
                "kind": "tool",
                "name": "zoom_crop_tool",
                "args": {"padding_ratio": zoom_info.get("padding_ratio", 0.15), "max_side": zoom_info.get("resize_max_side", 1024)},
                "expected_observation": "crop and crop metadata",
            }
        )
        sid += 1
    if code_info.get("need_code_analysis"):
        selected_skills.append({"name": "code_when_geometry_matters", "reason": "geometry verification"})
        selected_tools.append({"name": "code_visual_analyzer", "reason": "deterministic point proposal"})
        steps.append(
            {
                "step_id": sid,
                "kind": "tool",
                "name": "code_visual_analyzer",
                "args": {"analysis_type": code_info.get("analysis_type", "interior_region")},
                "expected_observation": "point proposal",
            }
        )

    return {
        "task_understanding": f"Rule planner for task type {task_type}",
        "task_type": task_type,
        "selected_skills": selected_skills,
        "selected_tools": selected_tools,
        "steps": steps,
        "final_strategy": {"type": "aggregate_point", "reason": "aggregate tool observations into benchmark points"},
        "fallback_policy": {"use_rules_if_invalid_plan": True},
    }


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
    grounding_backend = str(os.environ.get("AGENT_HARNESS_GROUNDING_BACKEND", orch.get("grounding_backend", "lami")))

    trace = HarnessTrace(sample_id=sample_id, image_path=image_path or "", question=question)
    skill_lib = build_skill_library()
    tool_lib = build_tool_library()
    library_ctx = build_library_context()
    trace.metadata["library"] = {"skills": skill_lib.names(), "tools": tool_lib.names()}

    state = ExecutionState(sample_id=sample_id, question=question, image=image, image_path=image_path)
    state.planner_input = {
        "question": question,
        "benchmark_category": benchmark_category,
        "library": library_ctx,
        "output_requirement": "normalized points in [0,1]",
    }
    trace.add_execution_step(phase="plan", action="planner_input", details={"question": question, "library_names": trace.metadata["library"]})

    reasoner = _get_reasoner(cfg)
    llm_plan_raw: Dict[str, Any] = {}
    llm_error: Optional[str] = None
    if reasoner is not None:
        try:
            llm_plan_raw = reasoner.plan(question, image, planner_context=state.planner_input)
            trace.metadata["llm_plan"] = llm_plan_raw
            trace.metadata["llm_dialog"] = reasoner.get_last_trace()
        except Exception as e:
            llm_error = str(e)
            trace.metadata["llm_plan_error"] = llm_error

    if not llm_plan_raw:
        llm_plan_raw = _build_rule_plan(question, benchmark_category, image.size)
        trace.metadata["plan_source"] = "rules_fallback"
    else:
        trace.metadata["plan_source"] = "llm"

    state.planner_plan_raw = llm_plan_raw
    plan: Plan = plan_from_dict(llm_plan_raw)
    validation = validate_plan(
        plan,
        valid_skills=skill_lib.names(),
        valid_tools=tool_lib.names(),
        tool_schemas=tool_required_args_hint(),
    )
    trace.metadata["plan_validation"] = {"ok": validation.ok, "errors": validation.errors, "warnings": validation.warnings}
    trace.add_execution_step(phase="decide", action="validate_plan", details=trace.metadata["plan_validation"])

    if not validation.ok:
        trace.failure_tags.append("invalid_plan")
        if not plan.fallback_policy.get("use_rules_if_invalid_plan", True):
            trace.failure_tags.append("fallback_disabled")
        fallback_raw = _build_rule_plan(question, benchmark_category, image.size)
        plan = plan_from_dict(fallback_raw)
        trace.metadata["plan_source"] = "rules_after_validation_failure"

    state.plan = llm_plan_raw
    trace.task_type = plan.task_type
    for c in plan.selected_skills:
        if c.name:
            trace.add_skill(c.name)
    for c in plan.selected_tools:
        if c.name:
            trace.add_tool_name(c.name)

    final_points, final_meta = execute_plan(
        plan,
        state,
        tool_lib=tool_lib,
        trace=trace,
        image=image,
        grounder=grounder or LaMIDetrGrounder(),
        nms_iou=nms_iou,
        score_th=score_th,
        topk=topk,
        grounding_backend=grounding_backend,
    )
    trace.final_points = final_points
    trace.metadata["execution_state"] = {
        "planner_input": state.planner_input,
        "plan": state.planner_plan_raw,
        "tool_inputs": state.tool_inputs,
        "tool_outputs": state.tool_outputs,
        "errors": state.errors,
        "fallback_info": state.fallback_info,
    }
    trace.metadata["final_points_1000"] = points_norm_to_1000(final_points)
    trace.metadata["final_strategy"] = final_meta
    if llm_error:
        trace.metadata["llm_plan_error"] = llm_error
    trace.reasoning_summary = plan.task_understanding or final_meta.get("strategy", "aggregate_point")
    return final_points, trace

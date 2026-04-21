"""Execute validated planner steps and finalize benchmark outputs."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from PIL import Image

from agent_afford_harness.agent.aggregator import aggregate_part, aggregate_recognition, aggregate_spatial
from agent_afford_harness.agent.execution_state import ExecutionState
from agent_afford_harness.agent.plan_schema import Plan
from agent_afford_harness.agent.skills_runtime import skill_point_output_normalization
from agent_afford_harness.agent import router as R
from agent_afford_harness.tools.lami_detr_tool import LaMIDetrGrounder


def _resolve_step_args(step_args: Dict[str, Any], state: ExecutionState, image_size: Tuple[int, int]) -> Dict[str, Any]:
    args = dict(step_args)
    if "from_step" in args:
        ref = str(args["from_step"])
        args["source_observation"] = state.observations.get(ref)
    return args


def execute_plan(
    plan: Plan,
    state: ExecutionState,
    *,
    tool_lib: Any,
    trace: Any,
    image: Image.Image,
    grounder: LaMIDetrGrounder | None,
    nms_iou: float,
    score_th: float,
    topk: int,
    grounding_backend: str,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    W, H = image.size
    context: Dict[str, Any] = {"crop_meta": None, "crop_image": image, "lami_result": {}, "code_result": {}}

    for step in plan.steps:
        step_key = f"step_{step.step_id}"
        args = _resolve_step_args(step.args, state, (W, H))
        trace.add_execution_step(phase="act", action=step_key, details={"tool": step.name, "args": args})
        t0 = time.time()
        if step.name == "lami_detr_grounder":
            prompt_info = args.get("prompt_info") or state.observations.get("prompt_info") or {"object_prompt": state.question}
            out = tool_lib.call(
                "lami_detr_grounder",
                image,
                prompt_info,
                nms_iou=nms_iou,
                score_threshold=score_th,
                topk=topk,
                backend=grounding_backend,
                grounder=grounder,
            )
            context["lami_result"] = out
        elif step.name == "zoom_crop_tool":
            lami_boxes = (context.get("lami_result") or {}).get("boxes") or []
            if args.get("bbox"):
                bbox = args["bbox"]
            elif lami_boxes:
                bbox = lami_boxes[0]["bbox"]
            else:
                bbox = [0, 0, W, H]
            out = tool_lib.call(
                "zoom_crop_tool",
                image,
                bbox,
                padding_ratio=float(args.get("padding_ratio", 0.15)),
                max_side=int(args.get("max_side", 1024)),
            )
            context["crop_meta"] = {k: v for k, v in out.items() if k != "crop"}
            context["crop_image"] = out["crop"]
        elif step.name == "code_visual_analyzer":
            out = tool_lib.call(
                "code_visual_analyzer",
                context["crop_image"],
                args.get("analysis_type", "interior_region"),
                candidate_boxes_xyxy=args.get("candidate_boxes_xyxy"),
                crop_meta=context.get("crop_meta"),
                full_image_size=(W, H),
                n_points=int(args.get("n_points", 3)),
            )
            context["code_result"] = out
        elif step.name == "web_search_tool":
            out = tool_lib.call("web_search_tool", args.get("queries") or [state.question])
        else:
            out = {"error": f"unsupported tool: {step.name}"}
        state.tool_inputs[step_key] = args
        state.tool_outputs[step_key] = out
        state.observations[step_key] = out
        trace.add_tool_call(step.name, args, out, time.time() - t0)

    task_type = plan.task_type or R.OBJECT_AFFORDANCE_PREDICTION
    strategy = (plan.final_strategy or {}).get("type", "aggregate_point")
    if strategy in ("aggregate_point", "classify_affordance"):
        if task_type == R.OBJECT_AFFORDANCE_RECOGNITION:
            raw = aggregate_recognition((W, H), context.get("lami_result") or {})
        elif task_type == R.SPATIAL_AFFORDANCE_LOCALIZATION:
            raw = aggregate_spatial((W, H), context.get("code_result") or {}, context.get("lami_result") or {})
        else:
            raw = aggregate_part((W, H), context.get("crop_meta"), context.get("code_result") or {}, context.get("lami_result") or {})
        final = skill_point_output_normalization(raw, W, H, task_type)
        return final, {"task_type": task_type, "strategy": strategy}
    return skill_point_output_normalization([(W * 0.5, H * 0.5)], W, H, task_type), {"task_type": task_type, "strategy": "fallback"}

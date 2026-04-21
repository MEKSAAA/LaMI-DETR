"""Validate planner plans against library availability and structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent_afford_harness.agent.plan_schema import Plan


@dataclass
class PlanValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_plan(
    plan: Plan,
    *,
    valid_skills: List[str],
    valid_tools: List[str],
    tool_schemas: Dict[str, Dict[str, Any]] | None = None,
) -> PlanValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    if not plan.task_type:
        errors.append("missing task_type")
    if not plan.steps:
        errors.append("empty steps")
    for c in plan.selected_skills:
        if c.name and c.name not in valid_skills:
            errors.append(f"unknown skill: {c.name}")
    for c in plan.selected_tools:
        if c.name and c.name not in valid_tools:
            errors.append(f"unknown tool: {c.name}")
    for i, step in enumerate(plan.steps):
        if step.kind != "tool":
            errors.append(f"step[{i}] unsupported kind: {step.kind}")
            continue
        if step.name not in valid_tools:
            errors.append(f"step[{i}] unknown tool: {step.name}")
        if not isinstance(step.args, dict):
            errors.append(f"step[{i}] args must be dict")
        schema = (tool_schemas or {}).get(step.name) or {}
        required = [str(k) for k in (schema.get("required_args") or [])]
        for key in required:
            if key not in step.args:
                warnings.append(f"step[{i}] missing recommended arg: {key}")
    if not isinstance(plan.final_strategy, dict) or not plan.final_strategy.get("type"):
        errors.append("missing final_strategy.type")
    return PlanValidationResult(ok=not errors, errors=errors, warnings=warnings)

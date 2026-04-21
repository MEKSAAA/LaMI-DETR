"""Schema helpers for planner-generated execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PlanChoice:
    name: str
    reason: str = ""


@dataclass
class PlanStep:
    step_id: int
    kind: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    expected_observation: str = ""


@dataclass
class Plan:
    task_understanding: str
    task_type: str
    selected_skills: List[PlanChoice]
    selected_tools: List[PlanChoice]
    steps: List[PlanStep]
    final_strategy: Dict[str, Any]
    fallback_policy: Dict[str, Any]


def plan_from_dict(blob: Dict[str, Any]) -> Plan:
    skills = [PlanChoice(name=str(x.get("name", "")), reason=str(x.get("reason", ""))) for x in (blob.get("selected_skills") or [])]
    tools = [PlanChoice(name=str(x.get("name", "")), reason=str(x.get("reason", ""))) for x in (blob.get("selected_tools") or [])]
    steps: List[PlanStep] = []
    for i, s in enumerate(blob.get("steps") or []):
        steps.append(
            PlanStep(
                step_id=int(s.get("step_id", i + 1)),
                kind=str(s.get("kind", "tool")),
                name=str(s.get("name", "")),
                args=dict(s.get("args") or {}),
                expected_observation=str(s.get("expected_observation", "")),
            )
        )
    return Plan(
        task_understanding=str(blob.get("task_understanding", "")),
        task_type=str(blob.get("task_type", "")),
        selected_skills=skills,
        selected_tools=tools,
        steps=steps,
        final_strategy=dict(blob.get("final_strategy") or {}),
        fallback_policy=dict(blob.get("fallback_policy") or {}),
    )

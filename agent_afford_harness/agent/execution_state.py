"""Runtime state for plan-driven harness execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from PIL import Image


@dataclass
class ExecutionState:
    sample_id: str
    question: str
    image: Image.Image
    image_path: str = ""
    planner_input: Dict[str, Any] = field(default_factory=dict)
    planner_plan_raw: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    selected_skills: List[str] = field(default_factory=list)
    selected_tools: List[str] = field(default_factory=list)
    observations: Dict[str, Any] = field(default_factory=dict)
    tool_inputs: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    fallback_info: Dict[str, Any] = field(default_factory=dict)
    final_result: Dict[str, Any] = field(default_factory=dict)

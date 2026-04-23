"""Trace and run state (future self-evolve hooks)."""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HarnessTrace:
    """Structured run log; keep fields stable for future library refinement."""

    sample_id: str
    image_path: str
    question: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_type: str = ""
    selected_skills: List[str] = field(default_factory=list)
    selected_tools: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    engineered_prompt: Optional[Dict[str, Any]] = None
    search_evidence: Optional[Dict[str, Any]] = None
    final_points: List[List[float]] = field(default_factory=list)
    reward: Optional[float] = None
    failure_tags: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    started_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)

    def add_skill(self, name: str) -> None:
        if name not in self.selected_skills:
            self.selected_skills.append(name)

    def add_tool_name(self, name: str) -> None:
        if name not in self.selected_tools:
            self.selected_tools.append(name)

    def add_tool_call(
        self,
        name: str,
        input_payload: Dict[str, Any],
        output_payload: Any,
        duration_s: float | None = None,
    ) -> None:
        self.add_tool_name(name)
        entry: Dict[str, Any] = {
            "name": name,
            "input": self._json_safe(input_payload),
            "output": self._json_safe(output_payload),
        }
        if duration_s is not None:
            entry["duration_s"] = duration_s
        self.tool_calls.append(entry)
        self.add_execution_step(
            phase="observe",
            action=name,
            details={
                "tool_input": self._json_safe(input_payload),
                "tool_output": self._json_safe(output_payload),
                "duration_s": duration_s,
            },
        )

    def add_execution_step(
        self,
        *,
        phase: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.execution_steps.append(
            {
                "phase": phase,
                "action": action,
                "details": self._json_safe(details or {}),
                "index": len(self.execution_steps),
            }
        )

    @staticmethod
    def _json_safe(x: Any) -> Any:
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, (list, tuple)):
            return [HarnessTrace._json_safe(i) for i in x]
        if isinstance(x, dict):
            return {str(k): HarnessTrace._json_safe(v) for k, v in x.items()}
        if hasattr(x, "to_dict"):
            return HarnessTrace._json_safe(x.to_dict())  # type: ignore[no-untyped-call]
        return str(x)

    def to_dict(self) -> Dict[str, Any]:
        tc = list(self.tool_calls)
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "question": self.question,
            "task_type": self.task_type,
            "selected_skills": list(self.selected_skills),
            "selected_tools": list(self.selected_tools),
            "tool_call_history": tc,
            "tool_calls": tc,
            "engineered_prompt": self.engineered_prompt,
            "search_evidence": self.search_evidence,
            "final_points": [list(p) for p in self.final_points],
            "reward": self.reward,
            "failure_tags": list(self.failure_tags),
            "reasoning_summary": self.reasoning_summary,
            "metadata": copy.deepcopy(self.metadata),
            "execution_steps": copy.deepcopy(self.execution_steps),
        }
        return d

"""Skill/Tool library registry for orchestrator-time selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from agent_afford_harness.agent.skills_runtime import (
    skill_code_when_geometry_matters,
    skill_lami_prompt_engineering,
    skill_point_output_normalization,
    skill_search_triggering,
    skill_task_type_classification,
    skill_zoom_before_detail,
)
from agent_afford_harness.tools.code_visual_analyzer import tool_code_visual_analyzer
from agent_afford_harness.tools.lami_detr_tool import tool_lami_detr_grounder
from agent_afford_harness.tools.web_search_tool import tool_web_search
from agent_afford_harness.tools.zoom_crop_tool import tool_zoom_crop


@dataclass
class CallableLibrary:
    """A small callable registry with explicit name-based invocation."""

    namespace: str
    registry: Dict[str, Callable[..., Any]]

    def has(self, name: str) -> bool:
        return name in self.registry

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self.registry:
            raise KeyError(f"{self.namespace} '{name}' is not registered")
        return self.registry[name](*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self.registry.keys())


def build_skill_library() -> CallableLibrary:
    return CallableLibrary(
        namespace="skill",
        registry={
            "task_type_classification": skill_task_type_classification,
            "search_triggering": skill_search_triggering,
            "lami_prompt_engineering": skill_lami_prompt_engineering,
            "zoom_before_detail": skill_zoom_before_detail,
            "code_when_geometry_matters": skill_code_when_geometry_matters,
            "point_output_normalization": skill_point_output_normalization,
        },
    )


def build_tool_library() -> CallableLibrary:
    return CallableLibrary(
        namespace="tool",
        registry={
            "lami_detr_grounder": tool_lami_detr_grounder,
            "zoom_crop_tool": tool_zoom_crop,
            "code_visual_analyzer": tool_code_visual_analyzer,
            "web_search_tool": tool_web_search,
        },
    )

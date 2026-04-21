"""Format skill/tool registries as planner-readable context."""

from __future__ import annotations

from typing import Any, Dict, List

from agent_afford_harness.config_load import _read_yaml


def build_library_context() -> Dict[str, Any]:
    skills_cfg = _read_yaml("skills.yaml")
    tools_cfg = _read_yaml("tools.yaml")
    skills = skills_cfg.get("skills") or []
    tools = tools_cfg.get("tools") or []
    return {
        "skills": [
            {
                "name": s.get("name"),
                "description": s.get("description", ""),
                "input_schema": s.get("input_schema", {}),
                "output_schema": s.get("output_schema", {}),
            }
            for s in skills
            if s.get("enabled", True)
        ],
        "tools": [
            {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", {}),
                "output_schema": t.get("output_schema", {}),
            }
            for t in tools
            if t.get("enabled", True)
        ],
    }


def tool_required_args_hint() -> Dict[str, Dict[str, List[str]]]:
    return {
        "lami_detr_grounder": {"required_args": ["prompt_info"]},
        "zoom_crop_tool": {"required_args": ["bbox"]},
        "code_visual_analyzer": {"required_args": ["analysis_type"]},
        "web_search_tool": {"required_args": ["queries"]},
    }

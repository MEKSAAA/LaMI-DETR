from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from agent_afford_harness.paths import harness_root

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _read_yaml(name: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    p = harness_root() / "configs" / name
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config() -> Dict[str, Any]:
    return _read_yaml("pipeline.yaml")


def load_prompt_hints() -> Dict[str, Any]:
    return _read_yaml("prompts.yaml")

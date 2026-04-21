"""Repository-root discovery for LaMI-DETR and harness assets."""

from __future__ import annotations

import os
from pathlib import Path


def harness_root() -> Path:
    """Directory containing this package (`agent_afford_harness/`)."""
    return Path(__file__).resolve().parent


def lami_det_repo_root() -> Path:
    """LaMI-DETR repo root (parent of `agent_afford_harness`)."""
    return harness_root().parent


def default_lami_config() -> Path:
    return lami_det_repo_root() / "lami_detr" / "configs" / "infer_dino_convnext_large.py"


def outputs_dir(sub: str | None = None) -> Path:
    base = harness_root() / "outputs"
    if sub:
        p = base / sub
        p.mkdir(parents=True, exist_ok=True)
        return p
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_env_overrides() -> None:
    """Optional: read .env next to harness if python-dotenv installed."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    env_path = harness_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path)

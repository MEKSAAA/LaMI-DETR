"""LLM-based planner for agentic affordance harness.

Supports:
- local_qwen_vl: local open-source VLM (e.g., Qwen2.5-VL)
- api_vlm: OpenAI-compatible vision API
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image


PLANNER_SYSTEM = """You are the orchestrator-planner for an affordance harness agent.
Given an image, question, and available skill/tool library, return STRICT JSON only (no markdown) with keys:
{
  "task_understanding": "one sentence",
  "task_type": "object_affordance_recognition|object_affordance_prediction|spatial_affordance_localization",
  "selected_skills": [{"name":"...", "reason":"..."}],
  "selected_tools": [{"name":"...", "reason":"..."}],
  "steps": [
    {"step_id":1, "kind":"tool", "name":"...", "args":{}, "expected_observation":"..."}
  ],
  "final_strategy": {"type":"aggregate_point", "reason":"..."},
  "fallback_policy": {"use_rules_if_invalid_plan": true}
}
Constraints:
- Steps must use only tools from provided library.
- Always include >=1 step and final_strategy.
- Output valid JSON only.
"""


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Some VLMs return "JSON-like" text with unquoted string values
    # (e.g., `"reason": text ...`). Patch common string fields and retry.
    repaired = blob
    for key in ("reason", "reasoning_summary"):
        repaired = re.sub(
            rf'("{key}"\s*:\s*)([^"\[\{{\n][^,\n\}}]*)',
            lambda mm: mm.group(1) + json.dumps(mm.group(2).strip()),
            repaired,
        )
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _pil_to_data_url(image: Image.Image, mime: str = "image/jpeg") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@dataclass
class ReasonerConfig:
    mode: str = "rules"  # rules | local_qwen_vl | api_vlm
    local_model_path: str = "/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct"
    api_model: str = "doubao-seed-2-0-lite-250821"
    api_base_url: Optional[str] = "https://ark.cn-beijing.volces.com/api/v3"
    api_key: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.1


class LLMReasoner:
    def __init__(self, cfg: ReasonerConfig):
        self.cfg = cfg
        self._local_loaded = False
        self._processor = None
        self._model = None
        self._last_trace: Dict[str, Any] = {}

    def get_last_trace(self) -> Dict[str, Any]:
        return dict(self._last_trace)

    def _ensure_local(self) -> None:
        if self._local_loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.cfg.local_model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._local_loaded = True

    def _local_plan(self, question: str, image: Image.Image, planner_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_local()
        assert self._processor is not None and self._model is not None
        import torch

        user_prompt = f"{PLANNER_SYSTEM}\nQuestion: {question}\nLibrary: {json.dumps(planner_context or {}, ensure_ascii=False)}"
        messages_for_trace = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "note": "input_image_attached"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}],
                }
            ]
            prompt_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt_text], images=[image], return_tensors="pt")
        else:
            inputs = self._processor(text=[user_prompt], images=[image], return_tensors="pt")

        # Move tensor inputs to model device
        model_device = next(self._model.parameters()).device
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(model_device)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=self.cfg.temperature > 0.0,
            )
        text = self._processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        parsed = _extract_first_json(text)
        self._last_trace = {
            "mode": "local_qwen_vl",
            "request": {
                "model_path": self.cfg.local_model_path,
                "temperature": self.cfg.temperature,
                "max_new_tokens": self.cfg.max_new_tokens,
                "messages": messages_for_trace,
            },
            "response": {"raw_text": text},
            "parsed_plan": parsed,
            "parse_ok": bool(parsed),
        }
        if not parsed:
            raise RuntimeError("Failed to parse JSON from local VLM output")
        return parsed

    def _api_plan(self, question: str, image: Image.Image, planner_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from openai import OpenAI

        api_key = self.cfg.api_key or os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ARK_API_KEY/OPENAI_API_KEY for api_vlm mode")
        base_url = self.cfg.api_base_url or os.environ.get("AGENT_HARNESS_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        image_url = _pil_to_data_url(image)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "text", "text": f"Library: {json.dumps(planner_context or {}, ensure_ascii=False)}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        resp = client.chat.completions.create(
            model=self.cfg.api_model,
            temperature=self.cfg.temperature,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        parsed = _extract_first_json(text)
        self._last_trace = {
            "mode": "api_vlm",
            "request": {
                "model": self.cfg.api_model,
                "base_url": base_url or "",
                "temperature": self.cfg.temperature,
                "messages": [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question: {question}"},
                            {"type": "image_url", "image_url": {"url": "<data_url_omitted>"}},
                        ],
                    },
                ],
            },
            "response": {
                "id": getattr(resp, "id", ""),
                "model": getattr(resp, "model", ""),
                "raw_text": text,
            },
            "parsed_plan": parsed,
            "parse_ok": bool(parsed),
        }
        if not parsed:
            raise RuntimeError("Failed to parse JSON from API VLM output")
        return parsed

    def plan(self, question: str, image: Image.Image, planner_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._last_trace = {}
        mode = (self.cfg.mode or "rules").lower()
        if mode == "local_qwen_vl":
            return self._local_plan(question, image, planner_context)
        if mode == "api_vlm":
            return self._api_plan(question, image, planner_context)
        raise RuntimeError(f"LLMReasoner mode not supported for planning: {mode}")


def load_reasoner_config(pipeline_cfg: Dict[str, Any]) -> ReasonerConfig:
    llm_cfg = dict((pipeline_cfg or {}).get("llm") or {})
    # Env overrides
    mode = os.environ.get("AGENT_HARNESS_LLM_MODE", llm_cfg.get("mode", "rules"))
    local_model_path = os.environ.get(
        "AGENT_HARNESS_LOCAL_VLM_PATH",
        llm_cfg.get("local_model_path", "/NEW_EDS/miaojw/models/Qwen2.5-VL-3B-Instruct"),
    )
    api_model = os.environ.get("AGENT_HARNESS_API_MODEL", llm_cfg.get("api_model", "doubao-seed-2-0-lite-250821"))
    api_base_url = os.environ.get(
        "AGENT_HARNESS_API_BASE_URL",
        llm_cfg.get("api_base_url", "https://ark.cn-beijing.volces.com/api/v3"),
    )
    api_key = os.environ.get("AGENT_HARNESS_API_KEY") or os.environ.get("ARK_API_KEY") or llm_cfg.get("api_key")
    max_new_tokens = int(llm_cfg.get("max_new_tokens", 512))
    temperature = float(llm_cfg.get("temperature", 0.1))
    return ReasonerConfig(
        mode=mode,
        local_model_path=local_model_path,
        api_model=api_model,
        api_base_url=api_base_url,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

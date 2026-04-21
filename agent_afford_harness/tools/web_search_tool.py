"""
Web search tool for the affordance harness.

Design:
- Prefer Zhipu Web Search API
- Return structured search evidence for downstream prompt engineering
- Fall back to a local stub if API is unavailable

No chat model is required in this version.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List


BIGMODEL_WEB_SEARCH_URL = os.environ.get(
    "BIGMODEL_WEB_SEARCH_URL",
    "https://open.bigmodel.cn/api/paas/v4/tools",
)

DEFAULT_SEARCH_ENGINE = os.environ.get("AGENT_HARNESS_SEARCH_ENGINE", "search_pro")
DEFAULT_CONTENT_SIZE = os.environ.get("AGENT_HARNESS_SEARCH_CONTENT_SIZE", "high")
DEFAULT_RECENCY = os.environ.get("AGENT_HARNESS_SEARCH_RECENCY", "noLimit")


def _safe_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


DEFAULT_SEARCH_COUNT = _safe_int(os.environ.get("AGENT_HARNESS_SEARCH_COUNT", "5"), 5)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _dedup_keep_order(items: List[str], max_items: int = 8) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        val = _normalize_text(item)
        key = val.lower()
        if not val or key in seen:
            continue
        seen.add(key)
        out.append(val)
        if len(out) >= max_items:
            break
    return out


def _normalize_queries(queries: List[str]) -> List[str]:
    return _dedup_keep_order([_normalize_text(q) for q in (queries or [])], max_items=32)


def _simple_keyword_extract(text: str, max_items: int = 6) -> List[str]:
    """
    Lightweight extractor for concepts / visual cues from titles and snippets.
    Suitable for harness prompt expansion, not for final-answer generation.
    """
    text = _normalize_text(text).lower()

    english_tokens = re.findall(r"[a-z][a-z0-9\-\+]{2,}", text)
    english_tokens = [
        t for t in english_tokens
        if t not in {"https", "http", "www", "com"}
    ]

    phrase_chunks = re.split(r"[，。；;,.|/()\[\]{}]+", text)
    phrase_chunks = [p.strip() for p in phrase_chunks if 2 <= len(p.strip()) <= 40]

    merged = phrase_chunks + english_tokens
    return _dedup_keep_order(merged, max_items=max_items)


def _stub(queries: List[str]) -> Dict[str, Any]:
    out = {}
    for q in queries:
        key_tokens = [t for t in re.split(r"\W+", q.lower()) if len(t) > 3][:6]
        concepts = list(dict.fromkeys(key_tokens))[:5]
        if not concepts:
            concepts = ["object", "visual cues"]
        out[q] = {
            "concepts": concepts,
            "visual_cues": [
                f"look for typical {concepts[0]} silhouette",
                "compare scale against nearby objects",
            ],
            "raw_search_results": [],
            "source": "stub",
        }
    return {"queries": queries, "results": out, "backend": "stub"}


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: int = 60) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _extract_from_search_results(search_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    concepts: List[str] = []
    visual_cues: List[str] = []

    for item in search_results:
        title = _normalize_text(item.get("title", ""))
        content = _normalize_text(item.get("content", ""))
        media = _normalize_text(item.get("media", ""))

        if title:
            concepts.extend(_simple_keyword_extract(title, max_items=4))
        if content:
            visual_cues.extend(_simple_keyword_extract(content, max_items=6))
        if media:
            concepts.append(media)

    return {
        "concepts": _dedup_keep_order(concepts, max_items=8),
        "visual_cues": _dedup_keep_order(visual_cues, max_items=8),
    }


def _glm_web_search_api(
    queries: List[str],
    api_key: str,
    search_engine: str = DEFAULT_SEARCH_ENGINE,
    count: int = DEFAULT_SEARCH_COUNT,
    content_size: str = DEFAULT_CONTENT_SIZE,
    search_recency_filter: str = DEFAULT_RECENCY,
) -> Dict[str, Any]:
    """
    Preferred backend: Zhipu Web Search API.

    According to the official docs, this API is designed to return structured
    search results (title / URL / snippet / media / etc.), which is exactly what
    our harness needs for downstream prompt engineering rather than direct answer
    generation.
    """
    results: Dict[str, Any] = {}

    for q in queries:
        payload = {
            "search_engine": search_engine,
            "search_query": q,
            "count": count,
            "search_recency_filter": search_recency_filter,
            "content_size": content_size,
        }

        body = _post_json(
            BIGMODEL_WEB_SEARCH_URL,
            payload,
            api_key=api_key,
        )

        raw_search = body.get("search_result") or []
        if not isinstance(raw_search, list):
            raw_search = []

        extracted = _extract_from_search_results(raw_search)

        results[q] = {
            "concepts": extracted["concepts"],
            "visual_cues": extracted["visual_cues"],
            "raw_search_results": raw_search,
            "source": "web_search_api",
        }

    return {
        "queries": queries,
        "results": results,
        "backend": "glm_web_search_api",
        "search_engine": search_engine,
    }


def tool_web_search(queries: List[str]) -> Dict[str, Any]:
    norm_queries = _normalize_queries(queries)
    if not norm_queries:
        return {"queries": [], "results": {}, "backend": "empty"}

    api_key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
    if not api_key:
        stub = _stub(norm_queries)
        stub["backend"] = "stub_no_key"
        stub["fallback_reason"] = "missing_api_key"
        return stub

    try:
        return _glm_web_search_api(norm_queries, api_key=api_key)
    except Exception as e:
        stub = _stub(norm_queries)
        stub["backend"] = "stub_api_error"
        stub["fallback_reason"] = str(e)
        return stub
# Tool: web_search_tool

`web_search_tool` is the harness knowledge-expansion module. It takes query strings and returns compact concept and visual-cue evidence intended for prompt enrichment, not for direct final-answer generation.

Implementation entrypoint: `tools/web_search_tool.py::tool_web_search`.

Input is a list of search queries. Output is a structured result map per query, including extracted `concepts`, `visual_cues`, optional raw search payloads, and backend metadata.

The tool is designed for robust degradation. When API credentials are missing or remote calls fail, it falls back to deterministic stub outputs and reports explicit backend/failure markers (`stub_no_key`, `stub_api_error`, `fallback_reason`). This keeps pipeline traces consistent across online and offline settings.

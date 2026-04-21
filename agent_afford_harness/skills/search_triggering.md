# Skill: search_triggering

This skill decides whether external knowledge expansion is worth the latency cost. It does not call search APIs directly; it only emits a policy decision (`need_search`, `queries`, and `reasons`) for the orchestrator.

Current heuristics trigger search for long-tail attributes, unusually long or abstract phrasing, and explicit rarity cues. The intent is to reserve search calls for samples where grounding is likely to benefit from auxiliary conceptual hints, while keeping straightforward cases fast.

The skill is designed for graceful degradation. If search is requested but unavailable, tool-layer fallback still returns structured evidence with explicit backend and failure reason tags (`stub_no_key`, `stub_api_error`, `fallback_reason`). This keeps traces analyzable even in offline runs.

In traces, this policy appears in `selected_skills`, and the resulting evidence is stored under `search_evidence`.

# Skill: lami_prompt_engineering

This skill rewrites raw benchmark language into detector-facing prompt structures. Instead of passing the original question directly to grounding, it builds `prompt_info` with `object_prompt`, `part_prompt`, `spatial_query`, `negative_hint`, and `lami_classes` (class names mapped to short descriptor lists).

The design objective is semantic fidelity with detector usability. For spatial questions, the skill tries to split anchor entities (for example, “between A and B”). For part questions, it emphasizes functional regions. If search evidence is available, it compresses extracted concepts into short visual hints that are still practical for grounding.

The output is backend-agnostic and can be consumed by local LaMI-DETR or Doubao grounding. That makes prompt engineering a reusable policy layer rather than a backend-specific hardcode.

If prompt construction fails, fallback is intentionally simple: `object_prompt = question`. This guarantees pipeline continuity while keeping failure visible in traces. Full prompt artifacts are always stored in `trace.engineered_prompt`.

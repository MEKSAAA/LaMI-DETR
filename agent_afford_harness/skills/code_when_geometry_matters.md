# Skill: code_when_geometry_matters

This skill determines whether explicit geometric analysis should be invoked and which deterministic template should be selected. It connects semantic grounding outputs to procedural geometry reasoning by producing `need_code_analysis` and `analysis_type`.

Current routing is intentionally interpretable: spatial tasks map to `free_space_between_boxes`; part tasks mentioning grasp/handle cues map to `elongated_grasp_region`; container/interior cues map to `interior_region`. The goal is not to be exhaustive, but to provide stable and auditable behavior.

When confidence is low, fallback can disable code analysis and let aggregators use heuristic point placement from existing evidence. This avoids introducing noisy template outputs in clearly underconstrained cases.

The chosen analysis type is always visible in tool call records, which makes failure triage and future policy refinement straightforward.

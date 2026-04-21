# Tool: code_visual_analyzer

`code_visual_analyzer` is a deterministic geometry reasoning module. It does not run free-form agent-generated code; instead, it executes a fixed set of reproducible templates and returns structured candidate points for aggregation.

Implementation entrypoint: `tools/code_visual_analyzer.py::tool_code_visual_analyzer`.

Inputs include an image (full frame or crop), `analysis_type`, and optional context (`candidate_boxes_xyxy`, `crop_meta`, `full_image_size`). Outputs include crop-normalized points, projected full-image points, confidence, and textual diagnostics.

Supported templates are:
- `interior_region`
- `elongated_grasp_region`
- `free_space_between_boxes`

The design constraint is stability over flexibility: deterministic templates make regression tracking and failure triage easier in benchmark-driven harness development.

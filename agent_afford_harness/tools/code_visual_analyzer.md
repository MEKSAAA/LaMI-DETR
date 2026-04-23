name: code_visual_analyzer
type: tool
description: Run deterministic geometry templates to produce candidate affordance points.
input_schema:
  image: PIL.Image
  analysis_type: enum[interior_region, elongated_grasp_region, free_space_between_boxes]
  candidate_boxes_xyxy: list[list[float]] | null
  crop_meta: dict | null
  full_image_size: [int, int] | null
  n_points: int
output_schema:
  analysis_type: str
  points_normalized_crop: list[list[float]]
  points_full_image: list[list[float]]
  confidence: float
  text: str
system_prompt: "Use deterministic templates only; no free-form code generation."

# Tool: code_visual_analyzer

Template-based deterministic visual analyzer.

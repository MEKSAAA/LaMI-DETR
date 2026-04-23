name: code_when_geometry_matters
type: skill
description: Decide whether deterministic geometric templates should be run.
input_schema:
  task_type: str
  question: str
  tool_outputs_hint: dict
  llm_analysis_type: str | null
output_schema:
  need_code_analysis: bool
  analysis_type: enum[interior_region, elongated_grasp_region, free_space_between_boxes]
policy:
  - Use geometry templates for spatial and detailed part localization.
  - Allow planner override of analysis_type when present.

# Skill: code_when_geometry_matters

Policy skill for routing to code_visual_analyzer.

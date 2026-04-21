name: point_output_normalization
type: skill
description: Normalize candidate points into benchmark-required [0,1] coordinates and expected count.
input_schema:
  points_xy: list[tuple[float, float]]
  image_width: int
  image_height: int
  task_type: str
output_schema:
  normalized_points: list[list[float]]
policy:
  - Clamp coordinates to [0,1].
  - Match expected point count by task type.

# Skill: point_output_normalization

Policy skill for final benchmark output formatting.

name: zoom_before_detail
type: skill
description: Decide whether ROI crop/zoom is needed before detailed localization.
input_schema:
  task_type: str
  question: str
  lami_boxes: list[dict]
  image_size: tuple[int, int]
  llm_need_zoom: bool | null
output_schema:
  need_zoom: bool
  crop_target_index: int
  padding_ratio: float
  resize_max_side: int
policy:
  - Prefer zoom for part/spatial fine-grained queries.
  - Increase zoom tendency when target box area is small.

# Skill: zoom_before_detail

Policy skill controlling pre-analysis crop.

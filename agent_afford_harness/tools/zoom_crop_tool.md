name: zoom_crop_tool
type: tool
description: Crop and optionally resize ROI while keeping reversible crop-to-full coordinate metadata.
input_schema:
  image: PIL.Image
  bbox_xyxy: [float, float, float, float]
  padding_ratio: float
  max_side: int
output_schema:
  crop: PIL.Image
  origin_xy: [float, float]
  crop_width: int
  crop_height: int
  scale: float
system_prompt: "Perform deterministic crop and return metadata for coordinate projection."

# Tool: zoom_crop_tool

Deterministic image pre-processing tool.

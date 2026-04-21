name: lami_detr_grounder
type: tool
description: Ground objects/parts via LaMI-DETR (or compatible backend) and return ranked boxes.
input_schema:
  image: PIL.Image
  prompt_info: dict
  nms_iou: float
  score_threshold: float
  topk: int
  backend: str
output_schema:
  boxes: list[{bbox: [x1,y1,x2,y2], score: float, label: str}]
  prompt_summary: dict
  backend: str
  image_size: [int, int]
system_prompt: "Ground relevant entities from prompt classes and return structured candidate boxes."

# Tool: lami_detr_grounder

Primary grounding action tool.

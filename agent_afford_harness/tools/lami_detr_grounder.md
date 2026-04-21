# Tool: lami_detr_grounder

`lami_detr_grounder` is the candidate proposal tool for visual grounding. It executes one of two backends behind a unified interface: local LaMI-DETR (`backend=lami`) or Doubao grounding (`backend=doubao_grounding`), then returns structured candidate boxes for downstream policies.

Implementation entrypoint: `tools/lami_detr_tool.py::tool_lami_detr_grounder`.

Inputs include an image, engineered prompt package (`prompt_info`), and optional control parameters (`nms_iou`, `score_threshold`, `topk`, `backend`). Outputs include candidate boxes, scores, class labels, image size, backend metadata, and coordinate-space annotations.

Coordinate contract is explicit. Internal pipeline processing uses pixel-space `bbox` values. When Doubao grounding is used, 1000-grid coordinates are converted to pixel coordinates on input and retained as `bbox_1000` for traceability and round-trip compatibility.

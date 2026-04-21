# Skill: zoom_before_detail

This skill decides whether the pipeline should switch from full-image processing to local high-resolution processing before detailed analysis. It consumes task type, question text, candidate boxes, and image size, then produces a deterministic crop policy (`need_zoom`, `crop_target_index`, `padding_ratio`, `resize_max_side`).

The policy favors zoom for part localization by default, and for spatial tasks when wording suggests fine geometry (for example: between, edge, interior, rim, handle). It also escalates to zoom when candidate region area is too small relative to the image, because part-level reasoning is often unstable at full-frame scale.

If no reliable candidate boxes exist, fallback is safe and explicit: do not crop, continue on the full image, and let later tools/aggregators handle the case. This prevents hard failures while preserving reproducibility.

All crop decisions and resulting crop metadata are persisted through `zoom_crop_tool` entries in `tool_call_history`.

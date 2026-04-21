# Tool: zoom_crop_tool

`zoom_crop_tool` is a deterministic preprocessing module for local refinement. Given a full image and a pixel-space bounding box, it applies padded cropping (and optional resizing) and returns both the cropped image and the metadata required for exact back-projection.

Implementation entrypoint: `tools/zoom_crop_tool.py::tool_zoom_crop`.

Inputs are `image`, `bbox_xyxy`, `padding_ratio`, and `max_side`. Outputs include `crop`, `origin_xy`, `crop_width`, `crop_height`, `scale`, and original bbox metadata.

The key engineering property is coordinate reversibility. Crop-normalized points can be mapped back to full-image pixel coordinates using `crop_coords_to_full()`, which keeps downstream geometric outputs consistent and auditable.

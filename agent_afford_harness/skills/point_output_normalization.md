# Skill: point_output_normalization

This skill enforces the benchmark output contract at the end of the pipeline. It takes candidate points from heterogeneous upstream sources (pixel-space or already normalized), converts them into normalized `[0,1]` coordinates, applies boundary-safe clipping, and ensures task-specific point counts.

The implementation automatically detects coordinate space, clips values into valid ranges, adds a small epsilon margin to avoid fragile border cases, and pads sparse outputs to expected counts (3 points for recognition/part tasks, 5 points for spatial tasks).

The primary benchmark-facing output remains normalized `final_points`. For interoperability with Doubao grounding conventions, the orchestrator also stores 1000-grid equivalents in `metadata.final_points_1000`.

If the input point list is empty, the skill returns an empty list and lets upstream aggregators define fallback behavior.

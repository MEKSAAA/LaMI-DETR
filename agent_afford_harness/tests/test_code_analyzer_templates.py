from PIL import Image

from agent_afford_harness.tools.code_visual_analyzer import tool_code_visual_analyzer


def test_free_space_template():
    im = Image.new("RGB", (100, 100), color="gray")
    out = tool_code_visual_analyzer(
        im,
        "free_space_between_boxes",
        candidate_boxes_xyxy=[[10, 10, 30, 80], [60, 10, 90, 80]],
        full_image_size=(100, 100),
    )
    assert len(out["points_normalized_crop"]) == 5

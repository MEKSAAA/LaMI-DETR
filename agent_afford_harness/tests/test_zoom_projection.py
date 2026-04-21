from PIL import Image

from agent_afford_harness.tools.zoom_crop_tool import tool_zoom_crop, crop_coords_to_full


def test_crop_project_roundtrip():
    im = Image.new("RGB", (200, 200), color="white")
    pack = tool_zoom_crop(im, [50, 50, 100, 100], padding_ratio=0.0, max_side=200)
    meta = {k: v for k, v in pack.items() if k != "crop"}
    x, y = crop_coords_to_full((0.5, 0.5), meta)
    # center of crop
    assert 60 < x < 100
    assert 60 < y < 100

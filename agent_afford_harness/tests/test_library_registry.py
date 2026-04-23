from agent_afford_harness.agent.library import build_skill_library, build_tool_library


def test_library_contains_expected_entries():
    skills = build_skill_library()
    tools = build_tool_library()
    assert skills.has("task_type_classification")
    assert skills.has("point_output_normalization")
    assert tools.has("lami_detr_grounder")
    assert tools.has("zoom_crop_tool")


def test_library_call_missing_name_raises():
    skills = build_skill_library()
    try:
        skills.call("not_exists")
        assert False, "call should fail for missing registry names"
    except KeyError:
        assert True

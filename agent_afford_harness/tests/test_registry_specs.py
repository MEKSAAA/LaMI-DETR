from pathlib import Path

import yaml


def test_skill_and_tool_yaml_use_standard_keys():
    skills_cfg = yaml.safe_load(Path("agent_afford_harness/configs/skills.yaml").read_text(encoding="utf-8"))
    tools_cfg = yaml.safe_load(Path("agent_afford_harness/configs/tools.yaml").read_text(encoding="utf-8"))

    for item in skills_cfg["skills"]:
        for key in ("name", "type", "description", "input_schema", "output_schema"):
            assert key in item
        assert item["type"] == "skill"

    for item in tools_cfg["tools"]:
        for key in ("name", "type", "description", "input_schema", "output_schema"):
            assert key in item
        assert item["type"] == "tool"


def test_skill_and_tool_markdown_have_standard_headers():
    root = Path("agent_afford_harness")
    required = ("name:", "type:", "description:", "input_schema:", "output_schema:")

    for folder in ("skills", "tools"):
        for md in (root / folder).glob("*.md"):
            text = md.read_text(encoding="utf-8")
            for token in required:
                assert token in text, f"{md} missing token: {token}"

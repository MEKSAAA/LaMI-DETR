from agent_afford_harness.agent.task_parser import strip_benchmark_format_instructions, normalize_category_label


def test_strip_long_suffix():
    q = "Point to the mug. Your answer should be formatted as a list of tuples"
    s = strip_benchmark_format_instructions(q)
    assert "mug" in s.lower() or "point" in s.lower()


def test_normalize_category():
    assert "recognition" in normalize_category_label("object reference").lower()

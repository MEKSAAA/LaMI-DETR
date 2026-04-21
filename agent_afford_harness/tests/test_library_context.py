from agent_afford_harness.agent.library_context import build_library_context


def test_library_context_contains_skills_and_tools():
    ctx = build_library_context()
    assert "skills" in ctx and ctx["skills"]
    assert "tools" in ctx and ctx["tools"]

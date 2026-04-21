from agent_afford_harness.agent.state import HarnessTrace


def test_trace_records_execution_steps():
    trace = HarnessTrace(sample_id="s1", image_path="a.jpg", question="q")
    trace.add_execution_step(phase="plan", action="task_type", details={"task_type": "x"})
    trace.add_tool_call("dummy_tool", {"x": 1}, {"ok": True}, 0.1)
    out = trace.to_dict()
    assert out["execution_steps"]
    assert out["execution_steps"][0]["phase"] == "plan"
    assert any(s["action"] == "dummy_tool" for s in out["execution_steps"])

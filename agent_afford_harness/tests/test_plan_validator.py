from agent_afford_harness.agent.plan_schema import plan_from_dict
from agent_afford_harness.agent.plan_validator import validate_plan


def test_plan_validator_accepts_minimal_valid_plan():
    p = plan_from_dict(
        {
            "task_understanding": "x",
            "task_type": "object_affordance_prediction",
            "selected_skills": [{"name": "task_type_classification"}],
            "selected_tools": [{"name": "lami_detr_grounder"}],
            "steps": [{"step_id": 1, "kind": "tool", "name": "lami_detr_grounder", "args": {"prompt_info": {}}}],
            "final_strategy": {"type": "aggregate_point"},
            "fallback_policy": {"use_rules_if_invalid_plan": True},
        }
    )
    res = validate_plan(
        p,
        valid_skills=["task_type_classification"],
        valid_tools=["lami_detr_grounder"],
        tool_schemas={"lami_detr_grounder": {"required_args": ["prompt_info"]}},
    )
    assert res.ok


def test_plan_validator_rejects_unknown_tool():
    p = plan_from_dict(
        {
            "task_understanding": "x",
            "task_type": "object_affordance_prediction",
            "selected_skills": [],
            "selected_tools": [],
            "steps": [{"step_id": 1, "kind": "tool", "name": "unknown_tool", "args": {}}],
            "final_strategy": {"type": "aggregate_point"},
            "fallback_policy": {"use_rules_if_invalid_plan": True},
        }
    )
    res = validate_plan(p, valid_skills=[], valid_tools=[])
    assert not res.ok

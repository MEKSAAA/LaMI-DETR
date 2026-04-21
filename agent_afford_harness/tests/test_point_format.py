from agent_afford_harness.agent.skills_runtime import skill_point_output_normalization
from agent_afford_harness.agent import router as R


def test_normalize_points_count():
    pts = [(0.2, 0.3), (0.22, 0.31)]
    out = skill_point_output_normalization(pts, 100, 100, R.SPATIAL_AFFORDANCE_LOCALIZATION)
    assert len(out) == 5
    for p in out:
        assert 0 < p[0] < 1 and 0 < p[1] < 1

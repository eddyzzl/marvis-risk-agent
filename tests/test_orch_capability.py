from types import SimpleNamespace

from marvis.orchestrator.capability import TIERS, auto_gate_budget, resolve_tier, tier_from_settings


def test_capability_tiers_resolve_and_unknown_falls_back_to_default(tmp_path):
    assert resolve_tier("conservative").name == "conservative"
    assert resolve_tier(" AUTONOMOUS ").name == "autonomous"
    assert resolve_tier("missing").name == "balanced"

    settings = SimpleNamespace(workspace=tmp_path)
    assert tier_from_settings(settings).name == "balanced"


def test_capability_tier_limits_are_monotonic():
    conservative = TIERS["conservative"]
    balanced = TIERS["balanced"]
    autonomous = TIERS["autonomous"]

    assert conservative.default_autonomy_level <= balanced.default_autonomy_level <= autonomous.default_autonomy_level
    assert conservative.max_replan_iterations <= balanced.max_replan_iterations <= autonomous.max_replan_iterations
    assert conservative.max_plan_depth <= balanced.max_plan_depth <= autonomous.max_plan_depth
    assert conservative.explore_segment_size <= balanced.explore_segment_size <= autonomous.explore_segment_size
    assert conservative.allow_explore_mode is False
    assert balanced.allow_explore_mode is True
    assert conservative.max_auto_gates <= balanced.max_auto_gates <= autonomous.max_auto_gates


def test_auto_gate_budget_scales_with_plan_gate_count_and_caps_at_tier():
    """AGT-7: the AUTO gate budget is gate_count + a fixed margin, capped by the
    tier's ceiling — not the old fixed 8 that silently exhausted on any plan
    with >=9 gates."""
    balanced = TIERS["balanced"]
    assert auto_gate_budget(balanced, 3) == 5  # 3 + margin(2)
    assert auto_gate_budget(balanced, 0) == 4  # floored at AUTO_GATE_BUDGET_MIN
    assert auto_gate_budget(balanced, 100) == balanced.max_auto_gates  # capped by tier

    conservative = TIERS["conservative"]
    autonomous = TIERS["autonomous"]
    assert auto_gate_budget(conservative, 100) == conservative.max_auto_gates
    assert auto_gate_budget(autonomous, 100) == autonomous.max_auto_gates
    assert auto_gate_budget(conservative, 100) < auto_gate_budget(autonomous, 100)

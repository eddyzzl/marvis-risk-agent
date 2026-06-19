from types import SimpleNamespace

from marvis.orchestrator.capability import TIERS, resolve_tier, tier_from_settings


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

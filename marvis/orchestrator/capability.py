from __future__ import annotations

from dataclasses import dataclass

from marvis.llm_settings import LLMSettingsError, load_llm_settings


@dataclass(frozen=True)
class CapabilityTier:
    name: str
    default_autonomy_level: int
    failure_driven_replan: bool
    allow_explore_mode: bool
    decision_point_replan: bool
    max_replan_iterations: int
    max_plan_depth: int
    explore_segment_size: int


TIERS: dict[str, CapabilityTier] = {
    "conservative": CapabilityTier(
        name="conservative",
        default_autonomy_level=0,
        failure_driven_replan=True,
        allow_explore_mode=False,
        decision_point_replan=True,
        max_replan_iterations=2,
        max_plan_depth=8,
        explore_segment_size=1,
    ),
    "balanced": CapabilityTier(
        name="balanced",
        default_autonomy_level=1,
        failure_driven_replan=True,
        allow_explore_mode=True,
        decision_point_replan=True,
        max_replan_iterations=4,
        max_plan_depth=16,
        explore_segment_size=3,
    ),
    "autonomous": CapabilityTier(
        name="autonomous",
        default_autonomy_level=2,
        failure_driven_replan=True,
        allow_explore_mode=True,
        decision_point_replan=True,
        max_replan_iterations=8,
        max_plan_depth=24,
        explore_segment_size=5,
    ),
}
DEFAULT_TIER = "balanced"


def resolve_tier(name: str | None) -> CapabilityTier:
    key = str(name or "").strip().lower()
    return TIERS.get(key, TIERS[DEFAULT_TIER])


def tier_from_settings(settings) -> CapabilityTier:
    explicit = getattr(settings, "capability_tier", None)
    if explicit:
        return resolve_tier(explicit)
    try:
        data = load_llm_settings(settings.workspace)
    except (LLMSettingsError, AttributeError):
        return TIERS[DEFAULT_TIER]
    return resolve_tier(data.get("capability_tier"))

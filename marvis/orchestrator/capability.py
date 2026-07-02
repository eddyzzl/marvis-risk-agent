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
    # AGT-7: ceiling on how many gates agent_autodrive_turn will auto-process in a
    # single AUTO(自动审查) turn before stopping for a human. The *effective*
    # per-turn budget is dynamic (plan gate count + a small margin, see
    # auto_gate_budget below) — this is only the tier's upper bound on that.
    max_auto_gates: int = 8


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
        max_auto_gates=6,
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
        max_auto_gates=10,
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
        max_auto_gates=16,
    ),
}
DEFAULT_TIER = "balanced"
# AGT-7: fixed margin added on top of a plan's own gate count (needs_confirmation
# steps + the plan-overview gate) so adjust/dedup re-stops at the same gate don't
# immediately exhaust the budget for an otherwise gate-count-accurate plan.
AUTO_GATE_BUDGET_MARGIN = 2
# Floor so a plan with very few gates (or none loaded yet, e.g. before the C1
# file-role gate builds the real plan) still gets a usable AUTO budget.
AUTO_GATE_BUDGET_MIN = 4


def auto_gate_budget(tier: CapabilityTier, gate_count: int) -> int:
    """Dynamic per-turn AUTO gate budget (AGT-7): the plan's own gate count plus a
    fixed margin for adjust/dedup re-stops, capped by the tier's ceiling so a
    misconfigured/huge plan still can't auto-drive unboundedly."""
    dynamic = max(int(gate_count), 0) + AUTO_GATE_BUDGET_MARGIN
    return max(min(dynamic, tier.max_auto_gates), AUTO_GATE_BUDGET_MIN)


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

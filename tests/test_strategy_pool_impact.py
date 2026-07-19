from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from marvis.packs.strategy.candidate_fragment import build_verified_candidate_fragment
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    reorder_strategy_pool,
)
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_SCHEMA_VERSION,
    build_strategy_pool_impact_assessment,
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _action(action_type: str) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": None if action_type == "approval" else action_type.upper(),
        "stop": True,
    }


def _identity() -> dict:
    return {
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 1,
        "semantic_mapping_hash": HASH_B,
        "sample_context_hash": HASH_C,
    }


def _sample_binding() -> dict:
    return {"task_id": "task-1", **_identity()}


def _condition(operator: str, value: int) -> dict:
    return {
        "op": "compare",
        "field": "score",
        "operator": operator,
        "value": value,
        "missing": "no_match",
    }


def _fragment(index: int, condition: dict) -> dict:
    suffix = f"{index:064x}"
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{index}",
            "artifact_kind": "test_candidate_json",
            "artifact_schema_version": "test.candidate-artifact.v1",
            "artifact_content_hash": suffix,
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "test.candidate.v1",
            "asset_id": f"candidate-asset-{index}",
            "asset_hash": suffix,
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=f"candidate-rule-{index}",
        condition=condition,
        requirements=[],
        effect_id=f"candidate-effect-{index}",
        evidence_id="candidate-evidence-1",
        evidence_hash=HASH_D,
        evidence_identity=_identity(),
    )


def _pool(*, broad_first: bool = False, strategy_type: str = "approval") -> dict:
    if strategy_type in {"approval", "reject"}:
        default = _action("approval")
        actions = (_action("reject"), _action("review"))
    else:
        default = {
            "type": "limit",
            "value": 1000,
            "reason_code": None,
            "stop": True,
        }
        actions = (
            {"type": "limit", "value": 500, "reason_code": None, "stop": True},
            {"type": "limit", "value": 800, "reason_code": None, "stop": True},
        )
    conditions = (
        (_condition("<", 8), _condition("<", 5))
        if broad_first
        else (_condition("<", 5), _condition("<", 8))
    )
    result = None
    for index, (condition, action) in enumerate(zip(conditions, actions, strict=True), 1):
        result = add_verified_candidate_fragment(
            result,
            task_id="task-1",
            strategy_type=strategy_type,
            default_action=default,
            verified_candidate_fragment=_fragment(index, condition),
            action=action,
        )
    assert result is not None
    return result


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bad": [1, 1, 0, 1, 0, 1, None, 0, 0, 1],
            "month": [
                "202601",
                "202601",
                "202601",
                "202601",
                "202601",
                "202602",
                "202602",
                "202602",
                "202602",
                "202602",
            ],
            "loan": [100, 100, 100, 100, None, 200, 200, 200, 200, 200],
            "overdue": [10, 0, 0, 5, 0, 20, None, 0, 0, 10],
        }
    )


def _build(**overrides) -> dict:
    values = {
        "pool": _pool(),
        "frame": _frame(),
        "sample_binding": _sample_binding(),
        "target_col": "bad",
        "month_col": "month",
        "loan_amount_col": "loan",
        "overdue_amount_col": "overdue",
    }
    values.update(overrides)
    return build_strategy_pool_impact_assessment(**values)


def test_pool_impact_first_match_waterfall_and_population_conserve() -> None:
    impact = _build()

    assert impact["schema_version"] == STRATEGY_POOL_IMPACT_SCHEMA_VERSION
    assert impact["population"] == {
        "population_count": 10,
        "labelled_count": 9,
        "unlabelled_count": 1,
        "label_coverage": 0.9,
    }
    first, second = impact["waterfall"]
    assert first["standalone"]["population_count"] == 4
    assert first["incremental"]["population_count"] == 4
    assert first["shadowed"]["population_count"] == 0
    assert second["standalone"]["population_count"] == 7
    assert second["incremental"]["population_count"] == 3
    assert second["shadowed"]["population_count"] == 4
    assert impact["default_unmatched"]["effect"]["population_count"] == 3
    assert sum(row["incremental"]["population_count"] for row in impact["waterfall"]) + 3 == 10
    assert impact["overall"]["actions"]["metrics"]["approve_count"] == 3
    assert impact["overall"]["actions"]["metrics"]["reject_count"] == 4
    assert impact["overall"]["actions"]["metrics"]["review_count"] == 3
    assert impact["overall"]["actions"]["metrics"]["bad_capture_rate"] == pytest.approx(
        3 / 5
    )


def test_pool_impact_reports_fully_shadowed_rule_and_reorder_changes_reach() -> None:
    pool = _pool(broad_first=True)
    broad = _build(pool=pool)
    assert broad["waterfall"][1]["incremental"]["population_count"] == 0
    assert any(flag["code"] == "rule_fully_shadowed" for flag in broad["red_flags"])

    reversed_pool = reorder_strategy_pool(
        pool, [pool["entries"][1]["entry_id"], pool["entries"][0]["entry_id"]]
    )
    reordered = _build(pool=reversed_pool)
    assert reordered["waterfall"][0]["incremental"]["population_count"] == 4
    assert reordered["waterfall"][1]["incremental"]["population_count"] == 3


def test_pool_impact_monthly_and_amounts_roll_to_overall() -> None:
    impact = _build()
    monthly = impact["monthly"]

    assert monthly["status"] == "available"
    assert [row["period"] for row in monthly["periods"]] == ["202601", "202602"]
    assert sum(row["effect"]["population_count"] for row in monthly["periods"]) == 10
    overall_amounts = impact["overall"]["effect"]["amounts"]
    assert overall_amounts["loan_amount"]["coverage_count"] == 9
    assert overall_amounts["loan_amount"]["sum"] == 1400.0
    assert overall_amounts["overdue_amount"]["coverage_count"] == 9
    assert overall_amounts["paired"]["coverage_count"] == 8
    assert overall_amounts["paired"]["loan_amount_sum"] == 1200.0
    assert overall_amounts["paired"]["overdue_amount_sum"] == 45.0
    assert overall_amounts["paired"]["overdue_rate"] == pytest.approx(45 / 1200)


def test_pool_impact_optional_columns_are_unavailable_not_zero() -> None:
    impact = _build(
        month_col=None,
        loan_amount_col=None,
        overdue_amount_col=None,
    )

    assert impact["monthly"] == {
        "status": "unavailable",
        "reason": "month_column_not_provided",
        "periods": [],
    }
    amounts = impact["overall"]["effect"]["amounts"]
    assert amounts["loan_amount"]["sum"] is None
    assert amounts["overdue_amount"]["coverage_count"] is None
    assert amounts["paired"]["overdue_rate"] is None


def test_pool_impact_rejects_invalid_month_and_amount_values() -> None:
    bad_month = _frame()
    bad_month.loc[0, "month"] = None
    with pytest.raises(StrategyError, match="unparseable"):
        _build(frame=bad_month)

    bad_amount = _frame()
    bad_amount.loc[0, "loan"] = -1
    with pytest.raises(StrategyError, match="non-negative"):
        _build(frame=bad_amount)


def _baseline_spec() -> dict:
    return StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(
            StrategyRuleSpec(
                rule_id="baseline-reject",
                priority=10,
                condition=_condition("<", 3),
                action=StrategyAction(type="reject", value="reject"),
            ),
        ),
    ).to_dict()


def test_pool_impact_baseline_deltas_are_explicit_and_monthly() -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    baseline = _baseline_spec()
    impact = _build(
        comparison_mode="vs_baseline",
        baseline_spec=baseline,
        baseline_binding={
            "strategy_id": "strategy-baseline",
            "strategy_type": "approval",
            "spec_hash": strategy_spec_hash(baseline),
        },
    )

    assert impact["baseline"]["status"] == "available"
    deltas = impact["baseline"]["overall"]["metric_deltas"]
    assert deltas["reject_count"] == 2
    assert deltas["approve_count"] == -5
    assert deltas["review_count"] == 3
    assert impact["baseline"]["monthly"]["status"] == "available"
    assert len(impact["baseline"]["monthly"]["periods"]) == 2


def test_pool_impact_baseline_cannot_silently_degrade_or_change_type() -> None:
    with pytest.raises(StrategyError, match="requires baseline_spec"):
        _build(comparison_mode="vs_baseline")

    other = StrategySpec(
        strategy_type="reject",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(),
    ).to_dict()
    from marvis.packs.strategy.dsl import strategy_spec_hash

    with pytest.raises(StrategyError, match="type must match"):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=other,
            baseline_binding={
                "strategy_id": "strategy-other",
                "strategy_type": "reject",
                "spec_hash": strategy_spec_hash(other),
            },
        )

    with pytest.raises(StrategyError, match="must not provide"):
        _build(
            baseline_spec=_baseline_spec(),
            baseline_binding={
                "strategy_id": "strategy-baseline",
                "strategy_type": "approval",
                "spec_hash": HASH_A,
            },
        )


def test_pool_impact_rejects_unsupported_type_and_sample_drift() -> None:
    with pytest.raises(StrategyError, match="approval/reject only"):
        _build(pool=_pool(strategy_type="limit"))

    sample = _sample_binding()
    sample["semantic_mapping_hash"] = HASH_D
    with pytest.raises(StrategyError, match="sample binding"):
        _build(sample_binding=sample)


def test_pool_impact_is_deterministic_canonical_and_tamper_evident() -> None:
    first = _build()
    second = _build()

    assert first == second
    raw = canonical_strategy_pool_impact_json(first)
    assert json.loads(raw) == first
    assert validate_strategy_pool_impact_assessment(first) == first

    forged = copy.deepcopy(first)
    forged["population"]["population_count"] += 1
    with pytest.raises(StrategyError, match="content_hash"):
        validate_strategy_pool_impact_assessment(forged)


def test_pool_impact_rejects_duplicate_or_conflicting_column_bindings() -> None:
    duplicate = _frame()
    duplicate.columns = ["score", "bad", "month", "loan", "loan"]
    with pytest.raises(StrategyError, match="duplicate columns"):
        _build(frame=duplicate)

    with pytest.raises(StrategyError, match="bindings must be distinct"):
        _build(month_col="bad")

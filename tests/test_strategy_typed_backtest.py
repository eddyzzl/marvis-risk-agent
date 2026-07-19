from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams
from marvis.packs.strategy.typed_backtest import (
    ApprovalProfitInputs,
    STRATEGY_BACKTEST_SCHEMA_VERSION,
    StrategyBacktestResult,
    run_typed_backtest,
)


def _rule(
    rule_id: str,
    priority: int,
    field: str,
    operator: str,
    value: object,
    action_type: str,
    action_value: object | None = None,
) -> StrategyRuleSpec:
    return StrategyRuleSpec(
        rule_id=rule_id,
        priority=priority,
        condition={
            "op": "compare",
            "field": field,
            "operator": operator,
            "value": value,
        },
        action=StrategyAction(type=action_type, value=action_value),
    )


def _decision_spec(strategy_type: str = "approval") -> StrategySpec:
    return StrategySpec(
        strategy_type=strategy_type,
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule("approve-high", 10, "score", ">=", 700, "approval"),
            _rule("review-low", 20, "score", "<", 500, "review"),
        ),
    )


def _segmentation_spec() -> StrategySpec:
    return StrategySpec(
        strategy_type="segmentation",
        default_action=StrategyAction(type="segment", value="C"),
        rules=(
            _rule("segment-b", 10, "x", "<", 2, "segment", "B"),
            _rule("segment-a", 20, "x", "<", 4, "segment", "A"),
        ),
    )


def _limit_spec(values: tuple[float, float, float, float]) -> StrategySpec:
    return StrategySpec(
        strategy_type="limit",
        default_action=StrategyAction(type="limit", value=values[3]),
        rules=tuple(
            _rule(
                f"limit-{position}",
                position,
                "x",
                "==",
                position,
                "limit",
                assigned,
            )
            for position, assigned in enumerate(values[:3])
        ),
    )


def _pricing_spec(values: tuple[float, float]) -> StrategySpec:
    return StrategySpec(
        strategy_type="pricing",
        default_action=StrategyAction(type="pricing", value=values[1]),
        rules=(
            _rule("price-first", 1, "x", "==", 0, "pricing", values[0]),
        ),
    )


def _breakdown(result: StrategyBacktestResult, key: str, value: object) -> dict:
    return next(row for row in result.breakdown if row[key] == value)


def test_approval_backtest_keeps_population_and_labeled_denominators_separate() -> None:
    frame = pd.DataFrame(
        {
            "score": [800, 650, 500, 750, 400, 600],
            "target": [0, 1, 1, None, math.nan, 0],
        },
        index=[10, 10, 20, 30, 40, 50],
    )

    result = run_typed_backtest(
        frame,
        _decision_spec(),
        target_col="target",
        strategy_id="approval-v1",
    )

    assert result.schema_version == STRATEGY_BACKTEST_SCHEMA_VERSION
    assert result.strategy_type == "approval"
    assert result.population_count == 6
    assert result.labeled_count == 4
    assert result.label_coverage == pytest.approx(4 / 6)
    assert result.metrics == {
        "overall_bad_count": 2,
        "overall_bad_rate": 0.5,
        "approve_count": 2,
        "approve_rate": pytest.approx(2 / 6),
        "approve_labeled_count": 1,
        "approve_bad_count": 0,
        "approve_bad_rate": 0.0,
        "reject_count": 3,
        "reject_rate": 0.5,
        "reject_labeled_count": 3,
        "reject_bad_count": 2,
        "reject_bad_rate": pytest.approx(2 / 3),
        "review_count": 1,
        "review_rate": pytest.approx(1 / 6),
        "review_labeled_count": 0,
        "review_bad_count": 0,
        "review_bad_rate": None,
    }
    assert _breakdown(result, "action", "review")["bad_rate"] is None
    assert "bad_capture_rate" not in result.metrics
    assert result.normalized_input["missing_label_policy"] == (
        "exclude_from_label_metrics"
    )
    assert result.warnings == ("2 population rows have no target label",)


def test_reject_backtest_adds_capture_metrics_without_counting_nan_as_good() -> None:
    frame = pd.DataFrame(
        {
            "score": [800, 650, 500, 750, 400, 600],
            "target": [0, 1, 1, None, math.nan, 0],
        }
    )

    result = run_typed_backtest(
        frame,
        _decision_spec("reject"),
        target_col="target",
        strategy_id="reject-v1",
    )

    assert result.metrics["bad_capture_rate"] == 1.0
    assert result.metrics["good_reject_rate"] == 0.5
    assert result.metrics["reject_bad_count"] == 2
    assert result.metrics["reject_labeled_count"] == 3


def test_approval_profit_uses_the_already_evaluated_approved_population() -> None:
    frame = pd.DataFrame(
        {
            "score": [800, 600],
            "target": [0, 1],
            "ead": [1000, 2000],
            "pd": [0.1, 0.2],
        }
    )

    result = run_typed_backtest(
        frame,
        _decision_spec(),
        target_col="target",
        approval_profit_inputs=ApprovalProfitInputs(
            params=ProfitParams(
                annual_rate=0.12,
                funding_rate=0.03,
                lgd=0.5,
                operating_cost_per_loan=10,
                term_months=12,
            ),
            ead_col="ead",
            pd_col="pd",
        ),
    )

    assert result.metrics["approve_count"] == 1
    assert result.economics["revenue"] == 120.0
    assert result.economics["expected_loss"] == 50.0
    assert result.economics["funding_cost"] == 30.0
    assert result.economics["operating_cost"] == 10.0
    assert result.economics["expected_profit"] == 30.0
    assert result.economics["profit_note"] is None
    assert result.normalized_input["approval_profit_input"] == {
        "ead_col": "ead",
        "pd_col": "pd",
        "params": {
            "annual_rate": 0.12,
            "funding_rate": 0.03,
            "lgd": 0.5,
            "operating_cost_per_loan": 10,
            "term_months": 12,
        },
    }


def test_empty_action_groups_have_undefined_bad_rate() -> None:
    spec = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval"),
    )

    result = run_typed_backtest(
        pd.DataFrame({"target": [0, 1]}),
        spec,
        target_col="target",
        strategy_id="approve-all",
    )

    assert result.metrics["reject_bad_rate"] is None
    assert result.metrics["review_bad_rate"] is None
    assert _breakdown(result, "action", "reject")["bad_rate"] is None


def test_segmentation_breakdown_is_stable_and_uses_labeled_bad_rate_for_lift() -> None:
    frame = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4, 5], "target": [1, None, 0, 1, 0, None]}
    )

    result = run_typed_backtest(
        frame,
        _segmentation_spec(),
        target_col="target",
        strategy_id="segments-v1",
    )

    assert [row["segment"] for row in result.breakdown] == ["A", "B", "C"]
    assert _breakdown(result, "segment", "A") == {
        "segment": "A",
        "count": 2,
        "share": pytest.approx(1 / 3),
        "labeled_count": 2,
        "bad_count": 1,
        "bad_rate": 0.5,
        "lift": 1.0,
    }
    assert _breakdown(result, "segment", "B")["bad_rate"] == 1.0
    assert _breakdown(result, "segment", "B")["lift"] == 2.0
    assert _breakdown(result, "segment", "C")["bad_rate"] == 0.0
    assert _breakdown(result, "segment", "C")["lift"] == 0.0


def test_unlabeled_segment_has_none_bad_rate_and_lift() -> None:
    spec = StrategySpec(
        strategy_type="segmentation",
        default_action=StrategyAction(type="segment", value="unlabeled"),
        rules=(_rule("known", 1, "x", "==", 1, "segment", "known"),),
    )

    result = run_typed_backtest(
        pd.DataFrame({"x": [1, 2], "target": [1, None]}),
        spec,
        target_col="target",
        strategy_id="segment-unlabeled",
    )

    row = _breakdown(result, "segment", "unlabeled")
    assert row["bad_count"] == 0
    assert row["bad_rate"] is None
    assert row["lift"] is None


def test_same_type_decision_baseline_emits_deterministic_three_by_three_matrix() -> None:
    baseline = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="review"),
        rules=(
            _rule("old-approve", 1, "x", "<", 2, "approval"),
            _rule("old-reject", 2, "x", "<", 4, "reject"),
        ),
    )
    candidate = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="reject"),
        rules=(
            _rule("new-review", 1, "x", "<", 1, "review"),
            _rule("new-approve", 2, "x", "<", 3, "approval"),
        ),
    )
    frame = pd.DataFrame({"x": range(6), "target": [0, 0, 1, 1, 0, 1]})

    first = run_typed_backtest(
        frame,
        candidate,
        target_col="target",
        strategy_id="candidate",
        baseline=baseline,
    )
    second = run_typed_backtest(
        frame,
        candidate,
        target_col="target",
        strategy_id="candidate",
        baseline=baseline,
    )

    expected_pairs = [
        (old, new)
        for old in ("approve", "reject", "review")
        for new in ("approve", "reject", "review")
    ]
    assert [
        (row["from_action"], row["to_action"]) for row in first.transitions
    ] == expected_pairs
    assert len(first.transitions) == 9
    assert sum(row["count"] for row in first.transitions) == len(frame)
    assert all(
        {"labeled_count", "bad_count", "bad_rate"} <= set(row)
        for row in first.transitions
    )
    assert first == second


def test_segmentation_baseline_emits_stably_sorted_segment_transitions() -> None:
    baseline = StrategySpec(
        strategy_type="segmentation",
        default_action=StrategyAction(type="segment", value="B"),
        rules=(_rule("old-a", 1, "x", "<", 2, "segment", "A"),),
    )
    frame = pd.DataFrame({"x": range(6), "target": [0, 0, 1, 1, 0, 1]})

    result = run_typed_backtest(
        frame,
        _segmentation_spec(),
        target_col="target",
        strategy_id="segments-v2",
        baseline=baseline,
    )

    assert [
        (row["from_segment"], row["to_segment"]) for row in result.transitions
    ] == [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A"), ("B", "B"), ("B", "C")]
    assert sum(row["count"] for row in result.transitions) == len(frame)


def test_cross_type_baseline_and_non_binary_target_fail_closed() -> None:
    frame = pd.DataFrame({"x": [0], "target": [2]})

    with pytest.raises(StrategyError, match="baseline strategy type"):
        run_typed_backtest(
            frame,
            _segmentation_spec(),
            target_col="target",
            baseline=_decision_spec(),
        )

    with pytest.raises(StrategyError, match="target must contain only 0, 1, or missing"):
        run_typed_backtest(
            frame,
            _segmentation_spec(),
            target_col="target",
        )


def test_result_round_trip_preserves_versioned_envelope() -> None:
    result = run_typed_backtest(
        pd.DataFrame({"target": [0, None]}),
        StrategySpec(
            strategy_type="approval",
            default_action=StrategyAction(type="approval"),
        ),
        target_col="target",
        strategy_id="roundtrip",
        economics={
            "expected_profit": None,
            "profit_note": "EAD/PD inputs were not supplied",
        },
    )

    assert StrategyBacktestResult.from_dict(result.to_dict()) == result
    with pytest.raises(StrategyError, match="unsupported backtest schema_version"):
        StrategyBacktestResult.from_dict(
            {**result.to_dict(), "schema_version": "strategy.backtest.v999"}
        )


def test_limit_backtest_reuses_typed_decisions_for_metrics_baseline_and_economics() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "target": [1, 1, None, 0],
        },
        index=[10, 10, 20, 30],
    )

    result = run_typed_backtest(
        frame,
        _limit_spec((2000, 1000, 2000, 0)),
        target_col="target",
        strategy_id="limit-v2",
        baseline=_limit_spec((1500, 1000, 2500, 0)),
        economics_inputs={
            "pd": pd.Series([0.1, 0.2, 0.3, 0.0], index=frame.index),
            "lgd": 0.5,
            "utilization": pd.Series([0.5, 0.25, 0.5, 0.0], index=frame.index),
        },
    )

    assert result.strategy_type == "limit"
    assert result.metrics == {
        "count": 4,
        "total_limit": 5000.0,
        "mean_limit": 1250.0,
        "min_limit": 0.0,
        "max_limit": 2000.0,
        "up_count": 1,
        "down_count": 1,
        "unchanged_count": 2,
        "total_limit_delta": 0.0,
    }
    assert [row["assigned_limit"] for row in result.breakdown] == [
        0.0,
        1000.0,
        2000.0,
    ]
    assert result.breakdown[-1]["labeled_count"] == 1
    assert result.breakdown[-1]["bad_rate"] == 1.0
    assert result.economics == {
        "expected_ead": 2250.0,
        "expected_loss": 225.0,
    }
    assert result.transitions == (
        {"direction": "up", "count": 1, "rate": 0.25},
        {"direction": "down", "count": 1, "rate": 0.25},
        {"direction": "unchanged", "count": 2, "rate": 0.5},
    )
    assert result.normalized_input["economics_input_kinds"] == {
        "lgd": "scalar",
        "pd": "series",
        "utilization": "series",
    }
    assert result.normalized_input["economics_input_evidence"]["lgd"] == {
        "kind": "scalar",
        "value": 0.5,
    }
    pd_evidence = result.normalized_input["economics_input_evidence"]["pd"]
    assert pd_evidence["kind"] == "series"
    assert pd_evidence["name"] is None
    assert pd_evidence["row_count"] == len(frame)
    assert len(pd_evidence["content_hash"]) == 64
    assert set(pd_evidence) == {"kind", "name", "row_count", "content_hash"}
    assert StrategyBacktestResult.from_dict(result.to_dict()) == result
    json.dumps(result.to_dict(), allow_nan=False)


def test_value_action_legacy_output_alias_cannot_override_typed_values() -> None:
    spec = StrategySpec(
        strategy_type="limit",
        default_action=StrategyAction(
            type="limit",
            value=1000,
            output_value={"legacy": "fallback"},
        ),
        rules=(
            StrategyRuleSpec(
                rule_id="limit-hit",
                priority=1,
                condition={
                    "op": "compare",
                    "field": "x",
                    "operator": "==",
                    "value": 0,
                },
                action=StrategyAction(
                    type="limit",
                    value=2000,
                    output_value=999999,
                ),
            ),
        ),
    )

    result = run_typed_backtest(
        pd.DataFrame({"x": [0, 1], "target": [1, 0]}),
        spec,
        target_col="target",
    )

    assert result.metrics["total_limit"] == 3000.0
    assert [row["assigned_limit"] for row in result.breakdown] == [1000.0, 2000.0]


def test_pricing_backtest_computes_risk_tiers_repricing_and_profit_chain() -> None:
    frame = pd.DataFrame({"x": [0, 1], "target": [0, 1]})

    result = run_typed_backtest(
        frame,
        _pricing_spec((0.12, 0.18)),
        target_col="target",
        strategy_id="pricing-v2",
        baseline=_pricing_spec((0.10, 0.16)),
        economics_inputs={
            "ead": pd.Series([1000, 2000]),
            "pd": pd.Series([0.10, 0.05]),
            "lgd": pd.Series([0.5, 0.4]),
            "funding_rate": pd.Series([0.03, 0.04]),
            "term_months": pd.Series([12, 6]),
            "operating_cost_per_loan": pd.Series([10, 20]),
        },
    )

    assert result.metrics == {
        "count": 2,
        "mean_rate": 0.15,
        "repriced_up_count": 2,
        "repriced_down_count": 0,
        "unchanged_count": 0,
    }
    assert [row["assigned_rate"] for row in result.breakdown] == [0.12, 0.18]
    assert result.economics["total_ead"] == 3000.0
    assert result.economics["ead_weighted_rate"] == pytest.approx(0.16)
    assert result.economics["revenue"] == 300.0
    assert result.economics["expected_loss"] == 90.0
    assert result.economics["profit"] == 110.0
    assert result.economics["baseline_profit"] == 70.0
    assert result.economics["profit_delta_vs_baseline"] == 40.0
    assert result.transitions == (
        {"direction": "repriced_up", "count": 2, "rate": 1.0},
        {"direction": "repriced_down", "count": 0, "rate": 0.0},
        {"direction": "unchanged", "count": 0, "rate": 0.0},
    )
    assert StrategyBacktestResult.from_dict(result.to_dict()) == result
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("spec", "expected_metrics"),
    [
        (
            _limit_spec((1000, 1000, 1000, 1000)),
            {
                "up_count": None,
                "down_count": None,
                "unchanged_count": None,
                "total_limit_delta": None,
            },
        ),
        (
            _pricing_spec((0.1, 0.1)),
            {
                "repriced_up_count": None,
                "repriced_down_count": None,
                "unchanged_count": None,
            },
        ),
    ],
)
def test_economic_strategy_without_economics_or_baseline_is_explicitly_unavailable(
    spec: StrategySpec, expected_metrics: dict[str, None]
) -> None:
    result = run_typed_backtest(
        pd.DataFrame({"x": [0], "target": [0]}),
        spec,
        target_col="target",
    )

    assert result.economics == {}
    assert all(result.metrics[key] is value for key, value in expected_metrics.items())
    assert result.transitions == ()
    assert result.warnings[-1] == (
        f"{spec.strategy_type} economics inputs were not supplied"
    )


@pytest.mark.parametrize(
    ("spec", "economics_inputs"),
    [
        (_limit_spec((1000, 1000, 1000, 1000)), {"pd": 0.1}),
        (
            _pricing_spec((0.1, 0.1)),
            {"ead": 1000, "pd": 0.1, "lgd": 0.5},
        ),
    ],
)
def test_economic_strategy_partial_economics_inputs_fail_closed(
    spec: StrategySpec, economics_inputs: dict[str, float]
) -> None:
    with pytest.raises(StrategyError, match="requires all inputs"):
        run_typed_backtest(
            pd.DataFrame({"x": [0], "target": [0]}),
            spec,
            target_col="target",
            economics_inputs=economics_inputs,
        )


def test_economics_input_contract_rejects_wrong_strategy_and_legacy_output_collision() -> None:
    frame = pd.DataFrame({"score": [800], "target": [0]})

    with pytest.raises(StrategyError, match="economics_inputs is only supported"):
        run_typed_backtest(
            frame,
            _decision_spec(),
            target_col="target",
            economics_inputs={"pd": 0.1},
        )

    with pytest.raises(StrategyError, match="cannot be combined"):
        run_typed_backtest(
            frame.assign(x=0),
            _limit_spec((1000, 1000, 1000, 1000)),
            target_col="target",
            economics={"expected_loss": 0},
            economics_inputs={"pd": 0.1, "lgd": 0.5, "utilization": 0.5},
        )


def test_economic_input_evidence_distinguishes_scalar_and_series_assumptions() -> None:
    frame = pd.DataFrame({"x": [0, 1], "target": [0, 1]})
    pd_first = pd.Series([0.1, 0.2], name="pd_12m", index=frame.index)
    pd_second = pd.Series([0.1, 0.3], name="pd_12m", index=frame.index)

    first = run_typed_backtest(
        frame,
        _limit_spec((1000, 2000, 2000, 2000)),
        target_col="target",
        economics_inputs={"pd": pd_first, "lgd": 0.5, "utilization": 0.5},
    )
    changed_series = run_typed_backtest(
        frame,
        _limit_spec((1000, 2000, 2000, 2000)),
        target_col="target",
        economics_inputs={"pd": pd_second, "lgd": 0.5, "utilization": 0.5},
    )
    changed_scalar = run_typed_backtest(
        frame,
        _limit_spec((1000, 2000, 2000, 2000)),
        target_col="target",
        economics_inputs={"pd": pd_first, "lgd": 0.6, "utilization": 0.5},
    )

    first_evidence = first.normalized_input["economics_input_evidence"]
    series_evidence = changed_series.normalized_input["economics_input_evidence"]
    scalar_evidence = changed_scalar.normalized_input["economics_input_evidence"]
    assert first_evidence["pd"]["name"] == "pd_12m"
    assert first_evidence["pd"]["row_count"] == len(frame)
    assert first_evidence["pd"]["content_hash"] != series_evidence["pd"][
        "content_hash"
    ]
    assert first_evidence["lgd"]["value"] == 0.5
    assert scalar_evidence["lgd"]["value"] == 0.6
    assert first.normalized_input != changed_series.normalized_input
    assert first.normalized_input != changed_scalar.normalized_input
    assert first.to_dict() != changed_series.to_dict()
    assert first.to_dict() != changed_scalar.to_dict()


def _semantic_validation_result(strategy_type: str) -> StrategyBacktestResult:
    frame = pd.DataFrame(
        {
            "score": [800, 600, 400, 750],
            "x": [0, 1, 2, 3],
            "target": [0, 1, None, 1],
        }
    )
    if strategy_type in {"approval", "reject"}:
        spec = _decision_spec(strategy_type)
        return run_typed_backtest(
            frame,
            spec,
            target_col="target",
            baseline=spec,
        )
    if strategy_type == "limit":
        spec = _limit_spec((1000, 1500, 2000, 2500))
        return run_typed_backtest(
            frame,
            spec,
            target_col="target",
            baseline=spec,
            economics_inputs={
                "pd": pd.Series([0.1] * len(frame), name="pd_12m"),
                "lgd": 0.5,
                "utilization": 0.5,
            },
        )
    if strategy_type == "pricing":
        spec = _pricing_spec((0.1, 0.2))
        return run_typed_backtest(
            frame,
            spec,
            target_col="target",
            baseline=spec,
            economics_inputs={
                "ead": 1000,
                "pd": 0.1,
                "lgd": 0.5,
                "funding_rate": 0.03,
                "term_months": 12,
                "operating_cost_per_loan": 10,
            },
        )
    spec = _segmentation_spec()
    return run_typed_backtest(
        frame,
        spec,
        target_col="target",
        baseline=spec,
    )


def _tampered_semantic_payload(strategy_type: str) -> dict[str, object]:
    payload = json.loads(
        json.dumps(_semantic_validation_result(strategy_type).to_dict())
    )
    if strategy_type == "approval":
        payload["metrics"]["approve_count"] += 1
    elif strategy_type == "reject":
        del payload["metrics"]["bad_capture_rate"]
    elif strategy_type == "limit":
        payload["metrics"]["unsupported_metric"] = 1
    elif strategy_type == "pricing":
        payload["breakdown"][0]["assigned_rate"] = 1.5
    else:
        payload["metrics"]["segment_count"] += 1
    return payload


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
def test_type_specific_semantics_fail_closed_for_deserialization_and_construction(
    strategy_type: str,
) -> None:
    payload = _tampered_semantic_payload(strategy_type)

    with pytest.raises(StrategyError):
        StrategyBacktestResult.from_dict(payload)
    with pytest.raises(StrategyError):
        StrategyBacktestResult(**payload)


def test_transition_economics_and_normalized_evidence_tampering_fail_closed() -> None:
    approval = _semantic_validation_result("approval").to_dict()
    approval["transitions"][0]["rate"] = 0.123
    with pytest.raises(StrategyError, match="transitions\\[0\\].rate"):
        StrategyBacktestResult.from_dict(approval)

    pricing = _semantic_validation_result("pricing").to_dict()
    pricing["economics"]["profit"] += 1
    with pytest.raises(StrategyError, match="economics.profit"):
        StrategyBacktestResult.from_dict(pricing)

    limit_result = _semantic_validation_result("limit").to_dict()
    limit_result["normalized_input"]["economics_input_evidence"]["pd"][
        "row_count"
    ] += 1
    with pytest.raises(StrategyError, match="row_count"):
        StrategyBacktestResult.from_dict(limit_result)


def test_action_transition_candidate_columns_must_match_current_metrics() -> None:
    payload = _semantic_validation_result("approval").to_dict()
    approve_to_approve = payload["transitions"][0]
    approve_to_reject = payload["transitions"][1]
    approve_to_approve.update(
        {
            "count": 1,
            "rate": 0.5,
            "population_share": 0.25,
            "labeled_count": 1,
            "bad_count": 0,
            "bad_rate": 0.0,
        }
    )
    approve_to_reject.update(
        {
            "count": 1,
            "rate": 0.5,
            "population_share": 0.25,
            "labeled_count": 1,
            "bad_count": 1,
            "bad_rate": 1.0,
        }
    )

    with pytest.raises(StrategyError, match="transitions.to_action.approve.count"):
        StrategyBacktestResult.from_dict(payload)


@pytest.mark.parametrize(
    "economics",
    [
        {"expected_profit": 1.0, "profit_note": None},
        {"expected_profit": 1.0, "profit_note": "caller supplied"},
        {"expected_profit": None, "profit_note": None},
        {"expected_profit": None, "profit_note": ""},
        {"expected_profit": None, "profit_note": "   "},
    ],
)
def test_caller_cannot_inject_profit_or_omit_unavailability_reason(
    economics: dict[str, object],
) -> None:
    with pytest.raises(StrategyError):
        run_typed_backtest(
            pd.DataFrame({"score": [800], "target": [0]}),
            _decision_spec(),
            target_col="target",
            economics=economics,
        )


def test_unknown_nested_fields_and_non_list_warnings_fail_closed() -> None:
    result = _semantic_validation_result("segmentation").to_dict()
    result["normalized_input"]["unsupported"] = True
    with pytest.raises(StrategyError, match="unsupported fields"):
        StrategyBacktestResult.from_dict(result)

    result = _semantic_validation_result("segmentation").to_dict()
    result["warnings"] = "not-a-list"
    with pytest.raises(StrategyError, match="warnings must be a list"):
        StrategyBacktestResult.from_dict(result)

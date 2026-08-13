"""Tests for the bounded counterfactual "what-if" sandbox kernel."""

from __future__ import annotations

import copy

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from marvis.decision_twin.counterfactual import (
    BaseStrategyRef,
    CanonicalStrategyRef,
    CounterfactualDelta,
    CounterfactualDeltaError,
    CounterfactualResult,
    CounterfactualRowBudgetError,
    CounterfactualSampleRef,
    CounterfactualSpec,
    CounterfactualSpecError,
    run_bounded_counterfactual,
)
from marvis.packs.strategy.dsl import strategy_spec_hash

_SHA = "0" * 64


def _base_strategy(*, threshold: int = 600) -> dict:
    """Single score floor rule: score >= threshold approves, else default reject."""
    return {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "approval",
        "match_policy": "first_match",
        "default_action": {
            "type": "reject",
            "value": "reject",
            "reason_code": None,
            "stop": True,
        },
        "rules": [
            {
                "rule_id": "score_floor",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": ">=",
                    "value": threshold,
                    "missing": "no_match",
                },
                "action": {
                    "type": "approval",
                    "value": "approve",
                    "reason_code": None,
                    "stop": True,
                },
            },
        ],
        "metadata": {"lineage": {"source": "test"}},
    }


def _reject_rule_strategy() -> dict:
    """Reject risky (score < 600), otherwise approve."""
    return {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "reject",
        "match_policy": "first_match",
        "default_action": {
            "type": "approval",
            "value": "approve",
            "reason_code": None,
            "stop": True,
        },
        "rules": [
            {
                "rule_id": "reject_risky",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "score",
                    "operator": "<",
                    "value": 600,
                    "missing": "no_match",
                },
                "action": {
                    "type": "reject",
                    "value": "reject",
                    "reason_code": None,
                    "stop": True,
                },
            },
        ],
        "metadata": {"lineage": {"source": "test"}},
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [500, 540, 600, 640, 680, 720, 760, 800],
            "target": [1, 0, 1, 1, 1, 0, 0, 0],
            "exposure": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        }
    )


def _sample_ref() -> CounterfactualSampleRef:
    return CounterfactualSampleRef(
        artifact_id="artifact-1",
        artifact_content_hash=_SHA,
        sample_design_id="sd-1",
        sample_design_content_hash="1" * 64,
        partition="development",
    )


def _base_ref(base_strategy: dict) -> BaseStrategyRef:
    return BaseStrategyRef(
        kind="canonical_strategy",
        canonical_strategy=CanonicalStrategyRef(
            strategy_id="strategy-1",
            strategy_version=1,
            strategy_content_hash=strategy_spec_hash(base_strategy),
        ),
    )


def test_threshold_delta_moves_rates_in_the_right_direction_with_golden_numbers():
    base_strategy = _base_strategy(threshold=600)
    spec = CounterfactualSpec(
        base=_base_ref(base_strategy),
        deltas=(
            CounterfactualDelta(
                kind="threshold",
                rule_id="score_floor",
                field="score",
                attribute="value",
                value=680,
            ),
        ),
        sample=_sample_ref(),
    )

    result = run_bounded_counterfactual(
        spec=spec,
        base_strategy=base_strategy,
        frame=_frame(),
        target_col="target",
        bad_value=1,
        exposure_col="exposure",
    )

    # Base: 6 approvals (score >= 600), 3 bads among them.
    assert result.base.count == 8
    assert result.base.approval_count == 6
    assert result.base.approval_rate == pytest.approx(0.75)
    assert result.base.bad_count == 3
    assert result.base.bad_rate == pytest.approx(0.5)
    assert result.base.exposure_sum == pytest.approx(3300.0)
    assert result.base.expected_loss == pytest.approx(1200.0)

    # Delta (threshold 680): 4 approvals, 1 bad among them.
    assert result.delta.approval_count == 4
    assert result.delta.approval_rate == pytest.approx(0.5)
    assert result.delta.bad_count == 1
    assert result.delta.bad_rate == pytest.approx(0.25)
    assert result.delta.exposure_sum == pytest.approx(2600.0)
    assert result.delta.expected_loss == pytest.approx(500.0)

    # Tightening the threshold lowers approval and bad rates.
    assert result.comparison.approval_rate_delta == pytest.approx(-0.25)
    assert result.comparison.bad_rate_delta == pytest.approx(-0.25)
    assert result.comparison.approval_count_delta == -2
    assert result.comparison.bad_count_delta == -2
    assert result.comparison.exposure_sum_delta == pytest.approx(-700.0)
    assert result.comparison.expected_loss_delta == pytest.approx(-700.0)


def test_action_flip_delta_increases_approvals():
    base_strategy = _reject_rule_strategy()
    spec = CounterfactualSpec(
        base=_base_ref(base_strategy),
        deltas=(CounterfactualDelta(kind="action_flip", rule_id="reject_risky"),),
        sample=_sample_ref(),
    )

    result = run_bounded_counterfactual(
        spec=spec,
        base_strategy=base_strategy,
        frame=_frame(),
        target_col="target",
        bad_value=1,
        exposure_col=None,
    )

    # Base: score < 600 rejected -> 6 approvals.
    assert result.base.approval_count == 6
    assert result.base.approval_rate == pytest.approx(0.75)

    # Flipping reject -> approve makes every row approve -> 8 approvals.
    assert result.delta.approval_count == 8
    assert result.delta.approval_rate == pytest.approx(1.0)
    assert result.comparison.approval_count_delta == 2
    assert result.comparison.approval_rate_delta == pytest.approx(0.25)
    assert result.delta.exposure_sum is None
    assert result.delta.expected_loss is None


def test_row_budget_exceeded_fails_closed():
    base_strategy = _base_strategy()
    spec = CounterfactualSpec(
        base=_base_ref(base_strategy),
        deltas=(
            CounterfactualDelta(
                kind="threshold",
                rule_id="score_floor",
                field="score",
                attribute="value",
                value=700,
            ),
        ),
        sample=_sample_ref(),
        row_budget=4,
    )

    with pytest.raises(CounterfactualRowBudgetError):
        run_bounded_counterfactual(
            spec=spec,
            base_strategy=base_strategy,
            frame=_frame(),
            target_col="target",
        )


def test_fuzzy_delta_kind_is_rejected():
    with pytest.raises(CounterfactualDeltaError):
        CounterfactualDelta.from_value(
            {"kind": "target_approval_rate", "value": 0.3}
        )


def test_threshold_delta_requires_exact_match():
    base_strategy = _base_strategy()
    spec = CounterfactualSpec(
        base=_base_ref(base_strategy),
        deltas=(
            CounterfactualDelta(
                kind="threshold",
                rule_id="score_floor",
                field="unknown_field",
                attribute="value",
                value=700,
            ),
        ),
        sample=_sample_ref(),
    )
    with pytest.raises(CounterfactualDeltaError):
        run_bounded_counterfactual(
            spec=spec,
            base_strategy=base_strategy,
            frame=_frame(),
            target_col="target",
        )


def test_threshold_delta_rejects_invalid_attribute_and_list_value():
    with pytest.raises(CounterfactualDeltaError):
        CounterfactualDelta(
            kind="threshold",
            rule_id="r",
            field="score",
            attribute="threshold",
            value=700,
        )
    with pytest.raises(CounterfactualDeltaError):
        CounterfactualDelta(
            kind="threshold",
            rule_id="r",
            field="score",
            attribute="value",
            value=[700],
        )


def test_action_flip_rejects_non_approval_reject_action():
    review_strategy = _base_strategy()
    review_strategy["rules"][0]["action"] = {
        "type": "review",
        "value": "review",
        "reason_code": None,
        "stop": True,
    }
    review_spec = CounterfactualSpec(
        base=BaseStrategyRef(
            kind="canonical_strategy",
            canonical_strategy=CanonicalStrategyRef(
                strategy_id="strategy-1",
                strategy_version=1,
                strategy_content_hash=strategy_spec_hash(review_strategy),
            ),
        ),
        deltas=(CounterfactualDelta(kind="action_flip", rule_id="score_floor"),),
        sample=_sample_ref(),
    )
    with pytest.raises(CounterfactualDeltaError):
        run_bounded_counterfactual(
            spec=review_spec,
            base_strategy=review_strategy,
            frame=_frame(),
            target_col="target",
        )


def test_canonical_ref_hash_mismatch_fails_closed():
    base_strategy = _base_strategy()
    wrong_ref = BaseStrategyRef(
        kind="canonical_strategy",
        canonical_strategy=CanonicalStrategyRef(
            strategy_id="strategy-1",
            strategy_version=1,
            strategy_content_hash="f" * 64,
        ),
    )
    spec = CounterfactualSpec(
        base=wrong_ref,
        deltas=(
            CounterfactualDelta(
                kind="threshold",
                rule_id="score_floor",
                field="score",
                attribute="value",
                value=700,
            ),
        ),
        sample=_sample_ref(),
    )
    with pytest.raises(CounterfactualSpecError):
        run_bounded_counterfactual(
            spec=spec,
            base_strategy=base_strategy,
            frame=_frame(),
            target_col="target",
        )


def test_empty_deltas_rejected():
    with pytest.raises(CounterfactualSpecError):
        CounterfactualSpec(
            base=_base_ref(_base_strategy()),
            deltas=(),
            sample=_sample_ref(),
        )


def test_run_has_no_state_change_and_is_proposal_only():
    base_strategy = _base_strategy(threshold=600)
    frame = _frame()
    base_before = copy.deepcopy(base_strategy)
    frame_before = frame.copy(deep=True)
    spec = CounterfactualSpec(
        base=_base_ref(base_strategy),
        deltas=(
            CounterfactualDelta(
                kind="threshold",
                rule_id="score_floor",
                field="score",
                attribute="value",
                value=680,
            ),
        ),
        sample=_sample_ref(),
    )

    result = run_bounded_counterfactual(
        spec=spec,
        base_strategy=base_strategy,
        frame=frame,
        target_col="target",
        bad_value=1,
        exposure_col="exposure",
    )

    assert isinstance(result, CounterfactualResult)
    assert result.counterfactual_only is True
    assert result.authority == "proposal_only"
    assert result.automatic_action_permitted is False
    assert base_strategy == base_before
    assert_frame_equal(frame, frame_before)

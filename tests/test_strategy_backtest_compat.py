from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from marvis.packs.strategy.backtest_compat import (
    approval_backtest_projection,
    backtest_record_payload,
)
from marvis.packs.strategy.contracts import BacktestResult
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams
from marvis.packs.strategy.typed_backtest import (
    ApprovalProfitInputs,
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.packs.strategy.tools import _backtest_id


def _rule(
    rule_id: str,
    priority: int,
    condition: dict,
    action_type: str,
) -> StrategyRuleSpec:
    return StrategyRuleSpec(
        rule_id=rule_id,
        priority=priority,
        condition=condition,
        action=StrategyAction(type=action_type),
    )


def _typed_result(*, approved_labels: bool = True) -> StrategyBacktestResult:
    candidate = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="review"),
        rules=(
            _rule(
                "candidate-approve-high",
                10,
                {"op": "compare", "field": "score", "operator": "==", "value": 800},
                "approval",
            ),
            _rule(
                "candidate-approve-mid",
                20,
                {"op": "compare", "field": "score", "operator": "==", "value": 550},
                "approval",
            ),
            _rule(
                "candidate-reject",
                30,
                {"op": "compare", "field": "score", "operator": "==", "value": 650},
                "reject",
            ),
        ),
    )
    baseline = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="review"),
        rules=(
            _rule(
                "baseline-approve",
                10,
                {"op": "compare", "field": "score", "operator": ">=", "value": 650},
                "approval",
            ),
            _rule(
                "baseline-reject",
                20,
                {"op": "compare", "field": "score", "operator": "<", "value": 500},
                "reject",
            ),
        ),
    )
    target: list[int | None] = [0, 1, 0, 1]
    if not approved_labels:
        target[0] = None
        target[3] = None
    return run_typed_backtest(
        pd.DataFrame(
            {
                "score": [800, 650, 400, 550],
                "bad": target,
                "ead": [100.0] * 4,
                "pd": [0.0] * 4,
            }
        ),
        candidate,
        target_col="bad",
        strategy_id="strategy-1",
        baseline=baseline,
        approval_profit_inputs=ApprovalProfitInputs(
            params=ProfitParams(
                annual_rate=0.125,
                funding_rate=0.0,
                lgd=0.5,
                operating_cost_per_loan=0.0,
                term_months=12,
            ),
            ead_col="ead",
            pd_col="pd",
        ),
    )


def _segmentation_result() -> StrategyBacktestResult:
    spec = StrategySpec(
        strategy_type="segmentation",
        default_action=StrategyAction(type="segment", value="standard"),
        rules=(
            StrategyRuleSpec(
                rule_id="prime",
                priority=10,
                condition={
                    "op": "compare",
                    "field": "score",
                    "operator": ">=",
                    "value": 650,
                },
                action=StrategyAction(type="segment", value="prime"),
            ),
        ),
    )
    return run_typed_backtest(
        pd.DataFrame({"score": [800, 650, 400, 550], "bad": [0, 1, 0, 1]}),
        spec,
        target_col="bad",
        strategy_id="segment-1",
    )


def test_typed_approval_projection_preserves_legacy_fields_without_overriding_envelope() -> None:
    result = _typed_result()

    projection = approval_backtest_projection(result)

    assert projection["approval_rate"] == 0.5
    assert projection["approved_bad_rate"] == 0.5
    assert projection["expected_profit"] == 25.0
    assert projection["swap_in_count"] == 1
    assert projection["swap_in_bad_rate"] == 1.0
    assert projection["swap_out_count"] == 1
    assert projection["swap_out_bad_rate"] == 1.0
    assert projection["by_segment"][2]["decision"] == "review"
    payload = backtest_record_payload(result)
    assert payload["schema_version"] == "strategy.backtest.v2"
    assert "approval_rate" not in payload


def test_projection_refuses_nonapproval_backtests() -> None:
    result = _segmentation_result()

    with pytest.raises(StrategyError, match="not defined"):
        approval_backtest_projection(result)


def test_canonical_projection_can_preserve_undefined_group_rates_for_memory() -> None:
    result = _typed_result(approved_labels=False)

    assert approval_backtest_projection(result)["approved_bad_rate"] == 0.0
    assert (
        approval_backtest_projection(
            result,
            preserve_undefined_rates=True,
        )["approved_bad_rate"]
        is None
    )


def test_legacy_payload_and_projection_remain_flat() -> None:
    result = BacktestResult(
        strategy_id="legacy",
        approval_rate=0.8,
        approved_count=8,
        approved_bad_rate=0.1,
        rejected_bad_rate=0.5,
        expected_profit=100.0,
        swap_in_count=1,
        swap_out_count=2,
        swap_in_bad_rate=0.0,
        swap_out_bad_rate=0.5,
        by_segment=({"decision": "approve", "count": 8},),
    )

    payload = backtest_record_payload(result)

    assert approval_backtest_projection(result) == payload
    assert payload["by_segment"] == [{"decision": "approve", "count": 8}]

    historical_payload = {"dataset_id": "dataset-1", "result": payload}
    historical_digest = hashlib.sha256(
        json.dumps(
            historical_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert _backtest_id("dataset-1", result) == f"backtest-{historical_digest[:12]}"


def test_typed_backtest_id_uses_canonical_envelope_and_dataset_identity() -> None:
    result = _typed_result()

    first = _backtest_id("dataset-1", result)

    assert first == _backtest_id(
        "dataset-1",
        StrategyBacktestResult.from_dict(result.to_dict()),
    )
    assert first != _backtest_id("dataset-2", result)

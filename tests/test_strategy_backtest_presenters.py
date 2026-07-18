from __future__ import annotations

import pandas as pd
import pytest

from marvis.agent.memory_bridge import _approval_backtest_memory_metrics
from marvis.agent.renderers import _render_backtest_strategy
from marvis.packs.strategy.contracts import BacktestResult
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.doc import render_strategy_doc_markdown
from marvis.packs.strategy.profit import ProfitParams
from marvis.packs.strategy.typed_backtest import (
    ApprovalProfitInputs,
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


def _spec(strategy_type: str, *, baseline: bool = False) -> StrategySpec:
    if strategy_type in {"approval", "reject"}:
        if baseline:
            return StrategySpec(
                strategy_type=strategy_type,
                default_action=StrategyAction(type="review"),
                rules=(
                    _rule("baseline-approve", 10, "score", ">=", 700, "approval"),
                    _rule("baseline-reject", 20, "score", "<", 500, "reject"),
                ),
            )
        return StrategySpec(
            strategy_type=strategy_type,
            default_action=StrategyAction(type="review"),
            rules=(
                _rule("approve", 10, "score", ">=", 600, "approval"),
                _rule("reject", 20, "score", "<", 400, "reject"),
            ),
        )
    if strategy_type == "limit":
        return StrategySpec(
            strategy_type="limit",
            default_action=StrategyAction(
                type="limit",
                value=1600.0 if baseline else 1500.0,
            ),
            rules=(
                _rule(
                    "limit-low",
                    10,
                    "x",
                    "<",
                    5,
                    "limit",
                    900.0 if baseline else 1000.0,
                ),
            ),
        )
    if strategy_type == "pricing":
        return StrategySpec(
            strategy_type="pricing",
            default_action=StrategyAction(
                type="pricing",
                value=0.20 if baseline else 0.18,
            ),
            rules=(
                _rule(
                    "price-low",
                    10,
                    "x",
                    "<",
                    5,
                    "pricing",
                    0.10 if baseline else 0.12,
                ),
            ),
        )
    return StrategySpec(
        strategy_type="segmentation",
        default_action=StrategyAction(
            type="segment",
            value="legacy" if baseline else "standard",
        ),
        rules=(
            _rule(
                "segment-prime",
                10,
                "x",
                "<",
                5,
                "segment",
                "standard" if baseline else "prime",
            ),
        ),
    )


def _frame(*, approved_labels: bool = True) -> pd.DataFrame:
    target: list[float | None] = [0, 0, 1, 0, 0, None, 1, 0, 1, None]
    if not approved_labels:
        target[:6] = [None] * 6
    return pd.DataFrame(
        {
            "score": [900, 800, 700, 650, 620, 600, 500, 400, 300, 200],
            "x": list(range(10)),
            "target": target,
            "ead": [164.0] * 10,
            "pd": [0.0] * 10,
        }
    )


def _approval_profit_inputs() -> ApprovalProfitInputs:
    return ApprovalProfitInputs(
        params=ProfitParams(
            annual_rate=0.125,
            funding_rate=0.0,
            lgd=0.50,
            operating_cost_per_loan=0.0,
            term_months=12,
        ),
        ead_col="ead",
        pd_col="pd",
    )


def _envelope(
    strategy_type: str,
    *,
    baseline: bool = False,
    approved_labels: bool = True,
    approval_profit_inputs: ApprovalProfitInputs | None = None,
) -> dict:
    result = run_typed_backtest(
        _frame(approved_labels=approved_labels),
        _spec(strategy_type),
        target_col="target",
        strategy_id=f"{strategy_type}-1",
        baseline=_spec(strategy_type, baseline=True) if baseline else None,
        approval_profit_inputs=approval_profit_inputs,
    )
    return result.to_dict()


def _approval_envelope(
    strategy_type: str = "approval",
    *,
    baseline: bool = False,
    approved_labels: bool = True,
    include_profit: bool = True,
) -> dict:
    return _envelope(
        strategy_type,
        baseline=baseline,
        approved_labels=approved_labels,
        approval_profit_inputs=_approval_profit_inputs() if include_profit else None,
    )


def test_typed_approval_renderer_prefers_canonical_envelope_over_flat_aliases() -> None:
    payload = {
        **_approval_envelope(),
        "approval_rate": 0.99,
        "approved_bad_rate": 0.99,
        "expected_profit": -999.0,
    }

    text, tables = _render_backtest_strategy(payload)

    assert "审批率 60.0%" in text
    assert "99.0%" not in text
    assert "预期利润 123.0000" in text
    rows = {row[0]: row[1] for row in tables[0]["rows"]}
    assert rows["审批率"] == "60.0%"
    assert tables[1]["title"] == "按决策分组"


@pytest.mark.parametrize(
    ("payload", "text_fragment", "table_title"),
    [
        (
            _approval_envelope("reject"),
            "拒绝策略回测完成",
            "策略回测摘要",
        ),
        (
            _envelope("limit"),
            "额度策略回测完成",
            "额度策略回测摘要",
        ),
        (
            _envelope("pricing"),
            "定价策略回测完成",
            "定价策略回测摘要",
        ),
        (
            _envelope("segmentation"),
            "分群策略回测完成",
            "客群风险分布",
        ),
    ],
)
def test_typed_renderers_use_strategy_specific_copy_and_tables(
    payload: dict,
    text_fragment: str,
    table_title: str,
) -> None:
    text, tables = _render_backtest_strategy(payload)

    assert text_fragment in text
    assert table_title in {table["title"] for table in tables}
    if payload["strategy_type"] not in {"approval", "reject"}:
        assert "审批率" not in text


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_approval_envelope(), "回测类型：准入策略"),
        (
            _envelope("limit"),
            "回测类型：额度策略",
        ),
        (
            _envelope("pricing"),
            "回测类型：定价策略",
        ),
        (
            _envelope("segmentation"),
            "回测类型：分群策略",
        ),
    ],
)
def test_strategy_document_renders_typed_backtest_without_recalculation(
    payload: dict,
    expected: str,
) -> None:
    markdown, _sections = render_strategy_doc_markdown(
        strategy={
            "id": "strategy-1",
            "strategy_type": payload["strategy_type"],
            "rules": [],
            "default_decision": "n/a",
        },
        meta={"version": 1, "status": "draft"},
        backtests=[payload],
        artifacts=[],
        band_stats=[],
    )

    assert expected in markdown
    if payload["strategy_type"] not in {"approval", "reject"}:
        assert "- 审批率：" not in markdown


def test_nonapproval_document_never_renders_approval_band_schema() -> None:
    markdown, sections = render_strategy_doc_markdown(
        strategy={
            "id": "limit-1",
            "strategy_type": "limit",
            "rules": [],
            "default_decision": 1000,
        },
        meta={"version": 1, "status": "draft"},
        backtests=[_envelope("limit")],
        artifacts=[],
        band_stats=[
            {
                "lo": 0,
                "hi": 100,
                "cum_approval_rate": 0.99,
                "decision": "approve",
            }
        ],
    )

    assert "类型化分布" in sections
    assert "累计审批率" not in markdown
    assert "0.99" not in markdown
    assert "审批" not in markdown


def test_typed_document_keeps_profit_reason_and_baseline_transition_evidence() -> None:
    payload = _approval_envelope(baseline=True, include_profit=False)
    payload["economics"] = {
        "expected_profit": None,
        "profit_note": "缺少 EAD/PD，利润不可用",
    }

    text, _tables = _render_backtest_strategy(payload)
    markdown, _sections = render_strategy_doc_markdown(
        strategy={
            "id": "approval-1",
            "strategy_type": "approval",
            "rules": [],
            "default_decision": "approve",
        },
        meta={"version": 1, "status": "draft"},
        backtests=[payload],
        artifacts=[],
        band_stats=[],
    )

    assert "利润口径提示：缺少 EAD/PD，利润不可用" in text
    assert "利润口径提示：缺少 EAD/PD，利润不可用" in markdown
    assert "相对基线的决策迁移" in markdown
    assert "| review | approve | 3 |" in markdown
    assert "expected_profit_unavailable" in markdown
    assert "无红旗记录" not in markdown


def test_memory_skips_typed_approval_when_canonical_bad_rate_is_undefined() -> None:
    payload = _approval_envelope(approved_labels=False)

    assert (
        _approval_backtest_memory_metrics(
            StrategyBacktestResult.from_dict(payload),
            strategy_type="approval",
        )
        is None
    )


@pytest.mark.parametrize("strategy_type", ["limit", "pricing"])
def test_economic_presenters_show_na_when_no_baseline_exists(strategy_type: str) -> None:
    payload = _envelope(strategy_type)

    _text, tables = _render_backtest_strategy(payload)
    markdown, _sections = render_strategy_doc_markdown(
        strategy={
            "id": f"{strategy_type}-1",
            "strategy_type": strategy_type,
            "rules": [],
            "default_decision": "n/a",
        },
        meta={"version": 1, "status": "draft"},
        backtests=[payload],
        artifacts=[],
        band_stats=[],
    )

    summary_values = {row[0]: row[1] for row in tables[0]["rows"]}
    assert "n/a" in summary_values.values()
    assert "None" not in markdown


def test_presenters_surface_canonical_warnings_and_tool_risk_flags() -> None:
    payload = _approval_envelope()
    payload["warnings"] = ["2 population rows have no target label"]
    payload["red_flags"] = [
        {
            "code": "expected_profit_unavailable",
            "level": "amber",
            "message": "缺少 EAD/PD，利润不可用",
        }
    ]

    text, tables = _render_backtest_strategy(payload)
    markdown, _sections = render_strategy_doc_markdown(
        strategy={
            "id": "approval-1",
            "strategy_type": "approval",
            "rules": [],
            "default_decision": "approve",
        },
        meta={"version": 1, "status": "draft"},
        backtests=[payload],
        artifacts=[],
        band_stats=[],
        red_flags=payload["red_flags"],
    )

    assert "2 population rows have no target label" in text
    assert "2 population rows have no target label" in markdown
    assert "回测风险提示" in {table["title"] for table in tables}
    assert "expected_profit_unavailable" in markdown


def test_legacy_document_keeps_profit_unavailable_reason() -> None:
    markdown, _sections = render_strategy_doc_markdown(
        strategy={
            "id": "legacy-1",
            "strategy_type": "approval",
            "rules": [],
            "default_decision": "approve",
        },
        meta={"version": 1, "status": "draft"},
        backtests=[
            {
                "strategy_id": "legacy-1",
                "approval_rate": 0.5,
                "approved_bad_rate": 0.1,
                "rejected_bad_rate": 0.3,
                "expected_profit": None,
                "profit_note": "旧回测缺少 pd_col",
                "swap_in_count": 0,
                "swap_out_count": 0,
                "swap_in_bad_rate": None,
                "swap_out_bad_rate": None,
            }
        ],
        artifacts=[],
        band_stats=[],
    )

    assert "利润口径提示：旧回测缺少 pd_col" in markdown
    assert "expected_profit_unavailable" in markdown


def test_memory_metrics_accept_new_and_legacy_approval_but_skip_nonapproval() -> None:
    expected = {
        "approval_rate": 0.6,
        "approved_bad_rate": 0.2,
        "expected_profit": 123.0,
    }
    assert _approval_backtest_memory_metrics(
        StrategyBacktestResult.from_dict(_approval_envelope()),
        strategy_type="approval",
    ) == expected
    legacy = BacktestResult(
        strategy_id="legacy-approval",
        approval_rate=0.7,
        approved_count=7,
        approved_bad_rate=0.04,
        rejected_bad_rate=0.2,
        expected_profit=50.0,
        swap_in_count=0,
        swap_out_count=0,
        swap_in_bad_rate=None,
        swap_out_bad_rate=None,
        by_segment=(),
    )
    assert _approval_backtest_memory_metrics(
        legacy,
        strategy_type="approval",
    ) == {
        "approval_rate": 0.7,
        "approved_bad_rate": 0.04,
        "expected_profit": 50.0,
    }
    assert (
        _approval_backtest_memory_metrics(
            StrategyBacktestResult.from_dict(_envelope("limit")),
            strategy_type="limit",
        )
        is None
    )

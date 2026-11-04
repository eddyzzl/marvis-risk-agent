"""Regression: pooled migration matrix, undefined rows and their EL handling.

Reproduces the 6-loan x 4-month panel from the 2026-10-19 agent eval
(e2e_ui_20260729 ``inputs/portfolio/portfolio_panel.csv``). The old
``bucket_migration`` averaged each cell over every month pair, counting months
where the from-state had no loans as 0%, so the 61-90 row showed only 66.7% and
the charged_off row only 33.3%. Expected values below are hand-counted from the
raw transitions, never from the kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marvis.agent.renderers import render_tool_output
from marvis.packs.analysis.flow import bucket_migration, flow_rate
from marvis.packs.analysis.loss import expected_loss_estimate
from marvis.packs.analysis.report import gate_summary_payload
from marvis.packs.analysis.segment import concentration_thresholds, segment_profile

_STATES = ["current", "1-30", "31-60", "61-90", "charged_off"]
_LOSS = "charged_off"

# loan -> [(month, bucket, balance)], verbatim from portfolio_panel.csv
_PANEL = {
    "L001": [("2025-10", "current", 10000), ("2025-11", "1-30", 9200),
             ("2025-12", "31-60", 8400), ("2026-01", "charged_off", 7600)],
    "L002": [("2025-10", "current", 12000), ("2025-11", "current", 11200),
             ("2025-12", "1-30", 10400), ("2026-01", "current", 9600)],
    "L003": [("2025-10", "1-30", 8000), ("2025-11", "31-60", 7400),
             ("2025-12", "61-90", 6800), ("2026-01", "charged_off", 6200)],
    "L004": [("2025-10", "current", 15000), ("2025-11", "current", 14000),
             ("2025-12", "current", 13000), ("2026-01", "1-30", 12000)],
    "L005": [("2025-10", "31-60", 9000), ("2025-11", "61-90", 8300),
             ("2025-12", "charged_off", 7600), ("2026-01", "charged_off", 7000)],
    "L006": [("2025-10", "current", 11000), ("2025-11", "1-30", 10300),
             ("2025-12", "current", 9600), ("2026-01", "current", 8900)],
}
_SEGMENT = {"L001": "A", "L002": "A", "L003": "B", "L004": "B", "L005": "A", "L006": "B"}

# Hand-counted pooled count transitions over 2025-10..2025-12 (to_states + exited):
#   current: 8 pairs -> 4 current, 4 1-30
#   1-30:    4 pairs -> 2 current, 2 31-60
#   31-60:   3 pairs -> 2 61-90, 1 charged_off
#   61-90:   2 pairs (Nov, Dec only) -> 2 charged_off
#   charged_off: 1 pair (Dec only) -> 1 charged_off
_POOLED_COUNT = [
    [4 / 8, 4 / 8, 0, 0, 0, 0],
    [2 / 4, 0, 2 / 4, 0, 0, 0],
    [0, 0, 0, 2 / 3, 1 / 3, 0],
    [0, 0, 0, 0, 1.0, 0],
    [0, 0, 0, 0, 1.0, 0],
]


def _frame() -> pd.DataFrame:
    rows = [
        {"loan_id": loan, "snapshot_month": month, "bucket": bucket,
         "balance": float(balance), "segment": _SEGMENT[loan]}
        for loan, seq in _PANEL.items()
        for month, bucket, balance in seq
    ]
    return pd.DataFrame(rows)


def _kw(**extra):
    return {"id_col": "loan_id", "snapshot_col": "snapshot_month",
            "bucket_col": "bucket", "states": _STATES, **extra}


def test_pooled_count_matrix_rows_sum_to_one_and_match_hand_counts():
    result = bucket_migration(_frame(), **_kw())
    assert result.unobserved_states == ()
    for row, expected in zip(result.avg_matrix, _POOLED_COUNT, strict=True):
        assert sum(row) == pytest.approx(1.0, abs=1e-12)
        assert row == pytest.approx(expected, abs=1e-12)
    loss_j = result.to_states.index(_LOSS)
    # the two cells the eval reported as 66.7% / 33.3%
    assert result.avg_matrix[_STATES.index("61-90")][loss_j] == pytest.approx(1.0)
    assert result.avg_matrix[_STATES.index(_LOSS)][loss_j] == pytest.approx(1.0)
    # heat table carries the same pooled values
    heat = {row["from"]: row for row in result.heat_table}
    assert heat["61-90"][_LOSS] == pytest.approx(1.0)
    assert sum(heat["31-60"][state] for state in result.to_states) == pytest.approx(1.0)


def test_pooled_balance_matrix_uses_flow_rate_balance_weighting():
    result = bucket_migration(_frame(), **_kw(balance_col="balance"))
    for row in result.avg_matrix:
        assert sum(row) == pytest.approx(1.0, abs=1e-12)
    current = result.avg_matrix[0]
    # current-origin balances: stay = 12000+15000+14000+9600, roll = 10000+11000+11200+13000
    assert current[0] == pytest.approx(50600 / 95800)
    assert current[1] == pytest.approx(45200 / 95800)
    # 31-60: L005 9000 + L003 7400 -> 61-90, L001 8400 -> charged_off
    assert result.avg_matrix[2][3] == pytest.approx(16400 / 24800)
    assert result.avg_matrix[3][4] == pytest.approx(1.0)
    assert result.avg_matrix[4][4] == pytest.approx(1.0)


def test_worst_matrix_only_uses_months_with_mass():
    result = bucket_migration(_frame(), **_kw())
    # 1-30: Oct 100% -> 31-60, Nov 50/50, Dec 100% -> current
    assert result.worst_matrix[1][:3] == pytest.approx([1.0, 0.0, 1.0])
    # 61-90 only observed Nov/Dec (both 100% charged_off): no empty-month zeros
    assert result.worst_matrix[3] == pytest.approx([0, 0, 0, 0, 1.0, 0])


def test_unobserved_from_state_is_an_explicit_undefined_row():
    result = bucket_migration(_frame(), **_kw(window=["2025-10"]))
    # Oct origins are current, 1-30, 31-60 only
    assert result.unobserved_states == ("61-90", "charged_off")
    for state in ("61-90", "charged_off"):
        index = _STATES.index(state)
        assert result.avg_matrix[index] == [None] * len(result.to_states)
        assert result.worst_matrix[index] == [None] * len(result.to_states)
    for index in range(3):
        assert sum(result.avg_matrix[index]) == pytest.approx(1.0)
    # per-month sparse_month stays with flow_rate; not re-emitted by migration
    kinds = [flag["kind"] for flag in result.red_flags]
    assert kinds == ["unobserved_state"]
    assert result.red_flags[0]["states"] == ["61-90", "charged_off"]


def _hand_absorption(horizon: int) -> dict[str, float]:
    square = np.array([row[:5] for row in _POOLED_COUNT], dtype=float)
    powered = np.linalg.matrix_power(square, horizon)
    return {state: float(powered[i][4]) for i, state in enumerate(_STATES)}


def test_absorption_and_el_follow_pooled_matrix():
    lgd, horizon = 0.6, 12
    result = expected_loss_estimate(
        _frame(), **_kw(balance_col="balance", loss_state=_LOSS, lgd=lgd, horizon_months=horizon)
    )
    expected = _hand_absorption(horizon)
    chain = {row.from_state: row.p_to_loss for row in result.chain}
    assert chain == pytest.approx(expected, abs=1e-12)
    assert chain["61-90"] == pytest.approx(1.0) and chain[_LOSS] == pytest.approx(1.0)
    for month_row in result.el_by_month:
        month = month_row.month
        balances = [
            (bucket, balance)
            for seq in _PANEL.values()
            for m, bucket, balance in seq
            if m == month
        ]
        assert month_row.balance == pytest.approx(sum(b for _, b in balances))
        assert month_row.expected_loss == pytest.approx(
            sum(balance * expected[bucket] * lgd for bucket, balance in balances)
        )
    assert result.total_el == pytest.approx(result.el_by_month[-1].expected_loss)
    assert {flag["kind"] for flag in result.red_flags} == set()


def test_el_uses_pooled_rates_not_month_equal_weighting():
    # 2025-01: one current loan rolls to bad; 2025-02: nine current loans stay.
    # Pooled current->bad = 1/10; the old per-month mean gave (1.0 + 0.0) / 2.
    rows = [{"loan_id": "a", "snapshot_month": m, "bucket": b, "balance": 1.0}
            for m, b in (("2025-01", "current"), ("2025-02", "bad"), ("2025-03", "bad"))]
    rows += [{"loan_id": f"n{i}", "snapshot_month": m, "bucket": "current", "balance": 1.0}
             for i in range(9) for m in ("2025-02", "2025-03")]
    result = expected_loss_estimate(
        pd.DataFrame(rows), id_col="loan_id", snapshot_col="snapshot_month",
        bucket_col="bucket", states=["current", "bad"], balance_col="balance",
        loss_state="bad", lgd=1.0, horizon_months=1,
    )
    chain = {row.from_state: row.p_to_loss for row in result.chain}
    assert chain["current"] == pytest.approx(0.1)


def test_unobserved_non_loss_state_is_undefined_and_flagged_in_el():
    lgd = 0.5
    result = expected_loss_estimate(
        _frame(), **_kw(balance_col="balance", loss_state=_LOSS, lgd=lgd,
                        horizon_months=12, window=["2025-10"])
    )
    chain = {row.from_state: row.p_to_loss for row in result.chain}
    # 61-90 has no out-flow evidence: undefined, not a silent 0%
    assert chain["61-90"] is None
    # declared loss state with no evidence is absorbing by declaration
    assert chain[_LOSS] == pytest.approx(1.0)
    flag = next(flag for flag in result.red_flags if flag["kind"] == "unestimated_state")
    assert flag["states"] == ["61-90"]
    # 31-60 rolls into 61-90 in Oct, and current/1-30 reach 31-60 -> lower bounds
    assert flag["affected_states"] == ["current", "1-30", "31-60"]
    assert flag["reference_snapshot"] == "2026-01"
    assert flag["unestimated_balance"] == pytest.approx(0.0)
    assert "matrix_not_absorbing" not in {f["kind"] for f in result.red_flags}
    # Dec holds L003 in 61-90 (6800): its balance is excluded, not multiplied by 0
    dec = next(row for row in result.el_by_month if row.month == "2025-12")
    dec_rows = [(bucket, balance) for seq in _PANEL.values()
                for m, bucket, balance in seq if m == "2025-12" and bucket != "61-90"]
    assert dec.expected_loss == pytest.approx(
        sum(balance * chain[bucket] * lgd for bucket, balance in dec_rows)
    )


def test_gate_checklist_has_no_duplicated_alignment_flags():
    frame = _frame()
    flow = flow_rate(frame, **_kw(balance_col="balance"))
    migration = bucket_migration(frame, **_kw(balance_col="balance"))
    payload = gate_summary_payload(
        flow={"red_flags": flow.red_flags}, migration={"red_flags": migration.red_flags},
        segment=None, trend=None, expected_loss=None,
    )
    messages = payload["checklist"]
    assert len(messages) == len(set(messages)) == 3  # one sparse_month per month pair
    assert payload["red_flag_count"] == 3


@pytest.mark.parametrize(
    ("segments", "flagged"),
    [
        ({"A": 3, "B": 3}, False),           # two even segments: HHI 0.5 is the floor
        ({"A": 7, "B": 3}, False),           # 70% < 75% threshold for n=2
        ({"A": 8, "B": 2}, True),            # 80% > 75%
        ({f"s{i}": 1 for i in range(10)}, False),
        ({"big": 9, **{f"s{i}": 1 for i in range(11)}}, True),  # 45% > 40% for n=12
    ],
)
def test_concentration_threshold_accounts_for_segment_count(segments, flagged):
    frame = pd.DataFrame({"seg": [name for name, n in segments.items() for _ in range(n)]})
    result = segment_profile(frame, segment_col="seg")
    flags = [flag for flag in result.red_flags if flag["kind"] == "high_concentration"]
    assert bool(flags) is flagged
    if flags:
        n = len(segments)
        assert flags[0]["segment_count"] == n
        assert f"{n} 个细分" in flags[0]["message"]
        assert "均匀分布" in flags[0]["message"]


def test_concentration_thresholds_scale_with_even_share():
    assert concentration_thresholds(2) == pytest.approx((0.75, 0.75))
    assert concentration_thresholds(3) == pytest.approx((0.5, 0.5))
    assert concentration_thresholds(6) == pytest.approx((0.4, 0.25))
    assert concentration_thresholds(20) == pytest.approx((0.4, 0.25))
    # one segment never trips (concentration across one category is not informative)
    assert concentration_thresholds(1)[0] > 1.0


def test_panel_segments_of_equal_size_are_not_flagged():
    result = segment_profile(_frame(), segment_col="segment", ead_col="balance")
    assert result.concentration.top1_pct == pytest.approx(0.5)
    assert "high_concentration" not in {flag["kind"] for flag in result.red_flags}


def test_renderers_label_flags_in_chinese_and_show_undefined_rows():
    frame = _frame()
    migration = bucket_migration(frame, **_kw(window=["2025-10"]))
    output = {
        "states": list(migration.states), "to_states": list(migration.to_states),
        "window_months": list(migration.window_months), "heat_table": migration.heat_table,
        "red_flags": migration.red_flags,
    }
    text, tables = render_tool_output("bucket_migration", output)
    heat = next(table for table in tables if table["title"] == "平均迁徙率矩阵")
    row = next(r for r in heat["rows"] if r[0] == "61-90")
    assert row[1:] == ["无观测"] * len(migration.to_states)
    assert "61-90" in text and "无观测" in text
    flag_table = next(table for table in tables if table["title"].startswith("数据质量红旗"))
    assert flag_table["title"] == "数据质量红旗 · 迁徙热力"
    assert flag_table["rows"][0][0] == "迁徙行无观测"

    flow = flow_rate(frame, **_kw())
    duplicated = [*flow.red_flags, *flow.red_flags]
    _, flow_tables = render_tool_output(
        "flow_rate", {"months": list(flow.months), "net_flows": [], "red_flags": duplicated}
    )
    flow_flags = flow_tables[-1]
    assert flow_flags["title"] == "数据质量红旗 · 流量分析"
    assert len(flow_flags["rows"]) == len(flow.red_flags)  # identical flags shown once
    rendered = repr(flow_flags["rows"])
    assert "sparse_month" not in rendered and "月度样本稀疏" in rendered

    el = expected_loss_estimate(
        frame, **_kw(balance_col="balance", loss_state=_LOSS, lgd=0.5,
                     horizon_months=12, window=["2025-10"])
    )
    el_output = {
        "loss_state": el.loss_state, "total_el": el.total_el,
        "chain": [{"from_state": r.from_state, "p_to_loss": r.p_to_loss} for r in el.chain],
        "el_by_month": [], "assumptions": el.assumptions, "red_flags": el.red_flags,
    }
    _, el_tables = render_tool_output("expected_loss_estimate", el_output)
    chain_table = next(table for table in el_tables if table["title"] == "各状态到损失态的吸收概率")
    assert ["61-90", "无法估计"] in chain_table["rows"]
    el_flags = el_tables[-1]
    assert el_flags["rows"][0][0] == "损失概率无法估计"
    assert "unestimated_state" not in repr(el_flags)


@pytest.mark.parametrize("window", [None, ["2025-10"], ["2025-11", "2025-12"]])
def test_independent_reference_agrees_with_kernels(window):
    """The archive revalidation reference must encode the same pooled semantics."""
    from marvis.orchestrator.eval.runtime_portfolio_reference import (
        reference_flag_text, segment_reference, transition_reference,
    )

    frame = _frame()
    records = frame.to_dict("records")
    reference = transition_reference(
        records, id_col="loan_id", snapshot_col="snapshot_month", bucket_col="bucket",
        balance_col="balance", states=_STATES, loss_state=_LOSS, lgd=0.45,
        horizon_months=18, window=window,
    )
    migration = bucket_migration(frame, **_kw(balance_col="balance", window=window))
    el = expected_loss_estimate(
        frame, **_kw(balance_col="balance", loss_state=_LOSS, lgd=0.45,
                     horizon_months=18, window=window)
    )

    def same(actual, expected):
        if isinstance(expected, list):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected):
                same(left, right)
        elif expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)

    same(migration.avg_matrix, reference["migration"]["avg_matrix"])
    same(migration.worst_matrix, reference["migration"]["worst_matrix"])
    same([row.p_to_loss for row in el.chain],
         [row["p_to_loss"] for row in reference["expected_loss"]["chain"]])
    same([row.expected_loss for row in el.el_by_month],
         [row["expected_loss"] for row in reference["expected_loss"]["el_by_month"]])
    for produced, expected in ((migration.red_flags, reference["migration"]["red_flags"]),
                               (el.red_flags, reference["expected_loss"]["red_flags"])):
        assert [flag["message"] for flag in produced] == [reference_flag_text(f) for f in expected]
        assert [{k: v for k, v in flag.items() if k != "message"} for flag in produced] == expected

    segments = segment_reference(records, segment_col="segment", balance_col="balance")
    produced = segment_profile(frame, segment_col="segment", ead_col="balance")
    assert produced.red_flags == segments["red_flags"] == []
    skewed = [{**row, "segment": "A" if row["loan_id"] != "L006" else "B"} for row in records]
    skewed_ref = segment_reference(skewed, segment_col="segment", balance_col="balance")
    skewed_out = segment_profile(pd.DataFrame(skewed), segment_col="segment", ead_col="balance")
    assert [f["message"] for f in skewed_out.red_flags] == [
        reference_flag_text(f) for f in skewed_ref["red_flags"]
    ]

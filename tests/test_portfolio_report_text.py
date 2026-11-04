"""Portfolio report labels are literal text on the actual XLSX carrier."""
import pytest
from openpyxl import load_workbook

from marvis.output.portfolio_report import PortfolioReportPayload, render_portfolio_report


@pytest.mark.parametrize("text", ["=1+1", "#N/A"])
def test_uploaded_labels_are_literal_in_values_and_dynamic_headers(tmp_path, text):
    path = render_portfolio_report(PortfolioReportPayload(
        project_meta={text: text},
        migration={"heat_table": [{"from": text, text: 0.125}]},
        segment={"segments": [{"segment": text, "count": 2, "net_profit": -0.25}]},
        red_flags=[{"message": text}],
    ), tmp_path / "report.xlsx")
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        for sheet, address in (("组合概览", "A2"), ("组合概览", "B2"), ("桶迁徙", "B1"),
                               ("桶迁徙", "A2"), ("细分画像", "A2"), ("数据质量红旗", "A2")):
            cell = book[sheet][address]
            assert cell.value == text and cell.data_type == "s"
        assert book["桶迁徙"]["B2"].value == 0.125 and book["桶迁徙"]["B2"].data_type == "n"
        assert book["细分画像"]["C2"].value == -0.25 and book["细分画像"]["C2"].data_type == "n"
    finally:
        book.close()


def test_red_flag_and_migration_sheets_read_in_chinese(tmp_path):
    path = render_portfolio_report(PortfolioReportPayload(
        migration={"heat_table": [{"from": "current", "current": 0.9, "1-30": 0.1}]},
        red_flags=[
        {"source": "flow_rate", "kind": "sparse_month", "message": "月 2026-01 对齐月对仅 3 对。"},
        {"source": "segment_profile", "kind": "high_concentration", "message": "细分高度集中。"},
    ]), tmp_path / "report.xlsx")
    book = load_workbook(path, read_only=True)
    try:
        rows = [list(row) for row in book["数据质量红旗"].iter_rows(values_only=True)]
        migration_header = [cell.value for cell in next(book["桶迁徙"].iter_rows(max_row=1))]
    finally:
        book.close()
    assert rows == [
        ["说明", "类型", "来源步骤"],
        ["月 2026-01 对齐月对仅 3 对。", "月度样本稀疏", "流量分析"],
        ["细分高度集中。", "细分集中度高", "细分画像"],
    ]
    assert migration_header == ["起始状态", "current", "1-30"]


def test_portfolio_summary_states_migration_and_concentration_findings():
    # Eval S9: the user asked about migration and concentration and got no
    # sentence about either. The summary reads them off the steps' own output.
    from marvis.agent.renderers import _render_portfolio_gate_summary
    from marvis.packs.analysis.report import gate_summary_payload

    payload = gate_summary_payload(
        flow=None,
        migration={
            "states": ["current", "1-30", "31-60", "charged_off"],
            "heat_table": [
                {"from": "current", "current": 0.9, "1-30": 0.1},
                {"from": "1-30", "current": 0.4, "31-60": 0.6},
                {"from": "31-60", "charged_off": 0.3, "exited": 0.7},
            ],
            "window_months": ["2025-10"],
        },
        segment={
            "segments": [{"segment": "A", "pop_pct": 0.5}, {"segment": "B", "pop_pct": 0.5}],
            "concentration": {"hhi": 0.5, "top1_pct": 0.5},
            "red_flags": [],
        },
        trend=None,
        expected_loss=None,
    )
    assert payload["highlights"]["worst_migration"] == {"from": "1-30", "to": "31-60", "rate": 0.6}
    assert payload["highlights"]["top1_segment"] == "A、B"
    text, _tables = _render_portfolio_gate_summary(payload)
    assert "最大的恶化流向是 1-30 → 31-60（窗口平均 60.0%）" in text
    assert "最大细分 A、B 占 50.0%，没有触发集中度红旗" in text


def test_bucket_order_gate_offers_a_deterministic_confirm_chip():
    # Eval S9: the inferred order could only be confirmed by typing.
    from marvis.agent.portfolio_setup import parse_states_reply
    from marvis.agent.turn_handlers.portfolio import _states_order_replies

    states = ["current", "1-30", "31-60", "charged_off"]
    [reply] = _states_order_replies(states)
    assert reply["label"] == "顺序没问题"
    assert parse_states_reply(reply["command"], {"proposed_states": states}) == states
    assert _states_order_replies([]) == ()


def test_step_results_state_the_migration_and_concentration_findings():
    # The summary gate opens before its own step runs, so the findings also
    # sit in the step results the gate shows.
    from marvis.agent.renderers import _render_bucket_migration, _render_segment_profile

    migration, _ = _render_bucket_migration({
        "states": ["current", "1-30", "31-60"],
        "to_states": ["current", "1-30", "31-60", "exited"],
        "heat_table": [{"from": "current", "current": 0.9, "1-30": 0.1}, {"from": "1-30", "current": 0.4, "31-60": 0.6}],
        "window_months": ["2025-10"],
    })
    assert "最大的恶化流向是 1-30 → 31-60（窗口平均 60.0%）" in migration
    segment, _ = _render_segment_profile({
        "segments": [{"segment": "A", "pop_pct": 0.7}, {"segment": "B", "pop_pct": 0.3}],
        "concentration": {"top1_pct": 0.7, "hhi": 0.58},
        "red_flags": [{"kind": "high_concentration", "message": "x"}],
    })
    assert "占比最大的细分是 A（70.0%），已触发集中度红旗" in segment

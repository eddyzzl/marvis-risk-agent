"""VD-4: calibration reliability curve + score-band chart payload shaping.

generate_model_report's calibration/score_bands rows (report_tools.py::
_artifact_calibration_rows / _score_band_rows) already reach the tool output
but never reached the agent-conversation table payload -- these tests pin the
{title, columns, rows, chart} reshape in renderers.py::_render_report so the
frontend chart renderers (metric-tables.js::renderCalibrationCard /
renderScoreBandCard) get coordinate-ready data. No new numbers are computed
here (INV-1) -- only reshaping of the same fields already in the tool output.
"""

from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _calibration_rows():
    return [
        {
            "score_type": "summary", "method": "isotonic", "split": "test",
            "fit_split": "train", "eval_split": "test", "evaluated_on": "test",
            "sample_count": 200, "positive_count": 20,
            "brier_raw": 0.1234, "brier_calibrated": 0.0456,
            "ece_raw": 0.0789, "ece_calibrated": 0.0123,
            "pmml_includes_calibration": False,
        },
        {
            "score_type": "raw", "bin": 1, "prob_lower": 0.0, "prob_upper": 0.1,
            "sample_count": 120, "positive_count": 4,
            "avg_predicted_pd": 0.05, "observed_bad_rate": 0.033,
            "calibration_gap": 0.017, "abs_gap": 0.017,
        },
        {
            "score_type": "raw", "bin": 2, "prob_lower": 0.1, "prob_upper": 0.2,
            "sample_count": 80, "positive_count": 16,
            "avg_predicted_pd": 0.15, "observed_bad_rate": 0.20,
            "calibration_gap": -0.05, "abs_gap": 0.05,
        },
        {
            "score_type": "calibrated", "bin": 1, "prob_lower": 0.0, "prob_upper": 0.1,
            "sample_count": 120, "positive_count": 4,
            "avg_predicted_pd": 0.04, "observed_bad_rate": 0.033,
            "calibration_gap": 0.007, "abs_gap": 0.007,
        },
    ]


def _score_band_rows():
    return [
        {
            "split": "oot", "bin": 1, "score_lower": 0.0, "score_upper": 0.1,
            "sample_count": 500, "labeled_count": 500, "bad_count": 10, "bad_rate": 0.02,
            "avg_score": 0.05, "cum_count_pct": 0.5, "cum_bad_rate": 0.02,
            "cum_bad_capture": 0.4, "cum_reject_rate": 0.5,
            "cum_pass_rate": 0.5, "lift": 0.5,
            "bin_edges_source": "train", "cum_direction": "higher_is_riskier",
        },
        {
            "split": "oot", "bin": 2, "score_lower": 0.1, "score_upper": 0.2,
            "sample_count": 480, "labeled_count": 480, "bad_count": 24, "bad_rate": 0.05,
            "avg_score": 0.15, "cum_count_pct": 1.0, "cum_bad_rate": 0.035,
            "cum_bad_capture": 1.0, "cum_reject_rate": 1.0,
            "cum_pass_rate": 0.0, "lift": 1.2,
            "bin_edges_source": "train", "cum_direction": "higher_is_riskier",
        },
        {
            "split": "train", "bin": 1, "score_lower": 0.0, "score_upper": 0.1,
            "sample_count": 900, "labeled_count": 900, "bad_count": 18, "bad_rate": 0.02,
            "avg_score": 0.05, "cum_count_pct": 0.5, "cum_bad_rate": 0.02,
            "cum_bad_capture": 0.4, "cum_reject_rate": 0.5,
            "cum_pass_rate": 0.5, "lift": 0.5,
            "bin_edges_source": "train", "cum_direction": "higher_is_riskier",
        },
    ]


def test_generate_model_report_renderer_adds_calibration_and_score_band_tables():
    text, tables = render_tool_output("generate_model_report", {
        "report_path": "/tmp/model_report.xlsx",
        "section_status": [{"section": "汇总", "available": True}],
        "calibration": _calibration_rows(),
        "score_bands": _score_band_rows(),
    })

    assert "模型开发报告已生成" in text
    titles = [table["title"] for table in tables]
    assert "报告章节状态" in titles
    assert "概率校准（可靠性曲线）" in titles
    assert "评分分段（oot）" in titles


def test_calibration_table_chart_points_come_straight_from_raw_reliability_rows():
    _text, tables = render_tool_output("generate_model_report", {
        "calibration": _calibration_rows(),
    })
    calibration_table = next(t for t in tables if t["title"] == "概率校准（可靠性曲线）")
    chart = calibration_table["chart"]

    assert chart["kind"] == "calibration_curve"
    # Only "raw" score_type rows become chart points (not "calibrated" or "summary").
    assert chart["points"] == [
        {"avg_predicted_pd": 0.05, "observed_bad_rate": 0.033, "sample_count": 120, "bin": 1},
        {"avg_predicted_pd": 0.15, "observed_bad_rate": 0.20, "sample_count": 80, "bin": 2},
    ]
    # Brier/ECE summary numbers are the exact ones from the summary row (INV-1:
    # no recomputation, straight passthrough).
    assert chart["brier_raw"] == 0.1234
    assert chart["brier_calibrated"] == 0.0456
    assert chart["ece_raw"] == 0.0789
    assert chart["ece_calibrated"] == 0.0123

    # The flat table underneath keeps both raw and calibrated rows (chart is
    # an addition, not a replacement of the existing tabular view).
    assert len(calibration_table["rows"]) == 3


def test_score_band_table_prefers_oot_split_and_orders_bands_by_bin():
    _text, tables = render_tool_output("generate_model_report", {
        "score_bands": _score_band_rows(),
    })
    score_band_table = next(t for t in tables if t["title"].startswith("评分分段"))
    chart = score_band_table["chart"]

    assert chart["kind"] == "score_band_bars"
    assert chart["split"] == "oot"
    assert [band["bin"] for band in chart["bands"]] == [1, 2]
    assert chart["bands"][0]["sample_count"] == 500
    assert chart["bands"][0]["bad_rate"] == 0.02
    assert chart["bands"][1]["sample_count"] == 480
    assert chart["bands"][1]["bad_rate"] == 0.05
    assert score_band_table["columns"][4:6] == ["累计拒绝率", "拒绝人群坏率"]
    assert [row[4] for row in score_band_table["rows"]] == ["0.5000", "1.0000"]
    # train-split rows must not leak into the oot chart.
    assert all(band["sample_count"] != 900 for band in chart["bands"])


def test_score_band_table_surfaces_unscored_population():
    rows = _score_band_rows()
    for row in rows:
        if row["split"] == "oot":
            row["score_coverage"] = 0.8
            row["unscored_count"] = 20
    _text, tables = render_tool_output("generate_model_report", {"score_bands": rows})

    score_band_table = next(t for t in tables if t["title"].startswith("评分分段"))
    assert score_band_table["columns"][-2:] == ["评分覆盖率", "未评分数"]
    assert score_band_table["rows"][0][-2:] == ["0.8000", "20"]


def test_score_band_table_falls_back_to_first_available_split_when_no_oot():
    train_only = [row for row in _score_band_rows() if row["split"] == "train"]
    _text, tables = render_tool_output("generate_model_report", {
        "score_bands": train_only,
    })
    score_band_table = next(t for t in tables if t["title"].startswith("评分分段"))
    assert score_band_table["chart"]["split"] == "train"
    assert score_band_table["title"] == "评分分段（train）"


def test_report_renderer_degrades_gracefully_with_no_calibration_or_score_band_data():
    """Empty/missing calibration and score_bands must not add empty tables or
    crash -- generate_model_report tool output for a non-binary target returns
    calibration/score_bands as [] (report_tools.py's minimal-report branch)."""
    text, tables = render_tool_output("generate_model_report", {
        "report_path": "/tmp/model_report.xlsx",
        "section_status": [{"section": "汇总", "status": "ok"}],
        "scorecard_table": [],
        "score_bands": [],
    })
    assert "模型开发报告已生成" in text
    titles = [table["title"] for table in tables]
    assert "概率校准（可靠性曲线）" not in titles
    assert not any(title.startswith("评分分段") for title in titles)


def test_report_renderer_ignores_missing_calibration_and_score_band_keys():
    """Older tool outputs (or tools that never populate these keys) must not
    error -- render_tool_output already falls back to _render_generic on any
    exception, but the explicit .get(...) here means the happy path degrades
    cleanly instead of relying on that fallback."""
    text, tables = render_tool_output("generate_model_report", {
        "report_path": "/tmp/model_report.xlsx",
        "section_status": [],
    })
    assert "模型开发报告已生成" in text
    assert tables == []

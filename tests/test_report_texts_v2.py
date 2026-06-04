from dataclasses import replace

from riskmodel_checker.report_texts import report_text_values_from_results
from riskmodel_checker.validation.results import SplitRow
from tests.output.test_excel import _make_results


def test_text_values_cover_core_placeholders():
    values = report_text_values_from_results(_make_results())
    expected_keys = {
        "TEXT:report_title",
        "TEXT:model_name",
        "TEXT:sample_period",
        "TEXT:sample_start_month",
        "TEXT:sample_end_month",
        "TEXT:train_count",
        "TEXT:test_count",
        "TEXT:oot_count",
        "TEXT:oot_ks",
        "TEXT:oot_psi",
        "TEXT:reproducibility_summary",
        "TEXT:stress_test_summary",
        "TEXT:pressure_test_summary",
    }
    assert expected_keys.issubset(values.keys())


def test_oot_metrics_formatted_to_four_decimals():
    values = report_text_values_from_results(_make_results())
    assert values["TEXT:oot_ks"] == "0.2500"
    assert values["TEXT:oot_psi"] == "0.1200"


def test_reproducibility_summary_reflects_status():
    values = report_text_values_from_results(_make_results())
    assert "通过" in values["TEXT:reproducibility_summary"]


def test_stress_summary_mentions_negative_9999_sentinel():
    values = report_text_values_from_results(_make_results())

    assert "置 -9999 1 个特征" in values["TEXT:stress_test_summary"]
    assert values["TEXT:pressure_test_summary"] == values["TEXT:stress_test_summary"]
    assert "置 null" not in values["TEXT:stress_test_summary"]


def test_pressure_test_summary_placeholder_gets_platform_fallback():
    values = report_text_values_from_results(
        _make_results(),
        manual_values={"TEXT:pressure_test_summary": "不应替换原始占位符"},
    )

    assert values["TEXT:pressure_test_summary"] == values["TEXT:stress_test_summary"]


def test_manual_report_values_override_and_extend_generated_text():
    values = report_text_values_from_results(
        _make_results(),
        report_values={"TEXT:report_title": "人工标题"},
        manual_values={
            "model_overview": "人工模型概述",
            "TEXT:revision_version": "V0.1",
        },
    )

    assert values["TEXT:report_title"] == "人工标题"
    assert values["TEXT:model_overview"] == "人工模型概述"
    assert values["TEXT:revision_version"] == "V0.1"
    assert values["TEXT:oot_ks"] == "0.2500"


def test_empty_model_version_does_not_add_version_suffix_to_report_title():
    results = replace(
        _make_results(),
        model_name="贷前评分卡 MOB3 v202604",
        model_version="",
    )

    values = report_text_values_from_results(results)

    assert values["TEXT:report_title"] == "贷前评分卡 MOB3 v202604模型验证文档"
    assert "版验证文档" not in values["TEXT:report_title"]


def test_platform_generated_sample_period_text_values_use_split_stats():
    base = _make_results()
    results = replace(
        base,
        basic_info=replace(
            base.basic_info,
            sample_period=("20250301", "20250831"),
            split_summary=[
                SplitRow("train", 100, 10, 0.1, period_start="20250301", period_end="20250531"),
                SplitRow("test", 50, 5, 0.1, period_start="20250601", period_end="20250630"),
                SplitRow("oot", 25, 2, 0.08, period_start="20250701", period_end="20250831"),
            ],
        ),
    )

    values = report_text_values_from_results(results)

    assert values["TEXT:sample_start_month"] == "20250301"
    assert values["TEXT:sample_end_month"] == "20250831"
    assert values["TEXT:train_test_period"] == "20250301-20250630"
    assert values["TEXT:train_test_ratio"] == "66.67%:33.33%"
    assert values["TEXT:oot_period"] == "20250701-20250831"


def test_model_training_description_uses_algorithm_default_text():
    results = replace(_make_results(), algorithm="xgb")

    values = report_text_values_from_results(results)

    assert "XGBoost" in values["TEXT:model_training_description"]
    assert "信贷风控" in values["TEXT:model_training_description"]


def test_pressure_recommendation_alias_tracks_manual_summary():
    values = report_text_values_from_results(
        _make_results(),
        manual_values={"TEXT:pressure_recommendation_summary": "重点关注征信类变量冲击。"},
    )

    assert values["TEXT:pressure_impact_recommendation"] == "重点关注征信类变量冲击。"

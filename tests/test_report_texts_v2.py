from dataclasses import replace

from marvis.report_texts import merge_report_text_values, report_text_values_from_results
from marvis.validation.results import SplitRow
from tests.output.test_excel import _make_pmml_results, _make_results


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


def test_legacy_reproducibility_summary_wording_is_unchanged():
    values = report_text_values_from_results(_make_results())

    assert values["TEXT:reproducibility_summary"] == (
        "对 2 行抽样进行三方分数对比，对齐 2 行，差异 0 行，"
        "最大绝对差 0.000000，可复现性验证 通过。"
    )
    assert "TEXT:pmml_scoring_summary" not in values


def test_pmml_results_use_full_sample_scoring_summary_without_reproducibility():
    values = report_text_values_from_results(_make_pmml_results())

    summary = values["TEXT:pmml_scoring_summary"]
    assert values["TEXT:reproducibility_summary"] == summary
    assert "PMML打分测试" in summary
    assert "全量 3 行" in summary
    assert "成功 3 行，失败 0 行" in summary
    assert "probability_1" in summary
    assert "测试通过" in summary
    assert "三方分数对比" not in summary


def test_stress_summary_mentions_negative_9999_sentinel():
    values = report_text_values_from_results(_make_results())

    assert "置 -9999 1 个特征" in values["TEXT:stress_test_summary"]
    assert values["TEXT:pressure_test_summary"] == values["TEXT:stress_test_summary"]
    assert "置 null" not in values["TEXT:stress_test_summary"]


def test_stress_summary_names_unclassified_features():
    base = _make_results()
    results = replace(
        base,
        stress_test=replace(
            base.stress_test,
            status="partial",
            unclassified_features=["BH_A044_C0580"],
        ),
    )

    values = report_text_values_from_results(results)

    assert "未分类特征 1 个：BH_A044_C0580" in values["TEXT:stress_test_summary"]


def test_pressure_test_summary_placeholder_gets_platform_fallback():
    values = report_text_values_from_results(
        _make_results(),
        manual_values={"TEXT:pressure_test_summary": "不应替换原始占位符"},
    )

    assert values["TEXT:pressure_test_summary"] == values["TEXT:stress_test_summary"]


def test_agent_report_values_cannot_replace_platform_pressure_test_summary():
    values = report_text_values_from_results(
        _make_results(),
        report_values={
            "TEXT:pressure_test_summary": "Agent 不应覆盖平台摘要",
            "TEXT:pressure_impact_recommendation": "Agent 可补充影响建议",
        },
    )

    assert values["TEXT:pressure_test_summary"] == values["TEXT:stress_test_summary"]
    assert values["TEXT:pressure_impact_recommendation"] == "Agent 可补充影响建议"


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


def test_platform_generated_period_text_collapses_single_month_range():
    base = _make_results()
    results = replace(
        base,
        basic_info=replace(
            base.basic_info,
            split_summary=[
                SplitRow("train", 100, 10, 0.1, period_start="202503", period_end="202503"),
                SplitRow("test", 50, 5, 0.1, period_start="202503", period_end="202503"),
                SplitRow("oot", 25, 2, 0.08, period_start="202504", period_end="202504"),
            ],
        ),
    )

    values = report_text_values_from_results(results)

    assert values["TEXT:train_test_period"] == "202503"
    assert values["TEXT:oot_period"] == "202504"


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


def test_merge_report_text_values_uses_named_permission_sets():
    values = merge_report_text_values(
        {"TEXT:oot_ks": "0.2500", "TEXT:report_title": "平台标题"},
        report_values={
            "TEXT:final_validation_conclusion": "Agent 已确认结论",
            "TEXT:oot_ks": "不允许",
        },
        manual_values={
            "TEXT:report_title": "人工标题",
            "TEXT:final_validation_conclusion": "不允许覆盖 Agent 确认结论",
            "TEXT:oot_ks": "不允许",
        },
    )

    assert values["TEXT:oot_ks"] == "0.2500"
    assert values["TEXT:report_title"] == "人工标题"
    assert values["TEXT:final_validation_conclusion"] == "Agent 已确认结论"

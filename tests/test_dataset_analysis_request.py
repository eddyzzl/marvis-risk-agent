from marvis.agent.dataset_analysis import (
    build_dataset_analysis_request,
    detect_dataset_analysis_intent,
)


def test_detect_dataset_analysis_intent_is_specific_and_does_not_hijack_strategy_work():
    assert detect_dataset_analysis_intent("分析这份样本的数据概况")
    assert detect_dataset_analysis_intent("看看 target 分布和缺失情况")
    assert detect_dataset_analysis_intent("生成完整相关矩阵")
    assert not detect_dataset_analysis_intent("回测当前策略并比较通过率")
    assert not detect_dataset_analysis_intent("确认")


def test_build_general_request_uses_all_report_ready_sections_and_bound_target():
    result = build_dataset_analysis_request(
        "分析当前样本",
        columns=("score", "income", "bad"),
        target_col="bad",
        business_names={"score": "模型分"},
    )

    assert result.clarification is None
    assert result.request is not None
    assert result.request.sections == (
        "overview",
        "target",
        "missing",
        "distribution",
        "correlation",
    )
    assert result.request.columns is None
    assert result.request.target_col == "bad"


def test_build_specific_request_resolves_raw_and_business_column_names():
    result = build_dataset_analysis_request(
        "看模型分和 income 的相关性与空值",
        columns=("score", "income", "bad"),
        target_col="bad",
        business_names={"score": "模型分"},
    )

    assert result.clarification is None
    assert result.request is not None
    assert result.request.sections == ("missing", "correlation")
    assert result.request.columns == ("score", "income")


def test_target_distribution_requires_confirmed_target_mapping():
    result = build_dataset_analysis_request(
        "查看目标分布",
        columns=("score", "bad"),
        target_col=None,
        business_names={},
    )

    assert result.request is None
    assert result.clarification is not None
    assert "target" in result.clarification.lower()


def test_unknown_named_column_clarifies_instead_of_silently_analyzing_everything():
    result = build_dataset_analysis_request(
        "分析 region 字段的缺失情况",
        columns=("score", "bad"),
        target_col="bad",
        business_names={},
    )

    assert result.request is None
    assert result.clarification is not None
    assert "region" in result.clarification


def test_mixed_known_and_unknown_columns_fail_closed_without_dropping_the_unknown():
    result = build_dataset_analysis_request(
        "看 score 和 region 的相关性",
        columns=("score", "bad"),
        target_col="bad",
        business_names={},
    )

    assert result.request is None
    assert result.clarification is not None
    assert "region" in result.clarification


def test_common_english_analysis_words_are_not_mistaken_for_columns():
    result = build_dataset_analysis_request(
        "show correlation between score and income columns",
        columns=("score", "income", "bad"),
        target_col="bad",
        business_names={},
    )

    assert result.clarification is None
    assert result.request is not None
    assert result.request.columns == ("score", "income")

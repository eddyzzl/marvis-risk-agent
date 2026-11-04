from __future__ import annotations

import pytest

from marvis.agent.strategy_control_labels import (
    control_label,
    control_labels,
    join_control_labels,
)


@pytest.mark.parametrize(
    ("item", "label"),
    [
        ("operation", "策略操作"),
        ("workflow_inputs", "流程参数"),
        ("strategy_type", "策略类型"),
        ("strategy_type unknown", "策略类型"),
        ("strategy_type approval", "策略类型（审批策略）"),
        ("partition oot", "回放分区（OOT）"),
        ("partitions", "分区范围"),
        ("comparison_mode vs_baseline", "对比方式（与基线策略对比）"),
        ("month_col apply_month", "月份列「apply_month」"),
        ("group_col a/b", "分组列「a/b」"),
        ("output_prefix", "输出前缀"),
        ("output_prefix safe_", "输出前缀「safe_」"),
        ("drop_nan_labels=true", "空标签处理（删除空标签样本）"),
        ("economics_inputs", "经济参数"),
        ("economics_inputs.lgd", "经济参数 LGD"),
        ("pd_col/pd_value", "PD"),
        ("utilization_col/utilization_value", "额度使用率"),
        ("lgd", "LGD"),
        ("ead_col", "EAD 列"),
        ("default_action", "默认动作"),
        ("action", "命中动作"),
        ("typed action review", "动作类型（转人工复核）"),
        ("typed action unknown", "动作类型"),
        ("typed action value 3000", "动作取值「3000」"),
        ("typed action output_value TIER_A", "动作输出值「TIER_A」"),
        ("features includes income", "特征「income」"),
        ("sample_weight_col=weight", "样本权重列「weight」"),
        ("age=decreasing", "特征「age」的方向（单调递减）"),
        ("directions.age=increasing", "特征「age」的方向（单调递增）"),
        ("max_depth=4", "最大树深「4」"),
        ("strategy-base-a", "「strategy-base-a」"),
        ("模型建议转人工", "「模型建议转人工」"),
        ("用户原话中的完整顺序", "用户原话中的完整顺序"),
    ],
)
def test_control_label_names_raw_controls_in_business_terms(
    item: str,
    label: str,
) -> None:
    assert control_label(item) == label


def test_control_label_prefixes_bare_values_with_the_callers_control() -> None:
    assert control_label("income", bare_value="特征") == "特征「income」"
    # Known control keys keep their own label even when a bare-value hint exists.
    assert (
        control_label("current_strategy_id", bare_value="当前策略 ID")
        == "当前策略 ID"
    )


def test_feature_columns_named_like_controls_stay_features() -> None:
    # Automatic-tree features are user columns; ``pd`` must not read as the
    # PD economics control, with or without a direction.
    assert control_label("pd", bare_value="特征", bare_keys=False) == "特征「pd」"
    assert control_label("pd=increasing") == "特征「pd」的方向（单调递增）"
    assert control_label("seed=unordered") == "特征「seed」的方向（不约束方向）"
    assert control_label("seed=42") == "随机种子「42」"


def test_user_values_starting_with_a_control_name_are_not_read_as_controls() -> None:
    # Pool reasons/reason codes and tree features are user data, even when
    # they begin with a word from the control vocabulary.
    assert control_label("pd 过高", bare_keys=False) == "「pd 过高」"
    assert control_label("lgd", bare_keys=False) == "「lgd」"
    assert (
        control_label("action count", bare_value="特征", bare_keys=False)
        == "特征「action count」"
    )
    assert control_label("strategy_type approval", bare_keys=False) == (
        "策略类型（审批策略）"
    )


def test_unknown_enum_values_stay_verbatim_rather_than_being_dropped() -> None:
    assert control_label("strategy_type collection") == "策略类型（collection）"
    assert control_label("typed action escalate") == "动作类型（escalate）"


def test_control_labels_deduplicate_after_translation_in_order() -> None:
    items = ("ead_col/ead_value", "economics_inputs.ead", "ead_col/ead_value")

    assert control_labels(items) == ["EAD", "经济参数 EAD"]
    assert join_control_labels(("strategy_type", "default_action")) == "策略类型、默认动作"

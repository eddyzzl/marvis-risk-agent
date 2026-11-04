"""Business wording for strategy controls named in user-facing prose.

Product rule (DESIGN.md, workbench redesign plan D11): the sentence a user
reads names controls in business terms. Raw control paths, enum values and
hashes stay machine-readable in clarification ``code``/``fields``, metadata
and evidence tables; only prose goes through these labels, and a label never
replaces the raw value anywhere else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

STRATEGY_TYPE_LABELS = {
    "approval": "审批策略",
    "reject": "拒绝策略",
    "limit": "额度策略",
    "pricing": "定价策略",
    "segmentation": "分群策略",
}

POOL_ACTION_TYPE_LABELS = {
    "approval": "通过",
    "reject": "拒绝",
    "review": "转人工复核",
    "limit": "额度",
    "pricing": "定价",
    "segment": "分群",
}

PARTITION_LABELS = {
    "development": "开发集",
    "validation": "验证集",
    "oot": "OOT",
}

COMPARISON_MODE_LABELS = {
    "absolute": "单独测算",
    "vs_baseline": "与基线策略对比",
}

TREE_DIRECTION_LABELS = {
    "increasing": "单调递增",
    "decreasing": "单调递减",
    "unordered": "不约束方向",
}

DROP_NAN_LABELS = {
    "true": "删除空标签样本",
    "false": "保留空标签样本",
}

ECONOMICS_LABELS = {
    "ead": "EAD",
    "pd": "PD",
    "lgd": "LGD",
    "utilization": "额度使用率",
    "funding_rate": "资金成本率",
    "annual_rate": "年化利率",
    "term_months": "期限",
    "operating_cost_per_loan": "单笔运营成本",
}

SAMPLE_SCOPE_LABELS = {
    "strategy_development": "策略开发",
    "exploration_only": "仅探索分析",
}

SAMPLE_MATURITY_LABELS = {
    "confirmed_matured": "已确认成熟",
    "not_matured": "未成熟",
    "unknown": "成熟度未知",
}

CONTROL_LABELS = {
    "operation": "策略操作",
    "strategy_type": "策略类型",
    "workflow": "要执行的流程",
    "workflow_inputs": "流程参数",
    "output_prefix": "输出前缀",
    "partition": "回放分区",
    "partitions": "分区范围",
    "comparison_mode": "对比方式",
    "baseline_strategy_id": "基线策略 ID",
    "current_strategy_id": "当前策略 ID",
    "month_col": "月份列",
    "group_col": "分组列",
    "segment_col": "分群列",
    "loan_amount_col": "放款金额列",
    "overdue_amount_col": "逾期金额列",
    "sample_weight_col": "样本权重列",
    "drop_nan_labels": "空标签处理",
    "economics_inputs": "经济参数",
    "default_action": "默认动作",
    "action": "命中动作",
    "max_depth": "最大树深",
    "min_leaf_count": "叶子最少样本数",
    "min_weight_fraction_leaf": "叶子最小权重占比",
    "seed": "随机种子",
    "ead_col": "EAD 列",
    "pd_col": "PD 列",
    **ECONOMICS_LABELS,
}

# Keys the grounding checks emit as ``key value`` / ``key=value``. Only these
# split, so a user value such as the reason ``pd 过高`` or the feature column
# ``action count`` is never read as a control.
_SPACE_KEYS = frozenset(
    {
        "strategy_type",
        "partition",
        "comparison_mode",
        "output_prefix",
        "month_col",
        "group_col",
        "segment_col",
        "loan_amount_col",
        "overdue_amount_col",
    }
)
_EQUALS_KEYS = frozenset(
    {
        "drop_nan_labels",
        "sample_weight_col",
        "loan_amount_col",
        "overdue_amount_col",
        "max_depth",
        "min_leaf_count",
        "min_weight_fraction_leaf",
        "seed",
    }
)

# Already-Chinese entries (Pool reorder reports a broken order this way).
_PHRASE_LABELS = {"用户原话中的完整顺序": "用户原话中的完整顺序"}

# Controls whose value is a closed enum; every other value is user data
# (column, ID, prefix, number) and is quoted verbatim.
_ENUM_VALUE_LABELS: Mapping[str, Mapping[str, str]] = {
    "strategy_type": STRATEGY_TYPE_LABELS,
    "partition": PARTITION_LABELS,
    "comparison_mode": COMPARISON_MODE_LABELS,
    "drop_nan_labels": DROP_NAN_LABELS,
}

_ECONOMICS_PAIR_RE = re.compile(r"^(?P<name>[a-z_]+)_col/(?P=name)_value$")


def value_label(labels: Mapping[str, str], value: object) -> str:
    """Business label for one enum value; unknown values stay verbatim."""

    return labels.get(str(value), str(value))


def control_label(
    item: str,
    *,
    bare_value: str | None = None,
    bare_keys: bool = True,
) -> str:
    """Translate one ungrounded-control entry into business wording.

    Entries follow the shapes the request grounding checks emit: ``key``,
    ``key value``, ``key=value``, ``economics_inputs.<name>``,
    ``directions.<feature>=<direction>``, ``<feature>=<direction>``,
    ``features includes <column>``, ``typed action ...`` and
    ``<name>_col/<name>_value``. Anything else is a user-supplied value
    (strategy/rule ID, reason text, column) and is quoted verbatim, prefixed
    with ``bare_value`` when the caller knows which control it belongs to.
    ``bare_keys=False`` tells the parser that a bare entry is always such a
    value (an automatic-tree feature named ``pd``, a Pool reason code ``lgd``),
    never a control key.
    """

    text = str(item)
    if text in _PHRASE_LABELS:
        return _PHRASE_LABELS[text]
    if text.startswith("typed action "):
        return _typed_action_label(text[len("typed action ") :])
    if text.startswith("features includes "):
        return f"特征「{text[len('features includes ') :]}」"
    if (pair := _ECONOMICS_PAIR_RE.match(text)) and pair["name"] in ECONOMICS_LABELS:
        return ECONOMICS_LABELS[pair["name"]]
    if text.startswith("economics_inputs."):
        name = text[len("economics_inputs.") :]
        return f"经济参数 {ECONOMICS_LABELS.get(name, name)}"
    if text.startswith("directions.") and "=" in text:
        feature, direction = text[len("directions.") :].rsplit("=", 1)
        return _direction_label(feature, direction)
    # No control takes a direction word as its value, so ``<x>=increasing``
    # is a feature direction even when the column is named like a control.
    feature, found, direction = text.rpartition("=")
    if found and feature and direction in TREE_DIRECTION_LABELS:
        return _direction_label(feature, direction)
    for separator, keys in ((" ", _SPACE_KEYS), ("=", _EQUALS_KEYS)):
        key, found, value = text.partition(separator)
        if found and key in keys:
            return _keyed_value_label(key, value)
    if bare_keys and text in CONTROL_LABELS:
        return CONTROL_LABELS[text]
    return f"{bare_value or ''}「{text}」"


def control_labels(
    items: Iterable[str],
    *,
    bare_value: str | None = None,
    bare_keys: bool = True,
) -> list[str]:
    """Ordered, de-duplicated business labels for ungrounded controls."""

    return list(
        dict.fromkeys(
            control_label(item, bare_value=bare_value, bare_keys=bare_keys)
            for item in items
        )
    )


def join_control_labels(
    items: Iterable[str],
    *,
    bare_value: str | None = None,
    bare_keys: bool = True,
) -> str:
    return "、".join(
        control_labels(items, bare_value=bare_value, bare_keys=bare_keys)
    )


def _keyed_value_label(key: str, value: str) -> str:
    label = CONTROL_LABELS[key]
    if not value or value == "unknown":
        return label
    enum_labels = _ENUM_VALUE_LABELS.get(key)
    if enum_labels is not None:
        return f"{label}（{value_label(enum_labels, value)}）"
    return f"{label}「{value}」"


def _typed_action_label(rest: str) -> str:
    for field, label in (("output_value ", "动作输出值"), ("value ", "动作取值")):
        if rest.startswith(field):
            return f"{label}「{rest[len(field) :]}」"
    if not rest or rest == "unknown":
        return "动作类型"
    return f"动作类型（{value_label(POOL_ACTION_TYPE_LABELS, rest)}）"


def _direction_label(feature: str, direction: str) -> str:
    return f"特征「{feature}」的方向（{value_label(TREE_DIRECTION_LABELS, direction)}）"


__all__ = [
    "COMPARISON_MODE_LABELS",
    "CONTROL_LABELS",
    "DROP_NAN_LABELS",
    "ECONOMICS_LABELS",
    "PARTITION_LABELS",
    "POOL_ACTION_TYPE_LABELS",
    "SAMPLE_MATURITY_LABELS",
    "SAMPLE_SCOPE_LABELS",
    "STRATEGY_TYPE_LABELS",
    "TREE_DIRECTION_LABELS",
    "control_label",
    "control_labels",
    "join_control_labels",
    "value_label",
]

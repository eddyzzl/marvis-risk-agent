"""Deterministic split configuration shared by intake, gates and native tools."""
from __future__ import annotations

import math
from numbers import Real


class SplitConfigurationError(ValueError):
    """A declared split cannot be admitted without changing its meaning."""


_FIELDS = frozenset({"method", "test_size", "oot_size", "oot_by_time", "random_oot", "group_cols", "rules"})
_OPS = frozenset({"eq", "ne", "in", "lt", "le", "gt", "ge"})
_EXISTING_LABELS = {
    "train": frozenset({"train", "training", "dev", "develop", "development", "build"}),
    "test": frozenset({"test", "testing", "valid", "validation", "val", "holdout"}),
    "oot": frozenset({"oot", "ootest", "out_of_time", "oos", "time_oot"}),
}


def _column(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 128:
        raise SplitConfigurationError("切分字段必须是有效列名字符串，未执行。")
    return value


def _literal(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real) and math.isfinite(float(value)):
        return value
    raise SplitConfigurationError("切分规则值必须是有限数字、字符串、布尔值或空值，未执行。")


def validate_split_configuration(value, *, columns=None, split_col=None, require_existing_binding=False):
    """Return an exact typed copy; unknown controls never become a new split.

    Empty configuration retains the legacy default/passthrough behavior. The
    explicit existing descriptor is distinct: it requires an existing column
    at execution and cannot be combined with new partition controls.
    """
    if split_col is not None and split_col != "":
        _column(split_col)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SplitConfigurationError("split_config 必须是对象，未执行。")
    if set(value) - _FIELDS:
        raise SplitConfigurationError("split_config 包含不支持字段，未执行。")
    config = dict(value)
    if "method" in config:
        if config["method"] != "existing" or len(config) != 1:
            raise SplitConfigurationError("existing 仅用于保留现有切分，不能包含新划分或未知方式，未执行。")
        if require_existing_binding:
            _column(split_col)
            if columns is not None and split_col not in columns:
                raise SplitConfigurationError("existing 必须绑定已存在的切分列，未执行。")
        return config
    for key in ("test_size", "oot_size"):
        if key in config:
            number = config[key]
            if isinstance(number, bool) or not isinstance(number, Real) or not math.isfinite(float(number)) or not 0 <= number <= 1:
                raise SplitConfigurationError(f"split_config.{key} 必须是 0 到 1 之间的有限数字，未执行。")
    time_col = config.get("oot_by_time")
    if time_col is not None:
        _column(time_col)
        if columns is not None and time_col not in columns:
            raise SplitConfigurationError("时间 OOT 列不在当前已认证数据中，未执行。")
    if "random_oot" in config and type(config["random_oot"]) is not bool:
        raise SplitConfigurationError("split_config.random_oot 必须是布尔值，未执行。")
    if time_col and config.get("random_oot"):
        raise SplitConfigurationError("时间 OOT 与随机 OOT 不能同时启用，未执行。")
    groups = config.get("group_cols")
    if groups is not None:
        # The deterministic kernel historically supports one field or a list.
        groups = [groups] if isinstance(groups, str) else groups
        if not isinstance(groups, list):
            raise SplitConfigurationError("分组列必须是非空字段名或字段名列表，未执行。")
        config["group_cols"] = [_column(column) for column in groups]
        if columns is not None:
            missing = [column for column in groups if column not in columns]
            if missing:
                raise SplitConfigurationError(f"分组列缺失: {', '.join(missing)}；请补充隔离键后重新切分。")
    rules = config.get("rules")
    if rules is not None:
        if not isinstance(rules, list):
            raise SplitConfigurationError("切分规则必须是列表，未执行。")
        admitted = []
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {"when", "assign"}:
                raise SplitConfigurationError("切分规则必须仅包含 when 与 assign，未执行。")
            if not isinstance(rule["assign"], str) or rule["assign"] not in {"train", "test", "oot"}:
                raise SplitConfigurationError("split rule has invalid assign; expected train/test/oot")
            conditions = rule["when"]
            if not isinstance(conditions, list) or not conditions:
                raise SplitConfigurationError("split rule must have a non-empty 'when' list")
            clean = []
            for condition in conditions:
                if not isinstance(condition, dict) or set(condition) != {"col", "op", "val"}:
                    raise SplitConfigurationError("切分条件必须仅包含 col、op 与 val，未执行。")
                column = _column(condition["col"])
                op = condition["op"]
                if not isinstance(op, str) or op not in _OPS:
                    raise SplitConfigurationError("split rule uses unknown op")
                if columns is not None and column not in columns:
                    raise SplitConfigurationError(f"missing columns: {column}")
                literal = condition["val"]
                if op == "in":
                    if not isinstance(literal, list):
                        raise SplitConfigurationError("切分 in 条件必须使用值列表，未执行。")
                    literal = [_literal(item) for item in literal]
                else:
                    literal = _literal(literal)
                clean.append({"col": column, "op": op, "val": literal})
            admitted.append({"when": clean, "assign": rule["assign"]})
        config["rules"] = admitted
    return config


def existing_split_values(values):
    """Authenticate every present label, preserving original membership literals.

    These are the established sample-setup aliases. Multiple literals for one
    role cannot be represented by the native one-value-per-role contract.
    """
    mapping = {}
    for value in values:
        if not isinstance(value, str):
            raise SplitConfigurationError("现有切分包含空值或非字符串成员标签，请确认后再建模。")
        role = next((name for name, labels in _EXISTING_LABELS.items() if value.strip().lower() in labels), None)
        if role is None or (role in mapping and mapping[role] != value):
            raise SplitConfigurationError("现有切分包含未知或同角色重复成员标签，请确认后再建模。")
        mapping[role] = value
    if "train" not in mapping or not ({"test", "oot"} & mapping.keys()):
        raise SplitConfigurationError("现有切分必须包含 train 和 test/OOT 成员，请确认后再建模。")
    return mapping


def effective_split_column(split_col, split_config):
    """Name the actual output column for a typed split descriptor."""
    split_config = validate_split_configuration(split_config, split_col=split_col)
    if not split_config or split_config.get("method") == "existing":
        return split_col or "split"
    return "split"

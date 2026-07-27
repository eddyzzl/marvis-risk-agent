from __future__ import annotations

from marvis.agent.dataset_transform import (
    build_dataset_transform_request,
    detect_dataset_transform_intent,
)
from marvis.data.workspace import DataSemanticMapping


COLUMNS = (
    "customer_id",
    "amount",
    "income",
    "channel",
    "apply_time",
    "bad",
    "mobile",
)
BUSINESS_NAMES = {
    "customer_id": "客户编号",
    "amount": "申请金额",
    "income": "月收入",
    "channel": "申请渠道",
    "apply_time": "申请时间",
    "bad": "风险标签",
    "mobile": "手机号",
}
SEMANTICS = DataSemanticMapping(
    target_col="bad",
    field_roles={
        "customer_id": "id",
        "bad": "target",
        "mobile": "phone",
    },
    business_names=BUSINESS_NAMES,
)


def _build(text: str):
    return build_dataset_transform_request(
        text,
        columns=COLUMNS,
        business_names=BUSINESS_NAMES,
        semantic_mapping=SEMANTICS,
    )


def test_detect_transform_intent_is_specific_to_governed_data_changes():
    assert detect_dataset_transform_intent("把申请金额重命名为 loan_amount")
    assert detect_dataset_transform_intent("fill missing amount with median")
    assert detect_dataset_transform_intent("filter rows where amount >= 1000")
    assert detect_dataset_transform_intent("按客户编号去重，保留申请时间最新一条")
    assert not detect_dataset_transform_intent("回测策略并比较通过率")
    assert not detect_dataset_transform_intent("删除拒绝规则")
    assert not detect_dataset_transform_intent("确认")
    assert not detect_dataset_transform_intent(None)


def test_rename_and_drop_resolve_business_names_to_raw_columns():
    renamed = _build("把申请金额重命名为 loan_amount，并将风险标签 rename to label")
    assert renamed.clarification is None
    assert renamed.request is not None
    assert renamed.request.operations == (
        {
            "op": "rename_columns",
            "mapping": {"amount": "loan_amount", "bad": "label"},
        },
    )
    assert renamed.request.confirm_protected_drop is False

    dropped = _build("drop columns 月收入, 申请渠道")
    assert dropped.clarification is None
    assert dropped.request is not None
    assert dropped.request.operations == (
        {"op": "drop_columns", "columns": ["income", "channel"]},
    )


def test_mixed_known_and_unknown_fields_fail_closed_for_the_whole_request():
    result = _build("删除申请金额和 ghost_field")
    assert result.request is None
    assert result.clarification is not None
    assert "ghost_field" in result.clarification

    result = _build("筛选申请金额 >= 1000 且 unknown_score < 20")
    assert result.request is None
    assert result.clarification is not None
    assert "unknown_score" in result.clarification


def test_fill_missing_supports_statistics_and_typed_constants():
    result = _build(
        "用中位数填充申请金额缺失值，并用常量 'UNKNOWN' 填充申请渠道缺失值"
    )
    assert result.clarification is None
    assert result.request is not None
    assert result.request.operations == (
        {
            "op": "fill_missing",
            "fills": [
                {"column": "amount", "method": "median"},
                {
                    "column": "channel",
                    "method": "constant",
                    "value": "UNKNOWN",
                },
            ],
        },
    )

    zero = _build("fill missing 月收入 with 0")
    assert zero.request is not None
    assert zero.request.operations == (
        {
            "op": "fill_missing",
            "fills": [{"column": "income", "method": "constant", "value": 0}],
        },
    )


def test_cast_requires_explicit_mode_and_emits_safe_canonical_types():
    result = _build("将申请金额尝试转换为 DOUBLE，并把风险标签严格转为 INTEGER")
    assert result.clarification is None
    assert result.request is not None
    assert result.request.operations == (
        {
            "op": "cast_columns",
            "casts": [
                {"column": "amount", "to_type": "DOUBLE", "mode": "try"},
                {"column": "bad", "to_type": "INTEGER", "mode": "strict"},
            ],
        },
    )

    missing_mode = _build("cast amount to DOUBLE")
    assert missing_mode.request is None
    assert missing_mode.clarification is not None
    assert "strict" in missing_mode.clarification
    assert "try" in missing_mode.clarification


def test_single_and_multi_condition_filters_emit_typed_predicate_ast():
    single = _build("筛选申请金额大于等于1000")
    assert single.request is not None
    assert single.request.operations == (
        {
            "op": "filter_rows",
            "predicate": {
                "op": "gte",
                "left": {"column": "amount"},
                "right": {"literal": 1000},
            },
        },
    )

    multiple = _build("filter rows where 申请金额 >= 1000 and 申请渠道 = 'APP'")
    assert multiple.clarification is None
    assert multiple.request is not None
    assert multiple.request.operations == (
        {
            "op": "filter_rows",
            "predicate": {
                "op": "and",
                "args": [
                    {
                        "op": "gte",
                        "left": {"column": "amount"},
                        "right": {"literal": 1000},
                    },
                    {
                        "op": "eq",
                        "left": {"column": "channel"},
                        "right": {"literal": "APP"},
                    },
                ],
            },
        },
    )

    null_filter = _build("过滤手机号不为空")
    assert null_filter.request is not None
    assert null_filter.request.operations == (
        {
            "op": "filter_rows",
            "predicate": {
                "op": "is_not_null",
                "arg": {"column": "mobile"},
            },
        },
    )


def test_derive_accepts_only_one_bounded_arithmetic_expression():
    result = _build("新增字段 debt_ratio = 申请金额 / 月收入")
    assert result.clarification is None
    assert result.request is not None
    assert result.request.operations == (
        {
            "op": "derive_columns",
            "derivations": [
                {
                    "name": "debt_ratio",
                    "expression": {
                        "op": "divide",
                        "left": {"column": "amount"},
                        "right": {"column": "income"},
                    },
                }
            ],
        },
    )

    unsafe = _build("新增字段 hacked = python(__import__('os'))")
    assert unsafe.request is None
    assert unsafe.clarification is not None


def test_later_steps_can_reference_columns_renamed_or_derived_earlier():
    renamed_then_dropped = _build("把申请金额重命名为 loan_amount；再删除 loan_amount")
    assert renamed_then_dropped.clarification is None
    assert renamed_then_dropped.request is not None
    assert renamed_then_dropped.request.operations == (
        {"op": "rename_columns", "mapping": {"amount": "loan_amount"}},
        {"op": "drop_columns", "columns": ["loan_amount"]},
    )

    derived_then_cast = _build(
        "新增字段 debt_ratio = 申请金额 / 月收入；再把 debt_ratio 尝试转为 DOUBLE"
    )
    assert derived_then_cast.clarification is None
    assert derived_then_cast.request is not None
    assert derived_then_cast.request.operations == (
        {
            "op": "derive_columns",
            "derivations": [
                {
                    "name": "debt_ratio",
                    "expression": {
                        "op": "divide",
                        "left": {"column": "amount"},
                        "right": {"column": "income"},
                    },
                }
            ],
        },
        {
            "op": "cast_columns",
            "casts": [
                {"column": "debt_ratio", "to_type": "DOUBLE", "mode": "try"}
            ],
        },
    )

    chained_derivation = _build(
        "新增字段 debt_ratio = 申请金额 / 月收入；"
        "再新增字段 adjusted_ratio = debt_ratio * 100"
    )
    assert chained_derivation.request is not None
    assert [item["op"] for item in chained_derivation.request.operations] == [
        "derive_columns",
        "derive_columns",
    ]

    renamed_target = _build("把风险标签重命名为 label；再删除 label")
    assert renamed_target.request is None
    assert renamed_target.protected_fields == ("label",)
    assert "确认" in str(renamed_target.clarification)


def test_repeated_casts_and_fills_on_one_column_remain_ordered_steps():
    casts = _build(
        "把申请金额严格转为 VARCHAR；再把申请金额严格转为 DOUBLE"
    )
    assert casts.request is not None
    assert casts.request.operations == (
        {
            "op": "cast_columns",
            "casts": [
                {"column": "amount", "to_type": "VARCHAR", "mode": "strict"}
            ],
        },
        {
            "op": "cast_columns",
            "casts": [
                {"column": "amount", "to_type": "DOUBLE", "mode": "strict"}
            ],
        },
    )

    fills = _build("用常量 0 填充申请金额缺失值；再用均值填充申请金额缺失值")
    assert fills.request is not None
    assert [item["op"] for item in fills.request.operations] == [
        "fill_missing",
        "fill_missing",
    ]


def test_deduplicate_requires_keys_and_explicit_order():
    result = _build("按客户编号去重，保留申请时间最新一条，空值最后")
    assert result.clarification is None
    assert result.request is not None
    assert result.request.operations == (
        {
            "op": "deduplicate",
            "keys": ["customer_id"],
            "order_by": [
                {"column": "apply_time", "direction": "desc", "nulls": "last"}
            ],
        },
    )

    english = _build(
        "deduplicate by customer_id order by apply_time asc nulls first"
    )
    assert english.request is not None
    assert english.request.operations == (
        {
            "op": "deduplicate",
            "keys": ["customer_id"],
            "order_by": [
                {"column": "apply_time", "direction": "asc", "nulls": "first"}
            ],
        },
    )

    missing_order = _build("按客户编号去重")
    assert missing_order.request is None
    assert missing_order.clarification is not None
    assert "排序" in missing_order.clarification


def test_protected_drop_needs_explicit_confirmation_and_sets_request_flag():
    blocked = _build("删除风险标签和手机号")
    assert blocked.request is None
    assert blocked.clarification is not None
    assert "确认" in blocked.clarification

    confirmed = _build("我确认删除风险标签和手机号")
    assert confirmed.clarification is None
    assert confirmed.request is not None
    assert confirmed.request.operations == (
        {"op": "drop_columns", "columns": ["bad", "mobile"]},
    )
    assert confirmed.request.confirm_protected_drop is True


def test_protected_drop_confirmation_is_bound_to_the_specific_drop_action():
    mixed = _build("我确认删除月收入；删除风险标签")

    assert mixed.request is None
    assert mixed.clarification is not None
    assert "确认" in mixed.clarification
    assert mixed.protected_fields == ("bad",)
    assert mixed.operations == (
        {"op": "drop_columns", "columns": ["income", "bad"]},
    )

    fully_confirmed = _build("我确认删除风险标签和手机号")
    assert fully_confirmed.request is not None
    assert fully_confirmed.protected_fields == ()
    assert fully_confirmed.request.confirm_protected_drop is True


def test_missing_parameters_and_sql_or_python_requests_return_clarification():
    missing_target = _build("把申请金额重命名")
    assert missing_target.request is None
    assert missing_target.clarification is not None

    sql = _build("用 SQL 删除 amount 列")
    assert sql.request is None
    assert sql.clarification is not None
    assert "SQL" in sql.clarification

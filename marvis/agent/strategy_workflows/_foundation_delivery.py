"""Canonical foundation and delivery workflow specifications.

This module owns validation, deterministic confirmation, and plan preparation
only.  Platform-owned evidence enters through preparation callbacks; no adapter
invokes a Tool or reaches into the legacy compiler/turn-handler implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
import json
import math
import re
from typing import Any

from marvis.data.predicate_ast import PredicateAstError, canonicalize_predicate

from .contracts import (
    PreparedStrategyPlan,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowSpec,
    StrategyWorkflowValidationError,
    deep_freeze,
    deep_thaw,
)


_PROJECT_CONTEXT_FIELD_PATH_RE = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,7}$"
)
_SAMPLE_DESIGN_STATUS_VALUES = {
    "performance_window_status": frozenset({"provided", "unavailable"}),
    "observation_window_status": frozenset({"provided", "unavailable"}),
    "maturity_status": frozenset({"confirmed_matured", "not_matured", "unknown"}),
}
_STRATEGY_DSL_DELIVERY_STRATEGY_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])strategy-[A-Za-z0-9][A-Za-z0-9_-]*"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_STRATEGY_REPORT_DEFAULT_TITLE = "策略迭代评审报告"
_STRATEGY_REPORT_DEFAULT_STATUS = "partial"
_STRATEGY_REPORT_STATUSES = frozenset({"draft", "partial", "final"})


def validate_project_context_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate only project facts owned by the user."""

    workflow = "strategy_project_context"
    allowed = {
        "as_of",
        "scope",
        "business_context",
        "explicit_unavailable",
        "external_report_filenames",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    if "as_of" not in inputs:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少 as_of；请明确本次项目现状的截止日期。"
        )
    as_of = _required_text(inputs["as_of"], name=f"{workflow} as_of")
    try:
        parsed_as_of = date.fromisoformat(as_of)
    except ValueError as exc:
        raise StrategyWorkflowValidationError(
            f"{workflow} as_of 必须是 YYYY-MM-DD ISO 日期。"
        ) from exc
    if parsed_as_of.isoformat() != as_of:
        raise StrategyWorkflowValidationError(
            f"{workflow} as_of 必须是 YYYY-MM-DD ISO 日期。"
        )

    normalized: dict[str, Any] = {"as_of": as_of}
    if "scope" in inputs:
        raw_scope = inputs["scope"]
        if raw_scope is None:
            normalized["scope"] = None
        else:
            scope = _required_text(raw_scope, name=f"{workflow} scope")
            if len(scope) > 4000:
                raise StrategyWorkflowValidationError(
                    f"{workflow} scope 最多 4000 个字符。"
                )
            normalized["scope"] = scope

    raw_context = inputs.get("business_context", {})
    if not isinstance(raw_context, Mapping) or len(raw_context) > 50:
        raise StrategyWorkflowValidationError(
            f"{workflow} business_context 必须是最多 50 个字段的对象。"
        )
    business_context: dict[str, str | None] = {}
    for raw_path, raw_value in raw_context.items():
        if not isinstance(raw_path, str):
            raise StrategyWorkflowValidationError(
                f"{workflow} business_context 字段名必须是文本路径。"
            )
        field_path = raw_path.strip()
        if (
            not _PROJECT_CONTEXT_FIELD_PATH_RE.fullmatch(field_path)
            or len(field_path) > 256
        ):
            raise StrategyWorkflowValidationError(
                f"{workflow} business_context 字段路径无效：{raw_path!r}。"
            )
        if raw_value is None:
            business_context[field_path] = None
            continue
        value = _required_text(
            raw_value,
            name=f"{workflow} business_context.{field_path}",
        )
        if len(value) > 4000:
            raise StrategyWorkflowValidationError(
                f"{workflow} business_context.{field_path} 最多 4000 个字符。"
            )
        business_context[field_path] = value
    normalized["business_context"] = business_context
    normalized["explicit_unavailable"] = _project_context_text_list(
        inputs.get("explicit_unavailable", []),
        name=f"{workflow} explicit_unavailable",
        maximum=100,
        field_paths=True,
    )
    normalized["external_report_filenames"] = _project_context_text_list(
        inputs.get("external_report_filenames", []),
        name=f"{workflow} external_report_filenames",
        maximum=20,
        field_paths=False,
    )
    return normalized


def validate_report_bundle_v2_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, str]:
    """Validate the only two caller-owned report controls."""

    workflow = "strategy_report_bundle_v2"
    allowed = {"title", "status"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 只允许 title/status；平台字段 "
            + "、".join(unexpected)
            + " 由 Agent 在计划创建时绑定。",
            code="strategy_report_bundle_v2_platform_binding_forbidden",
            fields=unexpected,
        )
    title = inputs.get("title", _STRATEGY_REPORT_DEFAULT_TITLE)
    if not isinstance(title, str) or not title.strip() or "\x00" in title:
        raise StrategyWorkflowValidationError(
            "strategy_report_bundle_v2 title 必须是非空文本。",
            code="strategy_report_bundle_v2_title_invalid",
            fields=("title",),
        )
    title = title.strip()
    if len(title) > 200:
        raise StrategyWorkflowValidationError(
            "strategy_report_bundle_v2 title 最多 200 个字符。",
            code="strategy_report_bundle_v2_title_invalid",
            fields=("title",),
        )
    status = inputs.get("status", _STRATEGY_REPORT_DEFAULT_STATUS)
    if not isinstance(status, str) or status not in _STRATEGY_REPORT_STATUSES:
        raise StrategyWorkflowValidationError(
            "strategy_report_bundle_v2 status 只能是 draft、partial 或 final。",
            code="strategy_report_bundle_v2_status_invalid",
            fields=("status",),
        )
    return {"title": title, "status": status}


def validate_dsl_delivery_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, str]:
    """Accept only the optional strategy identifier owned by the user."""

    workflow = "strategy_dsl_delivery"
    allowed = {"strategy_id"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 只允许 strategy_id；平台字段 "
            + "、".join(unexpected)
            + " 由 Agent 在计划创建时绑定。",
            code="strategy_dsl_delivery_platform_binding_forbidden",
            fields=unexpected,
        )
    if "strategy_id" not in inputs:
        return {}
    strategy_id = _required_text(
        inputs["strategy_id"],
        name="strategy_dsl_delivery strategy_id",
    )
    if (
        len(strategy_id) > 128
        or _STRATEGY_DSL_DELIVERY_STRATEGY_ID_RE.fullmatch(strategy_id) is None
    ):
        raise StrategyWorkflowValidationError(
            "strategy_dsl_delivery strategy_id 必须是完整的 strategy-* ID。",
            code="strategy_dsl_delivery_strategy_id_invalid",
            fields=("strategy_id",),
        )
    return {"strategy_id": strategy_id}


def validate_model_evidence_v2_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    if inputs:
        fields = tuple(sorted(inputs))
        raise StrategyWorkflowValidationError(
            "strategy_model_evidence_v2 workflow_inputs 必须是空对象；"
            "SampleDesign 与认证单变量候选引用全部由当前 task 绑定。",
            code="strategy_model_evidence_v2_platform_binding_forbidden",
            fields=fields,
        )
    return {}


def validate_legacy_sample_design_replay_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Compatibility adapter retained only for persisted V1 request replay."""

    workflow = "strategy_sample_design"
    allowed = {
        "performance_window_status",
        "performance_window_days",
        "observation_window_status",
        "observation_start",
        "observation_end",
        "maturity_status",
        "target_bad_value",
        "split_col",
        "development_values",
        "validation_values",
        "oot_values",
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    required_controls = {*_SAMPLE_DESIGN_STATUS_VALUES, "target_bad_value"}
    missing_controls = sorted(required_controls - set(inputs))
    if missing_controls:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少必需口径：" + "、".join(missing_controls) + "。"
        )

    normalized: dict[str, Any] = {}
    for field, values in _SAMPLE_DESIGN_STATUS_VALUES.items():
        value = inputs[field]
        if not isinstance(value, str) or value not in values:
            raise StrategyWorkflowValidationError(
                f"{workflow} {field} 只能是：" + "、".join(sorted(values)) + "。"
            )
        normalized[field] = value

    target_bad_value = inputs["target_bad_value"]
    if (
        isinstance(target_bad_value, bool)
        or not isinstance(target_bad_value, (int, float))
        or not math.isfinite(float(target_bad_value))
        or float(target_bad_value) not in {0.0, 1.0}
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} target_bad_value 必须是整数 0 或 1，不能是布尔值。"
        )
    normalized["target_bad_value"] = int(target_bad_value)

    performance_status = normalized["performance_window_status"]
    if performance_status == "provided":
        days = inputs.get("performance_window_days")
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise StrategyWorkflowValidationError(
                f"{workflow} performance_window_days 必须是正整数。"
            )
        normalized["performance_window_days"] = days
    elif "performance_window_days" in inputs:
        raise StrategyWorkflowValidationError(
            f"{workflow} 表现窗 unavailable 时不能填写 performance_window_days。"
        )

    observation_status = normalized["observation_window_status"]
    observation_fields = {"observation_start", "observation_end"}
    if observation_status == "provided":
        missing = sorted(observation_fields - set(inputs))
        if missing:
            raise StrategyWorkflowValidationError(
                f"{workflow} 观察窗 provided 时缺少：" + "、".join(missing) + "。"
            )
        parsed_dates: dict[str, date] = {}
        for field in sorted(observation_fields):
            value = inputs[field]
            if not isinstance(value, str):
                raise StrategyWorkflowValidationError(
                    f"{workflow} {field} 必须是 ISO 日期。"
                )
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise StrategyWorkflowValidationError(
                    f"{workflow} {field} 必须是 YYYY-MM-DD ISO 日期。"
                ) from exc
            if parsed.isoformat() != value:
                raise StrategyWorkflowValidationError(
                    f"{workflow} {field} 必须是 YYYY-MM-DD ISO 日期。"
                )
            parsed_dates[field] = parsed
            normalized[field] = value
        if parsed_dates["observation_start"] > parsed_dates["observation_end"]:
            raise StrategyWorkflowValidationError(
                f"{workflow} observation_start 不能晚于 observation_end。"
            )
    else:
        unexpected = sorted(observation_fields & set(inputs))
        if unexpected:
            raise StrategyWorkflowValidationError(
                f"{workflow} 观察窗 unavailable 时不能填写："
                + "、".join(unexpected)
                + "。"
            )

    split_fields = {
        "split_col",
        "development_values",
        "validation_values",
        "oot_values",
    }
    supplied_split_fields = split_fields & set(inputs)
    if supplied_split_fields and supplied_split_fields != split_fields:
        missing = sorted(split_fields - supplied_split_fields)
        raise StrategyWorkflowValidationError(
            f"{workflow} 指定切分时必须同时提供 split_col 与三组值；缺少："
            + "、".join(missing)
            + "。"
        )
    if supplied_split_fields:
        split_col = _column(
            inputs["split_col"],
            name=f"{workflow} split_col",
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and split_col == context.target_col:
            raise StrategyWorkflowValidationError(
                f"{workflow} split_col 不能使用目标列。"
            )
        normalized["split_col"] = split_col
        value_sets: dict[str, set[tuple[str, object]]] = {}
        for field in (
            "development_values",
            "validation_values",
            "oot_values",
        ):
            values = _sample_design_value_sequence(
                inputs[field],
                name=field,
                minimum_items=1 if field == "development_values" else 0,
            )
            normalized[field] = values
            value_sets[field] = {_sample_design_value_identity(item) for item in values}
        overlaps: list[str] = []
        for left, right in (
            ("development_values", "validation_values"),
            ("development_values", "oot_values"),
            ("validation_values", "oot_values"),
        ):
            if value_sets[left] & value_sets[right]:
                overlaps.append(f"{left}/{right}")
        if overlaps:
            raise StrategyWorkflowValidationError(
                f"{workflow} 三组 split values 必须互不重叠："
                + "、".join(overlaps)
                + "。"
            )

    for field in (
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        if field not in inputs:
            continue
        column = _column(
            inputs[field],
            name=f"{workflow} {field}",
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and column == context.target_col:
            raise StrategyWorkflowValidationError(
                f"{workflow} {field} 不能使用目标列。"
            )
        normalized[field] = column

    bound_columns = [
        normalized[field]
        for field in (
            "split_col",
            "month_col",
            "weight_col",
            "loan_amount_col",
            "overdue_amount_col",
        )
        if field in normalized
    ]
    if len(bound_columns) != len(set(bound_columns)):
        raise StrategyWorkflowValidationError(
            f"{workflow} 切分、月份、权重和金额字段必须彼此不同。"
        )
    if "drop_nan_labels" in inputs:
        value = inputs["drop_nan_labels"]
        if not isinstance(value, bool):
            raise StrategyWorkflowValidationError(
                f"{workflow} drop_nan_labels 必须是布尔值。"
            )
        normalized["drop_nan_labels"] = value
    return normalized


def validate_sample_design_v2_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate the bounded fresh V2 sample-design request surface."""

    return _validate_sample_design_v2_inputs(
        inputs,
        context,
        allow_raw_population_ast=False,
        allow_recursive_partition_ast=False,
    )


def validate_sample_design_v2_replay_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate normalized persisted V2 input without weakening fresh intake.

    The current workflow contract does not carry a fresh/replay mode into a
    ``StrategyWorkflowSpec`` validator.  Main integration must route persisted
    replay to this adapter (or add an equivalent mode) before deleting the
    legacy compiler branch.
    """

    return _validate_sample_design_v2_inputs(
        inputs,
        context,
        allow_raw_population_ast=True,
        allow_recursive_partition_ast=True,
    )


def _validate_sample_design_v2_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
    *,
    allow_raw_population_ast: bool,
    allow_recursive_partition_ast: bool,
) -> dict[str, Any]:
    workflow = "strategy_sample_design_v2"
    allowed = {
        "target_bad_value",
        "drop_nan_labels",
        "relationship",
        "approval_population",
        "risk_population",
        "partitioning",
        "maturity",
        "performance_window",
        "observation_window",
        "field_bindings",
        "historical_score",
    }
    unexpected = tuple(sorted(set(inputs) - allowed))
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 只能包含用户拥有的样本口径；平台字段不允许："
            + "、".join(unexpected)
            + "。",
            code="strategy_sample_design_v2_platform_binding_forbidden",
            fields=unexpected,
        )
    missing = tuple(sorted(allowed - set(inputs)))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少必需口径：" + "、".join(missing) + "。",
            fields=missing,
        )

    bad_value = inputs["target_bad_value"]
    if (
        isinstance(bad_value, bool)
        or not isinstance(bad_value, int)
        or bad_value not in {0, 1}
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} target_bad_value 必须是整数 0 或 1。",
            fields=("target_bad_value",),
        )
    drop_missing = inputs["drop_nan_labels"]
    if not isinstance(drop_missing, bool):
        raise StrategyWorkflowValidationError(
            f"{workflow} drop_nan_labels 必须是布尔值。",
            fields=("drop_nan_labels",),
        )
    relationship = inputs["relationship"]
    if relationship not in {"nested_same_cohort", "parallel_time_cohorts"}:
        raise StrategyWorkflowValidationError(
            f"{workflow} relationship 只能是 nested_same_cohort 或 "
            "parallel_time_cohorts。",
            fields=("relationship",),
        )

    approval = _validate_sample_v2_population(
        inputs["approval_population"],
        name="approval_population",
        context=context,
        allow_raw_ast=allow_raw_population_ast,
    )
    risk = _validate_sample_v2_population(
        inputs["risk_population"],
        name="risk_population",
        context=context,
        allow_raw_ast=allow_raw_population_ast,
    )
    partitioning = _validate_sample_v2_partitioning(
        inputs["partitioning"],
        context=context,
        allow_recursive_ast=allow_recursive_partition_ast,
    )
    maturity = _validate_sample_v2_maturity(inputs["maturity"])
    performance = _validate_sample_v2_performance_window(inputs["performance_window"])
    observation = _validate_sample_v2_observation_window(inputs["observation_window"])
    if maturity["status"] in {"confirmed_matured", "not_matured"}:
        if performance["status"] != "provided":
            raise StrategyWorkflowValidationError(
                f"{workflow} 已评估成熟度必须同时提供表现窗。",
                fields=("maturity", "performance_window"),
            )
        if maturity["performance_window_days"] != performance["days"]:
            raise StrategyWorkflowValidationError(
                f"{workflow} maturity 与 performance_window 天数必须一致。",
                fields=(
                    "maturity.performance_window_days",
                    "performance_window.days",
                ),
            )
    fields = _validate_sample_v2_field_bindings(
        inputs["field_bindings"],
        context=context,
    )
    if partitioning["method"] == "time_ranges" and (
        fields["time_field"] is None or partitioning["column"] != fields["time_field"]
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} time_ranges.column 必须与非空的 "
            "field_bindings.time_field 完全相同。",
            fields=("partitioning.column", "field_bindings.time_field"),
        )
    if (
        maturity["status"] in {"confirmed_matured", "not_matured"}
        and fields["time_field"] is None
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} 已评估成熟度需要 time_field。",
            fields=("field_bindings.time_field",),
        )
    if observation["status"] == "provided" and fields["time_field"] is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} 已提供观察窗需要 time_field。",
            fields=("field_bindings.time_field",),
        )
    historical = _validate_sample_v2_historical_score(
        inputs["historical_score"],
        context=context,
    )
    return {
        "target_bad_value": bad_value,
        "drop_nan_labels": drop_missing,
        "relationship": relationship,
        "approval_population": approval,
        "risk_population": risk,
        "partitioning": partitioning,
        "maturity": maturity,
        "performance_window": performance,
        "observation_window": observation,
        "field_bindings": fields,
        "historical_score": historical,
    }


def _validate_sample_v2_population(
    value: object,
    *,
    name: str,
    context: StrategyWorkflowResolutionContext,
    allow_raw_ast: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"inclusion", "exclusion"}:
        raise StrategyWorkflowValidationError(
            f"{name} 必须且只能包含 inclusion 与 exclusion。",
            fields=(name,),
        )
    normalized: dict[str, Any] = {}
    for field in ("inclusion", "exclusion"):
        predicate = value[field]
        if predicate is None:
            normalized[field] = None
            continue
        if not allow_raw_ast:
            predicate = _sample_v2_population_filter_to_ast(
                predicate,
                name=f"{name}.{field}",
                context=context,
            )
        try:
            canonical = canonicalize_predicate(
                predicate,
                context.allowed_columns,
                max_nodes=256,
                max_depth=12,
            )
        except PredicateAstError as exc:
            raise StrategyWorkflowValidationError(
                f"{name}.{field} 不是受支持的严格 predicate AST：{exc}",
                fields=(f"{name}.{field}",),
            ) from exc
        if (
            context.target_col is not None
            and context.target_col in canonical.required_columns
        ):
            raise StrategyWorkflowValidationError(
                f"{name}.{field} 不能用目标列定义样本总体。",
                fields=(f"{name}.{field}",),
            )
        normalized[field] = canonical.canonical
    return normalized


def _sample_v2_population_filter_to_ast(
    value: object,
    *,
    name: str,
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"match", "conditions"}
        or value.get("match") not in {"all", "any"}
    ):
        raise StrategyWorkflowValidationError(
            f"{name} 必须是只含 match 与 conditions 的受限条件对象；"
            "fresh 请求不能直接提交 predicate AST。",
            code="strategy_sample_design_v2_population_dto_required",
            fields=(name,),
        )
    conditions = value["conditions"]
    if (
        not isinstance(conditions, Sequence)
        or isinstance(conditions, str | bytes)
        or not 1 <= len(conditions) <= 8
    ):
        raise StrategyWorkflowValidationError(
            f"{name}.conditions 必须包含 1 到 8 个简单条件。",
            code="strategy_sample_design_v2_population_dto_required",
            fields=(name,),
        )

    compiled: list[dict[str, Any]] = []
    identities: set[str] = set()
    comparison_ops = {"eq", "ne", "gt", "gte", "lt", "lte"}
    null_ops = {"is_null", "is_not_null"}
    for index, condition in enumerate(conditions):
        condition_name = f"{name}.conditions[{index}]"
        if not isinstance(condition, Mapping):
            raise StrategyWorkflowValidationError(
                f"{condition_name} 必须是简单条件对象。",
                code="strategy_sample_design_v2_population_dto_required",
                fields=(name,),
            )
        operator = condition.get("operator")
        expected = (
            {"column", "operator", "value"}
            if operator in comparison_ops
            else {"column", "operator"}
            if operator in null_ops
            else set()
        )
        if not expected or set(condition) != expected:
            raise StrategyWorkflowValidationError(
                f"{condition_name} 只支持简单比较或空值判断。",
                code="strategy_sample_design_v2_population_dto_required",
                fields=(name,),
            )
        column = _column(
            condition["column"],
            name=f"{condition_name}.column",
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and column == context.target_col:
            raise StrategyWorkflowValidationError(
                f"{condition_name} 不能使用目标列定义样本总体。",
                fields=(name,),
            )
        if operator in comparison_ops:
            literal = condition["value"]
            if (
                literal is None
                or not isinstance(literal, str | int | float | bool)
                or isinstance(literal, float)
                and not math.isfinite(literal)
                or isinstance(literal, str)
                and not literal.strip()
            ):
                raise StrategyWorkflowValidationError(
                    f"{condition_name}.value 必须是有限的非空字符串、数字或布尔值。",
                    code="strategy_sample_design_v2_population_dto_required",
                    fields=(name,),
                )
            node = {
                "op": operator,
                "left": {"column": column},
                "right": {"literal": literal},
            }
        else:
            node = {"op": operator, "arg": {"column": column}}
        identity = json.dumps(
            node,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in identities:
            raise StrategyWorkflowValidationError(
                f"{name}.conditions 不能包含重复条件。",
                code="strategy_sample_design_v2_population_dto_required",
                fields=(name,),
            )
        identities.add(identity)
        compiled.append(node)
    if len(compiled) == 1:
        return compiled[0]
    return {
        "op": "and" if value["match"] == "all" else "or",
        "args": compiled,
    }


def _validate_sample_v2_partitioning(
    value: object,
    *,
    context: StrategyWorkflowResolutionContext,
    allow_recursive_ast: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyWorkflowValidationError(
            "partitioning 必须是对象。",
            fields=("partitioning",),
        )
    method = value.get("method")
    if method == "time_ranges":
        return _validate_sample_v2_time_ranges(value, context=context)
    if method != "predicate_ast" or set(value) != {"method", "selectors"}:
        raise StrategyWorkflowValidationError(
            "partitioning 必须是严格 predicate_ast 或 time_ranges 对象。",
            fields=("partitioning",),
        )
    selectors = value["selectors"]
    partition_names = ("development", "validation", "oot")
    if not isinstance(selectors, Mapping) or set(selectors) != set(partition_names):
        raise StrategyWorkflowValidationError(
            "partitioning.selectors 必须完整包含 development、validation、oot。",
            fields=("partitioning.selectors",),
        )
    normalized: dict[str, Any] = {}
    for partition in partition_names:
        if (
            not allow_recursive_ast
            and not is_sample_design_v2_fresh_partition_selector(
                selectors[partition]
            )
        ):
            raise StrategyWorkflowValidationError(
                f"partitioning.selectors.{partition} 只能是单个简单条件，"
                "或由同一种 and/or 连接的单层 2 到 8 个简单条件；"
                "fresh 请求禁止嵌套逻辑、not 和列间比较。",
                fields=(f"partitioning.selectors.{partition}",),
            )
        try:
            canonical = canonicalize_predicate(
                selectors[partition],
                context.allowed_columns,
                max_nodes=256,
                max_depth=12,
            )
        except PredicateAstError as exc:
            raise StrategyWorkflowValidationError(
                f"partitioning.selectors.{partition} 不是严格 predicate AST：{exc}",
                fields=(f"partitioning.selectors.{partition}",),
            ) from exc
        if (
            context.target_col is not None
            and context.target_col in canonical.required_columns
        ):
            raise StrategyWorkflowValidationError(
                "partitioning 不能使用目标列。",
                fields=(f"partitioning.selectors.{partition}",),
            )
        normalized[partition] = canonical.canonical
    return {"method": "predicate_ast", "selectors": normalized}


def is_sample_design_v2_fresh_partition_selector(value: object) -> bool:
    """Return whether a fresh partition uses one leaf or one flat logic row."""

    if _sample_v2_fresh_partition_leaf_shape(value):
        return True
    if not isinstance(value, Mapping) or set(value) != {"op", "args"}:
        return False
    if value.get("op") not in {"and", "or"}:
        return False
    args = value.get("args")
    return (
        isinstance(args, Sequence)
        and not isinstance(args, str | bytes | bytearray)
        and 2 <= len(args) <= 8
        and all(_sample_v2_fresh_partition_leaf_shape(arg) for arg in args)
    )


def _sample_v2_fresh_partition_leaf_shape(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    op = value.get("op")
    if op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        return (
            set(value) == {"op", "left", "right"}
            and isinstance(value.get("left"), Mapping)
            and set(value["left"]) == {"column"}
            and isinstance(value.get("right"), Mapping)
            and set(value["right"]) == {"literal"}
        )
    if op in {"is_null", "is_not_null"}:
        return (
            set(value) == {"op", "arg"}
            and isinstance(value.get("arg"), Mapping)
            and set(value["arg"]) == {"column"}
        )
    return False


def _validate_sample_v2_time_ranges(
    value: Mapping[str, Any],
    *,
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    if set(value) != {"method", "column", "ranges"}:
        raise StrategyWorkflowValidationError(
            "time_ranges 字段不完整。",
            fields=("partitioning",),
        )
    column = _column(
        value["column"],
        name="partitioning.column",
        whitelist=context.allowed_columns,
    )
    if context.target_col is not None and column == context.target_col:
        raise StrategyWorkflowValidationError(
            "partitioning 不能使用目标列。",
            fields=("partitioning.column",),
        )
    ranges = value["ranges"]
    if not isinstance(ranges, Mapping) or set(ranges) != {
        "development",
        "validation",
        "oot",
    }:
        raise StrategyWorkflowValidationError(
            "time_ranges.ranges 必须完整包含三组。",
            fields=("partitioning.ranges",),
        )
    normalized_ranges: dict[str, dict[str, str | None]] = {}
    for name, raw in ranges.items():
        if not isinstance(raw, Mapping) or set(raw) != {"start", "end"}:
            raise StrategyWorkflowValidationError(
                f"partitioning.ranges.{name} 字段无效。"
            )
        bounds = [
            _strict_optional_iso_date(
                raw[field],
                f"partitioning.ranges.{name}.{field}",
            )
            for field in ("start", "end")
        ]
        if bounds == [None, None] or (
            bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]
        ):
            raise StrategyWorkflowValidationError(
                f"partitioning.ranges.{name} 日期范围无效。"
            )
        normalized_ranges[str(name)] = {
            "start": bounds[0],
            "end": bounds[1],
        }
    return {
        "method": "time_ranges",
        "column": column,
        "ranges": normalized_ranges,
    }


def _validate_sample_v2_maturity(value: object) -> dict[str, Any]:
    fields = {"status", "performance_window_days", "cutoff_date", "reason"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise StrategyWorkflowValidationError(
            "maturity 字段必须完整且精确。",
            fields=("maturity",),
        )
    status = value["status"]
    if status not in {
        "confirmed_matured",
        "not_matured",
        "unknown",
        "unavailable",
    }:
        raise StrategyWorkflowValidationError(
            "maturity.status 不受支持。",
            fields=("maturity.status",),
        )
    if status in {"confirmed_matured", "not_matured"}:
        days = value["performance_window_days"]
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise StrategyWorkflowValidationError(
                "maturity.performance_window_days 必须是正整数。"
            )
        cutoff = _strict_iso_date(value["cutoff_date"], "maturity.cutoff_date")
        if status == "confirmed_matured":
            if value["reason"] is not None:
                raise StrategyWorkflowValidationError(
                    "confirmed_matured 的 reason 必须为 null。"
                )
            reason = None
        else:
            reason = _required_text(value["reason"], name="maturity.reason")
        return {
            "status": status,
            "performance_window_days": days,
            "cutoff_date": cutoff,
            "reason": reason,
        }
    if value["performance_window_days"] is not None or value["cutoff_date"] is not None:
        raise StrategyWorkflowValidationError(
            "unknown/unavailable maturity 的天数与截止日必须为 null。"
        )
    return {
        "status": status,
        "performance_window_days": None,
        "cutoff_date": None,
        "reason": _required_text(value["reason"], name="maturity.reason"),
    }


def _validate_sample_v2_performance_window(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"status", "days"}:
        raise StrategyWorkflowValidationError("performance_window 字段必须完整且精确。")
    status = value["status"]
    if status not in {"provided", "unavailable"}:
        raise StrategyWorkflowValidationError("performance_window.status 不受支持。")
    days = value["days"]
    if status == "provided":
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise StrategyWorkflowValidationError(
                "performance_window.days 必须是正整数。"
            )
    elif days is not None:
        raise StrategyWorkflowValidationError(
            "unavailable performance_window.days 必须为 null。"
        )
    return {"status": status, "days": days}


def _validate_sample_v2_observation_window(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"status", "start", "end"}:
        raise StrategyWorkflowValidationError("observation_window 字段必须完整且精确。")
    status = value["status"]
    if status not in {"provided", "unavailable"}:
        raise StrategyWorkflowValidationError("observation_window.status 不受支持。")
    if status == "provided":
        start = _strict_iso_date(value["start"], "observation_window.start")
        end = _strict_iso_date(value["end"], "observation_window.end")
        if start > end:
            raise StrategyWorkflowValidationError(
                "observation_window.start 不能晚于 end。"
            )
    else:
        if value["start"] is not None or value["end"] is not None:
            raise StrategyWorkflowValidationError(
                "unavailable observation_window 边界必须为 null。"
            )
        start = end = None
    return {"status": status, "start": start, "end": end}


def _validate_sample_v2_field_bindings(
    value: object,
    *,
    context: StrategyWorkflowResolutionContext,
) -> dict[str, str | None]:
    fields = {
        "entity_field",
        "time_field",
        "group_field",
        "month_field",
        "weight_field",
        "loan_amount_field",
        "overdue_amount_field",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise StrategyWorkflowValidationError(
            "field_bindings 字段必须完整且精确。",
            fields=("field_bindings",),
        )
    normalized: dict[str, str | None] = {}
    for field in sorted(fields):
        raw = value[field]
        if raw is None:
            normalized[field] = None
            continue
        column = _column(
            raw,
            name=f"field_bindings.{field}",
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and column == context.target_col:
            raise StrategyWorkflowValidationError(
                f"field_bindings.{field} 不能使用目标列。"
            )
        normalized[field] = column
    return normalized


def _validate_sample_v2_historical_score(
    value: object,
    *,
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "column",
        "direction",
        "reason",
    }:
        raise StrategyWorkflowValidationError("historical_score 字段必须完整且精确。")
    status = value["status"]
    if status not in {"available", "unavailable", "not_applicable"}:
        raise StrategyWorkflowValidationError("historical_score.status 不受支持。")
    if status == "available":
        column = _column(
            value["column"],
            name="historical_score.column",
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and column == context.target_col:
            raise StrategyWorkflowValidationError(
                "historical_score.column 不能使用目标列。"
            )
        direction = value["direction"]
        if direction not in {"higher_is_riskier", "lower_is_riskier"}:
            raise StrategyWorkflowValidationError(
                "historical_score.direction 不受支持。"
            )
        if value["reason"] is not None:
            raise StrategyWorkflowValidationError(
                "available historical_score.reason 必须为 null。"
            )
        reason = None
    else:
        if value["column"] is not None or value["direction"] is not None:
            raise StrategyWorkflowValidationError(
                "非 available historical_score 的 column/direction 必须为 null。"
            )
        column = direction = None
        reason = _required_text(
            value["reason"],
            name="historical_score.reason",
        )
    return {
        "status": status,
        "column": column,
        "direction": direction,
        "reason": reason,
    }


def project_context_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔策略项目上下文 Workflow〕",
        f"现状截止日：{inputs['as_of']}",
        "本步骤只收集并绑定项目现状、历史策略和缺失信息，不计算或复制业务指标",
    ]
    if inputs.get("scope"):
        details.append(f"分析范围：{inputs['scope']}")
    if inputs["business_context"]:
        details.append("用户提供字段：" + "、".join(inputs["business_context"].keys()))
    if inputs["explicit_unavailable"]:
        details.append("明确暂缺字段：" + "、".join(inputs["explicit_unavailable"]))
    if inputs["external_report_filenames"]:
        details.append(
            "外部报告仅作为不透明证据复制并绑定："
            + "、".join(inputs["external_report_filenames"])
        )
    details.extend(
        [
            "当前样本、Pool、回测、监控与内部历史由平台自动发现并校验",
            "revision/CAS、消息哈希、artifact id、来源引用和所有指标由平台拥有",
        ]
    )
    return _confirmation_text(details)


def sample_design_v2_confirmation(inputs: Mapping[str, Any]) -> str:
    performance = inputs["performance_window"]
    observation = inputs["observation_window"]
    maturity = inputs["maturity"]
    historical_score = inputs["historical_score"]
    details = [
        "已识别为〔V2 双总体策略样本设计 Workflow〕",
        f"总体关系：{inputs['relationship']}",
        (
            "总体纳排：两者均无纳排"
            if all(
                population[field] is None
                for population in (
                    inputs["approval_population"],
                    inputs["risk_population"],
                )
                for field in ("inclusion", "exclusion")
            )
            else "总体纳排：已按用户提供的受限条件固化"
        ),
        (
            f"表现窗：{performance['days']} 天"
            if performance["status"] == "provided"
            else "表现窗：unavailable"
        ),
        (
            f"观察窗：{observation['start']} 至 {observation['end']}"
            if observation["status"] == "provided"
            else "观察窗：unavailable"
        ),
        f"成熟度：{maturity['status']}；坏样本值：{inputs['target_bad_value']}",
        (
            f"历史分：{historical_score['status']}；"
            f"字段：{historical_score['column'] or 'null'}；"
            f"方向：{historical_score['direction'] or 'null'}"
        ),
        "approval/risk 总体与 development/validation/OOT 三分区将分别固化",
        "平台将按请求语义选择无损 compatibility 链或原生 V2 执行；"
        "不会删减总体纳排、时间切分或复杂 selector",
        "legacy ref、scope、policy、数据/workspace 身份和所有 id/hash 由平台绑定",
    ]
    return _confirmation_text(details)


def model_evidence_v2_confirmation(_inputs: Mapping[str, Any]) -> str:
    return _confirmation_text(
        [
            "已识别为〔Strategy ModelEvidence V2 Workflow〕",
            "workflow_inputs 为空；平台从当前 task 绑定已认证 SampleDesign V2 与单变量候选",
            "当前只归集 univariate evidence，不训练模型、不比较模型、不生成月度/OOT 模型证据",
            "不报告、不采纳、不部署",
        ]
    )


def dsl_delivery_confirmation(inputs: Mapping[str, Any]) -> str:
    return _confirmation_text(
        [
            "已识别为〔离线 Strategy DSL 交付 Workflow〕",
            (
                f"策略 ID：{inputs['strategy_id']}"
                if "strategy_id" in inputs
                else "策略 ID：由平台仅在当前任务恰有一个可交付策略时唯一绑定"
            ),
            "平台将原子绑定策略类型、version/spec hash 与当前活动 task-owned "
            "dataset id/content hash",
            "将生成 Python、DuckDB SQL、canonical JSON 和等价证据四个"
            "content-hash 固定下载文件",
            "等价证据会明确显示校验样本数、源数据行数及是否为最多 4096 行的"
            "受治理有界样本，不会把抽样校验称为全量校验",
            "本步骤只生成离线代码；不会应用、写回、采纳、晋级或部署策略",
        ]
    )


def report_bundle_v2_confirmation(inputs: Mapping[str, Any]) -> str:
    return _confirmation_text(
        [
            "已识别为〔StrategyReportBundle V2 Workflow〕",
            f"报告标题：{inputs['title']}",
            f"报告状态：{inputs['status']}",
            "平台将绑定当前认证 ProjectContext、最新精确 SampleDesign V2、"
            "当前非空 approval/reject Pool 及其最新精确 PoolImpact",
            "只有与同一 SampleDesign/模型链完全兼容的最新认证模型证据才会纳入；"
            "不兼容证据保持缺省，损坏的最新证据不会回退到旧版本",
            "策略身份、report head revision/CAS、generated_at、artifact id/hash "
            "和所有指标均由平台拥有",
            "本步骤只生成 JSON、Markdown、XLSX 三种受治理报告；"
            "不会创建策略，也不代表采纳或部署",
        ]
    )


def legacy_sample_design_replay_confirmation(inputs: Mapping[str, Any]) -> str:
    """Echo persisted V1 sample semantics without re-exposing fresh intake."""

    performance = (
        f"已提供 {inputs['performance_window_days']} 天"
        if inputs["performance_window_status"] == "provided"
        else "unavailable"
    )
    observation = (
        f"{inputs['observation_start']} 至 {inputs['observation_end']}"
        if inputs["observation_window_status"] == "provided"
        else "unavailable"
    )
    maturity_labels = {
        "confirmed_matured": "已确认成熟",
        "not_matured": "尚未成熟",
        "unknown": "未知",
    }
    details = [
        "已识别为〔策略样本设计 Workflow〕",
        f"表现窗：{performance}",
        f"观察窗：{observation}",
        f"成熟度：{maturity_labels[inputs['maturity_status']]}",
        f"坏样本值：{inputs['target_bad_value']}；"
        f"好样本值：{1 - inputs['target_bad_value']}",
    ]
    if "split_col" in inputs:
        validation_values = (
            "、".join(str(value) for value in inputs["validation_values"])
            if inputs["validation_values"]
            else "unavailable"
        )
        oot_values = (
            "、".join(str(value) for value in inputs["oot_values"])
            if inputs["oot_values"]
            else "unavailable"
        )
        details.append(
            f"切分列 {inputs['split_col']}；开发样本值 "
            + "、".join(str(value) for value in inputs["development_values"])
            + f"；验证样本值 {validation_values}；OOT 样本值 {oot_values}"
        )
    else:
        details.append("未声明开发/验证/OOT 切分，平台只冻结 overall 样本边界")
    for field, label in (
        ("month_col", "月份列"),
        ("weight_col", "权重列"),
        ("loan_amount_col", "放款金额列"),
        ("overdue_amount_col", "逾期金额列"),
    ):
        if field in inputs:
            details.append(f"{label}：{inputs[field]}")
    if inputs.get("drop_nan_labels") is True:
        details.append("标签缺失处理：保留总体样本行，仅从好坏/风险分母排除空标签")
    elif inputs.get("drop_nan_labels") is False:
        details.append("标签缺失处理：保留 NaN 标签行，遇到缺失时不会自动排除")
    if (
        inputs["performance_window_status"] == "unavailable"
        or inputs["observation_window_status"] == "unavailable"
        or inputs["maturity_status"] != "confirmed_matured"
    ):
        details.append(
            "本次仅按 exploration-only 降级固化；不能据此声称样本已成熟、"
            "完成独立验证或晋级"
        )
    details.extend(
        [
            "数据集/hash、workspace、语义映射和目标列由平台绑定",
            "本步骤只冻结活动样本/派生数据边界并计算样本证据；"
            "不做自由文本过滤或派生，不建模、不建树、不入池、不采纳、不部署、"
            "不生成最终报告",
        ]
    )
    return _confirmation_text(details)


_PROJECT_PLATFORM_FIELDS = frozenset(
    {
        "expected_revision",
        "expected_revision_id",
        "expected_state_hash",
        "user_message_ref",
    }
)
_SAMPLE_IDENTITY_PLATFORM_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
    }
)
_SAMPLE_V2_BASE_PLATFORM_FIELDS = frozenset(
    {*_SAMPLE_IDENTITY_PLATFORM_FIELDS, "scope", "policy"}
)
_SAMPLE_V2_COMPATIBILITY_PLATFORM_FIELDS = frozenset(
    {
        "compatibility_performance_window_status",
        "compatibility_performance_window_days",
        "compatibility_observation_window_status",
        "compatibility_observation_start",
        "compatibility_observation_end",
        "compatibility_maturity_status",
        "compatibility_split_col",
        "compatibility_development_values",
        "compatibility_validation_values",
        "compatibility_oot_values",
        "compatibility_month_col",
        "compatibility_weight_col",
        "compatibility_loan_amount_col",
        "compatibility_overdue_amount_col",
    }
)
_MODEL_EVIDENCE_PLATFORM_FIELDS = frozenset(
    {"sample_design_ref", "univariate_sources", "expected_registry_token"}
)
_DSL_DELIVERY_PLATFORM_FIELDS = frozenset(
    {
        "strategy_ref",
        "dataset_ref",
        "workspace_ref",
        "maximum_equivalence_rows",
    }
)
_REPORT_BUNDLE_PLATFORM_FIELDS = frozenset(
    {
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
        "pool_validation_refs",
        "candidate_stability_ref",
        "pool_stability_ref",
        "voting_candidate_search_ref",
        "cross_candidate_search_ref",
        "cross_rule_search_ref",
        "pool_impact_ref",
        "impact_cube_ref",
        "report_revision",
        "previous_report_id",
        "previous_report_content_hash",
        "generated_at",
        "strategy_identity",
        "model_evidence_ref",
        "training_evidence_ref",
        "score_evidence_ref",
    }
)


def prepare_project_context(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    canonical_slots = {
        "as_of": inputs["as_of"],
        "scope": inputs.get("scope"),
        "business_context": deep_thaw(inputs["business_context"]),
        "explicit_unavailable": deep_thaw(inputs["explicit_unavailable"]),
        "external_report_filenames": deep_thaw(inputs["external_report_filenames"]),
    }
    evidence = _platform_evidence(
        "strategy_project_context",
        inputs,
        context,
        canonical_slots=canonical_slots,
        expected_fields=_PROJECT_PLATFORM_FIELDS,
    )
    return _prepared_plan(
        "strategy_project_context",
        "strategy_project_context",
        {**evidence, **canonical_slots},
    )


def prepare_sample_design_v2(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    template_id = select_sample_design_v2_template(inputs)
    canonical_slots = {
        key: deep_thaw(value)
        for key, value in inputs.items()
        if key != "drop_nan_labels"
    }
    # This flag is resolved by the confirmation/NaN gate before preparation;
    # using the request copy here would bypass that platform-owned decision.
    canonical_slots["drop_nan_labels"] = bool(context.drop_nan_labels)
    expected_fields = _SAMPLE_V2_BASE_PLATFORM_FIELDS
    if template_id == "strategy_sample_design_v2":
        expected_fields = frozenset(
            {
                *expected_fields,
                *_SAMPLE_V2_COMPATIBILITY_PLATFORM_FIELDS,
            }
        )
    evidence = _platform_evidence(
        "strategy_sample_design_v2",
        inputs,
        context,
        canonical_slots=canonical_slots,
        expected_fields=expected_fields,
    )
    return _prepared_plan(
        "strategy_sample_design_v2",
        template_id,
        {**evidence, **canonical_slots},
    )


def prepare_model_evidence_v2(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    evidence = _platform_evidence(
        "strategy_model_evidence_v2",
        inputs,
        context,
        canonical_slots={},
        expected_fields=_MODEL_EVIDENCE_PLATFORM_FIELDS,
    )
    return _prepared_plan(
        "strategy_model_evidence_v2",
        "strategy_model_evidence_v2",
        evidence,
    )


def prepare_dsl_delivery(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    # strategy_id is a selector only.  The exact strategy identity is supplied
    # by the platform callback as strategy_ref, matching the legacy runtime.
    evidence = _platform_evidence(
        "strategy_dsl_delivery",
        inputs,
        context,
        canonical_slots=inputs,
        expected_fields=_DSL_DELIVERY_PLATFORM_FIELDS,
    )
    return _prepared_plan(
        "strategy_dsl_delivery",
        "strategy_dsl_delivery",
        evidence,
    )


def prepare_report_bundle_v2(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    canonical_slots = deep_thaw(inputs)
    evidence = _platform_evidence(
        "strategy_report_bundle_v2",
        inputs,
        context,
        canonical_slots=canonical_slots,
        expected_fields=_REPORT_BUNDLE_PLATFORM_FIELDS,
    )
    return _prepared_plan(
        "strategy_report_bundle_v2",
        "strategy_report_bundle_v2",
        {**evidence, **canonical_slots},
    )


def prepare_legacy_sample_design_replay(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    """Compatibility preparation for already-persisted V1 sample requests."""

    canonical_slots = {
        key: deep_thaw(value)
        for key, value in inputs.items()
        if key != "drop_nan_labels"
    }
    canonical_slots["drop_nan_labels"] = bool(context.drop_nan_labels)
    evidence = _platform_evidence(
        "strategy_sample_design",
        inputs,
        context,
        canonical_slots=canonical_slots,
        expected_fields=_SAMPLE_IDENTITY_PLATFORM_FIELDS,
    )
    return _prepared_plan(
        "strategy_sample_design",
        "strategy_sample_design",
        {**evidence, **canonical_slots},
    )


def _platform_evidence(
    workflow_id: str,
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
    *,
    canonical_slots: Mapping[str, Any],
    expected_fields: frozenset[str],
) -> dict[str, Any]:
    binder = context.bind_workflow_evidence
    if binder is None:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 缺少平台认证的证据绑定。",
            code="strategy_workflow_evidence_binding_required",
        )
    frozen_inputs = deep_freeze(deep_thaw(inputs))
    evidence = binder(workflow_id, frozen_inputs)
    if not isinstance(evidence, Mapping):
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据绑定必须返回对象。",
            code="strategy_workflow_evidence_invalid",
        )
    if any(not isinstance(key, str) for key in evidence):
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据绑定字段名必须是文本。",
            code="strategy_workflow_evidence_invalid",
        )
    conflicts = tuple(sorted(set(evidence) & set(canonical_slots)))
    if conflicts:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据不能覆盖 canonical slots："
            + "、".join(conflicts)
            + "。",
            code="strategy_workflow_evidence_conflict",
            fields=conflicts,
        )
    actual_fields = frozenset(evidence)
    if actual_fields != expected_fields:
        fields = tuple(sorted(actual_fields ^ expected_fields))
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据绑定字段不完整或超出契约："
            + "、".join(fields)
            + "。",
            code="strategy_workflow_evidence_invalid",
            fields=fields,
        )
    copied = deep_thaw(deep_freeze(evidence))
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guarded above
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据绑定无法复制。",
            code="strategy_workflow_evidence_invalid",
        )
    return copied


def _prepared_plan(
    workflow_id: str,
    template_id: str,
    slots: Mapping[str, Any],
) -> PreparedStrategyPlan:
    return PreparedStrategyPlan(
        workflow_id=workflow_id,
        template_id=template_id,
        slots=deep_freeze(slots),
        success_criteria=(),
    )


def select_sample_design_v2_template(inputs: Mapping[str, Any]) -> str:
    """Select the lossless compatibility template or the native V2 template."""

    native = "strategy_sample_design_v2_native"
    if inputs.get("relationship") != "nested_same_cohort":
        return native
    empty_population = {"inclusion": None, "exclusion": None}
    if (
        inputs.get("approval_population") != empty_population
        or inputs.get("risk_population") != empty_population
    ):
        return native
    partitioning = inputs.get("partitioning")
    if (
        not isinstance(partitioning, Mapping)
        or set(partitioning) != {"method", "selectors"}
        or partitioning.get("method") != "predicate_ast"
        or not isinstance(partitioning.get("selectors"), Mapping)
    ):
        return native
    selectors = partitioning["selectors"]
    if set(selectors) != {"development", "validation", "oot"}:
        return native
    columns: list[str] = []
    values: list[object] = []
    for partition in ("development", "validation", "oot"):
        predicate = selectors[partition]
        if (
            not isinstance(predicate, Mapping)
            or set(predicate) != {"op", "left", "right"}
            or predicate.get("op") != "eq"
            or not isinstance(predicate.get("left"), Mapping)
            or set(predicate["left"]) != {"column"}
            or not isinstance(predicate.get("right"), Mapping)
            or set(predicate["right"]) != {"literal"}
        ):
            return native
        column = predicate["left"]["column"]
        literal = predicate["right"]["literal"]
        if (
            not isinstance(column, str)
            or not column
            or literal is None
            or isinstance(literal, Mapping | Sequence)
            and not isinstance(literal, str)
        ):
            return native
        columns.append(column)
        values.append(literal)
    if len(set(columns)) != 1:
        return native
    identities = {
        json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values
    }
    if len(identities) != 3:
        return native
    field_bindings = inputs.get("field_bindings")
    if not isinstance(field_bindings, Mapping):
        return native
    projected_fields = [
        field_bindings.get(name)
        for name in (
            "month_field",
            "weight_field",
            "loan_amount_field",
            "overdue_amount_field",
        )
        if field_bindings.get(name) is not None
    ]
    if columns[0] in projected_fields or len(projected_fields) != len(
        set(projected_fields)
    ):
        return native
    return "strategy_sample_design_v2"


def _confirmation_text(details: Sequence[str]) -> str:
    return "；".join(
        [
            *details,
            "请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。",
        ]
    )


def _project_context_text_list(
    value: object,
    *,
    name: str,
    maximum: int,
    field_paths: bool,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise StrategyWorkflowValidationError(f"{name} 必须是数组。")
    if len(value) > maximum:
        raise StrategyWorkflowValidationError(f"{name} 最多包含 {maximum} 项。")
    normalized: list[str] = []
    for raw_item in value:
        item = _required_text(raw_item, name=name)
        if len(item) > 512:
            raise StrategyWorkflowValidationError(f"{name} 每项最多 512 个字符。")
        if field_paths:
            if len(item) > 256 or not _PROJECT_CONTEXT_FIELD_PATH_RE.fullmatch(item):
                raise StrategyWorkflowValidationError(
                    f"{name} 包含无效字段路径：{item!r}。"
                )
        else:
            normalized_path = item.replace("\\", "/")
            parts = normalized_path.split("/")
            if (
                normalized_path.startswith("/")
                or any(part in {"", ".", ".."} for part in parts)
                or "\x00" in item
            ):
                raise StrategyWorkflowValidationError(
                    f"{name} 包含不安全的相对文件名。"
                )
            item = normalized_path
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise StrategyWorkflowValidationError(f"{name} 不能包含重复项。")
    return normalized


def _sample_design_value_sequence(
    value: object,
    *,
    name: str,
    minimum_items: int,
) -> list[object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or not minimum_items <= len(value) <= 100
    ):
        raise StrategyWorkflowValidationError(
            f"{name} 必须是包含 {minimum_items} 到 100 个标量值的数组。"
        )
    normalized: list[object] = []
    identities: set[tuple[str, object]] = set()
    for item in value:
        if item is None or not isinstance(item, str | int | float | bool):
            raise StrategyWorkflowValidationError(
                f"{name} 只能包含文本、布尔值或有限数字。"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise StrategyWorkflowValidationError(
                f"{name} 只能包含文本、布尔值或有限数字。"
            )
        if (
            isinstance(item, int)
            and not isinstance(item, bool)
            and abs(item) > 2**53 - 1
        ):
            raise StrategyWorkflowValidationError(
                f"{name} 中的整数超出精确 JSON 范围。"
            )
        identity = _sample_design_value_identity(item)
        if identity in identities:
            raise StrategyWorkflowValidationError(f"{name} 不能包含重复值。")
        identities.add(identity)
        normalized.append(item)
    return normalized


def _sample_design_value_identity(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        return ("number", Decimal(str(value)).normalize())
    return ("string", value)


def _strict_iso_date(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise StrategyWorkflowValidationError(f"{name} 必须是 YYYY-MM-DD ISO 日期。")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise StrategyWorkflowValidationError(
            f"{name} 必须是 YYYY-MM-DD ISO 日期。"
        ) from exc
    if parsed.isoformat() != value:
        raise StrategyWorkflowValidationError(f"{name} 必须是 YYYY-MM-DD ISO 日期。")
    return value


def _strict_optional_iso_date(value: object, name: str) -> str | None:
    return None if value is None else _strict_iso_date(value, name)


def _reject_fields(
    inputs: Mapping[str, Any],
    allowed: set[str],
    *,
    workflow: str,
) -> None:
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyWorkflowValidationError("workflow_inputs 的字段名必须是文本。")
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。"
        )


def _column(
    value: object,
    *,
    name: str,
    whitelist: Sequence[str],
) -> str:
    column = _required_text(value, name=name)
    if column not in whitelist:
        raise StrategyWorkflowValidationError(
            f"{name} 使用了数据集中不存在的列「{column}」。"
        )
    return column


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyWorkflowValidationError(f"{name} 必须是非空文本。")
    return value.strip()


_NO_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=False,
    target=False,
    complete_labels=False,
)
_SAMPLE_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=True,
    target=True,
    complete_labels=True,
)

STRATEGY_PROJECT_CONTEXT_SPEC = StrategyWorkflowSpec(
    workflow_id="strategy_project_context",
    fresh=True,
    replayable=True,
    manual=True,
    requirements=_NO_REQUIREMENTS,
    template_ids=("strategy_project_context",),
    validator=validate_project_context_inputs,
    confirmation=project_context_confirmation,
    preparer=prepare_project_context,
)
STRATEGY_SAMPLE_DESIGN_V2_SPEC = StrategyWorkflowSpec(
    workflow_id="strategy_sample_design_v2",
    fresh=True,
    replayable=True,
    manual=True,
    requirements=_SAMPLE_REQUIREMENTS,
    template_ids=("strategy_sample_design_v2", "strategy_sample_design_v2_native"),
    validator=validate_sample_design_v2_inputs,
    confirmation=sample_design_v2_confirmation,
    preparer=prepare_sample_design_v2,
)
STRATEGY_MODEL_EVIDENCE_V2_SPEC = StrategyWorkflowSpec(
    workflow_id="strategy_model_evidence_v2",
    fresh=True,
    replayable=True,
    manual=False,
    requirements=_NO_REQUIREMENTS,
    template_ids=("strategy_model_evidence_v2",),
    validator=validate_model_evidence_v2_inputs,
    confirmation=model_evidence_v2_confirmation,
    preparer=prepare_model_evidence_v2,
)
STRATEGY_DSL_DELIVERY_SPEC = StrategyWorkflowSpec(
    workflow_id="strategy_dsl_delivery",
    fresh=True,
    replayable=True,
    manual=True,
    requirements=StrategyWorkflowRequirements(
        dataset=True,
        target=False,
        complete_labels=False,
    ),
    template_ids=("strategy_dsl_delivery",),
    validator=validate_dsl_delivery_inputs,
    confirmation=dsl_delivery_confirmation,
    preparer=prepare_dsl_delivery,
)
STRATEGY_REPORT_BUNDLE_V2_SPEC = StrategyWorkflowSpec(
    workflow_id="strategy_report_bundle_v2",
    fresh=True,
    replayable=True,
    manual=True,
    requirements=_NO_REQUIREMENTS,
    template_ids=("strategy_report_bundle_v2",),
    validator=validate_report_bundle_v2_inputs,
    confirmation=report_bundle_v2_confirmation,
    preparer=prepare_report_bundle_v2,
)
LEGACY_SAMPLE_DESIGN_REPLAY_SPEC = StrategyWorkflowSpec(
    workflow_id="strategy_sample_design",
    fresh=False,
    replayable=True,
    manual=False,
    requirements=_SAMPLE_REQUIREMENTS,
    template_ids=("strategy_sample_design",),
    validator=validate_legacy_sample_design_replay_inputs,
    confirmation=legacy_sample_design_replay_confirmation,
    preparer=prepare_legacy_sample_design_replay,
)

FOUNDATION_DELIVERY_SPECS = (
    STRATEGY_PROJECT_CONTEXT_SPEC,
    STRATEGY_SAMPLE_DESIGN_V2_SPEC,
    STRATEGY_MODEL_EVIDENCE_V2_SPEC,
    STRATEGY_DSL_DELIVERY_SPEC,
    STRATEGY_REPORT_BUNDLE_V2_SPEC,
    LEGACY_SAMPLE_DESIGN_REPLAY_SPEC,
)


__all__ = [
    "FOUNDATION_DELIVERY_SPECS",
    "LEGACY_SAMPLE_DESIGN_REPLAY_SPEC",
    "STRATEGY_DSL_DELIVERY_SPEC",
    "STRATEGY_MODEL_EVIDENCE_V2_SPEC",
    "STRATEGY_PROJECT_CONTEXT_SPEC",
    "STRATEGY_REPORT_BUNDLE_V2_SPEC",
    "STRATEGY_SAMPLE_DESIGN_V2_SPEC",
    "dsl_delivery_confirmation",
    "legacy_sample_design_replay_confirmation",
    "model_evidence_v2_confirmation",
    "is_sample_design_v2_fresh_partition_selector",
    "prepare_dsl_delivery",
    "prepare_legacy_sample_design_replay",
    "prepare_model_evidence_v2",
    "prepare_project_context",
    "prepare_report_bundle_v2",
    "prepare_sample_design_v2",
    "project_context_confirmation",
    "report_bundle_v2_confirmation",
    "sample_design_v2_confirmation",
    "select_sample_design_v2_template",
    "validate_dsl_delivery_inputs",
    "validate_legacy_sample_design_replay_inputs",
    "validate_model_evidence_v2_inputs",
    "validate_project_context_inputs",
    "validate_report_bundle_v2_inputs",
    "validate_sample_design_v2_inputs",
    "validate_sample_design_v2_replay_inputs",
]

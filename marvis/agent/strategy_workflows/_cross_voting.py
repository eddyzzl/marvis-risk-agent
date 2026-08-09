from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any
import unicodedata

from ._univariate_scorecard import (
    UNIVARIATE_REFINEMENT_METHODS,
    validate_univariate_analysis_inputs,
)
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


VOTING_CANDIDATE_SEARCH_WORKFLOW_ID = "voting_candidate_search"
VOTING_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID = (
    "voting_candidate_build_from_search"
)
VOTING_CANDIDATE_BUILD_WORKFLOW_ID = "voting_candidate_build"
CROSS_MATRIX_CANDIDATE_SEARCH_WORKFLOW_ID = "cross_matrix_candidate_search"
CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID = (
    "cross_matrix_candidate_build_from_search"
)
CROSS_RULE_SEARCH_WORKFLOW_ID = "cross_rule_search"
CROSS_RULE_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID = (
    "cross_rule_candidate_build_from_search"
)
CROSS_MATRIX_ANALYSIS_WORKFLOW_ID = "cross_matrix_analysis"
CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID = "cross_matrix_cell_selection"
STRATEGY_TYPES = (
    "approval",
    "reject",
    "limit",
    "pricing",
    "segmentation",
)

_VOTING_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_VOTING_SEARCH_ID_RE = re.compile(r"^voting-search-[0-9a-f]{32}$")
_VOTING_COMBO_ID_RE = re.compile(r"^voting-combo-[0-9a-f]{32}$")
_CROSS_SEARCH_ID_RE = re.compile(r"^cross-search-[0-9a-f]{32}$")
_CROSS_PAIR_ID_RE = re.compile(r"^cross-pair-[0-9a-f]{32}$")
_CROSS_RULE_SEARCH_ID_RE = re.compile(r"^cross-rule-search-[0-9a-f]{32}$")
_CROSS_RULE_ID_RE = re.compile(r"^cross-rule-[0-9a-f]{32}$")
_CANDIDATE_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_CROSS_MATRIX_CELL_ID_RE = re.compile(r"^cross-cell-[0-9a-f]{32}$")
_POOL_ENTRY_ID_RE = re.compile(r"^pool-entry-[0-9a-f]{32}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_VOTING_SEARCH_METRICS = frozenset(
    {
        "hit_count",
        "hit_share",
        "good_count",
        "bad_count",
        "bad_rate",
        "lift",
        "bad_capture_rate",
        "weighted_hit_total",
        "weighted_hit_share",
        "weighted_good_total",
        "weighted_bad_total",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount",
        "hit_amount_share",
        "good_amount",
        "bad_amount",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }
)
_VOTING_SEARCH_RATE_METRICS = frozenset(
    {
        "hit_share",
        "bad_rate",
        "bad_capture_rate",
        "weighted_hit_share",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount_share",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }
)
_VOTING_SEARCH_REQUIRED_MINIMUM_SHARE = {
    "bad_rate": "hit_share",
    "weighted_bad_rate": "weighted_hit_share",
    "bad_amount_rate": "hit_amount_share",
}
_NO_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=False,
    target=False,
    complete_labels=False,
)
_DATA_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=True,
    target=True,
    complete_labels=True,
)


def validate_voting_candidate_search_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate human search controls and inject documented safe defaults."""

    workflow = VOTING_CANDIDATE_SEARCH_WORKFLOW_ID
    required = {"strategy_type", "member_count", "n", "objective"}
    allowed = {
        *required,
        "constraints",
        "include_rule_ids",
        "exclude_rule_ids",
        "max_combinations",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted(required - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    strategy_type = _required_text(
        inputs["strategy_type"], name=f"{workflow} strategy_type"
    )
    if strategy_type not in STRATEGY_TYPES:
        raise StrategyWorkflowValidationError(
            f"{workflow} strategy_type 只能是："
            + "、".join(STRATEGY_TYPES)
            + "。"
        )
    member_count = inputs["member_count"]
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or not 2 <= member_count <= 50
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} member_count 必须是 2 到 50 的整数。"
        )
    n = inputs["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= member_count:
        raise StrategyWorkflowValidationError(
            f"{workflow} n 必须是 1 到 member_count 的整数。"
        )

    objective_raw = inputs["objective"]
    if not isinstance(objective_raw, Mapping):
        raise StrategyWorkflowValidationError(f"{workflow} objective 必须是对象。")
    if set(objective_raw) != {"metric", "direction"}:
        raise StrategyWorkflowValidationError(
            f"{workflow} objective 必须且只能包含 metric、direction。"
        )
    objective_metric = _required_text(
        objective_raw["metric"], name=f"{workflow} objective.metric"
    )
    if objective_metric not in _VOTING_SEARCH_METRICS:
        raise StrategyWorkflowValidationError(
            f"{workflow} objective.metric 不受支持。"
        )
    objective_direction = _required_text(
        objective_raw["direction"], name=f"{workflow} objective.direction"
    )
    if objective_direction not in {"maximize", "minimize"}:
        raise StrategyWorkflowValidationError(
            f"{workflow} objective.direction 只能是 maximize 或 minimize。"
        )
    objective = {
        "metric": objective_metric,
        "direction": objective_direction,
    }

    constraints_raw = inputs.get("constraints", [])
    if (
        not isinstance(constraints_raw, Sequence)
        or isinstance(constraints_raw, str | bytes | bytearray)
        or len(constraints_raw) > 32
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} constraints 必须是最多 32 项的数组。"
        )
    constraints: list[dict[str, Any]] = []
    constraint_identities: set[tuple[str, str]] = set()
    for index, value in enumerate(constraints_raw):
        if not isinstance(value, Mapping) or set(value) != {
            "metric",
            "operator",
            "value",
        }:
            raise StrategyWorkflowValidationError(
                f"{workflow} constraints[{index}] 必须且只能包含 "
                "metric、operator、value。"
            )
        metric = _required_text(
            value["metric"], name=f"{workflow} constraints[{index}].metric"
        )
        if metric not in _VOTING_SEARCH_METRICS:
            raise StrategyWorkflowValidationError(
                f"{workflow} constraints[{index}].metric 不受支持。"
            )
        operator = _required_text(
            value["operator"],
            name=f"{workflow} constraints[{index}].operator",
        )
        if operator not in {"gte", "lte"}:
            raise StrategyWorkflowValidationError(
                f"{workflow} constraints[{index}].operator 只能是 gte 或 lte。"
            )
        number = _bounded_number(
            value["value"],
            name=f"{workflow} constraints[{index}].value",
            maximum=1.0 if metric in _VOTING_SEARCH_RATE_METRICS else None,
        )
        identity = (metric, operator)
        if identity in constraint_identities:
            raise StrategyWorkflowValidationError(
                f"{workflow} constraints 不能重复 metric/operator。"
            )
        constraint_identities.add(identity)
        constraints.append(
            {"metric": metric, "operator": operator, "value": number}
        )
    constraints.sort(
        key=lambda item: (item["metric"], item["operator"], item["value"])
    )

    include_rule_ids = _voting_search_rule_id_array(
        inputs.get("include_rule_ids", []),
        name=f"{workflow} include_rule_ids",
    )
    exclude_rule_ids = _voting_search_rule_id_array(
        inputs.get("exclude_rule_ids", []),
        name=f"{workflow} exclude_rule_ids",
    )
    if set(include_rule_ids) & set(exclude_rule_ids):
        raise StrategyWorkflowValidationError(
            f"{workflow} include_rule_ids 与 exclude_rule_ids 不能重叠。"
        )
    if len(include_rule_ids) > member_count:
        raise StrategyWorkflowValidationError(
            f"{workflow} include_rule_ids 数量不能超过 member_count。"
        )

    max_combinations = inputs.get("max_combinations", 10_000)
    if (
        isinstance(max_combinations, bool)
        or not isinstance(max_combinations, int)
        or not 1 <= max_combinations <= 10_000
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} max_combinations 必须是 1 到 10000 的整数。"
        )

    required_share = _VOTING_SEARCH_REQUIRED_MINIMUM_SHARE.get(objective_metric)
    if objective_direction == "minimize" and required_share is not None:
        has_positive_share = any(
            item["metric"] == required_share
            and item["operator"] == "gte"
            and item["value"] > 0
            for item in constraints
        )
        if not has_positive_share:
            raise StrategyWorkflowValidationError(
                f"最小化 {objective_metric} 必须提供正数 {required_share} gte "
                "约束，绝对命中量不能替代占比下限。",
                code="voting_search_minimum_share_required",
                fields=("constraints", required_share),
            )

    return {
        "strategy_type": strategy_type,
        "member_count": member_count,
        "n": n,
        "objective": objective,
        "constraints": constraints,
        "include_rule_ids": include_rule_ids,
        "exclude_rule_ids": exclude_rule_ids,
        "max_combinations": max_combinations,
    }


def validate_voting_candidate_build_from_search_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate exact authenticated search pointers and optional Pool type."""

    workflow = VOTING_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID
    allowed = {"search_id", "combo_id", "strategy_type"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"search_id", "combo_id"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    search_id = _required_text(
        inputs["search_id"], name=f"{workflow} search_id"
    )
    if _VOTING_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} search_id 必须是完整的 voting-search ID。"
        )
    combo_id = _required_text(inputs["combo_id"], name=f"{workflow} combo_id")
    if _VOTING_COMBO_ID_RE.fullmatch(combo_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} combo_id 必须是完整的 voting-combo ID。"
        )
    normalized: dict[str, Any] = {
        "search_id": search_id,
        "combo_id": combo_id,
    }
    if "strategy_type" in inputs:
        strategy_type = _required_text(
            inputs["strategy_type"], name=f"{workflow} strategy_type"
        )
        if strategy_type not in STRATEGY_TYPES:
            raise StrategyWorkflowValidationError(
                f"{workflow} strategy_type 只能是："
                + "、".join(STRATEGY_TYPES)
                + "。"
            )
        normalized["strategy_type"] = strategy_type
    return normalized


def validate_voting_candidate_build_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate only the explicit human controls for one n-of-k candidate."""

    workflow = VOTING_CANDIDATE_BUILD_WORKFLOW_ID
    allowed = {"strategy_type", "rule_ids", "n"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    strategy_type = _required_text(
        inputs["strategy_type"], name=f"{workflow} strategy_type"
    )
    if strategy_type not in STRATEGY_TYPES:
        raise StrategyWorkflowValidationError(
            f"{workflow} strategy_type 只能是："
            + "、".join(STRATEGY_TYPES)
            + "。"
        )
    raw_rule_ids = inputs["rule_ids"]
    if (
        not isinstance(raw_rule_ids, Sequence)
        or isinstance(raw_rule_ids, str | bytes | bytearray)
        or not 2 <= len(raw_rule_ids) <= 50
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} rule_ids 必须是 2 到 50 个完整 rule_id 的数组。"
        )
    rule_ids: list[str] = []
    for value in raw_rule_ids:
        rule_id = _required_text(value, name=f"{workflow} rule_ids")
        if _VOTING_RULE_ID_RE.fullmatch(rule_id) is None:
            raise StrategyWorkflowValidationError(
                f"{workflow} rule_ids 必须是完整的 candidate-rule ID。"
            )
        rule_ids.append(rule_id)
    if len(set(rule_ids)) != len(rule_ids):
        raise StrategyWorkflowValidationError(
            f"{workflow} rule_ids 不能包含重复 ID。"
        )
    n = inputs["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(rule_ids):
        raise StrategyWorkflowValidationError(
            f"{workflow} n 必须是 1 到规则数 {len(rule_ids)} 的整数。"
        )
    return {"strategy_type": strategy_type, "rule_ids": rule_ids, "n": n}


def validate_cross_matrix_candidate_search_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate explicit feature names and the bounded pair budget."""

    workflow = CROSS_MATRIX_CANDIDATE_SEARCH_WORKFLOW_ID
    allowed = {"features", "max_pairs"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 2 <= len(raw_features) <= 20
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 必须是 2 到 20 个明确字段的数组。"
        )
    features = [
        _column(
            value,
            name=f"{workflow} features",
            whitelist=context.allowed_columns,
        )
        for value in raw_features
    ]
    if len(set(features)) != len(features):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含重复字段。"
        )
    if context.target_col is not None and context.target_col in features:
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含当前目标列「{context.target_col}」。"
        )
    max_pairs = inputs["max_pairs"]
    if (
        isinstance(max_pairs, bool)
        or not isinstance(max_pairs, int)
        or not 1 <= max_pairs <= 190
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} max_pairs 必须是 1 到 190 的整数。"
        )
    return {"features": features, "max_pairs": max_pairs}


def validate_cross_matrix_candidate_build_from_search_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, str]:
    """Validate the two exact authenticated Cross pair pointers."""

    workflow = CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID
    allowed = {"search_id", "pair_id"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    search_id = _required_text(
        inputs["search_id"], name=f"{workflow} search_id"
    )
    pair_id = _required_text(inputs["pair_id"], name=f"{workflow} pair_id")
    if _CROSS_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} search_id 必须是完整的 cross-search ID。"
        )
    if _CROSS_PAIR_ID_RE.fullmatch(pair_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} pair_id 必须是完整的 cross-pair ID。"
        )
    return {"search_id": search_id, "pair_id": pair_id}


def validate_cross_rule_search_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate the human-owned rule-search universe and hard budget."""

    workflow = CROSS_RULE_SEARCH_WORKFLOW_ID
    allowed = {"features", "dimension", "constraints", "max_trials"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 2 <= len(raw_features) <= 12
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 必须是 2 到 12 个明确字段的数组。"
        )
    features = [
        _column(
            value,
            name=f"{workflow} features",
            whitelist=context.allowed_columns,
        )
        for value in raw_features
    ]
    if len(set(features)) != len(features):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含重复字段。"
        )
    if context.target_col is not None and context.target_col in features:
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含当前目标列「{context.target_col}」。"
        )
    dimension = inputs["dimension"]
    if isinstance(dimension, bool) or dimension not in {2, 3}:
        raise StrategyWorkflowValidationError(
            f"{workflow} dimension 只能是整数 2 或 3。"
        )
    raw_constraints = inputs["constraints"]
    constraint_fields = {
        "min_lift",
        "min_bad_count",
        "max_hit_share",
        "min_amount_lift",
    }
    if not isinstance(raw_constraints, Mapping):
        raise StrategyWorkflowValidationError(
            f"{workflow} constraints 必须是对象。"
        )
    unknown = sorted(set(raw_constraints) - constraint_fields)
    missing_constraints = sorted(constraint_fields - set(raw_constraints))
    if unknown or missing_constraints:
        details = []
        if unknown:
            details.append("不支持 " + "、".join(unknown))
        if missing_constraints:
            details.append("缺少 " + "、".join(missing_constraints))
        raise StrategyWorkflowValidationError(
            f"{workflow} constraints 字段无效：" + "；".join(details) + "。"
        )

    def finite_constraint(
        name: str,
        *,
        minimum: float,
        maximum: float,
        optional: bool = False,
    ) -> float | None:
        value = raw_constraints[name]
        if optional and value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            raise StrategyWorkflowValidationError(
                f"{workflow} constraints.{name} 必须是 "
                f"{minimum:g} 到 {maximum:g} 的有限数值"
                + ("或 null。" if optional else "。")
            )
        return float(value)

    min_bad_count = raw_constraints["min_bad_count"]
    if (
        isinstance(min_bad_count, bool)
        or not isinstance(min_bad_count, int)
        or min_bad_count < 0
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} constraints.min_bad_count 必须是非负整数。"
        )
    constraints = {
        "min_lift": finite_constraint("min_lift", minimum=0.0, maximum=1_000.0),
        "min_bad_count": min_bad_count,
        "max_hit_share": finite_constraint(
            "max_hit_share", minimum=0.0, maximum=1.0
        ),
        "min_amount_lift": finite_constraint(
            "min_amount_lift",
            minimum=0.0,
            maximum=1_000.0,
            optional=True,
        ),
    }
    max_trials = inputs["max_trials"]
    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, int)
        or not 1 <= max_trials <= 5_000
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} max_trials 必须是 1 到 5000 的整数。"
        )
    return {
        "features": features,
        "dimension": dimension,
        "constraints": constraints,
        "max_trials": max_trials,
    }


def validate_cross_rule_candidate_build_from_search_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate one exact Cross rule pointer and optional audit reason."""

    workflow = CROSS_RULE_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID
    allowed = {"search_id", "rule_id", "selection_reason"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"search_id", "rule_id"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    search_id = _required_text(
        inputs["search_id"], name=f"{workflow} search_id"
    )
    rule_id = _required_text(inputs["rule_id"], name=f"{workflow} rule_id")
    if _CROSS_RULE_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} search_id 必须是完整的 cross-rule-search ID。"
        )
    if _CROSS_RULE_ID_RE.fullmatch(rule_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} rule_id 必须是完整的 cross-rule ID。"
        )
    normalized: dict[str, Any] = {"search_id": search_id, "rule_id": rule_id}
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _selection_reason(
            inputs["selection_reason"], workflow_id=workflow
        )
    return normalized


def validate_cross_matrix_analysis_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate explicit 2D axes while deriving univariate Tool controls."""

    workflow = CROSS_MATRIX_ANALYSIS_WORKFLOW_ID
    axis_fields = {"x_feature", "x_method", "y_feature", "y_method"}
    derived_fields = {"features", "methods"}
    analysis_fields = {
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
        "manual_breakpoints",
    }
    _reject_fields(
        inputs,
        axis_fields | analysis_fields | derived_fields,
        workflow=workflow,
    )
    missing = sorted(axis_fields - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    x_feature = _column(
        inputs["x_feature"],
        name=f"{workflow} x_feature",
        whitelist=context.allowed_columns,
    )
    y_feature = _column(
        inputs["y_feature"],
        name=f"{workflow} y_feature",
        whitelist=context.allowed_columns,
    )
    if x_feature == y_feature:
        raise StrategyWorkflowValidationError(
            f"{workflow} 两个轴必须使用不同字段。"
        )
    if context.target_col is not None and context.target_col in {
        x_feature,
        y_feature,
    }:
        raise StrategyWorkflowValidationError(
            f"{workflow} 交叉轴不能使用目标列。"
        )

    def axis_method(field: str) -> str:
        method = _required_text(inputs[field], name=f"{workflow} {field}")
        if method not in UNIVARIATE_REFINEMENT_METHODS:
            raise StrategyWorkflowValidationError(
                f"{workflow} {field} 只能是："
                + "、".join(UNIVARIATE_REFINEMENT_METHODS)
                + "。"
            )
        return method

    x_method = axis_method("x_method")
    y_method = axis_method("y_method")
    numeric_methods = list(
        dict.fromkeys(
            method
            for method in (x_method, y_method)
            if method != "categorical"
        )
    )
    if "features" in inputs and inputs["features"] != [x_feature, y_feature]:
        raise StrategyWorkflowValidationError(
            f"{workflow} features 只能是平台派生的有序轴字段。"
        )
    if "methods" in inputs and inputs["methods"] != numeric_methods:
        raise StrategyWorkflowValidationError(
            f"{workflow} methods 只能是平台派生的数值轴方法。"
        )
    analysis_inputs: dict[str, Any] = {
        "features": [x_feature, y_feature],
        **{field: inputs[field] for field in analysis_fields if field in inputs},
    }
    if numeric_methods:
        analysis_inputs["methods"] = numeric_methods
    normalized = validate_univariate_analysis_inputs(
        analysis_inputs,
        context,
        manual_features=[
            feature
            for feature, method in (
                (x_feature, x_method),
                (y_feature, y_method),
            )
            if method == "manual"
        ],
    )
    return {
        **normalized,
        "x_feature": x_feature,
        "x_method": x_method,
        "y_feature": y_feature,
        "y_method": y_method,
    }


def validate_cross_matrix_cell_selection_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate exact user-owned Cross asset and cell pointers."""

    workflow = CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID
    allowed = {"cross_asset_id", "cell_ids", "selection_reason"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"cross_asset_id", "cell_ids"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    cross_asset_id = _required_text(
        inputs["cross_asset_id"], name=f"{workflow} cross_asset_id"
    )
    if _CANDIDATE_ASSET_ID_RE.fullmatch(cross_asset_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} cross_asset_id 必须是完整的 Cross candidate asset id。"
        )
    raw_cell_ids = inputs["cell_ids"]
    if (
        not isinstance(raw_cell_ids, Sequence)
        or isinstance(raw_cell_ids, str | bytes | bytearray)
        or not 1 <= len(raw_cell_ids) <= 400
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} cell_ids 必须是 1 到 400 个完整 cross-cell ID 的数组。"
        )
    cell_ids: list[str] = []
    for value in raw_cell_ids:
        cell_id = _required_text(value, name=f"{workflow} cell_ids")
        if _CROSS_MATRIX_CELL_ID_RE.fullmatch(cell_id) is None:
            raise StrategyWorkflowValidationError(
                f"{workflow} cell_ids 必须是 cross-cell- 后接 32 位小写十六进制字符。"
            )
        cell_ids.append(cell_id)
    if len(set(cell_ids)) != len(cell_ids):
        raise StrategyWorkflowValidationError(
            f"{workflow} cell_ids 不能包含重复 ID。"
        )
    normalized: dict[str, Any] = {
        "cross_asset_id": cross_asset_id,
        "cell_ids": cell_ids,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _cross_matrix_cell_selection_reason(
            inputs["selection_reason"]
        )
    return normalized


def voting_candidate_search_confirmation(inputs: Mapping[str, Any]) -> str:
    objective = inputs["objective"]
    details = [
        "已识别为〔Voting 组合搜索 Workflow〕",
        f"来源 Strategy Pool 类型：{inputs['strategy_type']}",
        f"组合参数：K={inputs['member_count']}，n={inputs['n']}",
        f"排序目标：{objective['metric']} / {objective['direction']}",
        f"确定性评估预算：最多 {inputs['max_combinations']:,} 个组合",
    ]
    if inputs["constraints"]:
        details.append(
            "资格约束："
            + "、".join(
                f"{item['metric']} {item['operator']} {item['value']:g}"
                for item in inputs["constraints"]
            )
        )
    else:
        details.append("资格约束：无")
    if inputs["include_rule_ids"]:
        details.append("必须包含：" + "、".join(inputs["include_rule_ids"]))
    if inputs["exclude_rule_ids"]:
        details.append("排除：" + "、".join(inputs["exclude_rule_ids"]))
    details.extend(
        [
            "平台将在计划创建前绑定当前 Pool 身份与受治理 development 样本；"
            "无需用户或模型提供 dataset pointer",
            "本步骤只搜索并发布聚合证据；不会构建候选或选择冠军，"
            "不会修改 Pool、入池、应用、采纳或部署",
        ]
    )
    return _confirmation(details)


def voting_candidate_build_from_search_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    details = [
        "已识别为〔Voting 搜索结果精确构建 Workflow〕",
        f"搜索证据 pointer：{inputs['search_id']}",
        f"组合 pointer：{inputs['combo_id']}",
    ]
    if "strategy_type" in inputs:
        details.append(f"来源 Strategy Pool 类型：{inputs['strategy_type']}")
    details.extend(
        [
            "平台将在计划开始前重新校验搜索证据、组合与当前 Pool；"
            "不会采用排名、最好或冠军等启发式选择",
            "本步骤只构建 development/backtested/unvalidated Voting 候选；"
            "不会加入或修改 Pool，不会设置动作、应用、采纳或部署",
        ]
    )
    return _confirmation(details)


def voting_candidate_build_confirmation(inputs: Mapping[str, Any]) -> str:
    return _confirmation(
        [
            "已识别为〔Voting / n-of-k 候选构建 Workflow〕",
            f"来源 Strategy Pool 类型：{inputs['strategy_type']}",
            "精确成员规则：" + "、".join(inputs["rule_ids"]),
            f"组合条件：{len(inputs['rule_ids'])} 条规则中至少命中 "
            f"{inputs['n']} 条",
            "平台将绑定当前 Pool revision/hash 和原始样本，逐行计算命中数与风险效果",
            "本步骤只生成 development/backtested/unvalidated 候选；"
            "不会入池、设置业务动作、采纳或部署",
        ]
    )


def cross_matrix_candidate_search_confirmation(inputs: Mapping[str, Any]) -> str:
    return _confirmation(
        [
            "已识别为〔Cross Matrix 自动组合搜索 Workflow〕",
            "显式候选字段：" + "、".join(inputs["features"]),
            f"确定性评估预算：最多 {inputs['max_pairs']} 个特征对",
            "平台将在计划创建时绑定最新精确单变量候选证据，并仅使用其"
            " risk/development 样本；每个字段的轴方法由受控 Tool 从父证据中"
            "选择最高排名的可用方法",
            "本步骤只发布聚合搜索证据；不会构建候选或选择候选，不会入池、"
            "应用、采纳或部署",
        ]
    )


def cross_matrix_candidate_build_from_search_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    return _confirmation(
        [
            "已识别为〔Cross 搜索结果精确构建 Workflow〕",
            f"搜索证据 pointer：{inputs['search_id']}",
            f"特征对 pointer：{inputs['pair_id']}",
            "平台将在计划开始前重新认证完整搜索证据、父候选、数据与"
            " risk/development 样本，并重新计算精确 Cross 资产",
            "不会采用排名、最好或冠军等启发式选择；本步骤只构建一个 "
            "development/backtested/unvalidated Cross 候选",
            "不会加入或修改 Pool，不会设置动作、应用、采纳或部署",
        ]
    )


def cross_rule_search_confirmation(inputs: Mapping[str, Any]) -> str:
    constraints = inputs["constraints"]
    amount_lift = constraints["min_amount_lift"]
    return _confirmation(
        [
            "已识别为〔2D/3D Cross 阈值规则搜索 Workflow〕",
            "显式候选字段：" + "、".join(inputs["features"]),
            f"组合维度：{inputs['dimension']}D；最多评估 "
            f"{inputs['max_trials']} 条确定性试验",
            "约束："
            f"min_lift={constraints['min_lift']:g}，"
            f"min_bad_count={constraints['min_bad_count']}，"
            f"max_hit_share={constraints['max_hit_share']:g}，"
            "min_amount_lift="
            + ("null" if amount_lift is None else f"{amount_lift:g}"),
            "平台将绑定最新精确单变量证据与 risk/development 样本，"
            "从认证分箱边界和风险方向生成有预算的 2D/3D 阈值组合",
            "本步骤只发布全部已评估规则的聚合证据与排序；不会自动选择、"
            "构建候选、入池、应用、采纳或部署",
        ]
    )


def cross_rule_candidate_build_from_search_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    details = [
        "已识别为〔Cross 阈值规则精确候选构建 Workflow〕",
        f"搜索证据 pointer：{inputs['search_id']}",
        f"规则 pointer：{inputs['rule_id']}",
        "平台将重新认证并完整重放搜索、数据、样本和规则条件；"
        "不会采用排名、最好或冠军等启发式选择",
        "本步骤只构建一个 development/unvalidated 候选；不会自动入池、"
        "应用、采纳或部署",
    ]
    if "selection_reason" in inputs:
        details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    return _confirmation(details)


def cross_matrix_analysis_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔二维 Cross Matrix 候选分析 Workflow〕",
        f"X 轴：{inputs['x_feature']} / {inputs['x_method']}；"
        f"Y 轴：{inputs['y_feature']} / {inputs['y_method']}",
        f"数值目标箱数 {inputs['bin_count']}，"
        f"最小箱占比 {inputs['min_bin_pct']:.2%}",
        "平台会先生成两个轴的不可变单变量证据，再逐行重放完整二维矩阵",
        "只生成 development/backtested/unvalidated Cross evidence；"
        "不会选择格子、入池、采纳或部署",
    ]
    if "loan_amount_col" in inputs:
        details.append(f"放款金额列：{inputs['loan_amount_col']}")
    if "overdue_amount_col" in inputs:
        details.append(f"逾期金额列：{inputs['overdue_amount_col']}")
    if inputs["sentinel_values"]:
        details.append(
            "独立哨兵值："
            + "、".join(str(value) for value in inputs["sentinel_values"])
        )
    if "manual_breakpoints" in inputs:
        details.append(
            "手工轴切点："
            + "；".join(
                feature
                + "=["
                + "、".join(f"{value:g}" for value in points)
                + "]"
                for feature, points in inputs["manual_breakpoints"].items()
            )
        )
    return _confirmation(details)


def cross_matrix_cell_selection_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔Cross Matrix 精确单元格选择 Workflow〕",
        f"完整 Cross 候选资产 pointer：{inputs['cross_asset_id']}",
        "精确 cell pointers：" + "、".join(inputs["cell_ids"]),
        "多个 cell 按确定性 OR 语义组成一个不可变选择；平台按源矩阵顺序归一化",
        "本步骤不排名、不推荐、不生成业务动作，也不会入池、采纳或部署",
    ]
    if "selection_reason" in inputs:
        details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    return _confirmation(details)


def prepare_voting_candidate_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    evidence = _workflow_evidence(
        VOTING_CANDIDATE_SEARCH_WORKFLOW_ID,
        inputs,
        context,
        slots,
    )
    _require_evidence_fields(
        VOTING_CANDIDATE_SEARCH_WORKFLOW_ID,
        evidence,
        ("pool_ref",),
    )
    if not isinstance(evidence["pool_ref"], Mapping):
        _invalid_evidence(
            VOTING_CANDIDATE_SEARCH_WORKFLOW_ID,
            "pool_ref 必须是对象。",
            fields=("pool_ref",),
        )
    slots.update(evidence)
    return PreparedStrategyPlan(
        workflow_id=VOTING_CANDIDATE_SEARCH_WORKFLOW_ID,
        template_id="strategy_voting_candidate_search",
        slots=deep_freeze(slots),
    )


def prepare_voting_candidate_build_from_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    slots.update(
        _workflow_evidence(
            VOTING_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
            inputs,
            context,
            slots,
        )
    )
    return PreparedStrategyPlan(
        workflow_id=VOTING_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
        template_id="strategy_voting_candidate_build_from_search",
        slots=deep_freeze(slots),
    )


def prepare_voting_candidate_build(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots: dict[str, Any] = {
        "strategy_type": inputs["strategy_type"],
        "n": inputs["n"],
    }
    evidence = _workflow_evidence(
        VOTING_CANDIDATE_BUILD_WORKFLOW_ID,
        inputs,
        context,
        {**slots, "rule_ids": inputs["rule_ids"]},
    )
    _require_evidence_fields(
        VOTING_CANDIDATE_BUILD_WORKFLOW_ID,
        evidence,
        (
            "expected_pool_revision",
            "expected_pool_snapshot_hash",
            "selected_entry_ids",
        ),
    )
    revision = evidence["expected_pool_revision"]
    snapshot_hash = evidence["expected_pool_snapshot_hash"]
    entry_ids = evidence["selected_entry_ids"]
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(snapshot_hash, str)
        or _SHA256_RE.fullmatch(snapshot_hash) is None
        or not isinstance(entry_ids, Sequence)
        or isinstance(entry_ids, str | bytes | bytearray)
        or len(entry_ids) != len(inputs["rule_ids"])
        or any(
            not isinstance(entry_id, str)
            or _POOL_ENTRY_ID_RE.fullmatch(entry_id) is None
            for entry_id in entry_ids
        )
        or len(set(entry_ids)) != len(entry_ids)
    ):
        _invalid_evidence(
            VOTING_CANDIDATE_BUILD_WORKFLOW_ID,
            "Pool revision/hash 或 selected_entry_ids 绑定无效。",
            fields=(
                "expected_pool_revision",
                "expected_pool_snapshot_hash",
                "selected_entry_ids",
            ),
        )
    slots.update(evidence)
    return PreparedStrategyPlan(
        workflow_id=VOTING_CANDIDATE_BUILD_WORKFLOW_ID,
        template_id="strategy_voting_candidate_build",
        slots=deep_freeze(slots),
    )


def prepare_cross_matrix_candidate_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    evidence = _workflow_evidence(
        CROSS_MATRIX_CANDIDATE_SEARCH_WORKFLOW_ID,
        inputs,
        context,
        slots,
    )
    _validate_cross_source_evidence(
        CROSS_MATRIX_CANDIDATE_SEARCH_WORKFLOW_ID,
        evidence,
    )
    slots.update(evidence)
    return PreparedStrategyPlan(
        workflow_id=CROSS_MATRIX_CANDIDATE_SEARCH_WORKFLOW_ID,
        template_id="strategy_cross_matrix_candidate_search",
        slots=deep_freeze(slots),
    )


def prepare_cross_matrix_candidate_build_from_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    slots.update(
        _workflow_evidence(
            CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
            inputs,
            context,
            slots,
        )
    )
    return PreparedStrategyPlan(
        workflow_id=CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
        template_id="strategy_cross_matrix_candidate_build_from_search",
        slots=deep_freeze(slots),
    )


def prepare_cross_rule_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    evidence = _workflow_evidence(
        CROSS_RULE_SEARCH_WORKFLOW_ID,
        inputs,
        context,
        slots,
    )
    _validate_cross_source_evidence(CROSS_RULE_SEARCH_WORKFLOW_ID, evidence)
    slots.update(evidence)
    return PreparedStrategyPlan(
        workflow_id=CROSS_RULE_SEARCH_WORKFLOW_ID,
        template_id="strategy_cross_rule_search",
        slots=deep_freeze(slots),
    )


def prepare_cross_rule_candidate_build_from_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots: dict[str, Any] = {
        "search_id": inputs["search_id"],
        "rule_id": inputs["rule_id"],
        "selection_reason": inputs.get("selection_reason"),
    }
    slots.update(
        _workflow_evidence(
            CROSS_RULE_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
            inputs,
            context,
            slots,
        )
    )
    return PreparedStrategyPlan(
        workflow_id=CROSS_RULE_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
        template_id="strategy_cross_rule_candidate_build_from_search",
        slots=deep_freeze(slots),
    )


def prepare_cross_matrix_analysis(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    evidence = _workflow_evidence(
        CROSS_MATRIX_ANALYSIS_WORKFLOW_ID,
        inputs,
        context,
        {**slots, "drop_nan_labels": context.drop_nan_labels},
    )
    _validate_cross_matrix_dataset_evidence(evidence, context)
    slots.update(evidence)
    if context.drop_nan_labels:
        slots["drop_nan_labels"] = True
    return PreparedStrategyPlan(
        workflow_id=CROSS_MATRIX_ANALYSIS_WORKFLOW_ID,
        template_id="strategy_cross_matrix_analysis",
        slots=deep_freeze(slots),
    )


def prepare_cross_matrix_cell_selection(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    canonical_slots: dict[str, Any] = {}
    if "selection_reason" in inputs:
        canonical_slots["selection_reason"] = inputs["selection_reason"]
    evidence = _workflow_evidence(
        CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
        inputs,
        context,
        {"selection_reason": inputs.get("selection_reason")},
    )
    if "cross_asset_id" in evidence:
        raise StrategyWorkflowValidationError(
            f"{CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID} 平台证据不能保留原始 "
            "cross_asset_id pointer。",
            code="strategy_workflow_evidence_invalid",
            fields=("cross_asset_id",),
        )
    _require_evidence_fields(
        CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
        evidence,
        (
            "source_artifact_id",
            "expected_artifact_content_hash",
            "expected_asset_id",
            "expected_asset_hash",
            "expected_candidate_id",
            "expected_evidence_hash",
            "cell_ids",
        ),
    )
    if evidence["expected_asset_id"] != inputs["cross_asset_id"]:
        _invalid_evidence(
            CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
            "expected_asset_id 与用户请求的 cross_asset_id 不一致。",
            fields=("expected_asset_id",),
        )
    for field in (
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_hash",
        "expected_evidence_hash",
    ):
        if not isinstance(evidence[field], str) or not evidence[field]:
            _invalid_evidence(
                CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
                f"{field} 必须是非空文本。",
                fields=(field,),
            )
    expected_candidate_id = evidence["expected_candidate_id"]
    if (
        not isinstance(expected_candidate_id, str)
        or _CANDIDATE_ID_RE.fullmatch(expected_candidate_id) is None
    ):
        _invalid_evidence(
            CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
            "expected_candidate_id 必须是完整 candidate id。",
            fields=("expected_candidate_id",),
        )
    ordered_cell_ids = evidence.get("cell_ids")
    requested_cell_ids = inputs["cell_ids"]
    if (
        not isinstance(ordered_cell_ids, Sequence)
        or isinstance(ordered_cell_ids, str | bytes | bytearray)
        or any(not isinstance(cell_id, str) for cell_id in ordered_cell_ids)
        or len(ordered_cell_ids) != len(requested_cell_ids)
        or len(set(ordered_cell_ids)) != len(ordered_cell_ids)
        or set(ordered_cell_ids) != set(requested_cell_ids)
    ):
        raise StrategyWorkflowValidationError(
            f"{CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID} 平台证据必须返回"
            "与请求精确一致且按源矩阵排序的 cell_ids。",
            code="strategy_workflow_evidence_invalid",
            fields=("cell_ids",),
        )
    slots = {**evidence, **canonical_slots}
    return PreparedStrategyPlan(
        workflow_id=CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
        template_id="strategy_cross_matrix_cell_selection",
        slots=deep_freeze(slots),
    )


def _workflow_evidence(
    workflow_id: str,
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
    canonical_slots: Mapping[str, Any],
) -> dict[str, Any]:
    binder = context.bind_workflow_evidence
    if binder is None:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 缺少平台认证的证据绑定。",
            code="strategy_workflow_evidence_binding_required",
        )
    evidence = binder(workflow_id, deep_freeze(deep_thaw(inputs)))
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
    conflicts = sorted(set(evidence) & set(canonical_slots))
    if conflicts:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据不能覆盖 canonical slots："
            + "、".join(conflicts)
            + "。",
            code="strategy_workflow_evidence_conflict",
            fields=conflicts,
        )
    copied = deep_thaw(deep_freeze(evidence))
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guarded above
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据绑定无法复制。",
            code="strategy_workflow_evidence_invalid",
        )
    return copied


def _confirmation(details: list[str]) -> str:
    details.append(
        "请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。"
    )
    return "；".join(details)


def _require_evidence_fields(
    workflow_id: str,
    evidence: Mapping[str, Any],
    required: Sequence[str],
) -> None:
    missing = sorted(set(required) - set(evidence))
    if missing:
        _invalid_evidence(
            workflow_id,
            "平台证据缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )


def _validate_cross_source_evidence(
    workflow_id: str,
    evidence: Mapping[str, Any],
) -> None:
    required = (
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
    )
    _require_evidence_fields(workflow_id, evidence, required)
    patterns = {
        "source_artifact_id": _SHA256_RE,
        "expected_artifact_content_hash": _SHA256_RE,
        "expected_candidate_id": _CANDIDATE_ID_RE,
        "expected_evidence_hash": _SHA256_RE,
    }
    invalid = [
        field
        for field, pattern in patterns.items()
        if not isinstance(evidence[field], str)
        or pattern.fullmatch(evidence[field]) is None
    ]
    if invalid:
        _invalid_evidence(
            workflow_id,
            "Cross source identity 绑定无效。",
            fields=invalid,
        )


def _validate_cross_matrix_dataset_evidence(
    evidence: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> None:
    workflow_id = CROSS_MATRIX_ANALYSIS_WORKFLOW_ID
    required = (
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
    )
    _require_evidence_fields(workflow_id, evidence, required)
    invalid: list[str] = []
    dataset_id = evidence["dataset_id"]
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or (context.dataset_id is not None and dataset_id != context.dataset_id)
    ):
        invalid.append("dataset_id")
    for field in ("expected_content_hash", "semantic_mapping_hash"):
        if (
            not isinstance(evidence[field], str)
            or _SHA256_RE.fullmatch(evidence[field]) is None
        ):
            invalid.append(field)
    for field in ("workspace_revision", "analysis_generation"):
        if isinstance(evidence[field], bool) or not isinstance(evidence[field], int):
            invalid.append(field)
    if not isinstance(evidence["target_col"], str) or not evidence["target_col"]:
        invalid.append("target_col")
    if not isinstance(evidence["sample_design_ref"], Mapping):
        invalid.append("sample_design_ref")
    if invalid:
        _invalid_evidence(
            workflow_id,
            "dataset/workspace/target/sample 证据绑定无效。",
            fields=invalid,
        )


def _invalid_evidence(
    workflow_id: str,
    detail: str,
    *,
    fields: Sequence[str],
) -> None:
    raise StrategyWorkflowValidationError(
        f"{workflow_id} {detail}",
        code="strategy_workflow_evidence_invalid",
        fields=fields,
    )


def _voting_search_rule_id_array(value: object, *, name: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or len(value) > 50
    ):
        raise StrategyWorkflowValidationError(
            f"{name} 必须是最多 50 个完整 rule_id 的数组。"
        )
    normalized: list[str] = []
    for raw in value:
        rule_id = _required_text(raw, name=name)
        if _VOTING_RULE_ID_RE.fullmatch(rule_id) is None:
            raise StrategyWorkflowValidationError(
                f"{name} 必须只包含完整的 candidate-rule ID。"
            )
        normalized.append(rule_id)
    if len(set(normalized)) != len(normalized):
        raise StrategyWorkflowValidationError(f"{name} 不能包含重复 ID。")
    return sorted(normalized)


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


def _selection_reason(value: object, *, workflow_id: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} selection_reason 必须是文本。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or len(canonical) > 500:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} selection_reason 必须是 1 到 500 个字符。"
        )
    return canonical


def _cross_matrix_cell_selection_reason(value: object) -> str:
    workflow = CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID
    if not isinstance(value, str):
        raise StrategyWorkflowValidationError(
            f"{workflow} selection_reason 必须是文本。"
        )
    if "\x00" in value:
        raise StrategyWorkflowValidationError(
            f"{workflow} selection_reason 不能包含 NUL。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical:
        raise StrategyWorkflowValidationError(
            f"{workflow} selection_reason 必须是非空文本。"
        )
    if len(canonical) > 500:
        raise StrategyWorkflowValidationError(
            f"{workflow} selection_reason 最多 500 个字符。"
        )
    return canonical


def _reject_fields(
    inputs: Mapping[str, Any],
    allowed: set[str],
    *,
    workflow: str,
) -> None:
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 字段名必须是文本。"
        )
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{workflow} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。"
        )


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyWorkflowValidationError(f"{name} 必须是非空文本。")
    return value.strip()


def _bounded_number(
    value: object,
    *,
    name: str,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StrategyWorkflowValidationError(f"{name} 必须是有限数字。")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < 0
        or (maximum is not None and number > maximum)
    ):
        if maximum is None:
            raise StrategyWorkflowValidationError(
                f"{name} 必须是大于等于 0 的有限数字。"
            )
        raise StrategyWorkflowValidationError(
            f"{name} 必须是 0 到 {maximum:g} 之间的有限数字。"
        )
    return number


CROSS_VOTING_WORKFLOW_SPECS = (
    StrategyWorkflowSpec(
        workflow_id=VOTING_CANDIDATE_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_voting_candidate_search",),
        validator=validate_voting_candidate_search_inputs,
        confirmation=voting_candidate_search_confirmation,
        preparer=prepare_voting_candidate_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=VOTING_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_voting_candidate_build_from_search",),
        validator=validate_voting_candidate_build_from_search_inputs,
        confirmation=voting_candidate_build_from_search_confirmation,
        preparer=prepare_voting_candidate_build_from_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=VOTING_CANDIDATE_BUILD_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=False,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_voting_candidate_build",),
        validator=validate_voting_candidate_build_inputs,
        confirmation=voting_candidate_build_confirmation,
        preparer=prepare_voting_candidate_build,
    ),
    StrategyWorkflowSpec(
        workflow_id=CROSS_MATRIX_CANDIDATE_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_cross_matrix_candidate_search",),
        validator=validate_cross_matrix_candidate_search_inputs,
        confirmation=cross_matrix_candidate_search_confirmation,
        preparer=prepare_cross_matrix_candidate_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=CROSS_MATRIX_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_cross_matrix_candidate_build_from_search",),
        validator=validate_cross_matrix_candidate_build_from_search_inputs,
        confirmation=cross_matrix_candidate_build_from_search_confirmation,
        preparer=prepare_cross_matrix_candidate_build_from_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=CROSS_RULE_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_cross_rule_search",),
        validator=validate_cross_rule_search_inputs,
        confirmation=cross_rule_search_confirmation,
        preparer=prepare_cross_rule_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=CROSS_RULE_CANDIDATE_BUILD_FROM_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_cross_rule_candidate_build_from_search",),
        validator=validate_cross_rule_candidate_build_from_search_inputs,
        confirmation=cross_rule_candidate_build_from_search_confirmation,
        preparer=prepare_cross_rule_candidate_build_from_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=CROSS_MATRIX_ANALYSIS_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_DATA_REQUIREMENTS,
        template_ids=("strategy_cross_matrix_analysis",),
        validator=validate_cross_matrix_analysis_inputs,
        confirmation=cross_matrix_analysis_confirmation,
        preparer=prepare_cross_matrix_analysis,
    ),
    StrategyWorkflowSpec(
        workflow_id=CROSS_MATRIX_CELL_SELECTION_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=False,
        requirements=_NO_REQUIREMENTS,
        template_ids=("strategy_cross_matrix_cell_selection",),
        validator=validate_cross_matrix_cell_selection_inputs,
        confirmation=cross_matrix_cell_selection_confirmation,
        preparer=prepare_cross_matrix_cell_selection,
    ),
)


__all__ = [
    "CROSS_VOTING_WORKFLOW_SPECS",
    "cross_matrix_analysis_confirmation",
    "cross_matrix_candidate_build_from_search_confirmation",
    "cross_matrix_candidate_search_confirmation",
    "cross_matrix_cell_selection_confirmation",
    "cross_rule_candidate_build_from_search_confirmation",
    "cross_rule_search_confirmation",
    "prepare_cross_matrix_analysis",
    "prepare_cross_matrix_candidate_build_from_search",
    "prepare_cross_matrix_candidate_search",
    "prepare_cross_matrix_cell_selection",
    "prepare_cross_rule_candidate_build_from_search",
    "prepare_cross_rule_search",
    "prepare_voting_candidate_build",
    "prepare_voting_candidate_build_from_search",
    "prepare_voting_candidate_search",
    "validate_cross_matrix_analysis_inputs",
    "validate_voting_candidate_build_from_search_inputs",
    "validate_voting_candidate_build_inputs",
    "validate_voting_candidate_search_inputs",
    "validate_cross_matrix_candidate_build_from_search_inputs",
    "validate_cross_matrix_candidate_search_inputs",
    "validate_cross_matrix_cell_selection_inputs",
    "validate_cross_rule_candidate_build_from_search_inputs",
    "validate_cross_rule_search_inputs",
    "voting_candidate_build_confirmation",
    "voting_candidate_build_from_search_confirmation",
    "voting_candidate_search_confirmation",
]

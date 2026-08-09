from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .contracts import (
    PreparedStrategyPlan,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_freeze,
    deep_thaw,
)


_PROFIT_PARAMETER_FIELDS = {
    "annual_rate",
    "funding_rate",
    "lgd",
    "operating_cost_per_loan",
    "term_months",
}


def validate_profit_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    allowed = {"segment_col", "ead_col", "pd_col", "profit_params"}
    _reject_fields(inputs, allowed, workflow="profit_calc")
    missing = sorted({"ead_col", "pd_col", "profit_params"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            "profit_calc 缺少字段：" + "、".join(missing) + "。"
        )
    params = inputs["profit_params"]
    if not isinstance(params, Mapping):
        raise StrategyWorkflowValidationError(
            "profit_calc 的 profit_params 必须是对象。"
        )
    if any(not isinstance(key, str) for key in params):
        raise StrategyWorkflowValidationError(
            "profit_calc 的 profit_params 字段名必须是文本。"
        )
    missing_params = sorted(_PROFIT_PARAMETER_FIELDS - set(params))
    unexpected_params = sorted(set(params) - _PROFIT_PARAMETER_FIELDS)
    if missing_params:
        raise StrategyWorkflowValidationError(
            "profit_calc 的 profit_params 缺少字段："
            + "、".join(missing_params)
            + "。"
        )
    if unexpected_params:
        raise StrategyWorkflowValidationError(
            "profit_calc 的 profit_params 包含不支持的字段："
            + "、".join(unexpected_params)
            + "。"
        )
    normalized_params = {
        "annual_rate": _bounded_number(
            params["annual_rate"], name="利润 annual_rate", maximum=1
        ),
        "funding_rate": _bounded_number(
            params["funding_rate"], name="利润 funding_rate", maximum=1
        ),
        "lgd": _bounded_number(params["lgd"], name="利润 lgd", maximum=1),
        "operating_cost_per_loan": _bounded_number(
            params["operating_cost_per_loan"],
            name="利润 operating_cost_per_loan",
        ),
    }
    term_months = params["term_months"]
    if (
        isinstance(term_months, bool)
        or not isinstance(term_months, int)
        or term_months < 1
    ):
        raise StrategyWorkflowValidationError(
            "利润 term_months 必须是大于等于 1 的整数。"
        )
    normalized_params["term_months"] = term_months
    normalized: dict[str, Any] = {
        "ead_col": _column(
            inputs["ead_col"],
            name="利润 EAD 列 ead_col",
            whitelist=context.allowed_columns,
        ),
        "pd_col": _column(
            inputs["pd_col"],
            name="利润 PD 列 pd_col",
            whitelist=context.allowed_columns,
        ),
        "profit_params": normalized_params,
    }
    if "segment_col" in inputs:
        normalized["segment_col"] = _column(
            inputs["segment_col"],
            name="profit_calc segment_col",
            whitelist=context.allowed_columns,
        )
    return normalized


def validate_roll_rate_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    allowed = {
        "id_col",
        "time_col",
        "status_col",
        "states",
        "balance_col",
        "observation_semantics",
    }
    _reject_fields(inputs, allowed, workflow="roll_rate_matrix")
    required = {"id_col", "time_col", "status_col", "states"}
    missing = sorted(required - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            "roll_rate_matrix 缺少字段：" + "、".join(missing) + "。"
        )
    normalized: dict[str, Any] = {
        key: _column(
            inputs[key],
            name=f"roll_rate_matrix {key}",
            whitelist=context.allowed_columns,
        )
        for key in ("id_col", "time_col")
    }
    status_whitelist = context.allowed_columns
    if context.target_col is not None and context.target_col not in status_whitelist:
        status_whitelist = (*status_whitelist, context.target_col)
    normalized["status_col"] = _column(
        inputs["status_col"],
        name="roll_rate_matrix status_col",
        whitelist=status_whitelist,
    )
    if len(set(normalized.values())) != len(normalized):
        raise StrategyWorkflowValidationError(
            "roll_rate_matrix 的 id_col、time_col、status_col 必须互不相同。"
        )
    states = inputs["states"]
    if (
        not isinstance(states, Sequence)
        or isinstance(states, str | bytes | bytearray)
        or not 2 <= len(states) <= 50
    ):
        raise StrategyWorkflowValidationError(
            "roll_rate_matrix states 必须是包含 2 到 50 个状态的有序数组。"
        )
    normalized_states = [
        _required_text(state, name="roll_rate_matrix states 状态")
        for state in states
    ]
    if len(set(normalized_states)) != len(normalized_states):
        raise StrategyWorkflowValidationError(
            "roll_rate_matrix states 不能包含重复状态。"
        )
    normalized["states"] = normalized_states
    semantics = inputs.get("observation_semantics", "adjacent_observation")
    if semantics != "adjacent_observation":
        raise StrategyWorkflowValidationError(
            "roll_rate_matrix observation_semantics 只能是 adjacent_observation；"
            "固定月末快照迁徙应使用 portfolio Workflow。"
        )
    normalized["observation_semantics"] = semantics
    if "balance_col" in inputs:
        balance_col = _column(
            inputs["balance_col"],
            name="roll_rate_matrix balance_col",
            whitelist=context.allowed_columns,
        )
        if balance_col in {
            normalized["id_col"],
            normalized["time_col"],
            normalized["status_col"],
        }:
            raise StrategyWorkflowValidationError(
                "roll_rate_matrix balance_col 不能复用 ID、时间或状态列。"
            )
        normalized["balance_col"] = balance_col
    return normalized


def validate_limit_pricing_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    allowed = {
        "score_col",
        "pd_col",
        "target_col",
        "band_edges",
        "n_bands",
        "limit_grid",
        "rate_grid",
        "lgd",
        "funding_rate",
        "term_months",
        "cost_per_loan",
        "el_ead_max",
        "strategy_id",
        "drop_nan_labels",
    }
    _reject_fields(inputs, allowed, workflow="limit_pricing_matrix")
    required = {
        "score_col",
        "limit_grid",
        "rate_grid",
        "lgd",
        "funding_rate",
        "term_months",
        "cost_per_loan",
        "el_ead_max",
    }
    missing = sorted(required - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            "limit_pricing_matrix 缺少字段：" + "、".join(missing) + "。"
        )
    has_pd = "pd_col" in inputs
    has_target = "target_col" in inputs
    if has_pd == has_target:
        raise StrategyWorkflowValidationError(
            "limit_pricing_matrix 的 pd_col 与 target_col 必须且只能二选一。"
        )
    has_edges = "band_edges" in inputs
    has_band_count = "n_bands" in inputs
    if has_edges == has_band_count:
        raise StrategyWorkflowValidationError(
            "limit_pricing_matrix 的 band_edges 与 n_bands 必须且只能二选一。"
        )
    normalized: dict[str, Any] = {
        "score_col": _column(
            inputs["score_col"],
            name="limit_pricing_matrix score_col",
            whitelist=context.allowed_columns,
        ),
        "limit_grid": _number_sequence(
            inputs["limit_grid"],
            name="limit_pricing_matrix limit_grid",
            minimum=0,
            exclusive_minimum=True,
            maximum_items=50,
        ),
        "rate_grid": _number_sequence(
            inputs["rate_grid"],
            name="limit_pricing_matrix rate_grid",
            minimum=0,
            maximum=1,
            maximum_items=50,
        ),
        "lgd": _bounded_number(
            inputs["lgd"], name="limit_pricing_matrix lgd", maximum=1
        ),
        "funding_rate": _bounded_number(
            inputs["funding_rate"],
            name="limit_pricing_matrix funding_rate",
            maximum=1,
        ),
        "cost_per_loan": _bounded_number(
            inputs["cost_per_loan"],
            name="limit_pricing_matrix cost_per_loan",
        ),
        "el_ead_max": _bounded_number(
            inputs["el_ead_max"],
            name="limit_pricing_matrix el_ead_max",
            maximum=1,
        ),
    }
    term_months = inputs["term_months"]
    if (
        isinstance(term_months, bool)
        or not isinstance(term_months, int)
        or not 1 <= term_months <= 600
    ):
        raise StrategyWorkflowValidationError(
            "limit_pricing_matrix term_months 必须是 1 到 600 的整数。"
        )
    normalized["term_months"] = term_months
    if has_pd:
        normalized["pd_col"] = _column(
            inputs["pd_col"],
            name="limit_pricing_matrix pd_col",
            whitelist=context.allowed_columns,
        )
    else:
        requested_target = _required_text(
            inputs["target_col"], name="limit_pricing_matrix target_col"
        )
        if context.target_col is None or requested_target != context.target_col:
            raise StrategyWorkflowValidationError(
                "limit_pricing_matrix target_col 必须与任务当前确认的目标列一致。"
            )
        normalized["target_col"] = requested_target
    if has_edges:
        edges = _number_sequence(
            inputs["band_edges"],
            name="limit_pricing_matrix band_edges",
            minimum=None,
            maximum_items=51,
            minimum_items=2,
        )
        if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
            raise StrategyWorkflowValidationError(
                "limit_pricing_matrix band_edges 必须严格递增。"
            )
        normalized["band_edges"] = edges
        band_count = len(edges) - 1
    else:
        n_bands = inputs["n_bands"]
        if (
            isinstance(n_bands, bool)
            or not isinstance(n_bands, int)
            or not 1 <= n_bands <= 20
        ):
            raise StrategyWorkflowValidationError(
                "limit_pricing_matrix n_bands 必须是 1 到 20 的整数。"
            )
        normalized["n_bands"] = n_bands
        band_count = n_bands
    if band_count * len(normalized["limit_grid"]) * len(normalized["rate_grid"]) > 2000:
        raise StrategyWorkflowValidationError(
            "limit_pricing_matrix 网格最多允许 2000 个组合。"
        )
    if "strategy_id" in inputs:
        normalized["strategy_id"] = _required_text(
            inputs["strategy_id"], name="limit_pricing_matrix strategy_id"
        )
    if "drop_nan_labels" in inputs:
        if not isinstance(inputs["drop_nan_labels"], bool):
            raise StrategyWorkflowValidationError(
                "limit_pricing_matrix drop_nan_labels 必须是布尔值。"
            )
        if has_pd:
            raise StrategyWorkflowValidationError(
                "limit_pricing_matrix 使用 pd_col 时不会读取标签，"
                "请删除未使用的 drop_nan_labels。"
            )
        normalized["drop_nan_labels"] = inputs["drop_nan_labels"]
    return normalized


def profit_confirmation(inputs: Mapping[str, Any]) -> str:
    params = inputs["profit_params"]
    details = [
        "已识别为〔标准利润分析 Workflow〕",
        f"EAD 列 {inputs['ead_col']}，PD 列 {inputs['pd_col']}",
        "分析范围："
        + (
            f"按 {inputs['segment_col']} 分组"
            if "segment_col" in inputs
            else "全样本"
        ),
        (
            f"年利率 {params['annual_rate']:.2%}，资金成本率 {params['funding_rate']:.2%}，"
            f"LGD {params['lgd']:.2%}，单笔成本 {params['operating_cost_per_loan']:g}，"
            f"期限 {params['term_months']} 个月"
        ),
    ]
    return _confirmation(details)


def roll_rate_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔标准滚动率矩阵 Workflow〕",
        (
            f"客户 ID 列 {inputs['id_col']}，时间列 {inputs['time_col']}，"
            f"状态列 {inputs['status_col']}"
        ),
        "状态顺序：" + " → ".join(inputs["states"]),
        "观测口径：相邻观测记录，不等同于固定月末快照迁徙",
    ]
    if "balance_col" in inputs:
        details.append(f"余额加权列：{inputs['balance_col']}")
    return _confirmation(details)


def limit_pricing_confirmation(inputs: Mapping[str, Any]) -> str:
    risk_source = (
        f"PD 列 {inputs['pd_col']}"
        if "pd_col" in inputs
        else f"目标列 {inputs['target_col']}"
    )
    banding = (
        "分箱边界 " + "、".join(f"{value:g}" for value in inputs["band_edges"])
        if "band_edges" in inputs
        else f"等频分为 {inputs['n_bands']} 档"
    )
    details = [
        "已识别为〔标准额度定价矩阵 Workflow〕",
        f"评分列 {inputs['score_col']}，风险来源 {risk_source}，{banding}",
        "额度网格：" + "、".join(f"{value:,.12g}" for value in inputs["limit_grid"]),
        "利率网格：" + "、".join(f"{value:.2%}" for value in inputs["rate_grid"]),
        (
            f"LGD {inputs['lgd']:.2%}，资金成本率 {inputs['funding_rate']:.2%}，"
            f"期限 {inputs['term_months']} 个月，单笔成本 {inputs['cost_per_loan']:g}，"
            f"EL/EAD 上限 {inputs['el_ead_max']:.2%}"
        ),
    ]
    if "target_col" in inputs:
        details.append(
            "标签缺失处理："
            + (
                "按明确授权丢弃 NaN 标签行"
                if inputs.get("drop_nan_labels")
                else "不自动丢弃 NaN 标签行"
            )
        )
    if "strategy_id" in inputs:
        details.append(f"关联策略 ID：{inputs['strategy_id']}")
    details.append("平台先计算完整矩阵；接受或导出矩阵仍需第二次明确确认")
    return _confirmation(details)


def prepare_profit(
    inputs: Mapping[str, Any], context: StrategyWorkflowPreparationContext
) -> PreparedStrategyPlan:
    return _prepare_dataset_plan(
        "profit_calc", "strategy_profit_analysis", inputs, context
    )


def prepare_roll_rate(
    inputs: Mapping[str, Any], context: StrategyWorkflowPreparationContext
) -> PreparedStrategyPlan:
    return _prepare_dataset_plan(
        "roll_rate_matrix", "strategy_roll_rate_analysis", inputs, context
    )


def prepare_limit_pricing(
    inputs: Mapping[str, Any], context: StrategyWorkflowPreparationContext
) -> PreparedStrategyPlan:
    if context.dataset_id is None:
        raise StrategyWorkflowValidationError(
            "额度定价矩阵需要当前任务内唯一且已认证的数据集。",
            code="strategy_dataset_required",
        )
    if context.bind_sample_design is None:
        raise StrategyWorkflowValidationError(
            "额度定价矩阵需要当前数据与标签口径匹配的已认证 SampleDesign。",
            code="strategy_sample_design_required",
        )
    thawed = deep_thaw(inputs)
    strategy_id = thawed.get("strategy_id")
    if strategy_id is not None and context.validate_strategy_ref is not None:
        context.validate_strategy_ref(
            str(strategy_id),
            frozenset({"limit", "pricing"}),
        )
    sample_design_ref = context.bind_sample_design(True)
    slots = {
        "dataset_id": context.dataset_id,
        **thawed,
        "sample_design_ref": deep_thaw(sample_design_ref),
    }
    if context.drop_nan_labels:
        slots["drop_nan_labels"] = True
    return PreparedStrategyPlan(
        workflow_id="limit_pricing_matrix",
        template_id="strategy_limit_pricing_analysis",
        slots=deep_freeze(slots),
    )


def _prepare_dataset_plan(
    workflow_id: str,
    template_id: str,
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    if context.dataset_id is None:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 需要当前任务内唯一且已认证的数据集。",
            code="strategy_dataset_required",
        )
    return PreparedStrategyPlan(
        workflow_id=workflow_id,
        template_id=template_id,
        slots=deep_freeze({"dataset_id": context.dataset_id, **deep_thaw(inputs)}),
    )


def _confirmation(details: list[str]) -> str:
    details.append(
        "请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。"
    )
    return "；".join(details)


def _reject_fields(
    inputs: Mapping[str, Any], allowed: set[str], *, workflow: str
) -> None:
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


def _column(value: object, *, name: str, whitelist: tuple[str, ...]) -> str:
    column = _required_text(value, name=name)
    if column not in whitelist:
        raise StrategyWorkflowValidationError(
            f"{name} 使用了数据集中不存在的列「{column}」。"
        )
    return column


def _bounded_number(
    value: object, *, name: str, maximum: float | None = None
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


def _number_sequence(
    value: object,
    *,
    name: str,
    minimum: float | None,
    maximum_items: int,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    minimum_items: int = 1,
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or not minimum_items <= len(value) <= maximum_items
    ):
        raise StrategyWorkflowValidationError(
            f"{name} 必须是包含 {minimum_items} 到 {maximum_items} 个有限数字的数组。"
        )
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise StrategyWorkflowValidationError(f"{name} 只能包含有限数字。")
        number = float(item)
        if not math.isfinite(number):
            raise StrategyWorkflowValidationError(f"{name} 只能包含有限数字。")
        if minimum is not None and (
            number < minimum or (exclusive_minimum and number == minimum)
        ):
            relation = "大于" if exclusive_minimum else "大于等于"
            raise StrategyWorkflowValidationError(
                f"{name} 中每个值都必须{relation} {minimum:g}。"
            )
        if maximum is not None and number > maximum:
            raise StrategyWorkflowValidationError(
                f"{name} 中每个值都必须小于等于 {maximum:g}。"
            )
        numbers.append(number)
    if len(set(numbers)) != len(numbers):
        raise StrategyWorkflowValidationError(f"{name} 不能包含重复值。")
    return numbers

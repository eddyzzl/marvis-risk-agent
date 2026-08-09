from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any

from .contracts import (
    PreparedStrategyPlan,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_freeze,
    deep_thaw,
)


UNIVARIATE_ANALYSIS_WORKFLOW_ID = "univariate_candidate_analysis"
UNIVARIATE_REFINEMENT_WORKFLOW_ID = "univariate_candidate_refinement"
CANDIDATE_STABILITY_WORKFLOW_ID = "candidate_monthly_stability"
SCORECARD_EVIDENCE_WORKFLOW_ID = "scorecard_model_score_evidence_build"
SCORECARD_BAND_WORKFLOW_ID = "scorecard_band_build"
SCORECARD_CUTOFF_WORKFLOW_ID = "scorecard_cutoff_selection"

UNIVARIATE_BINNING_METHODS = (
    "equal_frequency",
    "equal_width",
    "chimerge",
    "tree",
    "manual",
)
UNIVARIATE_REFINEMENT_METHODS = (*UNIVARIATE_BINNING_METHODS, "categorical")
STRATEGY_TYPES = (
    "approval",
    "reject",
    "limit",
    "pricing",
    "segmentation",
)

_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_CANDIDATE_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_POOL_ENTRY_ID_RE = re.compile(r"^pool-entry-[0-9a-f]{32}$")
_SCORECARD_BAND_ASSET_ID_RE = re.compile(
    r"^scorecard-band-asset-[0-9a-f]{32}$"
)
_SCORECARD_CUTOFF_ID_RE = re.compile(r"^scorecard-cutoff-[0-9a-f]{32}$")

_DATA_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=True,
    target=True,
    complete_labels=True,
)
_NO_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=False,
    target=False,
    complete_labels=False,
)


def univariate_refinement_requirements(
    inputs: Mapping[str, Any],
) -> StrategyWorkflowRequirements:
    """Preserve the legacy fresh-vs-existing source dependency boundary."""

    required = {"feature", "method", "selection"}
    if not isinstance(inputs, Mapping) or not required <= set(inputs):
        raise RuntimeError(
            "univariate_candidate_refinement requirements need normalized inputs"
        )
    return _NO_REQUIREMENTS if "source_candidate_id" in inputs else _DATA_REQUIREMENTS


def validate_univariate_analysis_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
    *,
    manual_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    allowed = {
        "features",
        "methods",
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
        "manual_breakpoints",
    }
    workflow = UNIVARIATE_ANALYSIS_WORKFLOW_ID
    _reject_fields(inputs, allowed, workflow=workflow)

    raw_features = inputs.get("features", [])
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or len(raw_features) > 50
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 必须是最多 50 个字段的有序数组。"
        )
    features = [
        _column(
            value,
            name=f"{workflow} features",
            whitelist=context.allowed_columns,
        )
        for value in raw_features
    ]
    if len(features) != len(set(features)):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含重复字段。"
        )
    if context.target_col is not None and context.target_col in features:
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含目标列 {context.target_col}。"
        )

    methods_supplied = "methods" in inputs
    raw_methods = inputs.get("methods", [])
    if (
        not isinstance(raw_methods, Sequence)
        or isinstance(raw_methods, str | bytes | bytearray)
        or (
            methods_supplied
            and not 1 <= len(raw_methods) <= len(UNIVARIATE_BINNING_METHODS)
        )
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} methods 必须包含 1 到 "
            f"{len(UNIVARIATE_BINNING_METHODS)} 个分箱方法。"
        )
    methods = [
        _required_text(value, name=f"{workflow} methods")
        for value in raw_methods
    ]
    unknown_methods = sorted(set(methods) - set(UNIVARIATE_BINNING_METHODS))
    if unknown_methods:
        raise StrategyWorkflowValidationError(
            f"{workflow} 不支持分箱方法：" + "、".join(unknown_methods) + "。"
        )
    if len(methods) != len(set(methods)):
        raise StrategyWorkflowValidationError(
            f"{workflow} methods 不能包含重复方法。"
        )

    bin_count = inputs.get("bin_count", 10)
    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, int)
        or not 3 <= bin_count <= 20
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} bin_count 必须是 3 到 20 的整数。"
        )
    min_bin_pct = _bounded_number(
        inputs.get("min_bin_pct", 0.02),
        name=f"{workflow} min_bin_pct",
        maximum=0.5,
    )
    normalized: dict[str, Any] = {
        "features": features,
        "methods": methods,
        "bin_count": bin_count,
        "min_bin_pct": min_bin_pct,
        "sentinel_values": _sentinel_sequence(
            inputs.get("sentinel_values", []),
            name=f"{workflow} sentinel_values",
        ),
    }
    expected_manual_features = (
        features if manual_features is None else list(manual_features)
    )
    manual_breakpoints = _validate_manual_breakpoint_mapping(
        inputs.get("manual_breakpoints"),
        manual_requested="manual" in methods,
        expected_features=expected_manual_features,
        workflow=workflow,
    )
    if manual_breakpoints:
        normalized["manual_breakpoints"] = manual_breakpoints
    for field in ("loan_amount_col", "overdue_amount_col"):
        if field in inputs:
            normalized[field] = _column(
                inputs[field],
                name=f"{workflow} {field}",
                whitelist=context.allowed_columns,
            )
            if (
                context.target_col is not None
                and normalized[field] == context.target_col
            ):
                raise StrategyWorkflowValidationError(
                    f"{workflow} {field} 不能使用目标列。"
                )
    if normalized.get("loan_amount_col") is not None and normalized.get(
        "loan_amount_col"
    ) == normalized.get("overdue_amount_col"):
        raise StrategyWorkflowValidationError(
            f"{workflow} loan_amount_col 与 overdue_amount_col 必须是不同字段。"
        )
    return normalized


def validate_univariate_refinement_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = UNIVARIATE_REFINEMENT_WORKFLOW_ID
    analysis_fields = {
        "features",
        "methods",
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
        "manual_breakpoints",
    }
    allowed = analysis_fields | {
        "feature",
        "method",
        "merge_groups",
        "selection",
        "selection_reason",
        "source_candidate_id",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"feature", "method", "selection"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    source_candidate_id = None
    if "source_candidate_id" in inputs:
        source_candidate_id = _required_text(
            inputs["source_candidate_id"],
            name=f"{workflow} source_candidate_id",
        )
        if _CANDIDATE_ID_RE.fullmatch(source_candidate_id) is None:
            raise StrategyWorkflowValidationError(
                f"{workflow} source_candidate_id 必须是完整 candidate id。"
            )
        ignored_analysis_fields = sorted(set(inputs) & analysis_fields)
        if ignored_analysis_fields:
            raise StrategyWorkflowValidationError(
                f"{workflow} 已绑定已有 candidate 时不能重设分析参数："
                + "、".join(ignored_analysis_fields)
                + "。"
            )

    feature = (
        _required_text(inputs["feature"], name=f"{workflow} feature")
        if source_candidate_id is not None
        else _column(
            inputs["feature"],
            name=f"{workflow} feature",
            whitelist=context.allowed_columns,
        )
    )
    if context.target_col is not None and feature == context.target_col:
        raise StrategyWorkflowValidationError(
            f"{workflow} feature 不能使用目标列。"
        )
    method = _required_text(inputs["method"], name=f"{workflow} method")
    if method not in UNIVARIATE_REFINEMENT_METHODS:
        raise StrategyWorkflowValidationError(
            f"{workflow} 不支持分箱方法 {method}；可选值为："
            + "、".join(UNIVARIATE_REFINEMENT_METHODS)
            + "。"
        )

    analysis: dict[str, Any] = {}
    if source_candidate_id is None:
        analysis_inputs = {
            field: inputs[field] for field in analysis_fields if field in inputs
        }
        if "features" not in analysis_inputs:
            analysis_inputs["features"] = [feature]
        if "methods" not in analysis_inputs and method != "categorical":
            analysis_inputs["methods"] = [method]
        analysis = validate_univariate_analysis_inputs(
            analysis_inputs,
            context,
        )
        if feature not in analysis["features"]:
            raise StrategyWorkflowValidationError(
                f"{workflow} feature 必须包含在本次候选字段 features 中。"
            )
        if method != "categorical" and method not in analysis["methods"]:
            raise StrategyWorkflowValidationError(
                f"{workflow} method 必须包含在本次数值分箱方法 methods 中。"
            )

    merge_groups = _candidate_merge_groups(
        inputs.get("merge_groups", []),
        name=f"{workflow} merge_groups",
    )
    selection = _candidate_selection(
        inputs["selection"],
        name=f"{workflow} selection",
    )
    uses_source_bin_ids = "source_bin_ids" in selection or bool(merge_groups)
    if uses_source_bin_ids and source_candidate_id is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} 使用 source bin id 时必须提供用户已查看证据的 "
            "source_candidate_id，不能重新分析后猜测绑定。"
        )
    normalized = {
        **analysis,
        "feature": feature,
        "method": method,
        "merge_groups": merge_groups,
        "selection": selection,
    }
    if source_candidate_id is not None:
        normalized["source_candidate_id"] = source_candidate_id
    if "selection_reason" in inputs:
        reason = _required_text(
            inputs["selection_reason"],
            name=f"{workflow} selection_reason",
        )
        if len(reason) > 500:
            raise StrategyWorkflowValidationError(
                f"{workflow} selection_reason 最多 500 个字符。"
            )
        normalized["selection_reason"] = reason
    return normalized


def validate_candidate_monthly_stability_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    """Validate only the user's immutable source pointer."""

    workflow = CANDIDATE_STABILITY_WORKFLOW_ID
    if any(not isinstance(key, str) for key in inputs):
        raise StrategyWorkflowValidationError(f"{workflow} 字段名必须是文本。")
    fields = set(inputs)
    if fields == {"asset_id"}:
        asset_id = _required_text(
            inputs["asset_id"],
            name=f"{workflow} asset_id",
        )
        if _CANDIDATE_ASSET_ID_RE.fullmatch(asset_id) is None:
            raise StrategyWorkflowValidationError(
                f"{workflow} asset_id 必须是 candidate-asset- 后接 "
                "32 位小写十六进制字符。"
            )
        return {"asset_id": asset_id}
    if fields == {"strategy_type", "entry_id"}:
        strategy_type = _required_text(
            inputs["strategy_type"],
            name=f"{workflow} strategy_type",
        )
        if strategy_type not in STRATEGY_TYPES:
            raise StrategyWorkflowValidationError(
                f"{workflow} strategy_type 只能是："
                + "、".join(STRATEGY_TYPES)
                + "。"
            )
        entry_id = _required_text(
            inputs["entry_id"],
            name=f"{workflow} entry_id",
        )
        if _POOL_ENTRY_ID_RE.fullmatch(entry_id) is None:
            raise StrategyWorkflowValidationError(
                f"{workflow} entry_id 必须是 pool-entry- 后接 "
                "32 位小写十六进制字符。"
            )
        return {
            "strategy_type": strategy_type,
            "entry_id": entry_id,
        }
    platform_fields = sorted(
        fields
        & {
            "source_kind",
            "source_artifact_id",
            "expected_artifact_content_hash",
            "expected_asset_id",
            "expected_asset_hash",
            "expected_pool_revision",
            "expected_pool_snapshot_hash",
            "dataset_id",
            "expected_dataset_content_hash",
            "workspace_revision",
            "workspace_generation",
            "analysis_generation",
            "semantic_mapping_hash",
            "sample_design_ref",
            "target_col",
            "month_col",
            "metrics",
            "psi",
        }
    )
    if platform_fields:
        raise StrategyWorkflowValidationError(
            f"{workflow} 包含平台拥有的字段："
            + "、".join(platform_fields)
            + "；这些字段必须由 preflight 恢复。",
            code="candidate_monthly_stability_platform_binding_forbidden",
            fields=platform_fields,
        )
    raise StrategyWorkflowValidationError(
        f"{workflow} 必须且只能提供一个完整 asset_id，"
        "或同时提供明确 strategy_type 与一个完整 entry_id。",
        code="candidate_monthly_stability_source_required",
        fields=("asset_id", "strategy_type", "entry_id"),
    )


def validate_scorecard_model_score_evidence_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = SCORECARD_EVIDENCE_WORKFLOW_ID
    allowed = {
        "features",
        "sample_weight_col",
        "seed",
        "max_iter",
        "scorecard_max_bins",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted(
        {"features", "seed", "max_iter", "scorecard_max_bins"} - set(inputs)
    )
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 1 <= len(raw_features) <= 50
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 必须是包含 1 到 50 个字段的有序数组。"
        )
    features = [
        _column(
            value,
            name=f"{workflow} features",
            whitelist=context.allowed_columns,
        )
        for value in raw_features
    ]
    if len(features) != len(set(features)):
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含重复字段。"
        )
    if context.target_col is not None and context.target_col in features:
        raise StrategyWorkflowValidationError(
            f"{workflow} features 不能包含目标列 {context.target_col}。"
        )

    seed = inputs["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 4_294_967_295
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} seed 必须是 0 到 4294967295 的整数。"
        )
    max_iter = inputs["max_iter"]
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, int)
        or not 20 <= max_iter <= 5_000
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} max_iter 必须是 20 到 5000 的整数。"
        )
    max_bins = inputs["scorecard_max_bins"]
    if (
        isinstance(max_bins, bool)
        or not isinstance(max_bins, int)
        or not 2 <= max_bins <= 20
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} scorecard_max_bins 必须是 2 到 20 的整数。"
        )

    normalized: dict[str, Any] = {
        "features": features,
        "seed": seed,
        "max_iter": max_iter,
        "scorecard_max_bins": max_bins,
    }
    if "sample_weight_col" in inputs:
        weight_col = _column(
            inputs["sample_weight_col"],
            name=f"{workflow} sample_weight_col",
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and weight_col == context.target_col:
            raise StrategyWorkflowValidationError(
                f"{workflow} sample_weight_col 不能使用目标列。"
            )
        if weight_col in features:
            raise StrategyWorkflowValidationError(
                f"{workflow} sample_weight_col 不能同时作为建模特征。"
            )
        normalized["sample_weight_col"] = weight_col
    return normalized


def validate_scorecard_band_build_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = SCORECARD_BAND_WORKFLOW_ID
    allowed = {"bin_count", "raw_pd_band_edges"}
    _reject_fields(inputs, allowed, workflow=workflow)
    if set(inputs) == allowed:
        raise StrategyWorkflowValidationError(
            f"{workflow} bin_count 与 raw_pd_band_edges 必须二选一；"
            "也可均省略以使用平台默认等频 10 档。"
        )
    if "bin_count" in inputs:
        value = inputs["bin_count"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 2 <= value <= 20
        ):
            raise StrategyWorkflowValidationError(
                f"{workflow} bin_count 必须是 2 到 20 的整数。"
            )
        return {"bin_count": value}
    if "raw_pd_band_edges" not in inputs:
        return {}
    raw = inputs["raw_pd_band_edges"]
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, str | bytes | bytearray)
        or not 3 <= len(raw) <= 21
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} raw_pd_band_edges 必须包含 3 到 21 个数字。"
        )
    edges: list[float] = []
    for value in raw:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise StrategyWorkflowValidationError(
                f"{workflow} raw_pd_band_edges 只能包含有限数字。"
            )
        edges.append(float(value))
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise StrategyWorkflowValidationError(
            f"{workflow} raw_pd_band_edges 必须从 0.0 开始并以 1.0 结束。"
        )
    if any(
        left >= right for left, right in zip(edges, edges[1:], strict=False)
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} raw_pd_band_edges 必须严格递增。"
        )
    return {"raw_pd_band_edges": edges}


def validate_scorecard_cutoff_selection_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = SCORECARD_CUTOFF_WORKFLOW_ID
    allowed = {"asset_id", "cutoff_id", "reason"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"asset_id", "cutoff_id"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    asset_id = _required_text(inputs["asset_id"], name=f"{workflow} asset_id")
    cutoff_id = _required_text(
        inputs["cutoff_id"],
        name=f"{workflow} cutoff_id",
    )
    if _SCORECARD_BAND_ASSET_ID_RE.fullmatch(asset_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} asset_id 必须是 scorecard-band-asset- 后接 "
            "32 位小写十六进制字符。"
        )
    if _SCORECARD_CUTOFF_ID_RE.fullmatch(cutoff_id) is None:
        raise StrategyWorkflowValidationError(
            f"{workflow} cutoff_id 必须是 scorecard-cutoff- 后接 "
            "32 位小写十六进制字符。"
        )
    normalized = {"asset_id": asset_id, "cutoff_id": cutoff_id}
    if "reason" in inputs:
        reason = _required_text(inputs["reason"], name=f"{workflow} reason")
        if len(reason) > 500:
            raise StrategyWorkflowValidationError(
                f"{workflow} reason 最多 500 个字符。"
            )
        normalized["reason"] = reason
    return normalized


def univariate_analysis_confirmation(inputs: Mapping[str, Any]) -> str:
    feature_text = (
        "、".join(inputs["features"])
        if inputs["features"]
        else "当前语义映射中的全部候选字段"
    )
    method_text = (
        "数值字段自动比较等频、等距、ChiMerge、决策树；类别字段使用等值箱"
        if not inputs["methods"]
        else "、".join(inputs["methods"]) + "；类别字段仍使用等值箱"
    )
    details = [
        "已识别为〔单变量候选分析 Workflow〕",
        f"候选字段：{feature_text}",
        "分箱方法：" + method_text,
        f"目标箱数 {inputs['bin_count']}，最小箱占比 {inputs['min_bin_pct']:.2%}",
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
            "手工切点："
            + "；".join(
                feature
                + "=["
                + "、".join(f"{value:g}" for value in points)
                + "]"
                for feature, points in inputs["manual_breakpoints"].items()
            )
        )
    details.append("只生成 development/unvalidated 候选证据，不冒充独立验证结果")
    return "；".join(details)


def univariate_refinement_confirmation(inputs: Mapping[str, Any]) -> str:
    merge_text = (
        "；".join(" + ".join(group) for group in inputs["merge_groups"])
        if inputs["merge_groups"]
        else "不合并，保留原分箱"
    )
    if "source_bin_ids" in inputs["selection"]:
        selection_text = "显式选择 " + "、".join(
            inputs["selection"]["source_bin_ids"]
        )
    else:
        threshold = inputs["selection"]["risk_threshold"]
        selection_text = (
            f"按观测坏率 {threshold['operator']} "
            f"{threshold['value']:.2%} 确定性选择"
        )
    details = [
        "已识别为〔单变量候选选择与合并 Workflow〕",
        f"候选字段与方法：{inputs['feature']} / {inputs['method']}",
        f"分箱合并：{merge_text}",
        f"候选选择：{selection_text}",
        "平台会从任务自有证据重放样本并重新计算全部指标",
        "只生成 development/unvalidated 候选资产，不代表独立验证、采纳或上线",
    ]
    if "selection_reason" in inputs:
        details.append(f"选择说明：{inputs['selection_reason']}")
    if "manual_breakpoints" in inputs:
        details.append(
            "手工切点："
            + "；".join(
                feature
                + "=["
                + "、".join(f"{value:g}" for value in points)
                + "]"
                for feature, points in inputs["manual_breakpoints"].items()
            )
        )
    return "；".join(details)


def candidate_monthly_stability_confirmation(inputs: Mapping[str, Any]) -> str:
    details = ["已识别为〔候选逐月稳定性 Workflow〕"]
    if "asset_id" in inputs:
        details.extend(
            [
                f"来源：已有单变量候选资产 {inputs['asset_id']}",
                "统计口径：该候选规则命中/未命中的逐月分布与 PSI",
            ]
        )
    else:
        details.extend(
            [
                (
                    "来源：当前 "
                    f"{inputs['strategy_type']} Strategy Pool 条目 "
                    f"{inputs['entry_id']}"
                ),
                "统计口径：该条目按当前 Pool 精确顺序的增量首次命中/未命中分布与 PSI",
            ]
        )
    details.extend(
        [
            "平台将在计划创建前恢复并认证 artifact/Pool CAS、活动 workspace、"
            "成熟 development SampleDesign 与月份字段",
            "基准固定为完整 development 样本；每个 YYYYMM 与同一基准比较，"
            "不会滚动改基准",
            "本步骤只生成只读稳定性证据；不会修改 Pool、入池、采纳或部署",
        ]
    )
    return "；".join(details)


def scorecard_model_score_evidence_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    details = [
        "已识别为〔Scorecard 训练与模型评分证据 Workflow〕",
        "建模特征：" + "、".join(inputs["features"]),
        f"随机种子：{inputs['seed']}",
        f"最大迭代次数：{inputs['max_iter']}",
        f"Scorecard 最大分箱数：{inputs['scorecard_max_bins']}",
        "平台将绑定最新完整认证的 StrategySampleDesign V2，先训练原生"
        " Scorecard，再用同一训练证据生成完整任务级原始坏账概率向量",
        "模型、训练证据、评分向量与评分证据均由平台发布并逐项校验",
        "本步骤不比较、不选择、不采纳、不部署模型，也不自动选择 cutoff",
    ]
    if "sample_weight_col" in inputs:
        details.append(f"样本权重列：{inputs['sample_weight_col']}")
    return "；".join(details)


def scorecard_band_build_confirmation(inputs: Mapping[str, Any]) -> str:
    if "bin_count" in inputs:
        banding = f"等频 {inputs['bin_count']} 档"
    elif "raw_pd_band_edges" in inputs:
        banding = (
            "raw PD 边界 ["
            + "、".join(f"{value:g}" for value in inputs["raw_pd_band_edges"])
            + "]"
        )
    else:
        banding = "省略用户分带参数，由受控 Tool 使用默认等频 10 档"
    return "；".join(
        [
            "已识别为〔Scorecard 完整分数带 Workflow〕",
            f"分带方式：{banding}",
            "平台将绑定最新精确兼容且完整认证的 ScoreEvidence、原始分数向量"
            "和 StrategySampleDesign V2；损坏的最新证据不会回退",
            "本步骤只生成全部分带及全部可选 cutoff 证据；不会自动选择、"
            "排名或推荐 cutoff",
            "不会入池、应用、采纳或部署",
        ]
    )


def scorecard_cutoff_selection_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔Scorecard cutoff 精确选择 Workflow〕",
        f"完整分数带资产 pointer：{inputs['asset_id']}",
        f"精确 cutoff pointer：{inputs['cutoff_id']}",
        "平台将从当前任务严格恢复 source artifact/hash 与完整分数带；"
        "不会自动排名或推荐",
        "本步骤只物化 pointer，不复制全部分带；不会入池、应用、采纳或部署",
    ]
    if "reason" in inputs:
        details.append(f"用户原话选择说明：{inputs['reason']}")
    return "；".join(details)


def prepare_univariate_analysis(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    slots.update(_workflow_evidence(UNIVARIATE_ANALYSIS_WORKFLOW_ID, inputs, context, slots))
    if context.drop_nan_labels:
        slots["drop_nan_labels"] = True
    return PreparedStrategyPlan(
        workflow_id=UNIVARIATE_ANALYSIS_WORKFLOW_ID,
        template_id="strategy_univariate_candidate_analysis",
        slots=deep_freeze(slots),
    )


def prepare_univariate_refinement(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(inputs)
    assert isinstance(slots, dict)
    source_candidate_id = slots.pop("source_candidate_id", None)
    slots.update(
        _workflow_evidence(
            UNIVARIATE_REFINEMENT_WORKFLOW_ID,
            inputs,
            context,
            slots,
        )
    )
    existing = source_candidate_id is not None
    if context.drop_nan_labels and not existing:
        slots["drop_nan_labels"] = True
    return PreparedStrategyPlan(
        workflow_id=UNIVARIATE_REFINEMENT_WORKFLOW_ID,
        template_id=(
            "strategy_univariate_candidate_refinement_existing"
            if existing
            else "strategy_univariate_candidate_refinement"
        ),
        slots=deep_freeze(slots),
    )


def prepare_candidate_monthly_stability(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    if "asset_id" in inputs:
        slots: dict[str, Any] = {
            "source_kind": "univariate_asset",
        }
    else:
        slots = {
            "source_kind": "pool_entry",
            "strategy_type": inputs["strategy_type"],
            "entry_id": inputs["entry_id"],
        }
    slots.update(
        _workflow_evidence(CANDIDATE_STABILITY_WORKFLOW_ID, inputs, context, slots)
    )
    return PreparedStrategyPlan(
        workflow_id=CANDIDATE_STABILITY_WORKFLOW_ID,
        template_id="strategy_candidate_monthly_stability",
        slots=deep_freeze(slots),
    )


def prepare_scorecard_model_score_evidence(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    params: dict[str, Any] = {
        "max_iter": inputs["max_iter"],
        "scorecard_max_bins": inputs["scorecard_max_bins"],
    }
    if "sample_weight_col" in inputs:
        params["sample_weight_col"] = inputs["sample_weight_col"]
    slots: dict[str, Any] = {
        "features": deep_thaw(inputs["features"]),
        "params": params,
        "seed": inputs["seed"],
    }
    slots.update(
        _workflow_evidence(SCORECARD_EVIDENCE_WORKFLOW_ID, inputs, context, slots)
    )
    return PreparedStrategyPlan(
        workflow_id=SCORECARD_EVIDENCE_WORKFLOW_ID,
        template_id="strategy_scorecard_model_score_evidence_build",
        slots=deep_freeze(slots),
    )


def prepare_scorecard_band_build(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots: dict[str, Any] = {}
    if "bin_count" in inputs:
        slots["banding"] = {
            "method": "equal_frequency",
            "bin_count": inputs["bin_count"],
        }
    elif "raw_pd_band_edges" in inputs:
        slots["raw_pd_band_edges"] = deep_thaw(inputs["raw_pd_band_edges"])
    slots.update(
        _workflow_evidence(SCORECARD_BAND_WORKFLOW_ID, inputs, context, slots)
    )
    return PreparedStrategyPlan(
        workflow_id=SCORECARD_BAND_WORKFLOW_ID,
        template_id="strategy_scorecard_band_build",
        slots=deep_freeze(slots),
    )


def prepare_scorecard_cutoff_selection(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots: dict[str, Any] = {"cutoff_id": inputs["cutoff_id"]}
    if "reason" in inputs:
        slots["reason"] = inputs["reason"]
    slots.update(
        _workflow_evidence(SCORECARD_CUTOFF_WORKFLOW_ID, inputs, context, slots)
    )
    return PreparedStrategyPlan(
        workflow_id=SCORECARD_CUTOFF_WORKFLOW_ID,
        template_id="strategy_scorecard_cutoff_selection",
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


def _validate_manual_breakpoint_mapping(
    value: object,
    *,
    manual_requested: bool,
    expected_features: Sequence[str],
    workflow: str,
) -> dict[str, list[float]]:
    if value is None:
        if manual_requested:
            raise StrategyWorkflowValidationError(
                f"{workflow} manual 分箱必须提供 manual_breakpoints。"
            )
        return {}
    if not manual_requested:
        raise StrategyWorkflowValidationError(
            f"{workflow} 只有选择 manual 分箱时才能提供 manual_breakpoints。"
        )
    if not isinstance(value, Mapping) or not value:
        raise StrategyWorkflowValidationError(
            f"{workflow} manual_breakpoints 必须是非空字段到切点数组映射。"
        )
    expected = tuple(expected_features)
    if not expected or set(value) != set(expected) or len(value) != len(expected):
        raise StrategyWorkflowValidationError(
            f"{workflow} manual_breakpoints 必须且只能覆盖 manual 轴/字段："
            + "、".join(expected)
            + "。"
        )
    normalized: dict[str, list[float]] = {}
    for feature in expected:
        raw_points = value[feature]
        if (
            not isinstance(raw_points, Sequence)
            or isinstance(raw_points, str | bytes | bytearray)
            or not 1 <= len(raw_points) <= 19
        ):
            raise StrategyWorkflowValidationError(
                f"{workflow} manual_breakpoints.{feature} 必须包含 1 到 19 个切点。"
            )
        points: list[float] = []
        for item in raw_points:
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise StrategyWorkflowValidationError(
                    f"{workflow} manual_breakpoints.{feature} 只能包含有限数字。"
                )
            if isinstance(item, int) and abs(item) > 2**53 - 1:
                raise StrategyWorkflowValidationError(
                    f"{workflow} manual_breakpoints.{feature} 超出精确 JSON 范围。"
                )
            number = float(item)
            if not math.isfinite(number):
                raise StrategyWorkflowValidationError(
                    f"{workflow} manual_breakpoints.{feature} 只能包含有限数字。"
                )
            points.append(number)
        if any(
            left >= right
            for left, right in zip(points, points[1:], strict=False)
        ):
            raise StrategyWorkflowValidationError(
                f"{workflow} manual_breakpoints.{feature} 必须严格递增且不重复。"
            )
        normalized[feature] = points
    return normalized


def _sentinel_sequence(value: object, *, name: str) -> list[str | int | float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or len(value) > 20
    ):
        raise StrategyWorkflowValidationError(
            f"{name} 必须是最多 20 个文本或有限数字的数组。"
        )
    normalized: list[str | int | float] = []
    identities: set[str] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, str | int | float):
            raise StrategyWorkflowValidationError(
                f"{name} 只能包含文本或有限数字。"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise StrategyWorkflowValidationError(
                f"{name} 只能包含文本或有限数字。"
            )
        if isinstance(item, int) and abs(item) > 2**53 - 1:
            raise StrategyWorkflowValidationError(
                f"{name} 中的整数超出精确 JSON 范围。"
            )
        identity = json.dumps(
            [type(item).__name__, item],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if identity in identities:
            raise StrategyWorkflowValidationError(f"{name} 不能包含重复值。")
        identities.add(identity)
        normalized.append(item)
    return normalized


def _candidate_merge_groups(value: object, *, name: str) -> list[list[str]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or len(value) > 20
    ):
        raise StrategyWorkflowValidationError(
            f"{name} 必须是最多 20 个合并组的数组。"
        )
    normalized: list[list[str]] = []
    seen: set[str] = set()
    for group_index, raw_group in enumerate(value):
        if (
            not isinstance(raw_group, Sequence)
            or isinstance(raw_group, str | bytes | bytearray)
            or not 2 <= len(raw_group) <= 20
        ):
            raise StrategyWorkflowValidationError(
                f"{name}[{group_index}] 必须包含 2 到 20 个 source bin id。"
            )
        group: list[str] = []
        for raw_bin_id in raw_group:
            bin_id = _required_text(
                raw_bin_id,
                name=f"{name}[{group_index}]",
            )
            if len(bin_id) > 128:
                raise StrategyWorkflowValidationError(
                    f"{name} 中的 bin id 最多 128 个字符。"
                )
            if bin_id in seen:
                raise StrategyWorkflowValidationError(
                    f"{name} 不能重复使用 bin id {bin_id}。"
                )
            seen.add(bin_id)
            group.append(bin_id)
        normalized.append(group)
    return normalized


def _candidate_selection(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StrategyWorkflowValidationError(f"{name} 必须是对象。")
    keys = set(value)
    if keys not in ({"source_bin_ids"}, {"risk_threshold"}):
        raise StrategyWorkflowValidationError(
            f"{name} 必须在 source_bin_ids 与 risk_threshold 中严格二选一。"
        )
    if "source_bin_ids" in value:
        raw_ids = value["source_bin_ids"]
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, str | bytes | bytearray)
            or not 1 <= len(raw_ids) <= 50
        ):
            raise StrategyWorkflowValidationError(
                f"{name}.source_bin_ids 必须包含 1 到 50 个 source bin id。"
            )
        bin_ids = [
            _required_text(item, name=f"{name}.source_bin_ids")
            for item in raw_ids
        ]
        if any(len(bin_id) > 128 for bin_id in bin_ids):
            raise StrategyWorkflowValidationError(
                f"{name}.source_bin_ids 中每个值最多 128 个字符。"
            )
        if len(bin_ids) != len(set(bin_ids)):
            raise StrategyWorkflowValidationError(
                f"{name}.source_bin_ids 不能包含重复值。"
            )
        return {"source_bin_ids": bin_ids}

    threshold = value["risk_threshold"]
    if not isinstance(threshold, Mapping) or any(
        not isinstance(key, str) for key in threshold
    ):
        raise StrategyWorkflowValidationError(
            f"{name}.risk_threshold 必须是对象。"
        )
    if set(threshold) != {"operator", "value"}:
        raise StrategyWorkflowValidationError(
            f"{name}.risk_threshold 只能包含 operator 和 value。"
        )
    operator = _required_text(
        threshold["operator"],
        name=f"{name}.risk_threshold.operator",
    )
    if operator not in {">=", ">", "<=", "<"}:
        raise StrategyWorkflowValidationError(
            f"{name}.risk_threshold.operator 只能是 >=、>、<=、<。"
        )
    risk_value = _bounded_number(
        threshold["value"],
        name=f"{name}.risk_threshold.value",
        maximum=1.0,
    )
    return {"risk_threshold": {"operator": operator, "value": risk_value}}


__all__ = [
    "UNIVARIATE_BINNING_METHODS",
    "UNIVARIATE_REFINEMENT_METHODS",
    "candidate_monthly_stability_confirmation",
    "prepare_candidate_monthly_stability",
    "prepare_scorecard_band_build",
    "prepare_scorecard_cutoff_selection",
    "prepare_scorecard_model_score_evidence",
    "prepare_univariate_analysis",
    "prepare_univariate_refinement",
    "scorecard_band_build_confirmation",
    "scorecard_cutoff_selection_confirmation",
    "scorecard_model_score_evidence_confirmation",
    "univariate_analysis_confirmation",
    "univariate_refinement_confirmation",
    "univariate_refinement_requirements",
    "validate_candidate_monthly_stability_inputs",
    "validate_scorecard_band_build_inputs",
    "validate_scorecard_cutoff_selection_inputs",
    "validate_scorecard_model_score_evidence_inputs",
    "validate_univariate_analysis_inputs",
    "validate_univariate_refinement_inputs",
]

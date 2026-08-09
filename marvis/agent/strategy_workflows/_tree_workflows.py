"""Canonical adapters for the governed automatic/interactive tree workflows.

This module deliberately stops at plan preparation.  Platform-owned artifact,
workspace, SampleDesign, ancestry, node and current-projection facts enter only
through ``bind_workflow_evidence``; no Tool or legacy request handler is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any
import unicodedata

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


AUTOMATIC_TREE_BUILD_WORKFLOW_ID = "automatic_tree_candidate_build"
AUTOMATIC_TREE_APPLY_WORKFLOW_ID = "automatic_tree_apply"
AUTOMATIC_TREE_LEAF_WORKFLOW_ID = "automatic_tree_leaf_materialization"
INTERACTIVE_TREE_SPLIT_SEARCH_WORKFLOW_ID = "interactive_tree_split_search"
INTERACTIVE_TREE_CONTINUATION_WORKFLOW_ID = "interactive_tree_auto_continuation"
INTERACTIVE_TREE_REVISION_WORKFLOW_ID = "interactive_tree_revision"

AUTOMATIC_TREE_BUILD_TEMPLATE_ID = "strategy_automatic_tree_candidate_build"
AUTOMATIC_TREE_APPLY_TEMPLATE_ID = "strategy_automatic_tree_apply"
AUTOMATIC_TREE_LEAF_TEMPLATE_ID = "strategy_automatic_tree_leaf_materialization"
INTERACTIVE_TREE_SPLIT_SEARCH_TEMPLATE_ID = "strategy_interactive_tree_split_search"
INTERACTIVE_TREE_CONTINUATION_TEMPLATE_ID = (
    "strategy_interactive_tree_auto_continuation"
)
INTERACTIVE_TREE_REVISION_TEMPLATE_ID = "strategy_interactive_tree_revision"

AUTOMATIC_TREE_DIRECTIONS = (
    "increasing",
    "decreasing",
    "unordered",
)

_CANDIDATE_ASSET_ID_RE = re.compile(r"candidate-asset-[0-9a-f]{32}")
_AUTOMATIC_TREE_LEAF_ID_RE = re.compile(r"leaf-[0-9a-f]{20}")
_INTERACTIVE_TREE_SOURCE_ID_RE = re.compile(
    r"(?:candidate-asset|interactive-tree-revision)-[0-9a-f]{32}"
)
_INTERACTIVE_TREE_NODE_ID_RE = re.compile(r"node-[0-9a-f]{20}")
_INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE = re.compile(
    r"interactive-tree-split-search-[0-9a-f]{32}"
)
_INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE = re.compile(
    r"interactive-tree-split-candidate-[0-9a-f]{32}"
)
_APPLY_OUTPUT_COLUMN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

_BUILD_EVIDENCE_FIELDS = frozenset(
    {
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "target_col",
        "sample_design_ref",
    }
)
_APPLY_EVIDENCE_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
    }
)
_LEAF_EVIDENCE_FIELDS = frozenset(
    {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
    }
)

_DATA_TARGET_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=True,
    target=True,
    complete_labels=True,
)
_DATA_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=True,
    target=False,
    complete_labels=False,
)
_NO_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=False,
    target=False,
    complete_labels=False,
)


def validate_automatic_tree_candidate_build_inputs(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = AUTOMATIC_TREE_BUILD_WORKFLOW_ID
    allowed = {
        "features",
        "sample_weight_col",
        "directions",
        "max_depth",
        "min_leaf_count",
        "min_weight_fraction_leaf",
        "seed",
        "loan_amount_col",
        "overdue_amount_col",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    if "features" not in inputs:
        _invalid(workflow, "缺少必需字段 features。", "features")
    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 1 <= len(raw_features) <= 50
    ):
        _invalid(
            workflow,
            "features 必须是包含 1 到 50 个字段的有序数组。",
            "features",
        )
    features = [
        _column(
            value,
            name="features",
            workflow=workflow,
            whitelist=context.allowed_columns,
        )
        for value in raw_features
    ]
    if len(features) != len(set(features)):
        _invalid(workflow, "features 不能包含重复字段。", "features")
    if context.target_col is not None and context.target_col in features:
        _invalid(
            workflow,
            f"features 不能包含目标列 {context.target_col}。",
            "features",
        )

    normalized: dict[str, Any] = {"features": features}
    for field in ("sample_weight_col", "loan_amount_col", "overdue_amount_col"):
        if field not in inputs:
            continue
        column = _column(
            inputs[field],
            name=field,
            workflow=workflow,
            whitelist=context.allowed_columns,
        )
        if context.target_col is not None and column == context.target_col:
            _invalid(workflow, f"{field} 不能使用目标列。", field)
        normalized[field] = column

    if "directions" in inputs:
        raw_directions = inputs["directions"]
        if not isinstance(raw_directions, Mapping) or not raw_directions:
            _invalid(
                workflow,
                "directions 必须是至少包含一个特征方向的对象；"
                "没有风险方向诊断期望或检查时请省略该字段。",
                "directions",
            )
        if any(not isinstance(key, str) for key in raw_directions):
            _invalid(workflow, "directions 的字段名必须是文本。", "directions")
        unexpected_features = sorted(set(raw_directions) - set(features))
        if unexpected_features:
            _invalid(
                workflow,
                "directions 引用了未选择的特征："
                + "、".join(unexpected_features)
                + "。",
                "directions",
            )
        directions: dict[str, str] = {}
        for feature, value in raw_directions.items():
            if not isinstance(value, str) or value not in AUTOMATIC_TREE_DIRECTIONS:
                _invalid(
                    workflow,
                    f"directions.{feature} 只能是 increasing、decreasing 或 unordered。",
                    "directions",
                )
            directions[feature] = value
        normalized["directions"] = directions

    if "max_depth" in inputs:
        normalized["max_depth"] = _bounded_int(
            inputs["max_depth"],
            name="max_depth",
            workflow=workflow,
            minimum=1,
            maximum=8,
        )
    if "min_leaf_count" in inputs:
        value = inputs["min_leaf_count"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _invalid(workflow, "min_leaf_count 必须是正整数。", "min_leaf_count")
        normalized["min_leaf_count"] = value
    if "min_weight_fraction_leaf" in inputs:
        normalized["min_weight_fraction_leaf"] = _bounded_number(
            inputs["min_weight_fraction_leaf"],
            name="min_weight_fraction_leaf",
            workflow=workflow,
            maximum=0.5,
        )
    if "seed" in inputs:
        normalized["seed"] = _bounded_int(
            inputs["seed"],
            name="seed",
            workflow=workflow,
            minimum=0,
            maximum=4_294_967_295,
        )

    assigned_columns = [
        normalized[field]
        for field in ("sample_weight_col", "loan_amount_col", "overdue_amount_col")
        if field in normalized
    ]
    duplicate_roles = {
        column for column in assigned_columns if assigned_columns.count(column) > 1
    }
    feature_conflicts = set(features) & set(assigned_columns)
    if duplicate_roles or feature_conflicts:
        conflicts = sorted(duplicate_roles | feature_conflicts)
        raise StrategyWorkflowValidationError(
            f"{workflow} features、sample_weight_col、loan_amount_col 与 "
            "overdue_amount_col 必须使用不同字段：" + "、".join(conflicts) + "。",
            fields=(
                "features",
                "sample_weight_col",
                "loan_amount_col",
                "overdue_amount_col",
            ),
        )
    return normalized


def automatic_tree_candidate_build_confirmation(inputs: Mapping[str, Any]) -> str:
    direction_labels = {
        "increasing": "递增",
        "decreasing": "递减",
        "unordered": "无序",
    }
    details = [
        "已识别为〔自动决策树候选构建 Workflow〕",
        "候选特征：" + "、".join(inputs["features"]),
    ]
    if "sample_weight_col" in inputs:
        details.append(f"样本权重列 {inputs['sample_weight_col']}")
    if "directions" in inputs:
        details.append(
            "风险方向诊断期望："
            + "、".join(
                f"{feature}={direction_labels[direction]}"
                for feature, direction in inputs["directions"].items()
            )
        )
    if "max_depth" in inputs:
        details.append(f"最大深度 {inputs['max_depth']}")
    if "min_leaf_count" in inputs:
        details.append(f"最小叶样本数 {inputs['min_leaf_count']}")
    if "min_weight_fraction_leaf" in inputs:
        details.append(f"最小叶权重占比 {inputs['min_weight_fraction_leaf']:.2%}")
    if "seed" in inputs:
        details.append(f"随机种子 {inputs['seed']}")
    if "loan_amount_col" in inputs:
        details.append(f"放款金额列 {inputs['loan_amount_col']}")
    if "overdue_amount_col" in inputs:
        details.append(f"逾期金额列 {inputs['overdue_amount_col']}")
    details.extend(
        [
            "数据集、hash、workspace、目标列、标签处理和执行预算由平台绑定，"
            "LLM 不得填写",
            "本步骤只构建完整候选树及确定性证据；不会自动选择叶子、写入 "
            "Strategy Pool、采纳或部署",
            "平台不会给叶子生成“最佳”自动排名；后续操作必须由用户引用明确 leaf",
        ]
    )
    return "；".join(details)


def prepare_automatic_tree_candidate_build(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    _require_dataset_context(AUTOMATIC_TREE_BUILD_WORKFLOW_ID, context)
    slots = deep_thaw(deep_freeze(inputs))
    evidence = _workflow_evidence(
        AUTOMATIC_TREE_BUILD_WORKFLOW_ID,
        inputs,
        context,
        slots,
        required_fields=_BUILD_EVIDENCE_FIELDS,
    )
    _validate_evidence_values(
        AUTOMATIC_TREE_BUILD_WORKFLOW_ID,
        evidence,
        context=context,
        require_sample_design=True,
    )
    slots.update(evidence)
    if context.drop_nan_labels:
        slots["drop_nan_labels"] = True
    return _prepared(
        AUTOMATIC_TREE_BUILD_WORKFLOW_ID,
        AUTOMATIC_TREE_BUILD_TEMPLATE_ID,
        slots,
    )


def validate_automatic_tree_apply_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = AUTOMATIC_TREE_APPLY_WORKFLOW_ID
    allowed = {"tree_asset_id", "leaf_id_column", "rule_id_column"}
    _reject_fields(inputs, allowed, workflow=workflow)
    if "tree_asset_id" not in inputs:
        _invalid(workflow, "缺少字段：tree_asset_id。", "tree_asset_id")
    tree_asset_id = _required_text(
        inputs["tree_asset_id"], name="tree_asset_id", workflow=workflow
    )
    if _CANDIDATE_ASSET_ID_RE.fullmatch(tree_asset_id) is None:
        _invalid(
            workflow,
            "tree_asset_id 必须是完整的自动树 candidate asset id。",
            "tree_asset_id",
        )
    normalized: dict[str, Any] = {"tree_asset_id": tree_asset_id}
    for field in ("leaf_id_column", "rule_id_column"):
        if field not in inputs:
            continue
        value = _required_text(inputs[field], name=field, workflow=workflow)
        if _APPLY_OUTPUT_COLUMN_RE.fullmatch(value) is None:
            _invalid(
                workflow,
                f"{field} 必须是最多 64 位的 ASCII 标识符。",
                field,
            )
        normalized[field] = value
    if (
        "leaf_id_column" in normalized
        and "rule_id_column" in normalized
        and normalized["leaf_id_column"].casefold()
        == normalized["rule_id_column"].casefold()
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} leaf_id_column 与 rule_id_column 必须不同（忽略大小写）。",
            fields=("leaf_id_column", "rule_id_column"),
        )
    return normalized


def automatic_tree_apply_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔自动树全量写回 Workflow〕",
        f"完整树候选资产 pointer：{inputs['tree_asset_id']}",
        (
            f"叶节点输出列：{inputs['leaf_id_column']}"
            if "leaf_id_column" in inputs
            else "叶节点输出列：由受控 Tool 使用默认列名"
        ),
        (
            f"规则输出列：{inputs['rule_id_column']}"
            if "rule_id_column" in inputs
            else "规则输出列：由受控 Tool 使用默认列名"
        ),
        "平台将从当前任务重新校验 source artifact/hash、asset hash、"
        "原始数据集与 workspace lineage，LLM 不得填写或覆盖",
        "本步骤创建不可变派生数据集，但不会激活或替换当前 workspace",
        "结果仍是 development / unvalidated；不会入池、采纳或部署，也不生成业务动作",
    ]
    return "；".join(details)


def prepare_automatic_tree_apply(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    _require_dataset_context(AUTOMATIC_TREE_APPLY_WORKFLOW_ID, context)
    slots = {
        field: deep_thaw(deep_freeze(inputs[field]))
        for field in ("leaf_id_column", "rule_id_column")
        if field in inputs
    }
    evidence = _workflow_evidence(
        AUTOMATIC_TREE_APPLY_WORKFLOW_ID,
        inputs,
        context,
        slots,
        required_fields=_APPLY_EVIDENCE_FIELDS,
    )
    _validate_evidence_values(
        AUTOMATIC_TREE_APPLY_WORKFLOW_ID,
        evidence,
        context=context,
        expected_asset_id=inputs["tree_asset_id"],
    )
    slots.update(evidence)
    return _prepared(
        AUTOMATIC_TREE_APPLY_WORKFLOW_ID,
        AUTOMATIC_TREE_APPLY_TEMPLATE_ID,
        slots,
    )


def validate_automatic_tree_leaf_materialization_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = AUTOMATIC_TREE_LEAF_WORKFLOW_ID
    allowed = {"tree_asset_id", "leaf_id", "selection_reason"}
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"tree_asset_id", "leaf_id"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    tree_asset_id = _required_text(
        inputs["tree_asset_id"], name="tree_asset_id", workflow=workflow
    )
    if _CANDIDATE_ASSET_ID_RE.fullmatch(tree_asset_id) is None:
        _invalid(
            workflow,
            "tree_asset_id 必须是完整的自动树 candidate asset id。",
            "tree_asset_id",
        )
    leaf_id = _required_text(inputs["leaf_id"], name="leaf_id", workflow=workflow)
    if _AUTOMATIC_TREE_LEAF_ID_RE.fullmatch(leaf_id) is None:
        _invalid(
            workflow,
            "leaf_id 必须是 leaf- 后接 20 位小写十六进制字符。",
            "leaf_id",
        )
    normalized: dict[str, Any] = {
        "tree_asset_id": tree_asset_id,
        "leaf_id": leaf_id,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _canonical_reason(
            inputs["selection_reason"],
            name="selection_reason",
            workflow=workflow,
            maximum=None,
        )
    return normalized


def automatic_tree_leaf_materialization_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    details = [
        "已识别为〔自动树精确叶节点物化 Workflow〕",
        f"完整树候选资产 pointer：{inputs['tree_asset_id']}",
        f"精确叶节点 pointer：{inputs['leaf_id']}",
        "本步骤只创建指向完整树中该叶节点的不可变 pointer；"
        "不复制规则、条件、指标或业务动作",
        "不会加入 Strategy Pool，也不会采纳或部署策略",
    ]
    if "selection_reason" in inputs:
        details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    return "；".join(details)


def prepare_automatic_tree_leaf_materialization(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots: dict[str, Any] = {"leaf_id": inputs["leaf_id"]}
    if "selection_reason" in inputs:
        slots["selection_reason"] = inputs["selection_reason"]
    evidence = _workflow_evidence(
        AUTOMATIC_TREE_LEAF_WORKFLOW_ID,
        inputs,
        context,
        slots,
        required_fields=_LEAF_EVIDENCE_FIELDS,
    )
    _validate_evidence_values(
        AUTOMATIC_TREE_LEAF_WORKFLOW_ID,
        evidence,
        context=context,
        expected_asset_id=inputs["tree_asset_id"],
    )
    slots.update(evidence)
    return _prepared(
        AUTOMATIC_TREE_LEAF_WORKFLOW_ID,
        AUTOMATIC_TREE_LEAF_TEMPLATE_ID,
        slots,
    )


def validate_interactive_tree_split_search_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = INTERACTIVE_TREE_SPLIT_SEARCH_WORKFLOW_ID
    allowed = {
        "source_tree_id",
        "node_id",
        "mode",
        "features",
        "max_thresholds_per_feature",
        "max_row_evaluations",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    required = {
        "source_tree_id",
        "node_id",
        "mode",
        "max_thresholds_per_feature",
        "max_row_evaluations",
    }
    missing = sorted(required - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    source_tree_id = _source_tree_id(inputs["source_tree_id"], workflow=workflow)
    node_id = _node_id(inputs["node_id"], workflow=workflow)
    mode = _required_text(inputs["mode"], name="mode", workflow=workflow)
    if mode not in {"all_features", "selected_features"}:
        _invalid(
            workflow,
            "mode 只允许 all_features 或 selected_features。",
            "mode",
        )
    has_features = "features" in inputs
    if mode == "all_features" and has_features:
        _invalid(workflow, "all_features 不能提供 features。", "features")
    if mode == "selected_features" and not has_features:
        _invalid(workflow, "selected_features 必须提供 features。", "features")
    normalized: dict[str, Any] = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "mode": mode,
    }
    if has_features:
        raw_features = inputs["features"]
        if isinstance(raw_features, str | bytes | bytearray) or not isinstance(
            raw_features, Sequence
        ):
            _invalid(workflow, "features 必须是非空字符串数组。", "features")
        features = [
            _required_text(item, name="features", workflow=workflow)
            for item in raw_features
        ]
        if not features or len(features) > 50 or len(features) != len(set(features)):
            _invalid(
                workflow,
                "features 必须非空、唯一且不超过 50 个。",
                "features",
            )
        normalized["features"] = sorted(features)
    normalized["max_thresholds_per_feature"] = _bounded_int(
        inputs["max_thresholds_per_feature"],
        name="max_thresholds_per_feature",
        workflow=workflow,
        minimum=1,
        maximum=20,
    )
    normalized["max_row_evaluations"] = _bounded_int(
        inputs["max_row_evaluations"],
        name="max_row_evaluations",
        workflow=workflow,
        minimum=1,
        maximum=20_000_000,
    )
    return normalized


def interactive_tree_split_search_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔交互式树节点分裂候选搜索 Workflow〕",
        f"来源树 pointer：{inputs['source_tree_id']}",
        f"精确可见 node pointer：{inputs['node_id']}",
        (
            "搜索范围：认证树的全部特征"
            if inputs["mode"] == "all_features"
            else "搜索范围：用户明确的特征子集 " + "、".join(inputs["features"])
        ),
        f"每特征最大候选阈值数：{inputs['max_thresholds_per_feature']}",
        f"总行评估预算：{inputs['max_row_evaluations']}",
        "平台将认证完整树父链、数据集、workspace 与 SampleDesign，只"
        "持久化左右节点聚合风险证据，不输出客户明细",
        "排名只用于浏览；本步骤不会选择胜者、修改树、入池、应用、采纳或部署",
    ]
    return "；".join(details)


def prepare_interactive_tree_split_search(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    return _prepare_pointer_only_interactive(
        INTERACTIVE_TREE_SPLIT_SEARCH_WORKFLOW_ID,
        INTERACTIVE_TREE_SPLIT_SEARCH_TEMPLATE_ID,
        inputs,
        context,
    )


def validate_interactive_tree_auto_continuation_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = INTERACTIVE_TREE_CONTINUATION_WORKFLOW_ID
    allowed = {
        "search_id",
        "candidate_id",
        "max_additional_depth",
        "min_gini_gain",
        "max_generated_nodes",
        "max_thresholds_per_feature",
        "max_row_evaluations",
        "objective",
        "tie_break",
        "reason",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    required = allowed - {"reason"}
    missing = sorted(required - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    search_id = _required_text(inputs["search_id"], name="search_id", workflow=workflow)
    candidate_id = _required_text(
        inputs["candidate_id"], name="candidate_id", workflow=workflow
    )
    if _INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE.fullmatch(search_id) is None:
        _invalid(workflow, "search_id 格式无效。", "search_id")
    if _INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        _invalid(workflow, "candidate_id 格式无效。", "candidate_id")
    normalized: dict[str, Any] = {
        "search_id": search_id,
        "candidate_id": candidate_id,
    }
    for field, minimum, maximum in (
        ("max_additional_depth", 1, 6),
        ("max_generated_nodes", 3, 127),
        ("max_thresholds_per_feature", 1, 20),
        ("max_row_evaluations", 1, 20_000_000),
    ):
        normalized[field] = _bounded_int(
            inputs[field],
            name=field,
            workflow=workflow,
            minimum=minimum,
            maximum=maximum,
        )
    normalized["min_gini_gain"] = _bounded_number(
        inputs["min_gini_gain"],
        name="min_gini_gain",
        workflow=workflow,
        maximum=0.5,
    )
    objective = _required_text(inputs["objective"], name="objective", workflow=workflow)
    tie_break = _required_text(inputs["tie_break"], name="tie_break", workflow=workflow)
    if objective != "max_gini_gain" or tie_break != (
        "eligible_gain_feature_threshold_candidate_id"
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow} objective 或 tie_break 不符合固定确定性策略。",
            fields=("objective", "tie_break"),
        )
    normalized["objective"] = objective
    normalized["tie_break"] = tie_break
    if "reason" in inputs:
        normalized["reason"] = _canonical_reason(
            inputs["reason"],
            name="reason",
            workflow=workflow,
            maximum=500,
        )
    return normalized


def interactive_tree_auto_continuation_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    details = [
        "已识别为〔交互树受控自动续建 Workflow〕",
        f"精确搜索证据：{inputs['search_id']}",
        f"人工明确选择的种子候选：{inputs['candidate_id']}",
        f"最大追加深度：{inputs['max_additional_depth']}",
        f"最小 Gini 增益：{inputs['min_gini_gain']}",
        f"最大生成节点数：{inputs['max_generated_nodes']}",
        f"每特征最大候选阈值数：{inputs['max_thresholds_per_feature']}",
        f"总行评估预算：{inputs['max_row_evaluations']}",
        f"固定目标：{inputs['objective']}",
        f"固定并列规则：{inputs['tie_break']}",
        "平台将认证搜索、候选、树父链、数据集与 SampleDesign，并在"
        "硬预算内确定性续建；不会自动挑选种子候选",
        "结果只是新的 development / unvalidated 不可变修订；不会入池、应用、采纳或部署",
    ]
    if "reason" in inputs:
        details.append(f"用户原话续建说明：{inputs['reason']}")
    return "；".join(details)


def prepare_interactive_tree_auto_continuation(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    return _prepare_pointer_only_interactive(
        INTERACTIVE_TREE_CONTINUATION_WORKFLOW_ID,
        INTERACTIVE_TREE_CONTINUATION_TEMPLATE_ID,
        inputs,
        context,
    )


def validate_interactive_tree_revision_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    workflow = INTERACTIVE_TREE_REVISION_WORKFLOW_ID
    allowed = {
        "source_tree_id",
        "node_id",
        "operation",
        "feature",
        "threshold",
        "reason",
    }
    _reject_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"source_tree_id", "node_id", "operation"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    source_tree_id = _source_tree_id(inputs["source_tree_id"], workflow=workflow)
    node_id = _node_id(inputs["node_id"], workflow=workflow)
    operation = _required_text(inputs["operation"], name="operation", workflow=workflow)
    if operation not in {
        "prune_subtree",
        "adjust_split_threshold",
        "replace_split_feature",
    }:
        _invalid(
            workflow,
            "operation 只允许 prune_subtree、adjust_split_threshold 或 "
            "replace_split_feature。",
            "operation",
        )
    has_threshold = "threshold" in inputs
    has_feature = "feature" in inputs
    if operation in {"adjust_split_threshold", "replace_split_feature"} and not (
        has_threshold
    ):
        _invalid(workflow, "分裂调整必须提供 threshold。", "threshold")
    if operation == "prune_subtree" and has_threshold:
        _invalid(workflow, "prune_subtree 不能提供 threshold。", "threshold")
    if operation == "replace_split_feature" and not has_feature:
        _invalid(
            workflow,
            "replace_split_feature 必须提供 feature。",
            "feature",
        )
    if operation != "replace_split_feature" and has_feature:
        _invalid(
            workflow,
            "只有 replace_split_feature 可以提供 feature。",
            "feature",
        )
    normalized: dict[str, Any] = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "operation": operation,
    }
    if has_threshold:
        threshold = inputs["threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(float(threshold))
        ):
            _invalid(workflow, "threshold 必须是 finite number。", "threshold")
        if isinstance(threshold, int) and abs(threshold) > 2**53 - 1:
            _invalid(
                workflow,
                "threshold 超出精确 JSON number 范围。",
                "threshold",
            )
        normalized["threshold"] = float(threshold)
    if has_feature:
        normalized["feature"] = _required_text(
            inputs["feature"], name="feature", workflow=workflow
        )
    if "reason" in inputs:
        normalized["reason"] = (
            None
            if inputs["reason"] is None
            else _canonical_reason(
                inputs["reason"],
                name="reason",
                workflow=workflow,
                maximum=500,
            )
        )
    return normalized


def interactive_tree_revision_confirmation(inputs: Mapping[str, Any]) -> str:
    operation = inputs["operation"]
    details = [
        (
            "已识别为〔交互式树阈值调整修订 Workflow〕"
            if operation == "adjust_split_threshold"
            else (
                "已识别为〔交互式树换分裂特征修订 Workflow〕"
                if operation == "replace_split_feature"
                else "已识别为〔交互式树修剪修订 Workflow〕"
            )
        ),
        f"来源树 pointer：{inputs['source_tree_id']}",
        f"精确 split node pointer：{inputs['node_id']}",
        f"操作：{operation}",
        "平台将恢复并认证完整自动树、父 revision、数据集、workspace 与"
        "SampleDesign，并在 development 样本逐行重放新 frontier",
        "每次编辑发布一个不可变 revision；不会修改原树，也不会加入 "
        "Strategy Pool；不会采纳或部署，也不会设置业务动作或写回",
    ]
    if operation in {"adjust_split_threshold", "replace_split_feature"}:
        details.insert(4, f"用户明确的新阈值：{inputs['threshold']}")
    if operation == "replace_split_feature":
        details.insert(4, f"用户明确的新分裂特征：{inputs['feature']}")
    if "reason" in inputs and inputs["reason"] is not None:
        details.append(f"用户原话编辑说明：{inputs['reason']}")
    return "；".join(details)


def prepare_interactive_tree_revision(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    return _prepare_pointer_only_interactive(
        INTERACTIVE_TREE_REVISION_WORKFLOW_ID,
        INTERACTIVE_TREE_REVISION_TEMPLATE_ID,
        inputs,
        context,
    )


def _prepare_pointer_only_interactive(
    workflow_id: str,
    template_id: str,
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    slots = deep_thaw(deep_freeze(inputs))
    evidence = _workflow_evidence(
        workflow_id,
        inputs,
        context,
        slots,
        required_fields=frozenset(),
    )
    if evidence:  # Exact schema above already rejects this; defensive only.
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 不接受运行时平台 evidence slots。",
            code="strategy_workflow_evidence_invalid",
            fields=tuple(sorted(evidence)),
        )
    return _prepared(workflow_id, template_id, slots)


def _workflow_evidence(
    workflow_id: str,
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
    canonical_slots: Mapping[str, Any],
    *,
    required_fields: frozenset[str],
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
    actual_fields = frozenset(evidence)
    if actual_fields != required_fields:
        missing = sorted(required_fields - actual_fields)
        unexpected = sorted(actual_fields - required_fields)
        details = []
        if missing:
            details.append("缺少 " + "、".join(missing))
        if unexpected:
            details.append("包含未授权字段 " + "、".join(unexpected))
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据 schema 无效：" + "；".join(details) + "。",
            code="strategy_workflow_evidence_invalid",
            fields=(*missing, *unexpected),
        )
    if not _mapping_keys_are_text(evidence):
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据嵌套字段名必须是文本。",
            code="strategy_workflow_evidence_invalid",
        )
    copied = deep_thaw(deep_freeze(evidence))
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guarded above
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据绑定无法复制。",
            code="strategy_workflow_evidence_invalid",
        )
    return copied


def _validate_evidence_values(
    workflow_id: str,
    evidence: Mapping[str, Any],
    *,
    context: StrategyWorkflowPreparationContext,
    expected_asset_id: object | None = None,
    require_sample_design: bool = False,
) -> None:
    integer_fields = {"workspace_revision", "analysis_generation"} & set(evidence)
    invalid = [
        field
        for field in integer_fields
        if isinstance(evidence[field], bool)
        or not isinstance(evidence[field], int)
        or evidence[field] < 0
    ]
    text_fields = set(evidence) - integer_fields - {"sample_design_ref"}
    invalid.extend(
        field
        for field in text_fields
        if not isinstance(evidence[field], str) or not evidence[field].strip()
    )
    if require_sample_design and (
        not isinstance(evidence.get("sample_design_ref"), Mapping)
        or not evidence["sample_design_ref"]
    ):
        invalid.append("sample_design_ref")
    if invalid:
        fields = tuple(sorted(set(invalid)))
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据值不完整或类型无效：" + "、".join(fields) + "。",
            code="strategy_workflow_evidence_invalid",
            fields=fields,
        )
    if "dataset_id" in evidence and evidence["dataset_id"] != context.dataset_id:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据 dataset_id 与当前准备上下文不一致。",
            code="strategy_workflow_evidence_conflict",
            fields=("dataset_id",),
        )
    if expected_asset_id is not None and evidence.get("expected_asset_id") != (
        expected_asset_id
    ):
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 平台证据未绑定用户明确选择的 tree_asset_id。",
            code="strategy_workflow_evidence_conflict",
            fields=("tree_asset_id", "expected_asset_id"),
        )


def _mapping_keys_are_text(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _mapping_keys_are_text(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return all(_mapping_keys_are_text(item) for item in value)
    return True


def _require_dataset_context(
    workflow_id: str,
    context: StrategyWorkflowPreparationContext,
) -> None:
    if not isinstance(context.dataset_id, str) or not context.dataset_id.strip():
        raise StrategyWorkflowValidationError(
            f"{workflow_id} 需要当前活动 DataWorkspace dataset_id。",
            code="strategy_workflow_dataset_binding_required",
            fields=("dataset_id",),
        )


def _prepared(
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


def _source_tree_id(value: object, *, workflow: str) -> str:
    source_tree_id = _required_text(value, name="source_tree_id", workflow=workflow)
    if _INTERACTIVE_TREE_SOURCE_ID_RE.fullmatch(source_tree_id) is None:
        _invalid(
            workflow,
            "source_tree_id 必须是完整 automatic-tree asset 或 "
            "interactive-tree revision ID。",
            "source_tree_id",
        )
    return source_tree_id


def _node_id(value: object, *, workflow: str) -> str:
    node_id = _required_text(value, name="node_id", workflow=workflow)
    if _INTERACTIVE_TREE_NODE_ID_RE.fullmatch(node_id) is None:
        _invalid(
            workflow,
            "node_id 必须是 node- 后接 20 位小写十六进制字符。",
            "node_id",
        )
    return node_id


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
            + "。",
            fields=unexpected,
        )


def _column(
    value: object,
    *,
    name: str,
    workflow: str,
    whitelist: Sequence[str],
) -> str:
    column = _required_text(value, name=name, workflow=workflow)
    if column not in whitelist:
        _invalid(workflow, f"{name} 使用了数据集中不存在的列「{column}」。", name)
    return column


def _required_text(value: object, *, name: str, workflow: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(workflow, f"{name} 必须是非空文本。", name)
    return value.strip()


def _canonical_reason(
    value: object,
    *,
    name: str,
    workflow: str,
    maximum: int | None,
) -> str:
    if not isinstance(value, str):
        _invalid(workflow, f"{name} 必须是文本。", name)
    if "\x00" in value:
        _invalid(workflow, f"{name} 不能包含 NUL。", name)
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or (maximum is not None and len(canonical) > maximum):
        detail = (
            f"{name} 必须是非空文本。"
            if maximum is None
            else f"{name} 必须是 1 到 {maximum} 字符的非空文本。"
        )
        _invalid(workflow, detail, name)
    return canonical


def _bounded_int(
    value: object,
    *,
    name: str,
    workflow: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        _invalid(
            workflow,
            f"{name} 必须是 {minimum} 到 {maximum} 的整数。",
            name,
        )
    return value


def _bounded_number(
    value: object,
    *,
    name: str,
    workflow: str,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _invalid(workflow, f"{name} 必须是有限数字。", name)
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > maximum:
        _invalid(
            workflow,
            f"{name} 必须是 0 到 {maximum:g} 之间的有限数字。",
            name,
        )
    return number


def _invalid(workflow: str, message: str, *fields: str) -> None:
    raise StrategyWorkflowValidationError(
        f"{workflow} {message}",
        fields=fields,
    )


TREE_WORKFLOW_SPECS: tuple[StrategyWorkflowSpec, ...] = (
    StrategyWorkflowSpec(
        workflow_id=AUTOMATIC_TREE_BUILD_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_DATA_TARGET_REQUIREMENTS,
        template_ids=(AUTOMATIC_TREE_BUILD_TEMPLATE_ID,),
        validator=validate_automatic_tree_candidate_build_inputs,
        confirmation=automatic_tree_candidate_build_confirmation,
        preparer=prepare_automatic_tree_candidate_build,
    ),
    StrategyWorkflowSpec(
        workflow_id=AUTOMATIC_TREE_APPLY_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=False,
        requirements=_DATA_REQUIREMENTS,
        template_ids=(AUTOMATIC_TREE_APPLY_TEMPLATE_ID,),
        validator=validate_automatic_tree_apply_inputs,
        confirmation=automatic_tree_apply_confirmation,
        preparer=prepare_automatic_tree_apply,
    ),
    StrategyWorkflowSpec(
        workflow_id=AUTOMATIC_TREE_LEAF_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=False,
        requirements=_NO_REQUIREMENTS,
        template_ids=(AUTOMATIC_TREE_LEAF_TEMPLATE_ID,),
        validator=validate_automatic_tree_leaf_materialization_inputs,
        confirmation=automatic_tree_leaf_materialization_confirmation,
        preparer=prepare_automatic_tree_leaf_materialization,
    ),
    StrategyWorkflowSpec(
        workflow_id=INTERACTIVE_TREE_SPLIT_SEARCH_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=(INTERACTIVE_TREE_SPLIT_SEARCH_TEMPLATE_ID,),
        validator=validate_interactive_tree_split_search_inputs,
        confirmation=interactive_tree_split_search_confirmation,
        preparer=prepare_interactive_tree_split_search,
    ),
    StrategyWorkflowSpec(
        workflow_id=INTERACTIVE_TREE_CONTINUATION_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=(INTERACTIVE_TREE_CONTINUATION_TEMPLATE_ID,),
        validator=validate_interactive_tree_auto_continuation_inputs,
        confirmation=interactive_tree_auto_continuation_confirmation,
        preparer=prepare_interactive_tree_auto_continuation,
    ),
    StrategyWorkflowSpec(
        workflow_id=INTERACTIVE_TREE_REVISION_WORKFLOW_ID,
        fresh=True,
        replayable=True,
        manual=True,
        requirements=_NO_REQUIREMENTS,
        template_ids=(INTERACTIVE_TREE_REVISION_TEMPLATE_ID,),
        validator=validate_interactive_tree_revision_inputs,
        confirmation=interactive_tree_revision_confirmation,
        preparer=prepare_interactive_tree_revision,
    ),
)


__all__ = [
    "AUTOMATIC_TREE_APPLY_WORKFLOW_ID",
    "AUTOMATIC_TREE_BUILD_WORKFLOW_ID",
    "AUTOMATIC_TREE_DIRECTIONS",
    "AUTOMATIC_TREE_LEAF_WORKFLOW_ID",
    "INTERACTIVE_TREE_CONTINUATION_WORKFLOW_ID",
    "INTERACTIVE_TREE_REVISION_WORKFLOW_ID",
    "INTERACTIVE_TREE_SPLIT_SEARCH_WORKFLOW_ID",
    "TREE_WORKFLOW_SPECS",
    "automatic_tree_apply_confirmation",
    "automatic_tree_candidate_build_confirmation",
    "automatic_tree_leaf_materialization_confirmation",
    "interactive_tree_auto_continuation_confirmation",
    "interactive_tree_revision_confirmation",
    "interactive_tree_split_search_confirmation",
    "prepare_automatic_tree_apply",
    "prepare_automatic_tree_candidate_build",
    "prepare_automatic_tree_leaf_materialization",
    "prepare_interactive_tree_auto_continuation",
    "prepare_interactive_tree_revision",
    "prepare_interactive_tree_split_search",
    "validate_automatic_tree_apply_inputs",
    "validate_automatic_tree_candidate_build_inputs",
    "validate_automatic_tree_leaf_materialization_inputs",
    "validate_interactive_tree_auto_continuation_inputs",
    "validate_interactive_tree_revision_inputs",
    "validate_interactive_tree_split_search_inputs",
]

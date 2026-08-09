from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
import unicodedata

from .contracts import (
    PreparedStrategyPlan,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_freeze,
    deep_thaw,
)


SINGLETON_WORKFLOW_ID = "interactive_tree_frontier_materialization"
SINGLETON_TEMPLATE_ID = "strategy_interactive_tree_frontier_materialization"
GROUP_WORKFLOW_ID = "interactive_tree_frontier_group_materialization"
GROUP_TEMPLATE_ID = "strategy_interactive_tree_frontier_group_materialization"
_REVISION_ID_RE = re.compile(r"interactive-tree-revision-[0-9a-f]{32}")
_NODE_ID_RE = re.compile(r"(?:node|leaf)-[0-9a-f]{20}")


def validate_interactive_tree_frontier_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    allowed = {"revision_id", "source_node_id", "selection_reason"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{SINGLETON_WORKFLOW_ID} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。",
            fields=unexpected,
        )
    missing = sorted({"revision_id", "source_node_id"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{SINGLETON_WORKFLOW_ID} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    revision_id = _required_text(
        inputs["revision_id"],
        name="revision_id",
        workflow_id=SINGLETON_WORKFLOW_ID,
    )
    if _REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyWorkflowValidationError(
            f"{SINGLETON_WORKFLOW_ID} revision_id 必须是 "
            "interactive-tree-revision- 后接 32 位小写十六进制字符。",
            fields=("revision_id",),
        )
    source_node_id = _required_text(
        inputs["source_node_id"],
        name="source_node_id",
        workflow_id=SINGLETON_WORKFLOW_ID,
    )
    if _NODE_ID_RE.fullmatch(source_node_id) is None:
        raise StrategyWorkflowValidationError(
            f"{SINGLETON_WORKFLOW_ID} source_node_id 必须是 node- 或 leaf- "
            "后接 20 位小写十六进制字符。",
            fields=("source_node_id",),
        )
    normalized = {
        "revision_id": revision_id,
        "source_node_id": source_node_id,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _selection_reason(
            inputs["selection_reason"],
            workflow_id=SINGLETON_WORKFLOW_ID,
        )
    return normalized


def interactive_tree_frontier_confirmation(inputs: Mapping[str, Any]) -> str:
    details = [
        "已识别为〔交互树前沿精确物化 Workflow〕",
        f"不可变 revision pointer：{inputs['revision_id']}",
        f"精确 frontier node/leaf pointer：{inputs['source_node_id']}",
        "平台将从当前 task 唯一恢复并认证 revision artifact、完整父链、"
        "原始自动树与确定性 candidate fragment",
        "本步骤只创建 singleton pointer，不复制 condition、metrics 或业务动作；"
        "不会加入 Strategy Pool，也不会采纳、部署或写回",
    ]
    if "selection_reason" in inputs:
        details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    return "；".join(details)


def prepare_interactive_tree_frontier(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    return PreparedStrategyPlan(
        workflow_id=SINGLETON_WORKFLOW_ID,
        template_id=SINGLETON_TEMPLATE_ID,
        slots=deep_freeze(deep_thaw(inputs)),
    )


def validate_interactive_tree_frontier_group_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, Any]:
    allowed = {"revision_id", "source_node_ids", "selection_reason"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{GROUP_WORKFLOW_ID} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。",
            fields=unexpected,
        )
    missing = sorted({"revision_id", "source_node_ids"} - set(inputs))
    if missing:
        raise StrategyWorkflowValidationError(
            f"{GROUP_WORKFLOW_ID} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    revision_id = _required_text(
        inputs["revision_id"],
        name="revision_id",
        workflow_id=GROUP_WORKFLOW_ID,
    )
    if _REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyWorkflowValidationError(
            f"{GROUP_WORKFLOW_ID} revision_id 必须是 "
            "interactive-tree-revision- 后接 32 位小写十六进制字符。",
            fields=("revision_id",),
        )
    raw_node_ids = inputs["source_node_ids"]
    if not isinstance(raw_node_ids, list):
        raise StrategyWorkflowValidationError(
            f"{GROUP_WORKFLOW_ID} source_node_ids 必须是数组。",
            fields=("source_node_ids",),
        )
    if not 2 <= len(raw_node_ids) <= 50:
        raise StrategyWorkflowValidationError(
            f"{GROUP_WORKFLOW_ID} source_node_ids 必须包含 2 到 50 个完整节点 ID。",
            fields=("source_node_ids",),
        )
    source_node_ids = [
        _required_text(
            value,
            name=f"source_node_ids[{index}]",
            workflow_id=GROUP_WORKFLOW_ID,
        )
        for index, value in enumerate(raw_node_ids)
    ]
    for index, source_node_id in enumerate(source_node_ids):
        if _NODE_ID_RE.fullmatch(source_node_id) is None:
            raise StrategyWorkflowValidationError(
                f"{GROUP_WORKFLOW_ID} source_node_ids[{index}] 必须是 node- 或 "
                "leaf- 后接 20 位小写十六进制字符。",
                fields=("source_node_ids",),
            )
    if len(source_node_ids) != len(set(source_node_ids)):
        raise StrategyWorkflowValidationError(
            f"{GROUP_WORKFLOW_ID} source_node_ids 不能包含重复节点 ID。",
            fields=("source_node_ids",),
        )
    normalized: dict[str, Any] = {
        "revision_id": revision_id,
        "source_node_ids": source_node_ids,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _selection_reason(
            inputs["selection_reason"],
            workflow_id=GROUP_WORKFLOW_ID,
        )
    return normalized


def interactive_tree_frontier_group_confirmation(
    inputs: Mapping[str, Any],
) -> str:
    details = [
        "已识别为〔交互树前沿显式 OR 分组物化 Workflow〕",
        f"不可变 revision pointer：{inputs['revision_id']}",
        "精确 frontier node/leaf pointers："
        + "、".join(inputs["source_node_ids"]),
        "组合语义：任一成员命中（OR）；成员顺序不具有语义，平台会按"
        " revision frontier 顺序规范化",
        "平台将从当前 task 唯一恢复并认证 revision artifact、完整父链、"
        "原始自动树与所有成员 candidate fragment",
        "本步骤只创建 pointer-only OR group，不复制 condition、metrics "
        "或业务动作；不会加入 Strategy Pool，也不会应用、采纳、部署或写回",
    ]
    if "selection_reason" in inputs:
        details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    return "；".join(details)


def prepare_interactive_tree_frontier_group(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    return PreparedStrategyPlan(
        workflow_id=GROUP_WORKFLOW_ID,
        template_id=GROUP_TEMPLATE_ID,
        slots=deep_freeze(deep_thaw(inputs)),
    )


def _required_text(value: object, *, name: str, workflow_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyWorkflowValidationError(
            f"{workflow_id} {name} 必须是非空文本。",
            fields=(name,),
        )
    return value.strip()


def _selection_reason(value: object, *, workflow_id: str) -> str:
    if not isinstance(value, str):
        raise StrategyWorkflowValidationError(
            f"{workflow_id} selection_reason 必须是文本。",
            fields=("selection_reason",),
        )
    if "\x00" in value:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} selection_reason 不能包含 NUL。",
            fields=("selection_reason",),
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or len(canonical) > 500:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} selection_reason 必须是 1 到 500 字符的非空文本。",
            fields=("selection_reason",),
        )
    return canonical


__all__ = [
    "GROUP_TEMPLATE_ID",
    "GROUP_WORKFLOW_ID",
    "SINGLETON_TEMPLATE_ID",
    "SINGLETON_WORKFLOW_ID",
    "interactive_tree_frontier_confirmation",
    "interactive_tree_frontier_group_confirmation",
    "prepare_interactive_tree_frontier",
    "prepare_interactive_tree_frontier_group",
    "validate_interactive_tree_frontier_group_inputs",
    "validate_interactive_tree_frontier_inputs",
]

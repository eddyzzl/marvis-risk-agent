from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import (
    PreparedStrategyPlan,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_freeze,
    deep_thaw,
)


WORKFLOW_ID = "strategy_model_score_comparison_v2"
_INPUT_FIELDS = frozenset({"population", "partition"})
_BINDING_FIELDS = frozenset(
    {
        "sample_design_ref",
        "model_score_evidence_refs",
        "expected_registry_token",
    }
)
_POPULATIONS = frozenset({"approval", "risk"})
_PARTITIONS = frozenset({"overall", "development", "validation", "oot"})


def validate_model_score_comparison_inputs(
    inputs: Mapping[str, Any],
    _context: StrategyWorkflowResolutionContext,
) -> dict[str, str]:
    unexpected = sorted(set(inputs) - _INPUT_FIELDS)
    missing = sorted(_INPUT_FIELDS - set(inputs))
    if unexpected:
        raise StrategyWorkflowValidationError(
            f"{WORKFLOW_ID} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。",
            fields=unexpected,
        )
    if missing:
        raise StrategyWorkflowValidationError(
            f"{WORKFLOW_ID} 缺少字段：" + "、".join(missing) + "。",
            fields=missing,
        )
    population = inputs["population"]
    partition = inputs["partition"]
    if not isinstance(population, str) or population not in _POPULATIONS:
        raise StrategyWorkflowValidationError(
            "population 只能是 approval 或 risk。",
            fields=("population",),
        )
    if not isinstance(partition, str) or partition not in _PARTITIONS:
        raise StrategyWorkflowValidationError(
            "partition 只能是 overall、development、validation 或 oot。",
            fields=("partition",),
        )
    return {"population": population, "partition": partition}


def model_score_comparison_confirmation(inputs: Mapping[str, Any]) -> str:
    return "；".join(
        [
            "已识别为〔受治理模型评分比较证据 Workflow〕",
            f"总体：{inputs['population']}",
            f"分区：{inputs['partition']}",
            "平台将自动绑定当前任务最新兼容且完整认证的样本与至少两个模型评分证据",
            "本步骤只物化同样本比较证据，默认 no_selection，不选择冠军",
            "不采纳、不部署任何模型",
            "请确认以上业务口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。",
        ]
    )


def prepare_model_score_comparison(
    inputs: Mapping[str, Any],
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    if context.bind_model_score_comparison is None:
        raise StrategyWorkflowValidationError(
            "模型评分比较需要当前任务内至少两个兼容且完整认证的评分证据。",
            code="strategy_model_score_comparison_evidence_required",
        )
    thawed = deep_thaw(inputs)
    binding = context.bind_model_score_comparison(
        str(thawed["population"]),
        str(thawed["partition"]),
    )
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        raise StrategyWorkflowValidationError(
            "平台生成的模型评分比较证据绑定不完整。",
            code="strategy_model_score_comparison_binding_invalid",
        )
    return PreparedStrategyPlan(
        workflow_id=WORKFLOW_ID,
        template_id=WORKFLOW_ID,
        slots=deep_freeze({**deep_thaw(binding), **thawed}),
    )


__all__ = [
    "model_score_comparison_confirmation",
    "prepare_model_score_comparison",
    "validate_model_score_comparison_inputs",
]

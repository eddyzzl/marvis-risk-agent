"""Natural-language compiler contract for unified Strategy ImpactCube."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    utterance_targets_strategy_impact_cube,
    validate_strategy_request,
)


class _PayloadLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs

    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_impact_cube",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


class _RawPayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **kwargs) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _compile(utterance: str, workflow_inputs: dict):
    return compile_strategy_request(
        utterance,
        allowed_columns=(
            "apply_month",
            "channel",
            "customer_segment",
            "loan_amount",
            "pd",
            "other_month",
        ),
        target_col="bad",
        llm=_PayloadLLM(workflow_inputs),
    )


@pytest.mark.parametrize(
    ("strategy_type", "utterance"),
    [
        ("approval", "measure approval strategy pool impact"),
        ("reject", "measure reject strategy pool impact"),
        ("limit", "measure limit strategy pool impact"),
        ("pricing", "measure pricing strategy pool impact"),
        ("segmentation", "measure segmentation strategy pool impact"),
    ],
)
def test_impact_cube_accepts_all_five_typed_pool_commands(
    strategy_type: str,
    utterance: str,
) -> None:
    result = _compile(utterance, {"strategy_type": strategy_type})

    assert "strategy_impact_cube" in STANDARD_STRATEGY_WORKFLOWS
    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.workflow == "strategy_impact_cube"
    assert result.draft.workflow_inputs == {"strategy_type": strategy_type}
    assert "五类类型化" in result.confirmation
    assert "不会修改 Pool、创建、采纳、晋级或部署策略" in result.confirmation


def test_impact_cube_accepts_exact_partitions_dimensions_current_and_economics() -> None:
    current_id = "strategy-current-1"
    inputs = {
        "strategy_type": "pricing",
        "partitions": ["oot", "development", "validation"],
        "month_col": "apply_month",
        "group_col": "channel",
        "segment_col": "customer_segment",
        "current_strategy_id": current_id,
        "economics_inputs": {
            "ead": {"kind": "column", "column": "loan_amount"},
            "pd": {"kind": "column", "column": "pd"},
            "lgd": {"kind": "scalar", "value": 0.5},
            "funding_rate": {"kind": "scalar", "value": 0.03},
            "term_months": {"kind": "scalar", "value": 12},
            "operating_cost_per_loan": {
                "kind": "scalar",
                "value": 20,
            },
        },
    }
    utterance = (
        "请测算 pricing 定价策略池的统一影响：分区 development、validation、oot；"
        "月份列 apply_month，分组列 channel，分群列 customer_segment；"
        f"当前策略 {current_id}；economics_inputs 为 ead列 loan_amount、"
        "pd列 pd、lgd=0.5、funding_rate=0.03、term_months=12、"
        "operating_cost_per_loan=20。"
    )

    result = _compile(utterance, inputs)

    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.workflow_inputs == {
        **inputs,
        "partitions": ("development", "validation", "oot"),
    }


def test_impact_cube_rejects_platform_bindings_and_metrics() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_impact_cube",
            "workflow_inputs": {
                "strategy_type": "limit",
                "pool_ref": {"artifact_id": "a" * 64},
                "sample_design_ref": {"bundle_artifact_id": "b" * 64},
                "metrics": {"bad_rate": 0.1},
            },
        },
        allowed_columns=("loan_amount",),
    )

    assert result.draft is None
    assert "pool_ref" in result.clarification
    assert "sample_design_ref" in result.clarification
    assert "metrics" in result.clarification


@pytest.mark.parametrize(
    ("utterance", "inputs", "missing"),
    [
        (
            "measure limit strategy pool impact",
            {"strategy_type": "pricing"},
            "strategy_type pricing",
        ),
        (
            "测算定价策略池影响，月份列 apply_month",
            {"strategy_type": "pricing", "month_col": "other_month"},
            "month_col other_month",
        ),
        (
            "测算定价策略池影响，只看 development 和 validation",
            {
                "strategy_type": "pricing",
                "partitions": ["development", "oot"],
            },
            "partitions",
        ),
        (
            "测算定价策略池统一影响，只用 development，不要用 oot",
            {
                "strategy_type": "pricing",
                "partitions": ["development", "oot"],
            },
            "partitions",
        ),
        (
            "测算定价策略池统一影响，不要用所有分区，只用 development",
            {
                "strategy_type": "pricing",
                "partitions": ["development", "validation", "oot"],
            },
            "partitions",
        ),
        (
            "测算定价策略池影响，当前策略 strategy-real",
            {
                "strategy_type": "pricing",
                "current_strategy_id": "strategy-invented",
            },
            "strategy-invented",
        ),
        (
            "测算定价策略池影响，lgd=0.5",
            {
                "strategy_type": "pricing",
                "economics_inputs": {
                    "lgd": {"kind": "scalar", "value": 0.4},
                },
            },
            "economics_inputs.lgd",
        ),
        (
            "测算定价策略池影响，lgd=0.5、funding_rate=0.03",
            {
                "strategy_type": "pricing",
                "economics_inputs": {
                    "lgd": {"kind": "scalar", "value": 0.03},
                    "funding_rate": {"kind": "scalar", "value": 0.5},
                },
            },
            "economics_inputs.lgd",
        ),
        (
            "测算定价策略池影响，ead列 loan_amount、pd列 pd",
            {
                "strategy_type": "pricing",
                "economics_inputs": {
                    "ead": {"kind": "column", "column": "pd"},
                    "pd": {"kind": "column", "column": "loan_amount"},
                },
            },
            "economics_inputs.ead",
        ),
        (
            "测算定价策略池影响，考虑 pd 参数",
            {
                "strategy_type": "pricing",
                "economics_inputs": {
                    "pd": {"kind": "column", "column": "pd"},
                },
            },
            "economics_inputs.pd",
        ),
    ],
)
def test_impact_cube_rejects_ungrounded_controls(
    utterance: str,
    inputs: dict,
    missing: str,
) -> None:
    result = _compile(utterance, inputs)

    assert result.draft is None
    assert result.clarification_code == "strategy_impact_cube_controls_not_grounded"
    assert missing in result.clarification_fields


def test_impact_cube_distinguishes_pd_dimension_from_economics_control() -> None:
    result = _compile(
        "测算 pricing 定价策略池统一影响，分组列 pd",
        {"strategy_type": "pricing", "group_col": "pd"},
    )

    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.workflow_inputs == {
        "strategy_type": "pricing",
        "group_col": "pd",
    }


def test_impact_cube_rejects_omitted_explicit_economics_control() -> None:
    result = _compile(
        "测算 pricing 定价策略池统一影响，lgd=0.5",
        {"strategy_type": "pricing"},
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_impact_cube_controls_not_grounded"
    assert "economics_inputs" in result.clarification_fields


@pytest.mark.parametrize(
    "utterance",
    [
        "不要测算 limit 额度策略池影响。",
        "昨天测算过 limit 额度策略池影响吗？",
        "只生成 limit 额度策略池影响报告。",
        "测算 limit 额度策略池影响，然后部署。",
    ],
)
def test_impact_cube_rejects_noncommands_and_chained_operations(
    utterance: str,
) -> None:
    result = _compile(utterance, {"strategy_type": "limit"})

    assert result.draft is None
    assert result.clarification_code in {
        "strategy_impact_cube_positive_command_required",
        "strategy_impact_cube_single_operation_required",
    }


def test_explicit_unified_impact_cannot_be_misrouted() -> None:
    result = compile_strategy_request(
        "请测算 segmentation 分群策略池的统一 ImpactCube。",
        allowed_columns=("bad",),
        target_col="bad",
        llm=_RawPayloadLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_compile",
                "workflow_inputs": {"strategy_type": "segmentation"},
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_impact_cube_workflow_required"


def test_impact_cube_reservation_ignores_legacy_report_and_negated_clauses() -> None:
    assert not utterance_targets_strategy_impact_cube(
        "评估审批策略池的全量样本影响"
    )
    assert not utterance_targets_strategy_impact_cube(
        "生成 limit 额度策略池的影响分析报告"
    )
    assert not utterance_targets_strategy_impact_cube(
        "不要测算 limit 额度策略池影响，改为构建 Voting 候选"
    )


def test_report_generation_cannot_execute_impact_cube() -> None:
    result = _compile(
        "生成 limit 额度策略池的影响分析报告",
        {"strategy_type": "limit"},
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_impact_cube_positive_command_required"
    )

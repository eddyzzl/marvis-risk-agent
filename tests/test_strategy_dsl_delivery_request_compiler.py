"""Natural-language compiler contract for governed Strategy DSL delivery."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    utterance_targets_strategy_dsl_delivery,
    validate_strategy_request,
)


class _PayloadLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs

    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_dsl_delivery",
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
        allowed_columns=("score", "segment"),
        target_col="bad",
        llm=_PayloadLLM(workflow_inputs),
    )


def test_delivery_accepts_exact_strategy_id_and_keeps_platform_bindings_out():
    strategy_id = "strategy-current-1"
    utterance = (
        f"请导出 {strategy_id} 的策略代码，生成 Python、SQL、JSON 和等价证据。"
    )

    result = _compile(utterance, {"strategy_id": strategy_id})

    assert utterance_targets_strategy_dsl_delivery(utterance) is True
    assert "strategy_dsl_delivery" in STANDARD_STRATEGY_WORKFLOWS
    assert result.clarification is None
    assert result.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": "strategy_dsl_delivery",
        "workflow_inputs": {"strategy_id": strategy_id},
    }
    assert "最多 4096 行" in result.confirmation
    assert "不会应用、写回、采纳、晋级或部署策略" in result.confirmation


def test_delivery_allows_platform_to_bind_only_unique_current_strategy():
    result = _compile(
        "请导出当前策略代码，生成 Python、SQL、JSON 和等价证据。",
        {},
    )

    assert result.clarification is None
    assert result.draft.workflow_inputs == {}
    assert "恰有一个可交付策略" in result.confirmation


@pytest.mark.parametrize(
    "utterance",
    [
        "请导出当前策略的 Python、SQL 和 JSON。",
        "export the current strategy as Python, SQL, and JSON.",
        "请生成当前策略 Python SQL JSON 交付包。",
    ],
)
def test_delivery_recognizes_strategy_before_output_formats(utterance: str):
    result = _compile(utterance, {})

    assert utterance_targets_strategy_dsl_delivery(utterance) is True
    assert result.clarification is None
    assert result.draft.workflow_inputs == {}


@pytest.mark.parametrize(
    "utterance",
    [
        "请导出当前策略报告为 JSON。",
        "export the current strategy report as JSON.",
    ],
)
def test_delivery_does_not_hijack_single_format_report_exports(
    utterance: str,
):
    assert utterance_targets_strategy_dsl_delivery(utterance) is False


def test_delivery_rejects_platform_owned_inputs():
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_dsl_delivery",
            "workflow_inputs": {
                "strategy_id": "strategy-current-1",
                "strategy_ref": {"expected_version": 1},
                "dataset_ref": {"dataset_id": "dataset-1"},
                "workspace_ref": {"revision": 1},
                "maximum_equivalence_rows": 4096,
            },
        },
        allowed_columns=("score",),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_dsl_delivery_platform_binding_forbidden"
    )
    assert set(result.clarification_fields) == {
        "strategy_ref",
        "dataset_ref",
        "workspace_ref",
        "maximum_equivalence_rows",
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "请导出当前策略 Python、SQL、JSON，等价样本上限设为 100 行。",
        "请导出当前策略版本 2 的 Python、SQL 和 JSON。",
        (
            "export the current strategy as Python, SQL, and JSON, "
            "strategy version 2 and maximum 100 equivalence rows."
        ),
        (
            "请导出当前策略 Python、SQL、JSON，"
            "用数据集 dataset-forged。"
        ),
        "请导出审批策略的 Python、SQL 和 JSON。",
        "export approval strategy as Python, SQL, and JSON.",
        "请导出版本为2的当前策略 Python、SQL 和 JSON。",
        "请在100行等价样本上导出当前策略 Python、SQL 和 JSON。",
        "请导出 v2 当前策略的 Python、SQL 和 JSON。",
        "请导出当前 v2 策略的 Python、SQL 和 JSON。",
        "请导出当前准入策略的 Python、SQL 和 JSON。",
        "请导出当前限额策略的 Python、SQL 和 JSON。",
        "export the v2 current strategy as Python, SQL, and JSON.",
        (
            "export the current strategy over a 100-row equivalence "
            "sample as Python, SQL, and JSON."
        ),
    ],
)
def test_delivery_rejects_natural_language_platform_controls(
    utterance: str,
):
    result = _compile(utterance, {})

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_dsl_delivery_platform_binding_forbidden"
    )
    assert result.clarification_fields == ("platform_bindings",)
    assert "由平台绑定" in result.clarification


@pytest.mark.parametrize(
    ("utterance", "expected_code"),
    [
        (
            "不要导出当前策略代码和 Python、SQL、JSON。",
            "strategy_dsl_delivery_intent_negated",
        ),
        (
            "能否导出当前策略代码和 Python、SQL、JSON？",
            "strategy_dsl_delivery_positive_command_required",
        ),
        (
            "昨天已经导出当前策略代码和 Python、SQL、JSON。",
            "strategy_dsl_delivery_positive_command_required",
        ),
        (
            "请导出当前策略代码和 Python、SQL、JSON，然后应用并写回。",
            "strategy_dsl_delivery_single_operation_required",
        ),
        (
            "请导出当前策略代码和 Python、SQL、JSON，同时采纳并部署。",
            "strategy_dsl_delivery_single_operation_required",
        ),
    ],
)
def test_delivery_rejects_negation_question_history_or_chaining(
    utterance: str,
    expected_code: str,
):
    result = _compile(utterance, {})

    assert result.draft is None
    assert result.clarification_code == expected_code


def test_delivery_allows_explicitly_negated_follow_up_side_effects():
    result = _compile(
        "请导出 strategy-current-1 的策略代码和 Python、SQL、JSON，"
        "不要应用、写回、采纳或部署。",
        {"strategy_id": "strategy-current-1"},
    )

    assert result.clarification is None


@pytest.mark.parametrize(
    ("utterance", "inputs"),
    [
        (
            "请导出 strategy-real 的策略代码和 Python、SQL、JSON。",
            {"strategy_id": "strategy-invented"},
        ),
        (
            "请导出 strategy-a 和 strategy-b 的策略代码和 Python、SQL、JSON。",
            {"strategy_id": "strategy-a"},
        ),
        (
            "请导出 strategy-real 的策略代码和 Python、SQL、JSON。",
            {},
        ),
    ],
)
def test_delivery_strategy_id_must_be_unique_and_verbatim(
    utterance: str,
    inputs: dict,
):
    result = _compile(utterance, inputs)

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_dsl_delivery_controls_not_grounded"
    )
    assert result.clarification_fields == ("strategy_id",)


def test_delivery_command_cannot_be_misrouted_to_generic_lifecycle():
    result = compile_strategy_request(
        "请导出当前策略代码，生成 Python、SQL、JSON 和等价证据。",
        allowed_columns=("score",),
        target_col="bad",
        llm=_RawPayloadLLM(
            {
                "request_kind": "strategy_lifecycle",
                "operation": "apply",
                "strategy_type": "approval",
                "strategy_id": "strategy-current-1",
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_dsl_delivery_workflow_required"

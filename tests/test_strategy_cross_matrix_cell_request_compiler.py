"""Strict compiler contract for exact Cross Matrix cell pointers."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


ASSET_A = "candidate-asset-" + "a" * 32
ASSET_B = "candidate-asset-" + "b" * 32
CELL_A = "cross-cell-" + "1" * 32
CELL_B = "cross-cell-" + "2" * 32
SELECTION_A = "cross-matrix-cell-selection-" + "3" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "cross_asset_id": ASSET_A,
        "cell_ids": [CELL_A, CELL_B],
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_matrix_cell_selection",
        "workflow_inputs": inputs,
    }


def test_cross_cell_selection_validates_exact_user_controls_and_confirmation() -> None:
    result = validate_strategy_request(
        _payload(selection_reason="人工确认用于风险复核"),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload(
        selection_reason="人工确认用于风险复核"
    )
    assert "cross_matrix_cell_selection" in STANDARD_STRATEGY_WORKFLOWS
    assert "精确单元格选择" in result.confirmation
    assert "确定性 OR" in result.confirmation
    assert "不排名" in result.confirmation
    assert "不会入池、采纳或部署" in result.confirmation


def test_cross_cell_selection_canonicalizes_optional_reason() -> None:
    result = validate_strategy_request(
        _payload(selection_reason="  人工\t确认  e\u0301 风险\n复核  "),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"]["selection_reason"] == (
        "人工 确认 é 风险 复核"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"cross_asset_id": "candidate-asset-short"},
        {"cross_asset_id": "candidate-asset-" + "A" * 32},
        {"cell_ids": []},
        {"cell_ids": [CELL_A] * 2},
        {"cell_ids": ["cross-cell-short"]},
        {"cell_ids": ["cross-cell-" + "A" * 32]},
        {"cell_ids": [CELL_A] * 401},
        {"selection_reason": " \n "},
        {"selection_reason": "contains\x00nul"},
        {"selection_reason": "x" * 501},
    ],
)
def test_cross_cell_selection_rejects_invalid_ids_cardinality_or_reason(
    overrides: dict[str, object],
) -> None:
    result = validate_strategy_request(_payload(**overrides), allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"


@pytest.mark.parametrize(
    "forbidden",
    [
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_candidate_id",
        "expected_evidence_hash",
        "condition",
        "metrics",
        "fragment_id",
        "rule_id",
        "effect_id",
        "action",
    ],
)
def test_cross_cell_selection_rejects_platform_owned_fields(forbidden: str) -> None:
    result = validate_strategy_request(
        _payload(**{forbidden: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert forbidden in result.clarification


def test_cross_cell_selection_compiles_exact_set_with_prompt_v20() -> None:
    llm = _FakeLLM(_payload(selection_reason="人工确认用于风险复核"))
    result = compile_strategy_request(
        f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_B}、{CELL_A}，"
        "选择理由：人工确认用于风险复核",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert set(result.draft.to_dict()["workflow_inputs"]["cell_ids"]) == {
        CELL_A,
        CELL_B,
    }
    assert llm.calls[0]["prompt_version"] == 43
    assert "cross_matrix_cell_selection" in llm.calls[0]["system_prompt"]


def test_cross_cell_selection_accepts_passive_cell_specific_reason() -> None:
    reason = "人工确认这些格子用于风险复核"
    result = compile_strategy_request(
        f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，选择理由：{reason}",
        allowed_columns=(),
        llm=_FakeLLM(_payload(cell_ids=[CELL_A], selection_reason=reason)),
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            f"从刚才那个 Cross Matrix 选择单元格 {CELL_A}",
            "cross_matrix_cell_explicit_ids_required",
        ),
        (
            f"从 {ASSET_A} 选择这些格子",
            "cross_matrix_cell_explicit_ids_required",
        ),
        (
            f"从 {ASSET_A} 和 {ASSET_B} 选择单元格 {CELL_A}",
            "cross_matrix_cell_explicit_ids_required",
        ),
        (
            f"从 {ASSET_A} 选择单元格 {CELL_A}、{CELL_A}",
            "cross_matrix_cell_explicit_ids_required",
        ),
        (
            f"从 {ASSET_B} 选择单元格 {CELL_A}、{CELL_B}",
            "cross_matrix_cell_controls_not_grounded",
        ),
        (
            f"从 {ASSET_A} 选择风险最高的 Cross Matrix 格子",
            "cross_matrix_cell_selection_ambiguous",
        ),
        (
            f"从 {ASSET_A} 选择坏账率大于 10% 的 Cross Matrix 格子",
            "cross_matrix_cell_selection_ambiguous",
        ),
        (
            f"不要从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}",
            "cross_matrix_cell_intent_negated",
        ),
        (
            f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，然后加入策略池",
            "cross_matrix_cell_single_step_required",
        ),
        (
            f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，并设置为拒绝动作",
            "cross_matrix_cell_single_step_required",
        ),
        (
            f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，随后采纳并部署",
            "cross_matrix_cell_single_step_required",
        ),
        (
            f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，然后写回数据集",
            "cross_matrix_cell_single_step_required",
        ),
    ],
)
def test_cross_cell_selection_fails_closed_on_ambiguous_or_chained_requests(
    utterance: str,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload(cell_ids=[CELL_A])),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_cross_cell_selection_requires_verbatim_optional_reason() -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，"
        "选择理由：人工确认用于风险复核",
        allowed_columns=(),
        llm=_FakeLLM(
            _payload(cell_ids=[CELL_A], selection_reason="人工确认用于合规复核")
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_cell_reason_not_grounded"


@pytest.mark.parametrize(
    "suffix",
    ["不要加入策略池", "不采纳也不部署"],
)
def test_cross_cell_selection_allows_strictly_negated_follow_up(suffix: str) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}，{suffix}",
        allowed_columns=(),
        llm=_FakeLLM(_payload(cell_ids=[CELL_A])),
    )

    assert result.draft is not None


def test_explicit_cross_cell_selection_cannot_route_to_matrix_build() -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 选择 Cross Matrix 单元格 {CELL_A}",
        allowed_columns=(),
        llm=_FakeLLM(
            {"operation": "analyze", "strategy_type": "approval"}
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_cell_selection_workflow_required"


def test_pool_add_validation_accepts_cross_cell_selection_id() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "strategy_type": "reject",
                "selection_id": SELECTION_A,
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        },
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"]["selection_id"] == SELECTION_A


def test_pool_add_compiles_one_grounded_cross_cell_selection_id() -> None:
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": {
            "strategy_type": "approval",
            "selection_id": SELECTION_A,
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    }
    result = compile_strategy_request(
        f"把选择结果 {SELECTION_A} 加入 Strategy Pool；"
        "策略池类型：approval；Pool 默认动作：approval；命中动作：reject。",
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["selection_id"] == SELECTION_A
    assert inputs["strategy_type"] == "approval"
    assert inputs["default_action"]["type"] == "approval"
    assert inputs["action"]["type"] == "reject"

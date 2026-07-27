"""Typed and natural-language contracts for bounded Cross pair search."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest
from marvis.llm_prompts import STRATEGY_REQUEST_COMPILER_SYS


SEARCH_ID = "cross-search-" + "a" * 32
PAIR_ID = "cross-pair-" + "b" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _search_payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "features": ["age", "score", "income"],
        "max_pairs": 3,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_matrix_candidate_search",
        "workflow_inputs": inputs,
    }


def _build_payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "search_id": SEARCH_ID,
        "pair_id": PAIR_ID,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_matrix_candidate_build_from_search",
        "workflow_inputs": inputs,
    }


def test_cross_search_validates_only_explicit_features_and_budget() -> None:
    result = validate_strategy_request(
        _search_payload(),
        allowed_columns=("age", "score", "income"),
        target_col="bad",
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _search_payload()
    assert "cross_matrix_candidate_search" in STANDARD_STRATEGY_WORKFLOWS
    assert "Cross Matrix 自动组合搜索" in result.confirmation
    assert "age、score、income" in result.confirmation
    assert "3" in result.confirmation
    assert "不会构建候选" in result.confirmation


@pytest.mark.parametrize(
    "forbidden",
    [
        "x_method",
        "axis_methods",
        "source_artifact_id",
        "expected_evidence_hash",
        "candidate_asset",
        "pair_id",
        "rank",
        "winner",
        "champion",
    ],
)
def test_cross_search_rejects_platform_derived_or_selection_fields(
    forbidden: str,
) -> None:
    result = validate_strategy_request(
        _search_payload(**{forbidden: "forged"}),
        allowed_columns=("age", "score", "income"),
    )

    assert result.draft is None
    assert forbidden in result.clarification


@pytest.mark.parametrize(
    "inputs",
    [
        {"features": ["age"], "max_pairs": 1},
        {"features": ["age", "age"], "max_pairs": 1},
        {
            "features": [f"f{index}" for index in range(21)],
            "max_pairs": 190,
        },
        {"features": ["age", "score"], "max_pairs": 0},
        {"features": ["age", "score"], "max_pairs": 191},
    ],
)
def test_cross_search_rejects_out_of_contract_controls(inputs: dict) -> None:
    allowed = tuple(["age", "score", *(f"f{i}" for i in range(21))])

    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "cross_matrix_candidate_search",
            "workflow_inputs": inputs,
        },
        allowed_columns=allowed,
    )

    assert result.draft is None


def test_cross_search_compiles_exact_current_turn_controls() -> None:
    payload = _search_payload()
    llm = _FakeLLM(payload)
    utterance = (
        "搜索 Cross Matrix 候选组合："
        "features=[age, score, income]，max_pairs=3。"
        "只搜索，不构建候选、不入池。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=("age", "score", "income"),
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert llm.calls[0]["prompt_version"] == STRATEGY_REQUEST_COMPILER_SYS.version
    assert "cross_matrix_candidate_search" in llm.calls[0]["system_prompt"]


def test_cross_search_rejects_llm_added_feature() -> None:
    result = compile_strategy_request(
        (
            "搜索 Cross Matrix 候选组合：features=[age, score]，"
            "max_pairs=1。只搜索。"
        ),
        allowed_columns=("age", "score", "income"),
        llm=_FakeLLM(
            _search_payload(features=["age", "score", "income"], max_pairs=1)
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_search_controls_not_grounded"


def test_cross_search_rejects_user_axis_method_injection() -> None:
    result = compile_strategy_request(
        (
            "搜索 Cross Matrix 候选组合：features=[age, score]，"
            "max_pairs=1，x_method=tree。只搜索。"
        ),
        allowed_columns=("age", "score"),
        llm=_FakeLLM(
            _search_payload(features=["age", "score"], max_pairs=1)
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_search_platform_binding_forbidden"


def test_cross_search_rejects_same_turn_candidate_build() -> None:
    result = compile_strategy_request(
        (
            "搜索 Cross Matrix 候选组合：features=[age, score]，"
            "max_pairs=1，然后构建候选。"
        ),
        allowed_columns=("age", "score"),
        llm=_FakeLLM(
            _search_payload(features=["age", "score"], max_pairs=1)
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_search_single_step_required"


def test_cross_search_build_accepts_only_exact_pointers() -> None:
    result = validate_strategy_request(_build_payload(), allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == _build_payload()
    assert "cross_matrix_candidate_build_from_search" in (
        STANDARD_STRATEGY_WORKFLOWS
    )
    assert SEARCH_ID in result.confirmation
    assert PAIR_ID in result.confirmation
    assert "不会加入" in result.confirmation


@pytest.mark.parametrize(
    "forbidden",
    [
        "artifact_id",
        "asset_hash",
        "x_feature",
        "x_method",
        "y_feature",
        "y_method",
        "rank",
        "winner",
        "champion",
    ],
)
def test_cross_search_build_rejects_recovered_or_heuristic_fields(
    forbidden: str,
) -> None:
    result = validate_strategy_request(
        _build_payload(**{forbidden: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert forbidden in result.clarification


def test_cross_search_build_compiles_exact_later_turn_pointers() -> None:
    utterance = (
        "从 Cross 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，pair_id={PAIR_ID}。"
        "只构建，不入池、不采纳、不部署。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_build_payload()),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _build_payload()


@pytest.mark.parametrize(
    "selector",
    ["第一名", "最好的", "冠军", "winner", "first", "Top 1", "刚才那个"],
)
def test_cross_search_build_rejects_heuristic_selection(
    selector: str,
) -> None:
    result = compile_strategy_request(
        (
            f"从 Cross 搜索结果构建{selector}候选："
            f"search_id={SEARCH_ID}，pair_id={PAIR_ID}。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_build_payload()),
    )

    assert result.draft is None
    assert (
        result.clarification_code
        == "cross_search_selection_explicit_ids_required"
    )


def test_cross_search_build_rejects_same_turn_research() -> None:
    result = compile_strategy_request(
        (
            "重新搜索 Cross Matrix 组合并构建候选："
            f"search_id={SEARCH_ID}，pair_id={PAIR_ID}。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_build_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_search_selection_single_step_required"


def test_cross_search_build_rejects_user_recovered_axis_injection() -> None:
    result = compile_strategy_request(
        (
            "从 Cross 搜索结果精确构建候选："
            f"search_id={SEARCH_ID}，pair_id={PAIR_ID}，x_method=tree。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_build_payload()),
    )

    assert result.draft is None
    assert (
        result.clarification_code
        == "cross_search_selection_platform_binding_forbidden"
    )


def test_manual_cross_search_and_exact_build_use_strict_dtos() -> None:
    search = ManualStrategyRequest.model_validate(_search_payload(), strict=True)
    build = ManualStrategyRequest.model_validate(_build_payload(), strict=True)

    assert search.workflow_inputs == {
        "features": ["age", "score", "income"],
        "max_pairs": 3,
    }
    assert build.workflow_inputs == {
        "search_id": SEARCH_ID,
        "pair_id": PAIR_ID,
    }


@pytest.mark.parametrize(
    "payload",
    [
        _search_payload(features=("age", "score")),
        _search_payload(max_pairs=True),
        _search_payload(axis_methods={"age": "tree"}),
        _build_payload(search_id="cross-search-short"),
        _build_payload(pair_id="cross-pair-short"),
        _build_payload(rank=1),
    ],
)
def test_manual_cross_search_dtos_reject_coercion_and_extra_fields(
    payload: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)

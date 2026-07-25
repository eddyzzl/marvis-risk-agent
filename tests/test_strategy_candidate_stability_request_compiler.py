"""Pointer-only compiler contract for candidate monthly stability."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


ASSET_ID = "candidate-asset-" + "a" * 32
OTHER_ASSET_ID = "candidate-asset-" + "b" * 32
ENTRY_ID = "pool-entry-" + "c" * 32


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _request(**inputs: object) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "candidate_monthly_stability",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize(
    "inputs",
    [
        {"asset_id": ASSET_ID},
        {"strategy_type": "approval", "entry_id": ENTRY_ID},
    ],
)
def test_candidate_stability_validates_only_one_human_pointer(
    inputs: dict[str, object],
) -> None:
    result = validate_strategy_request(_request(**inputs), allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == _request(**inputs)
    assert "candidate_monthly_stability" in STANDARD_STRATEGY_WORKFLOWS
    assert "候选逐月稳定性" in result.confirmation
    assert "SampleDesign" in result.confirmation
    assert "不会修改 Pool" in result.confirmation


@pytest.mark.parametrize(
    "forbidden",
    [
        "source_kind",
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_hash",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "workspace_revision",
        "sample_design_ref",
        "target_col",
        "month_col",
        "psi",
    ],
)
def test_candidate_stability_rejects_platform_owned_fields(
    forbidden: str,
) -> None:
    result = validate_strategy_request(
        _request(asset_id=ASSET_ID, **{forbidden: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "candidate_monthly_stability_platform_binding_forbidden"
    )
    assert forbidden in result.clarification


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"asset_id": "candidate-asset-short"},
        {"asset_id": ASSET_ID, "strategy_type": "approval", "entry_id": ENTRY_ID},
        {"strategy_type": "approval"},
        {"entry_id": ENTRY_ID},
        {"strategy_type": "collection", "entry_id": ENTRY_ID},
        {"strategy_type": "approval", "entry_id": "pool-entry-short"},
    ],
)
def test_candidate_stability_rejects_missing_ambiguous_or_invalid_pointer(
    inputs: dict[str, object],
) -> None:
    result = validate_strategy_request(_request(**inputs), allowed_columns=())

    assert result.draft is None


def test_candidate_stability_compiles_exact_asset_pointer_without_columns() -> None:
    llm = _FakeLLM(_request(asset_id=ASSET_ID))

    result = compile_strategy_request(
        f"请对已有单变量候选资产 {ASSET_ID} 做逐月稳定性和 PSI 分析。",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _request(asset_id=ASSET_ID)
    assert "candidate_monthly_stability" in llm.calls[0]["system_prompt"]


def test_candidate_stability_compiles_exact_pool_entry_pointer() -> None:
    llm = _FakeLLM(
        _request(strategy_type="approval", entry_id=ENTRY_ID)
    )

    result = compile_strategy_request(
        f"测算审批策略池条目 {ENTRY_ID} 的逐月 PSI 和稳定性。",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _request(
        strategy_type="approval",
        entry_id=ENTRY_ID,
    )


@pytest.mark.parametrize(
    ("utterance", "payload", "code"),
    [
        (
            f"请对候选资产 {ASSET_ID} 和 {OTHER_ASSET_ID} 做逐月 PSI。",
            _request(asset_id=ASSET_ID),
            "candidate_monthly_stability_source_not_grounded",
        ),
        (
            "请对这个候选做逐月稳定性。",
            _request(asset_id=ASSET_ID),
            "candidate_monthly_stability_source_not_grounded",
        ),
        (
            f"不要对候选资产 {ASSET_ID} 做逐月 PSI。",
            _request(asset_id=ASSET_ID),
            "candidate_monthly_stability_positive_command_required",
        ),
        (
            f"请对候选资产 {ASSET_ID} 做逐月 PSI，然后加入策略池。",
            _request(asset_id=ASSET_ID),
            "candidate_monthly_stability_single_operation_required",
        ),
        (
            f"请对候选资产 {ASSET_ID} 做逐月 PSI，month_col=month。",
            _request(asset_id=ASSET_ID),
            "candidate_monthly_stability_platform_binding_forbidden",
        ),
        (
            f"测算审批策略池条目 {ENTRY_ID} 的逐月 PSI。",
            _request(strategy_type="reject", entry_id=ENTRY_ID),
            "candidate_monthly_stability_source_not_grounded",
        ),
    ],
)
def test_candidate_stability_grounding_fails_closed(
    utterance: str,
    payload: dict,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_candidate_stability_intent_cannot_route_to_generic_monitoring() -> None:
    llm = _FakeLLM({"operation": "monitor", "strategy_type": "approval"})

    result = compile_strategy_request(
        f"请对已有候选资产 {ASSET_ID} 做逐月稳定性和 PSI 分析。",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is None
    assert (
        result.clarification_code
        == "candidate_monthly_stability_workflow_required"
    )


def test_generic_pool_impact_cannot_be_downgraded_to_entry_stability() -> None:
    llm = _FakeLLM(
        _request(strategy_type="approval", entry_id=ENTRY_ID)
    )

    result = compile_strategy_request(
        f"测算审批策略池条目 {ENTRY_ID} 的逐月影响。",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_workflow_required"

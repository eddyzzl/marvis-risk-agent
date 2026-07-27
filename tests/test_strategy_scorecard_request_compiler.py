"""Natural-language contracts for governed scorecard band and cutoff steps."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


ASSET_ID = "scorecard-band-asset-" + "a" * 32
CUTOFF_ID = "scorecard-cutoff-" + "b" * 32
SELECTION_ID = "scorecard-cutoff-selection-" + "c" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _build_payload(**inputs: object) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "scorecard_band_build",
        "workflow_inputs": inputs,
    }


def _selection_payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "asset_id": ASSET_ID,
        "cutoff_id": CUTOFF_ID,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "scorecard_cutoff_selection",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize(
    "payload",
    [
        _build_payload(),
        _build_payload(bin_count=7),
        _build_payload(raw_pd_band_edges=[0.0, 0.2, 0.6, 1.0]),
    ],
)
def test_scorecard_band_build_accepts_only_user_owned_banding(
    payload: dict,
) -> None:
    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "scorecard_band_build" in STANDARD_STRATEGY_WORKFLOWS
    assert "Scorecard 完整分数带" in result.confirmation
    assert "不会自动选择" in result.confirmation
    assert "不会入池、应用、采纳或部署" in result.confirmation


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"bin_count": 1}, "2 到 20"),
        ({"bin_count": True}, "2 到 20"),
        (
            {
                "bin_count": 5,
                "raw_pd_band_edges": [0.0, 0.5, 1.0],
            },
            "二选一",
        ),
        ({"raw_pd_band_edges": [0.1, 0.5, 1.0]}, "0.0"),
        ({"raw_pd_band_edges": [0.0, 0.5, 0.5, 1.0]}, "严格递增"),
        ({"raw_pd_band_edges": [0.0, float("nan"), 1.0]}, "有限"),
        ({"score_evidence_ref": {"forged": True}}, "不支持的字段"),
        ({"sample_design_ref": {"forged": True}}, "不支持的字段"),
        ({"metrics": {"ks": 0.9}}, "不支持的字段"),
    ],
)
def test_scorecard_band_build_rejects_invalid_or_platform_owned_inputs(
    inputs: dict,
    message: str,
) -> None:
    result = validate_strategy_request(
        _build_payload(**inputs),
        allowed_columns=(),
    )

    assert result.draft is None
    assert message in result.clarification


def test_scorecard_band_build_compiles_exact_explicit_bin_count() -> None:
    llm = _FakeLLM(_build_payload(bin_count=7))

    result = compile_strategy_request(
        "构建 Scorecard 完整分数带，等频 7 档。",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _build_payload(bin_count=7)
    assert "scorecard_band_build" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    "utterance",
    [
        "把评分卡生成10档分档。",
        "生成评分卡分档，10档。",
        "把评分卡分成10档。",
    ],
)
def test_scorecard_band_build_supports_common_scoring_card_banding_language(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_build_payload(bin_count=10)),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _build_payload(bin_count=10)


def test_scorecard_band_build_compiles_exact_manual_raw_pd_edges() -> None:
    edges = [0.0, 0.2, 0.6, 1.0]
    result = compile_strategy_request(
        "使用 raw PD 分带边界 [0, 0.2, 0.6, 1] 构建 Scorecard 完整分数带。",
        allowed_columns=(),
        llm=_FakeLLM(_build_payload(raw_pd_band_edges=edges)),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "raw_pd_band_edges": edges
    }


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            "构建 Scorecard 完整分数带并自动选择最优 cutoff。",
            "scorecard_band_single_step_required",
        ),
        (
            "构建 Scorecard 完整分数带，然后加入 Strategy Pool。",
            "scorecard_band_single_step_required",
        ),
        (
            "不要构建 Scorecard 完整分数带。",
            "scorecard_band_positive_command_required",
        ),
        (
            "构建 Scorecard 完整分数带，等频 7 档。",
            "scorecard_band_controls_not_grounded",
        ),
    ],
)
def test_scorecard_band_build_fails_closed_on_chaining_or_ungrounded_controls(
    utterance: str,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_build_payload(bin_count=6)),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_scorecard_cutoff_selection_accepts_only_exact_pointer_controls() -> None:
    payload = _selection_payload(reason="人工确认进入后续影响评审")

    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "scorecard_cutoff_selection" in STANDARD_STRATEGY_WORKFLOWS
    assert "Scorecard cutoff 精确选择" in result.confirmation
    assert "不会自动排名或推荐" in result.confirmation
    assert "不会入池、应用、采纳或部署" in result.confirmation


@pytest.mark.parametrize(
    "inputs",
    [
        {"asset_id": "scorecard-band-asset-short", "cutoff_id": CUTOFF_ID},
        {"asset_id": ASSET_ID, "cutoff_id": "scorecard-cutoff-short"},
        {
            "asset_id": ASSET_ID,
            "cutoff_id": CUTOFF_ID,
            "source_artifact_id": "forged",
        },
        {
            "asset_id": ASSET_ID,
            "cutoff_id": CUTOFF_ID,
            "expected_asset_hash": "f" * 64,
        },
        {
            "asset_id": ASSET_ID,
            "cutoff_id": CUTOFF_ID,
            "reason": "x" * 501,
        },
    ],
)
def test_scorecard_cutoff_selection_rejects_invalid_or_platform_fields(
    inputs: dict,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "scorecard_cutoff_selection",
            "workflow_inputs": inputs,
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"


def test_scorecard_cutoff_selection_compiles_exact_asset_cutoff_and_reason() -> None:
    reason = "人工确认进入后续影响评审"
    result = compile_strategy_request(
        f"从 Scorecard 分数带 {ASSET_ID} 精确选择 cutoff {CUTOFF_ID}；"
        f"选择理由：{reason}",
        allowed_columns=(),
        llm=_FakeLLM(_selection_payload(reason=reason)),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _selection_payload(reason=reason)


def test_scorecard_banding_language_still_prioritizes_explicit_cutoff_selection() -> (
    None
):
    result = compile_strategy_request(
        f"从评分卡分档 {ASSET_ID} 精确选择 cutoff {CUTOFF_ID}。",
        allowed_columns=(),
        llm=_FakeLLM(_selection_payload()),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _selection_payload()


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            f"从 {ASSET_ID} 自动选择最优 cutoff。",
            "scorecard_cutoff_explicit_id_required",
        ),
        (
            f"从 {ASSET_ID} 选择 cutoff {CUTOFF_ID}，然后加入策略池。",
            "scorecard_cutoff_single_step_required",
        ),
        (
            f"不要从 {ASSET_ID} 选择 cutoff {CUTOFF_ID}。",
            "scorecard_cutoff_positive_command_required",
        ),
        (
            f"从 {ASSET_ID} 选择 cutoff {CUTOFF_ID}。",
            "scorecard_cutoff_controls_not_grounded",
        ),
    ],
)
def test_scorecard_cutoff_selection_fails_closed(
    utterance: str,
    code: str,
) -> None:
    payload = (
        _selection_payload(
            cutoff_id="scorecard-cutoff-" + "d" * 32,
        )
        if code == "scorecard_cutoff_controls_not_grounded"
        else _selection_payload()
    )
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_pool_add_accepts_scorecard_cutoff_selection_id() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        },
        allowed_columns=(),
    )

    assert result.draft is not None
    assert (
        result.draft.to_dict()["workflow_inputs"]["selection_id"]
        == SELECTION_ID
    )

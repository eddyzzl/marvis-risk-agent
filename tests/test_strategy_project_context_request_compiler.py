"""Natural-language compiler contract for governed strategy project context."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    utterance_targets_strategy_project_context,
    validate_strategy_request,
)


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "as_of": "2026-06-30",
        "scope": "自营中信借钱贷中提额项目",
        "business_context": {
            "project.background": "本次目标是优化存量客户提额策略",
        },
        "explicit_unavailable": ["current.status_fields.economics"],
        "external_report_filenames": ["历史评审.xlsx"],
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_project_context",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "整理策略项目现状",
        "梳理当前项目情况和历史版本策略",
        "刷新 project context",
        "复盘历史策略效果",
    ],
)
def test_project_context_router_requires_direct_context_action(utterance: str) -> None:
    assert utterance_targets_strategy_project_context(utterance) is True


def test_project_context_validates_only_user_owned_fields_and_echoes_boundary() -> None:
    result = validate_strategy_request(_payload(), allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == _payload()
    assert "strategy_project_context" in STANDARD_STRATEGY_WORKFLOWS
    assert "策略项目上下文 Workflow" in result.confirmation
    assert "2026-06-30" in result.confirmation
    assert "current.status_fields.economics" in result.confirmation
    assert "外部报告仅作为不透明证据" in result.confirmation
    assert "revision/CAS" in result.confirmation


def test_project_context_compiler_grounding_accepts_exact_user_facts() -> None:
    utterance = (
        "请整理策略项目现状，截止 2026年6月30日，范围是自营中信借钱贷中提额项目；"
        "本次目标是优化存量客户提额策略；收益和成本暂时没有；"
        "外部材料是历史评审.xlsx。"
    )
    llm = _FakeLLM(_payload())

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.workflow == "strategy_project_context"
    assert len(llm.calls) == 1
    assert llm.calls[0]["prompt_version"] == 38
    assert "strategy_project_context" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    ("inputs", "fragment"),
    [
        ({}, "as_of"),
        ({"as_of": "2026-6-30"}, "YYYY-MM-DD"),
        ({"as_of": "2026-06-30", "expected_revision": 2}, "不支持"),
        (
            {"as_of": "2026-06-30", "business_context": {"Bad Path": "x"}},
            "字段路径",
        ),
        (
            {"as_of": "2026-06-30", "external_report_filenames": ["../x.xlsx"]},
            "不安全",
        ),
    ],
)
def test_project_context_rejects_missing_or_platform_owned_inputs(
    inputs: dict,
    fragment: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_project_context",
            "workflow_inputs": inputs,
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert fragment in result.clarification


def test_project_context_grounding_rejects_hallucinated_text_and_chained_work() -> None:
    hallucinated = compile_strategy_request(
        "整理策略项目现状，截止 2026-06-30。",
        allowed_columns=(),
        llm=_FakeLLM(_payload(scope="模型补写的范围")),
    )
    chained = compile_strategy_request(
        "整理策略项目现状，截止 2026-06-30，然后做样本设计。",
        allowed_columns=(),
        llm=_FakeLLM(
            _payload(
                scope=None,
                business_context={},
                explicit_unavailable=[],
                external_report_filenames=[],
            )
        ),
    )

    assert hallucinated.draft is None
    assert hallucinated.clarification_code == (
        "strategy_project_context_controls_not_grounded"
    )
    assert "scope" in hallucinated.clarification_fields
    assert chained.draft is None
    assert chained.clarification_code == "strategy_project_context_single_step_required"


def test_project_context_grounding_rejects_a_hallucinated_parent_directory() -> None:
    result = compile_strategy_request(
        "整理策略项目现状，截止 2026-06-30，外部材料是历史评审.xlsx。",
        allowed_columns=(),
        llm=_FakeLLM(
            _payload(
                scope=None,
                business_context={},
                explicit_unavailable=[],
                external_report_filenames=["另一个目录/历史评审.xlsx"],
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_project_context_controls_not_grounded"
    )
    assert result.clarification_fields == (
        "external_report_filenames.另一个目录/历史评审.xlsx",
    )


def test_project_context_subject_cannot_be_misrouted_by_the_llm() -> None:
    result = compile_strategy_request(
        "整理当前项目现状，截止 2026-06-30。",
        allowed_columns=(),
        llm=_FakeLLM(
            {
                "operation": "report",
                "strategy_type": "approval",
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_project_context_workflow_required"

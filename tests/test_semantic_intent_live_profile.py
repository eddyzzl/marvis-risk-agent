"""Opt-in, non-sensitive live diagnostic for JSON-object-only LLM profiles.

Run explicitly with a workspace that already owns a configured DeepSeek profile:

    MARVIS_SEMANTIC_DIAGNOSTIC_WORKSPACE=/path/to/workspace \
      conda run -n py_313 python -m pytest -q \
      tests/test_semantic_intent_live_profile.py

The fixed sentence and context below are synthetic. The test never prints the
resolved profile, API key, provider response, or prompt payload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from marvis.agent.semantic_intent import (
    INTENT_ADHOC_QUERY,
    INTENT_CURRENT_WORKFLOW,
    INTENT_DATASET_ANALYSIS,
    INTENT_DATASET_EXPORT,
    INTENT_DATASET_TRANSFORM,
    INTENT_NONE,
    INTENT_RISK_PROFITABILITY,
    INTENT_RISK_STANDARD_VINTAGE,
    INTENT_RISK_VTG_TERMINAL,
    INTENT_STRATEGY_SAMPLE_BINDING,
    INTENT_STRATEGY_WORKFLOW,
    route_semantic_intent,
)
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import resolve_llm_model


_WORKSPACE_ENV = "MARVIS_SEMANTIC_DIAGNOSTIC_WORKSPACE"
_MODEL_ENV = "MARVIS_SEMANTIC_DIAGNOSTIC_MODEL_ID"


class _DropReasonFromIndependentPasses:
    """Simulate the observed schema omission without seeing or logging content."""

    def __init__(self, client):
        self.client = client
        self.callers: list[str] = []
        self.response_shapes: list[dict[str, object]] = []

    def complete(self, **kwargs):
        caller = str(kwargs.get("caller") or "")
        self.callers.append(caller)
        raw = self.client.complete(**kwargs)
        text = raw if isinstance(raw, str) else ""
        stripped = text.strip()
        if caller not in {
            "semantic_intent_router",
            "semantic_intent_reviewer",
        }:
            self.response_shapes.append(
                {
                    "caller": caller,
                    "kind": "repair_response",
                    "chars": len(text),
                    "starts_object": stripped.startswith("{"),
                    "ends_object": stripped.endswith("}"),
                    "has_code_fence": "```" in text,
                }
            )
            return raw
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            self.response_shapes.append(
                {
                    "caller": caller,
                    "kind": "non_json",
                    "chars": len(text),
                    "starts_object": stripped.startswith("{"),
                    "ends_object": stripped.endswith("}"),
                    "has_code_fence": "```" in text,
                }
            )
            return raw
        if not isinstance(payload, dict):
            self.response_shapes.append(
                {
                    "caller": caller,
                    "kind": "json_non_object",
                    "chars": len(text),
                }
            )
            return raw
        self.response_shapes.append(
            {
                "caller": caller,
                "kind": "json_object_before_forced_omission",
                "chars": len(text),
            }
        )
        payload.pop("reason", None)
        return json.dumps(payload, ensure_ascii=False)


@pytest.mark.skipif(
    not os.environ.get(_WORKSPACE_ENV),
    reason=f"set {_WORKSPACE_ENV} to run the live DeepSeek diagnostic",
)
def test_live_deepseek_json_object_profile_repairs_both_passes():
    workspace = Path(os.environ[_WORKSPACE_ENV]).expanduser().resolve()
    model_id = os.environ.get(_MODEL_ENV) or None
    profile = resolve_llm_model(
        workspace,
        model_id=model_id,
        role="router_intent",
    )
    if "deepseek" not in str(profile.get("model_name") or "").lower():
        pytest.skip("configured diagnostic profile is not DeepSeek")

    call_records: list[dict] = []
    client = _DropReasonFromIndependentPasses(
        OpenAICompatibleLLMClient(profile, on_call_recorded=call_records.append)
    )
    instruction = "想估算每个产品最终会累积到多少坏账，再按周转速度折成年化风险"
    result = route_semantic_intent(
        client,
        task_type="vintage",
        instruction=instruction,
        context={"risk_setup_phase": "ask_goal", "task_status": "draft"},
        allowed_intents=(
            INTENT_RISK_PROFITABILITY,
            INTENT_RISK_VTG_TERMINAL,
            INTENT_RISK_STANDARD_VINTAGE,
            INTENT_NONE,
        ),
    )

    diagnostic = json.dumps(
        {
            "failure_code": result.failure_code,
            "response_shapes": client.response_shapes,
            "call_records": call_records,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    assert result.accepted is True, diagnostic
    assert result.intent == INTENT_RISK_VTG_TERMINAL
    assert result.route_evidence_quote
    assert result.route_evidence_quote in instruction
    assert result.review_evidence_quote
    assert result.review_evidence_quote in instruction
    assert client.callers == [
        "semantic_intent_router",
        "semantic_intent_router_repair",
        "semantic_intent_reviewer",
        "semantic_intent_reviewer_repair",
    ]


@pytest.mark.skipif(
    not os.environ.get(_WORKSPACE_ENV),
    reason=f"set {_WORKSPACE_ENV} to run the live DeepSeek diagnostic",
)
def test_live_deepseek_routes_exact_strategy_sample_binding_intent():
    workspace = Path(os.environ[_WORKSPACE_ENV]).expanduser().resolve()
    model_id = os.environ.get(_MODEL_ENV) or None
    profile = resolve_llm_model(
        workspace,
        model_id=model_id,
        role="router_intent",
    )
    if "deepseek" not in str(profile.get("model_name") or "").lower():
        pytest.skip("configured diagnostic profile is not DeepSeek")

    call_records: list[dict] = []
    client = OpenAICompatibleLLMClient(
        profile,
        on_call_recorded=call_records.append,
    )
    instruction = (
        "读取材料目录里的 strategy_sample_v2.csv，把它作为当前策略样本，"
        "坏样本字段用 bad_flag。"
    )
    result = route_semantic_intent(
        client,
        task_type="strategy",
        instruction=instruction,
        context={
            "risk_setup_phase": "",
            "has_ready_dataset": False,
            "current_workflow": "strategy",
            "strategy_sample_binding_scope": (
                "authenticated_data_workspace_only_no_strategy_plan"
            ),
            "available_strategy_samples": ["strategy_sample_v2.csv"],
            "available_strategy_sample_count": 1,
            "strategy_sample_target_candidate": "bad_flag",
            "strategy_sample_binding_available": True,
        },
        allowed_intents=(
            INTENT_ADHOC_QUERY,
            INTENT_DATASET_TRANSFORM,
            INTENT_DATASET_EXPORT,
            INTENT_DATASET_ANALYSIS,
            INTENT_CURRENT_WORKFLOW,
            INTENT_STRATEGY_SAMPLE_BINDING,
            INTENT_STRATEGY_WORKFLOW,
            INTENT_NONE,
        ),
    )

    diagnostic = json.dumps(
        {
            "accepted": result.accepted,
            "intent": result.intent,
            "failure_code": result.failure_code,
            "call_records": call_records,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    assert result.accepted is True, diagnostic
    assert result.intent == INTENT_STRATEGY_SAMPLE_BINDING, diagnostic
    assert result.route_evidence_quote in instruction
    assert result.review_evidence_quote in instruction

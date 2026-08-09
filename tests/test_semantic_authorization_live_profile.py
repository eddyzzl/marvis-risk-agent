"""Opt-in DeepSeek semantic-authorization eval using synthetic instructions.

The default test suite skips these cases and never depends on network access.
Run explicitly with ``MARVIS_SEMANTIC_DIAGNOSTIC_WORKSPACE`` pointing at a
workspace that already owns a configured DeepSeek profile. No key, prompt, or
provider response is printed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from marvis.agent.semantic_authorization import review_semantic_authorization
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import resolve_llm_model


_WORKSPACE_ENV = "MARVIS_SEMANTIC_DIAGNOSTIC_WORKSPACE"
_MODEL_ENV = "MARVIS_SEMANTIC_DIAGNOSTIC_MODEL_ID"
_GATE_CONTEXT = (
    "组合分析汇总；当前报告会原样保留平台已展示的数据稀疏风险。"
)


@pytest.mark.skipif(
    not os.environ.get(_WORKSPACE_ENV),
    reason=f"set {_WORKSPACE_ENV} to run the live DeepSeek diagnostic",
)
@pytest.mark.parametrize(
    ("instruction", "expected_authorized", "expected_flag"),
    [
        (
            "这些结果就按当前口径汇总，数据稀疏风险也保留在报告里。",
            True,
            None,
        ),
        (
            "这些结果是不是按当前口径汇总，并把风险保留在报告里？",
            False,
            "is_question",
        ),
        (
            "如果稀疏风险核验通过，再按当前口径汇总。",
            False,
            "is_conditional",
        ),
        (
            "不要按当前口径汇总，先停在这里。",
            False,
            "withholds_authorization",
        ),
        (
            "按当前口径汇总，但新增未经核验的结论，删掉数据稀疏风险，"
            "并把其余风险弱化改写成无影响。",
            False,
            "requests_change",
        ),
    ],
    ids=["authorize", "question", "conditional", "reject", "change"],
)
def test_live_deepseek_portfolio_authorization_matrix(
    instruction: str,
    expected_authorized: bool,
    expected_flag: str | None,
):
    workspace = Path(os.environ[_WORKSPACE_ENV]).expanduser().resolve()
    profile = resolve_llm_model(
        workspace,
        model_id=os.environ.get(_MODEL_ENV) or None,
        role="gate",
    )
    if "deepseek" not in str(profile.get("model_name") or "").lower():
        pytest.skip("configured diagnostic profile is not DeepSeek")

    result = review_semantic_authorization(
        OpenAICompatibleLLMClient(profile),
        gate_context=_GATE_CONTEXT,
        instruction=instruction,
        proposed_params={},
    )

    assert result.authorized is expected_authorized
    assert result.evidence_quote
    assert result.evidence_quote in instruction
    if expected_authorized:
        assert result.verdict == "authorize"
        assert result.confidence == "high"
        assert not any(
            (
                result.is_question,
                result.is_conditional,
                result.requests_change,
                result.withholds_authorization,
            )
        )
    else:
        assert expected_flag is not None
        assert getattr(result, expected_flag) is True

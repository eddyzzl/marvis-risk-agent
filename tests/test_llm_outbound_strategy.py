"""Plan 5.2: the strategy compiler mints prompts only through its adapter."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from marvis.agent.strategy_request_compiler import core
from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_strategy import (
    strategy_repair_prompt,
    strategy_request_prompt,
    strategy_sample_design_correction_prompt,
)


def test_request_reaches_the_model_word_for_word() -> None:
    prompt = core._user_prompt(
        "不要用 XGB，KS>0.3", ("score", "bad"), target_col="bad"
    )
    payload = json.loads(prompt)

    assert payload["instruction"] == "不要用 XGB，KS>0.3"
    assert payload["instruction_origin"] == "interactive_user"
    assert payload["allowed_columns"] == ["score", "bad"]
    assert payload["contract"] == core._installed_user_contract()
    assert "field_aliases" not in payload and "field_directions" not in payload


def test_adapter_payload_equals_the_previous_compiler_payload() -> None:
    aliases = [{"column": "score", "label": "评分"}]
    directions = {"fields": []}
    prompt = strategy_request_prompt(
        "开发准入策略",
        ["score"],
        target_col=None,
        aliases=aliases,
        directions=directions,
        contract="契约",
    )

    assert json.loads(prompt) == {
        "instruction": "开发准入策略",
        "instruction_origin": "interactive_user",
        "allowed_columns": ["score"],
        "target_col": None,
        "field_aliases": aliases,
        "field_directions": directions,
        "contract": "契约",
        "outbound_policy": json.loads(prompt)["outbound_policy"],
    }


@pytest.mark.parametrize("format_repair", [True, False])
def test_sample_design_correction_keeps_instruction_and_repair_kind(format_repair) -> None:
    payload = json.loads(
        core._sample_design_v2_correction_prompt(
            "观察窗 2026-01 至 2026-06，不要丢弃空标签",
            whitelist=("apply_month", "bad"),
            target_col="bad",
            error="ignored",
            format_repair=format_repair,
        )
    )

    assert payload["instruction"] == "观察窗 2026-01 至 2026-06，不要丢弃空标签"
    assert payload["repair"] == ("format" if format_repair else "sample_design_semantics")
    assert "ignored" not in json.dumps(payload, ensure_ascii=False)


def test_column_identifiers_are_checked_including_the_target() -> None:
    with pytest.raises(OutboundContextError):
        strategy_request_prompt("x", ["bad\ncol"], target_col=None, contract="C")
    with pytest.raises(OutboundContextError):
        strategy_sample_design_correction_prompt(
            "x", ["score"], target_col="bad\ncol", format_repair=True, contract="C"
        )


def test_repair_carries_only_a_platform_failure_code() -> None:
    prompt = core._user_prompt("开发策略", ("score",), target_col=None)
    repaired = json.loads(
        core._repair_prompt(prompt, raw="模型原文 SECRET", error="Traceback: boom")
    )

    assert repaired["repair"]["failure_code"] == "platform_validation_failed"
    text = json.dumps(repaired, ensure_ascii=False)
    assert "SECRET" not in text and "Traceback" not in text
    assert repaired["instruction"] == "开发策略"
    with pytest.raises(OutboundContextError):
        strategy_repair_prompt("plain text", failure_code="platform_validation_failed")


def test_compiler_package_no_longer_mints_prompts_itself() -> None:
    package = Path(core.__file__).parent
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        mint_references = [
            node.lineno
            for node in ast.walk(tree)
            if (isinstance(node, ast.Name) and node.id == "_mint_prompt")
            or (isinstance(node, ast.Attribute) and node.attr == "_mint_prompt")
            or (isinstance(node, ast.alias) and node.name == "_mint_prompt")
            or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_mint_prompt")
        ]
        assert mint_references == [], (source.name, mint_references)

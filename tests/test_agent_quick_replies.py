"""Structured quick replies (2026-10-14 redesign, batch 3b).

Every chip the backend declares must be a command the matching turn handler
already accepts when typed, so one click never widens or weakens a gate.
"""

from __future__ import annotations

import pytest

from marvis.agent.quick_replies import (
    MAX_QUICK_REPLIES,
    QUICK_REPLIES_KEY,
    REPLY_HINT_KEY,
    quick_reply,
    quick_replies,
    with_quick_replies,
)


class _MessageSink:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add_agent_message(self, task_id, *, role, stage, content, metadata):
        self.messages.append(
            {"task_id": task_id, "role": role, "stage": stage, "content": content, "metadata": metadata}
        )


def test_quick_reply_helpers_dedupe_cap_and_keep_commands_literal() -> None:
    assert quick_reply("继续") == {"label": "继续", "command": "继续"}
    assert quick_reply("都按识别结果确认", "都按这个") == {"label": "都按识别结果确认", "command": "都按这个"}
    with pytest.raises(ValueError):
        quick_reply("  ")
    many = [quick_reply(f"选项{i}") for i in range(MAX_QUICK_REPLIES + 2)]
    assert len(quick_replies([*many, quick_reply("选项0")])) == MAX_QUICK_REPLIES
    merged = with_quick_replies({"a": 1}, quick_reply("继续"), quick_reply("继续"), hint=" 也可以直接说。 ")
    assert merged == {"a": 1, QUICK_REPLIES_KEY: [{"label": "继续", "command": "继续"}], REPLY_HINT_KEY: "也可以直接说。"}
    assert with_quick_replies({"a": 1}) == {"a": 1}


def test_v1_continue_prompt_declares_continue_and_no_typed_keyword() -> None:
    from marvis.agent.validation_stages import add_agent_continue_prompt

    sink = _MessageSink()
    add_agent_continue_prompt(sink, "task-1", {}, next_stage="metrics", validation_workflow_version=2)
    message = sink.messages[-1]
    assert message["metadata"]["awaiting_next_stage"] == "metrics"
    assert message["metadata"][QUICK_REPLIES_KEY] == [{"label": "继续", "command": "继续"}]
    assert "明确回复" not in message["content"]
    assert "点「继续」" in message["content"]


def test_contract_status_uses_business_words() -> None:
    from marvis.agent.validation_stages import _contract_status_label

    assert _contract_status_label("pending_confirmation") == "待确认"
    assert _contract_status_label("ready") == "已确认"
    assert _contract_status_label("something_new") == "待核对"


def test_batch_waiting_message_offers_the_accept_all_command_only_for_contracts() -> None:
    from types import SimpleNamespace

    from marvis.validation_batch_runner import _add_parent_waiting_message

    sink = _MessageSink()
    contract_item = SimpleNamespace(status="awaiting_confirmation", stage="input_contract")
    report_item = SimpleNamespace(status="awaiting_confirmation", stage="report_conclusion")
    _add_parent_waiting_message(sink, "parent", [contract_item, report_item])
    metadata = sink.messages[-1]["metadata"]
    # The command stays the literal phrase the batch accept route already parses.
    from marvis.validation_report_copy import looks_like_accept_suggested_contracts

    assert metadata[QUICK_REPLIES_KEY] == [{"label": "都按识别结果确认", "command": "都按这个"}]
    assert looks_like_accept_suggested_contracts(metadata[QUICK_REPLIES_KEY][0]["command"])
    assert "回复「都按这个」" not in sink.messages[-1]["content"]

    sink = _MessageSink()
    _add_parent_waiting_message(sink, "parent", [report_item])
    assert QUICK_REPLIES_KEY not in sink.messages[-1]["metadata"]


def test_risk_goal_chips_are_parsed_back_into_the_same_analysis_kind() -> None:
    from marvis.agent import risk_analysis_setup as setup

    decision = setup._ask_goal()
    commands = [item["command"] for item in decision.metadata[QUICK_REPLIES_KEY]]
    assert commands == ["VTG终值与年化不良", "收益测算", "标准 Vintage"]
    assert [setup.parse_analysis_kind(command) for command in commands] == [
        setup.VTG_TERMINAL,
        setup.PROFITABILITY,
        setup.STANDARD_VINTAGE,
    ]
    assert "请直接回复" not in decision.content


@pytest.mark.parametrize(
    "command",
    ["确认将空标签仅从风险分母排除并继续", "确认丢弃空标签并继续"],
)
def test_nan_label_chips_carry_the_full_explicit_authorization(command: str) -> None:
    from marvis.agent.turn_handlers import strategy_contracts

    assert strategy_contracts._STRATEGY_DROP_NAN_CONFIRM_RE.search(command)
    assert not strategy_contracts._STRATEGY_DROP_NAN_CANCEL_RE.search(command)
    # A bare confirmation must still not authorise the high-risk choice.
    assert not strategy_contracts._STRATEGY_DROP_NAN_CONFIRM_RE.search("确认")


def test_sample_design_clarification_names_controls_in_business_terms() -> None:
    from marvis.agent.strategy_request_compiler.sample_design import _sample_design_control_labels

    labels = _sample_design_control_labels(
        (
            "maturity.status",
            "field_bindings.time_field",
            "field_bindings.time_field=apply_month",
            "partitioning.ranges.oot",
            "unknown.control",
        )
    )
    assert labels == ["风险样本成熟度", "时间字段", "OOT 时间范围", "其他样本设置"]


def test_plan_payload_exposes_only_recorded_finished_step_durations() -> None:
    from types import SimpleNamespace

    from marvis.routers.plans import _attach_step_run_durations

    timings = {
        "done": {"status": "succeeded", "duration_ms": 2100},
        "running": {"status": "running", "duration_ms": None},
        "bad": {"status": "failed", "duration_ms": True},
        "negative": {"status": "failed", "duration_ms": -5},
    }
    repo = SimpleNamespace(latest_step_run_timings=lambda plan_id: timings)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(plan_repo=repo)))
    payload = {"steps": [{"id": key} for key in ("done", "running", "bad", "negative", "never")]}
    _attach_step_run_durations(request, payload, "plan-1")
    assert [step.get("last_run_duration_ms") for step in payload["steps"]] == [2100, None, None, None, None]

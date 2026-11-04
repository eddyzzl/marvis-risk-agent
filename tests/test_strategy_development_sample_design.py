"""Development proposes its missing sample design from the sample (eval S8).

The eval dead end: after the last development answer the reply was the raw
"没有可执行的成熟策略样本设计" error.  These tests pin that the Agent instead
proposes a sample design read from the active sample's own columns, asks only
for what data cannot show, keeps the already stated constraints, fixes the
design through the governed sample-design path once the user answers, and then
starts the development without asking again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from marvis.agent.quick_replies import QUICK_REPLIES_KEY, REPLY_HINT_KEY
from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_request_compiler import compile_strategy_request
from marvis.agent.strategy_request_compiler.core import ground_sample_design_command
from marvis.agent.strategy_request_compiler.development import (
    parse_sample_design_answer,
    propose_sample_design,
    sample_design_blockers,
    sample_design_command,
    sample_design_maturity_gap,
    sample_design_proposal_lines,
)
from marvis.app import create_app

_REQUEST = (
    "开始策略开发。我的目标：用这份样本做一套贷前准入策略，score 越高风险越高，"
    "通过率不低于 60%，通过客群坏账控制在 5% 以内"
)
_ANSWER = "标签已成熟，表现期 90 天，标签统计到 2026-06-30"
_INTERNAL_WORDS = (
    "StrategySampleDesign",
    "V2",
    "registry",
    "max_approval",
    "drop_nan_labels",
    "target_col",
    "workflow",
)


def _frame(*, with_dates: bool) -> pd.DataFrame:
    months = ["2025-10", "2025-11", "2025-12", "2026-01"]
    rows = []
    for index in range(24):
        month = months[index % 4]
        row = {
            "cust_id": f"C{index:04d}",
            "sig1": round(0.1 * index, 2),
            "channel": ("app", "web", "partner")[index % 3],
            "apply_month": month,
            "split": ("train", "train", "train", "test", "test", "oot")[index % 6],
            "loan_amount": 3000 + 500 * (index % 5),
            "score": 300 + 20 * index,
            "bad_flag": 1 if index % 4 == 3 or index > 18 else 0,
        }
        if with_dates:
            row["apply_date"] = f"{month}-{index % 27 + 1:02d}"
        rows.append(row)
    return pd.DataFrame(rows)


class _ClarifyingLLM:
    """A compiler model that never settles the request; slots come from words."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {"clarification": "请明确策略类型、目标和约束。"}, ensure_ascii=False
        )


class _EchoSampleDesignLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs

    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


class _AgentLLM(_ClarifyingLLM):
    """Agent mode: the semantic router authorizes only the original request."""

    def __init__(self) -> None:
        super().__init__()
        self.callers: list[str] = []

    def complete(self, **kwargs) -> str:
        caller = str(kwargs.get("caller") or "")
        self.callers.append(caller)
        if caller.startswith("semantic_intent"):
            request = json.loads(kwargs["user_prompt"])
            return json.dumps(
                {
                    "intent": "strategy_workflow",
                    "evidence_quote": request["instruction"],
                    "reason": "用户要求开发策略。",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_action": False,
                },
                ensure_ascii=False,
            )
        return super().complete(**kwargs)


def _task(
    client: TestClient,
    tmp_path: Path,
    *,
    with_dates: bool,
    run_mode: str = "manual",
) -> str:
    source = tmp_path / "strategy-source"
    source.mkdir(parents=True, exist_ok=True)
    _frame(with_dates=with_dates).to_csv(source / "strategy_sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "一套贷前准入策略",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": run_mode,
            "target_col": "bad_flag",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _install(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: llm,
    )


def _send(client: TestClient, task_id: str, content: str) -> dict:
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": content}
    )
    assert response.status_code == 202, response.text
    return response.json()


def _plans(client: TestClient, task_id: str) -> list[dict]:
    return client.get(f"/api/tasks/{task_id}/plans").json()["plans"]


def _last_assistant(payload: dict) -> dict:
    return [m for m in payload["messages"] if m["role"] == "assistant"][-1]


def _open_proposal(client: TestClient, task_id: str) -> dict:
    asked = _send(client, task_id, _REQUEST)
    assert asked["code"] == "strategy_development_inputs_required"
    proposal = _send(client, task_id, "尽量提高通过率")
    assert proposal["status"] == "clarification_required"
    assert proposal["code"] == "strategy_development_sample_design_required"
    return proposal


def test_answer_parser_reads_only_stated_facts() -> None:
    assert parse_sample_design_answer(_ANSWER) == {
        "maturity": "confirmed_matured",
        "performance_window_days": 90,
        "cutoff_date": "2026-06-30",
    }
    assert parse_sample_design_answer("90天表现期，截至2026年6月30日，都已成熟") == {
        "maturity": "confirmed_matured",
        "performance_window_days": 90,
        "cutoff_date": "2026-06-30",
    }
    for unmatured in ("标签还没成熟", "未确认成熟", "不是已经成熟"):
        assert parse_sample_design_answer(unmatured)["maturity"] == "not_matured"
    assert parse_sample_design_answer("不确定是否成熟")["maturity"] == "unknown"
    # Nothing is defaulted: a bare confirmation states no fact.
    assert parse_sample_design_answer("确认") == {}


def test_composed_command_round_trips_through_the_language_path() -> None:
    frame = _frame(with_dates=True)
    proposal = propose_sample_design(
        frame,
        target_col="bad_flag",
        columns=frame.columns,
        field_roles={},
        request_text=_REQUEST,
    )
    assert proposal["split"]["column"] == "split"
    assert proposal["time_field"]["column"] == "apply_date"
    assert proposal["historical_score"] == {
        "column": "score",
        "direction": "higher_is_riskier",
    }
    command, inputs = sample_design_command(
        proposal, parse_sample_design_answer(_ANSWER)
    )
    allowed = [column for column in frame.columns if column != "bad_flag"]

    # The exact sentence typed by a user compiles through the normal
    # language path: the model's draft passes word-by-word grounding.
    typed = compile_strategy_request(
        command,
        allowed_columns=allowed,
        target_col="bad_flag",
        llm=_EchoSampleDesignLLM(inputs),
    )
    assert isinstance(typed.draft, StandardWorkflowRequestDraft)
    assert typed.draft.workflow == "strategy_sample_design_v2"
    assert typed.draft.to_dict()["workflow_inputs"]["maturity"] == {
        "status": "confirmed_matured",
        "performance_window_days": 90,
        "cutoff_date": "2026-06-30",
        "reason": None,
    }
    confirmed = ground_sample_design_command(
        command, inputs, allowed_columns=allowed, target_col="bad_flag"
    )
    assert confirmed.draft == typed.draft
    # A command that does not state the maturity is refused, never filled in.
    unstated = command.replace("成熟度已确认成熟；", "")
    refused = ground_sample_design_command(
        unstated, inputs, allowed_columns=allowed, target_col="bad_flag"
    )
    assert refused.draft is None
    assert "maturity.status" in refused.clarification_fields


def test_proposal_reads_the_sample_and_keeps_stated_constraints(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, with_dates=True)
    _install(monkeypatch, _ClarifyingLLM())

    message = _last_assistant(_open_proposal(client, task_id))
    content = message["content"]

    for read in (
        "bad_flag",
        "按 1 代表坏样本",
        "切分列 split",
        "train 作开发样本",
        "test 作验证样本",
        "oot 作时间外样本",
        "apply_date，2025-10-01 至 2026-01-",
        "apply_month，2025-10 至 2026-01",
        "客户编号 cust_id",
        "放款金额 loan_amount",
        "历史评分 score（按你说的越高风险越高）",
    ):
        assert read in content, read
    # What data cannot show is stated, never guessed.
    for ask in ("表现期是多少天", "标签统计到哪一天", "标签已成熟"):
        assert ask in content, ask
    # Already stated constraints are echoed, not asked again.
    for kept in ("审批（准入）策略", "尽量提高通过率", "通过率不低于 60%", "坏账率不超过 5%"):
        assert kept in content, kept
    assert "还没有创建计划" in content
    for internal in _INTERNAL_WORDS:
        assert internal not in content, internal
    assert "成熟策略样本设计" not in content  # the raw prerequisite error is gone
    assert message["metadata"][REPLY_HINT_KEY]
    assert _plans(client, task_id) == []
    contract = client.get(f"/api/tasks/{task_id}").json()["strategy_input"]
    assert contract["objective"] == "max_approval"
    assert contract["min_approval_rate"] == 0.6
    assert contract["max_bad_rate"] == 0.05


def test_answer_fixes_the_sample_design_and_starts_development(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, with_dates=True)
    llm = _ClarifyingLLM()
    _install(monkeypatch, llm)
    _open_proposal(client, task_id)
    compiler_calls = len(llm.calls)

    _send(client, task_id, _ANSWER)

    plans = _plans(client, task_id)
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_development",
    ]
    assert plans[0]["status"] == "done"
    assert plans[1]["success_criteria"] == [
        {"metric": "approved_bad_rate", "max": 0.05},
        {"metric": "approval_rate", "min": 0.6},
    ]
    legacy = client.app.state.plan_repo.load_plan(plans[0]["id"]).steps[0].inputs
    assert legacy["maturity_status"] == "confirmed_matured"
    assert legacy["performance_window_days"] == 90
    assert legacy["split_col"] == "split"
    assert legacy["development_values"] == ["train"]
    assert legacy["oot_values"] == ["oot"]
    # The confirmed reply never went back through the compiler model.
    assert len(llm.calls) == compiler_calls


def test_agent_mode_answer_is_not_sent_back_to_top_level_routing(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, with_dates=True, run_mode="agent")
    llm = _AgentLLM()
    _install(monkeypatch, llm)
    _open_proposal(client, task_id)
    assert "semantic_intent_router" in llm.callers
    before = list(llm.callers)

    answered = _send(client, task_id, _ANSWER)

    assert llm.callers == before
    assert all(
        (m.get("metadata") or {}).get("code") != "semantic_intent_clarification"
        for m in answered["messages"]
    )
    assert [plan["template_id"] for plan in _plans(client, task_id)] == [
        "strategy_sample_design_v2",
        "strategy_development",
    ]


def test_questions_hold_backs_and_unmatured_labels_never_start_a_plan(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, with_dates=True)
    _install(monkeypatch, _ClarifyingLLM())
    _open_proposal(client, task_id)

    explained = _last_assistant(_send(client, task_id, "表现期是什么意思？"))
    assert "从申请开始观察多久" in explained["content"]
    unmatured = _last_assistant(_send(client, task_id, "标签还没成熟"))
    assert "只能做探索分析" in unmatured["content"]
    too_early = _last_assistant(
        _send(client, task_id, "标签已成熟，表现期 90 天，标签统计到 2026-02-01")
    )
    assert "还没满表现期" in too_early["content"]
    held = _last_assistant(_send(client, task_id, _ANSWER + "，但先别开始"))
    assert "先不开始" in held["content"]
    assert [reply["command"] for reply in held["metadata"][QUICK_REPLIES_KEY]] == [
        "开始策略开发"
    ]
    assert _plans(client, task_id) == []

    _send(client, task_id, "开始策略开发")
    assert [plan["template_id"] for plan in _plans(client, task_id)] == [
        "strategy_sample_design_v2",
        "strategy_development",
    ]


def test_month_only_sample_is_proposed_month_wise_and_round_trips() -> None:
    """D-2: a sample with only YYYY-MM months is proposed 按月口径."""

    frame = _frame(with_dates=False)
    proposal = propose_sample_design(
        frame,
        target_col="bad_flag",
        columns=frame.columns,
        field_roles={},
        request_text=_REQUEST,
    )

    assert proposal["time_field"] == {
        "column": "apply_month",
        "start": "2025-10-01",
        "end": "2026-01-31",
        "granularity": "month",
        "start_month": "2025-10",
        "end_month": "2026-01",
    }
    assert sample_design_blockers(proposal) == []
    lines = "".join(sample_design_proposal_lines(proposal))
    assert "按月口径" in lines
    assert "观察窗按整月计入（2025-10-01 至 2026-01-31）" in lines
    assert "按月末判断" in lines
    assert lines.count("apply_month") == 1  # one line, not a month line twice

    command, inputs = sample_design_command(
        proposal, parse_sample_design_answer(_ANSWER)
    )
    assert inputs["field_bindings"]["time_field"] == "apply_month"
    assert inputs["observation_window"] == {
        "status": "provided",
        "start": "2025-10-01",
        "end": "2026-01-31",
    }
    allowed = [column for column in frame.columns if column != "bad_flag"]
    grounded = ground_sample_design_command(
        command, inputs, allowed_columns=allowed, target_col="bad_flag"
    )
    assert isinstance(grounded.draft, StandardWorkflowRequestDraft)


def test_month_only_maturity_is_judged_at_month_end() -> None:
    month_frame = _frame(with_dates=False)
    day_frame = _frame(with_dates=True)
    months = propose_sample_design(
        month_frame, target_col="bad_flag", columns=month_frame.columns, field_roles={}
    )
    days = propose_sample_design(
        day_frame, target_col="bad_flag", columns=day_frame.columns, field_roles={}
    )
    # Latest day 2026-01-24 + 90 days is 2026-04-24; the latest month ends
    # 2026-01-31, so the same answer is mature by day but not by month.
    answer = parse_sample_design_answer("标签已成熟，表现期 90 天，标签统计到 2026-04-30")

    assert sample_design_maturity_gap(days, answer) is None
    gap = sample_design_maturity_gap(months, answer)
    assert gap is not None
    assert "按月口径" in gap
    assert "2026-01 按月末 2026-01-31 计" in gap
    assert "2026-05-01 才满" in gap
    assert sample_design_maturity_gap(
        months, parse_sample_design_answer("标签已成熟，表现期 90 天，标签统计到 2026-05-01")
    ) is None


def test_sample_without_day_or_month_column_is_still_blocked() -> None:
    frame = _frame(with_dates=False).drop(columns=["apply_month"])
    proposal = propose_sample_design(
        frame, target_col="bad_flag", columns=frame.columns, field_roles={}
    )

    assert proposal["time_field"] is None
    blockers = sample_design_blockers(proposal)
    assert len(blockers) == 1
    assert "申请月份列（格式如 2025-10）" in blockers[0]


def test_month_column_confirmed_with_date_role_is_still_read_month_wise() -> None:
    """A month column confirmed with the date role earlier never crashes the read."""

    frame = _frame(with_dates=False)
    proposal = propose_sample_design(
        frame,
        target_col="bad_flag",
        columns=frame.columns,
        field_roles={"apply_month": "date"},
    )

    assert proposal["time_field"]["granularity"] == "month"
    assert proposal["time_field"]["column"] == "apply_month"


def test_month_only_sample_is_fixed_month_wise_and_starts_development(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, with_dates=False)
    _install(monkeypatch, _ClarifyingLLM())

    message = _last_assistant(_open_proposal(client, task_id))

    content = message["content"]
    assert "申请月份：apply_month，2025-10 至 2026-01" in content
    assert "按月口径" in content
    assert "不能固化为正式开发样本" not in content
    for internal in _INTERNAL_WORDS:
        assert internal not in content, internal

    _send(client, task_id, _ANSWER)

    plans = _plans(client, task_id)
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_development",
    ]
    assert plans[0]["status"] == "done"

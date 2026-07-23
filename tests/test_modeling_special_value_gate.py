from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from marvis.agent.gates.adapters import parse_special_value_instruction
from marvis.agent.plan_driver import DriverError, PlanDriver
from marvis.app import create_app
from marvis.db import PlanRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.plugins.manifest import GovernancePolicy, ToolRef
from marvis.plugins.runner import ToolResult


class _ReviewerLLM:
    def complete(self, **_kwargs):
        return '{"summary":"done","open_items":[],"goal_doubt":false,"goal_met":true}'


class _Hooks:
    def dispatch(self, _event, _payload, *, task_id):
        return []


class _Tools:
    def resolve(self, _ref):
        return SimpleNamespace(
            failure_policy="fail",
            policy=GovernancePolicy(),
        )


class _Runner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[tuple[str, dict]] = []
        self._tools = _Tools()

    def invoke(self, ref, inputs, *, task_id, execution_context=None):
        self.calls.append((ref.tool, inputs))
        return ToolResult(
            ok=True,
            output=self.outputs.pop(0),
            error=None,
            error_kind=None,
            duration_ms=1,
        )


def _special_plan(*, governed: bool) -> Plan:
    screen = PlanStep(
        id="screen",
        plan_id="plan-special",
        index=0,
        title="特征筛选",
        tool_ref=ToolRef("modeling", "screen_features"),
        inputs={},
        depends_on=[],
        post_checks=[],
        phase="特征",
    )
    special = PlanStep(
        id="special",
        plan_id="plan-special",
        index=1,
        title="治理特殊值",
        tool_ref=ToolRef("modeling", "resolve_special_values"),
        inputs={
            "dataset_id": "ds-split",
            "features": "$ref:screen.output.selected",
            "sentinel_columns": "$ref:screen.output.sentinel_columns",
            "decisions": {},
        },
        depends_on=["screen"],
        post_checks=[],
        needs_confirmation=True,
        phase="特征",
        policy=GovernancePolicy(
            human_decision_gate="required" if governed else "none"
        ),
    )
    return Plan(
        id="plan-special",
        task_id="task-special",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[screen, special],
    )


def _screen_output(*, sentinel: bool) -> dict:
    return {
        "selected": ["x1", "x2"],
        "sentinel_columns": (
            {"x1": [[-999.0, 0.125]], "not_selected": [[-888.0, 0.2]]}
            if sentinel
            else {}
        ),
        "leakage": [],
        "suspected": [],
        "unusable": [],
        "ranked": [],
        "scores": {},
        "n_screened": 2,
    }


def _special_output() -> dict:
    return {
        "result_dataset_id": "ds-special",
        "selected": ["x1", "x2"],
        "governance": {},
        "policy_fingerprint": "",
        "masked": [],
        "retained": [],
        "dropped": [],
    }


def _mixed_screen_output() -> dict:
    return {
        "selected": ["x_mask", "x_retain", "x_drop"],
        "sentinel_columns": {
            "x_mask": [[-999.0, 0.125]],
            "x_retain": [[9999.0, 0.08]],
            "x_drop": [[-8888.0, 0.04]],
            "not_selected": [[7777.0, 0.2]],
        },
        "leakage": [],
        "suspected": [],
        "unusable": [],
        "ranked": [],
        "scores": {},
        "n_screened": 4,
    }


def _driver(
    tmp_path,
    *,
    sentinel: bool,
    governed: bool = False,
    screen_output: dict | None = None,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_special_plan(governed=governed))
    runner = _Runner([
        screen_output
        if screen_output is not None
        else _screen_output(sentinel=sentinel),
        _special_output(),
    ])
    executor = PlanExecutor(
        repo,
        runner,
        Reviewer(lambda: _ReviewerLLM()),
        None,
        _Hooks(),
        HarnessState(repo),
    )
    driver = PlanDriver(repo, executor)
    repo.confirm_plan("plan-special")
    return driver, repo, runner


def test_empty_special_value_gate_executes_noop_without_human_pause_or_component(tmp_path):
    driver, repo, runner = _driver(tmp_path, sentinel=False, governed=True)

    turn = driver._run_and_handle("plan-special", run_seq=0)

    assert turn.status == PlanStatus.DONE.value
    assert [tool for tool, _inputs in runner.calls] == [
        "screen_features",
        "resolve_special_values",
    ]
    assert all(message.metadata.get("special_values") is None for message in turn.messages)
    plan = repo.load_plan("plan-special")
    assert plan.status == PlanStatus.DONE
    assert next(step for step in plan.steps if step.id == "special").status == StepStatus.DONE
    assert repo.is_step_confirmed("special") is False


def test_detected_special_value_gate_exposes_only_selected_evidence_and_halts_auto(tmp_path):
    driver, repo, runner = _driver(tmp_path, sentinel=True, governed=True)

    turn = driver._run_and_handle("plan-special", run_seq=0)

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [tool for tool, _inputs in runner.calls] == ["screen_features"]
    message = turn.messages[-1]
    assert message.metadata.get("screen") is None
    assert message.metadata["special_values"]["columns"] == [
        {
            "column": "x1",
            "values": [{"value": -999.0, "share": 0.125}],
        }
    ]
    assert message.metadata["editable_input_schema"]["properties"]["decisions"][
        "required"
    ] == ["x1"]
    controls = {
        control["id"]: control
        for control in message.metadata["gate_envelope"]["controls"]
    }
    assert controls["decisions"]["required"] is True
    assert message.metadata["gate_envelope"]["human_decision_gate"] == "required"
    assert "仅回复「确认」不会越过" in message.content

    with pytest.raises(DriverError, match="AUTO.*强制人工"):
        driver.resume(
            plan_id="plan-special",
            user_text="确认",
            run_seq=1,
            expected_step_id="special",
            confirmation_source="auto",
        )
    assert repo.is_step_confirmed("special") is False


@pytest.mark.parametrize(
    "screen_output",
    [
        {"selected": ["x1"]},
        {"selected": ["x1"], "sentinel_columns": ["malformed"]},
        {"sentinel_columns": {"x1": [[-999.0, 0.125]]}},
    ],
)
def test_executor_special_value_requires_decision_fails_closed_on_malformed_evidence(
    tmp_path,
    screen_output,
):
    driver, repo, runner = _driver(
        tmp_path,
        sentinel=False,
        governed=True,
        screen_output=screen_output,
    )

    turn = driver._run_and_handle("plan-special", run_seq=0)

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert [tool for tool, _inputs in runner.calls] == ["screen_features"]
    assert repo.is_step_confirmed("special") is False
    with pytest.raises(DriverError, match="缺少有效"):
        driver.resume(
            plan_id="plan-special",
            user_text="确认",
            run_seq=1,
            expected_step_id="special",
        )


@pytest.mark.parametrize(
    ("decisions", "message"),
    [
        ({}, "必须逐列选择"),
        ([], "decisions 必须"),
        ({"ghost": {"action": "drop"}}, "必须逐列选择"),
        (
            {
                "x1": {"action": "mask"},
                "ghost": {"action": "drop"},
            },
            "未选或未检测到特殊值",
        ),
        (
            {"x1": {"action": "retain", "confirmed": True}},
            "需要填写理由",
        ),
    ],
)
def test_special_value_structured_decisions_fail_closed_without_mutation(
    tmp_path,
    decisions,
    message,
):
    driver, repo, runner = _driver(tmp_path, sentinel=True)
    driver._run_and_handle("plan-special", run_seq=0)

    with pytest.raises(DriverError, match=message):
        driver.resume(
            plan_id="plan-special",
            user_text="确认",
            run_seq=1,
            expected_step_id="special",
            adjust_params={"decisions": decisions},
        )

    special = next(
        step for step in repo.load_plan("plan-special").steps if step.id == "special"
    )
    assert special.status == StepStatus.AWAITING_CONFIRM
    assert special.inputs["decisions"] == {}
    assert repo.is_step_confirmed("special") is False
    assert [tool for tool, _inputs in runner.calls] == ["screen_features"]


def test_special_value_manual_submission_atomically_confirms_inputs_without_values(tmp_path):
    driver, repo, runner = _driver(tmp_path, sentinel=True)
    driver._run_and_handle("plan-special", run_seq=0)

    turn = driver.resume(
        plan_id="plan-special",
        user_text="确认",
        run_seq=1,
        expected_step_id="special",
        adjust_params={"decisions": {"x1": {"action": "mask"}}},
    )

    assert turn.status == PlanStatus.DONE.value
    special_call = runner.calls[-1]
    assert special_call[0] == "resolve_special_values"
    assert special_call[1]["decisions"] == {"x1": {"action": "mask"}}
    assert "values" not in special_call[1]["decisions"]["x1"]
    persisted = next(
        step for step in repo.load_plan("plan-special").steps if step.id == "special"
    )
    assert persisted.inputs["decisions"] == {"x1": {"action": "mask"}}
    assert repo.is_step_confirmed("special") is True


def test_special_value_manual_mixed_policy_is_complete_and_never_trusts_ui_values(
    tmp_path,
):
    driver, repo, runner = _driver(
        tmp_path,
        sentinel=False,
        screen_output=_mixed_screen_output(),
    )
    driver._run_and_handle("plan-special", run_seq=0)
    decisions = {
        "x_mask": {"action": "mask"},
        "x_retain": {
            "action": "retain",
            "confirmed": True,
            "reason": "该编码由上游业务字典明确定义",
        },
        "x_drop": {"action": "drop"},
    }

    turn = driver.resume(
        plan_id="plan-special",
        user_text="确认",
        run_seq=1,
        expected_step_id="special",
        adjust_params={"decisions": decisions},
    )

    assert turn.status == PlanStatus.DONE.value
    assert runner.calls[-1][1]["decisions"] == decisions
    assert all(
        "values" not in decision
        for decision in runner.calls[-1][1]["decisions"].values()
    )
    persisted = next(
        step for step in repo.load_plan("plan-special").steps
        if step.id == "special"
    )
    assert persisted.inputs["decisions"] == decisions


def test_special_value_partial_policy_is_rejected_against_all_relevant_columns(
    tmp_path,
):
    driver, repo, runner = _driver(
        tmp_path,
        sentinel=False,
        screen_output=_mixed_screen_output(),
    )
    driver._run_and_handle("plan-special", run_seq=0)

    with pytest.raises(DriverError, match="x_retain.*x_drop"):
        driver.resume(
            plan_id="plan-special",
            user_text="确认",
            run_seq=1,
            expected_step_id="special",
            adjust_params={
                "decisions": {"x_mask": {"action": "mask"}},
            },
        )

    assert repo.is_step_confirmed("special") is False
    assert [tool for tool, _inputs in runner.calls] == ["screen_features"]


def test_special_value_bare_confirmation_is_rejected(tmp_path):
    driver, repo, runner = _driver(tmp_path, sentinel=True)
    driver._run_and_handle("plan-special", run_seq=0)

    with pytest.raises(DriverError, match="必须逐列选择"):
        driver.resume(
            plan_id="plan-special",
            user_text="确认",
            run_seq=1,
            expected_step_id="special",
        )

    assert repo.is_step_confirmed("special") is False
    assert [tool for tool, _inputs in runner.calls] == ["screen_features"]


def test_special_value_agent_language_compiles_complete_policy_and_rejects_unknown(tmp_path):
    driver, _repo, runner = _driver(
        tmp_path,
        sentinel=False,
        screen_output=_mixed_screen_output(),
    )
    driver._run_and_handle("plan-special", run_seq=0)

    turn = driver.resume(
        plan_id="plan-special",
        user_text=(
            "x_mask 转为空值；"
            "x_retain 保留，原因：业务约定值；"
            "x_drop 删除"
        ),
        run_seq=1,
        expected_step_id="special",
    )

    assert turn.status == PlanStatus.DONE.value
    assert runner.calls[-1][1]["decisions"] == {
        "x_mask": {
            "action": "mask",
            "values": [-999.0],
        },
        "x_retain": {
            "action": "retain",
            "values": [9999.0],
            "confirmed": True,
            "reason": "业务约定值",
        },
        "x_drop": {
            "action": "drop",
            "values": [-8888.0],
        },
    }

    parsed = parse_special_value_instruction(
        "x_mask 转为空值；ghost 删除",
        selected=["x_mask"],
        sentinel_columns={"x_mask": [[-999.0, 0.125]]},
    )
    assert parsed == {
        "x_mask": {
            "action": "mask",
            "values": [-999.0],
        },
        "ghost": {
            "action": "drop",
        },
    }


def test_special_value_agent_unknown_column_and_retain_without_reason_stay_at_gate(
    tmp_path,
):
    unknown_driver, unknown_repo, unknown_runner = _driver(
        tmp_path / "unknown",
        sentinel=True,
    )
    unknown_driver._run_and_handle("plan-special", run_seq=0)

    unknown_turn = unknown_driver.resume(
        plan_id="plan-special",
        user_text="x1 转为空值；ghost 删除",
        run_seq=1,
        expected_step_id="special",
    )

    assert unknown_turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert "未选或未检测到特殊值" in unknown_turn.messages[-1].content
    assert unknown_repo.is_step_confirmed("special") is False
    assert [tool for tool, _inputs in unknown_runner.calls] == ["screen_features"]

    retain_driver, retain_repo, retain_runner = _driver(
        tmp_path / "retain",
        sentinel=True,
    )
    retain_driver._run_and_handle("plan-special", run_seq=0)

    retain_turn = retain_driver.resume(
        plan_id="plan-special",
        user_text="x1 保留",
        run_seq=1,
        expected_step_id="special",
    )

    assert retain_turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert retain_repo.is_step_confirmed("special") is False
    assert [tool for tool, _inputs in retain_runner.calls] == ["screen_features"]


def test_special_value_language_parser_is_exact_and_does_not_infer_negated_or_global_actions():
    sentinel_columns = {
        "x1": [[-999.0, 0.1]],
        "x10": [[9999.0, 0.2]],
    }

    assert parse_special_value_instruction(
        "x10 删除",
        selected=["x1", "x10"],
        sentinel_columns=sentinel_columns,
    ) == {
        "x10": {"action": "drop", "values": [9999.0]},
    }
    assert parse_special_value_instruction(
        "x1 不删除",
        selected=["x1"],
        sentinel_columns={"x1": sentinel_columns["x1"]},
    ) is None
    assert parse_special_value_instruction(
        "x1 保留，原因：都是业务约定值",
        selected=["x1", "x10"],
        sentinel_columns=sentinel_columns,
    ) == {
        "x1": {
            "action": "retain",
            "values": [-999.0],
            "confirmed": True,
            "reason": "都是业务约定值",
        },
    }


def test_special_value_manual_http_submission_confirms_and_executes_atomically(
    tmp_path,
):
    app = create_app(tmp_path)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="special-value-http",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="modeling",
            run_mode="manual",
        )
    )
    plan = _special_plan(governed=True)
    plan.task_id = task.id
    plan.status = PlanStatus.AWAITING_CONFIRM
    plan.steps[0].status = StepStatus.DONE
    plan.steps[1].status = StepStatus.AWAITING_CONFIRM
    repo = app.state.plan_repo
    repo.create_plan(plan)
    repo.store_step_output("screen", _screen_output(sentinel=True))
    runner = _Runner([_special_output()])
    app.state.plan_executor._runner = runner

    with TestClient(app) as client:
        response = client.post(
            f"/api/tasks/{task.id}/agent/messages",
            json={
                "content": "确认",
                "ui_action": "confirm_gate",
                "expected_step_id": "special",
                "adjust_params": {
                    "decisions": {"x1": {"action": "mask"}},
                },
            },
        )

    assert response.status_code == 202, response.text
    persisted = next(
        step for step in repo.load_plan("plan-special").steps
        if step.id == "special"
    )
    assert persisted.inputs["decisions"] == {"x1": {"action": "mask"}}
    assert repo.is_step_confirmed("special") is True
    assert runner.calls[-1] == (
        "resolve_special_values",
        {
            "dataset_id": "ds-split",
            "features": ["x1", "x2"],
            "sentinel_columns": {
                "x1": [[-999.0, 0.125]],
                "not_selected": [[-888.0, 0.2]],
            },
            "decisions": {"x1": {"action": "mask"}},
        },
    )
    assert "values" not in runner.calls[-1][1]["decisions"]["x1"]

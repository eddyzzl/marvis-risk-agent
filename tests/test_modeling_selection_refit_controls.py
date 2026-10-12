"""Exact candidate/refit controls share the existing selection confirmation CAS."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from marvis.agent.plan_driver import DriverError
from marvis.api_schemas import AgentMessageRequest
from marvis.orchestrator.contracts import PlanStatus, plan_fingerprint, plan_step_confirmation_fingerprint
from marvis.plugins.manifest import ToolRef
from marvis.routers.validation_agent import _validate_ui_action_contract
from tests.test_plan_driver import _selection_gate_driver


@pytest.mark.parametrize("params", [
    {"selected_experiment_id": "exp-current"},
    {"selected_experiment_id": "exp-current", "refit_on_train_plus_test": False},
    {"selected_experiment_id": "exp-current", "refit_on_train_plus_test": True},
    {"selected_experiment_id": "exp-current", "refit_on_train_plus_test": None},
    {"decisions": {"field": "keep"}},
])
def test_http_confirm_gate_accepts_exact_optional_refit_command(params):
    payload = AgentMessageRequest(content="确认", ui_action="confirm_gate", adjust_params=params)
    _validate_ui_action_contract(payload)


@pytest.mark.parametrize("params", [
    {"refit_on_train_plus_test": False},
    {"selected_experiment_id": "exp-current", "refit_on_train_plus_test": False, "unknown": True},
    *({"selected_experiment_id": "exp-current", "refit_on_train_plus_test": value}
      for value in [0, 1, "false", "true", [], {}, 0.0]),
])
def test_http_confirm_gate_refuses_untyped_or_unknown_refit_control(params):
    payload = AgentMessageRequest(content="确认", ui_action="confirm_gate", adjust_params=params)
    with pytest.raises(HTTPException) as failure:
        _validate_ui_action_contract(payload)
    assert failure.value.status_code == 422


def _confirm(driver, selected_id, **kwargs):
    return driver.resume(
        plan_id="plan-1",
        user_text="确认",
        run_seq=1,
        expected_step_id="select",
        confirmation_source="human",
        adjust_params={"selected_experiment_id": selected_id, **kwargs},
    )


def _assert_not_selected(repo, runner):
    assert repo.load_plan("plan-1").status is PlanStatus.AWAITING_CONFIRM
    assert repo.is_step_confirmed("select") is False
    assert [call[0] for call in runner.calls] == ["train_models", "compare_experiments"]
    assert "selected_experiment_id" not in repo.load_plan("plan-1").steps[-1].inputs
    assert "refit_on_train_plus_test" not in repo.load_plan("plan-1").steps[-1].inputs


@pytest.mark.parametrize("refit", [False, True])
def test_explicit_refit_and_current_candidate_are_bound_once_in_same_cas(tmp_path, monkeypatch, refit):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    snapshot = repo.load_plan("plan-1")
    gate = snapshot.steps[-1]
    original_confirm = repo.confirm_step_with_inputs
    confirmations = []

    def record_confirmation(step_id, **kwargs):
        confirmations.append((step_id, kwargs))
        return original_confirm(step_id, **kwargs)

    monkeypatch.setattr(repo, "confirm_step_with_inputs", record_confirmation)
    turn = _confirm(driver, selected_id, refit_on_train_plus_test=refit)

    assert turn.status == PlanStatus.DONE.value
    assert len(confirmations) == 1
    step_id, binding = confirmations[0]
    assert step_id == "select"
    assert binding["input_updates"] == {
        "selected_experiment_id": selected_id, "refit_on_train_plus_test": refit,
    }
    assert binding["expected_plan_fingerprint"] == plan_fingerprint(snapshot)
    assert binding["expected_step_fingerprint"] == plan_step_confirmation_fingerprint(gate, confirmed=False)
    assert binding["expected_plan_revision"] == snapshot.replan_count
    assert binding["expected_plan_status"] == PlanStatus.AWAITING_CONFIRM.value
    assert repo.load_plan("plan-1").steps[-1].inputs["refit_on_train_plus_test"] is refit
    assert runner.calls[-1][1]["refit_on_train_plus_test"] is refit
    assert runner.calls[-1][1]["selected_experiment_id"] == selected_id


@pytest.mark.parametrize("params", [{}, {"refit_on_train_plus_test": None}])
def test_omitted_or_none_refit_retains_old_selection_inputs(tmp_path, monkeypatch, params):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    original_confirm = repo.confirm_step_with_inputs
    updates = []

    def record_confirmation(step_id, **kwargs):
        updates.append(kwargs["input_updates"])
        return original_confirm(step_id, **kwargs)

    monkeypatch.setattr(repo, "confirm_step_with_inputs", record_confirmation)
    assert _confirm(driver, selected_id, **params).status == PlanStatus.DONE.value
    assert updates == [{"selected_experiment_id": selected_id}]
    assert "refit_on_train_plus_test" not in runner.calls[-1][1]
    assert "refit_on_train_plus_test" not in repo.load_plan("plan-1").steps[-1].inputs


@pytest.mark.parametrize("value", [0, 1, "false", "true", "", [], {}, 0.0])
def test_nonboolean_refit_refuses_without_confirming_or_executing(tmp_path, value):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    with pytest.raises(DriverError, match="必须是布尔值"):
        _confirm(driver, selected_id, refit_on_train_plus_test=value)
    _assert_not_selected(repo, runner)


def test_unknown_selection_control_refuses_without_widening_adjustments(tmp_path):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    with pytest.raises(DriverError, match="仅可附带 refit_on_train_plus_test"):
        _confirm(driver, selected_id, refit_on_train_plus_test=False, new_policy="ignored")
    _assert_not_selected(repo, runner)


def test_refit_control_still_requires_an_authenticated_current_candidate(tmp_path):
    driver, repo, runner, _, _ = _selection_gate_driver(tmp_path)
    with pytest.raises(DriverError, match="不在当前任务的候选集合"):
        _confirm(driver, "experiment_obsolete", refit_on_train_plus_test=False)
    _assert_not_selected(repo, runner)


def test_refit_control_is_specific_to_installed_modeling_selection(tmp_path):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    gate = repo.load_plan("plan-1").steps[-1]
    gate.tool_ref = ToolRef("other_plugin", "select_experiment")
    repo.update_step(gate)
    with pytest.raises(DriverError, match="当前节点不接受带参数的确认"):
        _confirm(driver, selected_id, refit_on_train_plus_test=False)
    _assert_not_selected(repo, runner)


def test_refit_control_does_not_authorize_selection_without_confirmation(tmp_path):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    with pytest.raises(DriverError, match="必须同时明确授权"):
        driver.resume(
            plan_id="plan-1", user_text="暂不确认", expected_step_id="select",
            adjust_params={"selected_experiment_id": selected_id, "refit_on_train_plus_test": False},
        )
    _assert_not_selected(repo, runner)


def test_refit_control_cannot_confirm_an_old_step(tmp_path):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    with pytest.raises(DriverError, match="当前待确认步骤已变化"):
        driver.resume(
            plan_id="plan-1", user_text="确认", expected_step_id="old-step",
            adjust_params={"selected_experiment_id": selected_id, "refit_on_train_plus_test": False},
        )
    _assert_not_selected(repo, runner)


def test_refit_control_cannot_confirm_an_obsolete_ui_snapshot(tmp_path):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    snapshot = repo.load_plan("plan-1")
    gate = snapshot.steps[-1]
    rendered_step_fingerprint = plan_step_confirmation_fingerprint(gate, confirmed=False)
    rendered_plan_fingerprint = plan_fingerprint(snapshot)
    gate.inputs = {**gate.inputs, "target_type": "continuous"}
    repo.update_step(gate)
    with pytest.raises(DriverError, match="计划内容已变化"):
        driver.resume(
            plan_id="plan-1", user_text="确认", expected_step_id="select",
            expected_plan_status=snapshot.status.value,
            expected_plan_revision=snapshot.replan_count,
            expected_plan_fingerprint=rendered_plan_fingerprint,
            expected_step_fingerprint=rendered_step_fingerprint,
            _trusted_ui_action=True,
            adjust_params={"selected_experiment_id": selected_id, "refit_on_train_plus_test": False},
        )
    _assert_not_selected(repo, runner)


def test_refit_and_candidate_cas_rejects_intervening_change_atomically(tmp_path, monkeypatch):
    driver, repo, runner, selected_id, _ = _selection_gate_driver(tmp_path)
    original_confirm = repo.confirm_step_with_inputs

    def change_before_confirmation(step_id, **kwargs):
        gate = repo.load_plan("plan-1").steps[-1]
        gate.inputs = {**gate.inputs, "selection_policy": {"changed": True}}
        repo.update_step(gate)
        return original_confirm(step_id, **kwargs)

    monkeypatch.setattr(repo, "confirm_step_with_inputs", change_before_confirmation)
    with pytest.raises(DriverError, match="当前确认节点在执行前已变化"):
        _confirm(driver, selected_id, refit_on_train_plus_test=False)
    _assert_not_selected(repo, runner)
    assert repo.load_plan("plan-1").steps[-1].inputs["selection_policy"] == {"changed": True}

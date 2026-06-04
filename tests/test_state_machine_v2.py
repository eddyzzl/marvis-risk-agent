import pytest

from riskmodel_checker.domain import TaskStatus
from riskmodel_checker.state_machine import IllegalTransition, assert_transition


def test_pipeline_happy_path_transitions():
    assert_transition(TaskStatus.CREATED, TaskStatus.SCANNED)
    assert_transition(TaskStatus.SCANNED, TaskStatus.RUNNING)
    assert_transition(TaskStatus.RUNNING, TaskStatus.EXECUTED)
    assert_transition(TaskStatus.EXECUTED, TaskStatus.COMPUTING_METRICS)
    assert_transition(TaskStatus.COMPUTING_METRICS, TaskStatus.WRITING_ARTIFACTS)
    assert_transition(TaskStatus.WRITING_ARTIFACTS, TaskStatus.SUCCEEDED)


def test_writing_artifacts_to_review_required_allowed():
    assert_transition(TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED)


def test_any_state_to_failed_allowed():
    for state in TaskStatus:
        if state is TaskStatus.FAILED:
            continue
        assert_transition(state, TaskStatus.FAILED)


def test_failed_back_to_running_for_retry():
    assert_transition(TaskStatus.FAILED, TaskStatus.RUNNING)


@pytest.mark.parametrize("terminal", [TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED])
def test_terminal_tasks_can_rerun_prior_workflow_steps(terminal):
    assert_transition(terminal, TaskStatus.SCANNED)
    assert_transition(terminal, TaskStatus.RUNNING)
    assert_transition(terminal, TaskStatus.COMPUTING_METRICS)


def test_metrics_ready_task_can_rerun_earlier_workflow_steps():
    assert_transition(TaskStatus.WRITING_ARTIFACTS, TaskStatus.SCANNED)
    assert_transition(TaskStatus.WRITING_ARTIFACTS, TaskStatus.RUNNING)
    assert_transition(TaskStatus.WRITING_ARTIFACTS, TaskStatus.COMPUTING_METRICS)


def test_failed_back_to_computing_metrics_for_metrics_retry():
    assert_transition(TaskStatus.FAILED, TaskStatus.COMPUTING_METRICS)


def test_failed_can_stay_failed_for_repeated_scan_failures():
    assert_transition(TaskStatus.FAILED, TaskStatus.FAILED)


def test_running_back_to_scanned_after_notebook_cancel():
    assert_transition(TaskStatus.RUNNING, TaskStatus.SCANNED)


def test_disallowed_transition_raises():
    with pytest.raises(IllegalTransition):
        assert_transition(TaskStatus.CREATED, TaskStatus.SUCCEEDED)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.notebook_cancellation import (
    register_notebook_cancellation,
    request_active_notebook_cancellation,
    unregister_notebook_cancellation,
)
from marvis.pipeline import _metrics_cancel_marker_path
from marvis.routers.stage_controls import router
from marvis.settings import build_settings


def _client_and_task(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A-card",
            model_version="v1",
            validator="qa",
            source_dir=str(settings.workspace),
        )
    )
    with repo.transaction() as conn:
        conn.execute(
            "UPDATE tasks SET validation_workflow_version = 1 WHERE id = ?",
            (task.id,),
        )
    task = repo.get_task(task.id)
    return TestClient(app), settings, repo, task


def _advance_to_running(repo: TaskRepository, task_id: str) -> None:
    repo.update_status(task_id, TaskStatus.SCANNED, "scanned")
    repo.update_status(task_id, TaskStatus.RUNNING, "running")


def test_notebook_cancel_rejects_wrong_kind_job_without_pending_poison(tmp_path):
    client, _settings, repo, task = _client_and_task(tmp_path)
    _advance_to_running(repo, task.id)
    repo.start_job(task.id, "metrics")

    response = client.post(f"/api/tasks/{task.id}/notebook/cancel")
    token = register_notebook_cancellation(task.id)
    try:
        assert response.status_code == 409
        assert token.is_cancelled() is False
    finally:
        unregister_notebook_cancellation(task.id, token)


def test_metrics_cancel_rejects_missing_job_without_marker_or_pending_poison(tmp_path):
    client, settings, repo, task = _client_and_task(tmp_path)
    _advance_to_running(repo, task.id)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed")
    repo.update_status(task.id, TaskStatus.COMPUTING_METRICS, "metrics")
    marker = _metrics_cancel_marker_path(settings.tasks_dir / task.id)

    response = client.post(f"/api/tasks/{task.id}/metrics/cancel")
    token = register_notebook_cancellation(task.id)
    try:
        assert response.status_code == 409
        assert marker.exists() is False
        assert token.is_cancelled() is False
    finally:
        unregister_notebook_cancellation(task.id, token)


def test_notebook_cancel_does_not_poison_retry_when_job_outlives_token(tmp_path):
    client, _settings, repo, task = _client_and_task(tmp_path)
    _advance_to_running(repo, task.id)
    old_job_id = repo.start_job(task.id, "notebook")
    repo.mark_job_running(old_job_id)

    response = client.post(f"/api/tasks/{task.id}/notebook/cancel")
    repo.finish_job(old_job_id, status="succeeded")
    retry_token = register_notebook_cancellation(task.id)
    try:
        assert response.status_code == 409
        assert retry_token.is_cancelled() is False
    finally:
        unregister_notebook_cancellation(task.id, retry_token)


def test_metrics_cancel_does_not_leave_marker_or_poison_retry_without_token(tmp_path):
    client, settings, repo, task = _client_and_task(tmp_path)
    _advance_to_running(repo, task.id)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed")
    repo.update_status(task.id, TaskStatus.COMPUTING_METRICS, "metrics")
    old_job_id = repo.start_job(task.id, "metrics")
    repo.mark_job_running(old_job_id)
    marker = _metrics_cancel_marker_path(settings.tasks_dir / task.id)

    response = client.post(f"/api/tasks/{task.id}/metrics/cancel")
    repo.finish_job(old_job_id, status="succeeded")
    retry_token = register_notebook_cancellation(task.id)
    try:
        assert response.status_code == 409
        assert marker.exists() is False
        assert retry_token.is_cancelled() is False
    finally:
        unregister_notebook_cancellation(task.id, retry_token)


def test_notebook_cancel_cannot_hit_retry_registered_after_job_lookup(
    tmp_path,
    monkeypatch,
):
    client, _settings, repo, task = _client_and_task(tmp_path)
    _advance_to_running(repo, task.id)
    old_job_id = repo.start_job(task.id, "notebook")
    repo.mark_job_running(old_job_id)
    old_token = register_notebook_cancellation(task.id, job_id=old_job_id)
    retry: dict[str, object] = {}

    def swap_execution_before_delivery(task_id, *, expected_job_id=None):
        unregister_notebook_cancellation(task_id, old_token)
        repo.finish_job(old_job_id, status="cancelled")
        retry_job_id = repo.start_job(task_id, "notebook")
        repo.mark_job_running(retry_job_id)
        retry_token = register_notebook_cancellation(task_id, job_id=retry_job_id)
        retry.update(job_id=retry_job_id, token=retry_token)
        return request_active_notebook_cancellation(
            task_id,
            expected_job_id=expected_job_id,
        )

    monkeypatch.setattr(
        "marvis.routers.stage_controls.request_active_notebook_cancellation",
        swap_execution_before_delivery,
    )
    try:
        response = client.post(f"/api/tasks/{task.id}/notebook/cancel")

        assert response.status_code == 409
        assert retry["job_id"] != old_job_id
        assert retry["token"].is_cancelled() is False
    finally:
        retry_token = retry.get("token")
        if retry_token is not None:
            unregister_notebook_cancellation(task.id, retry_token)
        retry_job_id = retry.get("job_id")
        if retry_job_id is not None:
            repo.finish_job(str(retry_job_id), status="cancelled")


@pytest.mark.parametrize(
    ("status", "endpoint"),
    [
        (TaskStatus.RUNNING, "notebook"),
        (TaskStatus.COMPUTING_METRICS, "metrics"),
        (TaskStatus.WRITING_ARTIFACTS, "report"),
    ],
)
def test_stage_cancel_accepts_job_owned_by_full_pipeline(
    tmp_path,
    status,
    endpoint,
):
    client, _settings, repo, task = _client_and_task(tmp_path)
    _advance_to_running(repo, task.id)
    if status in {TaskStatus.COMPUTING_METRICS, TaskStatus.WRITING_ARTIFACTS}:
        repo.update_status(task.id, TaskStatus.EXECUTED, "executed")
        repo.update_status(task.id, TaskStatus.COMPUTING_METRICS, "metrics")
    if status == TaskStatus.WRITING_ARTIFACTS:
        repo.update_status(task.id, TaskStatus.WRITING_ARTIFACTS, "writing")
    job_id = repo.start_job(task.id, "pipeline")
    repo.mark_job_running(job_id)
    token = register_notebook_cancellation(task.id, job_id=job_id)
    try:
        response = client.post(f"/api/tasks/{task.id}/{endpoint}/cancel")

        assert response.status_code == 202
        assert token.is_cancelled() is True
    finally:
        unregister_notebook_cancellation(task.id, token)
        repo.finish_job(job_id, status="cancelled")

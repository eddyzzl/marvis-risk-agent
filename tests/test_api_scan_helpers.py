from types import SimpleNamespace

import pytest

from marvis.api_scan_helpers import perform_scan_task
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus


class FailingScanStatusRepository(TaskRepository):
    def update_status_on_connection(self, *args, **kwargs):
        raise RuntimeError("status write failed")


def test_perform_scan_task_rolls_back_artifacts_when_status_update_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(TaskCreate(
        model_name="A卡",
        model_version="v1",
        validator="qa",
        source_dir=str(tmp_path / "source"),
    ))
    task_dir = tmp_path / "tasks" / task.id
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    images_dir = task_dir / "images"
    execution_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    (execution_dir / "scan_result.json").write_text('{"old": true}', encoding="utf-8")
    (execution_dir / "notebook_steps.json").write_text('{"steps": ["old"]}', encoding="utf-8")
    (outputs_dir / "validation.xlsx").write_bytes(b"old-xlsx")
    (images_dir / "roc.png").write_bytes(b"old-png")
    monkeypatch.setattr("marvis.api_scan_helpers.scan_source_dir", lambda _path: [])

    failing_repo = FailingScanStatusRepository(db_path)
    settings = SimpleNamespace(tasks_dir=tmp_path / "tasks")
    with pytest.raises(RuntimeError, match="status write failed"):
        perform_scan_task(failing_repo, task, settings)

    assert (execution_dir / "scan_result.json").read_text(encoding="utf-8") == '{"old": true}'
    assert (execution_dir / "notebook_steps.json").read_text(encoding="utf-8") == '{"steps": ["old"]}'
    assert (outputs_dir / "validation.xlsx").read_bytes() == b"old-xlsx"
    assert (images_dir / "roc.png").read_bytes() == b"old-png"
    assert repo.get_task(task.id).status == TaskStatus.CREATED

from types import SimpleNamespace
import sqlite3
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.dataset_source_gc import DatasetSourceGarbageCollector
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, DraftRepository, TaskRepository, connect
from marvis.drafts.contracts import DraftRun, DraftTool
from marvis.notebooks import (
    close_live_notebook_session,
    get_live_notebook_session,
    register_live_notebook_session,
)
from marvis.state_machine import ConflictError


def _client(tmp_path, *, raise_server_exceptions: bool = True):
    app = create_app(tmp_path / "workspace")
    settings = app.state.settings
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), settings


def _create_task(client):
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "A card",
            "model_version": "v1",
            "validator": "qa",
            "source_dir": str(client.app.state.settings.workspace),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_dataset(client, task_id: str, frame: pd.DataFrame, *, name: str = "sample.csv"):
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    response = client.post(
        f"/api/tasks/{task_id}/datasets/upload",
        data={"role": "sample"},
        files={"file": (name, csv_bytes, "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()["datasets"][0]


def _activate_and_analyze(client, task_id: str, dataset: dict) -> str:
    activated = client.put(
        f"/api/tasks/{task_id}/data-workspace",
        headers={"If-Match": "0"},
        json={
            "active_dataset_id": dataset["id"],
            "active_dataset_content_hash": dataset["content_hash"],
            "page": "overview",
            "selected_field": None,
            "semantic_mapping": {
                "target_col": None,
                "field_roles": {},
                "business_names": {},
            },
        },
    )
    assert activated.status_code == 200, activated.text
    accepted = client.post(
        f"/api/tasks/{task_id}/data-analysis",
        headers={"If-Match": str(activated.json()["revision"])},
        json={"sections": ["overview"]},
    )
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run_id"]
    completed = client.get(f"/api/tasks/{task_id}/data-analysis/{run_id}")
    assert completed.status_code == 200
    assert completed.json()["status"] == "succeeded"
    return run_id


def test_purge_preview_endpoint_lists_expected_counts_without_deleting(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(client, task["id"], pd.DataFrame({"acct_id": ["A1", "B2"]}))
    dataset_path = settings.datasets_dir / dataset["source_path"]
    assert dataset_path.exists()

    preview = client.get(f"/api/tasks/{task['id']}/purge-preview")

    assert preview.status_code == 200
    body = preview.json()
    assert body["task_id"] == task["id"]
    assert body["purge_summary"]["datasets"] == 1
    assert "dataset_source_paths" not in body["purge_summary"]
    # dry-run leaves everything in place
    assert dataset_path.exists()
    assert TaskRepository(settings.db_path).get_task(task["id"]).id == task["id"]


def test_purge_preview_returns_404_for_unknown_task(tmp_path):
    client, _settings = _client(tmp_path)

    response = client.get("/api/tasks/does-not-exist/purge-preview")

    assert response.status_code == 404


def test_delete_task_removes_dataset_files_and_writes_delete_audit(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(client, task["id"], pd.DataFrame({"acct_id": ["A1", "B2"]}))
    dataset_path = settings.datasets_dir / dataset["source_path"]
    source_identity_dir = settings.datasets_dir / task["id"] / ".source-identities"
    assert dataset_path.exists()
    assert list(source_identity_dir.glob("*.json"))

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert not dataset_path.exists()
    assert not source_identity_dir.exists()
    assert not (settings.tasks_dir / task["id"]).exists()

    from marvis.repositories.audit import _list_audit_rows

    audit_rows = _list_audit_rows(settings.db_path, kind="task.delete")
    assert len(audit_rows) == 1
    assert audit_rows[0]["target_ref"] == task["id"]
    assert audit_rows[0]["detail"]["purge_summary"]["datasets"] == 1

    with pytest.raises(KeyError):
        TaskRepository(settings.db_path).get_task(task["id"])


def test_delete_task_purges_data_workspace_and_analysis_before_dataset(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    _activate_and_analyze(client, task["id"], dataset)

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    with connect(settings.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM data_analysis_runs WHERE task_id = ?",
            (task["id"],),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM data_workspaces WHERE task_id = ?",
            (task["id"],),
        ).fetchone()[0] == 0
    from marvis.repositories.audit import _list_audit_rows

    audit = _list_audit_rows(settings.db_path, kind="task.delete")[-1]
    assert audit["detail"]["purge_summary"]["data_analysis_runs"] == 1
    assert audit["detail"]["purge_summary"]["data_workspaces"] == 1


def test_task_purge_rolls_back_analysis_and_workspace_on_later_failure(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    _activate_and_analyze(client, task["id"], dataset)
    with connect(settings.db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER abort_test_dataset_delete
            BEFORE DELETE ON datasets
            BEGIN
                SELECT RAISE(ABORT, 'synthetic dataset purge failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="synthetic dataset purge"):
        TaskRepository(settings.db_path).purge_task(task["id"])

    with connect(settings.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE id = ?", (task["id"],)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE task_id = ?", (task["id"],)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM data_workspaces WHERE task_id = ?",
            (task["id"],),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM data_analysis_runs WHERE task_id = ?",
            (task["id"],),
        ).fetchone()[0] == 1


def test_delete_task_keeps_dataset_file_still_referenced_by_another_task(tmp_path):
    # GAP-2 x GAP-7: uses the real content-fingerprint dedup upload path (see
    # GET /api/tasks/{id}/purge-preview and register_from_upload) so both tasks'
    # dataset rows point at the same physical parquet file.
    client, settings = _client(tmp_path)
    owner_task = _create_task(client)
    frame = pd.DataFrame({"acct_id": ["A1", "B2"]})
    dataset = _upload_dataset(client, owner_task["id"], frame)
    dataset_path = settings.datasets_dir / dataset["source_path"]
    assert dataset_path.exists()

    other_task = _create_task(client)
    other_dataset = _upload_dataset(client, other_task["id"], frame)
    assert other_dataset["source_path"] == dataset["source_path"]

    response = client.delete(f"/api/tasks/{owner_task['id']}")

    assert response.status_code == 204
    # the physical file is still referenced by other_task's dataset row
    assert dataset_path.exists()

    second_response = client.delete(f"/api/tasks/{other_task['id']}")

    assert second_response.status_code == 204
    # once the last referencing task is gone, the file is finally removed
    assert not dataset_path.exists()


def test_delete_task_removes_last_content_addressed_snapshot_reference(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    pinned = registry.pin_authenticated_snapshot(dataset["id"])
    pinned_path = settings.datasets_dir / pinned.source_path
    pinned_dir = pinned_path.parent
    assert pinned_path.exists()

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert not pinned_path.exists()
    assert not pinned_dir.exists()


def test_delete_task_keeps_shared_content_addressed_snapshot_until_last_reference(
    tmp_path,
):
    client, settings = _client(tmp_path)
    frame = pd.DataFrame({"acct_id": ["A1", "B2"]})
    first_task = _create_task(client)
    first_dataset = _upload_dataset(client, first_task["id"], frame)
    second_task = _create_task(client)
    second_dataset = _upload_dataset(client, second_task["id"], frame)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    first_pinned = registry.pin_authenticated_snapshot(first_dataset["id"])
    second_pinned = registry.pin_authenticated_snapshot(second_dataset["id"])
    assert second_pinned.source_path == first_pinned.source_path
    pinned_path = settings.datasets_dir / first_pinned.source_path
    pinned_dir = pinned_path.parent

    first_response = client.delete(f"/api/tasks/{first_task['id']}")

    assert first_response.status_code == 204
    assert pinned_path.exists()

    second_response = client.delete(f"/api/tasks/{second_task['id']}")

    assert second_response.status_code == 204
    assert not pinned_path.exists()
    assert not pinned_dir.exists()


def test_delete_task_rechecks_cas_references_after_purge_before_unlink(
    tmp_path,
):
    client, settings = _client(tmp_path)
    frame = pd.DataFrame({"acct_id": ["A1", "B2"]})
    owner_task = _create_task(client)
    owner_dataset = _upload_dataset(client, owner_task["id"], frame)
    other_task = _create_task(client)
    other_dataset = _upload_dataset(client, other_task["id"], frame)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    owner_pinned = registry.pin_authenticated_snapshot(owner_dataset["id"])
    pinned_path = settings.datasets_dir / owner_pinned.source_path
    cleanup_entered = threading.Event()
    resume_cleanup = threading.Event()
    original_collector = client.app.state.dataset_source_gc

    class PausingCollector:
        def sweep(self, **kwargs):
            cleanup_entered.set()
            assert resume_cleanup.wait(timeout=5)
            return original_collector.sweep(**kwargs)

    client.app.state.dataset_source_gc = PausingCollector()
    outcome: dict[str, object] = {}

    def delete_owner():
        outcome["response"] = client.delete(f"/api/tasks/{owner_task['id']}")

    worker = threading.Thread(target=delete_owner)
    worker.start()
    assert cleanup_entered.wait(timeout=5)
    try:
        other_pinned = registry.pin_authenticated_snapshot(other_dataset["id"])
        assert other_pinned.source_path == owner_pinned.source_path
    finally:
        resume_cleanup.set()
        worker.join(timeout=10)

    assert worker.is_alive() is False
    assert outcome["response"].status_code == 204
    assert pinned_path.exists()


def test_pin_reauthenticates_cas_after_concurrent_last_reference_cleanup(
    tmp_path,
    monkeypatch,
):
    client, settings = _client(tmp_path)
    frame = pd.DataFrame({"acct_id": ["A1", "B2"]})
    owner_task = _create_task(client)
    owner_dataset = _upload_dataset(client, owner_task["id"], frame)
    other_task = _create_task(client)
    other_dataset = _upload_dataset(client, other_task["id"], frame)
    repo = DatasetRepository(settings.db_path)
    registry = DatasetRegistry(
        repo,
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    owner_pinned = registry.pin_authenticated_snapshot(owner_dataset["id"])
    pinned_path = settings.datasets_dir / owner_pinned.source_path
    pin_entered = threading.Event()
    resume_pin = threading.Event()
    original_pin = repo.pin_dataset_source_path

    def pause_after_materialization(*args, **kwargs):
        pin_entered.set()
        assert resume_pin.wait(timeout=5)
        return original_pin(*args, **kwargs)

    monkeypatch.setattr(repo, "pin_dataset_source_path", pause_after_materialization)
    outcome: dict[str, object] = {}

    def pin_other():
        try:
            outcome["dataset"] = registry.pin_authenticated_snapshot(
                other_dataset["id"]
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcome["error"] = exc

    worker = threading.Thread(target=pin_other)
    worker.start()
    assert pin_entered.wait(timeout=5)
    try:
        deleted = client.delete(f"/api/tasks/{owner_task['id']}")
        assert deleted.status_code == 204
        assert not pinned_path.exists()
    finally:
        resume_pin.set()
        worker.join(timeout=10)

    assert worker.is_alive() is False
    assert "error" not in outcome
    repinned = outcome["dataset"]
    assert repinned.source_path == owner_pinned.source_path
    assert registry.get(other_dataset["id"]).source_path == repinned.source_path
    assert pinned_path.exists()
    authenticated = registry.read_authenticated_parquet_snapshot(other_dataset["id"])
    assert authenticated["acct_id"].tolist() == ["A1", "B2"]


def test_dataset_repository_requires_authentication_before_cas_binding(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    repo = DatasetRepository(settings.db_path)
    digest = dataset["content_hash"]
    content_addressed_path = f"_cas/{digest}/{digest}.parquet"

    with pytest.raises(TypeError, match="validate_content_addressed_source"):
        repo.pin_dataset_source_path(
            dataset["id"],
            expected_source_path=dataset["source_path"],
            expected_content_hash=digest,
            content_addressed_source_path=content_addressed_path,
        )

    assert repo.get_dataset(dataset["id"]).source_path == dataset["source_path"]


def test_delete_task_retries_and_defers_post_commit_cleanup_lock_failure(
    tmp_path,
    caplog,
):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    pinned = registry.pin_authenticated_snapshot(dataset["id"])
    pinned_path = settings.datasets_dir / pinned.source_path
    calls = {"count": 0}

    def locked_cleanup(_path):
        calls["count"] += 1
        raise sqlite3.OperationalError("database is locked")

    client.app.state.dataset_source_gc = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        delete_source=locked_cleanup,
    )

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert calls["count"] == 1
    with pytest.raises(KeyError):
        TaskRepository(settings.db_path).get_task(task["id"])
    assert pinned_path.exists()
    status = client.app.state.dataset_source_gc.status()
    assert status.pending_count == 1
    assert status.entries[0].state == "retry_wait"
    assert "dataset source GC deferred" in caplog.text


def test_delete_task_retries_transient_post_commit_cleanup_lock(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    pinned = registry.pin_authenticated_snapshot(dataset["id"])
    pinned_path = settings.datasets_dir / pinned.source_path
    calls = {"count": 0}

    def transient_lock(_path):
        calls["count"] += 1
        raise sqlite3.OperationalError("database is locked")

    client.app.state.dataset_source_gc = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        delete_source=transient_lock,
    )

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert calls["count"] == 1
    assert pinned_path.exists()

    recovered = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
    ).sweep(force=True)

    assert recovered.deleted == 1
    assert not pinned_path.exists()


def test_delete_task_unlinks_task_directory_symlink_without_deleting_target(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    other_task = _create_task(client)
    other_task_dir = settings.tasks_dir / other_task["id"]
    other_task_dir.mkdir(parents=True)
    sentinel = other_task_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    task_link = settings.tasks_dir / task["id"]
    task_link.symlink_to(other_task_dir, target_is_directory=True)

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert task_link.exists() is False
    assert task_link.is_symlink() is False
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_delete_task_unlinks_dataset_symlink_without_deleting_target(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    protected_dir = settings.datasets_dir / "protected"
    protected_dir.mkdir(parents=True)
    protected_file = protected_dir / "shared.parquet"
    protected_file.write_bytes(b"protected")
    link_dir = settings.datasets_dir / task["id"]
    link_dir.mkdir(parents=True, exist_ok=True)
    dataset_link = link_dir / "linked.parquet"
    dataset_link.symlink_to(protected_file)
    relative_link = dataset_link.relative_to(settings.datasets_dir).as_posix()
    with connect(settings.db_path) as connection:
        connection.execute(
            "UPDATE datasets SET source_path = ? WHERE id = ?",
            (relative_link, dataset["id"]),
        )

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert dataset_link.exists() is False
    assert dataset_link.is_symlink() is False
    assert protected_file.read_bytes() == b"protected"


def test_task_purge_counts_and_deletes_task_drafts_and_runs(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    draft_repo = DraftRepository(settings.db_path)
    draft = DraftTool(
        id="draft-for-purge",
        task_id=task["id"],
        name="temporary helper",
        summary="task-local draft",
        code="def run(payload):\n    return payload\n",
        input_schema={},
        output_schema={},
        determinism="deterministic",
        source="user",
        learning_note_id=None,
        status="draft",
        created_at="2026-07-10T00:00:00+00:00",
    )
    draft_repo.save_draft(draft)
    draft_repo.save_draft_run(DraftRun(
        id="draft-run-for-purge",
        draft_id=draft.id,
        task_id=task["id"],
        inputs_hash="sha256:input",
        ok=True,
        output={"ok": True},
        error=None,
        at="2026-07-10T00:01:00+00:00",
    ))

    preview = client.get(f"/api/tasks/{task['id']}/purge-preview")

    assert preview.status_code == 200
    assert preview.json()["purge_summary"]["draft_tools"] == 1
    assert preview.json()["purge_summary"]["draft_runs"] == 1

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert draft_repo.get_draft(draft.id) is None
    assert draft_repo.list_runs(draft.id) == []
    from marvis.repositories.audit import _list_audit_rows

    audit = _list_audit_rows(settings.db_path, kind="task.delete")[-1]
    assert audit["detail"]["purge_summary"]["draft_tools"] == 1
    assert audit["detail"]["purge_summary"]["draft_runs"] == 1


def test_task_purge_rechecks_active_job_inside_write_transaction(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    repo = TaskRepository(settings.db_path)
    job_id = repo.start_job(task["id"], "metrics")

    with pytest.raises(ConflictError, match="active job"):
        repo.purge_task(task["id"])

    assert repo.get_task(task["id"]).id == task["id"]
    assert repo.get_job(job_id)["status"] == "queued"


def test_task_purge_holds_write_lock_before_dataset_summary_callback(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    _upload_dataset(client, task["id"], pd.DataFrame({"acct_id": ["A1", "B2"]}))
    other_task = _create_task(client)
    lock_observed = {"value": False}

    def validate_while_purge_is_in_progress(_source_path: str) -> None:
        connection = sqlite3.connect(settings.db_path, timeout=0.01)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                connection.execute(
                    "INSERT INTO jobs(id, task_id, kind, status, created_at) "
                    "VALUES ('concurrent-job', ?, 'metrics', 'queued', '2026-07-10')",
                    (other_task["id"],),
                )
            lock_observed["value"] = True
        finally:
            connection.close()

    TaskRepository(settings.db_path).purge_task(
        task["id"],
        validate_dataset_source_path=validate_while_purge_is_in_progress,
    )

    assert lock_observed["value"] is True


def test_concurrent_dedup_upload_does_not_reference_file_deleted_by_purge(
    tmp_path,
    monkeypatch,
):
    client, settings = _client(tmp_path)
    owner_task = _create_task(client)
    frame = pd.DataFrame({"acct_id": ["A1", "B2"]})
    owner_dataset = _upload_dataset(client, owner_task["id"], frame)
    owner_path = settings.datasets_dir / owner_dataset["source_path"]
    other_task = _create_task(client)
    upload_path = tmp_path / "same.csv"
    frame.to_csv(upload_path, index=False)
    repo = DatasetRepository(settings.db_path)
    registry = DatasetRegistry(repo, DataBackend(settings.datasets_dir), settings.datasets_dir)
    original_find = repo.find_dataset_by_content_hash
    found_existing = threading.Event()
    resume_upload = threading.Event()

    def pause_after_hash_lookup(content_hash):
        existing = original_find(content_hash)
        found_existing.set()
        assert resume_upload.wait(timeout=5)
        return existing

    monkeypatch.setattr(repo, "find_dataset_by_content_hash", pause_after_hash_lookup)
    outcome: dict[str, object] = {}

    def upload_duplicate():
        try:
            outcome["dataset"] = registry.register_from_upload(
                other_task["id"],
                upload_path,
                role="sample",
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcome["error"] = exc

    worker = threading.Thread(target=upload_duplicate)
    worker.start()
    assert found_existing.wait(timeout=5)
    try:
        deleted = client.delete(f"/api/tasks/{owner_task['id']}")
        assert deleted.status_code == 204
        assert owner_path.exists() is False
    finally:
        resume_upload.set()
        worker.join(timeout=10)

    assert worker.is_alive() is False
    assert "error" not in outcome
    dataset = outcome["dataset"]
    assert (settings.datasets_dir / dataset.source_path).exists()


def test_delete_task_rejects_escaping_dataset_path_before_database_purge(tmp_path):
    client, settings = _client(tmp_path, raise_server_exceptions=False)
    task = _create_task(client)
    dataset = _upload_dataset(
        client,
        task["id"],
        pd.DataFrame({"acct_id": ["A1", "B2"]}),
    )
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE datasets SET source_path = ? WHERE id = ?",
            ("../outside.parquet", dataset["id"]),
        )
    session_state = {"closed": False}
    session = SimpleNamespace(
        closed=False,
        close=lambda: session_state.__setitem__("closed", True),
    )
    register_live_notebook_session(task["id"], session)

    try:
        response = client.delete(f"/api/tasks/{task['id']}")

        assert response.status_code == 422
        assert TaskRepository(settings.db_path).get_task(task["id"]).id == task["id"]
        with connect(settings.db_path) as conn:
            dataset_row = conn.execute(
                "SELECT id FROM datasets WHERE id = ?",
                (dataset["id"],),
            ).fetchone()
        assert dataset_row is not None
        assert session_state["closed"] is False
        assert get_live_notebook_session(task["id"]) is session
    finally:
        close_live_notebook_session(task["id"])

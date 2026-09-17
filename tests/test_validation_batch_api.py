from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
import re
import sqlite3
import threading

from fastapi.testclient import TestClient
import httpx
import pytest

from marvis.api_task_helpers import format_validation_batch_parent_name
from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.db_schema import SCHEMA_VERSION, connect, init_db
from marvis.domain import TaskStatus
from marvis.files import sha256_file
from marvis.repositories.validation_batches import ValidationBatchRepository
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.settings import build_settings
from tests.validation_builders import (
    make_candidate_contract,
    make_validation_confirmation,
)


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(build_settings(tmp_path)))


def _item(root: Path, name: str) -> dict:
    source = root / name
    source.mkdir(parents=True)
    paths = {
        "notebook_path": source / f"{name}.ipynb",
        "sample_path": source / f"{name}.csv",
        "pmml_path": source / f"{name}.pmml",
        "dictionary_path": source / f"{name}.xlsx",
    }
    for path in paths.values():
        path.touch()
    return {
        "model_name": name,
        "model_version": "v1",
        "source_dir": str(source),
        **{key: str(value) for key, value in paths.items()},
    }


def _upload_item(client: TestClient, name: str = "model") -> dict:
    filenames = {
        "notebook_path": f"{name}.ipynb",
        "sample_path": f"{name}.csv",
        "pmml_path": f"{name}.pmml",
        "dictionary_path": f"{name}.xlsx",
    }
    response = client.post(
        "/api/validation-batches/material-uploads",
        files=[
            ("files", (filename, f"content:{filename}".encode(), "application/octet-stream"))
            for filename in filenames.values()
        ],
        data={"relative_paths": list(filenames.values())},
    )
    assert response.status_code == 201, response.text
    uploaded = response.json()
    assert re.fullmatch(r"[0-9a-f]{32}", uploaded["upload_token"])
    return {
        "model_name": name,
        "model_version": "v1",
        "source_dir": uploaded["source_dir"],
        "upload_token": uploaded["upload_token"],
        **filenames,
    }


def test_batch_material_upload_rejects_oversized_file_without_content_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client(tmp_path)
    object.__setattr__(client.app.state.settings, "max_csv_upload_bytes", 64)
    monkeypatch.setattr(
        "marvis.routers.validation_batches._MATERIAL_UPLOAD_CHUNK_SIZE",
        16,
    )
    multipart = httpx.Request(
        "POST",
        "http://testserver/irrelevant",
        files={"files": ("oversized.csv", b"x" * 80, "text/csv")},
        data={"relative_paths": "oversized.csv"},
    )
    body = multipart.read()
    content_type = multipart.headers["content-type"]

    def chunked_body():
        for offset in range(0, len(body), 11):
            yield body[offset:offset + 11]

    response = client.post(
        "/api/validation-batches/material-uploads",
        content=chunked_body(),
        headers={"content-type": content_type},
    )

    assert response.status_code == 413, response.text
    assert "单文件大小超过上限" in response.json()["detail"]
    staging_root = tmp_path / "material_uploads" / "validation-batches" / "staging"
    assert list(staging_root.iterdir()) == []


def test_batch_material_upload_rejects_combined_size_over_request_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client(tmp_path)
    object.__setattr__(client.app.state.settings, "max_csv_upload_bytes", 64)
    monkeypatch.setattr(
        "marvis.routers.validation_batches._MATERIAL_UPLOAD_CHUNK_SIZE",
        16,
    )

    response = client.post(
        "/api/validation-batches/material-uploads",
        files=[
            ("files", ("first.csv", b"a" * 40, "text/csv")),
            ("files", ("second.pmml", b"b" * 40, "application/xml")),
        ],
        data={"relative_paths": ["first.csv", "second.pmml"]},
    )

    assert response.status_code == 413, response.text
    assert "材料总大小超过上限" in response.json()["detail"]
    staging_root = tmp_path / "material_uploads" / "validation-batches" / "staging"
    assert list(staging_root.iterdir()) == []


def test_batch_upload_token_moves_materials_under_parent_owned_tree(tmp_path: Path):
    client = _client(tmp_path)
    item = _upload_item(client, "owned-model")
    staging = Path(item["source_dir"])

    response = client.post(
        "/api/validation-batches",
        json={"batch_name": "批次", "validator": "qa", "items": [item]},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    parent_task_id = body["batch"]["parent_task_id"]
    child_task_id = body["items"][0]["child_task_id"]
    parent = TaskRepository(client.app.state.settings.db_path).get_task(parent_task_id)
    child = TaskRepository(client.app.state.settings.db_path).get_task(child_task_id)
    parent_source = Path(parent.source_dir)
    assert not staging.exists()
    assert parent_source.parent == (
        tmp_path / "material_uploads" / "validation-batches"
    )
    assert re.fullmatch(r"[0-9a-f]{32}", parent_source.name)
    assert Path(child.source_dir) == parent_source / "01"
    assert Path(child.notebook_path).read_bytes() == b"content:owned-model.ipynb"
    with TaskRepository(client.app.state.settings.db_path).transaction() as conn:
        provision = conn.execute(
            """
            SELECT detail_json FROM audit
             WHERE kind = 'task.filesystem.provisioned' AND target_ref = ?
            """,
            (parent_task_id,),
        ).fetchone()
    assert provision is not None
    assert "validation_batch_source_dir" in provision["detail_json"]
    with TaskRepository(client.app.state.settings.db_path).transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM validation_batch_material_uploads"
        ).fetchone()[0] == 0


def test_batch_upload_staging_ledger_survives_response_and_restart_reclaims_it(
    tmp_path: Path,
):
    client = _client(tmp_path)
    item = _upload_item(client, "abandoned-model")
    staging = Path(item["source_dir"])
    token = item["upload_token"]

    with TaskRepository(client.app.state.settings.db_path).transaction() as conn:
        row = conn.execute(
            """
            SELECT relative_path, expires_at
              FROM validation_batch_material_uploads
             WHERE upload_token = ?
            """,
            (token,),
        ).fetchone()
    assert row is not None
    assert row["relative_path"] == f"staging/{token}"
    assert row["expires_at"]
    assert staging.is_dir()

    # A previous process cannot still own a staged browser upload.  Startup
    # reconciliation therefore removes both the durable claim and its files.
    restarted = create_app(build_settings(tmp_path))
    assert restarted.state.validation_batch_upload_recovery.removed == 1
    assert not staging.exists()
    with TaskRepository(restarted.state.settings.db_path).transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM validation_batch_material_uploads"
        ).fetchone()[0] == 0


def test_batch_upload_request_sweeps_expired_staging_ledger(tmp_path: Path):
    client = _client(tmp_path)
    abandoned = _upload_item(client, "expired-model")
    abandoned_path = Path(abandoned["source_dir"])
    with TaskRepository(client.app.state.settings.db_path).transaction() as conn:
        conn.execute(
            """
            UPDATE validation_batch_material_uploads
               SET expires_at = '1970-01-01T00:00:00+00:00'
             WHERE upload_token = ?
            """,
            (abandoned["upload_token"],),
        )

    current = _upload_item(client, "current-model")

    assert not abandoned_path.exists()
    assert Path(current["source_dir"]).is_dir()
    with TaskRepository(client.app.state.settings.db_path).transaction() as conn:
        tokens = {
            str(row[0])
            for row in conn.execute(
                "SELECT upload_token FROM validation_batch_material_uploads"
            ).fetchall()
        }
    assert tokens == {current["upload_token"]}


def test_schema_35_rebuilds_partial_upload_ledger_without_false_completion(
    tmp_path: Path,
):
    db_path = tmp_path / "partial-v34.sqlite"
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE validation_batch_material_uploads (
                upload_token TEXT PRIMARY KEY,
                expires_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO validation_batch_material_uploads(upload_token, expires_at)
            VALUES ('0123456789abcdef0123456789abcdef', '2099-01-01T00:00:00+00:00')
            """
        )
        conn.execute("PRAGMA user_version = 34")

    init_db(db_path)

    with connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(validation_batch_material_uploads)"
            ).fetchall()
        }
        indexes = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA index_list(validation_batch_material_uploads)"
            ).fetchall()
        }
        rows = conn.execute(
            "SELECT * FROM validation_batch_material_uploads"
        ).fetchall()

    assert version == SCHEMA_VERSION
    assert {
        "upload_token",
        "relative_path",
        "expires_at",
        "created_at",
        "updated_at",
    } <= columns
    # The partial row cannot prove its controlled staging path, so migration
    # must not invent ownership and later delete an arbitrary directory.
    assert rows == []
    assert "idx_validation_batch_material_uploads_expiry" in indexes


def test_batch_create_failure_cleans_claimed_upload_but_preserves_external_source(
    tmp_path: Path,
):
    client = _client(tmp_path)
    uploaded = _upload_item(client, "owned-model")
    staging = Path(uploaded["source_dir"])
    external = _item(tmp_path / "external", "external-model")
    sentinel = Path(external["sample_path"])
    uploaded["model_name"] = "bad/name"

    response = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [external, uploaded],
        },
    )

    assert response.status_code == 422, response.text
    assert not staging.exists()
    assert sentinel.is_file()
    assert client.get("/api/tasks").json() == []


def test_concurrent_batch_create_with_same_upload_token_has_one_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    app = create_app(build_settings(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)
    item = _upload_item(client, "concurrent-owner")
    staging = Path(item["source_dir"])
    payload = {
        "batch_name": "并发令牌批次",
        "validator": "qa",
        "items": [item],
    }

    losing_request_finished = threading.Event()
    rename_reached = threading.Event()
    original_rename = Path.rename

    def hold_winner_after_move(path: Path, target: Path):
        moved = original_rename(path, target)
        if path.absolute() == staging.absolute():
            rename_reached.set()
            assert losing_request_finished.wait(timeout=5.0)
        return moved

    monkeypatch.setattr(Path, "rename", hold_winner_after_move)
    start = threading.Barrier(2)

    def create_once():
        start.wait(timeout=5.0)
        response = client.post("/api/validation-batches", json=payload)
        if response.status_code != 201:
            losing_request_finished.set()
        return response

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _index: create_once(), range(2)))

    assert rename_reached.is_set()
    assert sorted(response.status_code for response in responses) == [201, 422]
    created = next(response.json() for response in responses if response.status_code == 201)
    child_task_id = created["items"][0]["child_task_id"]
    child = client.get(f"/api/tasks/{child_task_id}")
    assert child.status_code == 200, child.text
    assert Path(child.json()["sample_path"]).read_bytes() == b"content:concurrent-owner.csv"
    assert not staging.exists()


def test_batch_upload_token_cannot_claim_an_external_directory(tmp_path: Path):
    client = _client(tmp_path)
    item = _upload_item(client, "owned-model")
    staging = Path(item["source_dir"])
    external = tmp_path / "external-token-lookalike"
    external.mkdir()
    (external / "keep.txt").write_text("user-owned", encoding="utf-8")
    item["source_dir"] = str(external)

    response = client.post(
        "/api/validation-batches",
        json={"batch_name": "批次", "validator": "qa", "items": [item]},
    )

    assert response.status_code == 422
    assert not staging.exists()
    assert (external / "keep.txt").read_text(encoding="utf-8") == "user-owned"


def test_batch_upload_cancel_is_scoped_and_idempotent(tmp_path: Path):
    client = _client(tmp_path)
    item = _upload_item(client, "cancel-model")
    staging = Path(item["source_dir"])

    first = client.delete(
        f"/api/validation-batches/material-uploads/{item['upload_token']}"
    )
    second = client.delete(
        f"/api/validation-batches/material-uploads/{item['upload_token']}"
    )

    assert first.status_code == 204
    assert second.status_code == 204
    assert not staging.exists()
    with TaskRepository(client.app.state.settings.db_path).transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM validation_batch_material_uploads"
        ).fetchone()[0] == 0


def test_delete_batch_purges_family_and_owned_material_tree(tmp_path: Path):
    client = _client(tmp_path)
    item = _upload_item(client, "delete-model")
    created = client.post(
        "/api/validation-batches",
        json={"batch_name": "批次", "validator": "qa", "items": [item]},
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    child_task_id = created["items"][0]["child_task_id"]
    task_repo = TaskRepository(client.app.state.settings.db_path)
    parent_source = Path(task_repo.get_task(parent_task_id).source_dir)
    for task_id in (parent_task_id, child_task_id):
        output = client.app.state.settings.tasks_dir / task_id / "outputs" / "keep.txt"
        output.parent.mkdir(parents=True)
        output.write_text("remove", encoding="utf-8")

    preview = client.get(
        f"/api/validation-batches/{parent_task_id}/purge-preview"
    )
    response = client.delete(f"/api/validation-batches/{parent_task_id}")

    assert preview.status_code == 200, preview.text
    preview_body = preview.json()
    assert preview_body["task_count"] == 2
    assert preview_body["child_task_count"] == 1
    assert preview_body["owned_batch_material_tree_count"] == 1
    assert preview_body["purge_summary"]["child_tasks"] == 1
    assert preview_body["purge_summary"]["owned_batch_material_trees"] == 1
    assert "purge_summary" not in preview_body["purge_summary"]
    assert response.status_code == 204, response.text
    assert client.get(f"/api/tasks/{parent_task_id}").status_code == 404
    assert client.get(f"/api/tasks/{child_task_id}").status_code == 404
    assert not parent_source.exists()
    with task_repo.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM validation_batches WHERE parent_task_id = ?",
            (parent_task_id,),
        ).fetchone()[0] == 0
        audit = conn.execute(
            """
            SELECT outcome FROM audit
             WHERE kind = 'validation_batch.delete' AND target_ref = ?
            """,
            (parent_task_id,),
        ).fetchone()
    assert audit is not None


def test_delete_batch_preserves_all_external_manual_materials(tmp_path: Path):
    client = _client(tmp_path)
    external = _item(tmp_path / "external", "external-model")
    sentinel = Path(external["sample_path"])
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [external],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]

    response = client.delete(f"/api/validation-batches/{parent_task_id}")

    assert response.status_code == 204, response.text
    assert sentinel.is_file()


def test_delete_batch_rejects_active_job_anywhere_in_family(tmp_path: Path):
    client = _client(tmp_path)
    item = _upload_item(client, "active-model")
    created = client.post(
        "/api/validation-batches",
        json={"batch_name": "批次", "validator": "qa", "items": [item]},
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    child_task_id = created["items"][0]["child_task_id"]
    task_repo = TaskRepository(client.app.state.settings.db_path)
    parent_source = Path(task_repo.get_task(parent_task_id).source_dir)
    task_repo.start_job(child_task_id, "validation")

    response = client.delete(f"/api/validation-batches/{parent_task_id}")

    assert response.status_code == 409, response.text
    assert "active jobs" in response.json()["detail"]
    assert task_repo.get_task(parent_task_id)
    assert task_repo.get_task(child_task_id)
    assert parent_source.is_dir()


def test_delete_batch_commits_even_when_notebook_session_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    child_task_id = created["items"][0]["child_task_id"]

    def fail_close(_task_id: str) -> None:
        raise OSError("synthetic notebook close failure")

    monkeypatch.setattr(
        "marvis.routers.validation_batches.close_live_notebook_session",
        fail_close,
    )

    response = client.delete(f"/api/validation-batches/{parent_task_id}")

    assert response.status_code == 204, response.text
    assert client.get(f"/api/tasks/{parent_task_id}").status_code == 404
    assert client.get(f"/api/tasks/{child_task_id}").status_code == 404


def test_batch_create_remains_successful_when_optional_agent_message_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client(tmp_path)

    def fail_message(*_args, **_kwargs) -> None:
        raise sqlite3.OperationalError("synthetic message failure")

    monkeypatch.setattr(TaskRepository, "add_agent_message", fail_message)

    response = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    )

    assert response.status_code == 201, response.text
    parent_task_id = response.json()["batch"]["parent_task_id"]
    assert client.get(f"/api/tasks/{parent_task_id}").status_code == 200


def test_batch_creation_refuses_symlinked_owned_material_root(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    uploads = tmp_path / "material_uploads"
    uploads.mkdir()
    (uploads / "validation-batches").symlink_to(
        outside,
        target_is_directory=True,
    )
    client = TestClient(
        create_app(build_settings(tmp_path)),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    )

    assert response.status_code == 500
    assert list(outside.iterdir()) == []


def test_validation_batch_parent_name_uses_date_and_model_count():
    assert format_validation_batch_parent_name(
        2,
        datetime(2026, 8, 20, 15, 30, 0),
    ) == "2026-08-20 模型验证批次 (2个模型)"
    assert format_validation_batch_parent_name(
        0,
        datetime(2026, 8, 20),
    ) == "2026-08-20 模型验证批次"


def test_create_validation_batch_returns_parent_and_hidden_children(tmp_path: Path):
    client = _client(tmp_path)
    materials = tmp_path / "materials"
    payload = {
        "batch_name": "2026年8月模型验证批次",
        "validator": "qa",
        "items": [_item(materials, "模型A"), _item(materials, "模型B")],
    }

    response = client.post("/api/validation-batches", json=payload)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["batch"]["status"] == "created"
    assert body["batch"]["item_count"] == 2
    assert [item["model_name"] for item in body["items"]] == ["模型A", "模型B"]
    assert all(item["child_task_id"] for item in body["items"])
    parent_task = body["parent_task"]
    assert parent_task["id"] == body["batch"]["parent_task_id"]
    assert parent_task["task_type"] == "validation_batch"
    assert parent_task["run_mode"] == "agent"
    assert parent_task["model_name"] == (
        f"{date.today():%Y-%m-%d} 模型验证批次 (2个模型)"
    )
    assert parent_task["item_count"] == 2
    task_list = client.get("/api/tasks").json()
    assert [task["id"] for task in task_list] == [body["batch"]["parent_task_id"]]
    assert task_list[0]["model_name"] == parent_task["model_name"]
    assert task_list[0]["item_count"] == 2

    detail = client.get(
        f"/api/validation-batches/{body['batch']['parent_task_id']}"
    )
    assert detail.status_code == 200
    assert detail.json()["items"] == body["items"]


def test_task_list_rewrites_legacy_batch_parent_child_name(tmp_path: Path):
    client = _client(tmp_path)
    materials = tmp_path / "materials"
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(materials, "自营通用T卡多头 MOB6"), _item(materials, "模型B")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    expected_name = f"{date.today():%Y-%m-%d} 模型验证批次 (2个模型)"
    db_path = client.app.state.settings.db_path
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET model_name = ? WHERE id = ?",
            ("自营通用T卡多头 MOB6", parent_task_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT model_name FROM tasks WHERE id = ?",
            (parent_task_id,),
        ).fetchone()
    assert row["model_name"] == "自营通用T卡多头 MOB6"

    listed = client.get("/api/tasks").json()
    assert listed[0]["id"] == parent_task_id
    assert listed[0]["model_name"] == expected_name
    assert listed[0]["item_count"] == 2
    assert "自营通用T卡多头 MOB6" not in listed[0]["model_name"]

    detail = client.get(f"/api/tasks/{parent_task_id}").json()
    assert detail["model_name"] == expected_name
    assert detail["item_count"] == 2


def test_validation_batch_rejects_material_path_reuse_between_models(tmp_path: Path):
    client = _client(tmp_path)
    materials = tmp_path / "materials"
    first = _item(materials, "模型A")
    second = _item(materials, "模型B")
    second["notebook_path"] = first["notebook_path"]
    second["source_dir"] = str(materials)
    first["source_dir"] = str(materials)

    response = client.post(
        "/api/validation-batches",
        json={"batch_name": "批次", "validator": "qa", "items": [first, second]},
    )

    assert response.status_code == 422
    assert "同一批次的模型材料路径不能复用" in response.json()["detail"]


def test_validation_batch_rejects_more_than_ten_models(tmp_path: Path):
    client = _client(tmp_path)
    items = [_item(tmp_path / "materials", f"模型{index}") for index in range(11)]

    response = client.post(
        "/api/validation-batches",
        json={"batch_name": "批次", "validator": "qa", "items": items},
    )

    assert response.status_code == 422


def test_start_validation_batch_dispatches_one_ordered_background_run(tmp_path: Path):
    client = _client(tmp_path)
    materials = tmp_path / "materials"
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(materials, "模型A"), _item(materials, "模型B")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    calls: list[dict] = []
    client.app.state.validation_batch_job_runner = lambda **kwargs: calls.append(kwargs)

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={"model_id": "narrative-model", "effort": "medium"},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body == {
        "parent_task_id": parent_task_id,
        "job_id": body["job_id"],
        "status": "accepted",
        "batch_url": f"/api/validation-batches/{parent_task_id}",
        "message": "validation batch dispatched; poll batch_url for progress",
    }
    assert len(calls) == 1
    assert calls[0]["parent_task_id"] == parent_task_id
    assert calls[0]["model_id"] == "narrative-model"
    assert calls[0]["effort"] == "medium"
    assert calls[0]["manual_review_base_url"] == ""
    assert calls[0]["job_id"] == body["job_id"]
    assert client.get(
        f"/api/validation-batches/{parent_task_id}"
    ).json()["batch"]["status"] == "running"
    task_repo = TaskRepository(client.app.state.settings.db_path)
    parent = task_repo.get_task(parent_task_id)
    assert parent.status is TaskStatus.RUNNING
    assert task_repo.get_active_job_kind(parent_task_id) == "validation_batch"
    job = task_repo.get_job(body["job_id"])
    assert job is not None
    assert job["status"] == "queued"

    # Even if batch state were stale/startable, the active parent Job remains
    # the authoritative atomic duplicate-start guard.
    ValidationBatchRepository(client.app.state.settings.db_path).update_batch(
        parent_task_id,
        status="created",
    )
    duplicate = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )
    assert duplicate.status_code == 409
    assert task_repo.get_active_job_kind(parent_task_id) == "validation_batch"


@pytest.mark.parametrize(
    ("host", "expected_base_url"),
    [
        ("127.0.0.1:8123", "http://127.0.0.1:8123"),
        ("evil.example", ""),
    ],
)
def test_batch_manual_review_link_accepts_only_local_request_hosts(
    tmp_path: Path,
    host: str,
    expected_base_url: str,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    calls: list[dict] = []
    client.app.state.validation_batch_job_runner = lambda **kwargs: calls.append(kwargs)

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
        headers={"host": host},
    )

    assert response.status_code == 202, response.text
    assert calls[0]["manual_review_base_url"] == expected_base_url


def test_partial_failure_batch_can_be_dispatched_for_repaired_item_retry(
    tmp_path: Path,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    ValidationBatchRepository(client.app.state.settings.db_path).update_batch(
        parent_task_id,
        status="partial_failure",
    )
    calls: list[dict] = []
    client.app.state.validation_batch_job_runner = lambda **kwargs: calls.append(kwargs)

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )

    assert response.status_code == 202, response.text
    assert len(calls) == 1
    assert calls[0]["retry_failed_items"] is True


def test_failed_batch_can_be_dispatched_for_repaired_item_retry(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    batch_repo = ValidationBatchRepository(client.app.state.settings.db_path)
    [item] = batch_repo.list_items(parent_task_id)
    batch_repo.update_item(
        item.id,
        status="failed",
        stage="scan",
        outcome="failed",
        finished=True,
    )
    batch_repo.update_batch(parent_task_id, status="failed", finished=True)
    calls: list[dict] = []
    client.app.state.validation_batch_job_runner = lambda **kwargs: calls.append(kwargs)

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )

    assert response.status_code == 202, response.text
    assert len(calls) == 1
    assert calls[0]["retry_failed_items"] is True


def test_parent_transition_failure_restores_exact_batch_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    batch_repo = ValidationBatchRepository(client.app.state.settings.db_path)
    summary_path = tmp_path / "previous-summary.xlsx"
    summary_path.touch()
    batch_repo.update_batch(
        parent_task_id,
        status="failed",
        summary_path=str(summary_path),
        error_message="previous failure",
        started=True,
        finished=True,
    )
    before = batch_repo.get_batch(parent_task_id)

    def fail_parent_transition(*_args, **_kwargs):
        raise RuntimeError("parent transition failed")

    monkeypatch.setattr(
        "marvis.routers.validation_batches.mark_validation_batch_parent_running",
        fail_parent_transition,
    )

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )

    assert response.status_code == 409, response.text
    assert batch_repo.get_batch(parent_task_id) == before
    task_repo = TaskRepository(client.app.state.settings.db_path)
    assert task_repo.get_active_job_kind(parent_task_id) is None


def test_waiting_batch_requires_every_contract_confirmation_before_restart(
    tmp_path: Path,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    child_task_id = created["items"][0]["child_task_id"]
    settings = client.app.state.settings
    task_repo = TaskRepository(settings.db_path)
    child = task_repo.get_task(child_task_id)
    task_repo.update_status(
        child_task_id,
        TaskStatus.SCANNED,
        "scanned",
        expected=TaskStatus.CREATED,
    )
    hashes = {
        "notebook": sha256_file(Path(child.notebook_path)),
        "sample": sha256_file(Path(child.sample_path)),
        "pmml": sha256_file(Path(child.pmml_path)),
        "dictionary": sha256_file(Path(child.dictionary_path)),
    }
    contract_repo = ValidationContractRepository(settings.db_path)
    pending = contract_repo.replace_candidates(
        child_task_id,
        make_candidate_contract(material_hashes=hashes),
    )
    batch_repo = ValidationBatchRepository(settings.db_path)
    [item] = batch_repo.list_items(parent_task_id)
    batch_repo.update_item(
        item.id,
        status="awaiting_confirmation",
        stage="input_confirmation",
    )
    batch_repo.update_batch(parent_task_id, status="awaiting_confirmation")
    calls: list[dict] = []
    client.app.state.validation_batch_job_runner = lambda **kwargs: calls.append(kwargs)

    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()
    assert detail["items"][0]["input_contract_status"] == "pending_confirmation"
    blocked = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )
    assert blocked.status_code == 409
    assert "1 个模型等待确认" in blocked.json()["detail"]
    assert calls == []
    assert task_repo.get_active_job_kind(parent_task_id) is None

    batch_repo.update_batch(parent_task_id, status="partial_failure")
    blocked_retry = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )
    assert blocked_retry.status_code == 409
    assert task_repo.get_active_job_kind(parent_task_id) is None
    batch_repo.update_batch(parent_task_id, status="awaiting_confirmation")

    contract_repo.confirm(
        child_task_id,
        make_validation_confirmation(),
        expected_revision=pending.revision,
    )
    refreshed = client.get(f"/api/validation-batches/{parent_task_id}").json()
    assert refreshed["items"][0]["input_contract_status"] == "ready"
    started = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )
    assert started.status_code == 202, started.text
    assert len(calls) == 1


def test_batch_summary_from_previous_attempt_is_not_republished_after_retry_failure(
    tmp_path: Path,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    summary_path = (
        client.app.state.settings.tasks_dir
        / parent_task_id
        / "outputs"
        / "validation_batch_summary.xlsx"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_bytes(b"previous-attempt")
    batch_repo = ValidationBatchRepository(client.app.state.settings.db_path)
    batch_repo.update_batch(
        parent_task_id,
        status="failed",
        summary_path=str(summary_path),
        finished=True,
    )
    client.app.state.validation_batch_job_runner = lambda **_kwargs: None

    started = client.post(
        f"/api/validation-batches/{parent_task_id}/start",
        json={},
    )
    assert started.status_code == 202, started.text
    assert batch_repo.get_batch(parent_task_id).summary_path == ""
    batch_repo.update_batch(parent_task_id, status="failed", finished=True)

    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()
    assert detail["batch"]["summary_download_url"] == ""
    assert client.get(
        f"/api/validation-batches/{parent_task_id}/summary/download"
    ).status_code == 404


def test_batch_payload_exposes_only_existing_safe_report_urls(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    [item] = created["items"]
    child_task_id = item["child_task_id"]
    assert "summary_path" not in created["batch"]
    assert created["batch"]["summary_download_url"] == ""
    assert item["word_report_download_url"] == ""
    assert item["analysis_download_url"] == ""
    assert item["pending_report_draft"] is False

    parent_outputs = (
        client.app.state.settings.tasks_dir / parent_task_id / "outputs"
    )
    parent_outputs.mkdir(parents=True)
    (parent_outputs / "validation_batch_summary.xlsx").write_bytes(b"safe-summary")
    child_outputs = client.app.state.settings.tasks_dir / child_task_id / "outputs"
    child_outputs.mkdir(parents=True)
    (child_outputs / "validation_report.docx").touch()
    (child_outputs / "validation.xlsx").touch()
    forged = tmp_path / "outside-summary.xlsx"
    forged.write_bytes(b"forged-summary")
    ValidationBatchRepository(client.app.state.settings.db_path).update_batch(
        parent_task_id,
        status="completed",
        summary_path=str(forged),
        finished=True,
    )

    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()

    assert "summary_path" not in detail["batch"]
    assert detail["batch"]["summary_download_url"] == ""
    assert detail["items"][0]["word_report_download_url"] == (
        f"/api/tasks/{child_task_id}/report/download"
    )
    assert detail["items"][0]["analysis_download_url"] == (
        f"/api/tasks/{child_task_id}/analysis/download"
    )
    assert client.get(
        f"/api/validation-batches/{parent_task_id}/summary/download"
    ).status_code == 404

    ValidationBatchRepository(client.app.state.settings.db_path).update_batch(
        parent_task_id,
        status="completed",
        summary_path=str(parent_outputs / "validation_batch_summary.xlsx"),
        finished=True,
    )
    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()
    downloaded = client.get(detail["batch"]["summary_download_url"])
    assert downloaded.status_code == 200
    assert downloaded.content == b"safe-summary"


def test_batch_summary_download_returns_404_before_fixed_artifact_exists(
    tmp_path: Path,
):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]

    response = client.get(
        f"/api/validation-batches/{parent_task_id}/summary/download"
    )

    assert response.status_code == 404


def test_batch_summary_download_rejects_symlink_escape(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    summary_path = (
        client.app.state.settings.tasks_dir
        / parent_task_id
        / "outputs"
        / "validation_batch_summary.xlsx"
    )
    summary_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside-secret.xlsx"
    outside.write_bytes(b"OUTSIDE_SECRET")
    summary_path.symlink_to(outside)
    ValidationBatchRepository(client.app.state.settings.db_path).update_batch(
        parent_task_id,
        status="failed",
        summary_path=str(summary_path),
        finished=True,
    )

    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()
    response = client.get(
        f"/api/validation-batches/{parent_task_id}/summary/download"
    )

    assert detail["batch"]["summary_download_url"] == ""
    assert response.status_code == 404
    assert response.content != b"OUTSIDE_SECRET"


def test_batch_summary_download_rejects_outputs_directory_symlink(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A")],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    parent_task_dir = client.app.state.settings.tasks_dir / parent_task_id
    parent_task_dir.mkdir(parents=True, exist_ok=True)
    other_outputs = client.app.state.settings.tasks_dir / "other-task" / "outputs"
    other_outputs.mkdir(parents=True)
    (other_outputs / "validation_batch_summary.xlsx").write_bytes(
        b"OTHER_TASK_SECRET"
    )
    outputs = parent_task_dir / "outputs"
    outputs.symlink_to(other_outputs, target_is_directory=True)
    expected_path = outputs / "validation_batch_summary.xlsx"
    ValidationBatchRepository(client.app.state.settings.db_path).update_batch(
        parent_task_id,
        status="failed",
        summary_path=str(expected_path),
        finished=True,
    )

    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()
    response = client.get(
        f"/api/validation-batches/{parent_task_id}/summary/download"
    )

    assert detail["batch"]["summary_download_url"] == ""
    assert response.status_code == 404
    assert response.content != b"OTHER_TASK_SECRET"


_REQUIRED_AGENT_CONCLUSIONS = {
    "TEXT:pressure_test_summary": "压力测试显示模型整体稳定。",
    "TEXT:pressure_impact_recommendation": "建议继续监测缺失率较高的数据源。",
    "TEXT:final_validation_conclusion": "模型整体满足验证要求。",
}


def _advance_child_to_writing_artifacts(repo: TaskRepository, task_id: str) -> None:
    repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task_id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task_id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task_id,
        TaskStatus.COMPUTING_METRICS,
        "metrics",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task_id,
        TaskStatus.WRITING_ARTIFACTS,
        "writing",
        expected=TaskStatus.COMPUTING_METRICS,
    )


def _seed_pending_report_draft(repo: TaskRepository, task_id: str) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n压力测试显示模型整体稳定。",
        metadata={
            "draft_values": _REQUIRED_AGENT_CONCLUSIONS,
            "report_revision": 0,
        },
    )


def test_confirm_all_batch_report_drafts_rejects_unfinished_children(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [
                _item(tmp_path / "materials", "模型A"),
                _item(tmp_path / "materials", "模型B"),
            ],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    child_a, child_b = [item["child_task_id"] for item in created["items"]]
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    with connect(tmp_path / "marvis.sqlite") as conn:
        conn.execute(
            "UPDATE tasks SET validation_workflow_version = 1 WHERE id IN (?, ?)",
            (child_a, child_b),
        )
    _advance_child_to_writing_artifacts(repo, child_a)
    _seed_pending_report_draft(repo, child_a)

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/report-drafts/confirm-all",
        json={"overrides": {}},
    )

    assert response.status_code == 409, response.text
    assert "尚未完成报告结论草稿" in response.json()["detail"]


@pytest.mark.parametrize("first_fails", [False, True])
def test_confirm_all_batch_report_drafts_dispatches_reports_for_pending_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_fails: bool,
):
    calls: list[str] = []
    def run_report(**kwargs):
        calls.append(kwargs["task_id"])
        if first_fails and len(calls) == 1:
            TaskRepository(tmp_path / "marvis.sqlite").update_status(
                kwargs["task_id"], TaskStatus.FAILED, "report rendering failed",
                expected=TaskStatus.WRITING_ARTIFACTS,
            )
            raise RuntimeError("report rendering failed")
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_report_stage",
        run_report,
    )
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次",
            "validator": "qa",
            "items": [
                _item(tmp_path / "materials", "模型A"),
                _item(tmp_path / "materials", "模型B"),
            ],
        },
    ).json()
    parent_task_id = created["batch"]["parent_task_id"]
    child_a, child_b = [item["child_task_id"] for item in created["items"]]
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    with connect(tmp_path / "marvis.sqlite") as conn:
        conn.execute(
            "UPDATE tasks SET validation_workflow_version = 1 WHERE id IN (?, ?)",
            (child_a, child_b),
        )
    for child_id in (child_a, child_b):
        _advance_child_to_writing_artifacts(repo, child_id)
        _seed_pending_report_draft(repo, child_id)
        draft_message = repo.list_agent_messages(child_id)[-1]
        saved = client.put(f"/api/tasks/{child_id}/agent/report-draft", json={
            "revision": 0, "draft_message_id": draft_message["id"], "draft_edit_revision": 0,
            "text_values": {"TEXT:model_scope": f"已保存范围 {child_id}"},
        })
        assert saved.status_code == 200, saved.text

    detail = client.get(f"/api/validation-batches/{parent_task_id}").json()
    assert all(item["pending_report_draft"] for item in detail["items"])

    response = client.post(
        f"/api/validation-batches/{parent_task_id}/report-drafts/confirm-all",
        json={
            "overrides": {
                child_a: {
                    "revision": 0,
                    "text_values": {
                        **_REQUIRED_AGENT_CONCLUSIONS,
                        "TEXT:model_scope": "支用环节",
                    },
                }
            }
        },
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["confirmed_count"] == 2
    assert sorted(item["child_task_id"] for item in body["confirmed"]) == sorted(
        [child_a, child_b]
    )
    assert calls == [child_a, child_b]
    assert repo.get_active_job_kind(child_a) is None
    assert repo.get_active_job_kind(child_b) is None
    values_a, _ = repo.get_report_values(child_a)
    assert values_a["TEXT:model_scope"] == "支用环节"
    assert repo.get_report_values(child_b)[0]["TEXT:model_scope"] == f"已保存范围 {child_b}"
    assert repo.list_agent_messages(child_a)[-1]["stage"] in {
        "word_conclusion_confirmed",
        "word_report_ready",
    }


@pytest.mark.parametrize("invalid", ["stale_revision", "computed_value", "active_job", "unconfirmed_contract"])
def test_confirm_all_batch_report_drafts_rejects_atomically(tmp_path: Path, invalid):
    client = _client(tmp_path)
    created = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "批次", "validator": "qa",
            "items": [_item(tmp_path / "materials", "模型A"), _item(tmp_path / "materials", "模型B")],
        },
    ).json()
    parent_id = created["batch"]["parent_task_id"]
    child_a, child_b = [item["child_task_id"] for item in created["items"]]
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    with connect(repo.db_path) as conn:
        conn.execute("UPDATE tasks SET validation_workflow_version = 1 WHERE id IN (?, ?)", (child_a, child_b))
    for child_id in (child_a, child_b):
        _advance_child_to_writing_artifacts(repo, child_id)
        _seed_pending_report_draft(repo, child_id)
    before = {child_id: repo.get_report_values(child_id) for child_id in (child_a, child_b)}
    override = {"revision": 0, "text_values": dict(_REQUIRED_AGENT_CONCLUSIONS)}
    expected_status = 409
    if invalid == "stale_revision":
        override["revision"] = 99
    elif invalid == "computed_value":
        override["text_values"]["TEXT:oot_ks"] = "0.99"
        expected_status = 422
    elif invalid == "active_job":
        repo.start_job(child_b, "report")
    else:
        with connect(repo.db_path) as conn:
            conn.execute("UPDATE tasks SET validation_workflow_version = 2 WHERE id = ?", (child_b,))
        expected_status = 422

    response = client.post(
        f"/api/validation-batches/{parent_id}/report-drafts/confirm-all",
        json={"overrides": {child_b: override}},
    )

    assert response.status_code == expected_status, response.text
    assert repo.get_active_job_kind(child_a) is None
    for child_id in (child_a, child_b):
        assert repo.get_report_values(child_id) == before[child_id]
        assert not any(message["stage"] == "word_conclusion_confirmed" for message in repo.list_agent_messages(child_id))

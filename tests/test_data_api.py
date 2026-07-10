import asyncio
import hashlib
import json
import time

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.api import router
from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.join_engine import JoinEngine
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.job_cancellation import (
    register_job_cancellation,
    request_job_cancellation,
    unregister_job_cancellation,
)
from marvis.routers.data import _run_join_execute_job, router as data_router
from marvis.routers.stage_controls import router as stage_controls_router
from marvis.settings import build_settings


class FakeHookDispatcher:
    def __init__(self):
        self.calls = []

    def dispatch(self, event, payload, *, task_id):
        self.calls.append((event, payload, task_id))
        return []


def _client(tmp_path, *, raise_server_exceptions: bool = True):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    app.include_router(data_router)
    app.include_router(stage_controls_router)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), settings


def _create_task(settings):
    return TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="A-card",
            model_version="v1",
            validator="validator",
            source_dir=str(settings.workspace),
            algorithm="lgb",
            run_mode="manual",
            target_col="bad_flag",
            score_col="score",
            split_col="split",
            time_col="apply_month",
            feature_columns=[],
            notebook_path=None,
            sample_path=None,
            pmml_path=None,
            dictionary_path=None,
            report_values={},
        )
    )


def _registry(settings):
    repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    return DatasetRegistry(repo, backend, settings.datasets_dir)


def _register_csv(settings, tmp_path, task_id: str, name: str, frame: pd.DataFrame, role: str):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return _registry(settings).register_from_upload(task_id, path, role=role)


def _confirmed_join_plan(client, settings, tmp_path):
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        f"anchor_{task.id}",
        pd.DataFrame({"customer_id": [1, 2]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        f"feature_{task.id}",
        pd.DataFrame({"customer_id": [1, 2], "balance": [10, 20]}),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()
    client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True},
    )
    return task, plan


def test_data_routes_are_served_from_dedicated_router():
    routes = {
        (route.path, tuple(sorted(route.methods or []))): route.endpoint.__module__
        for route in data_router.routes
    }

    assert routes[("/api/tasks/{task_id}/datasets", ("GET",))] == "marvis.routers.data"
    assert routes[("/api/tasks/{task_id}/datasets/upload", ("POST",))] == "marvis.routers.data"
    assert routes[("/api/datasets/{dataset_id}/preview", ("GET",))] == "marvis.routers.data"
    assert routes[("/api/tasks/{task_id}/joins/propose", ("POST",))] == "marvis.routers.data"
    assert routes[("/api/joins/{join_plan_id}", ("GET",))] == "marvis.routers.data"
    assert routes[("/api/joins/{join_plan_id}/confirm", ("POST",))] == "marvis.routers.data"
    assert routes[("/api/joins/{join_plan_id}/execute", ("POST",))] == "marvis.routers.data"


def test_dataset_upload_list_and_preview_api(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    csv_bytes = b"mobile,bad_flag\n13800138000,0\n13900139000,1\n"

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )

    assert upload.status_code == 201
    dataset = upload.json()["datasets"][0]
    assert dataset["role"] == "sample"
    assert dataset["has_target"] is True
    assert dataset["target_col"] == "bad_flag"

    listed = client.get(f"/api/tasks/{task.id}/datasets")
    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=1")
    invalid_preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=9999")

    assert listed.status_code == 200
    assert listed.json()["datasets"][0]["id"] == dataset["id"]
    assert preview.status_code == 200
    assert preview.json()["columns"] == ["mobile", "bad_flag"]
    assert preview.json()["truncated"] is True
    assert invalid_preview.status_code == 422


def test_dataset_upload_csv_surfaces_encoding_and_long_id_warnings(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    raw = (
        "姓名,id_card,bad_flag\n"
        "张三,110101199001011234,0\n"
        "李四,,1\n"
        "王五,110101199001015678,0\n"
    ).encode("gbk")

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("gbk_ids.csv", raw, "text/csv")},
    )

    assert upload.status_code == 201
    body = upload.json()
    assert body["reports"], "CSV upload should surface an ingest report"
    report = body["reports"][0]
    assert report["encoding_used"] == "gb18030"
    assert "id_card" in report["long_id_columns"]
    assert any("id_card" in warning for warning in report["warnings"])


def test_dataset_upload_reuses_dataset_by_content_hash_across_tasks(tmp_path):
    client, settings = _client(tmp_path)
    task_a = _create_task(settings)
    task_b = _create_task(settings)
    csv_bytes = b"mobile,bad_flag\n13800138000,0\n13900139000,1\n"

    upload_a = client.post(
        f"/api/tasks/{task_a.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    upload_b = client.post(
        f"/api/tasks/{task_b.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )

    assert upload_a.status_code == 201
    assert upload_b.status_code == 201
    dataset_a = upload_a.json()["datasets"][0]
    dataset_b = upload_b.json()["datasets"][0]

    assert dataset_a["content_hash"] is not None
    assert dataset_a["content_hash"] == dataset_b["content_hash"]
    assert dataset_a["source_path"] == dataset_b["source_path"]
    assert dataset_b["task_id"] == task_b.id
    # no second parquet file was written for task_b's own directory
    assert not list((settings.datasets_dir / task_b.id).glob("*.parquet"))

    from marvis.repositories.audit import _list_audit_rows

    audit_rows = _list_audit_rows(settings.db_path, kind="dataset.dedup_reference")
    assert len(audit_rows) == 1
    assert audit_rows[0]["target_ref"] == dataset_b["id"]
    assert audit_rows[0]["detail"]["reused_dataset_id"] == dataset_a["id"]


def test_dataset_preview_returns_column_profiles_and_masked_samples(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    csv_bytes = b"mobile,bad_flag\n13800138000,0\n,1\n"

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    dataset = upload.json()["datasets"][0]

    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=2")

    assert preview.status_code == 200
    payload = preview.json()
    profiles = {profile["name"]: profile for profile in payload["column_profiles"]}
    assert profiles["mobile"]["semantic_role"] == "phone"
    assert profiles["mobile"]["null_rate"] == 0.5
    assert profiles["mobile"]["sample_values"] == ["138******00"]
    assert payload["rows"][0]["mobile"] == "138******00"
    assert payload["rows"][1]["mobile"] is None
    assert "13800138000" not in json.dumps(payload, ensure_ascii=False)


def test_dataset_preview_masks_names_and_long_card_values(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    csv_bytes = (
        "customer_name,bank_card,bad_flag\n"
        "张三丰,6222020202020202020,0\n"
        "李四,6222020202020202021,1\n"
    ).encode()

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    dataset = upload.json()["datasets"][0]

    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=2")

    assert preview.status_code == 200
    payload = preview.json()
    dumped = json.dumps(payload, ensure_ascii=False)
    profiles = {profile["name"]: profile for profile in payload["column_profiles"]}
    # customer_name is now detected as the 'name' identity element (join key §4) — still
    # masked as an opaque token (PII), never surfaced raw.
    assert profiles["customer_name"]["semantic_role"] == "name"
    assert profiles["customer_name"]["sample_values"][0].startswith("value:")
    assert payload["rows"][0]["customer_name"].startswith("value:")
    assert payload["rows"][0]["bank_card"].startswith("6222")
    assert "*" in payload["rows"][0]["bank_card"]
    assert "张三丰" not in dumped
    assert "李四" not in dumped
    assert "6222020202020202020" not in dumped


def test_dataset_upload_dispatches_dataset_registered_hook(tmp_path):
    client, settings = _client(tmp_path)
    dispatcher = FakeHookDispatcher()
    client.app.state.hook_dispatcher = dispatcher
    task = _create_task(settings)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", b"mobile,bad_flag\n13800138000,0\n", "text/csv")},
    )

    assert response.status_code == 201
    dataset = response.json()["datasets"][0]
    assert dispatcher.calls == [
        (
            "dataset.registered",
            {
                "task_id": task.id,
                "dataset_id": dataset["id"],
                "role": "sample",
            },
            task.id,
        )
    ]


def test_dataset_upload_rejects_invalid_excel_sheet(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    workbook_path = tmp_path / "book.xlsx"
    pd.DataFrame({"id": [1]}).to_excel(workbook_path, sheet_name="Present", index=False)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "feature", "sheet": "Missing"},
        files={
            "file": (
                "book.xlsx",
                workbook_path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 422
    upload_dir = settings.datasets_dir / task.id / "uploads"
    assert not any(upload_dir.rglob("*"))


def test_dataset_upload_streams_to_disk_without_full_buffer_read(tmp_path, monkeypatch):
    """TST-2: the upload endpoint must never pull the whole file into memory
    via a single unbounded read -- assert every read the router issues against
    the underlying SpooledTemporaryFile is bounded by a small chunk size, and
    that more than one chunked read happens for a file bigger than one chunk.
    A real 8MB UPLOAD_CHUNK_SIZE would need a multi-hundred-MB fixture file to
    force a second read, so the module constant is monkeypatched down to a
    trivially small size for this test only.
    """
    from tempfile import SpooledTemporaryFile

    monkeypatch.setattr("marvis.routers.data.UPLOAD_CHUNK_SIZE", 256)

    client, settings = _client(tmp_path)
    task = _create_task(settings)

    read_sizes = []
    original_read = SpooledTemporaryFile.read

    def tracking_read(self, size=-1, *args, **kwargs):
        read_sizes.append(size)
        assert size is not None and size != -1 and size <= 256, (
            f"upload read requested unbounded/oversized chunk: {size!r}"
        )
        return original_read(self, size, *args, **kwargs)

    monkeypatch.setattr(SpooledTemporaryFile, "read", tracking_read)

    # ~4KB, comfortably bigger than the 256-byte patched chunk size, so a
    # true streaming implementation must issue several bounded reads.
    multi_chunk_csv = b"mobile,bad_flag\n" + b"13800138000,0\n" * 300

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", multi_chunk_csv, "text/csv")},
    )

    assert response.status_code == 201
    assert len(read_sizes) > 1, "expected multiple chunked reads, not one full-buffer read"
    assert all(size <= 256 for size in read_sizes)


def test_dataset_upload_rejects_oversized_csv_via_streaming_check(tmp_path, monkeypatch):
    """TST-2: 413 with a Chinese guardrail message when the actual streamed
    bytes exceed the configured CSV limit. Content-Length is deliberately
    withheld (chunked-transfer request body) so this can only be caught by
    the streaming cumulative-byte-count check, not the Content-Length
    pre-check -- proving the streaming check is a real, independent guard and
    not just a formality behind the pre-check."""
    monkeypatch.setattr("marvis.routers.data.UPLOAD_CHUNK_SIZE", 1024)
    client, settings = _client(tmp_path)
    # Shrink the limit so the test doesn't need to generate gigabytes.
    object.__setattr__(settings, "max_csv_upload_bytes", 2048)
    task = _create_task(settings)
    oversized_csv = b"mobile,bad_flag\n" + b"13800138000,0\n" * 500  # well over 2048 bytes

    multipart_request = httpx.Request(
        "POST",
        "http://testserver/irrelevant",
        files={"file": ("big.csv", oversized_csv, "text/csv")},
        data={"role": "sample"},
    )
    body_bytes = multipart_request.read()
    content_type = multipart_request.headers["content-type"]

    def body_without_content_length():
        chunk_size = 200
        for offset in range(0, len(body_bytes), chunk_size):
            yield body_bytes[offset:offset + chunk_size]

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        content=body_without_content_length(),
        headers={"content-type": content_type},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "上传文件大小超过上限" in detail
    assert "2048" in detail
    upload_dir = settings.datasets_dir / task.id / "uploads"
    assert not any(upload_dir.rglob("*"))


def test_dataset_upload_rejects_oversized_content_length_before_reading(tmp_path):
    """TST-2: a declared Content-Length that already exceeds the limit is
    rejected before any staging/writing happens (fast pre-check)."""
    client, settings = _client(tmp_path)
    object.__setattr__(settings, "max_csv_upload_bytes", 100)
    task = _create_task(settings)
    csv_bytes = b"mobile,bad_flag\n13800138000,0\n"

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
        headers={"Content-Length": str(10_000)},
    )

    assert response.status_code == 413
    assert "上传内容大小超过上限" in response.json()["detail"]


def test_dataset_upload_rejects_excel_sheet_over_row_guardrail(tmp_path, monkeypatch):
    """TST-2: Excel sheets are row-probed before the full pd.read_excel load;
    an oversized sheet gets a clear typed error instead of an unbounded read."""
    client, settings = _client(tmp_path)
    object.__setattr__(settings, "max_excel_rows", 3)
    task = _create_task(settings)
    workbook_path = tmp_path / "book.xlsx"
    pd.DataFrame({"id": [1, 2, 3, 4, 5], "bad_flag": [0, 1, 0, 1, 0]}).to_excel(
        workbook_path, sheet_name="Sample", index=False
    )

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={
            "file": (
                "book.xlsx",
                workbook_path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "行数超过上限" in detail
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []


def test_build_settings_reads_upload_guardrails_from_env(tmp_path, monkeypatch):
    """TST-2: upload size/row guardrails are configurable via env var, with
    sane defaults (2GB CSV / 500MB Excel / 2M Excel rows) when unset."""
    from marvis.settings import build_settings

    defaults = build_settings(tmp_path / "defaults_workspace")
    assert defaults.max_csv_upload_bytes == 2 * 1024 * 1024 * 1024
    assert defaults.max_excel_upload_bytes == 500 * 1024 * 1024
    assert defaults.max_excel_rows == 2_000_000

    monkeypatch.setenv("MARVIS_MAX_CSV_UPLOAD_BYTES", "1234")
    monkeypatch.setenv("MARVIS_MAX_EXCEL_UPLOAD_BYTES", "5678")
    monkeypatch.setenv("MARVIS_MAX_EXCEL_ROWS", "42")
    overridden = build_settings(tmp_path / "overridden_workspace")
    assert overridden.max_csv_upload_bytes == 1234
    assert overridden.max_excel_upload_bytes == 5678
    assert overridden.max_excel_rows == 42


def test_register_path_dataset_end_to_end_then_appears_in_list_and_preview(tmp_path):
    """TST-2 (roadmap-1e): register a dataset directly from a local absolute
    path -- no HTTP upload -- then confirm it shows up via list/preview and
    that the source file was copied into the workspace (not registered
    in-place), and that an INV-8 audit row was written."""
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    local_dir = tmp_path / "outside_workspace"
    local_dir.mkdir()
    source_path = local_dir / "local_sample.csv"
    source_path.write_bytes(b"mobile,bad_flag\n13800138000,0\n13900139000,1\n")

    response = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(source_path), "role": "sample"},
    )

    assert response.status_code == 201
    dataset = response.json()["datasets"][0]
    assert dataset["role"] == "sample"
    assert dataset["task_id"] == task.id

    # copied into the workspace, not referencing the original path
    registered_path = settings.datasets_dir / dataset["source_path"]
    assert registered_path.exists()
    assert registered_path.resolve() != source_path.resolve()

    listing = client.get(f"/api/tasks/{task.id}/datasets").json()
    assert [item["id"] for item in listing["datasets"]] == [dataset["id"]]

    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=2")
    assert preview.status_code == 200
    assert preview.json()["columns"] == ["mobile", "bad_flag"]

    from marvis.repositories.audit import _list_audit_rows

    audit_rows = _list_audit_rows(settings.db_path, kind="dataset.registered_from_path")
    assert len(audit_rows) == 1
    assert audit_rows[0]["target_ref"] == dataset["id"]
    assert audit_rows[0]["detail"]["source_path"] == str(source_path.resolve())
    assert audit_rows[0]["detail"]["task_id"] == task.id


def test_register_path_dataset_reuses_existing_dataset_by_content_hash(tmp_path):
    """TST-2: local-path registration reuses register_from_upload's GAP-7
    content-hash dedup -- registering the same bytes twice does not write a
    second parquet file."""
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    source_path = tmp_path / "dup.csv"
    source_path.write_bytes(b"mobile,bad_flag\n13800138000,0\n")

    first = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(source_path), "role": "sample"},
    )
    second = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(source_path), "role": "feature"},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    dataset_a = first.json()["datasets"][0]
    dataset_b = second.json()["datasets"][0]
    assert dataset_a["content_hash"] == dataset_b["content_hash"]
    assert dataset_a["source_path"] == dataset_b["source_path"]
    assert len(list((settings.datasets_dir / task.id).glob("*.parquet"))) == 1


def test_register_path_rejects_nonexistent_path(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(tmp_path / "does_not_exist.csv"), "role": "sample"},
    )

    assert response.status_code == 422
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []


def test_register_path_rejects_directory_path(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    a_directory = tmp_path / "a_directory.csv"
    a_directory.mkdir()

    response = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(a_directory), "role": "sample"},
    )

    assert response.status_code == 422
    assert "not a regular file" in response.json()["detail"]


def test_register_path_rejects_disallowed_extension(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("not a dataset")

    response = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(bad_path), "role": "sample"},
    )

    assert response.status_code == 422
    assert "unsupported file extension" in response.json()["detail"]


def test_register_path_rejects_relative_path(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": "relative/sample.csv", "role": "sample"},
    )

    assert response.status_code == 422
    assert "must be absolute" in response.json()["detail"]


def test_register_path_rejects_oversized_file(tmp_path):
    client, settings = _client(tmp_path)
    object.__setattr__(settings, "max_csv_upload_bytes", 32)
    task = _create_task(settings)
    source_path = tmp_path / "too_big.csv"
    source_path.write_bytes(b"mobile,bad_flag\n" + b"13800138000,0\n" * 50)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/register-path",
        json={"path": str(source_path), "role": "sample"},
    )

    assert response.status_code == 413
    assert "本地路径注册文件大小超过上限" in response.json()["detail"]
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []


def test_dataset_upload_raw_file_rolls_back_when_registration_fails(
    tmp_path,
    monkeypatch,
):
    client, settings = _client(tmp_path, raise_server_exceptions=False)
    task = _create_task(settings)

    def fail_dataset_insert(self, conn, dataset):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(
        DatasetRepository,
        "create_dataset_on_connection",
        fail_dataset_insert,
    )

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", b"mobile,bad_flag\n13800138000,0\n", "text/csv")},
    )

    assert response.status_code == 500
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []
    task_dataset_dir = settings.datasets_dir / task.id
    assert not list((task_dataset_dir / "uploads").glob("*.csv"))
    assert not ((task_dataset_dir / "uploads") / ".staging").exists()
    assert not list(task_dataset_dir.glob("sample_*.parquet"))
    assert not (task_dataset_dir / ".staging").exists()


def test_dataset_upload_excel_multi_sheet_rolls_back_when_registration_fails(
    tmp_path,
    monkeypatch,
):
    client, settings = _client(tmp_path, raise_server_exceptions=False)
    task = _create_task(settings)
    workbook_path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame({"id": [1], "bad_flag": [0]}).to_excel(
            writer,
            sheet_name="Sample",
            index=False,
        )
        pd.DataFrame({"id": [2], "bad_flag": [1]}).to_excel(
            writer,
            sheet_name="Feature",
            index=False,
        )
    original_create = DatasetRepository.create_dataset_on_connection
    call_count = {"value": 0}

    def fail_second_dataset_insert(self, conn, dataset):
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise RuntimeError("db unavailable")
        return original_create(self, conn, dataset)

    monkeypatch.setattr(
        DatasetRepository,
        "create_dataset_on_connection",
        fail_second_dataset_insert,
    )

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "feature"},
        files={
            "file": (
                "book.xlsx",
                workbook_path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 500
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []
    upload_dir = settings.datasets_dir / task.id / "uploads"
    assert not list(upload_dir.glob("*.xlsx"))
    assert not (upload_dir / ".staging").exists()
    excel_dir = settings.datasets_dir / task.id / "excel"
    assert not list(excel_dir.glob("*.parquet"))
    assert not (excel_dir / ".staging").exists()
    assert not list(excel_dir.glob(".excel_ingest_*"))


def test_join_api_propose_confirm_execute_flow(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        "feature",
    )

    propose = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    )

    assert propose.status_code == 201
    plan = propose.json()
    assert plan["status"] == "draft"
    assert plan["anchor_dataset_id"] == anchor.id
    assert plan["joins"][0]["key_pairs"][0]["match_method"] == "hash:md5"

    blocked = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
    assert blocked.status_code == 409

    confirm = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True, "dedup_strategy": None},
    )
    fetched = client.get(f"/api/joins/{plan['join_plan_id']}")
    execute = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
    repeat = client.post(f"/api/joins/{plan['join_plan_id']}/execute")

    assert confirm.status_code == 200
    assert confirm.json()["anchor_dataset_id"] == anchor.id
    assert confirm.json()["joins"][0]["confirmed"] is True
    assert fetched.status_code == 200
    assert fetched.json()["anchor_dataset_id"] == anchor.id
    assert execute.status_code == 200
    assert execute.json()["anchor_rows"] == 2
    assert execute.json()["joined_rows"] == 2
    assert repeat.status_code == 409
    audits = PluginRepository(settings.db_path).list_audit()
    audit_kinds = [audit["kind"] for audit in audits]
    assert "join.confirmed" in audit_kinds
    assert "join.executed" in audit_kinds
    executed_audit = next(audit for audit in audits if audit["kind"] == "join.executed")
    assert executed_audit["target_ref"] == plan["join_plan_id"]
    assert executed_audit["detail"]["result_dataset_id"] == execute.json()["result_dataset_id"]


def test_join_confirm_recomputes_single_key_match_evidence_server_side(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "single_recount_anchor",
        pd.DataFrame({"proposal_id": [1, 2], "client_key": ["A", "B"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "single_recount_feature",
        pd.DataFrame({"proposal_id": [1, 2], "server_key": ["X", "Y"]}),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={
            "feature_id": feature.id,
            "confirmed": False,
            "dedup_strategy": None,
            "key_pairs": [
                {
                    "anchor_col": "client_key",
                    "feature_col": "server_key",
                    "match_method": "exact",
                    "transform_side": "both",
                    "match_rate": 1.0,
                    "resolved_by": "client-forged",
                }
            ],
        },
    )

    assert response.status_code == 200
    join = response.json()["joins"][0]
    assert join["key_pairs"][0]["match_rate"] == 0.0
    assert join["key_pairs"][0]["resolved_by"] == "user"
    assert join["diagnostics"]["matched_rows"] == 0
    assert join["diagnostics"]["match_rate"] == 0.0


def test_join_confirm_recount_uses_the_selected_hash_transform_side(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    phone_numbers = ["13800138000", "13900139000"]
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "transform_side_anchor",
        pd.DataFrame({"proposal_id": [1, 2], "phone": phone_numbers}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "transform_side_feature",
        pd.DataFrame({
            "proposal_id": [1, 2],
            "phone_md5": [hashlib.md5(value.encode()).hexdigest() for value in phone_numbers],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={
            "feature_id": feature.id,
            "confirmed": False,
            "dedup_strategy": None,
            "key_pairs": [
                {
                    "anchor_col": "phone",
                    "feature_col": "phone_md5",
                    "match_method": "hash:md5",
                    # Hashing both sides double-hashes the already-hashed feature.
                    "transform_side": "both",
                    "match_rate": 1.0,
                }
            ],
        },
    )

    assert response.status_code == 200
    join = response.json()["joins"][0]
    assert join["key_pairs"][0]["match_rate"] == 0.0
    assert join["diagnostics"]["matched_rows"] == 0
    assert join["diagnostics"]["match_rate"] == 0.0


def test_join_confirm_recomputes_each_key_and_combined_match_evidence(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "multi_recount_anchor",
        pd.DataFrame({
            "proposal_id": [1, 2, 3],
            "left_a": ["A", "B", "C"],
            "left_b": ["1", "2", "3"],
        }),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "multi_recount_feature",
        pd.DataFrame({
            "proposal_id": [1, 2, 3],
            "right_a": ["A", "B", "X"],
            "right_b": ["1", "9", "3"],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={
            "feature_id": feature.id,
            "confirmed": False,
            "dedup_strategy": None,
            "key_pairs": [
                {
                    "anchor_col": "left_a",
                    "feature_col": "right_a",
                    "match_method": "exact",
                    "transform_side": "both",
                    "match_rate": 1.0,
                },
                {
                    "anchor_col": "left_b",
                    "feature_col": "right_b",
                    "match_method": "exact",
                    "transform_side": "both",
                    "match_rate": 1.0,
                },
            ],
        },
    )

    assert response.status_code == 200
    join = response.json()["joins"][0]
    assert [pair["match_rate"] for pair in join["key_pairs"]] == pytest.approx(
        [2 / 3, 2 / 3],
        abs=0.0001,
    )
    assert {pair["resolved_by"] for pair in join["key_pairs"]} == {"user"}
    assert join["diagnostics"]["matched_rows"] == 1
    assert join["diagnostics"]["match_rate"] == pytest.approx(1 / 3, abs=0.0001)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("match_method", "client-python"),
        ("transform_side", "somewhere-else"),
    ],
)
def test_join_confirm_rejects_unknown_key_pair_enums(tmp_path, field, value):
    client, settings = _client(tmp_path, raise_server_exceptions=False)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        f"invalid_{field}_anchor",
        pd.DataFrame({"customer_id": [1, 2]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        f"invalid_{field}_feature",
        pd.DataFrame({"customer_id": [1, 2]}),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()
    key_pair = {
        "anchor_col": "customer_id",
        "feature_col": "customer_id",
        "match_method": "exact",
        "transform_side": "both",
    }
    key_pair[field] = value

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={
            "feature_id": feature.id,
            "confirmed": False,
            "dedup_strategy": None,
            "key_pairs": [key_pair],
        },
    )

    assert response.status_code == 422


def test_join_api_execute_supports_explicit_async_job(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"customer_id": [1, 2]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({"customer_id": [1, 2], "balance": [10, 20]}),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()
    client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True},
    )

    execute = client.post(f"/api/joins/{plan['join_plan_id']}/execute", json={"async": True})

    assert execute.status_code == 202
    payload = execute.json()
    assert payload["status"] == "accepted"
    assert payload["job_id"]
    assert payload["task_id"] == task.id
    loaded = DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"])
    assert loaded.status == "executed"
    assert loaded.result_dataset_id
    assert TaskRepository(settings.db_path).get_active_job_kind(task.id) is None


@pytest.mark.parametrize(
    ("payload", "expected_status_code"),
    [
        ({"async_execute": False}, 200),
        ({"async_execute": "false"}, 200),
        ({"async_execute": "0"}, 200),
        ({"async_execute": True}, 202),
        ({"async_execute": "true"}, 202),
        ({"async_execute": "1"}, 202),
    ],
)
def test_join_api_execute_coerces_async_flags(tmp_path, payload, expected_status_code):
    client, settings = _client(tmp_path)
    task, plan = _confirmed_join_plan(client, settings, tmp_path)

    execute = client.post(f"/api/joins/{plan['join_plan_id']}/execute", json=payload)

    assert execute.status_code == expected_status_code
    body = execute.json()
    if expected_status_code == 202:
        assert body["status"] == "accepted"
        assert body["job_id"]
    else:
        assert body["result_dataset_id"]
    assert DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"]).status == "executed"
    assert TaskRepository(settings.db_path).get_active_job_kind(task.id) is None


def test_join_api_execute_async_rejects_active_task_job(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"customer_id": [1, 2]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({"customer_id": [1, 2], "balance": [10, 20]}),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()
    client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True},
    )
    TaskRepository(settings.db_path).start_job(task.id, "metrics")

    execute = client.post(f"/api/joins/{plan['join_plan_id']}/execute", json={"async": True})

    assert execute.status_code == 409
    assert DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"]).status != "executed"


def test_join_api_execute_sync_rejects_active_task_job(tmp_path):
    client, settings = _client(tmp_path)
    task, plan = _confirmed_join_plan(client, settings, tmp_path)
    TaskRepository(settings.db_path).start_job(task.id, "metrics")

    execute = client.post(f"/api/joins/{plan['join_plan_id']}/execute")

    assert execute.status_code == 409
    assert DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"]).status != "executed"


def test_join_api_execute_sync_second_concurrent_call_gets_409_without_double_execution(tmp_path):
    client, settings = _client(tmp_path)
    task, plan = _confirmed_join_plan(client, settings, tmp_path)

    # Simulate a second synchronous request racing in while the first sync
    # execution's job is still claimed (TOCTOU window from REL-9).
    task_repo = TaskRepository(settings.db_path)
    job_id = task_repo.start_job(task.id, "join")
    try:
        second = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
        assert second.status_code == 409
        assert (
            DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"]).status
            != "executed"
        )
    finally:
        task_repo.finish_job(job_id, status="succeeded")

    # Once the guard clears, a legitimate execute call still succeeds exactly once.
    first = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
    assert first.status_code == 200
    assert TaskRepository(settings.db_path).get_active_job_kind(task.id) is None
    repeat = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
    assert repeat.status_code == 409


def test_join_cancel_endpoint_rejects_when_no_active_join_job(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)

    response = client.post(f"/api/tasks/{task.id}/join/cancel")

    assert response.status_code == 409
    assert "no active join job" in response.json()["detail"]


def test_join_cancel_endpoint_rejects_unknown_task(tmp_path):
    client, _settings = _client(tmp_path)

    response = client.post("/api/tasks/missing-task/join/cancel")

    assert response.status_code == 404


def test_join_cancel_endpoint_signals_the_running_jobs_cancellation_token(tmp_path):
    # REL-5: the cancel endpoint is cooperative — it flips the in-memory token
    # for the job actually recorded as active, it doesn't touch the DB status
    # itself (the join engine's own finish_job(status="cancelled") does that
    # once it observes the token at its next checkpoint).
    client, settings = _client(tmp_path)
    task, _plan = _confirmed_join_plan(client, settings, tmp_path)
    task_repo = TaskRepository(settings.db_path)
    job_id = task_repo.start_job(task.id, "join")
    task_repo.mark_job_running(job_id)
    token = register_job_cancellation(job_id)
    try:
        response = client.post(f"/api/tasks/{task.id}/join/cancel")

        assert response.status_code == 202
        body = response.json()
        assert body["job_id"] == job_id
        assert body["status"] == "accepted"
        assert token.is_cancelled() is True
    finally:
        unregister_job_cancellation(job_id, token)
        task_repo.finish_job(job_id, status="cancelled")


def test_join_cancel_endpoint_unlocks_task_after_running_job_is_cancelled(tmp_path):
    # End-to-end (task requirement c): a job is started, its cancellation token
    # is armed through the same cancel endpoint the frontend would call, and
    # once the background runner actually observes it at its checkpoint the
    # task must come out unlocked (idx_jobs_active_task released) with a
    # cancelled job status and no dangling "executed" join plan. TestClient's
    # BackgroundTasks run synchronously inside client.post() (Starlette detail),
    # so genuine cross-thread concurrency during the HTTP call isn't
    # observable here — this test instead proves the wiring end to end: cancel
    # endpoint -> token -> engine checkpoint -> job/task state, by arming the
    # token via the HTTP endpoint before the runner is invoked, which is
    # exactly the "cancel requested, then job's next checkpoint sees it" path.
    client, settings = _client(tmp_path)
    task, plan = _confirmed_join_plan(client, settings, tmp_path)
    task_repo = TaskRepository(settings.db_path)
    job_id = task_repo.start_job(task.id, "join")

    # Cancel arrives before the runner has registered its own token (the
    # queued-job window) — this is exactly what the "pending" cancel request
    # in JobCancellationRegistry exists for: the runner's later
    # register_job_cancellation(job_id) call picks up the already-requested
    # cancellation instead of silently starting fresh.
    cancel = client.post(f"/api/tasks/{task.id}/join/cancel")
    assert cancel.status_code == 202
    assert cancel.json()["job_id"] == job_id

    _run_join_execute_job(job_id, settings.db_path, settings.datasets_dir, plan["join_plan_id"])

    final_job = task_repo.get_job(job_id)
    assert final_job["status"] == "cancelled"
    assert task_repo.task_has_active_job(task.id) is False
    assert (
        DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"]).status
        != "executed"
    )


def test_join_callback_does_not_execute_after_queued_job_is_cancelled(tmp_path):
    client, settings = _client(tmp_path)
    task, plan = _confirmed_join_plan(client, settings, tmp_path)
    task_repo = TaskRepository(settings.db_path)
    job_id = task_repo.start_job(task.id, "join")
    request_job_cancellation(job_id)
    task_repo.finish_job(job_id, status="cancelled")

    _run_join_execute_job(
        job_id,
        settings.db_path,
        settings.datasets_dir,
        plan["join_plan_id"],
    )

    assert task_repo.get_job(job_id)["status"] == "cancelled"
    assert (
        DatasetRepository(settings.db_path).load_join_plan(plan["join_plan_id"]).status
        != "executed"
    )
    token = register_job_cancellation(job_id)
    try:
        assert token.is_cancelled() is False
    finally:
        unregister_job_cancellation(job_id, token)


def test_join_api_marks_aggregate_dedup_as_synthetic(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"customer_id": [1, 2]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "customer_id": [1, 1, 2],
            "balance": [10, 20, 30],
            "segment": ["old", "new", "steady"],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    confirm = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True, "dedup_strategy": "agg_mean"},
    )

    assert confirm.status_code == 200
    join = confirm.json()["joins"][0]
    assert join["dedup_strategy"] == "agg_mean"
    assert "synthesize" in join["dedup_strategy_warning"]
    fetched = client.get(f"/api/joins/{plan['join_plan_id']}")
    assert fetched.json()["joins"][0]["dedup_strategy_warning"] == join["dedup_strategy_warning"]


def test_join_confirm_dispatches_join_confirmed_hook(tmp_path):
    client, settings = _client(tmp_path)
    dispatcher = FakeHookDispatcher()
    client.app.state.hook_dispatcher = dispatcher
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True, "dedup_strategy": None},
    )

    assert response.status_code == 200
    assert dispatcher.calls == [
        (
            "join.confirmed",
            {
                "task_id": task.id,
                "join_plan_id": plan["join_plan_id"],
                "feature_id": feature.id,
                "confirmed": True,
            },
            task.id,
        )
    ]


def test_join_confirm_accepts_feature_dataset_id_alias(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_dataset_id": feature.id, "confirmed": True, "dedup_strategy": None},
    )

    assert response.status_code == 200
    assert response.json()["joins"][0]["confirmed"] is True


def test_join_api_keeps_task_dataset_boundaries(tmp_path):
    client, settings = _client(tmp_path)
    task_1 = _create_task(settings)
    task_2 = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task_1.id,
        "anchor",
        pd.DataFrame({"acct_num": ["A1", "B2"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task_2.id,
        "feature",
        pd.DataFrame({"acct_no": ["A1", "B2"]}),
        "feature",
    )

    response = client.post(
        f"/api/tasks/{task_1.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    )

    assert response.status_code == 404



@pytest.mark.asyncio
async def test_sync_join_execute_does_not_block_event_loop_for_health_polling(tmp_path, monkeypatch):
    """PERF-1 regression: propose/confirm/execute run heavy sync work inside plain
    ``def`` endpoints so FastAPI offloads them to a worker thread; a slow synchronous
    join execute must not freeze the single-process event loop that ``/api/health``
    polling depends on. A monkeypatched sleep stands in for "heavy sync work" so the
    assertion is deterministic instead of depending on a real dataset being slow
    enough on any given machine (DuckDB joins are fast even at millions of rows)."""
    app = create_app(tmp_path)
    settings = app.state.settings
    task, plan = _confirmed_join_plan(TestClient(app), settings, tmp_path)

    real_execute = JoinEngine.execute_join_plan

    def _slow_execute_join_plan(self, join_plan_id, *, out_dir, **kwargs):
        time.sleep(1.5)
        return real_execute(self, join_plan_id, out_dir=out_dir, **kwargs)

    monkeypatch.setattr(JoinEngine, "execute_join_plan", _slow_execute_join_plan)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Timed from BEFORE the join is even dispatched: if the join endpoint blocked
        # the event loop, the 0.2s head-start sleep below would itself absorb the
        # entire 1.5s block (since nothing else can run on a monopolized loop), making
        # a health_start timestamp taken *after* the head start a no-op measurement
        # that always looks fast. Measuring from here is what actually catches a
        # regression back to `async def` + synchronous heavy work.
        overall_start = time.monotonic()
        join_task = asyncio.create_task(
            client.post(f"/api/joins/{plan['join_plan_id']}/execute")
        )
        # Give the join request a head start so it is genuinely in flight (occupying
        # its worker thread) before health is polled.
        await asyncio.sleep(0.2)

        health_response = await client.get("/api/health")
        health_elapsed = time.monotonic() - overall_start

        assert health_response.status_code == 200
        # health should return right after the 0.2s head start (the join keeps
        # running in its own worker thread); a regression back to `async def` +
        # synchronous heavy work would make this take >=1.5s (the full join sleep)
        # instead, so 1.0s is a guardrail that is generous for CI jitter yet tight
        # enough to fail hard on that regression. The <2s target from the spec is
        # even more generous still and would not reliably catch this bug.
        assert health_elapsed < 1.0

        join_response = await join_task
        assert join_response.status_code == 200
        assert join_response.json()["result_dataset_id"]

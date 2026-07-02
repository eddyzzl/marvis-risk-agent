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
from marvis.job_cancellation import register_job_cancellation, unregister_job_cancellation
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
    task_repo.mark_job_running(job_id)

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

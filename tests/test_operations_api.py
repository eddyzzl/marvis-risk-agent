from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository, connect
from marvis.operations.contracts import (
    HumanEscalationSuggestion,
    MonitoringOutcome,
)


GOVERNANCE_ADMIN_HEADER = "x-marvis-governance-admin"
FIXED_NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _claim_role(app, client: TestClient, role: str) -> dict:
    response = client.post(
        "/api/production-governance/principals/claim",
        headers={GOVERNANCE_ADMIN_HEADER: app.state.plugin_admin_token},
        json={"display_name": f"Operations {role}", "role": role},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _publish_payload(
    *,
    monitoring_ref: str = "strategy-monitor-binding-one",
    schedule_id: str = "schedule-one",
) -> dict:
    anchor = FIXED_NOW - timedelta(minutes=2)
    return {
        "expected_revision": 0,
        "schedule": {
            "schedule_id": schedule_id,
            "revision": 1,
            "monitoring_ref": monitoring_ref,
            "active_from": anchor.isoformat(),
            "calendar": {
                "anchor_at": anchor.isoformat(),
                "interval_seconds": 60,
            },
            "catch_up_budget": 1,
            "lease_seconds": 30,
            "retry_policy": {
                "max_attempts": 2,
                "initial_backoff_seconds": 0,
                "multiplier": 1,
                "max_backoff_seconds": 0,
            },
            "enabled": True,
        },
    }


def _enqueue_dataset_source_gc_entry(
    app,
    *,
    source_path: str,
    state: str,
    next_attempt_at: str,
) -> None:
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO dataset_source_gc_queue(
                source_path, state, attempt_count, next_attempt_at,
                last_error_code, last_error_message, origin_task_id,
                enqueued_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_path,
                state,
                2,
                next_attempt_at,
                "PermissionError" if state == "quarantined" else None,
                "sharing violation" if state == "quarantined" else None,
                "task-source-gc",
                "2026-08-01T19:55:00+08:00",
                "2026-08-01T20:00:00+08:00",
            ),
        )


def _enqueue_task_filesystem_gc_entry(
    app,
    *,
    target_type: str,
    relative_path: str,
    state: str,
    next_attempt_at: str,
    origin_task_id: str,
) -> None:
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO task_fs_gc_queue(
                target_type, relative_path, state, attempt_count,
                next_attempt_at, last_error_code, last_error_message,
                origin_task_id, origin_task_created_at, enqueued_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target_type,
                relative_path,
                state,
                2,
                next_attempt_at,
                "PermissionError" if state == "quarantined" else None,
                "directory busy" if state == "quarantined" else None,
                origin_task_id,
                "2026-08-01T11:50:00Z",
                "2026-08-01T19:55:00+08:00",
                "2026-08-01T20:00:00+08:00",
            ),
        )


def test_operations_api_is_session_role_gated_and_keeps_escalation_advisory(tmp_path):
    calls = []

    def execute(request):
        calls.append(request)
        return MonitoringOutcome(
            level="amber",
            evidence_ref=f"local-monitoring://{request.run_id}",
            evidence_hash="b" * 64,
            escalation=HumanEscalationSuggestion(reason_code="monitoring_amber"),
        )

    app = create_app(
        tmp_path,
        operations_executor_allowlist={"strategy-monitor-binding-one": execute},
        operations_clock=lambda: FIXED_NOW,
    )
    unclaimed_client = TestClient(app)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, "maker")
    _claim_role(app, checker_client, "checker")
    _claim_role(app, admin_client, "admin")

    unclaimed = unclaimed_client.get("/api/operations/schedules/schedule-one")
    assert unclaimed.status_code == 403

    checker_publish = checker_client.post(
        "/api/operations/schedules",
        json=_publish_payload(),
    )
    assert checker_publish.status_code == 403

    published = maker_client.post(
        "/api/operations/schedules",
        json=_publish_payload(),
    )
    assert published.status_code == 201, published.text
    assert (
        published.json()["contract"]["monitoring_ref"] == "strategy-monitor-binding-one"
    )

    readable = checker_client.get("/api/operations/schedules/schedule-one")
    assert readable.status_code == 200, readable.text
    assert readable.json() == published.json()
    active = checker_client.get("/api/operations/schedules")
    assert active.status_code == 200, active.text
    assert active.json() == {"schedules": [published.json()], "count": 1}

    maker_tick = maker_client.post(
        "/api/operations/tick",
        json={"catch_up_budget": 1, "notification_budget": 0},
    )
    assert maker_tick.status_code == 403

    ticked = admin_client.post(
        "/api/operations/tick",
        json={"catch_up_budget": 1, "notification_budget": 0},
    )
    assert ticked.status_code == 200, ticked.text
    tick = ticked.json()
    assert tick["succeeded"] == 1
    assert len(calls) == 1

    period = admin_client.get(
        "/api/operations/schedules/schedule-one/period",
        params={"period_key": tick["claimed_period_keys"][0]},
    )
    assert period.status_code == 200, period.text
    escalation = period.json()["outcome"]["outcome"]["escalation"]
    assert escalation == {
        "schema_version": "operations.human_escalation.v1",
        "action": "review_monitoring_result",
        "reason_code": "monitoring_amber",
        "requires_human_confirmation": True,
        "automatic_action_permitted": False,
    }

    maker_recovery = maker_client.post("/api/operations/recover")
    assert maker_recovery.status_code == 403
    recovered = admin_client.post("/api/operations/recover")
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["period_leases_recovered"] == 0


def test_operations_api_rejects_unknown_or_request_supplied_callable_before_persisting(
    tmp_path,
):
    app = create_app(tmp_path)
    admin_client = TestClient(app)
    _claim_role(app, admin_client, "admin")

    assert app.state.operations_runtime.store.db_path == app.state.settings.db_path
    assert app.state.operations_runtime.allowed_monitoring_refs == ()

    unknown = admin_client.post(
        "/api/operations/schedules",
        json=_publish_payload(monitoring_ref="os.system"),
    )
    assert unknown.status_code == 422, unknown.text
    assert app.state.operations_runtime.store.get_schedule("schedule-one") is None

    supplied_callable = _publish_payload(schedule_id="supplied-callable")
    supplied_callable["executor"] = "attacker_module:run"
    rejected = admin_client.post(
        "/api/operations/schedules",
        json=supplied_callable,
    )
    assert rejected.status_code == 422
    assert app.state.operations_runtime.store.get_schedule("supplied-callable") is None


def test_operations_queries_stay_local_when_remote_read_is_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    app = create_app(tmp_path)
    remote = TestClient(app, client=("203.0.113.9", 5555))

    response = remote.get("/api/operations/schedules/schedule-one")

    assert response.status_code == 403
    assert response.json()["detail"] == "this endpoint is limited to local clients"


def test_operations_dataset_source_gc_status_is_bounded_and_serializes_utc(tmp_path):
    app = create_app(tmp_path)
    checker_client = TestClient(app)
    _claim_role(app, checker_client, "checker")
    _enqueue_dataset_source_gc_entry(
        app,
        source_path="pending/source.parquet",
        state="pending",
        next_attempt_at="2099-08-01T20:05:00+08:00",
    )
    _enqueue_dataset_source_gc_entry(
        app,
        source_path="quarantined/source.parquet",
        state="quarantined",
        next_attempt_at="2026-08-01T20:05:00+08:00",
    )

    response = checker_client.get(
        "/api/operations/dataset-source-gc",
        params={"limit": 1},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pending_count"] == 1
    assert body["quarantined_count"] == 1
    assert body["offset"] == 0
    assert body["returned_count"] == 1
    assert body["has_more"] is True
    assert body["entries"] == [
        {
            "source_path": "quarantined/source.parquet",
            "state": "quarantined",
            "attempt_count": 2,
            "next_attempt_at": "2026-08-01T12:05:00Z",
            "last_error_code": "PermissionError",
            "last_error_message": "sharing violation",
            "origin_task_id": "task-source-gc",
            "enqueued_at": "2026-08-01T11:55:00Z",
            "updated_at": "2026-08-01T12:00:00Z",
        }
    ]
    assert body["watchdog"]["alive"] is False
    assert body["watchdog"]["last_report"] is None
    assert body["startup"]["error"] is None
    assert body["startup"]["report"]["examined"] == 0

    second_page = checker_client.get(
        "/api/operations/dataset-source-gc",
        params={"limit": 1, "offset": 1},
    )
    assert second_page.status_code == 200, second_page.text
    assert second_page.json()["entries"][0]["source_path"] == (
        "pending/source.parquet"
    )
    assert second_page.json()["has_more"] is False


def test_operations_dataset_source_gc_sweep_is_admin_governed_and_bounded(tmp_path):
    app = create_app(tmp_path)
    maker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, "maker")
    admin = _claim_role(app, admin_client, "admin")
    source_path = "queued/source.parquet"
    source_file = app.state.settings.datasets_dir / source_path
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"dataset-source")
    _enqueue_dataset_source_gc_entry(
        app,
        source_path=source_path,
        state="pending",
        next_attempt_at="2099-08-01T20:05:00+08:00",
    )

    maker_response = maker_client.post(
        "/api/operations/dataset-source-gc/sweep",
        json={"limit": 1, "force": True},
    )
    not_due = admin_client.post(
        "/api/operations/dataset-source-gc/sweep",
        json={"limit": 1, "force": False},
    )
    forced = admin_client.post(
        "/api/operations/dataset-source-gc/sweep",
        json={"limit": 1, "force": True},
    )

    assert maker_response.status_code == 403
    assert not_due.status_code == 200, not_due.text
    assert not_due.json() == {
        "examined": 0,
        "deleted": 0,
        "cancelled_referenced": 0,
        "deferred": 0,
        "quarantined": 0,
    }
    assert forced.status_code == 200, forced.text
    assert forced.json() == {
        "examined": 1,
        "deleted": 1,
        "cancelled_referenced": 0,
        "deferred": 0,
        "quarantined": 0,
    }
    assert not source_file.exists()
    audit = TaskRepository(app.state.settings.db_path).list_audit(
        kind="dataset_source_gc.manual_sweep",
    )
    assert len(audit) == 4
    assert [row["outcome"] for row in audit] == [
        "started",
        "succeeded",
        "started",
        "succeeded",
    ]
    assert audit[-1]["actor"] == admin["id"]
    assert audit[-1]["outcome"] == "succeeded"
    assert audit[-1]["detail"] == {
        "limit": 1,
        "force": True,
        "report": forced.json(),
    }


def test_operations_dataset_source_gc_status_survives_corrupt_timestamp(tmp_path):
    app = create_app(tmp_path)
    admin_client = TestClient(app)
    _claim_role(app, admin_client, "admin")
    _enqueue_dataset_source_gc_entry(
        app,
        source_path="corrupt/time.parquet",
        state="pending",
        next_attempt_at="not-a-time",
    )
    with connect(app.state.settings.db_path) as connection:
        connection.execute(
            """
            UPDATE dataset_source_gc_queue
               SET enqueued_at = 'also-bad'
             WHERE source_path = 'corrupt/time.parquet'
            """
        )

    response = admin_client.get("/api/operations/dataset-source-gc")

    assert response.status_code == 200, response.text
    entry = response.json()["entries"][0]
    assert entry["next_attempt_at"] is None
    assert entry["enqueued_at"] is None
    assert entry["last_error_code"] == "CorruptDatasetSourceGcTimestamp"
    assert "next_attempt_at" in entry["last_error_message"]


def test_operations_dataset_source_gc_rejects_unbounded_or_coerced_controls(tmp_path):
    app = create_app(tmp_path)
    admin_client = TestClient(app)
    _claim_role(app, admin_client, "admin")

    for payload in (
        {"limit": 0, "force": False},
        {"limit": 501, "force": False},
        {"limit": True, "force": False},
        {"limit": 1, "force": "true"},
        {"limit": 1, "force": False, "source_paths": ["queued/source.parquet"]},
    ):
        response = admin_client.post(
            "/api/operations/dataset-source-gc/sweep",
            json=payload,
        )
        assert response.status_code == 422, (payload, response.text)

    assert (
        admin_client.get(
            "/api/operations/dataset-source-gc",
            params={"limit": 0},
        ).status_code
        == 422
    )
    assert (
        admin_client.get(
            "/api/operations/dataset-source-gc",
            params={"limit": 5001},
        ).status_code
        == 422
    )


def test_operations_dataset_source_gc_controls_stay_local(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    app = create_app(tmp_path)
    remote = TestClient(app, client=("203.0.113.9", 5555))

    status = remote.get("/api/operations/dataset-source-gc")
    sweep = remote.post(
        "/api/operations/dataset-source-gc/sweep",
        json={"limit": 1, "force": False},
    )

    assert status.status_code == 403
    assert status.json()["detail"] == "this endpoint is limited to local clients"
    assert sweep.status_code == 403
    assert sweep.json()["detail"] == "unsafe API methods are limited to local clients"


def test_operations_task_filesystem_gc_status_is_bounded_and_serializes_utc(
    tmp_path,
):
    app = create_app(tmp_path)
    checker_client = TestClient(app)
    _claim_role(app, checker_client, "checker")
    _enqueue_task_filesystem_gc_entry(
        app,
        target_type="task_dir",
        relative_path="pending-task",
        state="pending",
        next_attempt_at="2099-08-01T20:05:00+08:00",
        origin_task_id="pending-task",
    )
    _enqueue_task_filesystem_gc_entry(
        app,
        target_type="task_dir",
        relative_path="quarantined-task",
        state="quarantined",
        next_attempt_at="2026-08-01T20:05:00+08:00",
        origin_task_id="quarantined-task",
    )

    response = checker_client.get(
        "/api/operations/task-filesystem-gc",
        params={"limit": 1},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pending_count"] == 1
    assert body["quarantined_count"] == 1
    assert body["offset"] == 0
    assert body["returned_count"] == 1
    assert body["has_more"] is True
    assert body["entries"] == [
        {
            "target_type": "task_dir",
            "relative_path": "quarantined-task",
            "state": "quarantined",
            "attempt_count": 2,
            "next_attempt_at": "2026-08-01T12:05:00Z",
            "last_error_code": "PermissionError",
            "last_error_message": "directory busy",
            "origin_task_id": "quarantined-task",
            "origin_task_created_at": "2026-08-01T11:50:00Z",
            "enqueued_at": "2026-08-01T11:55:00Z",
            "updated_at": "2026-08-01T12:00:00Z",
        }
    ]
    assert body["watchdog"]["alive"] is False
    assert body["watchdog"]["last_report"] is None
    assert body["startup"]["error"] is None
    assert body["startup"]["report"]["examined"] == 0

    second_page = checker_client.get(
        "/api/operations/task-filesystem-gc",
        params={"limit": 1, "offset": 1},
    )
    assert second_page.status_code == 200, second_page.text
    assert second_page.json()["entries"][0]["relative_path"] == "pending-task"
    assert second_page.json()["has_more"] is False


def test_operations_task_filesystem_gc_sweep_is_admin_governed_and_audited(
    tmp_path,
):
    app = create_app(tmp_path)
    maker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, "maker")
    admin = _claim_role(app, admin_client, "admin")
    task_id = "orphan-task"
    task_dir = app.state.settings.tasks_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "result.txt").write_text("delete", encoding="utf-8")
    _enqueue_task_filesystem_gc_entry(
        app,
        target_type="task_dir",
        relative_path=task_id,
        state="pending",
        next_attempt_at="2099-08-01T20:05:00+08:00",
        origin_task_id=task_id,
    )

    denied = maker_client.post(
        "/api/operations/task-filesystem-gc/sweep",
        json={"limit": 1, "force": True},
    )
    not_due = admin_client.post(
        "/api/operations/task-filesystem-gc/sweep",
        json={"limit": 1, "force": False},
    )
    forced = admin_client.post(
        "/api/operations/task-filesystem-gc/sweep",
        json={"limit": 1, "force": True},
    )

    assert denied.status_code == 403
    assert not_due.status_code == 200, not_due.text
    assert not_due.json()["examined"] == 0
    assert forced.status_code == 200, forced.text
    assert forced.json() == {
        "examined": 1,
        "deleted": 1,
        "cancelled_referenced": 0,
        "deferred": 0,
        "quarantined": 0,
    }
    assert not task_dir.exists()
    audit = TaskRepository(app.state.settings.db_path).list_audit(
        kind="task_filesystem_gc.manual_sweep",
    )
    assert [row["outcome"] for row in audit] == [
        "started",
        "succeeded",
        "started",
        "succeeded",
    ]
    assert audit[-1]["actor"] == admin["id"]
    assert audit[-1]["detail"] == {
        "limit": 1,
        "force": True,
        "report": forced.json(),
    }


def test_operations_task_filesystem_gc_rejects_unbounded_or_coerced_controls(
    tmp_path,
):
    app = create_app(tmp_path)
    admin_client = TestClient(app)
    _claim_role(app, admin_client, "admin")

    for payload in (
        {"limit": 0, "force": False},
        {"limit": 501, "force": False},
        {"limit": True, "force": False},
        {"limit": 1, "force": "true"},
        {"limit": 1, "force": False, "origin_task_id": "task"},
    ):
        response = admin_client.post(
            "/api/operations/task-filesystem-gc/sweep",
            json=payload,
        )
        assert response.status_code == 422, (payload, response.text)

    assert (
        admin_client.get(
            "/api/operations/task-filesystem-gc",
            params={"limit": 0},
        ).status_code
        == 422
    )
    assert (
        admin_client.get(
            "/api/operations/task-filesystem-gc",
            params={"limit": 5001},
        ).status_code
        == 422
    )


def test_operations_task_filesystem_gc_requeues_one_quarantine_as_admin(
    tmp_path,
):
    app = create_app(tmp_path)
    maker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, "maker")
    admin = _claim_role(app, admin_client, "admin")
    task_id = "quarantined-task"
    _enqueue_task_filesystem_gc_entry(
        app,
        target_type="task_dir",
        relative_path=task_id,
        state="quarantined",
        next_attempt_at="2026-08-01T20:05:00+08:00",
        origin_task_id=task_id,
    )
    payload = {"target_type": "task_dir", "relative_path": task_id}

    denied = maker_client.post(
        "/api/operations/task-filesystem-gc/requeue",
        json=payload,
    )
    requeued = admin_client.post(
        "/api/operations/task-filesystem-gc/requeue",
        json=payload,
    )
    repeated = admin_client.post(
        "/api/operations/task-filesystem-gc/requeue",
        json=payload,
    )

    assert denied.status_code == 403
    assert requeued.status_code == 200, requeued.text
    assert requeued.json() == {
        "target_type": "task_dir",
        "relative_path": task_id,
        "state": "pending",
    }
    assert repeated.status_code == 409, repeated.text
    status = admin_client.get("/api/operations/task-filesystem-gc").json()
    [entry] = status["entries"]
    assert entry["state"] == "pending"
    assert entry["attempt_count"] == 0
    assert entry["last_error_code"] is None
    assert entry["last_error_message"] is None

    audit = TaskRepository(app.state.settings.db_path).list_audit(
        kind="task_filesystem_gc.manual_requeue",
    )
    assert [row["outcome"] for row in audit] == [
        "started",
        "succeeded",
        "started",
        "rejected",
    ]
    assert audit[-1]["actor"] == admin["id"]


def test_operations_requeues_validation_batch_source_quarantine(tmp_path):
    app = create_app(tmp_path)
    admin_client = TestClient(app)
    _claim_role(app, admin_client, "admin")
    task_id = "purged-validation-batch"
    relative_path = "validation-batches/0123456789abcdef0123456789abcdef"
    _enqueue_task_filesystem_gc_entry(
        app,
        target_type="validation_batch_source_dir",
        relative_path=relative_path,
        state="quarantined",
        next_attempt_at="2026-08-01T20:05:00+08:00",
        origin_task_id=task_id,
    )

    response = admin_client.post(
        "/api/operations/task-filesystem-gc/requeue",
        json={
            "target_type": "validation_batch_source_dir",
            "relative_path": relative_path,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "target_type": "validation_batch_source_dir",
        "relative_path": relative_path,
        "state": "pending",
    }


def test_operations_task_filesystem_gc_requeue_is_strict_and_local(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    app = create_app(tmp_path)
    admin_client = TestClient(app)
    _claim_role(app, admin_client, "admin")

    for payload in (
        {"target_type": "unknown", "relative_path": "task"},
        {"target_type": "task_dir", "relative_path": "../task"},
        {"target_type": "task_dir", "relative_path": "task", "force": True},
    ):
        response = admin_client.post(
            "/api/operations/task-filesystem-gc/requeue",
            json=payload,
        )
        assert response.status_code == 422, (payload, response.text)

    remote = TestClient(app, client=("203.0.113.9", 5555))
    blocked = remote.post(
        "/api/operations/task-filesystem-gc/requeue",
        json={"target_type": "task_dir", "relative_path": "task"},
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == "unsafe API methods are limited to local clients"


def test_operations_task_filesystem_gc_controls_stay_local(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    app = create_app(tmp_path)
    remote = TestClient(app, client=("203.0.113.9", 5555))

    status = remote.get("/api/operations/task-filesystem-gc")
    sweep = remote.post(
        "/api/operations/task-filesystem-gc/sweep",
        json={"limit": 1, "force": False},
    )

    assert status.status_code == 403
    assert status.json()["detail"] == "this endpoint is limited to local clients"
    assert sweep.status_code == 403
    assert sweep.json()["detail"] == "unsafe API methods are limited to local clients"

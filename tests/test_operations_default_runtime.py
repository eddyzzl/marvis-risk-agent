"""Standard app assembly, real monitor worker, and delivered local inbox receipts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import threading
import time

from fastapi.testclient import TestClient
import pytest
import pandas as pd

from marvis.app import create_app
from marvis.operations.contracts import MonitoringOutcome
from marvis.operations.integration import OperationsRuntime
from marvis.operations.monitoring import read_monitoring_evidence
from marvis.operations.repository import StaleRunLeaseError
from marvis.settings import build_settings

from test_modeling_monitor import _train_lr_experiment
from test_modeling_pack import _runtime
from test_operations_api import _claim_role, _publish_payload, FIXED_NOW
from test_operations_integration import _schedule


def _wait(condition, timeout=35):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = condition()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("condition did not become true before deadline")


@pytest.fixture(scope="module")
def trained_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("operations-model")
    runner, _, registry, _, settings, task = _runtime(root)
    trained, frame = _train_lr_experiment(runner, registry, root, task)
    frame["observed_at"] = (FIXED_NOW - timedelta(seconds=90)).isoformat()
    path = root / "monitoring.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(
        path, task_id=task.id, role="monitoring_source"
    )
    return settings, task, dataset, trained.output["experiment_id"]


def _binding(task, dataset, target):
    return {
        "task_id": task.id,
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash,
        "time_col": "observed_at",
        "timezone": "Asia/Shanghai",
        "complete_through": FIXED_NOW.isoformat(),
        "target_id": target,
        "target_col": "y",
        "label_mode": "required",
        "label_maturity_seconds": 0,
    }


def _publish(client, task, dataset, target, *, schedule_id, changes=None):
    payload = _publish_payload(
        monitoring_ref="modeling.monitor_run", schedule_id=schedule_id
    )
    binding = _binding(task, dataset, target)
    binding.update(changes or {})
    payload["schedule"]["monitoring_binding"] = binding
    response = client.post("/api/operations/schedules", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_normal_app_lifecycle_runs_real_monitor_and_delivers_once_after_restart(
    trained_source,
):
    settings, task, dataset, target = trained_source
    # No injected executors, no manual tick, no direct monitoring tool call.
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    with TestClient(app) as client:
        _claim_role(app, client, "admin")
        capabilities = client.get("/api/operations/capabilities").json()
        assert capabilities["monitoring_refs"] == [
            "modeling.monitor_run",
            "strategy.run_strategy_monitoring",
        ]
        assert capabilities["runtime"]["alive"]
        record = _publish(client, task, dataset, target, schedule_id="default-monitor")
        schedule = app.state.operations_runtime.store.get_schedule(
            "default-monitor"
        ).contract
        key = schedule.calendar.period(0).key

        def ready():
            response = client.get("/api/operations/inbox").json()
            return next(
                (
                    n
                    for n in response["notifications"]
                    if n["schedule_id"] == "default-monitor" and n["period_key"] == key
                ),
                None,
            )

        receipt = _wait(ready)
        assert receipt["delivery_target"] == "local_application_inbox"
        assert receipt["delivered_at"] and receipt["read_at"] is None
        evidence = client.get(
            "/api/operations/schedules/default-monitor/evidence",
            params={"period_key": key},
        )
        assert evidence.status_code == 200, evidence.text
        evidence = evidence.json()
        assert evidence["tool_output"]["row_count"] == 400
        assert evidence["tool_output"]["artifact_id"]
        assert evidence["tool_output"]["checks"]
        assert evidence["schedule_hash"] == record["contract_hash"]
        assert evidence["coverage_assurance"] == "publisher_declared"
        assert evidence["source_content_hash"] == dataset.content_hash
        assert evidence["window_content_hash"]
        assert "dataset_id" not in json.dumps(receipt["payload"])
        # Acknowledging local delivery does not execute a production action.
        assert (
            client.post(
                f"/api/operations/inbox/{receipt['notification_id']}/acknowledge"
            ).json()["automatic_action_permitted"]
            is False
        )
        assert ready()["read_by"]
    assert not app.state.operations_runtime.loop.status()["alive"]
    restarted = create_app(settings, operations_clock=lambda: FIXED_NOW)
    with TestClient(restarted) as client:
        time.sleep(1.2)
        store = restarted.state.operations_runtime.store
        assert len(store.list_runs("default-monitor", key)) == 1
        items = restarted.state.operations_runtime.local_inbox.list()
        assert (
            len(
                [
                    item
                    for item in items
                    if item["notification_id"] == receipt["notification_id"]
                ]
            )
            == 1
        )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"complete_through": (FIXED_NOW - timedelta(seconds=90)).isoformat()},
            "window_incomplete",
        ),
        (
            {"complete_through": (FIXED_NOW + timedelta(seconds=1)).isoformat()},
            "future_source_watermark",
        ),
        ({"label_maturity_seconds": 120}, "labels_immature"),
        ({"target_col": "missing_label"}, "labels_incomplete"),
    ],
)
def test_default_monitor_rejects_unready_window(trained_source, changes, reason):
    settings, task, dataset, target = trained_source
    # Do not enter lifespan: explicitly exercise one tick, retaining real binding.
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    name = "not-ready-" + reason
    _publish(client, task, dataset, target, schedule_id=name, changes=changes)
    app.state.operations_runtime.tick(catch_up_budget=100, notification_budget=100)
    schedule = app.state.operations_runtime.store.get_schedule(name).contract
    outcome = app.state.operations_runtime.store.get_outcome(
        name, schedule.calendar.period(0).key
    )
    assert outcome.outcome.level == "not_available"
    evidence = read_monitoring_evidence(settings.workspace, outcome.outcome)
    assert evidence["reason_code"] == reason
    assert "tool_output" not in evidence


def test_old_watermark_never_reuses_prior_rows(trained_source):
    settings, task, dataset, target = trained_source
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    _publish(client, task, dataset, target, schedule_id="old-data")
    runtime = app.state.operations_runtime
    runtime.tick(catch_up_budget=100, notification_budget=100)
    runtime.tick(catch_up_budget=100, notification_budget=100)
    schedule = runtime.store.get_schedule("old-data").contract
    outcome = runtime.store.get_outcome("old-data", schedule.calendar.period(1).key)
    assert outcome.outcome.level == "not_available"
    assert (
        read_monitoring_evidence(settings.workspace, outcome.outcome)["reason_code"]
        == "no_new_data"
    )


def test_binding_is_typed_and_task_hash_authenticated(trained_source):
    settings, task, dataset, target = trained_source
    app = create_app(settings)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    for change in (
        {"dataset_content_hash": "0" * 64},
        {"task_id": "other-task"},
        {"timezone": "unknown"},
        {"complete_through": "2026-08-01T12:00:00"},
        {"label_mode": "not_applicable"},
        {"label_maturity_seconds": True},
        {"label_maturity_seconds": None},
    ):
        payload = _publish_payload(
            monitoring_ref="modeling.monitor_run", schedule_id="invalid-binding"
        )
        payload["schedule"]["monitoring_binding"] = (
            _binding(task, dataset, target) | change
        )
        response = client.post("/api/operations/schedules", json=payload)
        assert response.status_code == 422, response.text
    assert app.state.operations_runtime.store.get_schedule("invalid-binding") is None


def test_two_live_workers_renew_a_short_lease_and_commit_once(tmp_path):
    calls = []
    started = threading.Event()

    def monitor(request):
        calls.append(request)
        started.set()
        time.sleep(1.6)
        return MonitoringOutcome(
            level="green", evidence_ref="test:result", evidence_hash="a" * 64
        )

    settings = build_settings(tmp_path)
    one = OperationsRuntime(settings, executor_allowlist={"bound": monitor})
    two = OperationsRuntime(settings, executor_allowlist={"bound": monitor})
    now = datetime.now(UTC)
    schedule = replace(
        _schedule(monitoring_ref="bound"),
        lease_seconds=1,
        active_from=now - timedelta(minutes=1),
        calendar=replace(
            _schedule(monitoring_ref="bound").calendar,
            anchor_at=now - timedelta(minutes=1),
        ),
    )
    one.publish_schedule(schedule, expected_revision=0)
    thread = threading.Thread(target=lambda: one.tick(catch_up_budget=1))
    thread.start()
    assert started.wait(5)
    time.sleep(1.2)
    report = two.tick(catch_up_budget=1)
    thread.join(5)
    assert report.attempted == 0
    assert len(calls) == 1
    assert (
        len(one.store.list_runs(schedule.schedule_id, schedule.calendar.period(0).key))
        == 1
    )
    assert one.store.get_outcome(schedule.schedule_id, schedule.calendar.period(0).key)


def test_expired_lease_cannot_be_renewed(tmp_path):
    now = [FIXED_NOW]
    runtime = OperationsRuntime(
        build_settings(tmp_path),
        executor_allowlist={"bound": lambda r: None},
        clock=lambda: now[0],
    )
    schedule = _schedule(monitoring_ref="bound")
    runtime.publish_schedule(schedule, expected_revision=0)
    claim = runtime.store.claim_period(
        schedule, schedule.calendar.period(0), owner_id="crashed"
    )
    now[0] += timedelta(seconds=31)
    with pytest.raises(StaleRunLeaseError):
        runtime.store.renew_run_lease(claim, lease_seconds=30)


def test_notification_generation_is_not_local_delivery_and_retry_is_idempotent(
    tmp_path, monkeypatch
):
    from marvis.operations.integration import build_operations_runtime

    runtime = build_operations_runtime(
        build_settings(tmp_path),
        executor_allowlist={
            "bound": lambda r: MonitoringOutcome(
                level="red", evidence_ref="test:result", evidence_hash="b" * 64
            )
        },
        clock=lambda: FIXED_NOW,
    )
    schedule = _schedule(monitoring_ref="bound")
    runtime.publish_schedule(schedule, expected_revision=0)
    runtime.tick(catch_up_budget=1, notification_budget=0)
    assert runtime.local_inbox.list() == []
    key = schedule.calendar.period(0).key
    assert (
        runtime.store.list_notifications(schedule.schedule_id, key)[0].status
        == "pending"
    )
    adapter = runtime.local_inbox
    original = adapter.send

    def fail(notification):
        raise OSError("local inbox temporarily unavailable")

    monkeypatch.setattr(adapter, "send", fail)
    report = runtime.tick(catch_up_budget=1, notification_budget=1)
    assert report.notifications_failed == 1
    assert adapter.list() == []
    monkeypatch.setattr(adapter, "send", original)
    # Advance the same scheduler/store clock after the recorded backoff.
    runtime.store._clock = lambda: FIXED_NOW + timedelta(seconds=6)
    report = runtime.tick(catch_up_budget=1, notification_budget=1)
    assert report.notifications_sent == 1
    item = adapter.list()[0]
    record = runtime.store.list_notifications(schedule.schedule_id, item["period_key"])[
        0
    ]
    assert record.status == "sent"
    original(record.notification)
    assert (
        len(
            [
                n
                for n in adapter.list()
                if n["notification_id"] == item["notification_id"]
            ]
        )
        == 1
    )
    with pytest.raises(ValueError):
        original(replace(record.notification, level="green"))


def test_default_strategy_binding_runs_real_registered_tool(tmp_path):
    from test_strategy_typed_monitoring import _adopt_typed_strategy, _register

    settings, task, registry, strategy, _, _ = _adopt_typed_strategy(
        tmp_path, "approval"
    )
    # Fixture adoption is prepared by existing deterministic helper; this test
    # verifies the unmodified production monitoring Tool manifest and worker.
    frame = pd.DataFrame(
        {
            "x": [0, 0, 1, 1],
            "bad": [0, 0, 1, 1],
            "observed_at": [(FIXED_NOW - timedelta(seconds=90)).isoformat()] * 4,
        }
    )
    dataset = _register(
        registry, tmp_path, task_id=task.id, name="monitor-source", frame=frame
    )
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    with TestClient(app) as client:
        _claim_role(app, client, "admin")
        payload = _publish_payload(
            monitoring_ref="strategy.run_strategy_monitoring",
            schedule_id="strategy-default",
        )
        payload["schedule"]["monitoring_binding"] = _binding(
            task, dataset, strategy.id
        ) | {"target_col": "bad"}
        response = client.post("/api/operations/schedules", json=payload)
        assert response.status_code == 201, response.text
        runtime = app.state.operations_runtime
        schedule = runtime.store.get_schedule("strategy-default").contract
        key = schedule.calendar.period(0).key
        outcome = _wait(lambda: runtime.store.get_outcome("strategy-default", key))
        evidence = read_monitoring_evidence(settings.workspace, outcome.outcome)
        assert evidence["tool_output"]["monitoring_run_id"]
        assert evidence["tool_output"]["strategy_id"] == strategy.id
        assert evidence["tool_output"]["plan_source"] == "ledger"
        assert evidence["manifest_hash"]


def test_failed_source_retries_and_never_generates_success_notification(trained_source):
    settings, task, dataset, target = trained_source
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    _publish(
        client,
        task,
        dataset,
        target,
        schedule_id="source-error",
        changes={"time_col": "absent_time"},
    )
    runtime = app.state.operations_runtime
    for _ in range(2):
        runtime.tick(catch_up_budget=100, notification_budget=100)
    schedule = runtime.store.get_schedule("source-error").contract
    key = schedule.calendar.period(0).key
    assert runtime.store.get_period("source-error", key).state == "terminal_failed"
    assert runtime.store.get_outcome("source-error", key) is None
    assert runtime.store.list_notifications("source-error", key) == ()
    events = runtime.store.list_events("source-error", key)
    assert [
        e.payload["error_code"] for e in events if e.event_type == "run_failed"
    ] == ["monitoring_source_error", "monitoring_source_error"]


def test_shutdown_stops_future_claims_and_disabled_schedule_never_runs(tmp_path):
    from marvis.operations.integration import build_operations_runtime

    calls = []
    runtime = build_operations_runtime(
        build_settings(tmp_path),
        executor_allowlist={"bound": lambda r: calls.append(r)},
        clock=lambda: FIXED_NOW,
    )
    runtime.publish_schedule(
        replace(_schedule(monitoring_ref="bound"), enabled=False), expected_revision=0
    )
    runtime.loop.start()
    _wait(lambda: runtime.loop.status()["last_finished_at"])
    runtime.loop.stop()
    assert runtime.loop.status()["alive"] is False
    assert calls == []
    runtime.publish_schedule(
        replace(_schedule(monitoring_ref="bound"), revision=2), expected_revision=1
    )
    assert runtime.tick(catch_up_budget=1).attempted == 0


def test_explicit_recheck_keeps_original_evidence_and_runs_one_window(trained_source):
    settings, task, dataset, target = trained_source
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    _publish(
        client,
        task,
        dataset,
        target,
        schedule_id="recheck-origin",
        changes={
            "complete_through": (FIXED_NOW - timedelta(seconds=90)).isoformat(),
        },
    )
    runtime = app.state.operations_runtime
    runtime.tick(catch_up_budget=100, notification_budget=100)
    source = runtime.store.get_schedule("recheck-origin").contract
    key = source.calendar.period(0).key
    original = runtime.store.get_outcome(source.schedule_id, key)
    assert original.outcome.level == "not_available"
    request = {
        "period_key": key,
        "expected_revision": 1,
        "idempotency_key": "one-review",
        "monitoring_binding": _binding(task, dataset, target),
    }
    url = "/api/operations/schedules/recheck-origin/rechecks"
    result = client.post(url, json=request)
    assert result.status_code == 201, result.text
    recheck = result.json()["contract"]
    assert recheck["recheck_of"]["period_key"] == key
    assert recheck["active_until"]
    repeated = client.post(url, json=request)
    assert repeated.status_code == 201
    assert repeated.json() == result.json()
    changed = request | {
        "monitoring_binding": request["monitoring_binding"] | {"time_col": "wrong"}
    }
    assert client.post(url, json=changed).status_code == 409
    assert client.post(url, json=request | {"expected_revision": 99}).status_code == 409
    runtime.tick(catch_up_budget=100, notification_budget=100)
    checked = runtime.store.get_outcome(recheck["schedule_id"], key)
    assert checked is not None
    assert "tool_output" in read_monitoring_evidence(
        settings.workspace, checked.outcome
    )
    runtime.tick(catch_up_budget=100, notification_budget=100)
    assert len(runtime.store.list_periods(recheck["schedule_id"])) == 1
    assert runtime.store.get_outcome(source.schedule_id, key) == original
    diagnostics = client.get("/api/operations/schedules/recheck-origin/periods").json()[
        "periods"
    ]
    assert any(
        p["diagnostic"]["code"] == "window_incomplete"
        and p["diagnostic"]["next_action"]["code"] == "publish_source"
        for p in diagnostics
    )


def test_inbox_recovers_send_before_outbox_commit_without_duplicate_delivery(
    tmp_path, monkeypatch
):
    from marvis.operations.integration import build_operations_runtime
    from marvis.operations.repository import StaleNotificationLeaseError

    current = [FIXED_NOW]
    runtime = build_operations_runtime(
        build_settings(tmp_path),
        executor_allowlist={
            "bound": lambda r: MonitoringOutcome(
                level="red", evidence_ref="test:result", evidence_hash="c" * 64
            )
        },
        clock=lambda: current[0],
    )
    schedule = _schedule(monitoring_ref="bound")
    schedule = replace(schedule, active_until=schedule.calendar.period(0).ends_at)
    runtime.publish_schedule(schedule, expected_revision=0)
    original = runtime.store.complete_notification

    def crash_after_send(claim):
        raise StaleNotificationLeaseError("simulated process loss after inbox commit")

    monkeypatch.setattr(runtime.store, "complete_notification", crash_after_send)
    runtime.tick(catch_up_budget=1, notification_budget=1)
    assert len(runtime.local_inbox.list()) == 1
    key = schedule.calendar.period(0).key
    assert (
        runtime.store.list_notifications(schedule.schedule_id, key)[0].status
        == "sending"
    )
    monkeypatch.setattr(runtime.store, "complete_notification", original)
    current[0] += timedelta(seconds=31)
    runtime.tick(catch_up_budget=1, notification_budget=1)
    current[0] += timedelta(seconds=6)
    runtime.tick(catch_up_budget=1, notification_budget=1)
    assert len(runtime.local_inbox.list()) == 1
    assert (
        runtime.store.list_notifications(schedule.schedule_id, key)[0].status == "sent"
    )


def test_stale_schedule_snapshot_cannot_claim_after_disable_or_republication(tmp_path):
    runtime = OperationsRuntime(
        build_settings(tmp_path),
        executor_allowlist={"bound": lambda r: None},
        clock=lambda: FIXED_NOW,
    )
    original = _schedule(monitoring_ref="bound")
    runtime.publish_schedule(original, expected_revision=0)
    runtime.publish_schedule(
        replace(original, revision=2, enabled=False), expected_revision=1
    )
    period = original.calendar.period(0)
    assert runtime.store.claim_period(original, period, owner_id="stale-worker") is None
    replacement = replace(original, revision=3)
    runtime.publish_schedule(replacement, expected_revision=2)
    assert runtime.store.claim_period(original, period, owner_id="stale-worker") is None
    assert runtime.store.claim_period(replacement, period, owner_id="current-worker")


def test_one_window_schedule_rejects_direct_claim_outside_window(tmp_path):
    runtime = OperationsRuntime(
        build_settings(tmp_path),
        executor_allowlist={"bound": lambda r: None},
        clock=lambda: FIXED_NOW,
    )
    original = _schedule(monitoring_ref="bound")
    bounded = replace(original, active_until=original.calendar.period(0).ends_at)
    runtime.publish_schedule(bounded, expected_revision=0)
    with pytest.raises(ValueError, match="after schedule deactivation"):
        runtime.store.claim_period(
            bounded, bounded.calendar.period(1), owner_id="worker"
        )

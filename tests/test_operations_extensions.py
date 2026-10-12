"""Actual HTTP confirmation and scheduler → native portfolio Tools."""

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.operations.monitoring import read_monitoring_evidence
from marvis.operations.month_calendar import CalendarMonthCalendar

from test_analysis_pack import _runtime, _four_loan_frame, _register
from test_operations_api import _claim_role

NOW = datetime(2025, 4, 1, tzinfo=UTC)


def portfolio_runtime(tmp_path):
    runner, plugins, registry, task = _runtime(tmp_path)
    frame = _four_loan_frame()
    frame["observed_at"] = frame["snapshot_month"] + "-01T00:00:00+00:00"
    frame["segment"] = ["north" if x in ("A", "B") else "south" for x in frame.loan_id]
    dataset = _register(registry, tmp_path, task.id, frame, "portfolio-source")
    settings = (
        runner._workspace if hasattr(runner, "_workspace") else tmp_path / "workspace"
    )
    app = create_app(settings, operations_clock=lambda: NOW)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    source = dict(
        task_id=task.id,
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        time_col="observed_at",
        timezone="UTC",
        complete_through=NOW.isoformat(),
        target_id="portfolio",
        label_mode="not_applicable",
        label_maturity_seconds=None,
    )
    body = dict(
        schedule_id="portfolio-monthly",
        revision=1,
        monitoring_ref="analysis.portfolio_run",
        active_from="2025-03-01T00:00:00Z",
        calendar=dict(
            anchor_at="2025-03-01T00:00:00Z", interval_months=1, timezone="UTC"
        ),
        catch_up_budget=2,
        lease_seconds=600,
        retry_policy=dict(
            max_attempts=1,
            initial_backoff_seconds=0.0,
            multiplier=1.0,
            max_backoff_seconds=0.0,
        ),
        portfolio_binding=dict(
            source=source,
            id_col="loan_id",
            bucket_col="bucket",
            balance_col="balance",
            segment_col="segment",
            states=["current", "M1", "bad"],
            loss_state="bad",
            lgd=0.5,
            horizon_months=1,
            lookback_seconds=62 * 86400,
            generate_report=True,
        ),
    )
    return app, client, registry, dataset, task, body


def test_actual_monthly_portfolio_proposal_confirmation_tick_and_notification(tmp_path):
    app, client, registry, dataset, task, body = portfolio_runtime(tmp_path)
    response = client.post(
        "/api/operations/proposals",
        json=dict(
            expected_revision=0,
            instruction="每 1 个月，从 2025-03-01T00:00:00Z 开始分析组合。",
            schedule=body,
        ),
    )
    assert response.status_code == 201, response.text
    proposal = response.json()
    assert proposal["state"] == "needs_confirmation"
    runtime = app.state.operations_runtime
    assert runtime.store.get_schedule(body["schedule_id"]) is None
    response = client.post(
        f"/api/operations/proposals/{proposal['proposal_id']}/confirm",
        json={"expected_contract_hash": proposal["contract_hash"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "published"
    replay = client.post(
        f"/api/operations/proposals/{proposal['proposal_id']}/confirm",
        json={"expected_contract_hash": proposal["contract_hash"]},
    )
    assert replay.json()["replayed"] is True
    runtime.tick(catch_up_budget=2, notification_budget=100)
    schedule = runtime.store.get_schedule(body["schedule_id"]).contract
    key = schedule.calendar.period(0).key
    record = runtime.store.get_outcome(body["schedule_id"], key)
    assert record is not None, [
        x for x in runtime.store.list_runs(body["schedule_id"], key)
    ]
    evidence = read_monitoring_evidence(app.state.settings.workspace, record.outcome)
    assert evidence["window_row_count"] == 11
    assert evidence["current_window_row_count"] == 3
    assert set(evidence["tool_results"]) == {
        "flow_rate",
        "bucket_migration",
        "segment_profile",
        "expected_loss_estimate",
        "portfolio_report",
    }
    flow = evidence["tool_results"]["flow_rate"]["output"]
    assert flow["months"] == ["2025-01", "2025-02"]
    feb = next(x for x in flow["matrix_by_month"] if x["month"] == "2025-02")
    assert feb["from_to_matrix"][0] == pytest.approx([0.5, 0.5, 0, 0])
    assert evidence["source_content_hash"] == dataset.content_hash
    assert evidence["coverage_assurance"] == "publisher_declared"
    assert record.outcome.level == "red"  # sparse 4-loan native warning
    notifications = runtime.local_inbox.list()
    assert (
        len(notifications) == 1
        and notifications[0]["delivery_target"] == "local_application_inbox"
    )
    runtime.tick(catch_up_budget=2, notification_budget=100)
    assert len(runtime.store.list_runs(body["schedule_id"], key)) == 1
    assert len(runtime.local_inbox.list()) == 1
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_schedule_confirmations"
            ).fetchone()[0]
            == 1
        )
        assert all(
            item["invocation"] and len(item["output_hash"]) == 64
            for item in evidence["tool_results"].values()
        )


def test_unknown_nl_clarifies_without_publish_and_credential_instruction_is_rejected(
    tmp_path,
):
    app, client, _, _, _, body = portfolio_runtime(tmp_path)
    r = client.post(
        "/api/operations/proposals",
        json=dict(expected_revision=0, instruction="有空时帮我分析", schedule=body),
    )
    assert r.status_code == 201 and r.json()["state"] == "clarification"
    r = client.post(
        "/api/operations/proposals",
        json=dict(
            expected_revision=0,
            instruction="API_KEY=private-value every 1 day 2025-03-01T00:00:00Z",
            schedule=body,
        ),
    )
    assert r.status_code == 422, r.text
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_schedule_proposals"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT count(*) FROM operations_schedules").fetchone()[0] == 0
        )


def test_proposal_source_and_head_cas_no_partial_confirmation(tmp_path):
    app, client, registry, dataset, _, body = portfolio_runtime(tmp_path)
    p = client.post(
        "/api/operations/proposals", json=dict(expected_revision=0, schedule=body)
    ).json()
    dataset_path = registry.authenticate_dataset_binding(
        dataset.id,
        expected_task_id=dataset.task_id,
        expected_content_hash=dataset.content_hash,
    ).path
    dataset_path.chmod(0o600)
    dataset_path.write_bytes(dataset_path.read_bytes() + b"drift")
    r = client.post(
        f"/api/operations/proposals/{p['proposal_id']}/confirm",
        json={"expected_contract_hash": p["contract_hash"]},
    )
    assert r.status_code in (409, 422), r.text
    assert app.state.operations_runtime.store.get_schedule(body["schedule_id"]) is None
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_schedule_confirmations"
            ).fetchone()[0]
            == 0
        )


def test_true_calendar_months_preserve_dst_boundary_and_old_wire_unchanged():
    c = CalendarMonthCalendar(
        datetime(
            2026, 3, 1, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York")
        ),
        1,
        "America/New_York",
    )
    p = c.period(0)
    assert (p.ends_at - p.starts_at).total_seconds() == 31 * 86400 - 3600
    assert CalendarMonthCalendar.from_dict(c.to_dict()).period(0) == p


def test_two_proposals_exact_actor_head_expiry_and_read_only_confirmation_retry(
    tmp_path, monkeypatch
):
    from copy import deepcopy

    app, client, _, _, _, body = portfolio_runtime(tmp_path)
    request = dict(expected_revision=0, schedule=body)
    p1 = client.post("/api/operations/proposals", json=request).json()
    p2 = client.post("/api/operations/proposals", json=request).json()

    def confirm(p):
        return client.post(
            f"/api/operations/proposals/{p['proposal_id']}/confirm",
            json={"expected_contract_hash": p["contract_hash"]},
        )

    assert confirm(p1).status_code == 200
    assert confirm(p2).status_code == 409
    other = TestClient(app)
    _claim_role(app, other, "admin")
    assert (
        other.post(
            f"/api/operations/proposals/{p1['proposal_id']}/confirm",
            json={"expected_contract_hash": p1["contract_hash"]},
        ).status_code
        == 409
    )
    body = deepcopy(body)
    body["revision"] = 2
    p3 = client.post(
        "/api/operations/proposals", json={"expected_revision": 1, "schedule": body}
    ).json()
    runtime = app.state.operations_runtime
    monkeypatch.setattr(runtime.store, "_clock", lambda: NOW + timedelta(minutes=31))
    assert confirm(p3).status_code == 409
    assert runtime.store.get_schedule(body["schedule_id"]).contract.revision == 1
    # Successful retries read authenticated immutable confirmation only. They
    # do not reread a source, reprepare native execution, or republish schedule.
    import marvis.operations.schedule_proposals as service

    monkeypatch.setattr(
        service,
        "binding_identity",
        lambda *a, **kw: pytest.fail("confirmation replay rechecks source"),
    )
    assert confirm(p1).json()["replayed"] is True
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_schedule_confirmations"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "watermark", [NOW - timedelta(seconds=1), NOW + timedelta(seconds=1)]
)
def test_portfolio_unready_source_never_invokes_analysis_tools(tmp_path, watermark):
    app, client, _, _, _, body = portfolio_runtime(tmp_path)
    body["portfolio_binding"]["source"]["complete_through"] = watermark.isoformat()
    p = client.post(
        "/api/operations/proposals", json={"expected_revision": 0, "schedule": body}
    ).json()
    r = client.post(
        f"/api/operations/proposals/{p['proposal_id']}/confirm",
        json={"expected_contract_hash": p["contract_hash"]},
    )
    assert r.status_code == 200, r.text
    runtime = app.state.operations_runtime
    runtime.tick(catch_up_budget=1, notification_budget=100)
    key = (
        runtime.store.get_schedule(body["schedule_id"]).contract.calendar.period(0).key
    )
    outcome = runtime.store.get_outcome(body["schedule_id"], key)
    assert outcome.outcome.level == "not_available"
    e = read_monitoring_evidence(app.state.settings.workspace, outcome.outcome)
    assert e["reason_code"] == (
        "window_incomplete" if watermark < NOW else "future_source_watermark"
    )
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_native_invocations"
            ).fetchone()[0]
            == 0
        )


def test_original_trusted_local_native_schedule_without_actor_remains_executable(
    tmp_path,
):
    from marvis.operations.http_schemas import SchedulePayload
    from marvis.operations.router import _schedule_contract

    app, _, _, _, _, body = portfolio_runtime(tmp_path)
    legacy = _schedule_contract(SchedulePayload.model_validate(body))
    assert (
        legacy.execution_actor_id is None
        and "execution_actor_id" not in legacy.to_dict()
    )
    runtime = app.state.operations_runtime
    runtime.publish_schedule(legacy, expected_revision=0)
    runtime.tick(catch_up_budget=1, notification_budget=100)
    outcome = runtime.store.get_outcome(
        legacy.schedule_id, legacy.calendar.period(0).key
    )
    assert outcome is not None and outcome.outcome.level == "red"
    with connect(app.state.settings.db_path) as conn:
        rows = conn.execute(
            "SELECT actor_id FROM operations_native_invocations"
        ).fetchall()
        assert len(rows) == 5 and all(
            row["actor_id"] == "trusted-local-legacy-scheduler" for row in rows
        )

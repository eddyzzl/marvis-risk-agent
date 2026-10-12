"""Real signed identity sessions bound to actual schedule/task/source evidence."""

import pytest

from marvis.db_schema import connect
from marvis.institution.policy import claim_resource
from marvis.operations.monitoring import read_monitoring_evidence

from test_institution_identity import institution as identity_fixture, _login
from test_operations_extensions import portfolio_runtime


def test_institution_exact_task_visibility_human_confirm_and_current_revocation(
    tmp_path,
):
    old_app, _, registry, dataset, task, body = portfolio_runtime(tmp_path)
    fixture = identity_fixture.__wrapped__(tmp_path)
    app, _, subjects, _ = fixture
    claim_resource(
        app.state.settings.db_path,
        subjects["admin"]["id"],
        "task",
        task.id,
        subjects["alice"]["id"],
        0,
    )
    alice, bob, admin, service = (
        _login(fixture, name) for name in ("alice", "bob", "admin", "automation")
    )
    response = alice.post(
        "/api/operations/proposals", json={"expected_revision": 0, "schedule": body}
    )
    assert response.status_code == 201, response.text
    p = response.json()
    assert (
        service.post(
            f"/api/operations/proposals/{p['proposal_id']}/confirm",
            json={"expected_contract_hash": p["contract_hash"]},
        ).status_code
        == 403
    )
    assert (
        bob.post(
            f"/api/operations/proposals/{p['proposal_id']}/confirm",
            json={"expected_contract_hash": p["contract_hash"]},
        ).status_code
        == 403
    )
    c = alice.post(
        f"/api/operations/proposals/{p['proposal_id']}/confirm",
        json={"expected_contract_hash": p["contract_hash"]},
    )
    assert c.status_code == 200, c.text
    for other in (bob, admin):
        assert other.get("/api/operations/schedules").json()["schedules"] == []
        assert (
            other.get("/api/operations/schedules/portfolio-monthly").status_code == 403
        )
        assert (
            other.get(
                "/api/operations/extensions/sources", params={"task_id": task.id}
            ).status_code
            == 403
        )
    assert len(alice.get("/api/operations/schedules").json()["schedules"]) == 1
    # Native background invocation must use the persisted human subject and the
    # exact leased run rather than a request/thread's incidental session.
    runtime = app.state.operations_runtime
    runtime.tick(catch_up_budget=1, notification_budget=100)
    contract = runtime.store.get_schedule("portfolio-monthly").contract
    key = contract.calendar.period(0).key
    outcome = runtime.store.get_outcome("portfolio-monthly", key)
    assert outcome is not None, list(
        runtime.store.list_events("portfolio-monthly", key)
    )
    evidence = read_monitoring_evidence(app.state.settings.workspace, outcome.outcome)
    assert evidence["tool_results"]["flow_rate"]["output"]["months"] == [
        "2025-01",
        "2025-02",
    ]
    assert len(alice.get("/api/operations/inbox").json()["notifications"]) == 1
    for other in (bob, admin):
        assert other.get("/api/operations/inbox").json()["notifications"] == []
        assert (
            other.get(
                "/api/operations/schedules/portfolio-monthly/evidence",
                params={"period_key": key},
            ).status_code
            == 403
        )
    with connect(app.state.settings.db_path) as conn:
        rows = conn.execute("SELECT * FROM operations_native_invocations").fetchall()
        assert len(rows) == 5 and all(
            r["actor_id"] == subjects["alice"]["id"]
            and r["task_id"] == task.id
            and r["dispatch_started_at"]
            for r in rows
        )
        with pytest.raises(Exception):
            conn.execute(
                "UPDATE operations_native_invocations SET inputs_hash=?", ("0" * 64,)
            )
    # Revoking the scheduled subject prevents fresh native reads and calls.
    revoked = admin.post(
        "/api/institution/revoke",
        json={"kind": "subject", "resource_id": subjects["alice"]["id"]},
    )
    assert revoked.status_code == 200, revoked.text
    runtime.tick(catch_up_budget=1, notification_budget=100)
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_native_invocations"
            ).fetchone()[0]
            == 5
        )
    assert alice.get("/api/operations/schedules").status_code == 401

"""Red scheduled native monitor → human approval → one native disposition."""

from datetime import timedelta

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.operations.monitoring import read_monitoring_evidence

from test_operations_api import FIXED_NOW, _claim_role, _publish_payload
from test_strategy_typed_monitoring import _adopt_typed_strategy, _register


def red_carrier(tmp_path):
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "approval"
    )
    frame = pd.DataFrame(
        {
            "x": [1, 1, 1, 1],
            "bad": [1, 1, 1, 1],
            "ead": [1000.0] * 4,
            "pd": [0.8] * 4,
            "observed_at": [(FIXED_NOW - timedelta(seconds=90)).isoformat()] * 4,
        }
    )
    dataset = _register(
        registry, tmp_path, task_id=task.id, name="red-monitor-source", frame=frame
    )
    app = create_app(settings, operations_clock=lambda: FIXED_NOW)
    client = TestClient(app)
    principal = _claim_role(app, client, "admin")
    body = _publish_payload(
        monitoring_ref="strategy.run_strategy_monitoring", schedule_id="scheduled-red"
    )
    body["schedule"]["retry_policy"]["max_attempts"] = 1
    body["schedule"]["monitoring_binding"] = dict(
        task_id=task.id,
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        time_col="observed_at",
        timezone="UTC",
        complete_through=FIXED_NOW.isoformat(),
        target_id=strategy.id,
        target_col="bad",
        label_mode="required",
        label_maturity_seconds=0,
    )
    r = client.post("/api/operations/schedules", json=body)
    assert r.status_code == 201, r.text
    runtime = app.state.operations_runtime
    runtime.tick(catch_up_budget=1, notification_budget=100)
    key = runtime.store.get_schedule("scheduled-red").contract.calendar.period(0).key
    outcome = runtime.store.get_outcome("scheduled-red", key)
    assert outcome is not None, list(runtime.store.list_runs("scheduled-red", key))
    evidence = read_monitoring_evidence(settings.workspace, outcome.outcome)
    assert outcome.outcome.level == "red", evidence
    out = evidence["tool_output"]
    payload = dict(
        period_key=key,
        expected_outcome_hash=outcome.outcome_hash,
        idempotency_key="human-one",
        expected_plan_id=out["monitoring_plan_id"],
        expected_plan_revision=out["monitoring_plan_revision"],
        expected_plan_hash=out["monitoring_plan_hash"],
        disposition="observe",
        reason="已核对本期原生证据，持续观察。",
    )
    return app, client, principal, task, strategy, evidence, payload


@pytest.mark.parametrize("action", ["observe", "new_version", "adjust_threshold"])
def test_native_red_disposition_exact_cas_human_governance_and_zero_retry(
    tmp_path, action, monkeypatch
):
    app, client, principal, task, strategy, evidence, payload = red_carrier(tmp_path)
    payload["disposition"] = action
    if action == "adjust_threshold":
        from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository

        plan = (
            StrategyMonitoringRepository(app.state.settings.db_path)
            .get_plan(payload["expected_plan_id"])
            .plan
        )
        check_id = next(
            key for key, v in plan.thresholds.items() if v["direction"] == "min"
        )
        payload["threshold_patch"] = {check_id: {"warn": 0.0, "fail": 0.0}}
    r = client.post(
        "/api/operations/schedules/scheduled-red/dispositions", json=payload
    )
    assert r.status_code == 200, r.text
    value = r.json()
    assert (
        value["state"] == "completed" and value["automatic_action_permitted"] is False
    )
    assert (
        value["native_result"]["status"]
        == {
            "observe": "observed",
            "new_version": "new_version_created",
            "adjust_threshold": "threshold_adjusted",
        }[action]
    )
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit WHERE kind='strategy.monitoring.disposition'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_disposition_claims WHERE state='completed'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM decision_records WHERE plan_id=?",
                (value["execution_plan_id"],),
            ).fetchone()[0]
            == 1
        )
    monkeypatch.setattr(
        app.state.plan_executor,
        "run",
        lambda *a, **kw: pytest.fail("committed disposition executed twice"),
    )
    r = client.post(
        "/api/operations/schedules/scheduled-red/dispositions", json=payload
    )
    assert r.status_code == 200 and r.json()["replayed"] is True, r.text
    other = {**payload, "idempotency_key": "second-one"}
    assert (
        client.post(
            "/api/operations/schedules/scheduled-red/dispositions", json=other
        ).status_code
        == 409
    )


@pytest.mark.parametrize("action", ["observe", "new_version", "adjust_threshold"])
def test_native_commit_lost_response_reconciles_exact_receipt_without_invoke(
    tmp_path, monkeypatch, action
):
    app, client, principal, _, _, _, payload = red_carrier(tmp_path)
    import marvis.operations.dispositions as service

    payload["disposition"] = action
    if action == "adjust_threshold":
        from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository

        plan = StrategyMonitoringRepository(app.state.settings.db_path).get_plan(
            payload["expected_plan_id"]
        ).plan
        check_id = next(
            key for key, spec in plan.thresholds.items() if spec["direction"] == "min"
        )
        payload["threshold_patch"] = {check_id: {"warn": 0.0, "fail": 0.0}}

    original = service._complete
    monkeypatch.setattr(
        service,
        "_complete",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("response lost after native commit")
        ),
    )
    r = client.post(
        "/api/operations/schedules/scheduled-red/dispositions", json=payload
    )
    assert r.status_code == 409, r.text
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit WHERE kind='strategy.monitoring.disposition'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT state FROM operations_disposition_claims").fetchone()[
                0
            ]
            == "claimed"
        )
    monkeypatch.setattr(service, "_complete", original)
    monkeypatch.setattr(
        app.state.plan_executor,
        "run",
        lambda *a, **kw: pytest.fail("recovery invoked native again"),
    )
    assert (
        client.post(
            "/api/operations/schedules/scheduled-red/dispositions", json=payload
        ).status_code
        == 409
    )
    r = client.post("/api/operations/dispositions/human-one/reconcile")
    assert r.status_code == 200 and r.json()["state"] == "completed", r.text
    assert r.json()["native_receipt"]["reason"] == payload["reason"]
    assert r.json()["native_result"]["disposition"] == action
    assert r.json()["native_step_run_id"]
    assert r.json()["native_output_ref"].startswith(
        "metrics:" + r.json()["execution_plan_id"] + "_step:v"
    )
    assert (
        client.post("/api/operations/dispositions/human-one/reconcile").json()[
            "replayed"
        ]
        is True
    )


def test_unexecuted_claim_and_bad_cas_never_publish_native_effect(
    tmp_path, monkeypatch
):
    app, client, _, _, _, _, payload = red_carrier(tmp_path)
    bad = {**payload, "expected_plan_hash": "0" * 64}
    assert (
        client.post(
            "/api/operations/schedules/scheduled-red/dispositions", json=bad
        ).status_code
        == 422
    )
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_disposition_claims"
            ).fetchone()[0]
            == 0
        )
    monkeypatch.setattr(
        app.state.plan_executor,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("process failed before tool")
        ),
    )
    assert (
        client.post(
            "/api/operations/schedules/scheduled-red/dispositions", json=payload
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/operations/schedules/scheduled-red/dispositions", json=payload
        ).status_code
        == 409
    )
    r = client.post("/api/operations/dispositions/human-one/reconcile")
    assert r.status_code == 200 and r.json()["state"] == "consumed_unresolved", r.text
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit WHERE kind='strategy.monitoring.disposition'"
            ).fetchone()[0]
            == 0
        )


def test_other_native_plan_same_receipt_cannot_complete_unexecuted_claim(
    tmp_path, monkeypatch
):
    """A real independent approval may not prove this consumed request ran."""
    import json
    from dataclasses import replace

    from marvis.orchestrator.contracts import Plan, PlanStatus, StepStatus

    app, client, principal, task, _, _, payload = red_carrier(tmp_path)
    executor = app.state.plan_executor
    original_run = executor.run
    monkeypatch.setattr(
        executor,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("original request failed before invocation")
        ),
    )
    response = client.post(
        "/api/operations/schedules/scheduled-red/dispositions", json=payload
    )
    assert response.status_code == 409, response.text
    with connect(app.state.settings.db_path) as conn:
        claim = conn.execute(
            "SELECT request_json FROM operations_disposition_claims"
        ).fetchone()
        request = json.loads(claim["request_json"])
    original = app.state.plan_repo.load_plan(request["execution_plan_id"])
    other_id = "independent-human-disposition"
    other_step = replace(
        original.steps[0],
        id=other_id + "_step",
        plan_id=other_id,
        status=StepStatus.AWAITING_CONFIRM,
    )
    other = Plan(
        other_id, task.id, "独立人工处置", "builtin", "test-native-disposition",
        [other_step], 0, status=PlanStatus.AWAITING_CONFIRM,
    )
    app.state.plan_repo.create_plan(other)
    app.state.governance_service.authorize_step(
        plan_id=other.id,
        step_id=other_step.id,
        principal=app.state.governance_repo.get_local_principal(principal["id"]),
        reason=payload["reason"],
        expected_plan_revision=0,
    )
    monkeypatch.setattr(executor, "run", original_run)
    assert executor.run(other.id).status == PlanStatus.DONE
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute(
            "SELECT count(*) FROM audit WHERE kind='strategy.monitoring.disposition'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM plan_step_runs WHERE plan_id=?",
            (original.id,),
        ).fetchone()[0] == 0
    monkeypatch.setattr(
        executor, "run", lambda *a, **kw: pytest.fail("recovery invoked again")
    )
    response = client.post("/api/operations/dispositions/human-one/reconcile")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "consumed_unresolved", response.text
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute(
            "SELECT state FROM operations_disposition_claims"
        ).fetchone()[0] == "claimed"

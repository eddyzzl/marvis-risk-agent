"""Real HTTP, governance decision, isolated ToolRunner and registered-file path."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import json

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.domain import TASK_TYPE_MODELING, TaskCreate
from marvis.plugins.manifest import ToolRef
from marvis.repositories.tasks import TaskRepository
from marvis.risk_context.event_contracts import EventError, content_hash
from marvis.risk_context.event_service import EventService, KIND
from test_event_features import coverage, event, query, source
from test_operations_api import _claim_role


def iso(instant):
    return instant.isoformat()


@pytest.fixture
def runtime(tmp_path):
    app = create_app(tmp_path / "workspace")
    settings = app.state.settings
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="事件历史特征",
            model_version="test",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="manual",
            task_type=TASK_TYPE_MODELING,
        )
    )
    maker, admin, other = TestClient(app), TestClient(app), TestClient(app)
    principal = _claim_role(app, maker, "maker")
    _claim_role(app, admin, "admin")
    _claim_role(app, other, "maker")
    basis = admin.post(
        f"/api/tasks/{task.id}/risk-sources/authorization-bases",
        json={
            "basis_id": "event-basis",
            "reference": "synthetic-source-coverage-review",
            "declaration": "synthetic_reference_test",
        },
    )
    assert basis.status_code == 201, basis.text
    src = source(task_id=task.id)
    response = admin.post("/api/risk-events/sources", json=src.model_dump())
    assert response.status_code == 201, response.text
    now = datetime.now(UTC)
    grant = {
        "grant_id": "event-grant",
        "task_id": task.id,
        "source_id": src.source_id,
        "source_contract_hash": src.contract_hash,
        "grantee_id": principal["id"],
        "permissions": ["read", "write"],
        "purpose": "synthetic event history",
        "basis_artifact_id": basis.json()["id"],
        "starts_at": iso(now - timedelta(hours=1)),
        "expires_at": iso(now + timedelta(hours=1)),
    }
    response = admin.post("/api/risk-events/grants", json=grant)
    assert response.status_code == 201, response.text
    return SimpleNamespace(
        app=app,
        settings=settings,
        task=task,
        maker=maker,
        admin=admin,
        other=other,
        principal=principal,
        source=src,
        grant=grant,
        root=tmp_path,
        serial=0,
        anchor=now - timedelta(minutes=2),
    )


def ingest(rt, rows, kind="events", *, client=None):
    path = rt.root / f"event-{rt.serial}.parquet"
    rt.serial += 1
    pd.DataFrame({"claim": [json.dumps(row.model_dump()) for row in rows]}).to_parquet(
        path, index=False
    )
    record = rt.app.state.risk_events.repo.registry.register_existing(
        path, task_id=rt.task.id, role="event_history"
    )
    payload = {
        "grant_id": rt.grant["grant_id"],
        "dataset": {
            "source_id": rt.source.source_id,
            "source_contract_hash": rt.source.contract_hash,
            "dataset_id": record.id,
            "expected_content_hash": record.content_hash,
            "kind": kind,
            "json_column": "claim",
        },
    }
    return (client or rt.maker).post(
        f"/api/tasks/{rt.task.id}/risk-events/imports", json=payload
    )


def prepare(rt, *, complete=True, request_id="historical-events"):
    anchor = rt.anchor
    response = ingest(
        rt,
        [
            event(
                event_at=iso(anchor - timedelta(seconds=10)),
                available_at=iso(anchor - timedelta(seconds=9)),
            )
        ],
    )
    assert response.status_code == 201, response.text
    if complete:
        response = ingest(
            rt,
            [
                coverage(
                    start_exclusive=iso(anchor - timedelta(seconds=60)),
                    through_inclusive=iso(anchor),
                    declared_at=iso(anchor),
                    available_at=iso(anchor),
                )
            ],
            "coverage",
        )
        assert response.status_code == 201, response.text
    contract = query(
        rt.source,
        decision_at=iso(anchor),
        knowledge_cutoff=iso(datetime.now(UTC)),
        availability_mode="retrospective_declared",
    )
    response = rt.maker.post(
        f"/api/tasks/{rt.task.id}/risk-events/requests",
        json={
            "request_id": request_id,
            "grant_id": rt.grant["grant_id"],
            "contract": contract.model_dump(),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def gated_plan(rt, proposal):
    created = rt.maker.post(
        f"/api/tasks/{rt.task.id}/plans",
        json={
            "goal": "事件窗口特征回放",
            "slots": {
                "event_request_id": proposal["request_id"],
                "event_proposal_hash": proposal["proposal_hash"],
                "event_contract": proposal["contract"],
            },
        },
    )
    assert created.status_code == 201, created.text
    plan = created.json()["plan"]
    response = rt.maker.post(
        f"/api/plans/{plan['id']}/confirm", json=plan["confirmation_snapshot"]
    )
    assert response.status_code == 200, response.text
    response = rt.maker.post(f"/api/plans/{plan['id']}/run")
    assert response.status_code == 202, response.text
    current = rt.maker.get(f"/api/plans/{plan['id']}").json()["plan"]
    assert current["status"] == "awaiting_confirm", current
    assert not any(
        a["kind"] == KIND
        for a in rt.app.state.risk_events.artifacts.list_for_task(rt.task.id)
    )
    return current


def approve(rt, plan, *, expect_read_denied=False):
    step = plan["steps"][0]
    response = rt.maker.post(
        f"/api/plans/{plan['id']}/steps/{step['id']}/decisions",
        json={
            "decision": "approve",
            "reason": "已确认来源、历史声明、窗口与未知边界",
            **step["confirmation_snapshot"],
        },
    )
    assert response.status_code == 202, response.text
    response = rt.maker.get(f"/api/plans/{plan['id']}")
    if expect_read_denied:
        assert response.status_code == 403, response.text
        return private_plan_snapshot(rt.app, plan["id"])
    assert response.status_code == 200, response.text
    return response.json()["plan"]


def private_plan_snapshot(app, plan_id):
    # Tests may inspect the repository to assert no execution took place after
    # HTTP read authorization failed; this is never a production HTTP fallback.
    from marvis.orchestrator.contracts import plan_to_dict

    plan = app.state.plan_repo.load_plan(plan_id)
    return {
        **plan_to_dict(plan),
        "reconciliation": app.state.plan_executor.reconciler.describe(plan),
    }


def url(rt, proposal, suffix=""):
    return f"/api/tasks/{rt.task.id}/risk-events/requests/{proposal['request_id']}{suffix}?grant_id={rt.grant['grant_id']}"


@pytest.mark.parametrize("complete", [True, False])
def test_http_human_gate_real_toolrunner_and_authoritative_export(runtime, complete):
    rt = runtime
    proposal = prepare(rt, complete=complete)
    plan = gated_plan(rt, proposal)
    frozen = plan["steps"][0]["inputs"]
    assert frozen["contract"] == proposal["contract"]
    current = approve(rt, plan)
    assert current["status"] == "done", current
    summary = rt.maker.get(url(rt, proposal))
    assert summary.status_code == 200, summary.text
    result = summary.json()
    assert result["status"] == ("measured" if complete else "unknown")
    assert result["features"]["amount"]["value"] == (100 if complete else None)
    assert result["automated_clearance"] is False
    evidence = rt.maker.get(url(rt, proposal, "/evidence"))
    assert evidence.status_code == 200, evidence.text
    receipt = evidence.json()["receipt"]
    assert receipt["result"]["historical_platform_visibility"] == "not_asserted"
    assert (
        receipt["snapshot"]["claims"][0]["origin"] == "authenticated_registered_dataset"
    )
    assert evidence.json() == rt.maker.get(url(rt, proposal, "/export/json")).json()
    assert content_hash(receipt["snapshot"]) == result["snapshot_hash"]
    restarted = EventService(rt.settings)
    assert (
        restarted.evidence(
            rt.task.id, proposal["request_id"], rt.principal["id"], rt.grant["grant_id"]
        )
        == evidence.json()
    )
    assert rt.other.get(url(rt, proposal, "/evidence")).status_code == 403
    # A backdated publisher correction arriving later cannot alter this cutoff
    # or the already registered native snapshot, even on a fresh API read.
    correction = event(
        version=2,
        supersedes_version=1,
        event_at=iso(rt.anchor - timedelta(seconds=8)),
        available_at=iso(rt.anchor - timedelta(seconds=7)),
        values={"amount_minor": 999},
    )
    assert ingest(rt, [correction]).status_code == 201
    assert rt.maker.get(url(rt, proposal, "/evidence")).json() == evidence.json()


def test_direct_toolrunner_has_no_human_gate_bypass(runtime):
    rt = runtime
    proposal = prepare(rt)
    result = rt.app.state.tool_runner.invoke(
        ToolRef("risk_context", "replay_events"),
        {key: proposal[key] for key in ("request_id", "proposal_hash", "contract")},
        task_id=rt.task.id,
    )
    assert not result.ok
    assert "governance" in result.error
    with connect(rt.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM event_decisions").fetchone()[0] == 0


@pytest.mark.parametrize(
    "failure", ["revoked", "principal_revoked", "changed_contract", "changed_hash"]
)
def test_approved_plan_rechecks_live_authority_and_review_binding(runtime, failure):
    rt = runtime
    proposal = prepare(rt)
    if failure == "changed_contract":
        proposal["contract"]["window_seconds"] = 120
    if failure == "changed_hash":
        proposal["proposal_hash"] = "f" * 64
    if failure in {"changed_contract", "changed_hash"}:
        response = rt.maker.post(
            f"/api/tasks/{rt.task.id}/plans",
            json={
                "goal": "事件窗口特征回放",
                "slots": {
                    "event_request_id": proposal["request_id"],
                    "event_proposal_hash": proposal["proposal_hash"],
                    "event_contract": proposal["contract"],
                },
            },
        )
        assert response.status_code == 403, response.text
        assert rt.app.state.plan_repo.list_plans_for_task(rt.task.id) == []
        assert not any(
            a["kind"] == KIND
            for a in rt.app.state.risk_events.artifacts.list_for_task(rt.task.id)
        )
        return
    plan = gated_plan(rt, proposal)
    if failure == "revoked":
        response = rt.admin.post(
            f"/api/tasks/{rt.task.id}/risk-events/grants/{rt.grant['grant_id']}/revoke"
        )
        assert response.status_code == 200
    if failure == "principal_revoked":
        with connect(rt.settings.db_path) as conn:
            conn.execute(
                "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
                (rt.principal["id"],),
            )
    result = approve(rt, plan, expect_read_denied=True)
    assert result["status"] != "done", result
    assert not any(
        a["kind"] == KIND
        for a in rt.app.state.risk_events.artifacts.list_for_task(rt.task.id)
    )


def test_event_http_rejects_caller_snapshot_and_identity_without_echo(runtime):
    rt = runtime
    proposal = prepare(rt)
    payload = {
        "request_id": "forged",
        "grant_id": rt.grant["grant_id"],
        "contract": proposal["contract"],
        "snapshot": {},
        "actor_id": "sensitive-injected-identity",
    }
    response = rt.maker.post(
        f"/api/tasks/{rt.task.id}/risk-events/requests", json=payload
    )
    assert response.status_code == 422
    assert "sensitive-injected-identity" not in response.text
    denied = ingest(rt, [event()], client=rt.other)
    assert denied.status_code == 403
    denied = rt.maker.post("/api/risk-events/sources", json=rt.source.model_dump())
    assert denied.status_code == 403


def test_evidence_tamper_cannot_be_exported_or_repaired_by_replay(runtime):
    rt = runtime
    proposal = prepare(rt)
    assert approve(rt, gated_plan(rt, proposal))["status"] == "done"
    summary = rt.maker.get(url(rt, proposal)).json()
    record = rt.app.state.risk_events.artifacts.get_for_task(
        rt.task.id, summary["artifact_id"]
    )
    Path(record["path"]).write_text("{}")
    assert rt.maker.get(url(rt, proposal)).status_code == 409
    assert rt.maker.get(url(rt, proposal, "/export/json")).status_code == 409
    with pytest.raises(EventError, match="event_artifact_integrity_failed"):
        rt.app.state.risk_events.read(
            rt.task.id, proposal["request_id"], rt.principal["id"], rt.grant["grant_id"]
        )

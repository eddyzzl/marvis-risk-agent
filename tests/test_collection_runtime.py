"""Actual HTTP Plans, live human/effect gates, isolated worker and native exports."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.collection.actions import CollectionPolicy
from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.service import CollectionService
from marvis.db_schema import connect
from marvis.repositories.tasks import TaskRepository
from tests.test_collection_cashflows import UNIT, flow, request as feedback_request
from tests.test_collection_planning import AT, case, history, policy, spec
from tests.test_db import _task_create
from tests.test_operations_api import _claim_role


@pytest.fixture
def runtime(tmp_path):
    app = create_app(tmp_path / "workspace")
    settings = app.state.settings
    task = TaskRepository(settings.db_path).create_task(
        _task_create(task_type="strategy")
    )
    maker, checker, other = TestClient(app), TestClient(app), TestClient(app)
    principal = _claim_role(app, maker, "maker")
    _claim_role(app, checker, "checker")
    _claim_role(app, other, "maker")
    base = f"/api/tasks/{task.id}/collection"
    material = maker.post(
        base + "/materials",
        json={
            "material_id": "history",
            "description": "Synthetic deidentified local test declarations",
            "declarations": {"historical": True},
        },
    )
    assert material.status_code == 201, material.text
    refs = {
        k: material.json()[k] for k in ("source_artifact_id", "source_artifact_hash")
    }
    created = maker.post(
        base + "/cases",
        json={
            "case_id": "case-1",
            "subject_namespace": "fixture",
            "subject_token": "a" * 64,
            "unit": UNIT,
            "opening_balance_minor": 10000,
            "opened_at": "2026-08-01T00:00:00Z",
            "source_assurance": "historical_import_unverified",
            **refs,
        },
    )
    assert created.status_code == 201, created.text
    p = policy(
        unit=UNIT,
        basis_artifact_id=refs["source_artifact_id"],
        basis_artifact_hash=refs["source_artifact_hash"],
        max_estimated_active_cost_minor=100,
    )
    raw = p.model_dump()
    raw["queues"][0]["max_active_actions"] = 5
    p = CollectionPolicy.model_validate(raw)
    batch = CollectionBatchRequest(
        batch_id="http-batch",
        execution_mode="local_reference",
        policy=p,
        strategy=spec(p).to_dict(),
        cases=[case("case-1", subject_namespace="fixture", **refs)],
        histories=[history(subject_namespace="fixture", **refs)],
        as_of=AT,
        knowledge_cutoff=AT,
    )
    response = maker.post(base + "/batches", json=batch.model_dump())
    assert response.status_code == 201, response.text
    return SimpleNamespace(
        app=app,
        settings=settings,
        task=task,
        maker=maker,
        checker=checker,
        other=other,
        principal=principal,
        refs=refs,
        base=base,
        proposal=response.json(),
    )


def plan(rt, operation, goal):
    result = rt.maker.post(
        f"/api/tasks/{rt.task.id}/plans",
        json={
            "goal": goal,
            "slots": {
                "collection_" + k: rt.proposal[k]
                for k in ("batch_id", "request_hash", "preview_hash")
            },
        },
    )
    assert result.status_code == 201, result.text
    p = result.json()["plan"]
    assert p["template_id"] == "collection_" + operation
    confirmed = rt.maker.post(
        f"/api/plans/{p['id']}/confirm", json=p["confirmation_snapshot"]
    )
    assert confirmed.status_code == 200, confirmed.text
    result = rt.maker.post(f"/api/plans/{p['id']}/run")
    assert result.status_code == 202, result.text
    p = rt.maker.get(f"/api/plans/{p['id']}").json()["plan"]
    assert p["status"] == "awaiting_confirm", p
    return p


def approve(rt, p):
    step = p["steps"][0]
    result = rt.checker.post(
        f"/api/plans/{p['id']}/steps/{step['id']}/decisions",
        json={
            "decision": "approve",
            "reason": "Reviewed local reference clock, declared history and resource limits",
            **step["confirmation_snapshot"],
        },
    )
    assert result.status_code == 202, result.text
    return rt.maker.get(f"/api/plans/{p['id']}").json()["plan"]


def test_native_http_human_gate_worker_and_cashflow_feedback(runtime):
    rt = runtime
    waiting = plan(rt, "queue_batch", "催收参考排队")
    before = rt.maker.get(rt.base + "/batches/http-batch").json()
    assert before["status"] == "proposed" and before["actions"] == []
    completed = approve(rt, waiting)
    assert completed["status"] == "done", completed
    queued = rt.maker.get(rt.base + "/batches/http-batch")
    assert queued.status_code == 200, queued.text
    body = queued.json()
    assert body["status"] == "queued" and len(body["actions"]) == 1
    assert body["actions"][0]["assigned_to"] == rt.principal["id"]
    original = body["effects"][0]
    evidence = rt.maker.get(original["evidence_url"])
    assert evidence.status_code == 200, evidence.text
    exported = rt.maker.get(original["evidence_url"] + "/export")
    assert exported.status_code == 200, exported.text
    assert exported.json() == evidence.json()["facts"]
    listed = rt.maker.get(f"/api/tasks/{rt.task.id}/task-artifacts").json()["artifacts"]
    artifact = next(item for item in listed if item["id"] == original["artifact_id"])
    assert artifact["available"] is True
    assert rt.maker.get(artifact["download_url"]).content == exported.content
    assert evidence.json()["facts"]["clock_scope"] == "declared_reference_simulation"
    assert rt.other.get(original["evidence_url"]).status_code in {403, 409}
    executed = approve(rt, plan(rt, "execute_reference", "催收参考执行"))
    assert executed["status"] == "done", executed
    after = rt.maker.get(rt.base + "/batches/http-batch").json()
    assert (
        after["status"] == "completed" and after["actions"][0]["state"] == "completed"
    )
    assert after["actions"][0]["actual_cost_minor"] is None
    assert after["actions"][0]["customer_contacted"] is False
    restarted = CollectionService(rt.settings)
    assert restarted.read(rt.task.id, "http-batch", rt.principal["id"]) == after
    assert rt.maker.get(original["evidence_url"]).json() == evidence.json()
    # Explicit observed historical cashflow is independent of reference actions.
    value = flow(rt.refs)
    response = rt.maker.post(
        rt.base + "/cashflows", json={"events": [value.model_dump()]}
    )
    assert response.status_code == 201, response.text
    feedback = rt.maker.post(
        rt.base + "/reconciliations", json=feedback_request(rt.refs).model_dump()
    )
    assert feedback.status_code == 201, feedback.text
    facts = feedback.json()
    assert facts["amounts"]["net_payments_minor"] == 3000
    assert facts["incremental_recovery_identified"] is False
    assert (
        rt.maker.get(
            rt.base + "/cases/case-1/reconciliations/" + facts["receipt_hash"]
        ).json()
        == facts
    )
    with connect(rt.settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM effect_executions WHERE status='committed'"
        ).fetchall()
        assert len(rows) == 2
        assert (
            conn.execute(
                "SELECT count(*) FROM approval_records WHERE status='consumed'"
            ).fetchone()[0]
            == 2
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM plan_step_runs WHERE status='succeeded'"
            ).fetchone()[0]
            == 2
        )


def test_http_cancellation_is_another_approved_effect(runtime):
    rt = runtime
    queued = approve(rt, plan(rt, "queue_batch", "催收参考排队"))
    assert queued["status"] == "done", queued
    waiting = plan(rt, "cancel_batch", "取消催收参考批次")
    assert rt.maker.get(rt.base + "/batches/http-batch").json()["status"] == "queued"
    cancelled = approve(rt, waiting)
    assert cancelled["status"] == "done", cancelled
    value = rt.maker.get(rt.base + "/batches/http-batch").json()
    assert value["status"] == "cancelled"
    assert value["actions"][0]["state"] == "cancelled"


def test_http_rejects_self_claimed_approval_identity_or_callback(runtime):
    rt = runtime
    invalid = {
        **rt.proposal["request"],
        "actor_id": rt.principal["id"],
        "approved": True,
    }
    response = rt.maker.post(rt.base + "/batches", json=invalid)
    assert response.status_code == 422, response.text
    assert rt.other.post(
        rt.base + "/cashflows", json={"events": [flow(rt.refs).model_dump()]}
    ).status_code in {403, 409}
    with connect(rt.settings.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_evidence WHERE kind='flow'"
            ).fetchone()[0]
            == 0
        )


def test_http_delete_retains_task_native_receipts_and_resources(runtime):
    rt = runtime
    assert approve(rt, plan(rt, "queue_batch", "催收参考排队"))["status"] == "done"
    before = rt.maker.get(rt.base + "/batches/http-batch").json()
    response = rt.maker.delete(f"/api/tasks/{rt.task.id}")
    assert response.status_code == 409, response.text
    assert "collection_reference_evidence_retained" in response.text
    assert "保留" in response.text
    assert rt.maker.get(rt.base + "/batches/http-batch").json() == before
    with connect(rt.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM tasks WHERE id=?", (rt.task.id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute("SELECT count(*) FROM collection_queue_items").fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM task_fs_gc_queue WHERE origin_task_id=?",
                (rt.task.id,),
            ).fetchone()[0]
            == 0
        )

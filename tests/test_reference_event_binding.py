"""Governed event receipts enter the same native package worker as raw inputs."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import httpx
import pytest
from pydantic import ValidationError

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.reference_decision.contracts import (
    DecisionError,
    DecisionRequest,
    PackageRequest,
    RulePackageRequest,
    digest,
)
from marvis.reference_decision.event_contracts import (
    EventEvidenceReference,
    EventFeatureRecipe,
)
from marvis.reference_decision.evaluation import evaluate
from marvis.risk_context.event_contracts import EventFeatureContract
import test_event_runtime as event_fixtures
from test_event_runtime import prepare, gated_plan, approve, url
from test_operations_api import _claim_role
from test_reference_deployment import _promotion
from test_reference_rule_packages import register
from test_risk_source_runtime import free_port


@pytest.fixture
def event_runtime(tmp_path):
    return event_fixtures.runtime.__wrapped__(tmp_path)


def build(rt, *, complete=True):
    proposal = prepare(rt, complete=complete)
    assert approve(rt, gated_plan(rt, proposal))["status"] == "done"
    evidence = rt.maker.get(url(rt, proposal, "/evidence")).json()
    contract = EventFeatureContract.model_validate(proposal["contract"])
    reference = EventEvidenceReference(
        task_id=rt.task.id,
        request_id=proposal["request_id"],
        grant_id=rt.grant["grant_id"],
        expected_content_hash=evidence["content_hash"],
        contract=contract,
    )
    spec = StrategySpec(
        strategy_type="approval",
        rules=(
            StrategyRuleSpec(
                rule_id="event-velocity",
                priority=1,
                condition={
                    "op": "compare",
                    "field": "transactions",
                    "operator": ">=",
                    "value": 1,
                },
                action=StrategyAction(type="review", reason_code="EVENT_REVIEW"),
            ),
        ),
        default_action=StrategyAction(type="approval", reason_code="NO_EVENT"),
    )
    register(rt.app, rt.task.id, spec, "online-strategy")
    request = RulePackageRequest(
        package_kind="rule_only",
        strategy_id="online-strategy",
        strategy_version=1,
        decision_node="underwriting",
        raw_schema=[],
        event_binding={
            "task_id": rt.task.id,
            "authoring_grant_id": rt.grant["grant_id"],
            "recipe": EventFeatureRecipe.from_contract(contract).model_dump(),
        },
    )
    response = rt.maker.post(
        "/api/reference-decision/packages", json=request.model_dump()
    )
    assert response.status_code == 201, response.text
    return request, reference, response.json(), evidence


def activate(rt, package, reference):
    checker = TestClient(rt.app)
    _claim_role(rt.app, checker, "checker")
    admin_id = rt.admin.get("/api/production-governance/me").json()["id"]
    response = rt.admin.post(
        "/api/risk-events/grants",
        json={
            **rt.grant,
            "grant_id": "admin-event-read",
            "permissions": ["read"],
            "grantee_id": admin_id,
        },
    )
    assert response.status_code == 201, response.text
    promotion = _promotion(
        (rt.app, rt.maker, checker, rt.admin), package["package_hash"]
    )
    probe_ref = reference.model_copy(update={"grant_id": "admin-event-read"})
    installed = rt.admin.post(
        "/api/reference-decision/installations",
        json={
            "promotion_id": promotion["id"],
            "probe_features": {},
            "event_evidence": probe_ref.model_dump(),
        },
    )
    assert installed.status_code == 201, installed.text
    receipt = installed.json()
    assert receipt["probe"] == {
        "execution": "completed",
        "event_assessment": "restricted_to_source_grant",
    }
    activated = rt.admin.post(
        f"/api/production-governance/environments/local-reference/promotion-requests/{promotion['id']}/activate",
        json={
            "verifier_id": "local-reference.v1",
            "receipt_id": receipt["activation_evidence"]["receipt_id"],
            "reason": "event reference acceptance",
        },
    )
    assert activated.status_code == 201, activated.text


def application(package, reference, **changes):
    return {
        "request_id": "event-application",
        "decision_node": "underwriting",
        "features": {},
        "expected_package_hash": package["package_hash"],
        "event_evidence": reference.model_dump(),
        **changes,
    }


def test_event_governed_receipt_to_package_http_worker_cli_and_restart(event_runtime):
    rt = event_runtime
    request, reference, package, evidence = build(rt)
    activate(rt, package, reference)
    payload = application(package, reference)
    assert (
        rt.other.post("/api/reference-decision/decisions", json=payload).status_code
        == 403
    )
    assert (
        rt.maker.post(
            "/api/reference-decision/decisions",
            json={**payload, "actor_id": rt.principal["id"]},
        ).status_code
        == 422
    )
    response = rt.maker.post("/api/reference-decision/decisions", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert (
        result["status"] == "decided"
        and result["action"]["reason_code"] == "EVENT_REVIEW"
    )
    assert result["score"] is None
    assert result["event_evidence"]["content_hash"] == evidence["content_hash"]
    assert (
        result["event_evidence"]["snapshot_hash"]
        == evidence["receipt"]["result"]["snapshot_hash"]
    )
    assert result["event_evidence"]["availability_mode"] == "retrospective_declared"
    assert result["event_evidence"]["fraud_or_identity_proof"] is False
    assert reference.contract.focus.token not in str(result)
    fresh = create_app(rt.settings)
    assert (
        fresh.state.reference_decision.decide(
            DecisionRequest(**payload), actor_id=rt.principal["id"]
        )
        == result
    )
    assert (
        rt.maker.post("/api/reference-decision/decisions", json=payload).json()
        == result
    )
    # A second real governed event request with an earlier knowledge cutoff has
    # no coverage evidence. It cannot take the rule's approving default.
    unknown_contract = reference.contract.model_copy(
        update={"knowledge_cutoff": reference.contract.decision_at}
    )
    prepared = rt.maker.post(
        f"/api/tasks/{rt.task.id}/risk-events/requests",
        json={
            "request_id": "unknown-cutoff",
            "grant_id": reference.grant_id,
            "contract": unknown_contract.model_dump(),
        },
    )
    assert prepared.status_code == 201, prepared.text
    proposal = prepared.json()
    plan_response = rt.maker.post(
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
    assert plan_response.status_code == 201, plan_response.text
    plan = plan_response.json()["plan"]
    assert (
        rt.maker.post(
            f"/api/plans/{plan['id']}/confirm", json=plan["confirmation_snapshot"]
        ).status_code
        == 200
    )
    assert rt.maker.post(f"/api/plans/{plan['id']}/run").status_code == 202
    assert (
        approve(rt, rt.maker.get(f"/api/plans/{plan['id']}").json()["plan"])["status"]
        == "done"
    )
    unknown = rt.maker.get(url(rt, proposal, "/evidence")).json()
    unknown_ref = reference.model_copy(
        update={
            "request_id": "unknown-cutoff",
            "contract": unknown_contract,
            "expected_content_hash": unknown["content_hash"],
        }
    )
    fallback = rt.maker.post(
        "/api/reference-decision/decisions",
        json=application(package, unknown_ref, request_id="unknown-event-application"),
    )
    assert fallback.status_code == 200, fallback.text
    assert (
        fallback.json()["status"] == "fallback"
        and fallback.json()["error_code"] == "event_features_unknown"
    )
    assert (
        fallback.json()["action"]["type"] == "review"
        and fallback.json()["score"] is None
    )
    port = free_port()
    with (rt.root / "event-reference-cli.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "marvis",
                "serve",
                "--workspace",
                str(rt.settings.workspace),
                "--port",
                str(port),
            ],
            cwd=Path(__file__).parents[1],
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                cookies=dict(rt.maker.cookies),
                timeout=30,
            ) as client:
                deadline = time.monotonic() + 25
                while True:
                    assert process.poll() is None, (
                        rt.root / "event-reference-cli.log"
                    ).read_text()
                    try:
                        assert (
                            client.get("/api/reference-decision/status").status_code
                            == 200
                        )
                        break
                    except httpx.TransportError:
                        assert time.monotonic() < deadline
                        time.sleep(0.05)
                assert (
                    client.post(
                        "/api/reference-decision/decisions", json=payload
                    ).json()
                    == result
                )
                new = client.post(
                    "/api/reference-decision/decisions",
                    json={**payload, "request_id": "event-cli-new"},
                )
                assert new.status_code == 200, new.text
                assert new.json()["action"] == result["action"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    # Current authority is required even when no new execution is needed.
    rt.app.state.risk_events.repo.revoke_grant(
        rt.task.id,
        rt.grant["grant_id"],
        rt.admin.get("/api/production-governance/me").json()["id"],
    )
    assert (
        rt.maker.post("/api/reference-decision/decisions", json=payload).status_code
        == 403
    )


def test_missing_unknown_and_forged_event_inputs_never_approve(event_runtime):
    rt = event_runtime
    request, reference, package, _ = build(rt, complete=False)
    service = rt.app.state.reference_decision
    with pytest.raises(DecisionError, match="event_features_unknown"):
        service.evaluate(
            package["package_hash"],
            {},
            20,
            event_evidence=reference,
            actor_id=rt.principal["id"],
        )
    # Installation cannot certify a package using an unknown coverage probe.
    with pytest.raises(DecisionError, match="event_evidence_required"):
        service.evaluate(package["package_hash"], {}, 20)
    with service.packages.snapshot(package["package_hash"]) as (manifest, directory):
        with pytest.raises(DecisionError, match="event_evidence_required"):
            evaluate(manifest, directory, {})
        with pytest.raises(DecisionError, match="raw_schema_mismatch"):
            evaluate(manifest, directory, {"transactions": 0})
    with pytest.raises(ValidationError):
        RulePackageRequest.model_validate(
            {
                **request.model_dump(),
                "raw_schema": [{"name": "transactions", "type": "integer"}],
            }
        )
    assert (
        rt.other.post(
            "/api/reference-decision/packages", json=request.model_dump()
        ).status_code
        == 403
    )


@pytest.mark.parametrize(
    "change,code",
    [
        ("hash", "event_evidence_content_hash_mismatch"),
        ("subject", "event_application_context_mismatch"),
        ("time", "event_application_context_mismatch"),
        ("window", "event_recipe_binding_mismatch"),
        ("task", "event_recipe_binding_mismatch"),
    ],
)
def test_receipt_is_bound_to_recipe_subject_times_and_source(
    event_runtime, change, code
):
    rt = event_runtime
    _, reference, package, _ = build(rt)
    payload = reference.model_dump()
    if change == "hash":
        payload["expected_content_hash"] = "0" * 64
    elif change == "subject":
        payload["contract"]["focus"]["token"] = "1" * 64
    elif change == "time":
        payload["contract"]["decision_at"] = (
            datetime.now(UTC) - timedelta(hours=1)
        ).isoformat()
    elif change == "window":
        payload["contract"]["window_seconds"] += 1
    else:
        payload["task_id"] = "another-task"
    with pytest.raises(DecisionError, match=code):
        rt.app.state.reference_decision.evaluate(
            package["package_hash"],
            {},
            20,
            event_evidence=EventEvidenceReference.model_validate(payload),
            actor_id=rt.principal["id"],
        )


def test_evidence_file_tampering_and_expired_principal_fail_closed(event_runtime):
    rt = event_runtime
    _, reference, package, evidence = build(rt)
    row = next(
        r
        for r in rt.app.state.risk_events.artifacts.list_for_task(rt.task.id)
        if r["id"] == evidence["artifact_id"]
    )
    path = Path(row["path"])
    path.write_bytes(b"changed")
    with pytest.raises(DecisionError, match="event_artifact_integrity_failed"):
        rt.app.state.reference_decision.evaluate(
            package["package_hash"],
            {},
            20,
            event_evidence=reference,
            actor_id=rt.principal["id"],
        )
    with connect(rt.settings.db_path) as conn:
        conn.execute(
            "UPDATE local_principals SET expires_at='2020-01-01T00:00:00Z' WHERE id=?",
            (rt.principal["id"],),
        )
    result = rt.maker.post(
        "/api/reference-decision/decisions", json=application(package, reference)
    )
    assert result.status_code in {401, 403}, result.text


def test_event_binding_and_native_model_share_one_scorer(event_runtime):
    from test_modeling_monitor import _train_lr_experiment

    rt = event_runtime
    request, reference, _, _ = build(rt)
    trained, _ = _train_lr_experiment(
        rt.app.state.tool_runner,
        rt.app.state.risk_events.repo.registry,
        rt.root,
        rt.task,
    )
    payload = request.model_dump(exclude={"package_kind"})
    payload.update(
        model_artifact_id=trained.output["artifact_id"],
        score_field="pd",
        raw_schema=[{"name": "x1", "type": "number"}, {"name": "x2", "type": "number"}],
    )
    req = PackageRequest.model_validate(payload)
    hash_, manifest = rt.app.state.reference_decision.packages.build(
        req, actor_id=rt.principal["id"]
    )
    result = rt.app.state.reference_decision.evaluate(
        hash_,
        {"x1": 0.1, "x2": 0.2},
        20,
        event_evidence=reference,
        actor_id=rt.principal["id"],
    )
    assert 0 <= result["score"] <= 1 and result["score_product"] == "raw_pd"
    assert result["action"]["reason_code"] == "EVENT_REVIEW"
    assert manifest["model_producer_receipt"] and result["event_evidence"]


def test_legacy_completed_request_is_read_without_changing_identity(tmp_path):
    app = create_app(tmp_path / "workspace")
    request = DecisionRequest(
        request_id="old",
        decision_node="underwriting",
        expected_package_hash="a" * 64,
        features={},
    )
    legacy = request.model_dump(exclude={"event_evidence"})
    ledger = app.state.reference_decision.ledger
    scope = "local-reference:production"
    owner, _ = ledger.claim(scope, "old", digest(legacy), "a" * 64, 10)
    saved = ledger.finish(
        scope, "old", owner, {"action": {"type": "review"}, "timing_ms": {}}
    )
    # No active deployment exists: only the original persisted identity can read this.
    assert app.state.reference_decision.decide(request) == saved

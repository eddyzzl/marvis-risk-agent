"""Two real independent local HTTP instances; synthetic interoperability evidence."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.institution_adapters.contracts import EndpointConfig
from marvis.institution_adapters.router import configure_institution_adapters, router
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.plugins.manifest import ToolRef
from marvis.reference_decision.contracts import digest
from marvis.production_governance.errors import GovernanceConflict
from marvis.repositories.modeling import ModelingRepository

from test_modeling_pack import _runtime, _register_modeling_sample
from test_operations_api import _claim_role
from test_risk_source_runtime import free_port, source_record

SCHEMA = [{"name": "x1", "type": "number", "nullable": False}, {"name": "x2", "type": "number", "nullable": False}]


@contextmanager
def endpoint(root, alias, *, drop=False):
    root.mkdir()
    records = root / "records.json"
    records.write_text(json.dumps([source_record()]))
    port = free_port()
    args = [sys.executable, "-m", "marvis.institution_adapters.reference_endpoint", "--state-dir", str(root), "--instance-id", alias, "--port", str(port), "--records", str(records)]
    if drop:
        args.append("--drop-every-new-operation-response")
    process = subprocess.Popen(args, cwd=Path(__file__).parents[1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                raise AssertionError("reference protocol endpoint exited before ready")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise AssertionError("reference endpoint startup deadline")
                time.sleep(0.03)
        yield EndpointConfig(alias=alias, instance_id=alias, environment=alias, mode="local_reference", base_url=f"http://127.0.0.1:{port}", credential_file=root / "access_token"), root
    finally:
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(5)


@pytest.fixture(scope="module")
def instances(tmp_path_factory):
    root = tmp_path_factory.mktemp("independent-institution-protocols")
    with endpoint(root / "a", "reference-a") as a, endpoint(root / "b", "reference-b", drop=True) as b:
        yield a, b


@pytest.fixture(scope="module")
def runtime(tmp_path_factory, instances):
    root = tmp_path_factory.mktemp("institution-adapters-native")
    runner, _, registry, _, settings, task = _runtime(root)
    data = _register_modeling_sample(registry, root, task.id)
    trained = runner.invoke(ToolRef("modeling", "train_model"), {"dataset_id": data.id, "recipe": "lr", "features": ["x1", "x2"], "target_col": "y", "split_col": "split", "split_values": {"train": "train", "test": "test", "oot": "oot"}, "seed": 20261009}, task_id=task.id)
    assert trained.ok, trained.error
    app = create_app(settings)
    if not any(getattr(r, "path", "") == "/api/tasks/{task_id}/institution-adapters/capabilities" for r in app.routes):
        app.include_router(router)
    configure_institution_adapters(app, [v[0] for v in instances])
    clients = [TestClient(app) for _ in range(3)]
    actors = {role: _claim_role(app, client, role)["id"] for role, client in zip(("maker", "checker", "admin"), clients, strict=True)}
    yield SimpleNamespace(app=app, settings=settings, task=task, root=root, maker=clients[0], checker=clients[1], admin=clients[2], actors=actors, artifact_id=trained.output["artifact_id"], instances=instances)


def prefix(rt, alias="reference-a"):
    return f"/api/tasks/{rt.task.id}/institution-adapters/{alias}"


def expect(response, status=200):
    assert response.status_code == status, response.text
    return response.json()


def effect_count(root, identity=None):
    with sqlite3.connect(root / "remote.sqlite") as conn:
        return conn.execute("SELECT count(*) FROM effects" + (" WHERE id=?" if identity else ""), (identity,) if identity else ()).fetchone()[0]


def post_count(root, identity):
    with sqlite3.connect(root / "remote.sqlite") as conn:
        return conn.execute("SELECT count(*) FROM requests WHERE operation_id=? AND method='POST'", (identity,)).fetchone()[0]


def promotion(rt, identity, *, alias="reference-a", registry_binding=None, threshold=0.5):
    metadata = {"institution_deployment": {"package_kind": "rule_only", "schema_hash": digest(SCHEMA)}}
    if registry_binding:
        metadata["institution_registry"] = registry_binding
    spec = StrategySpec(strategy_type="approval", rules=(StrategyRuleSpec(rule_id="low", priority=1, condition={"op": "compare", "field": "x1", "operator": "<", "value": threshold}, action=StrategyAction(type="approval", reason_code="LOW")),), default_action=StrategyAction(type="review", reason_code="REVIEW"), metadata=metadata)
    strategy = replace(build_strategy_from_spec(spec, score_col="x1", description="synthetic local protocol check"), id=identity)
    StrategyRepository(rt.settings.db_path).create_strategy(rt.task.id, strategy)
    with connect(rt.settings.db_path) as conn:
        conn.execute("UPDATE strategies SET asset_status='validated',status='draft' WHERE id=?", (identity,))
    created = expect(rt.maker.post("/api/production-governance/promotion-requests", json={"environment": alias, "deployment_slot": "production", "strategy_id": identity, "strategy_version": 1, "reason": "synthetic native protocol check"}), 201)
    for client in (rt.checker, rt.admin):
        created = expect(client.post(f"/api/production-governance/promotion-requests/{created['id']}/approvals", json={"decision": "approve", "reason": "independent synthetic review"}))
    return created


def upload(rt, identity, alias="reference-a"):
    return rt.maker.post(prefix(rt, alias) + "/registry/uploads", json={"operation_id": identity, "model_artifact_id": rt.artifact_id, "raw_schema": SCHEMA})


def test_registry_native_upload_ack_readback_promote_revoke(runtime):
    rt = runtime
    value = expect(upload(rt, "registry-a"), 201)
    manifest = value["result"]["manifest"]
    assert manifest["schema_hash"] == digest(SCHEMA)
    assert value["result"]["storage_only"] is True
    observed = expect(rt.maker.get(prefix(rt) + "/registry/uploads/registry-a/readback"))
    assert observed["manifest"] == manifest and observed["files_verified"] == manifest["files"]
    target = {"bundle_hash": observed["bundle_hash"], "model_artifact_id": rt.artifact_id, "schema_hash": digest(SCHEMA)}
    approved = promotion(rt, "registry-bound", registry_binding=target)
    common = {"upload_operation_id": "registry-a", "promotion_id": approved["id"]}
    promoted = expect(rt.admin.post(prefix(rt) + "/registry/promotions", json={"operation_id": "promote-a", **common}))
    assert promoted["result"]["state"] == "promoted" and promoted["result"]["serving"] is False
    revoked = expect(rt.admin.post(prefix(rt) + "/registry/revocations", json={"operation_id": "revoke-a", **common}))
    assert revoked["result"]["state"] == "revoked"
    assert expect(rt.maker.get(prefix(rt) + "/registry/uploads/registry-a/readback"))["state"] == "revoked"
    assert effect_count(rt.instances[0][1], "registry-a") == 1


def test_independent_second_instance_lost_ack_retry_never_resends_and_restart_recovers(runtime):
    rt = runtime
    failed = expect(upload(rt, "registry-b", "reference-b"), 409)
    assert failed["detail"]["code"] == "remote_effect_unknown"
    before = effect_count(rt.instances[1][1], "registry-b")
    again = expect(upload(rt, "registry-b", "reference-b"), 409)
    assert again["detail"]["code"] == "remote_effect_unknown" and before == 1
    assert effect_count(rt.instances[1][1], "registry-b") == before
    assert post_count(rt.instances[1][1], "registry-b") == 1
    configure_institution_adapters(rt.app, [v[0] for v in rt.instances])
    recovered = expect(rt.maker.post(prefix(rt, "reference-b") + "/operations/registry-b/recover"))
    assert recovered["status"] == "completed" and recovered["result"]["manifest"]["artifact_id"] == rt.artifact_id
    assert effect_count(rt.instances[1][1], "registry-b") == 1
    assert post_count(rt.instances[1][1], "registry-b") == 1
    assert effect_count(rt.instances[0][1], "registry-b") == 0
    assert expect(rt.maker.get(prefix(rt, "reference-b") + "/registry/uploads/registry-b/readback"))["storage_only"] is True


def test_real_rule_install_health_activate_second_version_and_rollback(runtime):
    rt = runtime
    first = promotion(rt, "deploy-first")
    body = {"operation_id": "install-first", "promotion_id": first["id"], "raw_schema": SCHEMA, "probe_features": {"x1": 0.1, "x2": 0.2}}
    installed = expect(rt.admin.post(prefix(rt) + "/deployments/install", json=body), 201)
    assert installed["result"]["probe_decision"]["action"]["type"] == "approval"
    assert expect(rt.admin.get(prefix(rt) + "/deployments/installations/install-first/readback"))["health_status"] == "healthy"
    activated = expect(rt.admin.post(prefix(rt) + "/deployments/activate", json={"operation_id": "activate-first", "install_operation_id": "install-first"}))
    assert activated["result"]["local_governance"]["status"] == "activated"
    assert expect(rt.admin.post(prefix(rt) + "/deployments/activate", json={"operation_id": "activate-first", "install_operation_id": "install-first"}))["result"] == activated["result"]
    second = promotion(rt, "deploy-second", threshold=0.05)
    expect(rt.admin.post(prefix(rt) + "/deployments/install", json={**body, "operation_id": "install-second", "promotion_id": second["id"]}), 201)
    expect(rt.admin.post(prefix(rt) + "/deployments/activate", json={"operation_id": "activate-second", "install_operation_id": "install-second"}))
    rollback = {"operation_id": "rollback-second", "deployment_operation_id": "activate-second", "expected_current_ref": "activate-second"}
    result = expect(rt.admin.post(prefix(rt) + "/deployments/rollback", json=rollback))
    assert result["result"]["active_ref"] == "activate-first"
    assert expect(rt.admin.post(prefix(rt) + "/deployments/rollback", json=rollback))["result"] == result["result"]
    native_head = expect(rt.admin.get("/api/production-governance/environments/reference-a"))
    assert native_head["active"]["external_deployment_ref"] == "activate-first"
    assert expect(rt.admin.get(prefix(rt) + "/deployments/installations/install-first/readback"))["active_ref"] == "activate-first"


def test_source_protocol_exact_grant_partial_unknown_and_revocation(runtime):
    rt = runtime
    basis = expect(rt.admin.post(f"/api/tasks/{rt.task.id}/risk-sources/authorization-bases", json={"basis_id": "configured-source-basis", "reference": "synthetic-reference-protocol", "declaration": "synthetic_reference_test"}), 201)
    now = datetime.now(UTC)
    grant = {"grant_id": "configured-grant", "profile_id": "reference-a", "task_id": rt.task.id, "grantee_id": rt.actors["maker"], "subject_namespace": "reference.people", "subject_token": "a" * 64, "purpose": "credit_review", "basis_artifact_id": basis["id"], "starts_at": (now - timedelta(minutes=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    expect(rt.admin.post(prefix(rt) + "/sources/grants", json=grant), 201)
    query = {"operation_id": "source-one", "grant_id": grant["grant_id"]}
    result = expect(rt.maker.post(prefix(rt) + "/sources/requests", json=query), 201)
    assert result["result"]["assessment"]["account_count"] is None
    assert result["result"]["automated_clearance"] is False
    expect(rt.maker.post(prefix(rt) + "/sources/requests", json=query), 201)
    assert effect_count(rt.instances[0][1], "source-one") == 1
    expect(rt.admin.post(prefix(rt) + "/sources/grants/configured-grant/revoke"))
    expect(rt.admin.post(prefix(rt) + "/sources/grants/configured-grant/revoke"))
    assert rt.maker.get(prefix(rt) + "/operations/source-one").status_code == 403
    assert rt.maker.post(prefix(rt) + "/sources/requests", json={**query, "operation_id": "source-two"}).status_code == 403
    assert effect_count(rt.instances[0][1], "source-two") == 0


def test_invalid_schema_roles_unknown_alias_and_arbitrary_network_are_explicit(runtime):
    rt = runtime
    assert rt.checker.post(prefix(rt) + "/registry/uploads", json={"operation_id": "bad-role", "model_artifact_id": rt.artifact_id, "raw_schema": SCHEMA}).status_code == 403
    assert rt.maker.post(prefix(rt) + "/registry/uploads", json={"operation_id": "bad-schema", "model_artifact_id": rt.artifact_id, "raw_schema": SCHEMA[::-1]}).status_code == 422
    assert upload(rt, "bad-alias", "unconfigured").status_code == 503
    assert rt.maker.post(prefix(rt) + "/registry/uploads", json={"operation_id": "bad-network", "model_artifact_id": rt.artifact_id, "raw_schema": SCHEMA, "endpoint": "http://127.0.0.1:1", "token": "secret"}).status_code == 422
    with pytest.raises(ValidationError):
        EndpointConfig(alias="x",instance_id="x",environment="x",mode="institution_https",base_url="http://example.com",credential_file=rt.root / "credential")
    capabilities = expect(rt.maker.get(f"/api/tasks/{rt.task.id}/institution-adapters/capabilities"))
    assert len(capabilities["endpoints"]) == 2 and capabilities["institution_production_acceptance"] == "not_established"
    assert "base_url" not in json.dumps(capabilities) and "credential_file" not in json.dumps(capabilities)


def test_every_other_remote_committed_family_recovers_only_by_observed_get(runtime):
    rt = runtime
    alias = "reference-b"
    root = rt.instances[1][1]
    base = prefix(rt, alias)
    expect(upload(rt, "registry-b-lifecycle", alias), 409)
    expect(rt.maker.post(base + "/operations/registry-b-lifecycle/recover"))
    observed = expect(rt.maker.get(base + "/registry/uploads/registry-b-lifecycle/readback"))
    target = {"bundle_hash": observed["bundle_hash"], "model_artifact_id": rt.artifact_id, "schema_hash": digest(SCHEMA)}
    approved = promotion(rt, "registry-bound-b", alias=alias, registry_binding=target)
    for path, identity in (("promotions", "promote-b"), ("revocations", "revoke-b")):
        body = {"operation_id": identity, "upload_operation_id": "registry-b-lifecycle", "promotion_id": approved["id"]}
        expect(rt.admin.post(base + "/registry/" + path, json=body), 409)
        expect(rt.admin.post(base + "/registry/" + path, json=body), 409)
        recovered = expect(rt.admin.post(base + f"/operations/{identity}/recover"))
        assert recovered["status"] == "completed" and post_count(root, identity) == effect_count(root, identity) == 1
    first = promotion(rt, "fault-deploy-first-b", alias=alias)
    installation = {"operation_id": "install-first-b", "promotion_id": first["id"], "raw_schema": SCHEMA, "probe_features": {"x1": 0.1, "x2": 0.2}}
    expect(rt.admin.post(base + "/deployments/install", json=installation), 409)
    expect(rt.admin.post(base + "/deployments/install", json=installation), 409)
    expect(rt.admin.post(base + "/operations/install-first-b/recover"))
    activation = {"operation_id": "activate-first-b", "install_operation_id": "install-first-b"}
    expect(rt.admin.post(base + "/deployments/activate", json=activation), 409)
    expect(rt.admin.post(base + "/deployments/activate", json=activation), 409)
    assert expect(rt.admin.get("/api/production-governance/environments/reference-b"))["active"] is None
    recovered = expect(rt.admin.post(base + "/operations/activate-first-b/recover"))
    assert recovered["result"]["local_governance"]["status"] == "activated"
    second = promotion(rt, "fault-deploy-second-b", alias=alias, threshold=0.05)
    expect(rt.admin.post(base + "/deployments/install", json={**installation, "operation_id": "install-second-b", "promotion_id": second["id"]}), 409)
    expect(rt.admin.post(base + "/operations/install-second-b/recover"))
    expect(rt.admin.post(base + "/deployments/activate", json={"operation_id": "activate-second-b", "install_operation_id": "install-second-b"}), 409)
    expect(rt.admin.post(base + "/operations/activate-second-b/recover"))
    rollback = {"operation_id": "rollback-second-b", "deployment_operation_id": "activate-second-b", "expected_current_ref": "activate-second-b"}
    expect(rt.admin.post(base + "/deployments/rollback", json=rollback), 409)
    expect(rt.admin.post(base + "/deployments/rollback", json=rollback), 409)
    expect(rt.admin.post(base + "/operations/rollback-second-b/recover"))
    assert expect(rt.admin.get("/api/production-governance/environments/reference-b"))["active"]["external_deployment_ref"] == "activate-first-b"
    for identity in ("install-first-b", "activate-first-b", "rollback-second-b"):
        assert post_count(root, identity) == effect_count(root, identity) == 1
    basis = expect(rt.admin.post(f"/api/tasks/{rt.task.id}/risk-sources/authorization-bases", json={"basis_id": "source-fault-basis", "reference": "synthetic-source", "declaration": "synthetic_reference_test"}), 201)
    now = datetime.now(UTC)
    grant = {"grant_id": "grant-b", "profile_id": alias, "task_id": rt.task.id, "grantee_id": rt.actors["maker"], "subject_namespace": "reference.people", "subject_token": "a" * 64, "purpose": "credit_review", "basis_artifact_id": basis["id"], "starts_at": (now - timedelta(minutes=1)).isoformat(), "expires_at": (now + timedelta(hours=1)).isoformat()}
    expect(rt.admin.post(base + "/sources/grants", json=grant), 201)
    query = {"operation_id": "source-b", "grant_id": "grant-b"}
    expect(rt.maker.post(base + "/sources/requests", json=query), 409)
    expect(rt.maker.post(base + "/sources/requests", json=query), 409)
    recovered = expect(rt.maker.post(base + "/operations/source-b/recover"))
    assert recovered["result"]["assessment"]["account_count"] is None
    assert post_count(root, "source-b") == effect_count(root, "source-b") == 1


def test_remote_actual_file_drift_is_rejected_and_never_becomes_verified(runtime):
    rt = runtime
    expect(upload(rt, "registry-tamper"), 201)
    saved = expect(rt.maker.get(prefix(rt) + "/registry/uploads/registry-tamper/readback"))
    file = rt.instances[0][1] / "assets" / saved["bundle_hash"] / saved["manifest"]["files"][0]["sha256"]
    original = file.read_bytes()
    try:
        file.write_bytes(original + b"tamper")
        response = expect(rt.admin.get(prefix(rt) + "/registry/uploads/registry-tamper/readback"), 503)
        assert response["detail"]["code"] == "remote_protocol_unavailable"
    finally:
        file.write_bytes(original)
    assert expect(rt.admin.get(prefix(rt) + "/registry/uploads/registry-tamper/readback"))["files_verified"] == saved["files_verified"]


def test_remote_commit_local_persistence_failure_is_visible_and_explicit_recovery(runtime, monkeypatch):
    rt = runtime
    service = rt.app.state.institution_adapters
    approved = promotion(rt, "local-persistence-failure")
    expect(rt.admin.post(prefix(rt) + "/deployments/install", json={"operation_id": "install-local-failure", "promotion_id": approved["id"], "raw_schema": SCHEMA, "probe_features": {"x1": 0.1, "x2": 0.2}}), 201)
    actual = service.governance.activate_promotion_request
    calls = []
    def fail_once(**kwargs):
        calls.append(kwargs["request_id"])
        raise GovernanceConflict("synthetic local persistence failure")
    monkeypatch.setattr(service.governance, "activate_promotion_request", fail_once)
    body = {"operation_id": "activate-local-failure", "install_operation_id": "install-local-failure"}
    failed = expect(rt.admin.post(prefix(rt) + "/deployments/activate", json=body), 409)
    assert failed["detail"]["code"] == "remote_committed_local_activation_pending"
    saved = expect(rt.admin.get(prefix(rt) + "/operations/activate-local-failure"))
    assert saved["status"] == "remote_committed_local_pending" and saved["result"]["local_governance"] is None
    assert len(calls) == 1  # GET neither re-sends nor silently writes local governance.
    monkeypatch.setattr(service.governance, "activate_promotion_request", actual)
    recovered = expect(rt.admin.post(prefix(rt) + "/operations/activate-local-failure/recover"))
    assert recovered["status"] == "completed" and recovered["result"]["local_governance"]["status"] == "activated"
    assert post_count(rt.instances[0][1], "activate-local-failure") == effect_count(rt.instances[0][1], "activate-local-failure") == 1


def test_native_committed_journal_tail_failure_recovers_without_new_native_or_remote_effect(runtime, monkeypatch):
    rt = runtime
    service = rt.app.state.institution_adapters
    approved = promotion(rt, "journal-tail-failure")
    expect(rt.admin.post(prefix(rt) + "/deployments/install", json={"operation_id": "install-tail-failure", "promotion_id": approved["id"], "raw_schema": SCHEMA, "probe_features": {"x1": 0.1, "x2": 0.2}}), 201)
    finalize = service.journal.finalize
    def fail(*args):
        raise OSError("synthetic durable tail unavailable")
    monkeypatch.setattr(service.journal, "finalize", fail)
    expect(rt.admin.post(prefix(rt) + "/deployments/activate", json={"operation_id": "activate-tail-failure", "install_operation_id": "install-tail-failure"}), 422)
    native_id = expect(rt.admin.get("/api/production-governance/environments/reference-a"))["active"]["id"]
    assert expect(rt.admin.get(prefix(rt) + "/operations/activate-tail-failure"))["status"] == "remote_committed_local_pending"
    monkeypatch.setattr(service.journal, "finalize", finalize)
    monkeypatch.setattr(service.governance, "activate_promotion_request", lambda **kwargs: pytest.fail("already committed native activation replayed"))
    recovered = expect(rt.admin.post(prefix(rt) + "/operations/activate-tail-failure/recover"))
    assert recovered["result"]["local_governance"]["deployment_id"] == native_id
    assert post_count(rt.instances[0][1], "activate-tail-failure") == effect_count(rt.instances[0][1], "activate-tail-failure") == 1


def test_corrupt_registered_model_refuses_before_any_network_or_pickle_load(runtime, monkeypatch):
    rt = runtime
    artifact = ModelingRepository(rt.settings.db_path).get_model_artifact(rt.artifact_id)
    path = rt.settings.tasks_dir / rt.task.id / "modeling_artifacts" / artifact.model_path
    data = path.read_bytes()
    monkeypatch.setattr("joblib.load", lambda *args, **kwargs: pytest.fail("registry deserialized native pickle"))
    try:
        path.write_bytes(data + b"tamper")
        response = expect(upload(rt, "corrupt-native"), 422)
        assert response["detail"]["code"] == "registered_model_native_closure_unavailable"
    finally:
        path.write_bytes(data)
    assert post_count(rt.instances[0][1], "corrupt-native") == effect_count(rt.instances[0][1], "corrupt-native") == 0


def test_peer_identity_and_local_credential_failures_precede_any_model_body_send(runtime):
    rt = runtime
    actual = rt.instances[0][0]
    wrong = actual.model_copy(update={"alias": "peer-wrong", "environment": "peer-wrong", "instance_id": "wrong-instance"})
    key = rt.root / "wrong-credential"
    key.write_text("invalid-reference-credential")
    key.chmod(0o600)
    bad_token = actual.model_copy(update={"alias": "token-wrong", "environment": "token-wrong", "credential_file": key})
    try:
        configure_institution_adapters(rt.app, [v[0] for v in rt.instances] + [wrong, bad_token])
        mismatch = expect(upload(rt, "peer-body-never-sent", "peer-wrong"), 409)
        assert mismatch["detail"]["code"] == "remote_instance_binding_mismatch"
        rejected = expect(upload(rt, "credential-body-never-sent", "token-wrong"), 503)
        assert rejected["detail"]["code"] == "remote_credential_rejected"
        assert "invalid-reference-credential" not in json.dumps(rejected)
        for identity in ("peer-body-never-sent", "credential-body-never-sent"):
            assert post_count(rt.instances[0][1], identity) == effect_count(rt.instances[0][1], identity) == 0
    finally:
        configure_institution_adapters(rt.app, [v[0] for v in rt.instances])

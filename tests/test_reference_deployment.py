from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.reference_decision.contracts import DecisionError, DecisionRequest
from marvis.reference_decision.deployment import ADAPTER_ID

import test_reference_decision as package_fixtures
from test_operations_api import _claim_role


@pytest.fixture
def packaged(tmp_path_factory):
    return package_fixtures.packaged.__wrapped__(tmp_path_factory)


@pytest.fixture
def governed(packaged):
    app = packaged[0]
    clients = [TestClient(app) for _ in range(3)]
    for client, role in zip(clients, ["maker", "checker", "admin"], strict=True):
        _claim_role(app, client, role)
    return app, *clients


def _promotion(governed, package_hash, *, slot="production"):
    _, maker, checker, admin = governed
    created = maker.post(
        "/api/production-governance/promotion-requests",
        json={
            "environment": "local-reference",
            "deployment_slot": slot,
            "strategy_id": "online-strategy",
            "strategy_version": 1,
            "decision_package_hash": package_hash,
            "reason": "local reference acceptance",
        },
    )
    assert created.status_code == 201, created.text
    promotion = created.json()
    for client in (checker, admin):
        result = client.post(
            f"/api/production-governance/promotion-requests/{promotion['id']}/approvals",
            json={"decision": "approve", "reason": "independent review"},
        )
        assert result.status_code == 200, result.text
    return result.json()


def _install_activate(governed, promotion):
    _, _, _, admin = governed
    installed = admin.post(
        "/api/reference-decision/installations",
        json={
            "promotion_id": promotion["id"],
            "probe_features": {"x1": 0.1, "x2": 0.2},
        },
    )
    assert installed.status_code == 201, installed.text
    receipt = installed.json()
    activated = admin.post(
        f"/api/production-governance/environments/local-reference/promotion-requests/{promotion['id']}/activate",
        json={
            "verifier_id": ADAPTER_ID,
            "receipt_id": receipt["activation_evidence"]["receipt_id"],
            "reason": "verified local package",
        },
    )
    assert activated.status_code == 201, activated.text
    return receipt, activated.json()


def _activate_baseline(packaged, governed):
    promotion = _promotion(governed, packaged[3])
    return _install_activate(governed, promotion)


def test_approved_install_is_real_idempotent_and_recovers_after_lost_response(
    packaged, governed, monkeypatch
):
    app, _, _, package_hash, _, _, _, _ = packaged
    promotion = _promotion(governed, package_hash)
    admin = governed[-1]
    install_payload = {
        "promotion_id": promotion["id"],
        "probe_features": {"x1": 0.1, "x2": 0.2},
    }
    first = admin.post("/api/reference-decision/installations", json=install_payload)
    assert first.status_code == 201, first.text
    assert first.json()["probe"]["action"]["reason_code"] == "LOW_PD"
    assert admin.get("/api/reference-decision/status").json()["state"] == "unavailable"
    fresh = create_app(app.state.settings)

    def cannot_repeat(*args, **kwargs):
        raise AssertionError("committed installation must be read back, not repeated")

    monkeypatch.setattr(fresh.state.reference_decision, "evaluate", cannot_repeat)
    assert (
        fresh.state.reference_deployment_adapter.install(promotion["id"], {})
        == first.json()
    )
    payload = {
        "verifier_id": ADAPTER_ID,
        "receipt_id": first.json()["activation_evidence"]["receipt_id"],
        "reason": "activate",
    }
    url = f"/api/production-governance/environments/local-reference/promotion-requests/{promotion['id']}/activate"
    activated = admin.post(url, json=payload)
    assert activated.status_code == 201, activated.text
    replay = admin.post(url, json=payload)
    assert replay.status_code == 201 and replay.json() == activated.json()
    assert (
        fresh.state.reference_decision.head()["deployment_id"] == activated.json()["id"]
    )


def test_application_http_decisions_are_pinned_logged_and_idempotent(
    packaged, governed
):
    _activate_baseline(packaged, governed)
    app, _, _, package_hash, _, _, _, _ = packaged
    admin = governed[-1]
    request = {
        "request_id": "application-1",
        "decision_node": "underwriting",
        "expected_package_hash": package_hash,
        "features": {"x1": 0.1, "x2": 0.2},
    }
    response = admin.post("/api/reference-decision/decisions", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "decided" and result["action"]["reason_code"] == "LOW_PD"
    assert result["package_hash"] == package_hash and result["output_hash"]
    assert result["execution_identity"] == "local_reference_worker.v1"
    assert (
        admin.post("/api/reference-decision/decisions", json=request).json() == result
    )
    request["features"]["x1"] = 0.9
    assert (
        admin.post("/api/reference-decision/decisions", json=request).status_code == 409
    )
    request["request_id"] = "forged-score"
    request["features"]["pd"] = 0.0
    assert (
        admin.post("/api/reference-decision/decisions", json=request).status_code == 422
    )
    fresh = create_app(app.state.settings)
    request["request_id"] = "application-1"
    request["features"] = {"x1": 0.1, "x2": 0.2}
    assert fresh.state.reference_decision.decide(DecisionRequest(**request)) == result


def test_shadow_switch_and_rollback_change_real_serving_head_without_resetting_decisions(
    packaged, governed
):
    _activate_baseline(packaged, governed)
    app, store, request, old_hash, _, _, _, _ = packaged
    original = DecisionRequest(
        request_id="application-1",
        decision_node="underwriting",
        expected_package_hash=old_hash,
        features={"x1": 0.1, "x2": 0.2},
    )
    original_result = app.state.reference_decision.decide(original)
    assert original_result["package_hash"] == old_hash
    newer, _ = store.build(
        request.model_copy(update={"failure_action": "reject"}), actor_id="test-maker"
    )
    shadow = _promotion(governed, newer, slot="shadow")
    _install_activate(governed, shadow)
    assert app.state.reference_decision.head("shadow")["package_hash"] == newer
    assert app.state.reference_decision.head()["package_hash"] == old_hash
    promotion = _promotion(governed, newer)
    _, active = _install_activate(governed, promotion)
    assert app.state.reference_decision.head()["package_hash"] == newer
    # Same request returns its original result even after a package switch.
    assert app.state.reference_decision.decide(original) == original_result
    admin = governed[-1]
    url = f"/api/production-governance/environments/local-reference/deployments/{active['id']}/rollback"
    payload = {
        "expected_active_deployment_id": active["id"],
        "reason": "local rollback drill",
    }
    rolled = admin.post(url, json=payload)
    assert rolled.status_code == 200, rolled.text
    assert app.state.reference_decision.head()["package_hash"] == old_hash
    assert admin.post(url, json=payload).json() == rolled.json()
    request = original.model_copy(update={"request_id": "after-rollback"})
    assert app.state.reference_decision.decide(request)["status"] == "decided"


def test_timeout_fallback_is_explicit_approved_and_durable(
    packaged, governed, monkeypatch
):
    _activate_baseline(packaged, governed)
    app, _, _, package_hash, _, _, _, _ = packaged

    def timeout(*args):
        raise DecisionError("scoring_timeout", 503)

    monkeypatch.setattr(app.state.reference_decision, "evaluate", timeout)
    request = DecisionRequest(
        request_id="timeout",
        decision_node="underwriting",
        expected_package_hash=package_hash,
        features={"x1": 0.1, "x2": 0.2},
    )
    response = app.state.reference_decision.decide(request)
    assert response["status"] == "fallback" and response["action"]["type"] == "review"
    assert response["score"] is None and response["error_code"] == "scoring_timeout"
    assert app.state.reference_decision.decide(request) == response


def test_package_binding_and_approval_are_required_before_install(packaged, governed):
    _, _, _, package_hash, _, _, _, _ = packaged
    _, maker, _, admin = governed
    payload = {
        "environment": "local-reference",
        "deployment_slot": "production",
        "strategy_id": "online-strategy",
        "strategy_version": 1,
        "reason": "must be bound",
    }
    assert (
        maker.post(
            "/api/production-governance/promotion-requests", json=payload
        ).status_code
        == 409
    )
    payload["decision_package_hash"] = package_hash
    created = maker.post("/api/production-governance/promotion-requests", json=payload)
    assert created.status_code == 201
    result = admin.post(
        "/api/reference-decision/installations",
        json={
            "promotion_id": created.json()["id"],
            "probe_features": {"x1": 0.1, "x2": 0.2},
        },
    )
    assert result.status_code == 409


def test_incompatible_schema_or_tampered_installed_package_cannot_activate(
    packaged, governed
):
    _activate_baseline(packaged, governed)
    app, store, request, _, _, _, _, _ = packaged
    incompatible = request.model_copy(update={"decision_node": "collections"})
    package_hash, manifest = store.build(incompatible, actor_id="test-maker")
    promotion = _promotion(governed, package_hash)
    admin = governed[-1]
    installed = admin.post(
        "/api/reference-decision/installations",
        json={
            "promotion_id": promotion["id"],
            "probe_features": {"x1": 0.1, "x2": 0.2},
        },
    )
    assert installed.status_code == 201
    url = f"/api/production-governance/environments/local-reference/promotion-requests/{promotion['id']}/activate"
    payload = {
        "verifier_id": ADAPTER_ID,
        "receipt_id": installed.json()["activation_evidence"]["receipt_id"],
        "reason": "incompatible",
    }
    assert admin.post(url, json=payload).status_code == 409
    target = store.root / package_hash / manifest["files"][0]["path"]
    target.write_bytes(b"changed installed package")
    assert admin.post(url, json=payload).status_code == 409


def test_governance_discovery_is_role_gated_paginated_and_uses_existing_heads(
    packaged, governed
):
    _activate_baseline(packaged, governed)
    app, _, _, package_hash, _, artifact, _, _ = packaged
    stranger = TestClient(app)
    endpoints = [
        "/api/reference-decision/packages",
        "/api/reference-decision/installations",
        "/api/production-governance/promotion-requests",
        "/api/production-governance/environments",
    ]
    for path in endpoints:
        assert stranger.get(path).status_code == 403
    admin = governed[-1]
    packages = admin.get(endpoints[0], params={"limit": 1}).json()
    assert packages["count"] >= 1 and len(packages["packages"]) == 1
    assert packages["packages"][0]["file_integrity"] == "verify_on_detail_read"
    assert (
        admin.get(endpoints[0], params={"task_id": "unrelated-task"}).json()["packages"]
        == []
    )
    assert admin.get(endpoints[0], params={"limit": 101}).status_code == 422
    requests = admin.get(endpoints[2], params={"environment": "local-reference"}).json()
    assert requests["count"] and all(
        item["environment"] == "local-reference" for item in requests["requests"]
    )
    installations = admin.get(endpoints[1]).json()
    assert installations["count"] and all(
        item["verifier_id"] == ADAPTER_ID for item in installations["installations"]
    )
    environments = admin.get(endpoints[3]).json()
    local = next(
        item
        for item in environments["environments"]
        if item["environment"] == "local-reference"
    )
    assert (
        local["active_deployment_id"]
        == app.state.reference_decision.head()["deployment_id"]
    )


def test_nonfinite_or_unserializable_request_is_a_controlled_error(packaged):
    app, _, _, package_hash, _, _, _, _ = packaged
    request = DecisionRequest(
        request_id="nonfinite",
        decision_node="underwriting",
        expected_package_hash=package_hash,
        features={"x1": float("nan"), "x2": 1},
    )
    with pytest.raises(DecisionError, match="invalid_feature_payload"):
        app.state.reference_decision.decide(request)

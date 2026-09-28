"""Ordered rejection and slot-local rollback remain governed and auditable."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from marvis.db_schema import connect
from marvis.production_governance.repository import ProductionGovernanceRepository
from marvis.production_governance.errors import GovernanceNotFound
from test_production_governance_api import (
    _activate,
    _activation_payload,
    _approved_promotion,
    _claim_role,
    _create_governed_app,
    _promotion_payload,
    _seed_strategy,
)


@pytest.fixture
def actors(tmp_path):
    app = _create_governed_app(tmp_path)
    clients, ids = {}, {}
    for role in ("maker", "checker", "admin"):
        clients[role] = TestClient(app)
        ids[role] = _claim_role(app, clients[role], display_name=role, role=role)["id"]
    _seed_strategy(app, "reviewed-strategy")
    return app, clients, ids


def create(clients):
    result = clients["maker"].post(
        "/api/production-governance/promotion-requests",
        json=_promotion_payload("reviewed-strategy"),
    )
    assert result.status_code == 201, result.text
    return result.json()


def review(
    client,
    promotion,
    decision="reject",
    reason="Independent rejection: missing policy evidence",
):
    return client.post(
        f"/api/production-governance/promotion-requests/{promotion['id']}/approvals",
        json={"decision": decision, "reason": reason},
    )


@pytest.mark.parametrize("stage", ["checker", "admin"])
def test_reject_is_ordered_terminal_audited_and_never_expires_into_another_state(
    actors, stage
):
    app, clients, ids = actors
    promotion = create(clients)
    if stage == "admin":
        assert review(clients["checker"], promotion, "approve").status_code == 200
    rejected = review(clients[stage], promotion)
    assert rejected.status_code == 200, rejected.text
    result = rejected.json()
    assert result["status"] == "rejected"
    assert len(result["approvals"]) == (0 if stage == "checker" else 1)
    assert result["rejection"]["principal_id"] == ids[stage]
    assert result["rejection"]["stage"] == stage
    assert (
        result["rejection"]["reason"]
        == "Independent rejection: missing policy evidence"
    )
    assert len(result["rejection"]["event_hash"]) == 64
    for decision in ("approve", "reject"):
        for role in ("checker", "admin"):
            assert review(clients[role], promotion, decision).status_code == 409
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE production_promotion_requests SET expires_at=? WHERE id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), promotion["id"]),
        )
    assert review(clients[stage], promotion).status_code == 409
    activate = clients["admin"].post(
        f"/api/production-governance/environments/production/promotion-requests/{promotion['id']}/activate",
        json=_activation_payload(app, promotion),
    )
    assert activate.status_code == 409
    stored = (
        clients["admin"]
        .get(f"/api/production-governance/promotion-requests/{promotion['id']}")
        .json()
    )
    assert stored["status"] == "rejected" and stored["rejection"] == result["rejection"]
    audit = ProductionGovernanceRepository(app.state.settings.db_path).export_audit()
    events = [e for e in audit["events"] if e["target_id"] == promotion["id"]]
    assert len([e for e in events if e["event_type"] == "promotion.rejected"]) == 1
    assert not any(e["event_type"] == "promotion.expired" for e in events)
    assert (
        clients["admin"]
        .get("/api/production-governance/environments/production")
        .json()["active"]
        is None
    )


def test_rejection_obeys_role_order_reason_expiry_and_revocation(actors):
    app, clients, ids = actors
    promotion = create(clients)
    assert review(clients["maker"], promotion).status_code == 403
    assert review(clients["admin"], promotion).status_code == 409
    assert review(clients["checker"], promotion, reason=" \n ").status_code == 422
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE production_principals SET status='revoked' WHERE local_principal_id=?",
            (ids["checker"],),
        )
    assert review(clients["checker"], promotion).status_code == 403
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE production_principals SET status='active' WHERE local_principal_id=?",
            (ids["checker"],),
        )
        conn.execute(
            "UPDATE production_promotion_requests SET expires_at=? WHERE id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), promotion["id"]),
        )
    assert review(clients["checker"], promotion).status_code == 409
    current = (
        clients["maker"]
        .get(f"/api/production-governance/promotion-requests/{promotion['id']}")
        .json()
    )
    assert (
        current["status"] == "expired"
        and current["approvals"] == []
        and "rejection" not in current
    )


def test_rejection_rechecks_local_principal_inside_transaction(actors):
    app, clients, ids = actors
    promotion = create(clients)
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE local_principals SET status='revoked' WHERE id=?", (ids["checker"],)
        )
    with pytest.raises(GovernanceNotFound):
        ProductionGovernanceRepository(
            app.state.settings.db_path
        ).review_promotion_request(
            request_id=promotion["id"],
            actor_principal_id=ids["checker"],
            decision="reject",
            reason="review",
        )


@pytest.mark.parametrize("slot", ["production", "shadow"])
def test_slot_rollback_cas_never_changes_other_head_and_audits_exact_predecessor(
    actors, slot
):
    app, clients, ids = actors

    def deploy(selected_slot, suffix):
        p = _approved_promotion(
            clients["maker"],
            clients["checker"],
            clients["admin"],
            strategy_id="reviewed-strategy",
            deployment_slot=selected_slot,
        )
        return _activate(
            clients["admin"], p, environment="production", ref_suffix=suffix
        )

    other_slot = "shadow" if slot == "production" else "production"
    first, other, second = (
        deploy(slot, "first"),
        deploy(other_slot, "other"),
        deploy(slot, "second"),
    )
    endpoint = f"/api/production-governance/environments/production/deployments/{second['id']}/rollback"
    payload = {
        "deployment_slot": slot,
        "expected_active_deployment_id": second["id"],
        "reason": "restore verified predecessor",
    }
    assert clients["checker"].post(endpoint, json=payload).status_code == 403
    assert (
        clients["admin"]
        .post(endpoint, json={**payload, "deployment_slot": other_slot})
        .status_code
        == 409
    )
    assert (
        clients["admin"]
        .post(endpoint, json={**payload, "expected_active_deployment_id": first["id"]})
        .status_code
        == 409
    )
    stale = endpoint.replace(second["id"], first["id"])
    assert (
        clients["admin"]
        .post(stale, json={**payload, "expected_active_deployment_id": first["id"]})
        .status_code
        == 409
    )
    rolled = clients["admin"].post(endpoint, json=payload)
    assert rolled.status_code == 200, rolled.text
    assert rolled.json()["deployment_slot"] == slot
    assert rolled.json()["restored_deployment"]["id"] == first["id"]
    assert rolled.json()["restored_deployment"]["status"] == (
        "shadow" if slot == "shadow" else "active"
    )
    heads = (
        clients["admin"]
        .get("/api/production-governance/environments/production")
        .json()
    )
    assert heads["shadow" if slot == "shadow" else "active"]["id"] == first["id"]
    assert heads["shadow" if other_slot == "shadow" else "active"]["id"] == other["id"]
    audit = ProductionGovernanceRepository(app.state.settings.db_path).export_audit()
    event = [e for e in audit["events"] if e["event_type"] == "deployment.rolled_back"][
        -1
    ]
    assert event["payload"]["deployment_slot"] == slot
    assert event["payload"]["restored_deployment_id"] == first["id"]


def test_concurrent_approve_and_reject_consume_checker_stage_once(actors):
    from concurrent.futures import ThreadPoolExecutor
    from marvis.production_governance.errors import GovernanceConflict

    app, clients, ids = actors
    promotion = create(clients)
    repo = ProductionGovernanceRepository(app.state.settings.db_path)

    def decide(decision):
        try:
            return repo.review_promotion_request(
                request_id=promotion["id"],
                actor_principal_id=ids["checker"],
                decision=decision,
                reason="concurrent review",
            )["status"]
        except GovernanceConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decide, ["approve", "reject"]))
    assert results.count("conflict") == 1
    current = repo.get_promotion_request(promotion["id"])
    assert current["status"] in {"awaiting_admin", "rejected"}
    events = [
        e
        for e in repo.export_audit()["events"]
        if e["target_id"] == promotion["id"]
        and e["event_type"] in {"promotion.approved", "promotion.rejected"}
    ]
    assert len(events) == 1

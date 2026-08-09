from __future__ import annotations

import hashlib
import json
import time

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.db_schema import connect
from marvis.production_governance.evidence import (
    ActivationVerificationContext,
    VerifiedActivationEvidence,
)
from marvis.production_governance.errors import GovernanceConflict


GOVERNANCE_ADMIN_HEADER = "x-marvis-governance-admin"
VERIFIER_ID = "test-verifier"


class _ReceiptVerifier:
    def __init__(self) -> None:
        self.receipts: dict[str, VerifiedActivationEvidence] = {}

    def add(self, evidence: VerifiedActivationEvidence) -> None:
        self.receipts[evidence.receipt_id] = evidence

    def verify(self, *, receipt_id, context):
        try:
            return self.receipts[receipt_id]
        except KeyError as exc:
            raise GovernanceConflict("activation receipt not found") from exc


def _create_governed_app(workspace):
    verifier = _ReceiptVerifier()
    return create_app(
        workspace,
        production_activation_verifiers={VERIFIER_ID: verifier},
    )


def _claim_role(
    app,
    client: TestClient,
    *,
    display_name: str,
    role: str,
) -> dict:
    response = client.post(
        "/api/production-governance/principals/claim",
        headers={GOVERNANCE_ADMIN_HEADER: app.state.plugin_admin_token},
        json={"display_name": display_name, "role": role},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _seed_strategy(app, strategy_id: str, *, version: int = 1) -> dict:
    StrategyRepository(app.state.settings.db_path).create_strategy(
        "governance-task",
        Strategy(
            id=strategy_id,
            strategy_type="approval",
            rules=(
                StrategyRule(
                    condition="score >= 700",
                    decision="approve",
                    value=True,
                ),
            ),
            score_col="score",
            default_decision="reject",
            description=f"governed strategy {strategy_id}",
        ),
    )
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            """
            UPDATE strategies
               SET version = ?, status = 'adopted', asset_status = 'adopted_local'
             WHERE id = ?
            """,
            (version, strategy_id),
        )
        row = conn.execute(
            "SELECT version, asset_status, dsl_content_hash FROM strategies WHERE id = ?",
            (strategy_id,),
        ).fetchone()
    return dict(row)


def _promotion_payload(strategy_id: str, *, environment: str = "production") -> dict:
    return {
        "environment": environment,
        "deployment_slot": "production",
        "strategy_id": strategy_id,
        "strategy_version": 1,
        "reason": "reviewed release candidate",
        "expires_in_seconds": 300,
    }


def _activation_payload(
    app,
    promotion: dict,
    *,
    environment: str | None = None,
    manifest_hash: str | None = None,
    ref_suffix: str = "release-1",
    verified_at: str = "2026-08-01T00:00:00+00:00",
) -> dict:
    receipt_id = f"receipt-{promotion['id']}-{ref_suffix}"
    context = ActivationVerificationContext(
        promotion_request_id=promotion["id"],
        environment=environment or promotion["environment"],
        deployment_slot=promotion["deployment_slot"],
        strategy_id=promotion["strategy_id"],
        strategy_version=promotion["strategy_version"],
        strategy_content_hash=promotion["strategy_content_hash"],
        manifest_hash=manifest_hash or promotion["manifest_hash"],
    )
    evidence = VerifiedActivationEvidence.issue(
        verifier_id=VERIFIER_ID,
        receipt_id=receipt_id,
        context=context,
        external_deployment_ref=f"local-adapter://deployment/{ref_suffix}",
        health_evidence_ref=f"local-evidence://health/{ref_suffix}",
        health_status="healthy",
        verified_at=verified_at,
    )
    app.state.production_activation_verifiers[VERIFIER_ID].add(evidence)
    return {
        "verifier_id": VERIFIER_ID,
        "receipt_id": receipt_id,
        "reason": "health evidence reviewed",
    }


def _approved_promotion(
    maker_client: TestClient,
    checker_client: TestClient,
    admin_client: TestClient,
    *,
    strategy_id: str,
    environment: str = "production",
    deployment_slot: str = "production",
) -> dict:
    payload = _promotion_payload(strategy_id, environment=environment)
    payload["deployment_slot"] = deployment_slot
    created = maker_client.post(
        "/api/production-governance/promotion-requests",
        json=payload,
    )
    assert created.status_code == 201, created.text
    request_id = created.json()["id"]
    checked = checker_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "checker review"},
    )
    assert checked.status_code == 200, checked.text
    approved = admin_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "admin review"},
    )
    assert approved.status_code == 200, approved.text
    return approved.json()


def _activate(
    admin_client: TestClient,
    promotion: dict,
    *,
    environment: str,
    ref_suffix: str,
) -> dict:
    payload = _activation_payload(
        admin_client.app,
        promotion,
        environment=environment,
        ref_suffix=ref_suffix,
    )
    response = admin_client.post(
        f"/api/production-governance/environments/{environment}/"
        f"promotion-requests/{promotion['id']}/activate",
        json=payload,
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_server_owned_session_identity_and_roles_cannot_be_spoofed_in_body(tmp_path):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)

    missing_admin = maker_client.post(
        "/api/production-governance/principals/claim",
        json={"display_name": "Maker One", "role": "maker"},
    )
    assert missing_admin.status_code == 403

    maker = _claim_role(
        app,
        maker_client,
        display_name="Maker One",
        role="maker",
    )
    assert maker["role"] == "maker"
    assert "session_token" not in maker
    assert "session_token_hash" not in maker

    current = maker_client.get("/api/production-governance/me")
    assert current.status_code == 200, current.text
    assert current.json() == maker


def test_maker_checker_admin_approval_is_ordered_one_shot_and_not_spoofable(tmp_path):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    maker = _claim_role(app, maker_client, display_name="Maker", role="maker")
    checker = _claim_role(app, checker_client, display_name="Checker", role="checker")
    admin = _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-approval-1")

    spoofed_payload = {
        **_promotion_payload("strategy-approval-1"),
        "principal_id": checker["id"],
        "role": "admin",
    }
    spoofed = maker_client.post(
        "/api/production-governance/promotion-requests",
        json=spoofed_payload,
    )
    assert spoofed.status_code == 422

    wrong_creator = checker_client.post(
        "/api/production-governance/promotion-requests",
        json=_promotion_payload("strategy-approval-1"),
    )
    assert wrong_creator.status_code == 403

    created = maker_client.post(
        "/api/production-governance/promotion-requests",
        json=_promotion_payload("strategy-approval-1"),
    )
    assert created.status_code == 201, created.text
    promotion = created.json()
    assert promotion["status"] == "pending_checker"
    assert promotion["maker_principal_id"] == maker["id"]
    assert len(promotion["manifest_hash"]) == 64
    assert promotion["manifest_hash"] != "a" * 64
    request_id = promotion["id"]

    maker_self_approval = maker_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "self approval"},
    )
    assert maker_self_approval.status_code == 403

    admin_out_of_order = admin_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "skip checker"},
    )
    assert admin_out_of_order.status_code == 409

    checker_approval = checker_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "independent review complete"},
    )
    assert checker_approval.status_code == 200, checker_approval.text
    assert checker_approval.json()["status"] == "awaiting_admin"

    replayed = checker_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "repeat approval"},
    )
    assert replayed.status_code == 409

    admin_approval = admin_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "final ordered approval"},
    )
    assert admin_approval.status_code == 200, admin_approval.text
    approved = admin_approval.json()
    assert approved["status"] == "approved"
    assert approved["approvals"] == [
        {"stage": "checker", "principal_id": checker["id"], "role": "checker"},
        {"stage": "admin", "principal_id": admin["id"], "role": "admin"},
    ]


def test_expired_promotion_request_fails_closed_without_an_approval(tmp_path):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _seed_strategy(app, "strategy-expiry")
    payload = {**_promotion_payload("strategy-expiry"), "expires_in_seconds": 1}
    created = maker_client.post(
        "/api/production-governance/promotion-requests",
        json=payload,
    )
    assert created.status_code == 201, created.text
    request_id = created.json()["id"]

    time.sleep(1.05)
    expired = checker_client.post(
        f"/api/production-governance/promotion-requests/{request_id}/approvals",
        json={"decision": "approve", "reason": "too late"},
    )
    assert expired.status_code == 409

    current = maker_client.get(
        f"/api/production-governance/promotion-requests/{request_id}"
    )
    assert current.status_code == 200, current.text
    assert current.json()["status"] == "expired"
    assert current.json()["approvals"] == []


def test_activation_requires_health_evidence_and_rejects_cross_environment_or_hash_drift(
    tmp_path,
):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-guarded-activation")
    approved = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-guarded-activation",
    )
    request_id = approved["id"]

    fabricated_body_evidence = admin_client.post(
        f"/api/production-governance/environments/production/promotion-requests/{request_id}/activate",
        json={
            "external_deployment_ref": "local-adapter://deployment/release-1",
            "health_evidence_ref": "local-evidence://health/release-1",
            "health_status": "healthy",
            "observed_manifest_hash": approved["manifest_hash"],
            "reason": "caller claims this is healthy",
        },
    )
    assert fabricated_body_evidence.status_code == 422

    unallowlisted_verifier = admin_client.post(
        f"/api/production-governance/environments/production/promotion-requests/{request_id}/activate",
        json={
            "verifier_id": "not-installed",
            "receipt_id": "fabricated-receipt",
            "reason": "caller claims this receipt exists",
        },
    )
    assert unallowlisted_verifier.status_code == 409

    missing_health_ref = admin_client.post(
        f"/api/production-governance/environments/production/promotion-requests/{request_id}/activate",
        json={
            key: value
            for key, value in _activation_payload(app, approved).items()
            if key != "receipt_id"
        },
    )
    assert missing_health_ref.status_code == 422

    cross_environment = admin_client.post(
        f"/api/production-governance/environments/staging/promotion-requests/{request_id}/activate",
        json=_activation_payload(app, approved),
    )
    assert cross_environment.status_code == 409

    hash_drift = admin_client.post(
        f"/api/production-governance/environments/production/promotion-requests/{request_id}/activate",
        json=_activation_payload(app, approved, manifest_hash="b" * 64),
    )
    assert hash_drift.status_code == 409

    environment = admin_client.get(
        "/api/production-governance/environments/production"
    )
    assert environment.status_code == 200, environment.text
    assert environment.json()["active"] is None
    assert environment.json()["shadow"] is None


def test_activation_evidence_verified_at_must_be_strict_utc_iso8601(tmp_path):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-evidence-time")
    approved = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-evidence-time",
    )

    invalid_timestamps = (
        "2026-08-01T08:00:00+08:00",
        "2026-08-01T00:00:00",
        "2026-08-01 00:00:00+00:00",
        "2026-02-30T00:00:00+00:00",
    )
    for index, verified_at in enumerate(invalid_timestamps):
        rejected = admin_client.post(
            "/api/production-governance/environments/production/"
            f"promotion-requests/{approved['id']}/activate",
            json=_activation_payload(
                app,
                approved,
                ref_suffix=f"invalid-time-{index}",
                verified_at=verified_at,
            ),
        )
        assert rejected.status_code == 409
    environment = admin_client.get(
        "/api/production-governance/environments/production"
    ).json()
    assert environment["active"] is None
    assert environment["shadow"] is None

    # Timestamp freshness belongs to the allowlisted verifier.  This layer
    # authenticates the receipt binding and requires only a strict UTC format.
    accepted = admin_client.post(
        "/api/production-governance/environments/production/"
        f"promotion-requests/{approved['id']}/activate",
        json=_activation_payload(
            app,
            approved,
            ref_suffix="strict-z-time",
            verified_at="2020-01-01T00:00:00Z",
        ),
    )
    assert accepted.status_code == 201, accepted.text


def test_strategy_hash_drift_after_approval_fails_closed(tmp_path):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-content-drift")
    approved = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-content-drift",
    )
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE strategies SET dsl_content_hash = ? WHERE id = ?",
            ("c" * 64, "strategy-content-drift"),
        )

    drifted = admin_client.post(
        "/api/production-governance/environments/production/"
        f"promotion-requests/{approved['id']}/activate",
        json=_activation_payload(app, approved),
    )
    assert drifted.status_code == 409
    environment = admin_client.get(
        "/api/production-governance/environments/production"
    ).json()
    assert environment["active"] is None
    assert environment["shadow"] is None


def test_strategy_dsl_drift_with_stale_cached_hash_after_approval_fails_closed(
    tmp_path,
):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-dsl-drift")
    approved = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-dsl-drift",
    )
    with connect(app.state.settings.db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json, dsl_content_hash FROM strategies WHERE id = ?",
            ("strategy-dsl-drift",),
        ).fetchone()
        assert row is not None
        cached_hash = str(row["dsl_content_hash"])
        tampered_dsl = json.loads(str(row["dsl_json"]))
        tampered_dsl["metadata"]["description"] = "tampered after approval"
        conn.execute(
            "UPDATE strategies SET dsl_json = ? WHERE id = ?",
            (
                json.dumps(tampered_dsl, ensure_ascii=False),
                "strategy-dsl-drift",
            ),
        )
        persisted = conn.execute(
            "SELECT dsl_content_hash FROM strategies WHERE id = ?",
            ("strategy-dsl-drift",),
        ).fetchone()
        assert persisted is not None
        assert str(persisted["dsl_content_hash"]) == cached_hash

    drifted = admin_client.post(
        "/api/production-governance/environments/production/"
        f"promotion-requests/{approved['id']}/activate",
        json=_activation_payload(app, approved),
    )
    assert drifted.status_code == 409
    environment = admin_client.get(
        "/api/production-governance/environments/production"
    ).json()
    assert environment["active"] is None
    assert environment["shadow"] is None


def test_target_environment_activation_keeps_asset_shadow_and_other_environment_separate(
    tmp_path,
):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-active")
    _seed_strategy(app, "strategy-shadow")

    production_request = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-active",
        environment="production",
        deployment_slot="production",
    )
    activated = admin_client.post(
        "/api/production-governance/environments/production/"
        f"promotion-requests/{production_request['id']}/activate",
        json=_activation_payload(app, production_request),
    )
    assert activated.status_code == 201, activated.text
    active_record = activated.json()
    assert active_record["status"] == "active"
    assert active_record["strategy_id"] == "strategy-active"
    assert len(active_record["activation_evidence_id"]) == 64

    replayed = admin_client.post(
        "/api/production-governance/environments/production/"
        f"promotion-requests/{production_request['id']}/activate",
        json=_activation_payload(app, production_request),
    )
    assert replayed.status_code == 409

    shadow_request = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-shadow",
        environment="production",
        deployment_slot="shadow",
    )
    shadow_payload = _activation_payload(
        app,
        shadow_request,
        ref_suffix="shadow-1",
    )
    shadowed = admin_client.post(
        "/api/production-governance/environments/production/"
        f"promotion-requests/{shadow_request['id']}/activate",
        json=shadow_payload,
    )
    assert shadowed.status_code == 201, shadowed.text
    assert shadowed.json()["status"] == "shadow"

    production = admin_client.get(
        "/api/production-governance/environments/production"
    )
    assert production.status_code == 200, production.text
    assert production.json()["active"]["id"] == active_record["id"]
    assert production.json()["shadow"]["id"] == shadowed.json()["id"]

    staging = admin_client.get("/api/production-governance/environments/staging")
    assert staging.status_code == 200, staging.text
    assert staging.json()["active"] is None
    assert staging.json()["shadow"] is None

    asset = admin_client.get(
        "/api/production-governance/strategies/strategy-active/deployments"
    )
    assert asset.status_code == 200, asset.text
    assert asset.json()["asset_status"] == "adopted_local"
    assert [item["id"] for item in asset.json()["deployments"]] == [
        active_record["id"]
    ]


def test_single_environment_rollback_restores_exact_predecessor_and_exports_hash_chain(
    tmp_path,
):
    app = _create_governed_app(tmp_path)
    maker_client = TestClient(app)
    checker_client = TestClient(app)
    admin_client = TestClient(app)
    _claim_role(app, maker_client, display_name="Maker", role="maker")
    _claim_role(app, checker_client, display_name="Checker", role="checker")
    _claim_role(app, admin_client, display_name="Admin", role="admin")
    _seed_strategy(app, "strategy-predecessor")
    _seed_strategy(app, "strategy-successor")

    predecessor_request = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-predecessor",
        environment="production",
    )
    predecessor = _activate(
        admin_client,
        predecessor_request,
        environment="production",
        ref_suffix="production-predecessor",
    )
    other_environment_request = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-predecessor",
        environment="staging",
    )
    other_environment = _activate(
        admin_client,
        other_environment_request,
        environment="staging",
        ref_suffix="staging-predecessor",
    )
    successor_request = _approved_promotion(
        maker_client,
        checker_client,
        admin_client,
        strategy_id="strategy-successor",
        environment="production",
    )
    successor = _activate(
        admin_client,
        successor_request,
        environment="production",
        ref_suffix="production-successor",
    )
    assert successor["predecessor_deployment_id"] == predecessor["id"]

    rolled_back = admin_client.post(
        "/api/production-governance/environments/production/"
        f"deployments/{successor['id']}/rollback",
        json={
            "reason": "health regression after activation",
            "expected_active_deployment_id": successor["id"],
        },
    )
    assert rolled_back.status_code == 200, rolled_back.text
    rollback = rolled_back.json()
    assert rollback["rolled_back_deployment"]["status"] == "rolled_back"
    assert rollback["restored_deployment"]["id"] == predecessor["id"]

    production = admin_client.get(
        "/api/production-governance/environments/production"
    ).json()
    for field in (
        "id",
        "strategy_id",
        "strategy_version",
        "strategy_content_hash",
        "manifest_hash",
        "external_deployment_ref",
        "health_evidence_ref",
    ):
        assert production["active"][field] == predecessor[field]
    assert production["active"]["status"] == "active"

    staging = admin_client.get(
        "/api/production-governance/environments/staging"
    ).json()
    assert staging["active"]["id"] == other_environment["id"]

    replayed = admin_client.post(
        "/api/production-governance/environments/production/"
        f"deployments/{successor['id']}/rollback",
        json={
            "reason": "replay",
            "expected_active_deployment_id": successor["id"],
        },
    )
    assert replayed.status_code == 409

    exported = admin_client.get(
        "/api/production-governance/audit/export"
    )
    assert exported.status_code == 200, exported.text
    audit = exported.json()
    assert audit["schema_version"] == "production-governance.audit.v1"
    assert audit["verification"]["valid"] is True
    assert audit["verification"]["event_count"] == len(audit["events"])
    previous_hash = "0" * 64
    for event in audit["events"]:
        assert event["previous_event_hash"] == previous_hash
        hash_input = {key: value for key, value in event.items() if key != "event_hash"}
        expected_hash = hashlib.sha256(
            json.dumps(
                hash_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert event["event_hash"] == expected_hash
        previous_hash = event["event_hash"]
    assert audit["verification"]["chain_head"] == previous_hash
    assert any(
        event["event_type"] == "deployment.rolled_back"
        for event in audit["events"]
    )
    approval_reasons = {
        event["payload"].get("reason")
        for event in audit["events"]
        if event["event_type"] == "promotion.approved"
    }
    assert {"checker review", "admin review"} <= approval_reasons
    serialized = json.dumps(audit, ensure_ascii=False).lower()
    assert "session_token" not in serialized
    assert app.state.plugin_admin_token.lower() not in serialized

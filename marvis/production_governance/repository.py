from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from marvis.db_schema import connect
from marvis.production_governance.errors import (
    GovernanceConflict,
    GovernanceForbidden,
    GovernanceNotFound,
)
from marvis.production_governance.evidence import (
    ActivationVerificationContext,
    VerifiedActivationEvidence,
    require_evidence_matches_context,
)


_ROLES = frozenset({"maker", "checker", "admin"})
_GENESIS_EVENT_HASH = "0" * 64


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class ProductionGovernanceRepository:
    """SQLite authority for local production roles and its hash-chained audit."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def claim_principal(
        self,
        *,
        local_principal_id: str,
        display_name: str,
        role: str,
    ) -> dict[str, Any]:
        principal_id = _required_text(local_principal_id, "local_principal_id")
        display = _required_text(display_name, "display_name")
        normalized_role = str(role).strip().lower()
        if normalized_role not in _ROLES:
            raise ValueError(f"unsupported production role: {role}")
        now = datetime.now(UTC).isoformat()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            local = conn.execute(
                "SELECT status FROM local_principals WHERE id = ?",
                (principal_id,),
            ).fetchone()
            if local is None or str(local["status"]) != "active":
                raise GovernanceNotFound("active server session principal not found")
            existing = conn.execute(
                "SELECT * FROM production_principals WHERE local_principal_id = ?",
                (principal_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["display_name"]) != display
                    or str(existing["role"]) != normalized_role
                ):
                    raise GovernanceConflict(
                        "the server session principal already has an immutable production role"
                    )
                return _principal_from_row(existing)
            conn.execute(
                """
                INSERT INTO production_principals(
                    local_principal_id, display_name, role, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?)
                """,
                (principal_id, display, normalized_role, now, now),
            )
            _append_event(
                conn,
                event_type="principal.claimed",
                actor_principal_id=principal_id,
                actor_role="workspace_admin",
                target_type="principal",
                target_id=principal_id,
                environment=None,
                payload={"display_name": display, "role": normalized_role},
                at=now,
            )
            row = conn.execute(
                "SELECT * FROM production_principals WHERE local_principal_id = ?",
                (principal_id,),
            ).fetchone()
        return _principal_from_row(row)

    def get_principal(self, local_principal_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM production_principals WHERE local_principal_id = ?",
                (str(local_principal_id),),
            ).fetchone()
        if row is None or str(row["status"]) != "active":
            raise GovernanceNotFound("production principal is not registered")
        return _principal_from_row(row)

    def create_promotion_request(
        self,
        *,
        actor_principal_id: str,
        environment: str,
        deployment_slot: str,
        strategy_id: str,
        strategy_version: int,
        reason: str,
        expires_in_seconds: int,
    ) -> dict[str, Any]:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=int(expires_in_seconds))).isoformat()
        request_id = uuid.uuid4().hex
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            actor = _require_principal_tx(conn, actor_principal_id)
            if actor["role"] != "maker":
                raise GovernanceForbidden("only a maker can create a promotion request")
            strategy = _strategy_binding_tx(
                conn,
                strategy_id=strategy_id,
                strategy_version=strategy_version,
            )
            manifest_hash = _register_deployment_manifest_tx(
                conn,
                environment=environment,
                deployment_slot=deployment_slot,
                strategy=strategy,
                created_at=now,
            )
            head = conn.execute(
                "SELECT * FROM production_environment_heads WHERE environment = ?",
                (environment,),
            ).fetchone()
            expected_current = None
            if head is not None:
                column = (
                    "active_deployment_id"
                    if deployment_slot == "production"
                    else "shadow_deployment_id"
                )
                expected_current = head[column]
            conn.execute(
                """
                INSERT INTO production_promotion_requests(
                    id, environment, deployment_slot, strategy_id,
                    strategy_version, strategy_content_hash,
                    asset_status_snapshot, manifest_hash,
                    expected_current_deployment_id, maker_principal_id,
                    reason, status, expires_at, promoted_deployment_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_checker', ?, NULL, ?, ?)
                """,
                (
                    request_id,
                    environment,
                    deployment_slot,
                    strategy["id"],
                    strategy["version"],
                    strategy["strategy_content_hash"],
                    strategy["asset_status"],
                    manifest_hash,
                    expected_current,
                    actor["id"],
                    reason,
                    expires_at,
                    now,
                    now,
                ),
            )
            _append_event(
                conn,
                event_type="promotion.requested",
                actor_principal_id=actor["id"],
                actor_role=actor["role"],
                target_type="promotion_request",
                target_id=request_id,
                environment=environment,
                payload={
                    "deployment_slot": deployment_slot,
                    "strategy_id": strategy["id"],
                    "strategy_version": strategy["version"],
                    "strategy_content_hash": strategy["strategy_content_hash"],
                    "manifest_hash": manifest_hash,
                    "expected_current_deployment_id": expected_current,
                    "reason": reason,
                },
                at=now,
            )
            row = conn.execute(
                "SELECT * FROM production_promotion_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            result = _promotion_from_row(conn, row)
        return result

    def approve_promotion_request(
        self,
        *,
        request_id: str,
        actor_principal_id: str,
        reason: str,
    ) -> dict[str, Any]:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        expired = False
        result: dict[str, Any] | None = None
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            actor = _require_principal_tx(conn, actor_principal_id)
            row = conn.execute(
                "SELECT * FROM production_promotion_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise GovernanceNotFound("promotion request not found")
            if _parse_time(str(row["expires_at"])) <= now_dt:
                if str(row["status"]) not in {"expired", "promoted"}:
                    conn.execute(
                        "UPDATE production_promotion_requests SET status = 'expired', updated_at = ? WHERE id = ?",
                        (now, request_id),
                    )
                    _append_event(
                        conn,
                        event_type="promotion.expired",
                        actor_principal_id=actor["id"],
                        actor_role=actor["role"],
                        target_type="promotion_request",
                        target_id=request_id,
                        environment=str(row["environment"]),
                        payload={"previous_status": str(row["status"])},
                        at=now,
                    )
                expired = True
            else:
                status = str(row["status"])
                maker_id = str(row["maker_principal_id"])
                if actor["id"] == maker_id:
                    raise GovernanceForbidden("maker cannot approve their own request")
                if status == "pending_checker":
                    if actor["role"] == "admin":
                        raise GovernanceConflict(
                            "checker approval is required before admin approval"
                        )
                    if actor["role"] != "checker":
                        raise GovernanceForbidden(
                            "only a checker can perform the first approval"
                        )
                    stage = "checker"
                    next_status = "awaiting_admin"
                elif status == "awaiting_admin":
                    if actor["role"] == "checker":
                        raise GovernanceConflict(
                            "checker approval was already consumed; admin approval is next"
                        )
                    if actor["role"] != "admin":
                        raise GovernanceForbidden(
                            "only an admin can perform the final approval"
                        )
                    checker_row = conn.execute(
                        """
                        SELECT principal_id
                          FROM production_promotion_approvals
                         WHERE request_id = ? AND stage = 'checker'
                        """,
                        (request_id,),
                    ).fetchone()
                    if checker_row is None:
                        raise GovernanceConflict("checker approval record is missing")
                    if actor["id"] == str(checker_row["principal_id"]):
                        raise GovernanceForbidden(
                            "checker and admin approval must use distinct principals"
                        )
                    stage = "admin"
                    next_status = "approved"
                else:
                    raise GovernanceConflict(
                        f"promotion request cannot be approved from status {status}"
                    )
                approval_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO production_promotion_approvals(
                        id, request_id, stage, principal_id, role, reason, approved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        approval_id,
                        request_id,
                        stage,
                        actor["id"],
                        actor["role"],
                        reason,
                        now,
                    ),
                )
                conn.execute(
                    """
                    UPDATE production_promotion_requests
                       SET status = ?, updated_at = ?
                     WHERE id = ? AND status = ?
                    """,
                    (next_status, now, request_id, status),
                )
                _append_event(
                    conn,
                    event_type="promotion.approved",
                    actor_principal_id=actor["id"],
                    actor_role=actor["role"],
                    target_type="promotion_request",
                    target_id=request_id,
                    environment=str(row["environment"]),
                    payload={
                        "approval_stage": stage,
                        "result_status": next_status,
                        "reason": reason,
                    },
                    at=now,
                )
                updated = conn.execute(
                    "SELECT * FROM production_promotion_requests WHERE id = ?",
                    (request_id,),
                ).fetchone()
                result = _promotion_from_row(conn, updated)
        if expired:
            raise GovernanceConflict("promotion request is expired")
        assert result is not None
        return result

    def get_promotion_request(self, request_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM production_promotion_requests WHERE id = ?",
                (str(request_id),),
            ).fetchone()
            if row is None:
                raise GovernanceNotFound("promotion request not found")
            return _promotion_from_row(conn, row)

    def activate_promotion_request(
        self,
        *,
        request_id: str,
        environment: str,
        actor_principal_id: str,
        evidence: VerifiedActivationEvidence,
        reason: str,
    ) -> dict[str, Any]:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        deployment_id = uuid.uuid4().hex
        expired = False
        result: dict[str, Any] | None = None
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            actor = _require_principal_tx(conn, actor_principal_id)
            if actor["role"] != "admin":
                raise GovernanceForbidden("only an admin can activate a deployment")
            request_row = conn.execute(
                "SELECT * FROM production_promotion_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if request_row is None:
                raise GovernanceNotFound("promotion request not found")
            if _parse_time(str(request_row["expires_at"])) <= now_dt:
                if str(request_row["status"]) not in {"expired", "promoted"}:
                    conn.execute(
                        "UPDATE production_promotion_requests SET status = 'expired', updated_at = ? WHERE id = ?",
                        (now, request_id),
                    )
                    _append_event(
                        conn,
                        event_type="promotion.expired",
                        actor_principal_id=actor["id"],
                        actor_role=actor["role"],
                        target_type="promotion_request",
                        target_id=request_id,
                        environment=str(request_row["environment"]),
                        payload={"previous_status": str(request_row["status"])},
                        at=now,
                    )
                expired = True
            else:
                if str(request_row["status"]) != "approved":
                    raise GovernanceConflict(
                        "promotion request must have ordered checker and admin approval"
                    )
                frozen_environment = str(request_row["environment"])
                if environment != frozen_environment:
                    raise GovernanceConflict(
                        "promotion request is bound to a different environment"
                    )
                strategy = _strategy_binding_tx(
                    conn,
                    strategy_id=str(request_row["strategy_id"]),
                    strategy_version=int(request_row["strategy_version"]),
                )
                if (
                    strategy["strategy_content_hash"]
                    != str(request_row["strategy_content_hash"])
                    or strategy["asset_status"]
                    != str(request_row["asset_status_snapshot"])
                ):
                    raise GovernanceConflict("strategy binding drifted after approval")
                slot = str(request_row["deployment_slot"])
                manifest_hash = _register_deployment_manifest_tx(
                    conn,
                    environment=environment,
                    deployment_slot=slot,
                    strategy=strategy,
                    created_at=now,
                )
                if manifest_hash != str(request_row["manifest_hash"]):
                    raise GovernanceConflict("deployment manifest binding drifted")
                evidence_context = ActivationVerificationContext(
                    promotion_request_id=request_id,
                    environment=environment,
                    deployment_slot=slot,
                    strategy_id=strategy["id"],
                    strategy_version=strategy["version"],
                    strategy_content_hash=strategy["strategy_content_hash"],
                    manifest_hash=manifest_hash,
                )
                require_evidence_matches_context(evidence, evidence_context)
                activation_evidence_id = _register_activation_evidence_tx(
                    conn,
                    evidence=evidence,
                    created_at=now,
                )
                external_deployment_ref = evidence.external_deployment_ref
                health_evidence_ref = evidence.health_evidence_ref
                health_status = evidence.health_status
                head = conn.execute(
                    "SELECT * FROM production_environment_heads WHERE environment = ?",
                    (environment,),
                ).fetchone()
                head_column = (
                    "active_deployment_id"
                    if slot == "production"
                    else "shadow_deployment_id"
                )
                current_id = None if head is None else head[head_column]
                expected_id = request_row["expected_current_deployment_id"]
                if current_id != expected_id:
                    raise GovernanceConflict(
                        "target environment changed after the promotion request was created"
                    )
                if current_id is not None:
                    predecessor = conn.execute(
                        "SELECT * FROM production_deployments WHERE id = ?",
                        (current_id,),
                    ).fetchone()
                    expected_status = "active" if slot == "production" else "shadow"
                    if (
                        predecessor is None
                        or str(predecessor["environment"]) != environment
                        or str(predecessor["deployment_slot"]) != slot
                        or str(predecessor["status"]) != expected_status
                    ):
                        raise GovernanceConflict(
                            "target environment predecessor is inconsistent"
                        )
                    conn.execute(
                        "UPDATE production_deployments SET status = 'superseded', updated_at = ? WHERE id = ? AND status = ?",
                        (now, current_id, expected_status),
                    )
                deployment_status = "active" if slot == "production" else "shadow"
                conn.execute(
                    """
                    INSERT INTO production_deployments(
                        id, environment, deployment_slot, strategy_id,
                        strategy_version, strategy_content_hash,
                        asset_status_snapshot, manifest_hash,
                        external_deployment_ref, health_evidence_ref,
                        health_status, status, predecessor_deployment_id,
                        promotion_request_id, activated_by, activation_reason,
                        created_at, updated_at, activation_evidence_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        deployment_id,
                        environment,
                        slot,
                        strategy["id"],
                        strategy["version"],
                        strategy["strategy_content_hash"],
                        strategy["asset_status"],
                        str(request_row["manifest_hash"]),
                        external_deployment_ref,
                        health_evidence_ref,
                        health_status,
                        deployment_status,
                        current_id,
                        request_id,
                        actor["id"],
                        reason,
                        now,
                        now,
                        activation_evidence_id,
                    ),
                )
                if head is None:
                    conn.execute(
                        """
                        INSERT INTO production_environment_heads(
                            environment, revision, active_deployment_id,
                            shadow_deployment_id, created_at, updated_at
                        ) VALUES (?, 1, ?, ?, ?, ?)
                        """,
                        (
                            environment,
                            deployment_id if slot == "production" else None,
                            deployment_id if slot == "shadow" else None,
                            now,
                            now,
                        ),
                    )
                else:
                    conn.execute(
                        f"""
                        UPDATE production_environment_heads
                           SET {head_column} = ?, revision = revision + 1,
                               updated_at = ?
                         WHERE environment = ? AND {head_column} IS ?
                        """,
                        (deployment_id, now, environment, current_id),
                    )
                cursor = conn.execute(
                    """
                    UPDATE production_promotion_requests
                       SET status = 'promoted', promoted_deployment_id = ?,
                           updated_at = ?
                     WHERE id = ? AND status = 'approved'
                    """,
                    (deployment_id, now, request_id),
                )
                if cursor.rowcount != 1:
                    raise GovernanceConflict("promotion request was already consumed")
                _append_event(
                    conn,
                    event_type="deployment.activated",
                    actor_principal_id=actor["id"],
                    actor_role=actor["role"],
                    target_type="deployment",
                    target_id=deployment_id,
                    environment=environment,
                    payload={
                        "deployment_slot": slot,
                        "status": deployment_status,
                        "strategy_id": strategy["id"],
                        "strategy_version": strategy["version"],
                        "strategy_content_hash": strategy["strategy_content_hash"],
                        "manifest_hash": str(request_row["manifest_hash"]),
                        "external_deployment_ref": external_deployment_ref,
                        "health_evidence_ref": health_evidence_ref,
                        "health_status": health_status,
                        "activation_evidence_id": activation_evidence_id,
                        "evidence_verifier_id": evidence.verifier_id,
                        "evidence_receipt_id": evidence.receipt_id,
                        "evidence_content_hash": evidence.content_hash,
                        "predecessor_deployment_id": current_id,
                        "promotion_request_id": request_id,
                        "reason": reason,
                    },
                    at=now,
                )
                deployment = conn.execute(
                    "SELECT * FROM production_deployments WHERE id = ?",
                    (deployment_id,),
                ).fetchone()
                result = _deployment_from_row(deployment)
        if expired:
            raise GovernanceConflict("promotion request is expired")
        assert result is not None
        return result

    def get_environment(self, environment: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            head = conn.execute(
                "SELECT * FROM production_environment_heads WHERE environment = ?",
                (str(environment),),
            ).fetchone()
            if head is None:
                return {
                    "environment": str(environment),
                    "revision": 0,
                    "active": None,
                    "shadow": None,
                }
            active = _deployment_by_id(conn, head["active_deployment_id"])
            shadow = _deployment_by_id(conn, head["shadow_deployment_id"])
            return {
                "environment": str(head["environment"]),
                "revision": int(head["revision"]),
                "active": active,
                "shadow": shadow,
            }

    def get_strategy_deployments(self, strategy_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            strategy = conn.execute(
                """
                SELECT id, version, status, asset_status, dsl_content_hash
                  FROM strategies
                 WHERE id = ?
                """,
                (str(strategy_id),),
            ).fetchone()
            if strategy is None:
                raise GovernanceNotFound("strategy asset not found")
            rows = conn.execute(
                """
                SELECT *
                  FROM production_deployments
                 WHERE strategy_id = ?
                 ORDER BY created_at, id
                """,
                (str(strategy_id),),
            ).fetchall()
        asset_status = str(strategy["asset_status"] or strategy["status"])
        if asset_status == "adopted":
            asset_status = "adopted_local"
        return {
            "strategy_id": str(strategy["id"]),
            "strategy_version": int(strategy["version"]),
            "strategy_content_hash": str(strategy["dsl_content_hash"] or ""),
            "asset_status": asset_status,
            "deployments": [_deployment_from_row(row) for row in rows],
        }

    def rollback_deployment(
        self,
        *,
        environment: str,
        deployment_id: str,
        expected_active_deployment_id: str,
        actor_principal_id: str,
        reason: str,
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            actor = _require_principal_tx(conn, actor_principal_id)
            if actor["role"] != "admin":
                raise GovernanceForbidden("only an admin can rollback a deployment")
            if deployment_id != expected_active_deployment_id:
                raise GovernanceConflict(
                    "rollback expected active deployment does not match the target"
                )
            head = conn.execute(
                "SELECT * FROM production_environment_heads WHERE environment = ?",
                (environment,),
            ).fetchone()
            if head is None or head["active_deployment_id"] != deployment_id:
                raise GovernanceConflict(
                    "deployment is not the current active record in this environment"
                )
            current = conn.execute(
                "SELECT * FROM production_deployments WHERE id = ?",
                (deployment_id,),
            ).fetchone()
            if (
                current is None
                or str(current["environment"]) != environment
                or str(current["deployment_slot"]) != "production"
                or str(current["status"]) != "active"
            ):
                raise GovernanceConflict("active deployment record is inconsistent")
            predecessor_id = current["predecessor_deployment_id"]
            if predecessor_id is None:
                raise GovernanceConflict(
                    "the first active deployment has no predecessor to restore"
                )
            predecessor = conn.execute(
                "SELECT * FROM production_deployments WHERE id = ?",
                (predecessor_id,),
            ).fetchone()
            if (
                predecessor is None
                or str(predecessor["environment"]) != environment
                or str(predecessor["deployment_slot"]) != "production"
                or str(predecessor["status"]) != "superseded"
            ):
                raise GovernanceConflict("rollback predecessor is inconsistent")
            rolled_back = conn.execute(
                """
                UPDATE production_deployments
                   SET status = 'rolled_back', updated_at = ?
                 WHERE id = ? AND status = 'active'
                """,
                (now, deployment_id),
            )
            restored = conn.execute(
                """
                UPDATE production_deployments
                   SET status = 'active', updated_at = ?
                 WHERE id = ? AND status = 'superseded'
                """,
                (now, predecessor_id),
            )
            head_update = conn.execute(
                """
                UPDATE production_environment_heads
                   SET active_deployment_id = ?, revision = revision + 1,
                       updated_at = ?
                 WHERE environment = ? AND active_deployment_id = ?
                """,
                (predecessor_id, now, environment, deployment_id),
            )
            if (
                rolled_back.rowcount != 1
                or restored.rowcount != 1
                or head_update.rowcount != 1
            ):
                raise GovernanceConflict("rollback compare-and-swap failed")
            _append_event(
                conn,
                event_type="deployment.rolled_back",
                actor_principal_id=actor["id"],
                actor_role=actor["role"],
                target_type="deployment",
                target_id=deployment_id,
                environment=environment,
                payload={
                    "rolled_back_deployment_id": deployment_id,
                    "restored_deployment_id": str(predecessor_id),
                    "restored_strategy_id": str(predecessor["strategy_id"]),
                    "restored_strategy_version": int(predecessor["strategy_version"]),
                    "restored_strategy_content_hash": str(
                        predecessor["strategy_content_hash"]
                    ),
                    "restored_manifest_hash": str(predecessor["manifest_hash"]),
                    "reason": reason,
                },
                at=now,
            )
            rolled_back_row = conn.execute(
                "SELECT * FROM production_deployments WHERE id = ?",
                (deployment_id,),
            ).fetchone()
            restored_row = conn.execute(
                "SELECT * FROM production_deployments WHERE id = ?",
                (predecessor_id,),
            ).fetchone()
            updated_head = conn.execute(
                "SELECT * FROM production_environment_heads WHERE environment = ?",
                (environment,),
            ).fetchone()
        return {
            "environment": environment,
            "revision": int(updated_head["revision"]),
            "rolled_back_deployment": _deployment_from_row(rolled_back_row),
            "restored_deployment": _deployment_from_row(restored_row),
        }

    def export_audit(self) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM production_governance_events ORDER BY sequence"
            ).fetchall()
        events = [_event_from_row(row) for row in rows]
        previous_hash = _GENESIS_EVENT_HASH
        for expected_sequence, event in enumerate(events, start=1):
            if event["sequence"] != expected_sequence:
                raise GovernanceConflict("production audit sequence is not contiguous")
            if event["previous_event_hash"] != previous_hash:
                raise GovernanceConflict("production audit predecessor hash drifted")
            hash_input = {
                key: value for key, value in event.items() if key != "event_hash"
            }
            calculated = hashlib.sha256(
                _canonical_json(hash_input).encode("utf-8")
            ).hexdigest()
            if calculated != event["event_hash"]:
                raise GovernanceConflict("production audit event hash drifted")
            previous_hash = event["event_hash"]
        return {
            "schema_version": "production-governance.audit.v1",
            "hash_algorithm": "sha256",
            "canonicalization": "utf8-json-sort-keys-compact",
            "genesis_hash": _GENESIS_EVENT_HASH,
            "events": events,
            "verification": {
                "valid": True,
                "event_count": len(events),
                "chain_head": previous_hash,
            },
        }


def _append_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    actor_principal_id: str,
    actor_role: str,
    target_type: str,
    target_id: str,
    environment: str | None,
    payload: dict[str, Any],
    at: str,
) -> dict[str, Any]:
    previous = conn.execute(
        """
        SELECT sequence, event_hash
          FROM production_governance_events
         ORDER BY sequence DESC
         LIMIT 1
        """
    ).fetchone()
    sequence = 1 if previous is None else int(previous["sequence"]) + 1
    previous_hash = (
        _GENESIS_EVENT_HASH if previous is None else str(previous["event_hash"])
    )
    event = {
        "id": uuid.uuid4().hex,
        "sequence": sequence,
        "event_type": _required_text(event_type, "event_type"),
        "actor_principal_id": _required_text(
            actor_principal_id, "actor_principal_id"
        ),
        "actor_role": _required_text(actor_role, "actor_role"),
        "target_type": _required_text(target_type, "target_type"),
        "target_id": _required_text(target_id, "target_id"),
        "environment": environment,
        "payload": dict(payload),
        "at": _required_text(at, "at"),
        "previous_event_hash": previous_hash,
    }
    event_hash = hashlib.sha256(_canonical_json(event).encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO production_governance_events(
            sequence, id, event_type, actor_principal_id, actor_role,
            target_type, target_id, environment, payload_json,
            previous_event_hash, event_hash, at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sequence,
            event["id"],
            event["event_type"],
            event["actor_principal_id"],
            event["actor_role"],
            event["target_type"],
            event["target_id"],
            event["environment"],
            _canonical_json(event["payload"]),
            previous_hash,
            event_hash,
            event["at"],
        ),
    )
    return {**event, "event_hash": event_hash}


def _principal_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["local_principal_id"]),
        "display_name": str(row["display_name"]),
        "role": str(row["role"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
    }


def _require_principal_tx(
    conn: sqlite3.Connection,
    principal_id: str,
) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM production_principals WHERE local_principal_id = ?",
        (str(principal_id),),
    ).fetchone()
    if row is None or str(row["status"]) != "active":
        raise GovernanceNotFound("active production principal not found")
    return _principal_from_row(row)


def _strategy_binding_tx(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    strategy_version: int,
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, version, status, asset_status,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies
         WHERE id = ?
        """,
        (strategy_id,),
    ).fetchone()
    if row is None:
        raise GovernanceNotFound("strategy asset not found")
    if int(row["version"]) != int(strategy_version):
        raise GovernanceConflict("strategy version drifted")
    asset_status = str(row["asset_status"] or row["status"])
    if asset_status not in {"validated", "adopted_local"}:
        raise GovernanceConflict(
            "strategy must be validated or adopted_local before promotion"
        )
    dsl_json = row["dsl_json"]
    dsl_schema_version = row["dsl_schema_version"]
    cached_content_hash = str(row["dsl_content_hash"] or "")
    if dsl_json is None or dsl_schema_version is None:
        raise GovernanceConflict("strategy canonical DSL is unavailable")
    if not _is_sha256(cached_content_hash):
        raise GovernanceConflict("strategy content hash is unavailable")
    from marvis.packs.strategy.dsl import (
        canonical_strategy_json,
        parse_strategy_spec,
    )
    from marvis.packs.strategy.errors import StrategyError

    try:
        spec = parse_strategy_spec(json.loads(str(dsl_json)))
        canonical_dsl = canonical_strategy_json(spec).encode("utf-8")
    except (TypeError, ValueError, json.JSONDecodeError, StrategyError) as exc:
        raise GovernanceConflict("strategy canonical DSL is invalid") from exc
    if str(dsl_schema_version) != spec.schema_version:
        raise GovernanceConflict("strategy canonical DSL schema version drifted")
    authenticated_content_hash = hashlib.sha256(canonical_dsl).hexdigest()
    if not hmac.compare_digest(cached_content_hash, authenticated_content_hash):
        raise GovernanceConflict("strategy canonical DSL content hash drifted")
    return {
        "id": str(row["id"]),
        "version": int(row["version"]),
        "asset_status": asset_status,
        "strategy_content_hash": authenticated_content_hash,
    }


def _register_deployment_manifest_tx(
    conn: sqlite3.Connection,
    *,
    environment: str,
    deployment_slot: str,
    strategy: dict[str, Any],
    created_at: str,
) -> str:
    manifest = {
        "schema_version": "production-deployment-manifest.v1",
        "environment": _required_text(environment, "environment"),
        "deployment_slot": _required_text(deployment_slot, "deployment_slot"),
        "strategy_id": _required_text(strategy["id"], "strategy_id"),
        "strategy_version": int(strategy["version"]),
        "strategy_content_hash": _required_text(
            strategy["strategy_content_hash"],
            "strategy_content_hash",
        ),
    }
    canonical = _canonical_json(manifest)
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    existing = conn.execute(
        "SELECT canonical_json FROM production_deployment_manifests WHERE id = ?",
        (manifest_hash,),
    ).fetchone()
    if existing is not None:
        if str(existing["canonical_json"]) != canonical:
            raise GovernanceConflict("deployment manifest hash collision")
        return manifest_hash
    conn.execute(
        """
        INSERT INTO production_deployment_manifests(
            id, strategy_id, strategy_version, strategy_content_hash,
            environment, deployment_slot, canonical_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            manifest_hash,
            manifest["strategy_id"],
            manifest["strategy_version"],
            manifest["strategy_content_hash"],
            manifest["environment"],
            manifest["deployment_slot"],
            canonical,
            created_at,
        ),
    )
    return manifest_hash


def _register_activation_evidence_tx(
    conn: sqlite3.Connection,
    *,
    evidence: VerifiedActivationEvidence,
    created_at: str,
) -> str:
    canonical = _canonical_json(evidence.canonical_payload())
    existing = conn.execute(
        """
        SELECT id, canonical_json
          FROM production_activation_evidence
         WHERE verifier_id = ? AND receipt_id = ?
        """,
        (evidence.verifier_id, evidence.receipt_id),
    ).fetchone()
    if existing is not None:
        if (
            str(existing["id"]) != evidence.content_hash
            or str(existing["canonical_json"]) != canonical
        ):
            raise GovernanceConflict("activation evidence receipt was replayed with drift")
        return str(existing["id"])
    conn.execute(
        """
        INSERT INTO production_activation_evidence(
            id, verifier_id, receipt_id, promotion_request_id,
            environment, manifest_hash, external_deployment_ref,
            health_evidence_ref, health_status, canonical_json,
            verified_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence.content_hash,
            evidence.verifier_id,
            evidence.receipt_id,
            evidence.promotion_request_id,
            evidence.environment,
            evidence.manifest_hash,
            evidence.external_deployment_ref,
            evidence.health_evidence_ref,
            evidence.health_status,
            canonical,
            evidence.verified_at,
            created_at,
        ),
    )
    return evidence.content_hash


def _promotion_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> dict[str, Any]:
    approvals = conn.execute(
        """
        SELECT stage, principal_id, role
          FROM production_promotion_approvals
         WHERE request_id = ?
         ORDER BY CASE stage WHEN 'checker' THEN 1 ELSE 2 END
        """,
        (row["id"],),
    ).fetchall()
    return {
        "id": str(row["id"]),
        "environment": str(row["environment"]),
        "deployment_slot": str(row["deployment_slot"]),
        "strategy_id": str(row["strategy_id"]),
        "strategy_version": int(row["strategy_version"]),
        "strategy_content_hash": str(row["strategy_content_hash"]),
        "asset_status_snapshot": str(row["asset_status_snapshot"]),
        "manifest_hash": str(row["manifest_hash"]),
        "expected_current_deployment_id": row["expected_current_deployment_id"],
        "maker_principal_id": str(row["maker_principal_id"]),
        "reason": str(row["reason"]),
        "status": str(row["status"]),
        "expires_at": str(row["expires_at"]),
        "promoted_deployment_id": row["promoted_deployment_id"],
        "created_at": str(row["created_at"]),
        "approvals": [
            {
                "stage": str(item["stage"]),
                "principal_id": str(item["principal_id"]),
                "role": str(item["role"]),
            }
            for item in approvals
        ],
    }


def _deployment_by_id(
    conn: sqlite3.Connection,
    deployment_id: object,
) -> dict[str, Any] | None:
    if deployment_id is None:
        return None
    row = conn.execute(
        "SELECT * FROM production_deployments WHERE id = ?",
        (str(deployment_id),),
    ).fetchone()
    if row is None:
        raise GovernanceConflict("environment head references a missing deployment")
    return _deployment_from_row(row)


def _deployment_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "environment": str(row["environment"]),
        "deployment_slot": str(row["deployment_slot"]),
        "strategy_id": str(row["strategy_id"]),
        "strategy_version": int(row["strategy_version"]),
        "strategy_content_hash": str(row["strategy_content_hash"]),
        "asset_status_snapshot": str(row["asset_status_snapshot"]),
        "manifest_hash": str(row["manifest_hash"]),
        "external_deployment_ref": str(row["external_deployment_ref"]),
        "health_evidence_ref": str(row["health_evidence_ref"]),
        "health_status": str(row["health_status"]),
        "activation_evidence_id": row["activation_evidence_id"],
        "status": str(row["status"]),
        "predecessor_deployment_id": row["predecessor_deployment_id"],
        "promotion_request_id": str(row["promotion_request_id"]),
        "activated_by": str(row["activated_by"]),
        "created_at": str(row["created_at"]),
    }


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(str(row["payload_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GovernanceConflict("production audit payload is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GovernanceConflict("production audit payload must be an object")
    return {
        "id": str(row["id"]),
        "sequence": int(row["sequence"]),
        "event_type": str(row["event_type"]),
        "actor_principal_id": str(row["actor_principal_id"]),
        "actor_role": str(row["actor_role"]),
        "target_type": str(row["target_type"]),
        "target_id": str(row["target_id"]),
        "environment": row["environment"],
        "payload": payload,
        "at": str(row["at"]),
        "previous_event_hash": str(row["previous_event_hash"]),
        "event_hash": str(row["event_hash"]),
    }


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _required_text(value: object, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


__all__ = ["ProductionGovernanceRepository"]

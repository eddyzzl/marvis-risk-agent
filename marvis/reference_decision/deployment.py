"""Trusted local deployment adapter; the governance head is the serving pointer.

There is no second approval state machine and no remote deployment claim. A
successful installation proves a real isolated probe of the frozen package.
Activation/rollback use the same transactional head that the HTTP scorer reads.
"""

from dataclasses import asdict
from datetime import UTC, datetime
import json
from typing import Protocol

from marvis.db_schema import connect
from marvis.production_governance.errors import GovernanceConflict
from marvis.production_governance.evidence import (
    ActivationVerificationContext,
    VerifiedActivationEvidence,
)
from marvis.production_governance.repository import _deployment_manifest_tx
from marvis.reference_decision.contracts import (
    DecisionError,
    canonical,
    digest,
    validate_features,
)
from marvis.reference_decision.service import ENVIRONMENT


ADAPTER_ID = "local-reference.v1"


class DecisionDeploymentAdapter(Protocol):
    """Institution adapters must be code-reviewed, registered trusted code.

    install is idempotent by approved promotion identity. readback must observe
    the deployed package, not echo a submitted version. verify must authenticate
    that readback against the complete ActivationVerificationContext. No import
    path supplied over HTTP selects executable code.
    """

    def install(self, promotion_id: str, probe_features: dict) -> dict: ...
    def readback(self, promotion_id: str) -> dict: ...
    def verify(
        self, *, receipt_id: str, context: ActivationVerificationContext
    ) -> VerifiedActivationEvidence: ...


class LocalReferenceAdapter:
    def __init__(self, service):
        self.service = service
        self.db_path = service.settings.db_path

    def _approved(self, conn, promotion_id):
        row = conn.execute(
            "SELECT * FROM production_promotion_requests WHERE id=?", (promotion_id,)
        ).fetchone()
        if row is None or row["environment"] != ENVIRONMENT:
            raise GovernanceConflict("approved local reference promotion required")
        if row["status"] not in {"approved", "promoted"} or (
            row["status"] != "promoted"
            and datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC)
        ):
            raise GovernanceConflict("promotion is not approved or has expired")
        manifest = _deployment_manifest_tx(conn, row["manifest_hash"])
        package_hash = manifest.get("decision_package_hash")
        if not package_hash:
            raise GovernanceConflict("promotion has no frozen decision package")
        return row, package_hash

    def install(self, promotion_id, probe_features):
        with connect(self.db_path) as conn:
            row, package_hash = self._approved(conn, promotion_id)
            existing = conn.execute(
                "SELECT 1 FROM reference_installations WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
        if existing:
            return self.readback(promotion_id)
        manifest = self.service.packages.get(package_hash)
        validate_features(probe_features, manifest)
        probe = self.service.evaluate(
            package_hash, probe_features, manifest["configuration"]["timeout_seconds"]
        )
        context = ActivationVerificationContext(
            promotion_request_id=promotion_id,
            environment=ENVIRONMENT,
            deployment_slot=row["deployment_slot"],
            strategy_id=row["strategy_id"],
            strategy_version=row["strategy_version"],
            strategy_content_hash=row["strategy_content_hash"],
            manifest_hash=row["manifest_hash"],
        )
        evidence = VerifiedActivationEvidence.issue(
            verifier_id=ADAPTER_ID,
            receipt_id=f"reference-install:{promotion_id}",
            context=context,
            external_deployment_ref=f"reference-install:{promotion_id}:package:{package_hash}",
            health_evidence_ref=f"reference-probe:{digest({'input_hash': digest(probe_features), 'output': probe})}",
            health_status="healthy",
            verified_at=datetime.now(UTC).isoformat(),
        )
        receipt = {
            "package_hash": package_hash,
            "state": "installed_not_activated",
            "adapter_id": ADAPTER_ID,
            "execution_identity": "local_reference_worker.v1",
            "assurance": "local_reference_only",
            "probe_input_hash": digest(probe_features),
            "probe_output_hash": digest(probe),
            "probe": probe,
            "activation_evidence": asdict(evidence),
        }
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current, current_hash = self._approved(conn, promotion_id)
            if (
                current_hash != package_hash
                or current["manifest_hash"] != context.manifest_hash
            ):
                raise GovernanceConflict("promotion changed during installation")
            self.service.packages.get(package_hash)
            conn.execute(
                "INSERT OR IGNORE INTO reference_installations VALUES (?,?,?,?)",
                (
                    promotion_id,
                    package_hash,
                    canonical(receipt),
                    evidence.verified_at,
                ),
            )
        return self.readback(promotion_id)

    def list(self, *, limit=100, offset=0):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN")
            total = conn.execute(
                "SELECT count(*) FROM reference_installations"
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT i.*, p.environment, p.deployment_slot, p.status FROM reference_installations i JOIN production_promotion_requests p ON p.id=i.promotion_id ORDER BY i.installed_at DESC, i.promotion_id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        items = []
        for row in rows:
            evidence = json.loads(row["receipt_json"])["activation_evidence"]
            items.append(
                {
                    "promotion_request_id": row["promotion_id"],
                    "receipt_id": evidence["receipt_id"],
                    "verifier_id": evidence["verifier_id"],
                    "environment": row["environment"],
                    "deployment_slot": row["deployment_slot"],
                    "package_hash": row["package_hash"],
                    "installed_at": row["installed_at"],
                    "promotion_status": row["status"],
                    "readback_state": "verify_on_detail_read",
                    "evidence_content_hash": evidence["content_hash"],
                }
            )
        return {
            "installations": items,
            "count": total,
            "next_offset": offset + len(items) if offset + len(items) < total else None,
        }

    def readback(self, promotion_id):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reference_installations WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            if row is None:
                raise GovernanceConflict("reference package is not installed")
            promotion, package_hash = self._approved(conn, promotion_id)
            receipt = json.loads(row["receipt_json"])
        manifest = self.service.packages.get(package_hash)
        if (
            row["package_hash"] != package_hash
            or receipt["package_hash"] != package_hash
        ):
            raise GovernanceConflict("installation package binding drifted")
        evidence = receipt["activation_evidence"]
        if evidence["manifest_hash"] != promotion["manifest_hash"]:
            raise GovernanceConflict("installation manifest binding drifted")
        return {
            **receipt,
            "readback": {
                "package_hash": package_hash,
                "files": manifest["files"],
                "raw_schema": manifest["configuration"]["raw_schema"],
                "strategy": manifest["strategy"],
                "model_artifact_id": manifest["model"]["id"],
                "preprocessing_receipt": manifest["preprocessing_receipt"],
            },
        }

    def verify(self, *, receipt_id, context):
        if (
            receipt_id != f"reference-install:{context.promotion_request_id}"
            or context.environment != ENVIRONMENT
        ):
            raise GovernanceConflict("local installation receipt binding invalid")
        receipt = self.readback(context.promotion_request_id)
        return VerifiedActivationEvidence(**receipt["activation_evidence"])

    def validate_deployment(self, conn, manifest_hash, current_deployment_id):
        """Called within the same head-CAS transaction for activation and rollback."""
        try:
            target = _deployment_manifest_tx(conn, manifest_hash)
            package_hash = target.get("decision_package_hash")
            manifest = self.service.packages.get(package_hash)
            installed = conn.execute(
                "SELECT 1 FROM reference_installations WHERE package_hash=?",
                (package_hash,),
            ).fetchone()
            if installed is None:
                raise GovernanceConflict(
                    "target reference package has no installation receipt"
                )
            if current_deployment_id:
                current = conn.execute(
                    "SELECT manifest_hash FROM production_deployments WHERE id=?",
                    (current_deployment_id,),
                ).fetchone()
                previous = _deployment_manifest_tx(conn, current["manifest_hash"])
                old = self.service.packages.get(previous["decision_package_hash"])
                # An in-place switch may not silently redefine an API's inputs
                # or business node. A different interface needs a new environment.
                keys = ("raw_schema", "decision_node", "score_product")
                if any(
                    manifest["configuration"][k] != old["configuration"][k]
                    for k in keys
                ):
                    raise GovernanceConflict(
                        "reference deployment interface is incompatible"
                    )
        except DecisionError as exc:
            raise GovernanceConflict(exc.code) from exc

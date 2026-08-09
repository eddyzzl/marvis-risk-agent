"""Server-resolved production activation evidence.

HTTP callers submit only an opaque receipt id and the name of an allowlisted
verifier.  The verifier's typed result is then bound to the immutable promotion
context and content-addressed before the repository may record activation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import re
from typing import Mapping, Protocol, runtime_checkable

from marvis.production_governance.errors import GovernanceConflict


_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_ISO8601_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|\+00:00)$"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class ActivationVerificationContext:
    promotion_request_id: str
    environment: str
    deployment_slot: str
    strategy_id: str
    strategy_version: int
    strategy_content_hash: str
    manifest_hash: str


@dataclass(frozen=True)
class VerifiedActivationEvidence:
    verifier_id: str
    receipt_id: str
    promotion_request_id: str
    environment: str
    deployment_slot: str
    strategy_id: str
    strategy_version: int
    strategy_content_hash: str
    manifest_hash: str
    external_deployment_ref: str
    health_evidence_ref: str
    health_status: str
    verified_at: str
    content_hash: str

    @classmethod
    def issue(
        cls,
        *,
        verifier_id: str,
        receipt_id: str,
        context: ActivationVerificationContext,
        external_deployment_ref: str,
        health_evidence_ref: str,
        health_status: str,
        verified_at: str,
    ) -> "VerifiedActivationEvidence":
        payload = {
            "schema_version": "production-activation-evidence.v1",
            "verifier_id": verifier_id,
            "receipt_id": receipt_id,
            "promotion_request_id": context.promotion_request_id,
            "environment": context.environment,
            "deployment_slot": context.deployment_slot,
            "strategy_id": context.strategy_id,
            "strategy_version": context.strategy_version,
            "strategy_content_hash": context.strategy_content_hash,
            "manifest_hash": context.manifest_hash,
            "external_deployment_ref": external_deployment_ref,
            "health_evidence_ref": health_evidence_ref,
            "health_status": health_status,
            "verified_at": verified_at,
        }
        content_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(**{key: value for key, value in payload.items() if key != "schema_version"}, content_hash=content_hash)

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("content_hash")
        return {
            "schema_version": "production-activation-evidence.v1",
            **payload,
        }


@runtime_checkable
class ActivationEvidenceVerifier(Protocol):
    def verify(
        self,
        *,
        receipt_id: str,
        context: ActivationVerificationContext,
    ) -> VerifiedActivationEvidence: ...


def resolve_activation_evidence(
    *,
    verifiers: Mapping[str, ActivationEvidenceVerifier],
    verifier_id: str,
    receipt_id: str,
    context: ActivationVerificationContext,
) -> VerifiedActivationEvidence:
    normalized_verifier = str(verifier_id).strip()
    normalized_receipt = str(receipt_id).strip()
    if _IDENTIFIER_RE.fullmatch(normalized_verifier) is None:
        raise GovernanceConflict("activation verifier id is invalid")
    if not normalized_receipt or len(normalized_receipt) > 240:
        raise GovernanceConflict("activation evidence receipt id is invalid")
    verifier = verifiers.get(normalized_verifier)
    if verifier is None:
        raise GovernanceConflict("activation evidence verifier is not allowlisted")
    try:
        evidence = verifier.verify(receipt_id=normalized_receipt, context=context)
    except GovernanceConflict:
        raise
    except Exception as exc:
        raise GovernanceConflict("activation evidence verification failed") from exc
    if not isinstance(evidence, VerifiedActivationEvidence):
        raise GovernanceConflict("activation verifier returned an invalid evidence type")
    _require_evidence_matches(
        evidence,
        verifier_id=normalized_verifier,
        receipt_id=normalized_receipt,
        context=context,
    )
    return evidence


def require_evidence_matches_context(
    evidence: VerifiedActivationEvidence,
    context: ActivationVerificationContext,
) -> None:
    _require_evidence_matches(
        evidence,
        verifier_id=evidence.verifier_id,
        receipt_id=evidence.receipt_id,
        context=context,
    )


def _require_evidence_matches(
    evidence: VerifiedActivationEvidence,
    *,
    verifier_id: str,
    receipt_id: str,
    context: ActivationVerificationContext,
) -> None:
    expected = {
        "verifier_id": verifier_id,
        "receipt_id": receipt_id,
        **asdict(context),
    }
    for field, value in expected.items():
        if getattr(evidence, field) != value:
            raise GovernanceConflict(f"activation evidence {field} binding drifted")
    if evidence.health_status != "healthy":
        raise GovernanceConflict("deployment cannot activate without healthy evidence")
    for field in ("strategy_content_hash", "manifest_hash", "content_hash"):
        if _SHA256_RE.fullmatch(str(getattr(evidence, field))) is None:
            raise GovernanceConflict(f"activation evidence {field} is invalid")
    for field in (
        "external_deployment_ref",
        "health_evidence_ref",
        "verified_at",
    ):
        if not str(getattr(evidence, field)).strip():
            raise GovernanceConflict(f"activation evidence {field} is missing")
    _require_strict_utc_timestamp(evidence.verified_at)
    actual_hash = hashlib.sha256(
        _canonical_json(evidence.canonical_payload()).encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(actual_hash, evidence.content_hash):
        raise GovernanceConflict("activation evidence content hash is invalid")


def _require_strict_utc_timestamp(value: str) -> None:
    timestamp = str(value)
    if _UTC_ISO8601_RE.fullmatch(timestamp) is None:
        raise GovernanceConflict(
            "activation evidence verified_at must be a strict ISO-8601 UTC timestamp"
        )
    normalized = f"{timestamp[:-1]}+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GovernanceConflict(
            "activation evidence verified_at must be a strict ISO-8601 UTC timestamp"
        ) from exc
    if parsed.utcoffset() != timedelta(0):
        raise GovernanceConflict(
            "activation evidence verified_at must be a strict ISO-8601 UTC timestamp"
        )

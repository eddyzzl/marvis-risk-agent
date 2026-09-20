"""Read-only audit evidence integrity and closure gate.

File hashes prove binding, not execution or reviewer identity. Trusted receipt
and review verifiers must be supplied by the acceptance environment, never by
the evidence JSON. Without them a bundle can be intact but cannot close a finding.
This module does not run an Agent, execute receipts, or promote ledger statuses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EVIDENCE_SCHEMA = "marvis.audit_evidence.v1"
TIERS = frozenset({"C", "R", "A", "H", "P"})
ENVIRONMENTS = frozenset({
    "test_process", "local_reference", "institution_test", "institution_production",
})
SOURCES = frozenset({
    "synthetic_fixture", "deidentified_historical", "reference_runtime", "institution_runtime",
})
STATUSES = frozenset({
    "planned", "in_progress", "ready_for_review", "failed", "awaiting_external", "closed",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
Verifier = Callable[[Mapping[str, Any], Path], bool]
_REQUIREMENT_FIELDS = (
    "required_evidence_tiers", "source_paths", "work_packages", "criterion_version",
)


@dataclass(frozen=True)
class EvidenceCheck:
    integrity_errors: tuple[str, ...]
    verification_errors: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return not self.integrity_errors and not self.verification_errors


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: object, *, nonempty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(_text(item) for item in value)
        and len(set(value)) == len(value)
    )


def _checked_path(root: Path, name: object, digest: object) -> Path:
    if not _text(name) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("path must be relative and contained in its evidence root")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("expected a lowercase SHA-256 digest")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("evidence file is missing or escapes its root")
    return path


def _bound_file(root: Path, name: object, digest: object) -> Path:
    path = _checked_path(root, name, digest)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("evidence hash mismatch")
    return path


def _read_bound_json(root: Path, name: object, digest: object) -> Any:
    path = _checked_path(root, name, digest)
    # Hash and decode the same bytes. A concurrent replacement after hashing
    # must not substitute weaker criteria or a different execution record.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("evidence hash mismatch")
    return json.loads(raw)


def _finding_errors(finding: Mapping[str, Any]) -> list[str]:
    errors = []
    for key in ("id", "criterion_version"):
        if not _text(finding.get(key)):
            errors.append(f"missing or invalid finding {key}")
    for key in ("work_packages", "source_paths", "required_evidence_tiers"):
        if not _strings(finding.get(key)):
            errors.append(f"missing or invalid finding {key}")
    tiers = finding.get("required_evidence_tiers")
    if _strings(tiers) and not set(tiers).issubset(TIERS):
        errors.append("invalid required evidence tiers")
    return errors


def load_acceptance_spec(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Load criteria against a digest supplied independently by the operator.

    The expected digest must not be read from the ledger under inspection.
    It is the acceptance environment's trust anchor, like receipt verifiers.
    """
    spec = _read_bound_json(path.parent, path.name, expected_sha256)
    if not isinstance(spec, dict):
        raise ValueError("acceptance spec must be an object")
    return spec


def validate_evidence(
    record: Mapping[str, Any],
    finding: Mapping[str, Any],
    *,
    source_root: Path,
    evidence_root: Path,
    expected_commit: str,
    receipt_verifiers: Mapping[str, Verifier] | None = None,
    review_verifier: Verifier | None = None,
) -> EvidenceCheck:
    """Check immutable bindings before consulting environment-owned verifiers."""
    errors = _finding_errors(finding)
    if errors:
        return EvidenceCheck(tuple(errors), ())
    for key in ("run_id", "target_id", "criterion_version", "work_package"):
        if not _text(record.get(key)):
            errors.append(f"missing or invalid {key}")
    if record.get("schema_version") != EVIDENCE_SCHEMA:
        errors.append("unsupported evidence schema")
    ids = record.get("finding_ids")
    if not _strings(ids) or finding.get("id") not in ids:
        errors.append("finding binding mismatch")
    if record.get("work_package") not in finding.get("work_packages", []):
        errors.append("work package binding mismatch")
    if record.get("criterion_version") != finding.get("criterion_version"):
        errors.append("criterion version mismatch")
    commit = record.get("commit")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit) or commit != expected_commit:
        errors.append("commit binding mismatch")
    tier = record.get("evidence_tier")
    if not isinstance(tier, str) or tier not in TIERS:
        errors.append("invalid evidence tier")
        tier = ""
    environment = record.get("environment_kind")
    source = record.get("source_kind")
    if not isinstance(environment, str) or environment not in ENVIRONMENTS:
        errors.append("invalid environment kind")
    if not isinstance(source, str) or source not in SOURCES:
        errors.append("invalid source kind")

    maps: dict[str, dict] = {}
    for key in ("source_hashes", "dataset_hashes", "input_hashes", "artifact_hashes"):
        hashes = record.get(key)
        if not isinstance(hashes, dict):
            errors.append(f"missing or invalid {key}")
            hashes = {}
        maps[key] = hashes
        if key != "dataset_hashes" and not hashes:
            errors.append(f"empty {key}")
        for name, digest in hashes.items():
            try:
                _bound_file(source_root if key == "source_hashes" else evidence_root, name, digest)
            except (OSError, ValueError) as exc:
                errors.append(f"{key}: {exc}")
    if not set(finding.get("source_paths", [])).issubset(maps["source_hashes"]):
        errors.append("source snapshot does not cover the finding")
    artifacts = maps["artifact_hashes"]
    for key in ("evidence_paths", "tool_receipts"):
        paths = record.get(key)
        if not _strings(paths, nonempty=key == "evidence_paths" or tier != "C"):
            errors.append(f"missing or invalid {key}")
        elif not set(paths).issubset(artifacts):
            errors.append(f"unhashed {key}")

    if tier != "C":
        executor = record.get("real_executor")
        if not isinstance(executor, dict) or not _text(executor.get("id")):
            errors.append("real executor identity required")
        if environment == "test_process":
            errors.append("test process cannot prove runtime acceptance")
    elif "real_executor" not in record:
        errors.append("missing real_executor declaration")
    if tier == "A":
        if not _text(record.get("model_profile")) or record.get("model_source") != "real_model":
            errors.append("real model profile required")
        if record.get("evaluation_mode") != "blind":
            errors.append("joint acceptance requires blind evaluation")
    if tier in {"A", "H", "P"} and source == "synthetic_fixture":
        errors.append("fixture provenance cannot satisfy this evidence tier")
    if tier == "H" and (source != "deidentified_historical" or not maps["dataset_hashes"]):
        errors.append("historical acceptance requires bound historical data")
    if tier == "P" and (environment != "institution_production" or source != "institution_runtime"):
        errors.append("production acceptance requires an institution production environment")

    review = record.get("independent_review")
    if not isinstance(review, dict) or not _text(review.get("reviewer_id")):
        errors.append("independent review required")
    elif not _text(review.get("artifact_path")) or review["artifact_path"] not in artifacts:
        errors.append("independent review artifact is not hash-bound")

    verification: list[str] = []
    # Invalid bundles never reach verification adapters. Verifier failures must
    # not cause a pass or silently discard this run from the ledger.
    if not errors:
        checks = (
            ("receipt", (receipt_verifiers or {}).get(tier)),
            ("independent review", review_verifier),
        )
        for label, verifier in checks:
            if verifier is None:
                verification.append(f"trusted {label} verifier unavailable")
                continue
            try:
                if verifier(record, evidence_root) is not True:
                    verification.append(f"{label} not verified")
            except Exception:
                verification.append(f"{label} verification failed")
    return EvidenceCheck(tuple(errors), tuple(verification))


def check_ledger(
    ledger: Mapping[str, Any],
    *,
    source_root: Path,
    evidence_root: Path,
    expected_commit: str,
    receipt_verifiers: Mapping[str, Verifier] | None = None,
    review_verifier: Verifier | None = None,
    trusted_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report open work and reject unsupported closure claims without mutation."""
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    findings = ledger.get("findings")
    if type(ledger.get("schema_version")) is not int or ledger["schema_version"] != 1 or not isinstance(findings, list) or not findings:
        return {"valid": False, "complete": False, "errors": ["invalid ledger schema"], "findings": []}
    if not isinstance(ledger.get("overall_status"), str) or ledger["overall_status"] not in STATUSES:
        errors.append("invalid overall status")
    requirements: dict[str, dict] = {}
    if trusted_spec is not None:
        spec_findings = trusted_spec.get("findings")
        if trusted_spec.get("schema_version") != "marvis.audit_requirements.v1" or not isinstance(spec_findings, list) or not spec_findings:
            errors.append("invalid trusted acceptance spec")
        else:
            for spec_finding in spec_findings:
                if not isinstance(spec_finding, dict) or _finding_errors(spec_finding):
                    errors.append("invalid trusted finding requirements")
                elif spec_finding["id"] in requirements:
                    errors.append("duplicate trusted finding requirements")
                else:
                    requirements[spec_finding["id"]] = spec_finding
    seen: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict) or not _text(finding.get("id")):
            errors.append("invalid finding")
            continue
        finding_id = finding["id"]
        if finding_id in seen:
            errors.append(f"duplicate finding: {finding_id}")
        seen.add(finding_id)
        schema_errors = _finding_errors(finding)
        if schema_errors:
            errors.extend(f"{finding_id}: {error}" for error in schema_errors)
            continue
        if trusted_spec is not None:
            frozen = requirements.get(finding_id)
            if frozen is None or any(
                (sorted(finding[key]) if isinstance(finding[key], list) else finding[key])
                != (sorted(frozen[key]) if isinstance(frozen[key], list) else frozen[key])
                for key in _REQUIREMENT_FIELDS
            ):
                errors.append(f"{finding_id}: frozen acceptance requirements mismatch")
        required = finding.get("required_evidence_tiers")
        if not _strings(required) or not set(required).issubset(TIERS):
            errors.append(f"{finding_id}: invalid required evidence tiers")
            required = []
        if not isinstance(finding.get("status"), str) or finding["status"] not in STATUSES:
            errors.append(f"{finding_id}: invalid status")
        verified: set[str] = set()
        records: list[dict[str, Any]] = []
        references = finding.get("evidence")
        if not isinstance(references, list):
            errors.append(f"{finding_id}: evidence must be a list")
            references = []
        for reference in references:
            try:
                if not isinstance(reference, dict):
                    raise ValueError("evidence reference must contain path and sha256")
                record = _read_bound_json(evidence_root, reference.get("path"), reference.get("sha256"))
                if not isinstance(record, dict):
                    raise ValueError("evidence record must be an object")
                check = validate_evidence(
                    record, finding, source_root=source_root, evidence_root=evidence_root,
                    expected_commit=expected_commit, receipt_verifiers=receipt_verifiers,
                    review_verifier=review_verifier,
                )
                records.append({
                    "run_id": record.get("run_id"), "verified": check.verified,
                    "integrity_errors": list(check.integrity_errors),
                    "verification_errors": list(check.verification_errors),
                })
                if check.verified:
                    verified.add(record["evidence_tier"])
                if check.integrity_errors:
                    errors.append(f"{finding_id}: invalid evidence bundle")
            except (OSError, ValueError) as exc:
                errors.append(f"{finding_id}: {exc}")
        missing = sorted(set(required) - verified)
        if finding.get("status") == "closed" and (missing or not required):
            errors.append(f"{finding_id}: unsupported closure claim")
        if finding.get("status") == "closed" and trusted_spec is None:
            errors.append(f"{finding_id}: trusted acceptance spec unavailable")
        results.append({
            "id": finding_id, "status": finding.get("status"),
            "missing_verified_tiers": missing, "evidence": records,
        })
    if trusted_spec is not None and seen != set(requirements):
        errors.append("finding set differs from frozen acceptance scope")
    complete = trusted_spec is not None and bool(results) and all(
        row["status"] == "closed" and not row["missing_verified_tiers"] for row in results
    )
    if ledger.get("overall_status") == "closed" and not complete:
        errors.append("unsupported overall closure claim")
    return {
        "valid": not errors, "complete": complete and not errors,
        "requirements_verified": trusted_spec is not None and not errors,
        "errors": errors, "findings": results,
    }

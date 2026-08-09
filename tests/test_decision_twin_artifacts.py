from datetime import UTC, datetime, timedelta
import json

import pytest

from marvis.decision_twin import (
    ContentAddressedAuditStore,
    IdempotencyConflict,
    TamperEvidenceError,
)


NOW = datetime(2026, 2, 1, tzinfo=UTC)


def test_artifact_write_is_content_addressed_hash_chained_and_idempotent(
    tmp_path,
) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin", clock=lambda: NOW)

    first = store.put(
        kind="comparison",
        payload={"manifest_hash": "a" * 64, "jointly_satisfied": True},
        idempotency_key="comparison:run-1",
    )
    repeated = ContentAddressedAuditStore(
        tmp_path / "decision-twin",
        clock=lambda: NOW + timedelta(hours=1),
    ).put(
        kind="comparison",
        payload={"jointly_satisfied": True, "manifest_hash": "a" * 64},
        idempotency_key="comparison:run-1",
    )
    second = store.put(
        kind="reconciliation",
        payload={"manifest_hash": "a" * 64, "actual_loss": 10.0},
        idempotency_key="reconciliation:run-1",
    )

    assert repeated == first
    assert first.artifact_uri == f"sha256://{first.artifact_hash}"
    assert second.sequence == 2
    assert second.previous_event_hash == first.event_hash
    verification = store.verify()
    assert verification.valid is True
    assert verification.event_count == 2
    assert verification.chain_head == second.event_hash
    assert store.get(first.artifact_hash) == {
        "jointly_satisfied": True,
        "manifest_hash": "a" * 64,
    }


def test_idempotency_key_cannot_be_reused_for_different_content(tmp_path) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin", clock=lambda: NOW)
    store.put("comparison", {"value": 1}, idempotency_key="run-1")

    with pytest.raises(IdempotencyConflict, match="different content"):
        store.put("comparison", {"value": 2}, idempotency_key="run-1")


def test_artifact_tampering_is_detected_before_the_next_write(tmp_path) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin", clock=lambda: NOW)
    receipt = store.put("comparison", {"value": 1}, idempotency_key="run-1")
    store.artifact_path(receipt.artifact_hash).write_text(
        '{"value":2}', encoding="utf-8"
    )

    with pytest.raises(TamperEvidenceError, match="artifact hash drifted"):
        store.verify()
    with pytest.raises(TamperEvidenceError, match="artifact hash drifted"):
        store.put("comparison", {"value": 3}, idempotency_key="run-2")


def test_audit_chain_tampering_is_detected(tmp_path) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin", clock=lambda: NOW)
    store.put("comparison", {"value": 1}, idempotency_key="run-1")
    store.put("comparison", {"value": 2}, idempotency_key="run-2")
    event_path = store.audit_event_path(2)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["previous_event_hash"] = "f" * 64
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(TamperEvidenceError, match="audit"):
        store.verify()


def test_noncanonical_audit_bytes_are_tamper_evidence(tmp_path) -> None:
    store = ContentAddressedAuditStore(tmp_path / "decision-twin", clock=lambda: NOW)
    store.put("comparison", {"value": 1}, idempotency_key="run-1")
    event_path = store.audit_event_path(1)
    event_path.write_bytes(event_path.read_bytes() + b"\n")

    with pytest.raises(TamperEvidenceError, match="audit"):
        store.verify()

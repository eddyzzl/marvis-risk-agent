"""One deterministic feature function for live ingestion snapshots and replay."""

from marvis.risk_context.event_contracts import EventSnapshot, content_hash
from marvis.risk_context.relation_features import relation_features
from marvis.risk_context.window_features import (
    members_hash,
    select_window,
    window_features,
)


def evaluate_event_snapshot(snapshot: EventSnapshot):
    """Calculate values, not trust: only EventRepository signs native evidence.

    A caller-constructed snapshot is not authenticated merely by passing this
    pure function. Tool/HTTP producers must use the repository's scoped methods.
    """
    window = select_window(snapshot)
    contract = window.snapshot.contract
    features = {**window_features(window), **relation_features(window)}
    missing = any(value["status"] == "unknown" for value in features.values())
    return {
        "schema_version": "risk-event.feature_result.v1",
        "contract": contract.model_dump(),
        "contract_hash": contract.contract_hash,
        "snapshot_hash": content_hash(window.snapshot.model_dump()),
        "availability_mode": contract.availability_mode,
        "source_assurance": "publisher_declared",
        "platform_ingestion_scope": "frozen_knowledge_cutoff",
        "historical_platform_visibility": "not_asserted"
        if contract.availability_mode == "retrospective_declared"
        else "ingested_by_declared_cutoff",
        "window_start_exclusive": window.start.isoformat(),
        "window_end_inclusive": window.end.isoformat(),
        "population_members_hash": members_hash(window.records),
        "population_count": len(window.records),
        "coverage_claims": [row.model_dump() for row in window.coverage],
        "status": "unknown" if missing else "measured",
        "next_action": "required_review" if missing else "none",
        "features": features,
        "automated_clearance": False,
        "fraud_or_identity_proof": False,
    }

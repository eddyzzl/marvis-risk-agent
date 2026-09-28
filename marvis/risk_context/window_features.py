"""Point-in-time selection and window aggregates shared by every execution mode."""

from dataclasses import dataclass
from datetime import timedelta

from marvis.risk_context.event_contracts import (
    CoverageClaim,
    EventSnapshot,
    at,
    content_hash,
)


def member(stored):
    return {
        "source_id": stored.source_id,
        "event_id": stored.claim.event_id,
        "version": stored.claim.version,
        "content_hash": stored.content_hash,
    }


def members_hash(records):
    return content_hash(
        sorted(
            [member(row) for row in records],
            key=lambda row: (row["source_id"], row["event_id"], row["version"]),
        )
    )


def identity_key(kind, identity):
    return None if identity is None else (kind, identity.namespace, identity.token)


@dataclass(frozen=True)
class SelectedWindow:
    snapshot: EventSnapshot
    records: tuple
    coverage: tuple
    uncertainty: tuple[str, ...]
    coverage_uncertainty: tuple[str, ...]
    start: object
    end: object


def _latest(claims, kind, contract, relevant):
    groups = {}
    for row in claims:
        if row.kind != kind or at(row.ingested_at) > at(contract.knowledge_cutoff):
            continue
        key = row.claim.event_id if kind == "events" else row.claim.coverage_id
        groups.setdefault(key, {})[row.claim.version] = row
    chosen, reasons = [], []
    for versions in groups.values():
        if not any(relevant(row.claim) for row in versions.values()):
            continue
        eligible = [
            row
            for row in versions.values()
            if row.claim.available_at is not None
            and at(row.claim.available_at) <= at(contract.decision_at)
        ]
        selected = max(eligible, key=lambda row: row.claim.version, default=None)
        selected_version = selected.claim.version if selected is not None else 0
        if any(
            row.claim.available_at is None and row.claim.version >= selected_version
            for row in versions.values()
        ):
            reasons.append("available_at_unknown")
            continue
        if selected is None:
            continue
        if any(version not in versions for version in range(1, selected_version + 1)):
            reasons.append("version_lineage_incomplete")
            continue
        lineage = [
            versions[version].claim for version in range(1, selected_version + 1)
        ]
        known_lineage = [claim for claim in lineage if claim.available_at is not None]
        if any(
            at(left.available_at) > at(right.available_at)
            for left, right in zip(known_lineage, known_lineage[1:])
        ):
            reasons.append("version_availability_conflict")
            continue
        chosen.append(selected)
    return tuple(
        sorted(
            chosen,
            key=lambda row: (
                row.claim.event_id if kind == "events" else row.claim.coverage_id,
                row.claim.version,
            ),
        )
    ), tuple(sorted(set(reasons)))


def select_window(snapshot: EventSnapshot):
    # Pure callers receive the same strict validation as repository-created snapshots.
    snapshot = EventSnapshot.model_validate(snapshot.model_dump())
    contract = snapshot.contract
    end = at(contract.decision_at)
    start = end - timedelta(seconds=contract.window_seconds)
    event_types = set(contract.event_types)

    def relevant_event(claim):
        return (
            claim.event_type in event_types
            and start < at(claim.event_at) <= end
            and (
                contract.include_current_event
                or claim.event_id != contract.current_event_id
            )
        )

    records, reasons = _latest(snapshot.claims, "events", contract, relevant_event)
    records = tuple(row for row in records if relevant_event(row.claim))
    if contract.include_current_event and contract.current_event_id is not None:
        current = next(
            (row for row in records if row.claim.event_id == contract.current_event_id),
            None,
        )
        if current is None:
            reasons = (*reasons, "current_event_not_available_in_window")
        elif getattr(current.claim, contract.focus_kind) != contract.focus:
            reasons = (*reasons, "current_event_focus_mismatch")

    def relevant_coverage(claim):
        return (
            bool(set(claim.event_types) & event_types)
            and at(claim.start_exclusive) < end
            and at(claim.through_inclusive) > start
        )

    coverage, coverage_reasons = _latest(
        snapshot.claims, "coverage", contract, relevant_coverage
    )
    coverage = tuple(row for row in coverage if relevant_coverage(row.claim))
    return SelectedWindow(
        snapshot, records, coverage, reasons, coverage_reasons, start, end
    )


def coverage_reasons(window, *, global_population=False):
    contract = window.snapshot.contract
    reasons = list(window.coverage_uncertainty)
    candidates = []
    for row in window.coverage:
        claim: CoverageClaim = row.claim
        if claim.subject is not None and (
            global_population
            or contract.focus_kind != "subject"
            or claim.subject != contract.focus
        ):
            continue
        candidates.append(claim)
        if claim.state != "complete":
            reasons.append("publisher_coverage_" + claim.state)
    for event_type in contract.event_types:
        cursor = window.start
        intervals = sorted(
            (at(claim.start_exclusive), at(claim.through_inclusive))
            for claim in candidates
            if claim.state == "complete" and event_type in claim.event_types
        )
        for start, end in intervals:
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor < window.end:
            reasons.append("publisher_coverage_gap:" + event_type)
    return sorted(set(reasons))


def feature_result(window, records, *, value, reasons, unit):
    contract = window.snapshot.contract
    reasons = sorted(set(reasons))
    return {
        "status": "unknown" if reasons else "measured",
        "value": None if reasons else value,
        "unit": unit,
        "next_action": "required_review" if reasons else "none",
        "missing_reasons": reasons,
        "observed_member_count": len(records),
        "members_hash": members_hash(records),
        "contract_hash": contract.contract_hash,
        "source_id": contract.source_id,
        "source_contract_hash": contract.source_contract_hash,
        "coverage_hash": content_hash([row.model_dump() for row in window.coverage]),
        "automated_clearance": False,
    }


def window_features(window: SelectedWindow):
    contract = window.snapshot.contract
    focus_key = identity_key(contract.focus_kind, contract.focus)
    records, missing_focus = [], False
    for row in window.records:
        identity = getattr(row.claim, contract.focus_kind)
        if identity is None:
            missing_focus = True
        elif identity_key(contract.focus_kind, identity) == focus_key:
            records.append(row)
    common = [*window.uncertainty, *coverage_reasons(window)]
    if missing_focus:
        common.append("focus_identity_missing")
    output = {}
    for spec in contract.window_features:
        reasons = list(common)
        unit = "count"
        if spec.operation == "count":
            value = len(records)
        elif spec.operation == "sum":
            unit = window.snapshot.source.numeric_fields.get(spec.field)
            if unit is None:
                raise ValueError(
                    "sum field is not defined in the frozen source contract"
                )
            values = [row.claim.values.get(spec.field) for row in records]
            if any(value is None for value in values):
                reasons.append("numeric_value_missing:" + spec.field)
            value = sum(value for value in values if value is not None)
        else:
            identities = [getattr(row.claim, spec.field) for row in records]
            if any(identity is None for identity in identities):
                reasons.append("distinct_identity_missing:" + spec.field)
            value = len(
                {
                    identity_key(spec.field, identity)
                    for identity in identities
                    if identity is not None
                }
            )
        output[spec.name] = feature_result(
            window, records, value=value, reasons=reasons, unit=unit
        )
    return output

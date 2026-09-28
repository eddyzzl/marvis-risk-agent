from datetime import timedelta

import pytest

from marvis.risk_context.event_contracts import (
    CoverageClaim,
    EventFeatureContract,
    EventRecord,
    EventSnapshot,
    EventSourceContract,
    OpaqueIdentity,
    StoredClaim,
    at,
    content_hash,
)
from marvis.risk_context.event_replay import evaluate_event_snapshot
from marvis.risk_context.window_features import members_hash


T = "2026-09-28T00:00:00+00:00"


def ago(seconds):
    return (at(T) - timedelta(seconds=seconds)).isoformat()


def identity(kind="subject", name="alice"):
    return {"namespace": "publisher." + kind, "token": content_hash(name)}


def source(**changes):
    value = {
        "source_id": "events-one",
        "task_id": "task-one",
        "entity_namespaces": {
            kind: "publisher." + kind for kind in ("subject", "device", "account")
        },
        "event_types": ["transaction", "login"],
        "numeric_fields": {"amount_minor": "CNY_minor"},
        "publisher_ref": "explicit publisher contract",
    }
    value.update(changes)
    return EventSourceContract.model_validate(value)


def event(event_id="evt-1", **changes):
    value = {
        "event_id": event_id,
        "version": 1,
        "event_type": "transaction",
        "event_at": ago(10),
        "available_at": ago(9),
        "subject": identity(),
        "device": identity("device", "phone-a"),
        "account": identity("account", "account-a"),
        "values": {"amount_minor": 100},
    }
    value.update(changes)
    return EventRecord.model_validate(value)


def coverage(**changes):
    value = {
        "coverage_id": "watermark-1",
        "version": 1,
        "event_types": ["transaction"],
        "start_exclusive": ago(60),
        "through_inclusive": T,
        "declared_at": T,
        "available_at": T,
        "state": "complete",
        "subject": None,
        "publisher_ref": "publisher window completeness declaration",
    }
    value.update(changes)
    return CoverageClaim.model_validate(value)


def query(src=None, **changes):
    src = src or source()
    value = {
        "source_id": src.source_id,
        "source_contract_hash": src.contract_hash,
        "decision_at": T,
        "knowledge_cutoff": T,
        "availability_mode": "platform_observed",
        "window_seconds": 60,
        "include_current_event": True,
        "current_event_id": None,
        "event_types": ["transaction"],
        "focus_kind": "subject",
        "focus": identity(),
        "window_features": [
            {"name": "transactions", "operation": "count"},
            {"name": "amount", "operation": "sum", "field": "amount_minor"},
            {"name": "devices", "operation": "distinct", "field": "device"},
        ],
        "relation_features": [
            {
                "name": "shared_devices",
                "operation": "shared_neighbor_count",
                "via_kind": "device",
                "target_kind": "subject",
            }
        ],
        "policy_ref": "reviewed window and identity policy",
    }
    value.update(changes)
    return EventFeatureContract.model_validate(value)


def stored(claim, src=None, ingested_at=None):
    src = src or source()
    return StoredClaim(
        source_id=src.source_id,
        source_contract_hash=src.contract_hash,
        kind="events" if isinstance(claim, EventRecord) else "coverage",
        claim=claim,
        content_hash=claim.contract_hash,
        ingested_at=ingested_at or ago(1),
        dataset_id="registered-file",
        dataset_content_hash="a" * 64,
        import_contract_hash="b" * 64,
    )


def snapshot(events=(), coverages=None, *, src=None, contract=None):
    src = src or source()
    claims = [stored(row, src) for row in events]
    claims += [
        stored(row, src)
        for row in (coverages if coverages is not None else [coverage()])
    ]
    return EventSnapshot(source=src, contract=contract or query(src), claims=claims)


def values(result):
    return {key: value["value"] for key, value in result["features"].items()}


def test_integer_window_aggregates_and_shared_relationships_are_exact():
    rows = [
        event(values={"amount_minor": 2**53 + 1}),
        event("evt-2", values={"amount_minor": 7}),
        event("evt-3", subject=identity(name="bob")),
    ]
    snap = snapshot(rows)
    result = evaluate_event_snapshot(snap)
    assert values(result) == {
        "transactions": 2,
        "amount": 2**53 + 8,
        "devices": 1,
        "shared_devices": 1,
    }
    assert result["features"]["amount"]["unit"] == "CNY_minor"
    assert result["features"]["transactions"]["members_hash"] == members_hash(
        snap.claims[:2]
    )
    assert result["features"]["shared_devices"]["observed_member_count"] == 3
    assert result["automated_clearance"] is False
    assert result["fraud_or_identity_proof"] is False
    assert result["contract_hash"] == snap.contract.contract_hash


@pytest.mark.parametrize(
    "kind,target,expected",
    [("device", "subject", 2), ("account", "subject", 2), ("account", "device", 2)],
)
def test_device_and_account_neighbors(kind, target, expected):
    rows = [
        event(),
        event("evt-2", subject=identity(name="bob")),
        event("evt-3", device=identity("device", "phone-b")),
    ]
    focus = identity(kind, "phone-a" if kind == "device" else "account-a")
    q = query(
        focus_kind=kind,
        focus=focus,
        window_features=[],
        relation_features=[
            {"name": "neighbors", "operation": "neighbor_count", "target_kind": target}
        ],
    )
    assert values(evaluate_event_snapshot(snapshot(rows, contract=q))) == {
        "neighbors": expected
    }


def test_interval_is_open_left_closed_right_and_current_identity_is_frozen():
    rows = [
        event("left", event_at=ago(60), available_at=ago(60)),
        event("inside", event_at=ago(59), available_at=ago(59)),
        event("current", event_at=T, available_at=T),
        event("future", event_at=ago(-1), available_at=ago(-1)),
    ]
    assert values(evaluate_event_snapshot(snapshot(rows)))["transactions"] == 2
    excluded = query(include_current_event=False, current_event_id="current")
    assert (
        values(evaluate_event_snapshot(snapshot(rows, contract=excluded)))[
            "transactions"
        ]
        == 1
    )
    assert excluded.contract_hash != query().contract_hash


def test_explicit_current_event_cannot_be_synthesized_or_bound_to_other_subject():
    for rows in ([], [event("current", subject=identity(name="other"))]):
        result = evaluate_event_snapshot(
            snapshot(rows, contract=query(current_event_id="current"))
        )
        assert set(values(result).values()) == {None}
        assert result["next_action"] == "required_review"


@pytest.mark.parametrize(
    "coverages",
    [
        [],
        [coverage(state="partial")],
        [coverage(state="unknown")],
        [coverage(through_inclusive=ago(1))],
        [coverage(event_types=["login"])],
        [coverage(available_at=None)],
    ],
)
def test_absence_without_complete_coverage_is_unknown(coverages):
    result = evaluate_event_snapshot(snapshot(coverages=coverages))
    assert result["status"] == "unknown"
    assert set(values(result).values()) == {None}
    assert all(
        value["next_action"] == "required_review"
        for value in result["features"].values()
    )


def test_complete_empty_coverage_is_true_zero_without_clearance():
    result = evaluate_event_snapshot(snapshot())
    assert set(values(result).values()) == {0}
    assert result["status"] == "measured"
    assert result["automated_clearance"] is False
    assert result["coverage_claims"][0]["claim"]["through_inclusive"] == T


def test_coverage_intervals_can_join_but_cannot_conceal_a_gap_or_partial_revision():
    first = coverage(through_inclusive=ago(30))
    second = coverage(coverage_id="watermark-2", start_exclusive=ago(30))
    assert (
        values(evaluate_event_snapshot(snapshot(coverages=[first, second])))[
            "transactions"
        ]
        == 0
    )
    gap = second.model_copy(update={"start_exclusive": ago(29)})
    assert (
        values(evaluate_event_snapshot(snapshot(coverages=[first, gap])))[
            "transactions"
        ]
        is None
    )
    revised = coverage(version=2, supersedes_version=1, state="partial")
    assert (
        values(evaluate_event_snapshot(snapshot(coverages=[coverage(), revised])))[
            "transactions"
        ]
        is None
    )


def test_subject_coverage_never_proves_shared_population_complete():
    result = evaluate_event_snapshot(
        snapshot([event()], [coverage(subject=identity())])
    )
    assert result["features"]["transactions"]["value"] == 1
    assert result["features"]["shared_devices"]["value"] is None


@pytest.mark.parametrize(
    "changes,unknown",
    [
        (
            {"available_at": None},
            {"transactions", "amount", "devices", "shared_devices"},
        ),
        ({"values": {}}, {"amount"}),
        ({"device": None}, {"devices", "shared_devices"}),
    ],
)
def test_missing_availability_values_or_identity_do_not_become_zero(changes, unknown):
    result = evaluate_event_snapshot(snapshot([event(**changes)]))
    assert {
        key for key, value in result["features"].items() if value["value"] is None
    } == unknown


def test_version_selection_preserves_original_ingestion_cutoff_and_requires_lineage():
    original = event()
    correction = event(
        version=2, supersedes_version=1, available_at=T, values={"amount_minor": 800}
    )
    assert values(evaluate_event_snapshot(snapshot([correction])))["amount"] is None
    assert (
        values(evaluate_event_snapshot(snapshot([correction, original])))["amount"]
        == 800
    )
    snap = snapshot([original])
    late = stored(correction, ingested_at=ago(-10))
    later = EventSnapshot.model_validate(
        {
            **snap.model_dump(),
            "claims": [*[row.model_dump() for row in snap.claims], late.model_dump()],
        }
    )
    assert values(evaluate_event_snapshot(later))["amount"] == 100
    retro = query(availability_mode="retrospective_declared", knowledge_cutoff=ago(-10))
    later = EventSnapshot.model_validate(
        {**later.model_dump(), "contract": retro.model_dump()}
    )
    result = evaluate_event_snapshot(later)
    assert values(result)["amount"] == 800
    assert result["historical_platform_visibility"] == "not_asserted"


def test_known_full_correction_can_replace_an_older_unknown_timestamp():
    rows = [
        event(available_at=None),
        event(
            version=2,
            supersedes_version=1,
            available_at=T,
            values={"amount_minor": 200},
        ),
    ]
    assert values(evaluate_event_snapshot(snapshot(rows)))["amount"] == 200
    rows.append(event(version=3, supersedes_version=2, available_at=None))
    assert values(evaluate_event_snapshot(snapshot(rows)))["amount"] is None


def test_cross_source_same_tokens_are_never_implicitly_merged():
    one = source()
    two = source(source_id="events-two")
    snap = snapshot([event()], src=one)
    assert (
        values(evaluate_event_snapshot(snapshot([event()], src=two)))["transactions"]
        == 1
    )
    with pytest.raises(ValueError, match="source mismatch"):
        EventSnapshot.model_validate(
            {
                **snap.model_dump(),
                "claims": [
                    *[row.model_dump() for row in snap.claims],
                    stored(event("other"), two).model_dump(),
                ],
            }
        )
    with pytest.raises(ValueError, match="namespace"):
        snapshot([event(subject={**identity(), "namespace": "other.publisher"})])


@pytest.mark.parametrize("value", [2**53 + 1, float(2**53 + 1), True])
def test_large_numeric_identities_cannot_be_coerced(value):
    with pytest.raises(ValueError):
        event(event_id=value)
    with pytest.raises(ValueError):
        OpaqueIdentity(namespace="subject", token=value)


def test_large_string_event_ids_remain_distinct():
    result = evaluate_event_snapshot(
        snapshot([event(str(2**53)), event(str(2**53 + 1))])
    )
    assert values(result)["transactions"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"event_at": "2026-01-01"},
        {"available_at": "unknown"},
        {"available_at": ago(20)},
        {"version": 2},
        {"version": True},
        {"values": {"amount_minor": 1.5}},
        {"values": {"amount_minor": True}},
        {"ingested_at": ago(60)},
    ],
)
def test_invalid_event_contracts_rejected(changes):
    with pytest.raises(ValueError):
        event(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"include_current_event": False},
        {"knowledge_cutoff": ago(-1)},
        {"window_seconds": 0},
        {"window_seconds": True},
        {"event_types": ["transaction", "transaction"]},
        {"policy_ref": "   "},
        {
            "relation_features": [
                {
                    "name": "bad",
                    "operation": "shared_neighbor_count",
                    "target_kind": "subject",
                }
            ]
        },
    ],
)
def test_ambiguous_window_contracts_rejected(changes):
    with pytest.raises(ValueError):
        query(**changes)


@pytest.mark.parametrize("build", [source, coverage])
def test_publisher_reference_must_be_explicit_not_whitespace(build):
    with pytest.raises(ValueError):
        build(publisher_ref="   ")

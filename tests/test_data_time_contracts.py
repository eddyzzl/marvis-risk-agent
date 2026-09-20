from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from marvis.data.asof_selection import select_asof_rows
from marvis.data.time_contracts import (
    AsOfJoinSpec, DatasetTimeContract, TimeColumn, TimestampEvidence,
)


def tc(name, *, basis="recorded", timezone="UTC"):
    return TimeColumn(column=name, timezone=timezone, evidence=TimestampEvidence(
        basis=basis, source_ref="source-receipt-v1", source_sha256="1" * 64,
        rule_version="inference-v1" if basis == "inferred" else None,
    ))


def contracts(**feature_overrides):
    return (
        DatasetTimeContract(dataset_id="decisions", content_hash="a" * 64, role="decision",
                            row_id="id", entity_columns=("subject",), decision_at=tc("decision")),
        DatasetTimeContract(dataset_id="features", content_hash="b" * 64, role="feature_snapshot",
                            row_id="id", entity_columns=("subject",), event_at=tc("event"),
                            version_column="version", **{"available_at": tc("available"), **feature_overrides}),
    )


def spec(**kwargs):
    return AsOfJoinSpec(**{"as_of": datetime(2026, 2, 1, tzinfo=UTC), "feature_columns": ("amount",), **kwargs})


def frames():
    return (
        pd.DataFrame({"id": ["d1", "d2"], "subject": ["s", "s"],
                      "decision": ["2026-01-10T12:00:00Z", "2026-01-11T12:00:00Z"]}),
        pd.DataFrame({"id": ["f1", "f2", "f3"], "subject": ["s"] * 3,
                      "event": ["2026-01-09T10:00:00Z"] * 3,
                      "available": ["2026-01-09T10:00:01Z", "2026-01-10T12:00:01Z", "2026-01-12T00:00:00Z"],
                      "version": [1, 2, 3], "amount": [10, 20, 999]}),
    )


def test_backfill_and_late_revision_use_exact_availability_not_event_date():
    d, f = frames()
    out = select_asof_rows(d, f, *contracts(), spec())
    assert out.frame.asof__amount.tolist() == [10, 20]
    assert out.memberships == ((0, 0), (1, 1))
    assert out.assurance == "verified"
    assert len(out.membership_sha256) == 64
    # A physical reordering does not change the feature matrix.
    reordered = select_asof_rows(d, f.iloc[::-1], *contracts(), spec())
    pd.testing.assert_frame_equal(out.frame, reordered.frame)


def test_same_instant_versions_choose_highest_version_without_date_rounding():
    d, f = frames()
    f["available"] = "2026-01-10T11:59:59.999999Z"
    assert select_asof_rows(d, f, *contracts(), spec()).frame.asof__amount.tolist() == [999, 999]


@pytest.mark.parametrize("change,match", [
    (lambda d, f: d.__setitem__("decision", ["2026-01-10T12:00:00Z"] * 2), "duplicate entity/decision"),
    (lambda d, f: f.__setitem__("id", ["same"] * 3), "duplicate row"),
    (lambda d, f: f.__setitem__("version", [1, 1, 3]), "duplicate entity/event/version"),
    (lambda d, f: f.__setitem__("version", [True, 2, 3]), "positive integers"),
    (lambda d, f: f.__setitem__("version", [1.0, 2.0, 3.0]), "positive integers"),
    (lambda d, f: f.__setitem__("available", ["2026-01-11Z", "2026-01-10Z", "2026-01-12Z"]), "timestamp"),
    (lambda d, f: f.__setitem__("available", ["2026-01-11", "2026-01-10", "2026-01-12"]), "version order"),
    (lambda d, f: f.__setitem__("available", "2026-01-01"), "precedes event"),
    (lambda d, f: d.__setitem__("subject", [None, "s"]), "non-null"),
    (lambda d, f: d.__setitem__("decision", "2026-03-01"), "duplicate entity/decision"),
])
def test_ambiguous_rows_fail_closed(change, match):
    d, f = frames()
    change(d, f)
    with pytest.raises(ValueError, match=match):
        select_asof_rows(d, f, *contracts(), spec())


def test_decision_after_global_cutoff_rejected_and_future_only_feature_unmatched():
    d, f = frames()
    with pytest.raises(ValueError, match="after replay as_of"):
        select_asof_rows(d, f, *contracts(), spec(as_of=datetime(2026, 1, 10, tzinfo=UTC)))
    f["event"] = "2026-01-20"
    f["available"] = "2026-01-21"
    with pytest.raises(ValueError, match="no eligible"):
        select_asof_rows(d, f, *contracts(), spec())
    out = select_asof_rows(d, f, *contracts(), spec(require_match=False))
    assert out.frame.asof__amount.isna().all()
    assert out.memberships == ((0, None), (1, None))


@pytest.mark.parametrize("basis,assurance", [("inferred", "inferred"), ("unknown", "unknown")])
def test_inferred_and_unknown_source_cannot_be_pit_verified(basis, assurance):
    d, f = frames()
    cs = contracts(available_at=tc("available", basis=basis))
    with pytest.raises(ValueError, match="recorded historical"):
        select_asof_rows(d, f, *cs, spec())
    assert select_asof_rows(d, f, *cs, spec(mode="exploration")).assurance == assurance


def test_missing_available_is_explicit_exploration_not_import_time_or_recorded_event():
    d, f = frames()
    f = f.drop(columns="available")
    cs = contracts(available_at=None)
    with pytest.raises(ValueError, match="recorded historical"):
        select_asof_rows(d, f, *cs, spec())
    out = select_asof_rows(d, f, *cs, spec(mode="exploration"))
    assert out.assurance == "unknown"
    assert "historical_available_at_missing" in out.reasons


def test_latest_visible_revision_cannot_fall_back_to_expired_or_cross_partition_old_version():
    d, f = frames()
    f["start"] = "2026-01-01"
    f["end"] = ["2026-02-01", "2026-01-11", "2026-02-01"]
    cs = contracts(effective_from=tc("start"), effective_to=tc("end"))
    with pytest.raises(ValueError, match="no eligible.*row 1"):
        select_asof_rows(d, f, *cs, spec())
    d["split"] = "train"
    f["split"] = ["train", "test", "test"]
    with pytest.raises(ValueError, match="no eligible.*row 1"):
        select_asof_rows(d, f, *contracts(), spec(partition_pairs=(("split", "split"),)))


def test_half_open_intervals_and_lookback_are_enforced():
    d, f = frames()
    f["start"], f["end"] = "2026-01-10T12:00:00Z", "2026-01-11T12:00:00Z"
    cs = contracts(effective_from=tc("start"), effective_to=tc("end"))
    out = select_asof_rows(d, f, *cs, spec(require_match=False))
    assert out.memberships == ((0, 0), (1, None))
    with pytest.raises(ValueError, match="no eligible"):
        select_asof_rows(d, f, *contracts(), spec(lookback_seconds=1))
    f["end"] = f["start"]
    with pytest.raises(ValueError, match="effective interval"):
        select_asof_rows(d, f, *cs, spec())


def test_future_aggregate_information_is_rejected_even_if_event_is_backdated():
    d, f = frames()
    f["start"], f["end"] = "2026-01-01", "2026-01-10"
    cs = contracts(window_start=tc("start"), window_end=tc("end"))
    with pytest.raises(ValueError, match="window crosses event"):
        select_asof_rows(d, f, *cs, spec())


@pytest.mark.parametrize("value", ["2026-11-01T01:30:00", "2026-03-08T02:30:00"])
def test_dst_ambiguous_or_nonexistent_local_clock_is_rejected(value):
    d, f = frames()
    d.loc[0, "decision"] = value
    dc, fc = contracts()
    dc = DatasetTimeContract.model_validate({**dc.model_dump(), "decision_at": tc("decision", timezone="America/New_York")})
    with pytest.raises(ValueError, match="invalid/ambiguous timestamp"):
        select_asof_rows(d, f, dc, fc, spec())


def test_explicit_timezone_normalizes_equivalent_instants():
    d, f = frames()
    d.loc[0, "decision"] = "2026-01-10T20:00:00+08:00"
    assert select_asof_rows(d, f, *contracts(), spec()).memberships == ((0, 0), (1, 1))


def test_strict_contract_roundtrip_hash_and_no_semantic_aliases():
    dc, fc = contracts()
    assert DatasetTimeContract.model_validate_json(fc.model_dump_json()) == fc
    assert len(fc.contract_sha256) == 64
    with pytest.raises(ValidationError, match="distinct time meanings"):
        contracts(available_at=tc("event"))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DatasetTimeContract.model_validate({**dc.model_dump(), "created_at": "2026-01-01"})
    with pytest.raises(ValidationError, match="IANA"):
        tc("event", timezone="LOCAL")
    with pytest.raises(ValidationError, match="timezone"):
        spec(as_of=datetime(2026, 1, 1))
    with pytest.raises(ValidationError, match="frozen source"):
        TimestampEvidence(basis="recorded")
    with pytest.raises(ValidationError, match="rule_version"):
        TimestampEvidence(basis="inferred", source_ref="x", source_sha256="a" * 64)
    with pytest.raises(ValidationError):
        spec(require_match="false")


def test_row_and_candidate_budgets_fail_closed():
    d, f = frames()
    with pytest.raises(ValueError, match="row budget"):
        select_asof_rows(d, f, *contracts(), spec(max_source_rows=2))
    with pytest.raises(ValueError, match="candidate check budget"):
        select_asof_rows(d, f, *contracts(), spec(max_candidate_checks=1))


def test_aggregate_window_cannot_cross_the_requested_lookback_partition():
    d, f = frames()
    f["start"], f["end"] = "2025-01-01", "2026-01-09"
    cs = contracts(window_start=tc("start"), window_end=tc("end"))
    with pytest.raises(ValueError, match="no eligible"):
        select_asof_rows(d, f, *cs, spec(lookback_seconds=7 * 86400))


def test_empty_cohort_cannot_create_vacuously_verified_output():
    d, f = frames()
    with pytest.raises(ValueError, match="must not be empty"):
        select_asof_rows(d.iloc[:0], f, *contracts(), spec())


def test_key_types_cannot_silently_coerce_subjects():
    d, f = frames()
    d["subject"] = 1
    f["subject"] = "1"
    with pytest.raises(ValueError, match="no eligible"):
        select_asof_rows(d, f, *contracts(), spec())


def test_no_implicit_deduplication_of_duplicate_rows():
    d, f = frames()
    f = pd.concat([f, f.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate row identity"):
        select_asof_rows(d, f, *contracts(), spec())

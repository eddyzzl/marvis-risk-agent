"""Two-level dedup (join spec §6): level-1 lossless whole-row dedup + level-2 same-key
value-conflict DETECTION (reported, never silently dropped)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marvis.data.dedup import two_level_dedup


def test_level1_drops_whole_row_duplicates_losslessly():
    frame = pd.DataFrame({"id": [1, 1, 2], "v": [10, 10, 20]})
    deduped, report = two_level_dedup(frame, ["id"])
    assert len(deduped) == 2  # the identical (1, 10) row collapsed
    assert report.safe_dropped == 1
    assert report.has_conflicts is False
    assert report.n_conflict_keys == 0


def test_level2_reports_same_key_value_conflicts_without_dropping():
    frame = pd.DataFrame({"id": [1, 1, 2], "v": [10, 11, 20]})  # key 1 disagrees on v
    deduped, report = two_level_dedup(frame, ["id"])
    assert len(deduped) == 3  # nothing dropped — a conflict is reported, never removed
    assert report.safe_dropped == 0
    assert report.has_conflicts is True
    assert report.n_conflict_keys == 1
    assert report.n_conflict_rows == 2
    assert report.conflict_columns == ("v",)
    assert report.sample_keys == ((1,),)


def test_mixed_safe_dups_and_conflicts():
    # key 1: two identical rows (safe drop 1); key 2: a value conflict; key 3: unique.
    frame = pd.DataFrame({"id": [1, 1, 2, 2, 3], "v": [10, 10, 20, 21, 30]})
    deduped, report = two_level_dedup(frame, ["id"])
    assert report.safe_dropped == 1
    assert len(deduped) == 4  # 5 - 1 safe drop; the conflicting key-2 rows BOTH survive
    assert report.n_conflict_keys == 1
    assert report.sample_keys == ((2,),)
    assert report.conflict_columns == ("v",)


def test_no_conflict_when_key_unique():
    _deduped, report = two_level_dedup(pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]}), ["id"])
    assert report.has_conflicts is False
    assert report.safe_dropped == 0


def test_composite_key_conflict():
    frame = pd.DataFrame({"id": [1, 1], "day": ["d", "d"], "v": [10, 11]})
    _deduped, report = two_level_dedup(frame, ["id", "day"])
    assert report.has_conflicts is True
    assert report.sample_keys == ((1, "d"),)
    assert report.conflict_columns == ("v",)


def test_report_is_deterministic():
    frame = pd.DataFrame({"id": [1, 1, 2, 2], "v": [10, 11, 20, 21]})
    assert two_level_dedup(frame, ["id"])[1] == two_level_dedup(frame, ["id"])[1]


def test_datetime_key_conflict_is_json_safe():
    """同人同天: the key includes a DATE column. sample_keys must be JSON-serializable
    (no pd.Timestamp) since it flows out through allow_nan=False JSONResponse."""
    frame = pd.DataFrame({
        "id": [1, 1],
        "day": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "v": [10, 11],  # same person, same day, disagreeing value
    })
    _deduped, report = two_level_dedup(frame, ["id", "day"])
    assert report.has_conflicts is True
    assert report.conflict_columns == ("v",)
    json.dumps(report.sample_keys, allow_nan=False)  # no Timestamp → serializable
    assert report.sample_keys == ((1, "2020-01-01T00:00:00"),)


def test_inf_float_key_is_json_safe():
    frame = pd.DataFrame({"ratio": [float("inf"), float("inf")], "v": [1, 2]})
    _deduped, report = two_level_dedup(frame, ["ratio"])
    assert report.has_conflicts is True
    json.dumps(report.sample_keys, allow_nan=False)  # inf coerced, not left non-finite
    assert report.sample_keys == ((None,),)


def test_nan_key_rows_are_not_treated_as_conflict():
    """A missing key can't match the 1:1 anchor, so NaN-key rows are unkeyable — not a
    same-entity value conflict — even when their values differ."""
    frame = pd.DataFrame({"id": [np.nan, np.nan, 1, 1], "v": [1, 2, 9, 9]})
    _deduped, report = two_level_dedup(frame, ["id"])
    assert report.has_conflicts is False
    assert report.n_conflict_keys == 0

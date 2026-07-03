"""S3 Commit 1: 表现期快照契约 + 合成数据 tests.

Covers validate_performance_frame typed-error文案 (缺列/坏月/未知桶/坏余额) and
generate_performance_frame determinism + the y=1 vs y=0 末月坏桶占比手算阈值断言.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marvis.data.errors import PerformanceFrameError
from marvis.data.performance import (
    PerformanceFrameContract,
    parse_snapshot_month,
    validate_performance_frame,
)
from marvis.sample_data import (
    PERFORMANCE_STATES,
    generate_performance_frame,
    generate_sample_frame,
)


_STATES = ["current", "M1", "M2", "M3+", "charged_off"]


def _good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["A", "A", "B", "B"],
            "snapshot_month": ["2025-01", "2025-02", "2025-01", "2025-02"],
            "bucket": ["current", "M1", "current", "current"],
            "balance": [1000.0, 900.0, 500.0, 450.0],
        }
    )


def test_validate_performance_frame_ok_returns_contract():
    contract = validate_performance_frame(
        _good_frame(),
        id_col="loan_id",
        snapshot_col="snapshot_month",
        bucket_col="bucket",
        states=_STATES,
        balance_col="balance",
    )
    assert isinstance(contract, PerformanceFrameContract)
    assert contract.row_count == 4
    assert contract.balance_col == "balance"
    # observed states preserve the declared order, only include present ones
    assert contract.observed_states == ("current", "M1")


def test_validate_performance_frame_missing_column():
    frame = _good_frame().drop(columns=["bucket"])
    with pytest.raises(PerformanceFrameError) as excinfo:
        validate_performance_frame(
            frame, id_col="loan_id", snapshot_col="snapshot_month", bucket_col="bucket", states=_STATES
        )
    detail = excinfo.value.to_detail()
    assert detail["kind"] == "performance_frame_invalid"
    assert detail["problem"] == "missing_columns"
    assert detail["missing_columns"] == ["bucket"]
    assert "缺少必需列" in detail["reason"]


def test_validate_performance_frame_bad_snapshot_month():
    frame = _good_frame()
    frame.loc[1, "snapshot_month"] = "not-a-month"
    with pytest.raises(PerformanceFrameError) as excinfo:
        validate_performance_frame(
            frame, id_col="loan_id", snapshot_col="snapshot_month", bucket_col="bucket", states=_STATES
        )
    detail = excinfo.value.to_detail()
    assert detail["problem"] == "bad_snapshot"
    assert detail["column"] == "snapshot_month"
    assert "not-a-month" in detail["samples"]


def test_validate_performance_frame_unknown_bucket():
    frame = _good_frame()
    frame.loc[1, "bucket"] = "M9-mystery"
    with pytest.raises(PerformanceFrameError) as excinfo:
        validate_performance_frame(
            frame, id_col="loan_id", snapshot_col="snapshot_month", bucket_col="bucket", states=_STATES
        )
    detail = excinfo.value.to_detail()
    assert detail["problem"] == "unknown_bucket"
    assert "M9-mystery" in detail["samples"]
    # message lists the declared states so the user can see what was expected
    assert "current" in detail["reason"]


def test_validate_performance_frame_bad_balance():
    frame = _good_frame()
    frame["balance"] = frame["balance"].astype(object)
    frame.loc[2, "balance"] = "abc"
    with pytest.raises(PerformanceFrameError) as excinfo:
        validate_performance_frame(
            frame,
            id_col="loan_id",
            snapshot_col="snapshot_month",
            bucket_col="bucket",
            states=_STATES,
            balance_col="balance",
        )
    detail = excinfo.value.to_detail()
    assert detail["problem"] == "bad_balance"
    assert "abc" in detail["samples"]


def test_validate_performance_frame_empty_states():
    with pytest.raises(PerformanceFrameError) as excinfo:
        validate_performance_frame(
            _good_frame(), id_col="loan_id", snapshot_col="snapshot_month", bucket_col="bucket", states=[]
        )
    assert excinfo.value.to_detail()["problem"] == "empty_states"


def test_parse_snapshot_month_variants():
    assert parse_snapshot_month("2025-03") == "2025-03"
    assert parse_snapshot_month("202503") == "2025-03"
    assert parse_snapshot_month(pd.Timestamp("2025-03-15")) == "2025-03"
    assert parse_snapshot_month("garbage") is None
    assert parse_snapshot_month(None) is None
    assert parse_snapshot_month(np.nan) is None


def test_generate_performance_frame_is_byte_deterministic():
    sample = generate_sample_frame()
    first = generate_performance_frame(sample)
    second = generate_performance_frame(sample)
    assert first.to_csv(index=False).encode() == second.to_csv(index=False).encode()


def test_generate_performance_frame_shape_and_buckets():
    sample = generate_sample_frame()
    frame = generate_performance_frame(sample, n_months=12)
    assert list(frame.columns) == ["loan_id", "snapshot_month", "bucket", "balance"]
    # 12 snapshots per loan
    assert len(frame) == len(sample) * 12
    # every bucket is a legal state
    assert set(frame["bucket"].unique()) <= set(PERFORMANCE_STATES)
    # frame passes its own contract with the canonical states order
    validate_performance_frame(
        frame,
        id_col="loan_id",
        snapshot_col="snapshot_month",
        bucket_col="bucket",
        states=list(PERFORMANCE_STATES),
        balance_col="balance",
    )


def test_generate_performance_frame_bad_loans_deteriorate_more():
    """手算阈值断言: y=1 群体末月坏桶(M2/M3+/charged_off)占比显著高于 y=0。"""
    sample = generate_sample_frame()
    frame = generate_performance_frame(sample)
    labels = {f"L{i:06d}": int(y) for i, y in enumerate(sample["y"].tolist())}
    last = frame.sort_values(["loan_id", "snapshot_month"]).groupby("loan_id").tail(1).copy()
    last["y"] = last["loan_id"].map(labels)
    bad_buckets = {"M2", "M3+", "charged_off"}
    share_bad = last.loc[last["y"] == 1, "bucket"].isin(bad_buckets).mean()
    share_good = last.loc[last["y"] == 0, "bucket"].isin(bad_buckets).mean()
    # generation is deterministic, so this is a fixed, reproducible separation
    assert share_bad > 0.5
    assert share_good < 0.2
    assert share_bad - share_good > 0.3


def test_charged_off_is_absorbing_with_zero_balance():
    sample = generate_sample_frame()
    frame = generate_performance_frame(sample)
    ordered = frame.sort_values(["loan_id", "snapshot_month"])
    for _loan_id, group in ordered.groupby("loan_id", sort=False):
        buckets = group["bucket"].tolist()
        balances = group["balance"].tolist()
        if "charged_off" in buckets:
            first = buckets.index("charged_off")
            # once charged off, stays charged off for all remaining snapshots
            assert all(b == "charged_off" for b in buckets[first:])
            # and balance is zeroed from that point on
            assert all(bal == 0.0 for bal in balances[first:])

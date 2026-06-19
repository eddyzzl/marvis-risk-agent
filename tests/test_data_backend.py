import hashlib

import pandas as pd
import pytest

from marvis.data.backend import (
    DataBackend,
    csv_rel,
    parquet_rel,
    sql_identifier,
    sql_string_literal,
)
from marvis.data.contracts import ColumnFingerprint, KeyPair
from marvis.data.errors import DataBackendError, DataSecurityError


def _fingerprint(*, is_hashed: bool, hash_type: str | None = None) -> ColumnFingerprint:
    return ColumnFingerprint(
        value_kind="hash" if is_hashed else "categorical",
        length_mode=None,
        regex_pattern=None,
        is_hashed=is_hashed,
        hash_type=hash_type,
        hex_case="lower" if is_hashed else None,
        date_format=None,
    )


def test_sql_helpers_escape_paths_and_require_allowed_columns(tmp_path):
    path = tmp_path / "customer's sample.csv"

    assert sql_string_literal(path.as_posix()).endswith("customer''s sample.csv'")
    assert csv_rel(path).startswith("read_csv_auto('")
    assert parquet_rel(tmp_path / "data.parquet").startswith("read_parquet('")
    assert sql_identifier('has " quote', {'has " quote'}) == '"has "" quote"'
    assert sql_identifier("select", {"select"}) == '"select"'

    with pytest.raises(DataSecurityError):
        sql_string_literal("bad\x00path")
    with pytest.raises(DataSecurityError):
        sql_identifier("missing", {"present"})


def test_backend_counts_columns_frames_and_uniqueness_for_csv_and_parquet(tmp_path):
    frame = pd.DataFrame({
        "id": [1, 2, 2],
        "select": ["a", "b", "c"],
        'has " quote': [10, 20, 30],
        "中文 列": ["x", "y", "z"],
    })
    csv_path = tmp_path / "customer's sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    backend = DataBackend(tmp_path)

    assert backend.row_count(csv_path) == 3
    assert backend.row_count(parquet_path) == 3
    assert backend.column_names(csv_path) == ["id", "select", 'has " quote', "中文 列"]
    assert backend.read_frame(csv_path, columns=["select"], nrows=2).shape == (2, 1)
    assert backend.distinct_count(parquet_path, ["id"]) == 2
    assert backend.is_key_unique(parquet_path, ["id"]) is False
    assert backend.is_key_unique(parquet_path, ["id", "select"]) is True

    with pytest.raises(DataSecurityError):
        backend.read_frame(csv_path, columns=["missing"])


def test_left_join_preserves_anchor_rows_and_supports_first_and_mean_dedup(tmp_path):
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    first_out = tmp_path / "joined_first.parquet"
    mean_out = tmp_path / "joined_mean.parquet"
    fanout_out = tmp_path / "joined_fanout.parquet"
    pd.DataFrame({"id": ["a", "b", "c"], "score": [1, 2, 3]}).to_parquet(
        anchor_path,
        index=False,
    )
    pd.DataFrame({"id": ["a", "a", "b"], "limit": [10, 12, 20]}).to_parquet(
        feature_path,
        index=False,
    )
    backend = DataBackend(tmp_path)
    key_pairs = [
        KeyPair(
            anchor_col="id",
            feature_col="id",
            match_method="exact",
            transform_side="both",
            match_rate=1.0,
            resolved_by="empirical",
        ),
    ]

    assert backend.left_join(
        anchor_path,
        feature_path,
        key_pairs,
        dedup_strategy="first",
        out_path=first_out,
    ) == 3
    first_joined = pd.read_parquet(first_out)
    assert first_joined["limit"].tolist()[:2] == [10, 20]
    assert pd.isna(first_joined["limit"].tolist()[2])

    assert backend.left_join(
        anchor_path,
        feature_path,
        key_pairs,
        dedup_strategy="agg_mean",
        out_path=mean_out,
    ) == 3
    mean_joined = pd.read_parquet(mean_out)
    assert mean_joined["limit"].tolist()[:2] == [11.0, 20.0]

    with pytest.raises(DataBackendError):
        backend.left_join(
            anchor_path,
            feature_path,
            key_pairs,
            dedup_strategy=None,
            out_path=fanout_out,
        )
    assert not fanout_out.exists()


def test_match_rate_normalizes_hash_case_and_dates(tmp_path):
    raw_fp = _fingerprint(is_hashed=False)
    md5_fp = _fingerprint(is_hashed=True, hash_type="md5")
    sha256_fp = _fingerprint(is_hashed=True, hash_type="sha256")
    date_fp = ColumnFingerprint(
        value_kind="date",
        length_mode=8,
        regex_pattern=None,
        is_hashed=False,
        hash_type=None,
        hex_case=None,
        date_format="%Y%m%d",
    )
    backend = DataBackend(tmp_path)

    raw_path = tmp_path / "raw.csv"
    md5_path = tmp_path / "md5.csv"
    sha256_path = tmp_path / "sha256.csv"
    pd.DataFrame({"customer_id": ["A1", "B2", "C3"]}).to_csv(raw_path, index=False)
    pd.DataFrame({
        "customer_hash": [
            hashlib.md5("A1".encode()).hexdigest().upper(),
            hashlib.md5("B2".encode()).hexdigest(),
        ],
    }).to_csv(md5_path, index=False)
    pd.DataFrame({
        "customer_hash": [
            hashlib.sha256(value.encode()).hexdigest()
            for value in ["A1", "B2", "C3"]
        ],
    }).to_csv(sha256_path, index=False)

    assert backend.match_rate_for_method(
        raw_path,
        ["customer_id"],
        md5_path,
        ["customer_hash"],
        method="hash:md5",
        key_fingerprints=[(raw_fp, md5_fp)],
        sample_n=10,
        seed=0,
    ) == (2, 3)
    assert backend.match_rate_for_method(
        raw_path,
        ["customer_id"],
        sha256_path,
        ["customer_hash"],
        method="hash:sha256",
        key_fingerprints=[(raw_fp, sha256_fp)],
        sample_n=10,
        seed=0,
    ) == (3, 3)

    anchor_dates = tmp_path / "anchor_dates.csv"
    feature_dates = tmp_path / "feature_dates.csv"
    pd.DataFrame({"date_key": ["20260101", "2026-01-02", "bad"]}).to_csv(
        anchor_dates,
        index=False,
    )
    pd.DataFrame({"date_key": ["2026-01-01", "2026/01/02"]}).to_csv(
        feature_dates,
        index=False,
    )
    assert backend.match_rate_for_method(
        anchor_dates,
        ["date_key"],
        feature_dates,
        ["date_key"],
        method="date",
        key_fingerprints=[(date_fp, date_fp)],
        sample_n=10,
        seed=0,
    ) == (2, 3)

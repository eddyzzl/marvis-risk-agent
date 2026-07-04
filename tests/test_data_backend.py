import hashlib
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from marvis.data.backend import (
    DUCKDB_MEMORY_LIMIT_ENV,
    DUCKDB_TEMP_DIR_NAME,
    DUCKDB_THREADS_ENV,
    DataBackend,
    connect_duckdb,
    csv_rel,
    default_duckdb_threads,
    duckdb_health,
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


def test_first_last_dedup_resolves_by_file_order_deterministically(tmp_path):
    """first/last resolve a same-key duplicate by FILE order (parquet file_row_number),
    reproducibly — not by non-deterministic scan order. File order here is (99, 11), the
    reverse of value order, so a content-based tie-break would give the opposite answer."""
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    pd.DataFrame({"id": ["a"], "score": [1]}).to_parquet(anchor_path, index=False)
    pd.DataFrame({"id": ["a", "a"], "limit": [99, 11]}).to_parquet(feature_path, index=False)
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
    first_out = tmp_path / "first.parquet"
    last_out = tmp_path / "last.parquet"
    backend.left_join(anchor_path, feature_path, key_pairs, dedup_strategy="first", out_path=first_out)
    backend.left_join(anchor_path, feature_path, key_pairs, dedup_strategy="last", out_path=last_out)
    assert pd.read_parquet(first_out)["limit"].tolist() == [99]  # file-first row
    assert pd.read_parquet(last_out)["limit"].tolist() == [11]   # file-last row


def test_left_join_raises_on_silent_row_loss_shrink(tmp_path, monkeypatch):
    """Spec §7 strict 1:1: a join that LOSES rows (shrink) must raise, not pass. A correct
    LEFT JOIN can't shrink, so simulate it by forcing the result count below the anchor —
    proving the defensive guard catches the shrink side, not just fan-out."""
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    out_path = tmp_path / "joined_shrink.parquet"
    pd.DataFrame({"id": ["a", "b"], "score": [1, 2]}).to_parquet(anchor_path, index=False)
    pd.DataFrame({"id": ["a", "b"], "limit": [10, 20]}).to_parquet(feature_path, index=False)
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
    real_row_count = backend.row_count
    monkeypatch.setattr(
        backend,
        "row_count",
        lambda path: real_row_count(path) - 1 if path == out_path else real_row_count(path),
    )
    with pytest.raises(DataBackendError, match="shrink"):
        backend.left_join(anchor_path, feature_path, key_pairs, dedup_strategy=None, out_path=out_path)


def test_first_last_dedup_survives_reserved_internal_column_names(tmp_path):
    """A feature column literally named 'file_row_number' or '__marvis_rn' must not break
    first/last dedup — the synthetic rank column is derived to avoid the collision, and
    file_row_number=true is skipped when that name is taken (the review caught both)."""
    anchor_path = tmp_path / "anchor.parquet"
    pd.DataFrame({"id": ["a", "b"], "score": [1, 2]}).to_parquet(anchor_path, index=False)
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
    for bad_col in ("file_row_number", "__marvis_rn"):
        feature_path = tmp_path / f"feat_{bad_col}.parquet"
        # 'id'=a is non-unique so the first/last dedup path actually runs.
        pd.DataFrame({"id": ["a", "a", "b"], bad_col: [10, 11, 20]}).to_parquet(feature_path, index=False)
        out_path = tmp_path / f"out_{bad_col}.parquet"
        backend.left_join(anchor_path, feature_path, key_pairs, dedup_strategy="first", out_path=out_path)
        joined = pd.read_parquet(out_path)
        # dedup ran (one row per anchor key) and the real data column survived intact
        assert joined[bad_col].tolist() == [10, 20]
        assert "__marvis_rn_" not in joined.columns  # the synthetic rank never leaks


def test_agg_mean_preserves_non_numeric_columns_instead_of_nulling(tmp_path):
    """agg_mean averages numeric columns but must NOT silently NULL non-numeric ones (the
    old try_cast(DOUBLE) dropped them) — they take a deterministic max() instead."""
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    out_path = tmp_path / "joined.parquet"
    pd.DataFrame({"id": ["a", "b"], "score": [1, 2]}).to_parquet(anchor_path, index=False)
    pd.DataFrame(
        {"id": ["a", "a", "b"], "limit": [10, 20, 30], "grade": ["A", "B", "C"]}
    ).to_parquet(feature_path, index=False)
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
    backend.left_join(anchor_path, feature_path, key_pairs, dedup_strategy="agg_mean", out_path=out_path)
    joined = pd.read_parquet(out_path).set_index("id")
    assert joined.loc["a", "limit"] == 15.0  # numeric column averaged (10, 20)
    assert joined.loc["a", "grade"] == "B"   # non-numeric column preserved (max), not NULL
    assert pd.notna(joined.loc["a", "grade"])


def test_numeric_columns_does_not_treat_nested_types_as_numeric(tmp_path):
    feature_path = tmp_path / "feature.parquet"
    pd.DataFrame({
        "id": [1, 2],
        "amount": [10.5, 20.5],
        "integer_list": [[1, 2], [3, 4]],
        "note": ["a", "b"],
    }).to_parquet(feature_path, index=False)
    backend = DataBackend(tmp_path)

    assert backend.numeric_columns(feature_path) == {"id", "amount"}


def test_left_join_collision_safe_alias_across_multiple_feature_tables(tmp_path):
    """A feature column colliding with the anchor is renamed feature_{col}; a SECOND
    feature table colliding again must not duplicate/overwrite — it gets feature_{col}_2,
    and every original value is preserved."""
    anchor_path = tmp_path / "anchor.parquet"
    f1_path = tmp_path / "f1.parquet"
    f2_path = tmp_path / "f2.parquet"
    pd.DataFrame({"id": ["a", "b"], "score": [1, 2]}).to_parquet(anchor_path, index=False)
    pd.DataFrame({"id": ["a", "b"], "score": [10, 20]}).to_parquet(f1_path, index=False)
    pd.DataFrame({"id": ["a", "b"], "score": [100, 200]}).to_parquet(f2_path, index=False)
    backend = DataBackend(tmp_path)

    def key_pairs():
        return [
            KeyPair(
                anchor_col="id",
                feature_col="id",
                match_method="exact",
                transform_side="both",
                match_rate=1.0,
                resolved_by="empirical",
            ),
        ]

    step1 = tmp_path / "step1.parquet"
    step2 = tmp_path / "step2.parquet"
    backend.left_join(anchor_path, f1_path, key_pairs(), dedup_strategy=None, out_path=step1)
    backend.left_join(step1, f2_path, key_pairs(), dedup_strategy=None, out_path=step2)

    result = pd.read_parquet(step2)
    cols = list(result.columns)
    assert {"score", "feature_score", "feature_score_2"} <= set(cols)
    assert len(cols) == len(set(cols))  # no duplicate columns
    row = result.set_index("id").loc["a"]
    assert row["score"] == 1 and row["feature_score"] == 10 and row["feature_score_2"] == 100


def test_left_join_exact_normalizes_integral_float_keys_like_diagnostics(tmp_path):
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    out_path = tmp_path / "joined.parquet"
    pd.DataFrame({"id": [5.0, 6.0], "score": [0.1, 0.2]}).to_parquet(anchor_path, index=False)
    pd.DataFrame({"id": ["5", "6"], "limit": [100, 200]}).to_parquet(feature_path, index=False)
    backend = DataBackend(tmp_path)

    joined_rows = backend.left_join(
        anchor_path,
        feature_path,
        [
            KeyPair(
                anchor_col="id",
                feature_col="id",
                match_method="exact",
                transform_side="both",
                match_rate=1.0,
                resolved_by="test",
            )
        ],
        dedup_strategy=None,
        out_path=out_path,
    )

    joined = pd.read_parquet(out_path)
    assert joined_rows == 2
    assert joined["limit"].tolist() == [100, 200]


def test_match_rate_pushes_feature_key_scan_to_duckdb(tmp_path, monkeypatch):
    anchor_path = tmp_path / "anchor.csv"
    feature_path = tmp_path / "feature.csv"
    pd.DataFrame({"id": ["A1", "B2", "C3"]}).to_csv(anchor_path, index=False)
    pd.DataFrame({"customer_id": ["a1", "b2"], "wide_payload": ["x" * 1000, "y" * 1000]}).to_csv(
        feature_path,
        index=False,
    )
    fp = _fingerprint(is_hashed=False)
    backend = DataBackend(tmp_path)
    original_read_frame = backend.read_frame

    def tracked_read_frame(path, *args, **kwargs):
        if Path(path) == feature_path:
            raise AssertionError("feature match-rate path should stay inside DuckDB")
        return original_read_frame(path, *args, **kwargs)

    monkeypatch.setattr(backend, "read_frame", tracked_read_frame)

    assert backend.match_rate_for_method(
        anchor_path,
        ["id"],
        feature_path,
        ["customer_id"],
        method="exact_lower",
        key_fingerprints=[(fp, fp)],
        sample_n=10,
        seed=0,
    ) == (2, 3)


def test_match_rate_duckdb_path_resolves_relative_dataset_paths(tmp_path):
    anchor_path = tmp_path / "anchor.csv"
    feature_path = tmp_path / "feature.csv"
    pd.DataFrame({"id": ["A1", "B2", "C3"]}).to_csv(anchor_path, index=False)
    pd.DataFrame({"customer_id": ["a1", "b2"]}).to_csv(feature_path, index=False)
    fp = _fingerprint(is_hashed=False)
    backend = DataBackend(tmp_path)

    assert backend.match_rate_for_method(
        Path("anchor.csv"),
        ["id"],
        Path("feature.csv"),
        ["customer_id"],
        method="exact_lower",
        key_fingerprints=[(fp, fp)],
        sample_n=10,
        seed=0,
    ) == (2, 3)


def test_match_rate_falls_back_for_hash_methods_not_supported_by_duckdb(tmp_path, monkeypatch):
    raw_fp = _fingerprint(is_hashed=False)
    sha1_fp = _fingerprint(is_hashed=True, hash_type="sha1")
    anchor_path = tmp_path / "anchor.csv"
    feature_path = tmp_path / "feature.csv"
    pd.DataFrame({"id": ["A1", "B2", "C3"]}).to_csv(anchor_path, index=False)
    pd.DataFrame({
        "customer_hash": [
            hashlib.sha1(value.encode()).hexdigest()
            for value in ["A1", "C3"]
        ],
    }).to_csv(feature_path, index=False)
    backend = DataBackend(tmp_path)

    def fail_if_duckdb_used(*args, **kwargs):
        raise AssertionError("sha1 match-rate should use Python fallback")

    monkeypatch.setattr(backend, "_duckdb_match_rate_for_method", fail_if_duckdb_used)

    assert backend.match_rate_for_method(
        anchor_path,
        ["id"],
        feature_path,
        ["customer_hash"],
        method="hash:sha1",
        key_fingerprints=[(raw_fp, sha1_fp)],
        sample_n=10,
        seed=0,
    ) == (2, 3)


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


def test_python_match_rate_date_fallback_uses_same_formats_as_duckdb_join(tmp_path):
    date_fp = ColumnFingerprint(
        value_kind="date",
        length_mode=8,
        regex_pattern=None,
        is_hashed=False,
        hash_type=None,
        hex_case=None,
        date_format="%Y%m%d",
    )
    anchor_dates = tmp_path / "anchor_dates.csv"
    feature_dates = tmp_path / "feature_dates.feather"
    pd.DataFrame({"date_key": ["20260102"]}).to_csv(anchor_dates, index=False)
    pd.DataFrame({"date_key": ["Jan 2, 2026"]}).to_feather(feature_dates)

    assert DataBackend(tmp_path).match_rate_for_method(
        anchor_dates,
        ["date_key"],
        feature_dates,
        ["date_key"],
        method="date",
        key_fingerprints=[(date_fp, date_fp)],
        sample_n=10,
        seed=0,
    ) == (0, 1)


def _exact_key_pair() -> KeyPair:
    return KeyPair(
        anchor_col="id",
        feature_col="id",
        match_method="exact",
        transform_side="both",
        match_rate=1.0,
        resolved_by="test",
    )


def test_left_join_treats_blank_keys_as_missing_not_matchable(tmp_path):
    # T1-A5: a blank/whitespace-only join key is MISSING (nullif), so a blank-keyed anchor
    # row must fall through to NULL feature columns rather than wrongly attaching to the
    # blank-keyed feature row. Row count is preserved (LEFT JOIN keeps the anchor row).
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    out_path = tmp_path / "joined.parquet"
    pd.DataFrame({"id": ["", "   ", "A1"], "score": [0.1, 0.2, 0.3]}).to_parquet(
        anchor_path, index=False
    )
    pd.DataFrame({"id": ["", "A1"], "limit": [999, 100]}).to_parquet(
        feature_path, index=False
    )
    backend = DataBackend(tmp_path)

    joined_rows = backend.left_join(
        anchor_path,
        feature_path,
        [_exact_key_pair()],
        dedup_strategy=None,
        out_path=out_path,
    )

    joined = pd.read_parquet(out_path).sort_values("score").reset_index(drop=True)
    assert joined_rows == 3  # 1:1 preservation -- blank-key anchor rows stay in the result
    limits = joined["limit"].tolist()
    # blank '' and whitespace '   ' anchor rows are UNMATCHED (NULL), not attached to 999.
    assert pd.isna(limits[0])
    assert pd.isna(limits[1])
    assert limits[2] == 100  # 'A1' still matches its feature row


def test_match_rate_and_left_join_agree_on_blank_keys(tmp_path):
    # T1-A5: the match-rate diagnostic and the executed join must AGREE on a dataset with
    # blank keys (previously diverged: SQL join matched blank=blank, diagnostics excluded it).
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    out_path = tmp_path / "joined.parquet"
    pd.DataFrame({"id": ["", "   ", "A1"], "score": [0.1, 0.2, 0.3]}).to_parquet(
        anchor_path, index=False
    )
    pd.DataFrame({"id": ["", "A1"], "limit": [999, 100]}).to_parquet(
        feature_path, index=False
    )
    backend = DataBackend(tmp_path)
    fp = _fingerprint(is_hashed=False)

    matched, sampled = backend.match_rate_for_method(
        anchor_path,
        ["id"],
        feature_path,
        ["id"],
        method="exact",
        key_fingerprints=[(fp, fp)],
        sample_n=10,
        seed=0,
    )
    assert (matched, sampled) == (1, 3)  # only 'A1' matches; blanks excluded both sides

    backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    joined = pd.read_parquet(out_path)
    realized = int(joined["limit"].notna().sum())
    assert realized == matched  # diagnostic prediction equals executed-join matched count


def test_blank_feature_keys_not_reported_as_duplicate_collision(tmp_path):
    # T1-A5: two blank feature keys are MISSING, not a duplicate-key collision -- lock the
    # chosen semantics so a feature with two '' keys still reads as key-unique.
    feature_path = tmp_path / "feature.parquet"
    pd.DataFrame({"id": ["", "   ", "A1"], "limit": [1, 2, 3]}).to_parquet(
        feature_path, index=False
    )
    backend = DataBackend(tmp_path)
    assert backend.is_key_unique(feature_path, ["id"], key_pairs=[_exact_key_pair()]) is True


@pytest.mark.parametrize(
    "digits",
    ["1234567890123456", "12345678901234567", "123456789012345678", "1234567890123456789", "9999999999999999"],
)
def test_sql_and_python_value_text_agree_on_long_integral_float(digits):
    # T1-A6: for a float64-stored long integer id, the DuckDB SQL normalizer and the Python
    # normalizer must produce the SAME string (both the precision-rounded integer, no
    # scientific notation, no trailing .0) so SQL-vs-Python key comparisons cannot diverge.
    from marvis.data.backend import _sql_value_text, _value_text

    value = float(digits)
    sql_expr = _sql_value_text('a."x"')
    con = duckdb.connect()
    con.register("t", pd.DataFrame({"x": [value]}))
    sql_result = con.execute(f"SELECT {sql_expr} FROM t a").fetchone()[0]
    assert sql_result == _value_text(value)


def test_left_join_matches_long_float_ids_both_sides_float(tmp_path):
    # T1-A6: both sides store an 18-digit id as float64; exact join must attach the feature
    # (today: SQL renders sci-notation, Python renders rounded int -> silent no-match/shrink).
    anchor_path = tmp_path / "anchor.parquet"
    feature_path = tmp_path / "feature.parquet"
    out_path = tmp_path / "joined.parquet"
    ids = [123456789012345678.0, 223456789012345678.0]
    pd.DataFrame({"id": ids, "score": [0.1, 0.2]}).to_parquet(anchor_path, index=False)
    pd.DataFrame({"id": ids, "limit": [100, 200]}).to_parquet(feature_path, index=False)
    backend = DataBackend(tmp_path)

    joined_rows = backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    joined = pd.read_parquet(out_path).sort_values("score").reset_index(drop=True)
    assert joined_rows == 2
    assert joined["limit"].tolist() == [100, 200]


def test_distinct_count_long_float_ids_consistent_sql_and_python(tmp_path):
    # T1-A6: two ids that round to the SAME float64 collide; distinct_count reflects the
    # collision the SAME way on the DuckDB (parquet) and pandas (feather) branches.
    frame = pd.DataFrame({"id": [123456789012345678.0, 123456789012345679.0, 5.0]})
    parquet_path = tmp_path / "feature.parquet"
    feather_path = tmp_path / "feature.feather"
    frame.to_parquet(parquet_path, index=False)
    frame.to_feather(feather_path)
    backend = DataBackend(tmp_path)

    sql_distinct = backend.distinct_count(parquet_path, ["id"], key_pairs=[_exact_key_pair()])
    py_distinct = backend.distinct_count(feather_path, ["id"], key_pairs=[_exact_key_pair()])
    # ...678 and ...679 both round to ...680 -> 2 distinct join keys (the collision + 5.0)
    assert sql_distinct == py_distinct == 2


def test_match_rate_matches_left_join_for_zero_padded_csv_keys(tmp_path):
    # T1-B7: for a zero-padded CSV key, the match-rate diagnostic must equal the actual
    # left_join result. Previously the feature was read all_varchar ("007") while the anchor
    # was pandas-typed (007 -> int 7) and execution read both typed -> diagnostic and
    # execution disagreed. Unifying on the typed reader makes them agree.
    anchor_path = tmp_path / "anchor.csv"
    feature_path = tmp_path / "feature.csv"
    out_path = tmp_path / "joined.parquet"
    pd.DataFrame({"id": ["007", "012", "099"]}).to_csv(anchor_path, index=False)
    pd.DataFrame({"id": ["007", "012"], "val": [1, 2]}).to_csv(feature_path, index=False)
    backend = DataBackend(tmp_path)
    fp = _fingerprint(is_hashed=False)

    matched, sampled = backend.match_rate_for_method(
        anchor_path, ["id"], feature_path, ["id"],
        method="exact", key_fingerprints=[(fp, fp)], sample_n=10, seed=0,
    )
    backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    realized = int(pd.read_parquet(out_path)["val"].notna().sum())
    assert matched == realized == 2


def test_match_rate_feature_scan_uses_typed_reader_like_execution(tmp_path):
    # T1-B7: pin diagnostics and execution to the SAME reader so the all_varchar split can't
    # silently return -- _duckdb_text_rel must no longer exist as a reader on the backend.
    backend = DataBackend(tmp_path)
    assert not hasattr(backend, "_duckdb_text_rel")


def test_match_rate_wide_varchar_feature_still_works_under_typed_reader(tmp_path):
    # T1-B7: the one concrete reason all_varchar existed was wide payload columns; the typed
    # read_csv_auto reader must still handle a very wide VARCHAR feature column.
    anchor_path = tmp_path / "anchor.csv"
    feature_path = tmp_path / "feature.csv"
    pd.DataFrame({"id": ["A1", "B2", "C3"]}).to_csv(anchor_path, index=False)
    pd.DataFrame(
        {"customer_id": ["a1", "b2"], "wide_payload": ["x" * 1000, "y" * 1000]}
    ).to_csv(feature_path, index=False)
    backend = DataBackend(tmp_path)
    fp = _fingerprint(is_hashed=False)

    assert backend.match_rate_for_method(
        anchor_path, ["id"], feature_path, ["customer_id"],
        method="exact_lower", key_fingerprints=[(fp, fp)], sample_n=10, seed=0,
    ) == (2, 3)


def test_match_rates_for_methods_zero_padded_parity(tmp_path):
    # T1-B7: the batched align._resolve_by_data shape must also see a diagnostic rate that
    # equals the realized join for a zero-padded CSV key (anchor sampled inside DuckDB).
    anchor_path = tmp_path / "anchor.csv"
    feature_path = tmp_path / "feature.csv"
    out_path = tmp_path / "joined.parquet"
    pd.DataFrame({"id": ["007", "012", "099"]}).to_csv(anchor_path, index=False)
    pd.DataFrame({"id": ["007", "012"], "val": [1, 2]}).to_csv(feature_path, index=False)
    backend = DataBackend(tmp_path)
    fp = _fingerprint(is_hashed=False)

    rates = backend.match_rates_for_methods(
        anchor_path, "id", feature_path, "id",
        methods=["exact", "exact_lower"], key_fingerprints=[(fp, fp), (fp, fp)],
        sample_n=10, seed=0,
    )
    backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    realized = int(pd.read_parquet(out_path)["val"].notna().sum())
    for matched, sampled in rates:
        assert (matched, sampled) == (realized, 3) == (2, 3)


def _duckdb_setting_on(temp_directory, name: str) -> str:
    # TST-9c: settings live on the per-operation connection connect_duckdb()
    # opens, NOT the process-wide implicit default connection -- read the same
    # way the backend's operations do.
    with connect_duckdb(temp_directory) as conn:
        row = conn.execute(
            f"SELECT value FROM duckdb_settings() WHERE name = '{name}'"
        ).fetchone()
    return str(row[0]) if row is not None else ""


def _default_connection_setting(name: str) -> str:
    row = duckdb.sql(
        f"SELECT value FROM duckdb_settings() WHERE name = '{name}'"
    ).fetchone()
    return str(row[0]) if row is not None else ""


def test_data_backend_configures_duckdb_memory_limit_threads_and_temp_directory(
    tmp_path, monkeypatch
):
    """PERF-8 regression: DataBackend must apply memory_limit, threads, and a
    workspace-scoped temp_directory to every DuckDB connection it opens instead
    of leaving DuckDB's own defaults in place (~80% of RAM, all cores, no durable
    spill directory). TST-9c: the config lives on per-operation connections, not
    the shared implicit default connection."""
    monkeypatch.delenv(DUCKDB_MEMORY_LIMIT_ENV, raising=False)
    monkeypatch.delenv(DUCKDB_THREADS_ENV, raising=False)
    datasets_root = tmp_path / "workspace" / "datasets"
    datasets_root.mkdir(parents=True)

    DataBackend(datasets_root)

    expected_temp_dir = tmp_path / "workspace" / DUCKDB_TEMP_DIR_NAME
    assert expected_temp_dir.is_dir()
    assert _duckdb_setting_on(expected_temp_dir, "temp_directory") == str(expected_temp_dir)
    assert _duckdb_setting_on(expected_temp_dir, "threads") == str(default_duckdb_threads())
    # DuckDB reports memory_limit in a human-readable unit (e.g. "3.7 GiB" for the
    # "4GB" default), so assert it moved off the library's own huge default rather
    # than an exact string match.
    memory_limit = _duckdb_setting_on(expected_temp_dir, "memory_limit")
    assert memory_limit != ""
    assert "GiB" in memory_limit or "GB" in memory_limit
    gib_value = float(memory_limit.split()[0])
    assert gib_value <= 8.0

    # TST-9c: constructing a DataBackend must NOT mutate the process-wide implicit
    # default connection -- that shared connection is exactly the cross-upload
    # contention source this fix removes.
    assert _default_connection_setting("temp_directory") != str(expected_temp_dir)

    health = duckdb_health(expected_temp_dir)
    assert health["duckdb_temp_directory"] == str(expected_temp_dir)
    assert health["duckdb_threads"] == str(default_duckdb_threads())
    assert health["duckdb_memory_limit"] == memory_limit


def test_data_backend_honors_duckdb_env_var_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(DUCKDB_MEMORY_LIMIT_ENV, "777MB")
    monkeypatch.setenv(DUCKDB_THREADS_ENV, "3")
    datasets_root = tmp_path / "workspace" / "datasets"
    datasets_root.mkdir(parents=True)

    DataBackend(datasets_root)

    expected_temp_dir = tmp_path / "workspace" / DUCKDB_TEMP_DIR_NAME
    assert _duckdb_setting_on(expected_temp_dir, "threads") == "3"
    memory_limit = _duckdb_setting_on(expected_temp_dir, "memory_limit")
    assert "MiB" in memory_limit or "MB" in memory_limit

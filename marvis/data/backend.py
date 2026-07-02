from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from marvis.data.contracts import (
    DATE_FORMATS,
    LARGE_ROW_THRESHOLD,
    ColumnFingerprint,
    ConflictReport,
    KeyPair,
)
from marvis.data.errors import DataBackendError, DataSecurityError


SUPPORTED_FRAME_SUFFIXES = {".csv", ".parquet", ".feather"}
SUPPORTED_DUCKDB_SUFFIXES = {".csv", ".parquet"}
DUCKDB_NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
    "NUMERIC",
}

# PERF-8: DuckDB defaults to memory_limit roughly 80% of physical RAM and
# threads=all cores, with a temp_directory that does not point at a durable,
# workspace-scoped location -- on a single machine that also runs training
# subprocesses and (often) a local LLM, that starves everything else and, once
# a large JOIN exceeds the worker's RLIMIT, fails as an opaque process death
# instead of spilling to disk. Configure the shared default connection (the one
# every ``duckdb.sql(...)`` call in this module implicitly uses) once with
# conservative, overridable settings instead.
DUCKDB_MEMORY_LIMIT_ENV = "MARVIS_DUCKDB_MEMORY_LIMIT"
DUCKDB_THREADS_ENV = "MARVIS_DUCKDB_THREADS"
DEFAULT_DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_TEMP_DIR_NAME = ".duckdb_tmp"

_duckdb_config_lock = threading.Lock()
_duckdb_configured_temp_dirs: set[str] = set()


def default_duckdb_threads() -> int:
    """max(2, cpu_count // 2): leaves headroom for training subprocesses / a
    co-located local LLM instead of DuckDB claiming every core by default."""
    cpu_count = os.cpu_count() or 2
    return max(2, cpu_count // 2)


def duckdb_runtime_config(temp_directory: Path) -> dict[str, str]:
    """The PRAGMA values actually applied to the shared default connection, so
    callers (health/audit endpoints) can report what is in effect."""
    return {
        "memory_limit": os.environ.get(DUCKDB_MEMORY_LIMIT_ENV, DEFAULT_DUCKDB_MEMORY_LIMIT),
        "threads": str(
            os.environ.get(DUCKDB_THREADS_ENV) or default_duckdb_threads()
        ),
        "temp_directory": str(temp_directory),
    }


def configure_duckdb_defaults(temp_directory: Path) -> dict[str, str]:
    """Idempotently apply memory_limit / threads / temp_directory PRAGMAs to the
    process-wide default DuckDB connection (the one ``duckdb.sql(...)`` uses).
    Safe to call from multiple DataBackend instances / threads: guarded by a lock
    and skipped once a given temp_directory has already been configured, so
    concurrent threadpool requests (PERF-1) never race on ``SET`` statements."""
    key = str(temp_directory)
    if key in _duckdb_configured_temp_dirs:
        return duckdb_runtime_config(temp_directory)
    with _duckdb_config_lock:
        if key in _duckdb_configured_temp_dirs:
            return duckdb_runtime_config(temp_directory)
        temp_directory.mkdir(parents=True, exist_ok=True)
        config = duckdb_runtime_config(temp_directory)
        duckdb.sql(f"SET memory_limit={sql_string_literal(config['memory_limit'])}")
        duckdb.sql(f"SET threads={int(config['threads'])}")
        duckdb.sql(f"SET temp_directory={sql_string_literal(config['temp_directory'])}")
        _duckdb_configured_temp_dirs.add(key)
        return config


def duckdb_health() -> dict[str, object]:
    """Current effective PRAGMA values on the shared default connection, for
    ``/api/health`` (PERF-8 audit visibility)."""
    rows = duckdb.sql(
        "SELECT name, value FROM duckdb_settings() "
        "WHERE name IN ('memory_limit', 'threads', 'temp_directory')"
    ).fetchall()
    settings = {str(name): str(value) for name, value in rows}
    return {
        "duckdb_memory_limit": settings.get("memory_limit", ""),
        "duckdb_threads": settings.get("threads", ""),
        "duckdb_temp_directory": settings.get("temp_directory", ""),
    }


def sql_string_literal(value: str) -> str:
    if "\x00" in value:
        raise DataSecurityError("SQL string literal cannot contain NUL bytes")
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(name: str, allowed_columns: set[str]) -> str:
    if name not in allowed_columns:
        raise DataSecurityError(f"unknown column: {name}")
    return _quote_identifier(name)


def parquet_rel(path: Path) -> str:
    return f"read_parquet({sql_string_literal(path.as_posix())})"


def csv_rel(path: Path) -> str:
    return f"read_csv_auto({sql_string_literal(path.as_posix())})"


class DataBackend:
    def __init__(self, datasets_root: Path):
        self._root = Path(datasets_root)
        configure_duckdb_defaults(self._root.parent / DUCKDB_TEMP_DIR_NAME)

    def row_count(self, path: Path) -> int:
        path = self._resolve_path(path)
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_DUCKDB_SUFFIXES:
            row = duckdb.sql(
                f"SELECT count(*) FROM {self._duckdb_rel(path)}",
            ).fetchone()
            return int(row[0])
        if suffix == ".feather":
            import pyarrow.feather as feather

            return int(feather.read_table(path).num_rows)
        raise DataBackendError(f"unsupported dataset format: {path.suffix}")

    def numeric_columns(self, path: Path) -> set[str]:
        """Names of columns whose DuckDB type is numeric (used so agg_mean averages only
        real numbers and leaves non-numeric columns intact instead of NULLing them)."""
        path = self._resolve_path(path)
        if path.suffix.lower() not in SUPPORTED_DUCKDB_SUFFIXES:
            return set()
        rows = duckdb.sql(f"DESCRIBE SELECT * FROM {self._duckdb_rel(path)}").fetchall()
        return {str(row[0]) for row in rows if _is_duckdb_numeric_type(str(row[1]))}

    def conflict_report(
        self,
        path: Path,
        key_columns: Sequence[str],
        *,
        sample_key_limit: int = 50,
        row_limit: int = 5000,
        key_pairs: Sequence[KeyPair] | None = None,
    ) -> ConflictReport:
        from marvis.data.dedup import two_level_dedup

        path = self._resolve_path(path)
        if path.suffix.lower() not in SUPPORTED_DUCKDB_SUFFIXES:
            frame = self.read_frame(path)
            if key_pairs is not None:
                frame = self.with_transformed_key_columns(frame, key_pairs)
                transformed_keys = transformed_key_names(key_pairs)
                _deduped, report = two_level_dedup(frame, transformed_keys)
                return replace(report, key_columns=tuple(str(col) for col in key_columns))
            _deduped, report = two_level_dedup(frame, list(key_columns))
            return report
        allowed_columns = set(self.column_names(path))
        self._validate_columns(key_columns, allowed_columns)
        if key_pairs is not None:
            key_exprs = self._transformed_key_exprs(key_pairs, allowed_columns, side="feature")
            keys_sql = ", ".join(f"{expr} AS {alias}" for expr, alias in key_exprs)
            keys_select_sql = ", ".join(alias for _expr, alias in key_exprs)
            key_not_null = " AND ".join(f"{alias} IS NOT NULL" for _expr, alias in key_exprs) or "TRUE"
        else:
            keys_sql = ", ".join(sql_identifier(col, allowed_columns) for col in key_columns)
            keys_select_sql = keys_sql
            key_not_null = " AND ".join(
                f"{sql_identifier(col, allowed_columns)} IS NOT NULL"
                for col in key_columns
            ) or "TRUE"
        duplicate_groups = (
            f"SELECT {keys_sql}, count(*) AS __n "
            f"FROM {self._duckdb_rel(path)} "
            f"WHERE {key_not_null} "
            f"GROUP BY {keys_select_sql} "
            "HAVING count(*) > 1"
        )
        count_row = duckdb.sql(
            "SELECT count(*) AS n_keys, coalesce(sum(__n), 0) AS n_rows "
            f"FROM ({duplicate_groups})"
        ).fetchone()
        n_conflict_keys = int(count_row[0] or 0)
        n_conflict_rows = int(count_row[1] or 0)
        if n_conflict_keys == 0:
            return ConflictReport(
                key_columns=tuple(str(col) for col in key_columns),
                conflict_columns=(),
                n_conflict_keys=0,
                n_conflict_rows=0,
                safe_dropped=0,
                sample_keys=(),
            )

        if key_pairs is not None:
            key_exprs = self._transformed_key_exprs(key_pairs, allowed_columns, side="feature")
            key_aliases = [alias for _expr, alias in key_exprs]
            join_condition = " AND ".join(
                f"f_key.{alias} IS NOT DISTINCT FROM d.{alias}" for alias in key_aliases
            )
            sample_rows = duckdb.sql(
                "WITH duplicate_keys AS ("
                # duplicate_groups already projects the transformed keys under their
                # aliases, so re-select the aliases here (not the raw expressions again —
                # the raw source column no longer exists in this subquery's output).
                f"SELECT {keys_select_sql} FROM ({duplicate_groups}) ORDER BY {keys_select_sql} "
                f"LIMIT {int(sample_key_limit)}"
                "), feature_with_keys AS ("
                f"SELECT f.*, {keys_sql} FROM {self._duckdb_rel(path)} f"
                ") "
                f"SELECT f_key.* EXCLUDE ({', '.join(key_aliases)}) "
                "FROM feature_with_keys f_key "
                f"JOIN duplicate_keys d ON {join_condition} "
                f"LIMIT {int(row_limit)}"
            ).df()
            frame = self.with_transformed_key_columns(sample_rows, key_pairs)
            transformed_keys = transformed_key_names(key_pairs)
            _deduped, sample_report = two_level_dedup(frame, transformed_keys)
            return ConflictReport(
                key_columns=tuple(str(col) for col in key_columns),
                conflict_columns=sample_report.conflict_columns,
                n_conflict_keys=n_conflict_keys,
                n_conflict_rows=n_conflict_rows,
                safe_dropped=sample_report.safe_dropped,
                sample_keys=sample_report.sample_keys,
            )

        join_condition = " AND ".join(
            f"f.{sql_identifier(col, allowed_columns)} IS NOT DISTINCT FROM "
            f"d.{sql_identifier(col, allowed_columns)}"
            for col in key_columns
        )
        sample_rows = duckdb.sql(
            "WITH duplicate_keys AS ("
            f"SELECT {keys_sql} FROM ({duplicate_groups}) ORDER BY {keys_sql} "
            f"LIMIT {int(sample_key_limit)}"
            ") "
            f"SELECT f.* FROM {self._duckdb_rel(path)} f "
            f"JOIN duplicate_keys d ON {join_condition} "
            f"LIMIT {int(row_limit)}"
        ).df()
        _deduped, sample_report = two_level_dedup(sample_rows, list(key_columns))
        return ConflictReport(
            key_columns=tuple(str(col) for col in key_columns),
            conflict_columns=sample_report.conflict_columns,
            n_conflict_keys=n_conflict_keys,
            n_conflict_rows=n_conflict_rows,
            safe_dropped=sample_report.safe_dropped,
            sample_keys=sample_report.sample_keys,
        )

    def column_names(self, path: Path) -> list[str]:
        path = self._resolve_path(path)
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_DUCKDB_SUFFIXES:
            rows = duckdb.sql(f"DESCRIBE SELECT * FROM {self._duckdb_rel(path)}").fetchall()
            return [str(row[0]) for row in rows]
        if suffix == ".feather":
            import pyarrow.feather as feather

            return [str(name) for name in feather.read_table(path).schema.names]
        raise DataBackendError(f"unsupported dataset format: {path.suffix}")

    def read_frame(
        self,
        path: Path,
        *,
        columns: Sequence[str] | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        path = self._resolve_path(path)
        allowed_columns = set(self.column_names(path))
        selected = self._validate_columns(columns, allowed_columns)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(
                path,
                usecols=selected,
                nrows=nrows,
                encoding="utf-8-sig",
            )
        if suffix == ".parquet":
            if nrows is not None:
                cols_sql = self._select_columns_sql(selected, allowed_columns)
                query = f"SELECT {cols_sql} FROM {parquet_rel(path)} LIMIT {int(nrows)}"
                return duckdb.sql(query).df()
            return pd.read_parquet(path, columns=selected)
        if suffix == ".feather":
            frame = pd.read_feather(path, columns=selected)
            return frame.head(nrows) if nrows is not None else frame
        raise DataBackendError(f"unsupported dataset format: {path.suffix}")

    def sample_rows(self, path: Path, n: int, *, seed: int) -> pd.DataFrame:
        path = self._resolve_path(path)
        total = self.row_count(path)
        if total <= n:
            return self.read_frame(path)
        if total > LARGE_ROW_THRESHOLD and path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES:
            query = (
                f"SELECT * FROM {self._duckdb_rel(path)} "
                f"USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})"
            )
            return duckdb.sql(query).df()
        return self.read_frame(path).sample(n=int(n), random_state=int(seed))

    def distinct_count(
        self,
        path: Path,
        columns: list[str],
        *,
        key_pairs: Sequence[KeyPair] | None = None,
    ) -> int:
        path = self._resolve_path(path)
        if not columns:
            raise DataBackendError("distinct_count requires at least one column")
        allowed_columns = set(self.column_names(path))
        self._validate_columns(columns, allowed_columns)
        if path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES:
            if key_pairs is not None:
                key_exprs = self._transformed_key_exprs(key_pairs, allowed_columns, side="feature")
                cols_sql = ", ".join(expr for expr, _alias in key_exprs)
            else:
                cols_sql = ", ".join(sql_identifier(col, allowed_columns) for col in columns)
            query = (
                "SELECT count(*) FROM ("
                f"SELECT DISTINCT {cols_sql} FROM {self._duckdb_rel(path)}"
                ")"
            )
            return int(duckdb.sql(query).fetchone()[0])
        if key_pairs is not None:
            frame = self.with_transformed_key_columns(self.read_frame(path), key_pairs)
            return int(frame[transformed_key_names(key_pairs)].drop_duplicates().shape[0])
        frame = self.read_frame(path, columns=columns)
        return int(frame.drop_duplicates().shape[0])

    def is_key_unique(
        self,
        path: Path,
        columns: list[str],
        *,
        key_pairs: Sequence[KeyPair] | None = None,
    ) -> bool:
        return self.distinct_count(path, columns, key_pairs=key_pairs) == self.row_count(path)

    def left_join(
        self,
        anchor_path: Path,
        feature_path: Path,
        key_pairs: list[KeyPair],
        *,
        dedup_strategy: str | None,
        out_path: Path,
    ) -> int:
        anchor_path = self._resolve_path(anchor_path)
        feature_path = self._resolve_path(feature_path)
        out_path = self._resolve_path(out_path)
        if not key_pairs:
            raise DataBackendError("left_join requires at least one key pair")
        if anchor_path.suffix.lower() not in SUPPORTED_DUCKDB_SUFFIXES:
            raise DataBackendError("left_join requires a CSV or parquet anchor")
        if feature_path.suffix.lower() not in SUPPORTED_DUCKDB_SUFFIXES:
            raise DataBackendError("left_join requires a CSV or parquet feature")

        anchor_columns = set(self.column_names(anchor_path))
        feature_columns = set(self.column_names(feature_path))
        feature_key_columns = [pair.feature_col for pair in key_pairs]
        self._validate_columns([pair.anchor_col for pair in key_pairs], anchor_columns)
        self._validate_columns(feature_key_columns, feature_columns)

        if dedup_strategy == "abort" and not self.is_key_unique(
            feature_path, feature_key_columns, key_pairs=key_pairs
        ):
            raise DataBackendError("feature keys are not unique")

        feature_rel = self._dedup_feature_rel(
            feature_path,
            feature_columns,
            feature_key_columns,
            dedup_strategy,
            key_pairs=key_pairs,
        )
        on_sql = " AND ".join(
            self._join_condition(pair, anchor_columns, feature_columns)
            for pair in key_pairs
        )
        feature_select = self._feature_projection(
            feature_columns,
            feature_key_columns,
            anchor_columns,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        query = (
            "COPY ("
            f"SELECT a.*{feature_select} "
            f"FROM {self._duckdb_rel(anchor_path)} a "
            f"LEFT JOIN ({feature_rel}) b ON {on_sql}"
            f") TO {sql_string_literal(out_path.as_posix())} (FORMAT parquet)"
        )
        duckdb.sql(query)
        result_rows = self.row_count(out_path)
        anchor_rows = self.row_count(anchor_path)
        # The sample must stay 1:1 (spec §7): assert strict equality, catching BOTH
        # fan-out (key not unique) AND silent row loss / shrink (a bad key transform or
        # an inner-join-like regression). A correct LEFT JOIN keeps anchor_rows exactly.
        if result_rows != anchor_rows:
            out_path.unlink(missing_ok=True)
            kind = "fan-out" if result_rows > anchor_rows else "row loss (shrink)"
            raise DataBackendError(
                f"left_join {kind}: produced {result_rows} rows from {anchor_rows} "
                f"anchor rows (must be 1:1)",
            )
        return result_rows

    def match_rate_for_method(
        self,
        anchor_path: Path,
        anchor_keys: Sequence[str],
        feature_path: Path,
        feature_keys: Sequence[str],
        *,
        method: str | Sequence[str],
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
    ) -> tuple[int, int]:
        anchor_path = self._resolve_path(anchor_path)
        feature_path = self._resolve_path(feature_path)
        if len(anchor_keys) != len(feature_keys):
            raise DataBackendError("anchor_keys and feature_keys must have the same length")
        if len(anchor_keys) != len(key_fingerprints):
            raise DataBackendError("key_fingerprints must align with key columns")
        methods = _methods_for_keys(method, len(anchor_keys))

        self._validate_columns(anchor_keys, set(self.column_names(anchor_path)))
        self._validate_columns(feature_keys, set(self.column_names(feature_path)))
        anchor_frame = self.sample_rows(anchor_path, sample_n, seed=seed)
        if (
            feature_path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES
            and _duckdb_supports_match_methods(methods)
        ):
            return self._duckdb_match_rate_for_method(
                anchor_frame,
                feature_path,
                anchor_keys,
                feature_keys,
                methods=methods,
                key_fingerprints=key_fingerprints,
            )
        feature_frame = self.read_frame(feature_path, columns=feature_keys)
        fingerprint_pairs = [
            _fingerprint_pair(item)
            for item in key_fingerprints
        ]
        feature_key_set = {
            tuple(
                self._normalize_value(
                    row[feature_col],
                    method=pair_method,
                    side="feature",
                    fingerprint=feature_fp,
                )
                for feature_col, pair_method, (_, feature_fp) in zip(feature_keys, methods, fingerprint_pairs)
            )
            for _, row in feature_frame.iterrows()
        }
        feature_key_set = {
            key
            for key in feature_key_set
            if all(value is not None for value in key)
        }

        matched = 0
        for _, row in anchor_frame.iterrows():
            key = tuple(
                self._normalize_value(
                    row[anchor_col],
                    method=pair_method,
                    side="anchor",
                    fingerprint=anchor_fp,
                )
                for anchor_col, pair_method, (anchor_fp, _) in zip(anchor_keys, methods, fingerprint_pairs)
            )
            if all(value is not None for value in key) and key in feature_key_set:
                matched += 1
        return matched, int(anchor_frame.shape[0])

    def _duckdb_match_rate_for_method(
        self,
        anchor_frame: pd.DataFrame,
        feature_path: Path,
        anchor_keys: Sequence[str],
        feature_keys: Sequence[str],
        *,
        methods: Sequence[str],
        key_fingerprints: Sequence[Any],
    ) -> tuple[int, int]:
        fingerprint_pairs = [_fingerprint_pair(item) for item in key_fingerprints]
        anchor_columns = set(str(column) for column in anchor_frame.columns)
        feature_columns = set(self.column_names(feature_path))
        anchor_exprs = []
        feature_exprs = []
        for index, (anchor_col, feature_col, method, (anchor_fp, feature_fp)) in enumerate(
            zip(anchor_keys, feature_keys, methods, fingerprint_pairs, strict=True),
        ):
            anchor_expr = _sql_normalized_key(
                method,
                "a." + sql_identifier(anchor_col, anchor_columns),
                fingerprint=anchor_fp,
            )
            feature_expr = _sql_normalized_key(
                method,
                "b." + sql_identifier(feature_col, feature_columns),
                fingerprint=feature_fp,
            )
            anchor_exprs.append(f"{anchor_expr} AS __key_{index}")
            feature_exprs.append(f"{feature_expr} AS __key_{index}")
        anchor_projection = ", ".join(anchor_exprs)
        feature_projection = ", ".join(feature_exprs)
        key_columns = [f"__key_{index}" for index in range(len(anchor_exprs))]
        join_condition = " AND ".join(f"a.{column} = f.{column}" for column in key_columns)
        anchor_not_null = " AND ".join(f"a.{column} IS NOT NULL" for column in key_columns)
        feature_rel = self._duckdb_text_rel(feature_path)
        query = (
            "WITH anchor_keys AS ("
            f"SELECT {anchor_projection} FROM anchor_sample a"
            "), feature_keys AS ("
            f"SELECT DISTINCT {feature_projection} FROM {feature_rel} b"
            "), joined AS ("
            f"SELECT a.*, f.{key_columns[0]} AS __matched_key "
            "FROM anchor_keys a "
            f"LEFT JOIN feature_keys f ON {join_condition}"
            ") "
            "SELECT count(*) FILTER ("
            f"WHERE {anchor_not_null} AND __matched_key IS NOT NULL"
            ") FROM joined a"
        )
        with duckdb.connect(database=":memory:") as conn:
            conn.register("anchor_sample", anchor_frame)
            matched = conn.execute(query).fetchone()[0]
        return int(matched), int(anchor_frame.shape[0])

    def _resolve_path(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self._root / path

    def _duckdb_rel(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return csv_rel(path)
        if suffix == ".parquet":
            return parquet_rel(path)
        raise DataBackendError(f"unsupported DuckDB dataset format: {path.suffix}")

    def _duckdb_text_rel(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return f"read_csv_auto({sql_string_literal(path.as_posix())}, all_varchar=true)"
        if suffix == ".parquet":
            return parquet_rel(path)
        raise DataBackendError(f"unsupported DuckDB dataset format: {path.suffix}")

    def _validate_columns(
        self,
        columns: Sequence[str] | None,
        allowed_columns: set[str],
    ) -> list[str] | None:
        if columns is None:
            return None
        selected = [str(column) for column in columns]
        for column in selected:
            sql_identifier(column, allowed_columns)
        return selected

    def _select_columns_sql(
        self,
        columns: Sequence[str] | None,
        allowed_columns: set[str],
    ) -> str:
        if columns is None:
            return "*"
        return ", ".join(sql_identifier(column, allowed_columns) for column in columns)

    def _dedup_feature_rel(
        self,
        feature_path: Path,
        feature_columns: set[str],
        feature_key_columns: Sequence[str],
        dedup_strategy: str | None,
        *,
        key_pairs: Sequence[KeyPair] | None = None,
    ) -> str:
        rel = self._duckdb_rel(feature_path)
        if dedup_strategy in (None, "abort"):
            return f"SELECT * FROM {rel}"
        if key_pairs is not None:
            partition_sql = ", ".join(
                expr for expr, _alias in
                self._transformed_key_exprs(key_pairs, feature_columns, side="feature")
            )
        else:
            partition_sql = ", ".join(
                sql_identifier(column, feature_columns)
                for column in feature_key_columns
            )
        if dedup_strategy in {"first", "last"}:
            order = "ASC" if dedup_strategy == "first" else "DESC"
            return self._first_last_dedup_sql(feature_path, rel, partition_sql, feature_columns, order)
        if dedup_strategy in {"agg_mean", "agg_max"}:
            numeric = self.numeric_columns(feature_path) if dedup_strategy == "agg_mean" else set()
            return self._agg_dedup_sql(
                rel, partition_sql, feature_columns, feature_key_columns, dedup_strategy, numeric
            )
        raise DataBackendError(f"unsupported dedup_strategy: {dedup_strategy}")

    def _first_last_dedup_sql(
        self,
        feature_path: Path,
        rel: str,
        key_sql: str,
        feature_columns: set[str],
        order: str,
    ) -> str:
        # row_number() OVER () has no ordering and is non-reproducible across DuckDB
        # versions/parallelism, so 'first'/'last' were non-deterministic. Prefer parquet's
        # physical file_row_number (true file order); for any non-parquet path fall back to
        # full-row content order, which is also deterministic.
        # The synthetic rank column name is derived to be PROVABLY absent from the feature
        # columns so it can never collide with (and silently drop or shadow) a real column.
        # key_sql is expressed in the TRANSFORMED key space (matches the actual JOIN
        # condition), so rows whose raw keys differ but transform to the same value
        # (e.g. 'ABC' vs 'abc' under exact_lower) land in the same partition.
        rank = _unique_internal_name("__marvis_rn", feature_columns)
        # file_row_number=true is rejected by DuckDB when a real column is already named
        # 'file_row_number', so only use it when that name is free.
        if feature_path.suffix.lower() == ".parquet" and "file_row_number" not in feature_columns:
            rownum_rel = f"read_parquet({sql_string_literal(feature_path.as_posix())}, file_row_number=true)"
            return (
                f"SELECT * EXCLUDE (file_row_number, {rank}) FROM ("
                "SELECT *, row_number() OVER ("
                f"PARTITION BY {key_sql} ORDER BY file_row_number {order}"
                f") AS {rank} FROM {rownum_rel}"
                f") WHERE {rank} = 1"
            )
        content_order = ", ".join(
            f"{sql_identifier(column, feature_columns)} {order}" for column in sorted(feature_columns)
        )
        return (
            f"SELECT * EXCLUDE ({rank}) FROM ("
            "SELECT *, row_number() OVER ("
            f"PARTITION BY {key_sql} ORDER BY {content_order}"
            f") AS {rank} FROM {rel}"
            f") WHERE {rank} = 1"
        )

    def _agg_dedup_sql(
        self,
        rel: str,
        key_sql: str,
        feature_columns: set[str],
        feature_key_columns: Sequence[str],
        dedup_strategy: str,
        numeric_columns: set[str] = frozenset(),
    ) -> str:
        # key_sql is expressed in the TRANSFORMED key space (matches the actual JOIN
        # condition). Raw key columns can disagree within a transformed-key group (e.g.
        # 'ABC' vs 'abc' under exact_lower), so they are no longer valid GROUP BY
        # projections on their own — aggregate them deterministically like any other
        # non-numeric column instead of selecting the raw column directly.
        projections = []
        for column in feature_key_columns:
            ident = sql_identifier(column, feature_columns)
            alias = _quote_identifier(column)
            projections.append(f"max({ident}) AS {alias}")
        for column in sorted(feature_columns - set(feature_key_columns)):
            ident = sql_identifier(column, feature_columns)
            alias = _quote_identifier(column)
            if dedup_strategy == "agg_mean" and column in numeric_columns:
                projections.append(f"avg({ident}) AS {alias}")
            else:
                # Non-numeric columns under agg_mean (and all columns under agg_max) take a
                # deterministic max() rather than being silently NULLed by try_cast(DOUBLE).
                projections.append(f"max({ident}) AS {alias}")
        projection_sql = ", ".join(projections)
        return f"SELECT {projection_sql} FROM {rel} GROUP BY {key_sql}"

    def _join_condition(
        self,
        pair: KeyPair,
        anchor_columns: set[str],
        feature_columns: set[str],
    ) -> str:
        anchor_col = "a." + sql_identifier(pair.anchor_col, anchor_columns)
        feature_col = "b." + sql_identifier(pair.feature_col, feature_columns)
        return (
            f"{_sql_transform(pair.match_method, anchor_col, side='anchor', pair=pair)} = "
            f"{_sql_transform(pair.match_method, feature_col, side='feature', pair=pair)}"
        )

    def _transformed_key_exprs(
        self,
        key_pairs: Sequence[KeyPair],
        columns: set[str],
        *,
        side: str,
    ) -> list[tuple[str, str]]:
        """SQL (expression, alias) pairs for each key pair's column, transformed with the
        SAME ``_sql_transform`` used by the actual JOIN condition (:meth:`_join_condition`).
        Uniqueness/dedup must compare keys the way the JOIN will compare them — an
        `exact_lower`/`hash`/`date` transform can make raw-distinct values collide (or vice
        versa), so grouping on raw columns silently disagrees with what the JOIN does."""
        exprs = []
        for index, pair in enumerate(key_pairs):
            column = pair.feature_col if side == "feature" else pair.anchor_col
            source = sql_identifier(column, columns)
            alias = _unique_internal_name(f"__marvis_key_{index}", columns)
            expr = _sql_transform(pair.match_method, source, side=side, pair=pair)
            exprs.append((expr, alias))
        return exprs

    def with_transformed_key_columns(
        self,
        frame: pd.DataFrame,
        key_pairs: Sequence[KeyPair],
        *,
        side: str = "feature",
    ) -> pd.DataFrame:
        """Pandas-fallback counterpart to :meth:`_transformed_key_exprs`: adds one column per
        key pair holding the value transformed the same way the DuckDB JOIN condition would
        (via ``_transformed_key_value``), so uniqueness/dedup computed on these columns stays
        consistent with the transformed-key-space JOIN even off the DuckDB code path."""
        frame = frame.copy()
        for name, pair in zip(transformed_key_names(key_pairs), key_pairs):
            column = pair.feature_col if side == "feature" else pair.anchor_col
            frame[name] = [
                _transformed_key_value(value, method=pair.match_method, side=side, pair=pair)
                for value in frame[column]
            ]
        return frame

    def _feature_projection(
        self,
        feature_columns: set[str],
        feature_key_columns: Sequence[str],
        anchor_columns: set[str],
    ) -> str:
        selections = []
        # Collision-safe aliasing: a feature column that collides with the (accumulating)
        # anchor is renamed feature_{col}; if THAT is also taken — e.g. a second feature
        # table in the same plan already contributed a feature_{col} — disambiguate with a
        # numeric suffix so columns are never silently overwritten or duplicated.
        taken = set(anchor_columns)
        for column in sorted(feature_columns - set(feature_key_columns)):
            source = "b." + sql_identifier(column, feature_columns)
            if column not in taken:
                alias = column
            else:
                alias = f"feature_{column}"
                suffix = 2
                while alias in taken:
                    alias = f"feature_{column}_{suffix}"
                    suffix += 1
            taken.add(alias)
            selections.append(f"{source} AS {_quote_identifier(alias)}")
        return ", " + ", ".join(selections) if selections else ""

    def _normalize_value(
        self,
        value: Any,
        *,
        method: str,
        side: str,
        fingerprint: ColumnFingerprint,
    ) -> str | None:
        if pd.isna(value):
            return None
        text = _value_text(value)
        if not text:
            return None
        if method == "exact":
            return text
        if method == "exact_lower":
            return text.lower()
        if method == "date":
            return _canonical_date(text)
        if method.startswith("hash:"):
            algorithm = method.split(":", 1)[1]
            if fingerprint.is_hashed:
                return text.lower()
            return _hash_text(text, algorithm)
        raise DataBackendError(f"unsupported match method: {method}")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _unique_internal_name(base: str, columns: set[str]) -> str:
    """A synthetic SQL column name guaranteed absent from ``columns`` (append underscores
    until unique), so an internal rank/rowid column never collides with a real feature
    column and silently drop or shadow it."""
    name = base
    while name in columns:
        name += "_"
    return name


def _value_text(value: Any) -> str:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if number.is_integer():
            return str(int(number))
    return str(value).strip()


def _methods_for_keys(method: str | Sequence[str], key_count: int) -> list[str]:
    if isinstance(method, str):
        return [method] * key_count
    methods = [str(item) for item in method]
    if len(methods) != key_count:
        raise DataBackendError("method list must align with key columns")
    return methods


def _duckdb_supports_match_methods(methods: Sequence[str]) -> bool:
    for method in methods:
        if not method.startswith("hash:"):
            continue
        algorithm = method.split(":", 1)[1]
        if algorithm not in {"md5", "sha256"}:
            return False
    return True


def _is_duckdb_numeric_type(type_name: str) -> bool:
    base = type_name.strip().upper().split("(", 1)[0].strip()
    return base in DUCKDB_NUMERIC_TYPES


def _sql_transform(method: str, expression: str, *, side: str, pair: KeyPair) -> str:
    trimmed = _sql_value_text(expression)
    if method == "exact":
        return trimmed
    if method == "exact_lower":
        return f"lower({trimmed})"
    if method == "date":
        date_exprs = ", ".join(
            f"try_strptime({trimmed}, {sql_string_literal(fmt)})"
            for fmt in DATE_FORMATS
        )
        return f"strftime(coalesce({date_exprs}), '%Y-%m-%d')"
    if method.startswith("hash:"):
        algorithm = method.split(":", 1)[1]
        if side == pair.transform_side or pair.transform_side == "both":
            if algorithm not in {"md5", "sha256"}:
                raise DataBackendError(f"DuckDB hash method is not supported: {method}")
            return f"lower({algorithm}({trimmed}))"
        return f"lower({trimmed})"
    raise DataBackendError(f"unsupported match method: {method}")


def _sql_value_text(expression: str) -> str:
    trimmed = f"trim(CAST({expression} AS VARCHAR))"
    return (
        "CASE "
        f"WHEN regexp_matches({trimmed}, '^-?[0-9]+\\.0+$') "
        f"THEN regexp_replace({trimmed}, '\\.0+$', '') "
        f"ELSE {trimmed} END"
    )


def _sql_normalized_key(method: str, expression: str, *, fingerprint: ColumnFingerprint) -> str:
    text = f"nullif({_sql_value_text(expression)}, '')"
    if method == "exact":
        return text
    if method == "exact_lower":
        return f"lower({text})"
    if method == "date":
        date_exprs = ", ".join(
            f"try_strptime({text}, {sql_string_literal(fmt)})"
            for fmt in DATE_FORMATS
        )
        return f"strftime(coalesce({date_exprs}), '%Y-%m-%d')"
    if method.startswith("hash:"):
        algorithm = method.split(":", 1)[1]
        if algorithm not in {"md5", "sha256"}:
            raise DataBackendError(f"DuckDB hash method is not supported: {method}")
        if fingerprint.is_hashed:
            return f"lower({text})"
        return f"lower({algorithm}({text}))"
    raise DataBackendError(f"unsupported match method: {method}")


def _fingerprint_pair(item: Any) -> tuple[ColumnFingerprint, ColumnFingerprint]:
    if isinstance(item, ColumnFingerprint):
        return item, item
    if isinstance(item, Sequence) and len(item) == 2:
        anchor_fp, feature_fp = item
        if isinstance(anchor_fp, ColumnFingerprint) and isinstance(feature_fp, ColumnFingerprint):
            return anchor_fp, feature_fp
    raise DataBackendError("key_fingerprints entries must be ColumnFingerprint pairs")


def _canonical_date(text: str) -> str | None:
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed.strftime("%Y-%m-%d")
    return None


def _hash_text(text: str, algorithm: str) -> str:
    if algorithm not in hashlib.algorithms_available:
        raise DataBackendError(f"unsupported hash method: hash:{algorithm}")
    digest = hashlib.new(algorithm)
    digest.update(text.encode("utf-8"))
    return digest.hexdigest().lower()


def _transformed_key_value(value: Any, *, method: str, side: str, pair: KeyPair) -> str | None:
    """Pandas-fallback counterpart to ``_sql_transform``: applies the SAME match-method
    transform (and the same ``pair.transform_side`` hash-skip rule) to a single Python
    value, so a uniqueness/dedup check computed off the DuckDB code path still agrees with
    what the actual JOIN condition would compute for this value."""
    if pd.isna(value):
        return None
    text = _value_text(value)
    if not text:
        return None
    if method == "exact":
        return text
    if method == "exact_lower":
        return text.lower()
    if method == "date":
        return _canonical_date(text)
    if method.startswith("hash:"):
        algorithm = method.split(":", 1)[1]
        if side == pair.transform_side or pair.transform_side == "both":
            return _hash_text(text, algorithm)
        return text.lower()
    raise DataBackendError(f"unsupported match method: {method}")


def transformed_key_names(key_pairs: Sequence[KeyPair]) -> list[str]:
    return [f"__marvis_key_{index}" for index in range(len(key_pairs))]


def _iter_strings(items: Iterable[str]) -> list[str]:
    return [str(item) for item in items]


__all__ = [
    "DataBackend",
    "SUPPORTED_DUCKDB_SUFFIXES",
    "SUPPORTED_FRAME_SUFFIXES",
    "configure_duckdb_defaults",
    "csv_rel",
    "duckdb_health",
    "duckdb_runtime_config",
    "parquet_rel",
    "sql_identifier",
    "sql_string_literal",
    "transformed_key_names",
]

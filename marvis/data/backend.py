from __future__ import annotations

import hashlib
import os
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
from marvis.data.csv_ingest import read_csv_with_fallback_encoding
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
# instead of spilling to disk. Apply conservative, overridable settings to every
# DuckDB connection this module opens instead.
#
# TST-9c: those connections are now per-operation (``duckdb.connect()``), NOT the
# single process-wide implicit default connection ``duckdb.sql(...)`` reuses.
# Concurrent uploads to the same task each profile their own file, and the shared
# default connection serialized to one in-flight query at a time -- a second
# thread issuing ``duckdb.sql(...)`` while the first's pending result was open
# raised ``InvalidInputException('Attempting to execute an unsuccessful or closed
# pending query result')``. A fresh connection per operation removes that shared
# mutable state; the runtime config is applied to each new connection so the
# PERF-8 memory_limit/threads/temp_directory guarantees still hold.
DUCKDB_MEMORY_LIMIT_ENV = "MARVIS_DUCKDB_MEMORY_LIMIT"
DUCKDB_THREADS_ENV = "MARVIS_DUCKDB_THREADS"
DEFAULT_DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_TEMP_DIR_NAME = ".duckdb_tmp"


def default_duckdb_threads() -> int:
    """max(2, cpu_count // 2): leaves headroom for training subprocesses / a
    co-located local LLM instead of DuckDB claiming every core by default."""
    cpu_count = os.cpu_count() or 2
    return max(2, cpu_count // 2)


def duckdb_runtime_config(temp_directory: Path) -> dict[str, str]:
    """The PRAGMA values applied to every connection this module opens, so
    callers (health/audit endpoints) can report what is in effect."""
    return {
        "memory_limit": os.environ.get(DUCKDB_MEMORY_LIMIT_ENV, DEFAULT_DUCKDB_MEMORY_LIMIT),
        "threads": str(
            os.environ.get(DUCKDB_THREADS_ENV) or default_duckdb_threads()
        ),
        "temp_directory": str(temp_directory),
    }


def configure_duckdb_defaults(temp_directory: Path) -> dict[str, str]:
    """Ensure the workspace-scoped temp_directory exists and return the runtime
    config that will be applied to each new connection. No longer mutates a shared
    process-wide connection (TST-9c): configuration is applied per connection by
    :func:`connect_duckdb`, so this is just the one filesystem side effect plus the
    resolved config, and is safe to call repeatedly from any thread."""
    temp_directory.mkdir(parents=True, exist_ok=True)
    return duckdb_runtime_config(temp_directory)


def connect_duckdb(temp_directory: Path) -> duckdb.DuckDBPyConnection:
    """Open a fresh in-memory DuckDB connection with the PERF-8 runtime config
    applied. Each operation gets its own connection so concurrent callers never
    share the single implicit default connection's mutable pending-result state
    (TST-9c). The caller owns the connection and must close it (use ``with``)."""
    config = duckdb_runtime_config(temp_directory)
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute(f"SET memory_limit={sql_string_literal(config['memory_limit'])}")
        conn.execute(f"SET threads={int(config['threads'])}")
        conn.execute(f"SET temp_directory={sql_string_literal(config['temp_directory'])}")
    except Exception:
        conn.close()
        raise
    return conn


def duckdb_health(temp_directory: Path) -> dict[str, object]:
    """Effective PRAGMA values on a connection opened the same way this module's
    operations open theirs, for ``/api/health`` (PERF-8 audit visibility)."""
    with connect_duckdb(temp_directory) as conn:
        rows = conn.execute(
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
        self._temp_directory = self._root.parent / DUCKDB_TEMP_DIR_NAME
        # Ensures the workspace-scoped DuckDB spill directory exists up front; the
        # runtime config itself is applied per operation by self._connect() (TST-9c).
        configure_duckdb_defaults(self._temp_directory)
        # PERF-4: a plain in-process dict, scoped to THIS DataBackend instance's lifetime
        # (one per request/job -- see routers/data.py, turn_handlers.py, packs/*/tools.py).
        # It is never persisted or shared across instances, so there is no cross-request
        # invalidation to get wrong: the cache is created and thrown away with the backend
        # that populated it. Keys are namespaced by call kind + a (path, mtime_ns, size)
        # file-identity tuple (plus call-specific extras), so a file changing on disk between
        # two DataBackend instances can never serve a stale value.
        self._cache: dict[tuple, Any] = {}

    def _memo(self, kind: str, path: Path, *extra: Any, compute):
        key = (kind, *self._file_identity(path), *extra)
        if key in self._cache:
            return self._cache[key]
        value = compute()
        self._cache[key] = value
        return value

    def _file_identity(self, path: Path) -> tuple[str, int, int]:
        """(path, mtime_ns, size) -- used as the cache-key prefix for every memoized call so
        a file that changes on disk (different mtime/size) never serves a stale cached value,
        upholding determinism (INV-1) the same way a fresh DataBackend would recompute it."""
        try:
            stat = path.stat()
            return (path.as_posix(), stat.st_mtime_ns, stat.st_size)
        except OSError:
            # Path doesn't exist yet or is unreadable -- let the real call raise the real
            # error; do not cache under a fake identity.
            return (path.as_posix(), -1, -1)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """A fresh, PERF-8-configured DuckDB connection scoped to a single
        operation. Never the shared implicit default connection (TST-9c), so
        concurrent DataBackend operations can't collide on one connection's
        pending-result state. Use as a context manager so it is always closed."""
        return connect_duckdb(self._temp_directory)

    def row_count(self, path: Path) -> int:
        path = self._resolve_path(path)
        return self._memo("row_count", path, compute=lambda: self._row_count_uncached(path))

    def _row_count_uncached(self, path: Path) -> int:
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_DUCKDB_SUFFIXES:
            with self._connect() as conn:
                row = conn.execute(
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
        return self._memo("numeric_columns", path, compute=lambda: self._numeric_columns_uncached(path))

    def _numeric_columns_uncached(self, path: Path) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(f"DESCRIBE SELECT * FROM {self._duckdb_rel(path)}").fetchall()
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
            # T1-B7: key-VARCHAR reader, so conflict detection sees keys typed as the join does.
            rel = self._duckdb_key_rel(path, [pair.feature_col for pair in key_pairs])
        else:
            keys_sql = ", ".join(sql_identifier(col, allowed_columns) for col in key_columns)
            keys_select_sql = keys_sql
            key_not_null = " AND ".join(
                f"{sql_identifier(col, allowed_columns)} IS NOT NULL"
                for col in key_columns
            ) or "TRUE"
            rel = self._duckdb_rel(path)
        duplicate_groups = (
            f"SELECT {keys_sql}, count(*) AS __n "
            f"FROM {rel} "
            f"WHERE {key_not_null} "
            f"GROUP BY {keys_select_sql} "
            "HAVING count(*) > 1"
        )
        with self._connect() as conn:
            count_row = conn.execute(
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
            with self._connect() as conn:
                sample_rows = conn.execute(
                    "WITH duplicate_keys AS ("
                    # duplicate_groups already projects the transformed keys under their
                    # aliases, so re-select the aliases here (not the raw expressions again —
                    # the raw source column no longer exists in this subquery's output).
                    f"SELECT {keys_select_sql} FROM ({duplicate_groups}) ORDER BY {keys_select_sql} "
                    f"LIMIT {int(sample_key_limit)}"
                    "), feature_with_keys AS ("
                    f"SELECT f.*, {keys_sql} FROM {rel} f"
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
        with self._connect() as conn:
            sample_rows = conn.execute(
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
        return list(self._memo("column_names", path, compute=lambda: self._column_names_uncached(path)))

    def _column_names_uncached(self, path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_DUCKDB_SUFFIXES:
            with self._connect() as conn:
                rows = conn.execute(f"DESCRIBE SELECT * FROM {self._duckdb_rel(path)}").fetchall()
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
            # GAP-1: fall back through the common Chinese-bank-export encodings
            # instead of hardcoding utf-8-sig -- see marvis.data.csv_ingest.
            frame, _report = read_csv_with_fallback_encoding(
                path,
                usecols=selected,
                nrows=nrows,
            )
            return frame
        if suffix == ".parquet":
            if nrows is not None:
                cols_sql = self._select_columns_sql(selected, allowed_columns)
                query = f"SELECT {cols_sql} FROM {parquet_rel(path)} LIMIT {int(nrows)}"
                with self._connect() as conn:
                    return conn.execute(query).df()
            return pd.read_parquet(path, columns=selected)
        if suffix == ".feather":
            frame = pd.read_feather(path, columns=selected)
            return frame.head(nrows) if nrows is not None else frame
        raise DataBackendError(f"unsupported dataset format: {path.suffix}")

    def sample_rows(self, path: Path, n: int, *, seed: int) -> pd.DataFrame:
        path = self._resolve_path(path)
        cached = self._memo(
            "sample_rows", path, int(n), int(seed),
            compute=lambda: self._sample_rows_uncached(path, n, seed=seed),
        )
        # Defensive copy: callers must never be able to mutate the cached frame in place
        # and corrupt a later cache hit (upholding determinism -- INV-1 -- across repeated
        # calls with the same path/n/seed within one diagnose session).
        return cached.copy()

    def _sample_rows_uncached(self, path: Path, n: int, *, seed: int) -> pd.DataFrame:
        total = self.row_count(path)
        if total <= n:
            return self.read_frame(path)
        if total > LARGE_ROW_THRESHOLD and path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES:
            query = (
                f"SELECT * FROM {self._duckdb_rel(path)} "
                f"USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})"
            )
            with self._connect() as conn:
                return conn.execute(query).df()
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
        cache_key_pairs = tuple(key_pairs) if key_pairs is not None else None
        return self._memo(
            "distinct_count", path, tuple(columns), cache_key_pairs,
            compute=lambda: self._distinct_count_uncached(path, columns, key_pairs=key_pairs),
        )

    def _distinct_count_uncached(
        self,
        path: Path,
        columns: list[str],
        *,
        key_pairs: Sequence[KeyPair] | None,
    ) -> int:
        allowed_columns = set(self.column_names(path))
        self._validate_columns(columns, allowed_columns)
        if path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES:
            if key_pairs is not None:
                key_exprs = self._transformed_key_exprs(key_pairs, allowed_columns, side="feature")
                cols_sql = ", ".join(expr for expr, _alias in key_exprs)
                # T1-A5: a blank/whitespace key transforms to NULL (missing) and never
                # participates in the LEFT JOIN, so it is not a distinct join key -- exclude
                # rows whose transformed key is NULL so two '' rows aren't counted as a
                # duplicate-key collision (and don't distort the fan-out duplicate_factor).
                not_null_sql = " AND ".join(
                    f"({expr}) IS NOT NULL" for expr, _alias in key_exprs
                )
                where_sql = f" WHERE {not_null_sql}"
                # T1-B7: read key columns with the same VARCHAR typing the executed dedup uses.
                rel = self._duckdb_key_rel(path, [pair.feature_col for pair in key_pairs])
            else:
                cols_sql = ", ".join(sql_identifier(col, allowed_columns) for col in columns)
                where_sql = ""
                rel = self._duckdb_rel(path)
            query = (
                "SELECT count(*) FROM ("
                f"SELECT DISTINCT {cols_sql} FROM {rel}{where_sql}"
                ")"
            )
            with self._connect() as conn:
                return int(conn.execute(query).fetchone()[0])
        if key_pairs is not None:
            frame = self.with_transformed_key_columns(self.read_frame(path), key_pairs)
            key_names = transformed_key_names(key_pairs)
            # Mirror the SQL branch: a missing (None) transformed key never joins, so drop
            # rows with any NULL key before counting distinct join keys.
            key_frame = frame[key_names].dropna()
            return int(key_frame.drop_duplicates().shape[0])
        frame = self.read_frame(path, columns=columns)
        return int(frame.drop_duplicates().shape[0])

    def is_key_unique(
        self,
        path: Path,
        columns: list[str],
        *,
        key_pairs: Sequence[KeyPair] | None = None,
    ) -> bool:
        if key_pairs is not None:
            # T1-A5: uniqueness is judged over JOINABLE rows only -- a blank/whitespace key
            # transforms to NULL and never joins, so it is neither a distinct key nor a
            # collision. Compare distinct non-missing keys against the count of rows that
            # actually carry a non-missing key (distinct_count already excludes NULL keys).
            basis = self._nonnull_key_row_count(path, key_pairs)
            return self.distinct_count(path, columns, key_pairs=key_pairs) == basis
        return self.distinct_count(path, columns, key_pairs=key_pairs) == self.row_count(path)

    def _nonnull_key_row_count(self, path: Path, key_pairs: Sequence[KeyPair]) -> int:
        path = self._resolve_path(path)
        return self._memo(
            "nonnull_key_row_count", path, tuple(key_pairs),
            compute=lambda: self._nonnull_key_row_count_uncached(path, key_pairs),
        )

    def _nonnull_key_row_count_uncached(self, path: Path, key_pairs: Sequence[KeyPair]) -> int:
        allowed_columns = set(self.column_names(path))
        if path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES:
            key_exprs = self._transformed_key_exprs(key_pairs, allowed_columns, side="feature")
            not_null_sql = " AND ".join(
                f"({expr}) IS NOT NULL" for expr, _alias in key_exprs
            )
            # T1-B7: key-VARCHAR reader, consistent with distinct_count and the executed dedup.
            rel = self._duckdb_key_rel(path, [pair.feature_col for pair in key_pairs])
            query = f"SELECT count(*) FROM {rel} WHERE {not_null_sql}"
            with self._connect() as conn:
                return int(conn.execute(query).fetchone()[0])
        frame = self.with_transformed_key_columns(self.read_frame(path), key_pairs)
        key_names = transformed_key_names(key_pairs)
        return int(frame[key_names].dropna().shape[0])

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
        # T1-B7: read the anchor key columns as VARCHAR (same reader the diagnostics and dedup
        # use) so a typing-sensitive key (zero-padded / mixed-date) joins consistently. Only the
        # anchor's KEY columns are forced to text; payload columns keep their sniffed types.
        anchor_key_columns = [pair.anchor_col for pair in key_pairs]
        query = (
            "COPY ("
            f"SELECT a.*{feature_select} "
            f"FROM {self._duckdb_key_rel(anchor_path, anchor_key_columns)} a "
            f"LEFT JOIN ({feature_rel}) b ON {on_sql}"
            f") TO {sql_string_literal(out_path.as_posix())} (FORMAT parquet)"
        )
        with self._connect() as conn:
            conn.execute(query)
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
        cache_key = (
            "match_rate_for_method",
            *self._file_identity(anchor_path),
            tuple(anchor_keys),
            *self._file_identity(feature_path),
            tuple(feature_keys),
            tuple(methods),
            tuple(key_fingerprints),
            int(sample_n),
            int(seed),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self._match_rate_for_method_uncached(
            anchor_path, anchor_keys, feature_path, feature_keys,
            methods=methods, key_fingerprints=key_fingerprints,
            sample_n=sample_n, seed=seed,
        )
        self._cache[cache_key] = result
        return result

    def match_rate_pandas(
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
        """T3-1: the pure-pandas match-rate path, ALWAYS forced (never delegates to
        DuckDB). :meth:`match_rate_for_method` picks DuckDB for supported methods; this
        is the genuinely independent second implementation the reconciliation layer
        compares it against -- same key normalization, but a python set-membership loop
        over sampled anchor rows instead of a SQL hash join. Same ``(matched, sampled)``
        contract, so a divergence between the two is a real bug, not a shape mismatch."""
        anchor_path = self._resolve_path(anchor_path)
        feature_path = self._resolve_path(feature_path)
        if len(anchor_keys) != len(feature_keys):
            raise DataBackendError("anchor_keys and feature_keys must have the same length")
        if len(anchor_keys) != len(key_fingerprints):
            raise DataBackendError("key_fingerprints must align with key columns")
        methods = _methods_for_keys(method, len(anchor_keys))
        self._validate_columns(anchor_keys, set(self.column_names(anchor_path)))
        self._validate_columns(feature_keys, set(self.column_names(feature_path)))
        return self._pandas_match_rate(
            anchor_path, anchor_keys, feature_path, feature_keys,
            methods=methods, key_fingerprints=key_fingerprints,
            sample_n=sample_n, seed=seed,
        )

    def reconcile_paths_are_independent(
        self,
        anchor_path: Path,
        feature_path: Path,
        method: str | Sequence[str],
    ) -> bool:
        """T3-2: True only when :meth:`match_rate_for_method` (the reconcile PRIMARY) would run
        the DuckDB SQL kernel -- i.e. both datasets are a DuckDB-readable format AND every match
        method is one DuckDB can express. In that case the pandas secondary is a genuinely
        independent implementation and a reconcile verdict is meaningful.

        When this is False the primary FALLS BACK to the same ``_pandas_match_rate`` kernel the
        secondary uses (unsupported hash algorithm, or a non-CSV/parquet feature such as .xlsx),
        so the "two paths" collapse to ONE function and always agree by construction. Presenting
        that as a passing reconciliation is false assurance; the caller must instead mark the
        number as not independently verified."""
        anchor_path = self._resolve_path(anchor_path)
        feature_path = self._resolve_path(feature_path)
        methods = [str(m) for m in ([method] if isinstance(method, str) else method)]
        return (
            anchor_path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES
            and feature_path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES
            and _duckdb_supports_match_methods(methods)
        )

    def match_rate_reconcile_secondary(
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
        """T3-1: the reconcile-only pandas recount, scored over the EXACT anchor rows the DuckDB
        primary sampled. Same pure-pandas set-membership kernel as :meth:`match_rate_pandas` (a
        genuinely independent implementation vs the SQL hash join), but fed the identical anchor
        subset via the shared DuckDB reservoir -- so a divergence now reflects a real
        implementation disagreement, never a sampling artifact. Used only by the join trust
        layer; :meth:`match_rate_pandas` remains the general-purpose forced-pandas path."""
        anchor_path = self._resolve_path(anchor_path)
        feature_path = self._resolve_path(feature_path)
        if len(anchor_keys) != len(feature_keys):
            raise DataBackendError("anchor_keys and feature_keys must have the same length")
        if len(anchor_keys) != len(key_fingerprints):
            raise DataBackendError("key_fingerprints must align with key columns")
        methods = _methods_for_keys(method, len(anchor_keys))
        self._validate_columns(anchor_keys, set(self.column_names(anchor_path)))
        self._validate_columns(feature_keys, set(self.column_names(feature_path)))
        # Distinct cache discriminator so this NEVER collides with match_rate_pandas cache
        # entries (same (sample_n, seed) but a different sampling semantics).
        cache_key = (
            "match_rate_reconcile_secondary",
            *self._file_identity(anchor_path),
            tuple(anchor_keys),
            *self._file_identity(feature_path),
            tuple(feature_keys),
            tuple(methods),
            tuple(key_fingerprints),
            int(sample_n),
            int(seed),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        anchor_frame = self._duckdb_reservoir_sample_frame(
            anchor_path, anchor_keys, sample_n=sample_n, seed=seed
        )
        result = self._pandas_match_rate(
            anchor_path, anchor_keys, feature_path, feature_keys,
            methods=methods, key_fingerprints=key_fingerprints,
            sample_n=sample_n, seed=seed, anchor_frame=anchor_frame,
        )
        self._cache[cache_key] = result
        return result

    def _match_rate_for_method_uncached(
        self,
        anchor_path: Path,
        anchor_keys: Sequence[str],
        feature_path: Path,
        feature_keys: Sequence[str],
        *,
        methods: Sequence[str],
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
    ) -> tuple[int, int]:
        self._validate_columns(anchor_keys, set(self.column_names(anchor_path)))
        self._validate_columns(feature_keys, set(self.column_names(feature_path)))
        if (
            anchor_path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES
            and feature_path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES
            and _duckdb_supports_match_methods(methods)
        ):
            # T1-B7: sample the anchor INSIDE DuckDB through the same key-VARCHAR reader as the
            # feature and the executed join, instead of a pandas frame whose CSV type inference
            # (007 -> int 7) diverged from execution's read_csv_auto. Now anchor + feature +
            # execution read keys identically, so the diagnostic rate equals the realized join.
            return self._duckdb_match_rate_for_method(
                anchor_path,
                feature_path,
                anchor_keys,
                feature_keys,
                methods=methods,
                key_fingerprints=key_fingerprints,
                sample_n=sample_n,
                seed=seed,
            )
        return self._pandas_match_rate(
            anchor_path, anchor_keys, feature_path, feature_keys,
            methods=methods, key_fingerprints=key_fingerprints,
            sample_n=sample_n, seed=seed,
        )

    def _pandas_match_rate(
        self,
        anchor_path: Path,
        anchor_keys: Sequence[str],
        feature_path: Path,
        feature_keys: Sequence[str],
        *,
        methods: Sequence[str],
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
        anchor_frame: pd.DataFrame | None = None,
    ) -> tuple[int, int]:
        # ``anchor_frame`` lets the reconcile secondary inject the SAME anchor rows the DuckDB
        # primary sampled (T3-1 defect: the two paths must score an identical anchor subset,
        # not two independently-drawn samples). When absent, sample independently as before.
        if anchor_frame is None:
            anchor_frame = self.sample_rows(anchor_path, sample_n, seed=seed)
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

    def match_rates_for_methods(
        self,
        anchor_path: Path,
        anchor_col: str,
        feature_path: Path,
        feature_col: str,
        *,
        methods: Sequence[str],
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        """Batched counterpart to :meth:`match_rate_for_method` for a SINGLE anchor/feature
        column pair tried under several candidate match methods (PERF-4: this is exactly the
        ``align._resolve_by_data`` loop -- one column pair x several methods). Instead of one
        DuckDB round-trip (anchor sample scan + feature DISTINCT scan) per method, this issues
        ONE query: one shared registered anchor sample, one feature-table scan producing all
        methods' normalized keys together, and one ``count(*) FILTER`` per method. Falls back
        to :meth:`match_rate_for_method` per-method for any method DuckDB can't express
        (unsupported hash algorithm) so semantics/results are identical either way.

        Returns a list of ``(matched, sampled)`` aligned with ``methods``.
        """
        anchor_path = self._resolve_path(anchor_path)
        feature_path = self._resolve_path(feature_path)
        if len(methods) != len(key_fingerprints):
            raise DataBackendError("key_fingerprints must align with methods")
        if not methods:
            return []
        cache_key = (
            "match_rates_for_methods",
            *self._file_identity(anchor_path),
            anchor_col,
            *self._file_identity(feature_path),
            feature_col,
            tuple(methods),
            tuple(key_fingerprints),
            int(sample_n),
            int(seed),
        )
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        if (
            feature_path.suffix.lower() not in SUPPORTED_DUCKDB_SUFFIXES
            or not _duckdb_supports_match_methods(methods)
        ):
            # Fall back to the per-method path (still benefits from the sample_rows /
            # column_names memoization above) when DuckDB can't express every method in
            # one batch -- e.g. an unsupported hash algorithm forces the Python fallback.
            results = [
                self.match_rate_for_method(
                    anchor_path, [anchor_col], feature_path, [feature_col],
                    method=method, key_fingerprints=[fingerprint],
                    sample_n=sample_n, seed=seed,
                )
                for method, fingerprint in zip(methods, key_fingerprints)
            ]
            self._cache[cache_key] = results
            return list(results)

        self._validate_columns([anchor_col], set(self.column_names(anchor_path)))
        self._validate_columns([feature_col], set(self.column_names(feature_path)))
        results = self._duckdb_match_rates_for_methods(
            anchor_path, feature_path, anchor_col, feature_col,
            methods=methods, key_fingerprints=key_fingerprints,
            sample_n=sample_n, seed=seed,
        )
        self._cache[cache_key] = results
        return list(results)

    def _duckdb_match_rates_for_methods(
        self,
        anchor_path: Path,
        feature_path: Path,
        anchor_col: str,
        feature_col: str,
        *,
        methods: Sequence[str],
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
    ) -> list[tuple[int, int]]:
        anchor_columns = set(self.column_names(anchor_path))
        feature_columns = set(self.column_names(feature_path))
        anchor_ident = "a." + sql_identifier(anchor_col, anchor_columns)
        feature_ident = "b." + sql_identifier(feature_col, feature_columns)
        # T1-B7: anchor + feature read through the SAME key-VARCHAR reader as the executed join
        # (previously the feature was all_varchar and the anchor a pandas frame -- both diverged
        # from execution's read_csv_auto for typing-sensitive keys). The anchor is sampled once
        # inside DuckDB and shared across every method's LEFT JOIN below.
        anchor_rel, sampled = self._duckdb_key_sample_cte(
            anchor_path, [anchor_col], sample_n=sample_n, seed=seed
        )
        feature_rel = self._duckdb_key_rel(feature_path, [feature_col])
        # feature_keys computes DISTINCT normalized keys for EVERY method from a SINGLE
        # scan of the feature table (one file read shared across all methods, instead of
        # one read per method as the single-method path would do if called N times), then
        # each method gets its own LEFT JOIN against the shared (small) anchor sample --
        # same hash-join shape as the proven single-method query below, just batched.
        feature_exprs = ", ".join(
            f"{_sql_normalized_key(method, feature_ident, fingerprint=_fingerprint_pair(fp)[1])} "
            f"AS __key_{index}"
            for index, (method, fp) in enumerate(zip(methods, key_fingerprints))
        )
        ctes = [
            f"anchor_sample AS (SELECT * FROM {anchor_rel})",
            f"feature_keys AS (SELECT {feature_exprs} FROM {feature_rel} b)",
        ]
        selects = []
        for index, (method, fp) in enumerate(zip(methods, key_fingerprints)):
            anchor_fp, _feature_fp = _fingerprint_pair(fp)
            anchor_expr = _sql_normalized_key(method, anchor_ident, fingerprint=anchor_fp)
            ctes.append(
                f"anchor_keys_{index} AS (SELECT {anchor_expr} AS __key FROM anchor_sample a)"
            )
            ctes.append(
                f"feature_distinct_{index} AS (SELECT DISTINCT __key_{index} AS __key FROM feature_keys)"
            )
            selects.append(
                f"(SELECT count(*) FILTER (WHERE a.__key IS NOT NULL AND f.__key IS NOT NULL) "
                f"FROM anchor_keys_{index} a LEFT JOIN feature_distinct_{index} f "
                f"ON a.__key = f.__key) AS __matched_{index}"
            )
        query = "WITH " + ", ".join(ctes) + " SELECT " + ", ".join(selects)
        with self._connect() as conn:
            row = conn.execute(query).fetchone()
        return [(int(row[index] or 0), int(sampled)) for index in range(len(methods))]

    def _duckdb_match_rate_for_method(
        self,
        anchor_path: Path,
        feature_path: Path,
        anchor_keys: Sequence[str],
        feature_keys: Sequence[str],
        *,
        methods: Sequence[str],
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
    ) -> tuple[int, int]:
        fingerprint_pairs = [_fingerprint_pair(item) for item in key_fingerprints]
        anchor_columns = set(self.column_names(anchor_path))
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
        # T1-B7: anchor + feature read through the SAME key-VARCHAR reader as the executed join.
        anchor_rel, sampled = self._duckdb_key_sample_cte(
            anchor_path, anchor_keys, sample_n=sample_n, seed=seed
        )
        feature_rel = self._duckdb_key_rel(feature_path, feature_keys)
        query = (
            "WITH anchor_keys AS ("
            f"SELECT {anchor_projection} FROM {anchor_rel} a"
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
        with self._connect() as conn:
            matched = conn.execute(query).fetchone()[0]
        return int(matched), int(sampled)

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

    def _duckdb_key_rel(self, path: Path, key_columns: Sequence[str]) -> str:
        """T1-B7: the SAME reader for every key-consuming SQL surface -- the executed
        ``left_join`` (anchor + deduped feature) AND the match-rate diagnostics (anchor +
        feature) -- so diagnostics and execution type join keys identically and cannot
        diverge on typing-sensitive keys.

        For CSV, force the KEY columns to VARCHAR via ``read_csv_auto(..., types=...)``: this
        keeps zero-padded ids ("007", not int 7), long-id text, and same-column mixed date
        formats ("2026-01-01" and "2026/01/02") as text -- the canonical form the shared
        normalizer ``_sql_value_text`` casts to anyway (so this only aligns the read with the
        cast already applied everywhere downstream). Non-key columns keep their sniffed types,
        so numeric dedup / payload typing is unaffected. Parquet keys are already exact on disk
        and the VARCHAR cast in ``_sql_value_text`` normalizes them, so parquet needs no
        override."""
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return parquet_rel(path)
        if suffix == ".csv":
            unique_cols = list(dict.fromkeys(str(column) for column in key_columns))
            if not unique_cols:
                return csv_rel(path)
            types_entries = ", ".join(
                f"{sql_string_literal(column)}: 'VARCHAR'" for column in unique_cols
            )
            return (
                f"read_csv_auto({sql_string_literal(path.as_posix())}, "
                f"types={{{types_entries}}})"
            )
        raise DataBackendError(f"unsupported DuckDB dataset format: {path.suffix}")

    def _duckdb_key_sample_cte(
        self, path: Path, key_columns: Sequence[str], *, sample_n: int, seed: int
    ) -> tuple[str, int]:
        """T1-B7: a DuckDB-native anchor sample scanned through ``_duckdb_key_rel`` (keys as
        VARCHAR), replacing the pandas-typed ``sample_rows`` frame in the DuckDB match-rate
        path so anchor+feature+execution read keys identically. Returns the relation SQL and
        the sampled row count (the match-rate denominator). Reservoir-samples deterministically
        (REPEATABLE(seed)) only when the table exceeds ``sample_n`` -- mirroring
        ``_sample_rows_uncached`` -- otherwise scans the whole (small) table."""
        rel = self._duckdb_key_rel(path, key_columns)
        total = self.row_count(path)
        if total <= sample_n:
            return rel, total
        sampled_rel = (
            f"(SELECT * FROM {rel} "
            f"USING SAMPLE reservoir({int(sample_n)} ROWS) REPEATABLE ({int(seed)}))"
        )
        return sampled_rel, int(sample_n)

    def _duckdb_reservoir_sample_frame(
        self, path: Path, columns: Sequence[str], *, sample_n: int, seed: int
    ) -> pd.DataFrame:
        """T3-1: materialize the SAME anchor rows the DuckDB match-rate primary path scores.

        The primary path samples the anchor with ``USING SAMPLE reservoir(sample_n) REPEATABLE
        (seed)`` whenever ``total > sample_n`` (:meth:`_duckdb_key_sample_cte`), while the pandas
        ``sample_rows`` only crosses to the DuckDB reservoir past ``LARGE_ROW_THRESHOLD`` -- so in
        the ``(sample_n, LARGE_ROW_THRESHOLD]`` band the two paths drew DIFFERENT subsets and a
        correct join looked like a reconcile mismatch. This reads the anchor through the SAME
        reservoir the primary uses so the reconcile secondary scores an IDENTICAL subset.

        Read through ``_duckdb_rel`` (sniffed types), NOT ``_duckdb_key_rel`` (forced VARCHAR):
        the pandas normalizer's numeric branches (``_value_text``) must see numeric scalars, or a
        float-scientific-notation / long-id key would re-diverge from the SQL normalizer on the
        Python side. Only which ROWS are compared changes; typing stays what the pandas path
        always used."""
        path = self._resolve_path(path)
        total = self.row_count(path)
        cols_sql = ", ".join(
            sql_identifier(str(column), set(self.column_names(path))) for column in columns
        )
        if total <= sample_n:
            query = f"SELECT {cols_sql} FROM {self._duckdb_rel(path)}"
        else:
            query = (
                f"SELECT {cols_sql} FROM {self._duckdb_rel(path)} "
                f"USING SAMPLE reservoir({int(sample_n)} ROWS) REPEATABLE ({int(seed)})"
            )
        with self._connect() as conn:
            return conn.execute(query).df()

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
        # T1-B7: read feature key columns as VARCHAR (same reader the anchor + diagnostics use)
        # so the deduped feature that feeds the join types keys identically to the join's ON.
        rel = self._duckdb_key_rel(feature_path, feature_key_columns)
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
    # T1-A5: nullif(...,'') is applied INSIDE the shared normalizer so blank/whitespace-only
    # keys become NULL on EVERY path -- the executed JOIN condition (_sql_transform) and the
    # match-rate/dedup diagnostics (_sql_normalized_key) alike. SQL NULL never equals NULL in
    # a join, so a blank-keyed anchor row falls through to NULL feature columns instead of
    # wrongly attaching to a blank-keyed feature row. This matches the Python fallback
    # (_value_text -> '' -> None) and the diagnostics, closing the three-way split.
    #
    # T1-A6: a float64-stored long integer id renders in scientific notation under a plain
    # CAST(... AS VARCHAR) (e.g. 1.2345678901234568e+17), which the trailing-.0 regex below
    # cannot strip -- so the SQL key never matched the Python key (str(int(float(...)))) nor a
    # string-stored id. When the column's runtime type is a floating type AND the value is a
    # finite whole number, render it through HUGEINT so it becomes the SAME precision-rounded
    # integer string Python produces (no exponent, no trailing .0). isfinite/floor operate on a
    # TRY_CAST(... AS DOUBLE) handle (NULL for non-numeric text) so this branch never raises a
    # binder error when the column is VARCHAR (DuckDB does not short-circuit those by typeof).
    # _sql_value_text always receives a bare column identifier, so typeof() is well-defined.
    text = f"CAST({expression} AS VARCHAR)"
    trimmed = f"trim({text})"
    as_double = f"TRY_CAST({expression} AS DOUBLE)"
    return (
        "nullif(trim(CASE "
        f"WHEN typeof({expression}) IN ('DOUBLE', 'FLOAT', 'REAL') "
        f"AND {as_double} IS NOT NULL AND isfinite({as_double}) "
        f"AND {as_double} = floor({as_double}) "
        f"THEN CAST(TRY_CAST({expression} AS HUGEINT) AS VARCHAR) "
        f"WHEN regexp_matches({trimmed}, '^-?[0-9]+\\.0+$') "
        f"THEN regexp_replace({trimmed}, '\\.0+$', '') "
        f"ELSE {trimmed} END), '')"
    )


def _sql_normalized_key(method: str, expression: str, *, fingerprint: ColumnFingerprint) -> str:
    # _sql_value_text already nullifs blank keys (T1-A5), so the diagnostics path no longer
    # needs its own nullif wrapper -- it inherits blank=missing from the shared normalizer.
    text = _sql_value_text(expression)
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
    "connect_duckdb",
    "csv_rel",
    "duckdb_health",
    "duckdb_runtime_config",
    "parquet_rel",
    "sql_identifier",
    "sql_string_literal",
    "transformed_key_names",
]

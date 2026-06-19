from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from marvis.data.contracts import (
    DATE_FORMATS,
    LARGE_ROW_THRESHOLD,
    ColumnFingerprint,
    KeyPair,
)
from marvis.data.errors import DataBackendError, DataSecurityError


SUPPORTED_FRAME_SUFFIXES = {".csv", ".parquet", ".feather"}
SUPPORTED_DUCKDB_SUFFIXES = {".csv", ".parquet"}


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

    def distinct_count(self, path: Path, columns: list[str]) -> int:
        path = self._resolve_path(path)
        if not columns:
            raise DataBackendError("distinct_count requires at least one column")
        allowed_columns = set(self.column_names(path))
        self._validate_columns(columns, allowed_columns)
        if path.suffix.lower() in SUPPORTED_DUCKDB_SUFFIXES:
            cols_sql = ", ".join(sql_identifier(col, allowed_columns) for col in columns)
            query = (
                "SELECT count(*) FROM ("
                f"SELECT DISTINCT {cols_sql} FROM {self._duckdb_rel(path)}"
                ")"
            )
            return int(duckdb.sql(query).fetchone()[0])
        frame = self.read_frame(path, columns=columns)
        return int(frame.drop_duplicates().shape[0])

    def is_key_unique(self, path: Path, columns: list[str]) -> bool:
        return self.distinct_count(path, columns) == self.row_count(path)

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

        if dedup_strategy == "abort" and not self.is_key_unique(feature_path, feature_key_columns):
            raise DataBackendError("feature keys are not unique")

        feature_rel = self._dedup_feature_rel(
            feature_path,
            feature_columns,
            feature_key_columns,
            dedup_strategy,
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
        if result_rows > anchor_rows:
            out_path.unlink(missing_ok=True)
            raise DataBackendError(
                f"left_join produced {result_rows} rows from {anchor_rows} anchor rows",
            )
        return result_rows

    def match_rate_for_method(
        self,
        anchor_path: Path,
        anchor_keys: Sequence[str],
        feature_path: Path,
        feature_keys: Sequence[str],
        *,
        method: str,
        key_fingerprints: Sequence[Any],
        sample_n: int,
        seed: int,
    ) -> tuple[int, int]:
        if len(anchor_keys) != len(feature_keys):
            raise DataBackendError("anchor_keys and feature_keys must have the same length")
        if len(anchor_keys) != len(key_fingerprints):
            raise DataBackendError("key_fingerprints must align with key columns")

        self._validate_columns(anchor_keys, set(self.column_names(anchor_path)))
        self._validate_columns(feature_keys, set(self.column_names(feature_path)))
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
                    method=method,
                    side="feature",
                    fingerprint=feature_fp,
                )
                for feature_col, (_, feature_fp) in zip(feature_keys, fingerprint_pairs)
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
                    method=method,
                    side="anchor",
                    fingerprint=anchor_fp,
                )
                for anchor_col, (anchor_fp, _) in zip(anchor_keys, fingerprint_pairs)
            )
            if all(value is not None for value in key) and key in feature_key_set:
                matched += 1
        return matched, int(anchor_frame.shape[0])

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
    ) -> str:
        rel = self._duckdb_rel(feature_path)
        if dedup_strategy in (None, "abort"):
            return f"SELECT * FROM {rel}"
        key_sql = ", ".join(
            sql_identifier(column, feature_columns)
            for column in feature_key_columns
        )
        if dedup_strategy in {"first", "last"}:
            order = "ASC" if dedup_strategy == "first" else "DESC"
            return (
                "SELECT * EXCLUDE (__marvis_rn) FROM ("
                "SELECT *, row_number() OVER ("
                f"PARTITION BY {key_sql} ORDER BY __marvis_rowid {order}"
                ") AS __marvis_rn FROM ("
                f"SELECT *, row_number() OVER () AS __marvis_rowid FROM {rel}"
                ")"
                ") WHERE __marvis_rn = 1"
            )
        if dedup_strategy in {"agg_mean", "agg_max"}:
            projections = list(feature_key_columns)
            for column in sorted(feature_columns - set(feature_key_columns)):
                ident = sql_identifier(column, feature_columns)
                alias = _quote_identifier(column)
                if dedup_strategy == "agg_mean":
                    projections.append(f"avg(try_cast({ident} AS DOUBLE)) AS {alias}")
                else:
                    projections.append(f"max({ident}) AS {alias}")
            projection_sql = ", ".join(
                sql_identifier(column, feature_columns)
                if column in feature_columns
                else column
                for column in projections
            )
            return f"SELECT {projection_sql} FROM {rel} GROUP BY {key_sql}"
        raise DataBackendError(f"unsupported dedup_strategy: {dedup_strategy}")

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

    def _feature_projection(
        self,
        feature_columns: set[str],
        feature_key_columns: Sequence[str],
        anchor_columns: set[str],
    ) -> str:
        selections = []
        for column in sorted(feature_columns - set(feature_key_columns)):
            source = "b." + sql_identifier(column, feature_columns)
            alias = column if column not in anchor_columns else f"feature_{column}"
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
        text = str(value).strip()
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


def _sql_transform(method: str, expression: str, *, side: str, pair: KeyPair) -> str:
    trimmed = f"trim(CAST({expression} AS VARCHAR))"
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
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _hash_text(text: str, algorithm: str) -> str:
    if algorithm not in hashlib.algorithms_available:
        raise DataBackendError(f"unsupported hash method: hash:{algorithm}")
    digest = hashlib.new(algorithm)
    digest.update(text.encode("utf-8"))
    return digest.hexdigest().lower()


def _iter_strings(items: Iterable[str]) -> list[str]:
    return [str(item) for item in items]


__all__ = [
    "DataBackend",
    "SUPPORTED_DUCKDB_SUFFIXES",
    "SUPPORTED_FRAME_SUFFIXES",
    "csv_rel",
    "parquet_rel",
    "sql_identifier",
    "sql_string_literal",
]

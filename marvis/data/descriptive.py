"""Exact, deterministic descriptive analysis for normalized Parquet datasets.

The core deliberately scans Parquet through DuckDB aggregate queries.  It never
loads a complete dataset into a pandas frame and never asks an LLM to calculate
or repair metrics.  Values that leave the core are tagged so a report renderer
can distinguish, for example, integer ``1`` from string ``"1"``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from marvis.data.backend import (
    DUCKDB_NUMERIC_TYPES,
    configure_duckdb_defaults,
    connect_duckdb,
    parquet_rel,
    sql_identifier,
)
from marvis.data.errors import DataLayerError


DATA_ANALYSIS_SCHEMA_VERSION = "data-analysis.v1"

# Every integral value in this closed interval has an exact IEEE-754 binary64
# representation and remains an exact integer when consumed by JSON/JavaScript.
# Values outside it must not enter DOUBLE-based aggregates: adjacent integers
# can otherwise collapse to the same value (for example 2**53 and 2**53 + 1).
_MAX_EXACT_DOUBLE_INTEGER = 2**53 - 1
_DECIMAL_TYPE_PATTERN = re.compile(r"(?:DECIMAL|NUMERIC)\(\d+,\d+\)", re.IGNORECASE)
_DESCRIPTIVE_CONFIG_MAXIMUMS = {
    "max_columns": 500,
    "max_numeric_columns": 128,
    "max_pairs": 8128,
    "frequency_top_k": 100,
    "low_cardinality_threshold": 1000,
    "histogram_bins": 200,
    "summary_batch_size": 64,
    "correlation_batch_size": 512,
}

_SCALAR_TYPES = frozenset(
    {"null", "bool", "int", "bigint", "float", "string", "date", "datetime"}
)
_TYPE_SORT_ORDER = {
    "null": 0,
    "bool": 1,
    "int": 2,
    "bigint": 2,
    "float": 3,
    "string": 4,
    "date": 5,
    "datetime": 6,
}

TaggedScalar = dict[str, object]
ValueSanitizer = Callable[[TaggedScalar], Mapping[str, object]]


class DescriptiveInputError(DataLayerError):
    """The requested Parquet analysis cannot be evaluated safely."""


class DescriptiveConfigError(ValueError):
    """A descriptive-analysis limit is invalid or exceeds a hard safety cap."""


class DescriptiveBudgetError(DataLayerError):
    """The exact analysis exceeds an explicit caller-controlled budget."""

    def __init__(self, *, dimension: str, actual: int, limit: int) -> None:
        self.dimension = str(dimension)
        self.actual = int(actual)
        self.limit = int(limit)
        super().__init__(
            f"descriptive analysis {self.dimension} budget exceeded: "
            f"actual={self.actual}, limit={self.limit}"
        )

    def to_detail(self) -> dict[str, object]:
        return {
            "kind": "descriptive_budget_exceeded",
            "dimension": self.dimension,
            "actual": self.actual,
            "limit": self.limit,
        }


class DescriptiveSanitizerError(DescriptiveInputError):
    """A configured frequency-value sanitizer failed closed."""


@dataclass(frozen=True)
class DescriptiveConfig:
    """Explicit resource and output budgets for an exact analysis.

    ``max_pairs`` counts unique off-diagonal numeric pairs.  Diagonal cells are
    derived from each column's exact finite count/range and therefore consume no
    pair query budget.
    """

    max_columns: int = 200
    max_numeric_columns: int = 64
    max_pairs: int = 2016
    frequency_top_k: int = 20
    low_cardinality_threshold: int = 20
    histogram_bins: int = 20
    summary_batch_size: int = 16
    correlation_batch_size: int = 32

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DescriptiveConfigError(
                    f"{field_name} must be an explicit positive integer"
                )
            maximum = _DESCRIPTIVE_CONFIG_MAXIMUMS[field_name]
            if value > maximum:
                raise DescriptiveConfigError(
                    f"{field_name} must be at most {maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class _ColumnSpec:
    name: str
    duckdb_type: str
    kind: str
    selection_role: str

    @property
    def is_numeric(self) -> bool:
        return self.kind == "numeric"


@dataclass(frozen=True)
class _FrequencyBucket:
    value: TaggedScalar
    count: int
    sort_key: tuple
    sanitizable: bool


def analyze_parquet(
    path: Path,
    *,
    temp_directory: Path,
    target_column: str | None = None,
    columns: Sequence[str] | None = None,
    config: DescriptiveConfig | None = None,
    value_sanitizers: Mapping[str, ValueSanitizer] | None = None,
) -> dict[str, object]:
    """Build a ``data-analysis.v1`` report from a normalized Parquet file.

    ``columns`` is a stable, order-preserving selection.  Duplicate names are
    removed, unknown names fail closed, and a present target is appended when it
    was not selected so target distribution evidence is never silently lost.

    ``value_sanitizers`` only transforms tagged values at the final frequency
    output boundary.  Counts, rates, top-k membership, and tie ordering are
    calculated from the original values first.  Null and non-finite sentinels
    stay explicit and are not handed to the sanitizer.  No raw canonical value
    is retained elsewhere in the returned frequency payload.
    """

    parquet_path = Path(path)
    temp_path = Path(temp_directory)
    effective_config = config or DescriptiveConfig()
    sanitizers = dict(value_sanitizers or {})
    if target_column is not None and not isinstance(target_column, str):
        raise DescriptiveInputError("target_column must be a string or None")
    _validate_parquet_path(parquet_path)
    configure_duckdb_defaults(temp_path)

    relation = parquet_rel(parquet_path)
    with connect_duckdb(temp_path) as conn:
        source_specs = _describe_columns(conn, relation)
        selected_specs, columns_requested, target_auto_included = _select_columns(
            source_specs,
            columns=columns,
            target_column=target_column,
        )
        _validate_sanitizers(sanitizers, {spec.name for spec in source_specs})
        _enforce_budgets(selected_specs, effective_config)

        row = conn.execute(f"SELECT count(*) FROM {relation}").fetchone()
        row_count = int(row[0]) if row is not None else 0
        summaries = _summarize_columns(
            conn,
            relation,
            selected_specs,
            config=effective_config,
        )

        fields: list[dict[str, object]] = []
        for spec in selected_specs:
            summary = summaries[spec.name]
            frequency = _build_frequency(
                conn,
                relation,
                spec,
                summary,
                row_count=row_count,
                config=effective_config,
                sanitizer=sanitizers.get(spec.name),
            )
            numeric = _public_numeric_summary(summary) if spec.is_numeric else None
            histogram = (
                _build_histogram(
                    conn,
                    relation,
                    spec,
                    summary,
                    config=effective_config,
                )
                if spec.is_numeric
                else None
            )
            fields.append(
                {
                    "name": spec.name,
                    "duckdb_type": spec.duckdb_type,
                    "kind": spec.kind,
                    "selection_role": spec.selection_role,
                    "row_count": row_count,
                    "null_count": int(summary["null_count"]),
                    "null_rate": (
                        int(summary["null_count"]) / row_count if row_count else 0.0
                    ),
                    "distinct_count": int(summary["distinct_count"]),
                    "numeric": numeric,
                    "frequency": frequency,
                    "histogram": histogram,
                }
            )

        numeric_specs = [spec for spec in selected_specs if spec.is_numeric]
        correlations = _build_correlations(
            conn,
            relation,
            numeric_specs,
            summaries,
            config=effective_config,
        )

    fields_by_name = {str(field["name"]): field for field in fields}
    target_distribution = _target_distribution(
        target_column,
        source_columns={spec.name for spec in source_specs},
        fields_by_name=fields_by_name,
        auto_included=target_auto_included,
    )
    result: dict[str, object] = {
        "schema_version": DATA_ANALYSIS_SCHEMA_VERSION,
        "config": effective_config.to_dict(),
        "dataset": {
            "source_format": "parquet",
            "row_count": row_count,
            "source_column_count": len(source_specs),
            "column_count": len(selected_specs),
            "numeric_column_count": len(numeric_specs),
            "columns_requested": columns_requested,
            "target_auto_included": target_auto_included,
        },
        "fields": fields,
        "target_distribution": target_distribution,
        "correlations": correlations,
    }
    _ensure_json_safe(result, context="descriptive result")
    return result


def _validate_parquet_path(path: Path) -> None:
    if path.suffix.lower() != ".parquet":
        raise DescriptiveInputError("descriptive analysis requires normalized Parquet input")
    if not path.is_file():
        raise DescriptiveInputError(f"normalized Parquet input does not exist: {path}")


def _describe_columns(
    conn: duckdb.DuckDBPyConnection,
    relation: str,
) -> list[_ColumnSpec]:
    rows = conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    return [
        _ColumnSpec(
            name=str(row[0]),
            duckdb_type=str(row[1]),
            kind=_column_kind(str(row[1])),
            selection_role="all",
        )
        for row in rows
    ]


def _column_kind(type_name: str) -> str:
    normalized = type_name.strip().upper()
    base = normalized.split("(", 1)[0].strip()
    if base in DUCKDB_NUMERIC_TYPES:
        return "numeric"
    if base == "BOOLEAN":
        return "bool"
    if base == "DATE":
        return "date"
    if base.startswith("TIMESTAMP"):
        return "datetime"
    return "string"


def _select_columns(
    source_specs: Sequence[_ColumnSpec],
    *,
    columns: Sequence[str] | None,
    target_column: str | None,
) -> tuple[list[_ColumnSpec], bool, bool]:
    by_name = {spec.name: spec for spec in source_specs}
    if columns is None:
        return list(source_specs), False, False
    if isinstance(columns, (str, bytes)):
        raise DescriptiveInputError("columns must be a sequence of column names, not a string")

    selected_names: list[str] = []
    seen: set[str] = set()
    for value in columns:
        if not isinstance(value, str):
            raise DescriptiveInputError("columns must contain only string column names")
        if value in seen:
            continue
        seen.add(value)
        selected_names.append(value)
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        raise DescriptiveInputError(
            "unknown descriptive column(s): " + ", ".join(repr(name) for name in unknown)
        )

    target_auto_included = bool(
        target_column is not None
        and target_column in by_name
        and target_column not in seen
    )
    if target_auto_included:
        selected_names.append(str(target_column))

    selected: list[_ColumnSpec] = []
    for name in selected_names:
        source = by_name[name]
        role = "target_auto_included" if target_auto_included and name == target_column else "requested"
        selected.append(
            _ColumnSpec(
                name=source.name,
                duckdb_type=source.duckdb_type,
                kind=source.kind,
                selection_role=role,
            )
        )
    return selected, True, target_auto_included


def _validate_sanitizers(
    sanitizers: Mapping[str, ValueSanitizer],
    source_columns: set[str],
) -> None:
    unknown = [name for name in sanitizers if name not in source_columns]
    if unknown:
        raise DescriptiveInputError(
            "value sanitizer configured for unknown column(s): "
            + ", ".join(repr(name) for name in unknown)
        )
    invalid = [name for name, sanitizer in sanitizers.items() if not callable(sanitizer)]
    if invalid:
        raise DescriptiveInputError(
            "value sanitizer must be callable for column(s): "
            + ", ".join(repr(name) for name in invalid)
        )


def _enforce_budgets(
    specs: Sequence[_ColumnSpec],
    config: DescriptiveConfig,
) -> None:
    _check_budget("columns", len(specs), config.max_columns)
    numeric_count = sum(spec.is_numeric for spec in specs)
    _check_budget("numeric_columns", numeric_count, config.max_numeric_columns)
    pair_count = numeric_count * (numeric_count - 1) // 2
    _check_budget("correlation_pairs", pair_count, config.max_pairs)


def _check_budget(dimension: str, actual: int, limit: int) -> None:
    if actual > limit:
        raise DescriptiveBudgetError(dimension=dimension, actual=actual, limit=limit)


def _unsafe_numeric_precision_count_expression(
    spec: _ColumnSpec,
    quoted: str,
) -> str:
    """Return an exact aggregate that detects unsafe DOUBLE conversion.

    Floating-point Parquet values are already representable by DuckDB DOUBLE
    (FLOAT only widens).  Integral values are safe only inside the interoperable
    JSON integer interval.  DECIMAL/NUMERIC values additionally have to survive
    a DOUBLE round trip at their declared scale.  The type interpolation below
    is deliberately restricted to DuckDB's canonical decimal spelling.
    """

    normalized = spec.duckdb_type.strip().upper()
    base = normalized.split("(", 1)[0].strip()
    if base in {"FLOAT", "DOUBLE", "REAL"}:
        return "CAST(0 AS BIGINT)"

    outside_safe_integer_range = (
        f"{quoted} < -{_MAX_EXACT_DOUBLE_INTEGER} "
        f"OR {quoted} > {_MAX_EXACT_DOUBLE_INTEGER}"
    )
    if base not in {"DECIMAL", "NUMERIC"}:
        return (
            "count(*) FILTER (WHERE "
            f"{quoted} IS NOT NULL AND ({outside_safe_integer_range}))"
        )

    if _DECIMAL_TYPE_PATTERN.fullmatch(normalized) is None:
        # DuckDB should always expose parameterized decimals.  If a future
        # version changes that contract, fail closed instead of interpolating an
        # unexpected type or assuming a lossy conversion is safe.
        return f"count(*) FILTER (WHERE {quoted} IS NOT NULL)"
    round_trip_changed = (
        f"TRY_CAST(CAST({quoted} AS DOUBLE) AS {normalized}) "
        f"IS DISTINCT FROM {quoted}"
    )
    return (
        "count(*) FILTER (WHERE "
        f"{quoted} IS NOT NULL AND "
        f"({outside_safe_integer_range} OR {round_trip_changed}))"
    )


def _summarize_columns(
    conn: duckdb.DuckDBPyConnection,
    relation: str,
    specs: Sequence[_ColumnSpec],
    *,
    config: DescriptiveConfig,
) -> dict[str, dict[str, object]]:
    allowed = {spec.name for spec in specs}
    summaries: dict[str, dict[str, object]] = {}
    for start in range(0, len(specs), config.summary_batch_size):
        batch = specs[start : start + config.summary_batch_size]
        selections: list[str] = []
        positions: dict[tuple[str, str], int] = {}

        def add(spec: _ColumnSpec, key: str, expression: str) -> None:
            positions[(spec.name, key)] = len(selections)
            selections.append(expression)

        for spec in batch:
            quoted = sql_identifier(spec.name, allowed)
            add(spec, "null_count", f"count(*) FILTER (WHERE {quoted} IS NULL)")
            add(spec, "distinct_count", f"count(DISTINCT {quoted})")
            if not spec.is_numeric:
                continue
            number = f"CAST({quoted} AS DOUBLE)"
            finite = f"{quoted} IS NOT NULL AND isfinite({number})"
            nonfinite = f"{quoted} IS NOT NULL AND NOT isfinite({number})"
            add(spec, "finite_count", f"count(*) FILTER (WHERE {finite})")
            add(spec, "nonfinite_count", f"count(*) FILTER (WHERE {nonfinite})")
            add(
                spec,
                "unsafe_precision_count",
                _unsafe_numeric_precision_count_expression(spec, quoted),
            )
            add(
                spec,
                "regular_distinct_count",
                f"count(DISTINCT {quoted}) FILTER (WHERE {finite})",
            )
            add(spec, "min", f"min({number}) FILTER (WHERE {finite})")
            add(spec, "max", f"max({number}) FILTER (WHERE {finite})")
            add(spec, "mean", f"avg({number}) FILTER (WHERE {finite})")
            add(spec, "stddev_pop", f"stddev_pop({number}) FILTER (WHERE {finite})")
            add(spec, "p25", f"quantile_cont({number}, 0.25) FILTER (WHERE {finite})")
            add(spec, "p50", f"quantile_cont({number}, 0.50) FILTER (WHERE {finite})")
            add(spec, "p75", f"quantile_cont({number}, 0.75) FILTER (WHERE {finite})")
            add(
                spec,
                "nan_count",
                f"count(*) FILTER (WHERE {quoted} IS NOT NULL AND isnan({number}))",
            )
            add(
                spec,
                "negative_infinity_count",
                "count(*) FILTER (WHERE "
                f"{nonfinite} AND NOT isnan({number}) AND {number} < 0)",
            )
            add(
                spec,
                "positive_infinity_count",
                "count(*) FILTER (WHERE "
                f"{nonfinite} AND NOT isnan({number}) AND {number} > 0)",
            )

        if not selections:
            continue
        row = conn.execute(f"SELECT {', '.join(selections)} FROM {relation}").fetchone()
        if row is None:
            raise DescriptiveInputError("DuckDB returned no aggregate row")
        for spec in batch:
            summary: dict[str, object] = {
                "null_count": int(row[positions[(spec.name, "null_count")]]),
                "distinct_count": int(row[positions[(spec.name, "distinct_count")]]),
            }
            if spec.is_numeric:
                for key in (
                    "finite_count",
                    "nonfinite_count",
                    "unsafe_precision_count",
                    "regular_distinct_count",
                    "nan_count",
                    "negative_infinity_count",
                    "positive_infinity_count",
                ):
                    summary[key] = int(row[positions[(spec.name, key)]])
                for key in ("min", "max", "mean", "stddev_pop", "p25", "p50", "p75"):
                    summary[key] = _json_number(row[positions[(spec.name, key)]])
            else:
                summary["regular_distinct_count"] = summary["distinct_count"]
            summaries[spec.name] = summary
    return summaries


def _public_numeric_summary(summary: Mapping[str, object]) -> dict[str, object]:
    if int(summary["unsafe_precision_count"]) > 0:
        return {
            "basis": "finite_only",
            "status": "unavailable",
            "reason": "unsafe_numeric_precision",
            "finite_count": int(summary["finite_count"]),
            "nonfinite_count": int(summary["nonfinite_count"]),
            "min": None,
            "max": None,
            "mean": None,
            "stddev_pop": None,
            "p25": None,
            "p50": None,
            "p75": None,
        }
    return {
        "basis": "finite_only",
        "finite_count": int(summary["finite_count"]),
        "nonfinite_count": int(summary["nonfinite_count"]),
        "min": summary["min"],
        "max": summary["max"],
        "mean": summary["mean"],
        "stddev_pop": summary["stddev_pop"],
        "p25": summary["p25"],
        "p50": summary["p50"],
        "p75": summary["p75"],
    }


def _build_frequency(
    conn: duckdb.DuckDBPyConnection,
    relation: str,
    spec: _ColumnSpec,
    summary: Mapping[str, object],
    *,
    row_count: int,
    config: DescriptiveConfig,
    sanitizer: ValueSanitizer | None,
) -> dict[str, object]:
    quoted = sql_identifier(spec.name, {spec.name})
    regular_distinct = int(summary["regular_distinct_count"])
    null_count = int(summary["null_count"])
    special_counts: list[tuple[str, int]] = []
    if spec.is_numeric:
        special_counts = [
            ("negative_infinity", int(summary["negative_infinity_count"])),
            ("positive_infinity", int(summary["positive_infinity_count"])),
            ("nan", int(summary["nan_count"])),
        ]
    bucket_count = (
        regular_distinct
        + int(null_count > 0)
        + sum(1 for _, count in special_counts if count > 0)
    )
    exact = bucket_count <= config.low_cardinality_threshold
    limit = regular_distinct if exact else config.frequency_top_k
    buckets: list[_FrequencyBucket] = []

    if limit > 0 and regular_distinct > 0:
        value_expression = quoted if spec.kind != "string" else f"CAST({quoted} AS VARCHAR)"
        predicates = [f"{quoted} IS NOT NULL"]
        if spec.is_numeric:
            predicates.append(f"isfinite(CAST({quoted} AS DOUBLE))")
        query = (
            f"SELECT {value_expression} AS value, count(*) AS n "
            f"FROM {relation} WHERE {' AND '.join(predicates)} "
            f"GROUP BY {value_expression} "
            f"ORDER BY n DESC, value ASC NULLS LAST LIMIT {int(limit)}"
        )
        for value, count in conn.execute(query).fetchall():
            encoded = _encode_scalar(value, declared_kind=spec.kind)
            buckets.append(
                _FrequencyBucket(
                    value=encoded,
                    count=int(count),
                    sort_key=_tagged_scalar_sort_key(encoded),
                    sanitizable=True,
                )
            )

    if null_count:
        value = {"type": "null", "value": None}
        buckets.append(
            _FrequencyBucket(
                value=value,
                count=null_count,
                sort_key=_tagged_scalar_sort_key(value),
                sanitizable=False,
            )
        )
    for nonfinite, count in special_counts:
        if not count:
            continue
        value = {"type": "float", "value": None, "nonfinite": nonfinite}
        buckets.append(
            _FrequencyBucket(
                value=value,
                count=count,
                sort_key=_tagged_scalar_sort_key(value),
                sanitizable=False,
            )
        )

    buckets.sort(key=lambda item: (-item.count, item.sort_key))
    non_missing_count = row_count - null_count
    items: list[dict[str, object]] = []
    for bucket in buckets:
        value = bucket.value
        if sanitizer is not None and bucket.sanitizable:
            value = _sanitize_value(spec.name, value, sanitizer)
        items.append(
            {
                "value": value,
                "count": bucket.count,
                "rate_all": bucket.count / row_count if row_count else 0.0,
                "rate_non_missing": (
                    bucket.count / non_missing_count
                    if value.get("type") != "null" and non_missing_count
                    else None
                ),
            }
        )

    displayed_count = sum(int(item["count"]) for item in items)
    other_count = row_count - displayed_count
    if other_count < 0:
        raise DescriptiveInputError(
            f"frequency count conservation failed for column {spec.name!r}"
        )
    return {
        "mode": "exact" if exact else "top_k",
        "top_k": None if exact else config.frequency_top_k,
        "distinct_bucket_count": bucket_count,
        "values_sanitized": sanitizer is not None,
        "items": items,
        "complete": other_count == 0,
        "other_count": other_count,
        "other_rate_all": other_count / row_count if row_count else 0.0,
        "other_rate_non_missing": (
            other_count / non_missing_count if non_missing_count else None
        ),
    }


def _sanitize_value(
    column: str,
    value: TaggedScalar,
    sanitizer: ValueSanitizer,
) -> TaggedScalar:
    try:
        sanitized = sanitizer(dict(value))
    except Exception as exc:
        raise DescriptiveSanitizerError(
            f"frequency value sanitizer failed for column {column!r}"
        ) from exc
    if not isinstance(sanitized, Mapping):
        raise DescriptiveSanitizerError(
            f"frequency value sanitizer for column {column!r} must return a mapping"
        )
    result = dict(sanitized)
    if not _is_valid_tagged_scalar(result):
        raise DescriptiveSanitizerError(
            f"frequency value sanitizer for column {column!r} must return a tagged scalar"
        )
    _ensure_json_safe(result, context=f"sanitized value for column {column!r}")
    return result


def _build_histogram(
    conn: duckdb.DuckDBPyConnection,
    relation: str,
    spec: _ColumnSpec,
    summary: Mapping[str, object],
    *,
    config: DescriptiveConfig,
) -> dict[str, object]:
    finite_count = int(summary["finite_count"])
    if finite_count == 0:
        return {
            "basis": "finite_only",
            "finite_count": 0,
            "reason": "empty",
            "bins": [],
        }
    if int(summary["unsafe_precision_count"]) > 0:
        return {
            "basis": "finite_only",
            "finite_count": finite_count,
            "reason": "unsafe_numeric_precision",
            "bins": [],
        }
    minimum = float(summary["min"])
    maximum = float(summary["max"])
    if minimum == maximum:
        return {
            "basis": "finite_only",
            "finite_count": finite_count,
            "reason": "constant",
            "bins": [],
        }

    bin_count = config.histogram_bins
    last_index = bin_count - 1
    scale = max(abs(minimum), abs(maximum), 1.0)
    scaled_minimum = minimum / scale
    scaled_range = maximum / scale - scaled_minimum
    quoted = sql_identifier(spec.name, {spec.name})
    query = (
        "WITH finite_values AS ("
        f"SELECT CAST({quoted} AS DOUBLE) AS value FROM {relation} "
        f"WHERE {quoted} IS NOT NULL AND isfinite(CAST({quoted} AS DOUBLE))"
        ") "
        "SELECT CASE "
        f"WHEN value >= ? THEN {last_index} "
        f"ELSE greatest(0, least({last_index}, "
        f"CAST(floor((((value / ?) - ?) / ?) * {bin_count}) AS BIGINT))) END AS bin_index, "
        "count(*) AS n FROM finite_values GROUP BY bin_index ORDER BY bin_index"
    )
    rows = conn.execute(
        query,
        [maximum, scale, scaled_minimum, scaled_range],
    ).fetchall()
    counts = {int(index): int(count) for index, count in rows}
    bins: list[dict[str, object]] = []
    for index in range(bin_count):
        lower = _interpolate(minimum, maximum, index / bin_count)
        upper = _interpolate(minimum, maximum, (index + 1) / bin_count)
        count = counts.get(index, 0)
        bins.append(
            {
                "index": index,
                "lower": lower,
                "upper": upper,
                "lower_inclusive": True,
                "upper_inclusive": index == last_index,
                "count": count,
                "rate_finite": count / finite_count,
            }
        )
    if sum(item["count"] for item in bins) != finite_count:
        raise DescriptiveInputError(
            f"histogram count conservation failed for column {spec.name!r}"
        )
    return {
        "basis": "finite_only",
        "finite_count": finite_count,
        "reason": None,
        "bins": bins,
    }


def _interpolate(minimum: float, maximum: float, fraction: float) -> float:
    if fraction <= 0:
        return minimum
    if fraction >= 1:
        return maximum
    difference = maximum - minimum
    value = (
        minimum + difference * fraction
        if math.isfinite(difference)
        else minimum * (1.0 - fraction) + maximum * fraction
    )
    return float(value)


def _build_correlations(
    conn: duckdb.DuckDBPyConnection,
    relation: str,
    specs: Sequence[_ColumnSpec],
    summaries: Mapping[str, Mapping[str, object]],
    *,
    config: DescriptiveConfig,
) -> dict[str, object]:
    size = len(specs)
    values: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
    pair_counts: list[list[int]] = [[0 for _ in range(size)] for _ in range(size)]
    reasons: list[list[str]] = [
        ["insufficient_pairs" for _ in range(size)] for _ in range(size)
    ]

    for index, spec in enumerate(specs):
        summary = summaries[spec.name]
        count = int(summary["finite_count"])
        pair_counts[index][index] = count
        if count < 2:
            continue
        if int(summary["unsafe_precision_count"]) > 0:
            reasons[index][index] = "unsafe_numeric_precision"
            continue
        if summary["min"] == summary["max"]:
            reasons[index][index] = "zero_variance_both"
            continue
        values[index][index] = 1.0
        reasons[index][index] = "ok"

    pairs = [(left, right) for left in range(size) for right in range(left + 1, size)]
    allowed = {spec.name for spec in specs}
    for start in range(0, len(pairs), config.correlation_batch_size):
        batch = pairs[start : start + config.correlation_batch_size]
        selections: list[str] = []
        for left_index, right_index in batch:
            left = f"CAST({sql_identifier(specs[left_index].name, allowed)} AS DOUBLE)"
            right = f"CAST({sql_identifier(specs[right_index].name, allowed)} AS DOUBLE)"
            finite = f"isfinite({left}) AND isfinite({right})"
            selections.extend(
                [
                    f"count(*) FILTER (WHERE {finite})",
                    f"min({left}) FILTER (WHERE {finite})",
                    f"max({left}) FILTER (WHERE {finite})",
                    f"min({right}) FILTER (WHERE {finite})",
                    f"max({right}) FILTER (WHERE {finite})",
                    f"corr({left}, {right}) FILTER (WHERE {finite})",
                ]
            )
        row = conn.execute(f"SELECT {', '.join(selections)} FROM {relation}").fetchone()
        if row is None:
            raise DescriptiveInputError("DuckDB returned no correlation aggregate row")
        for offset, (left_index, right_index) in enumerate(batch):
            base = offset * 6
            count = int(row[base])
            pair_counts[left_index][right_index] = count
            pair_counts[right_index][left_index] = count
            left_unsafe = int(
                summaries[specs[left_index].name]["unsafe_precision_count"]
            ) > 0
            right_unsafe = int(
                summaries[specs[right_index].name]["unsafe_precision_count"]
            ) > 0
            if count >= 2 and (left_unsafe or right_unsafe):
                if left_unsafe and right_unsafe:
                    reason, value = "unsafe_numeric_precision_both", None
                elif left_unsafe:
                    reason, value = "unsafe_numeric_precision_left", None
                else:
                    reason, value = "unsafe_numeric_precision_right", None
            else:
                reason, value = _correlation_result(
                    count=count,
                    left_min=row[base + 1],
                    left_max=row[base + 2],
                    right_min=row[base + 3],
                    right_max=row[base + 4],
                    correlation=row[base + 5],
                )
            values[left_index][right_index] = value
            values[right_index][left_index] = value
            reasons[left_index][right_index] = reason
            reasons[right_index][left_index] = _transpose_reason(reason)

    return {
        "method": "pearson",
        "basis": "pairwise_finite",
        "columns": [spec.name for spec in specs],
        "values": values,
        "pair_counts": pair_counts,
        "reasons": reasons,
    }


def _correlation_result(
    *,
    count: int,
    left_min: object,
    left_max: object,
    right_min: object,
    right_max: object,
    correlation: object,
) -> tuple[str, float | None]:
    if count < 2:
        return "insufficient_pairs", None
    left_constant = left_min == left_max
    right_constant = right_min == right_max
    if left_constant and right_constant:
        return "zero_variance_both", None
    if left_constant:
        return "zero_variance_left", None
    if right_constant:
        return "zero_variance_right", None
    value = _json_number(correlation)
    if value is None or not isinstance(value, (int, float)):
        return "nonfinite_result", None
    value = float(value)
    if not math.isfinite(value):
        return "nonfinite_result", None
    return "ok", max(-1.0, min(1.0, value))


def _transpose_reason(reason: str) -> str:
    if reason == "zero_variance_left":
        return "zero_variance_right"
    if reason == "zero_variance_right":
        return "zero_variance_left"
    if reason == "unsafe_numeric_precision_left":
        return "unsafe_numeric_precision_right"
    if reason == "unsafe_numeric_precision_right":
        return "unsafe_numeric_precision_left"
    return reason


def _target_distribution(
    target_column: str | None,
    *,
    source_columns: set[str],
    fields_by_name: Mapping[str, Mapping[str, object]],
    auto_included: bool,
) -> dict[str, object]:
    if target_column is None:
        return {"status": "not_configured", "column": None}
    if target_column not in source_columns:
        return {
            "status": "unavailable",
            "column": target_column,
            "reason": "column_not_found",
        }
    field = fields_by_name[target_column]
    return {
        "status": "available",
        "column": target_column,
        "auto_included": auto_included,
        "frequency": field["frequency"],
    }


def _encode_scalar(value: object, *, declared_kind: str) -> TaggedScalar:
    if value is None:
        return {"type": "null", "value": None}
    if declared_kind == "bool" or isinstance(value, bool):
        return {"type": "bool", "value": bool(value)}
    if declared_kind == "date" or isinstance(value, date) and not isinstance(value, datetime):
        date_value = value.isoformat() if isinstance(value, date) else str(value)
        return {"type": "date", "value": date_value}
    if declared_kind == "datetime" or isinstance(value, datetime):
        datetime_value = value.isoformat() if isinstance(value, datetime) else str(value)
        return {"type": "datetime", "value": datetime_value}
    if declared_kind == "numeric":
        if isinstance(value, int):
            integer = int(value)
            if abs(integer) > _MAX_EXACT_DOUBLE_INTEGER:
                return {"type": "bigint", "value": str(integer)}
            return {"type": "int", "value": integer}
        if isinstance(value, Decimal) and value == value.to_integral_value():
            integer = int(value)
            if abs(integer) > _MAX_EXACT_DOUBLE_INTEGER:
                return {"type": "bigint", "value": str(integer)}
            return {"type": "int", "value": integer}
        number = float(value)
        if not math.isfinite(number):
            nonfinite = "nan" if math.isnan(number) else (
                "positive_infinity" if number > 0 else "negative_infinity"
            )
            return {"type": "float", "value": None, "nonfinite": nonfinite}
        return {"type": "float", "value": number}
    return {"type": "string", "value": str(value)}


def _tagged_scalar_sort_key(value: Mapping[str, object]) -> tuple:
    scalar_type = str(value["type"])
    rank = _TYPE_SORT_ORDER[scalar_type]
    raw = value.get("value")
    if scalar_type == "null":
        return (rank, 0)
    if scalar_type == "bool":
        return (rank, int(bool(raw)))
    if scalar_type in {"int", "bigint"}:
        return (rank, Decimal(int(raw)))
    if scalar_type == "float":
        nonfinite = value.get("nonfinite")
        if nonfinite == "negative_infinity":
            return (rank, 0, Decimal(0))
        if nonfinite == "positive_infinity":
            return (rank, 2, Decimal(0))
        if nonfinite == "nan":
            return (rank, 3, Decimal(0))
        return (rank, 1, Decimal(str(raw)))
    return (rank, str(raw))


def _is_valid_tagged_scalar(value: Mapping[str, object]) -> bool:
    scalar_type = value.get("type")
    if scalar_type not in _SCALAR_TYPES or "value" not in value:
        return False
    raw = value.get("value")
    expected_keys = (
        {"type", "value", "nonfinite"}
        if scalar_type == "float" and value.get("nonfinite") is not None
        else {"type", "value"}
    )
    if set(value) != expected_keys:
        return False
    if scalar_type == "null":
        return raw is None and "nonfinite" not in value
    if scalar_type == "bool":
        return isinstance(raw, bool) and "nonfinite" not in value
    if scalar_type == "int":
        return isinstance(raw, int) and not isinstance(raw, bool) and "nonfinite" not in value
    if scalar_type == "bigint":
        return (
            isinstance(raw, str)
            and re.fullmatch(r"-?(?:0|[1-9]\d*)", raw) is not None
            and abs(int(raw)) > _MAX_EXACT_DOUBLE_INTEGER
        )
    if scalar_type == "float":
        nonfinite = value.get("nonfinite")
        if nonfinite is not None:
            return raw is None and nonfinite in {
                "negative_infinity",
                "positive_infinity",
                "nan",
            }
        return (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
        )
    return isinstance(raw, str) and "nonfinite" not in value


def _json_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        number = float(value)
    else:
        number = float(value)
    return number if math.isfinite(number) else None


def _ensure_json_safe(value: object, *, context: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DescriptiveSanitizerError(f"{context} is not strict JSON") from exc


__all__ = [
    "DATA_ANALYSIS_SCHEMA_VERSION",
    "DescriptiveBudgetError",
    "DescriptiveConfig",
    "DescriptiveConfigError",
    "DescriptiveInputError",
    "DescriptiveSanitizerError",
    "TaggedScalar",
    "ValueSanitizer",
    "analyze_parquet",
]

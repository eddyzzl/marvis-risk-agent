"""Governed, deterministic Parquet-to-Parquet data transformations.

The module is the execution kernel for Agent-authored data preparation plans.
It accepts only a closed JSON operation grammar, validates every identifier and
type against the live input schema, and executes through DuckDB.  It never
materializes the full dataset in pandas and never accepts SQL or Python source.

Operations are applied in list order.  Multi-column members of one operation
(``casts``, ``fills``, and ``derivations``) are simultaneous: an item cannot
refer to another item created in that same operation.  A later operation may of
course refer to the preceding operation's output.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from marvis.data.backend import (
    DUCKDB_NUMERIC_TYPES,
    configure_duckdb_defaults,
    connect_duckdb,
    parquet_rel,
    sql_identifier,
    sql_string_literal,
)
from marvis.data.errors import DataLayerError
from marvis.data.predicate_ast import (
    CompiledExpression,
    ExpressionAstBudget,
    PredicateAstBudgetError,
    PredicateAstError,
    compile_expression,
)
from marvis.files import sha256_file


TRANSFORM_RESULT_SCHEMA_VERSION = "transform-result.v1"
TRANSFORM_EXECUTION_MODE = "duckdb-single-thread-v1"
_DETERMINISTIC_DUCKDB_THREADS = 1
_MAX_SAFE_JSON_INTEGER = 2**53 - 1
_TYPE_PATTERN = re.compile(
    r"^(?:"
    r"BOOLEAN|TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|"
    r"UTINYINT|USMALLINT|UINTEGER|UBIGINT|"
    r"REAL|FLOAT|DOUBLE|VARCHAR|DATE|TIMESTAMP|TIMESTAMP WITH TIME ZONE|"
    r"DECIMAL\([1-9]\d*,\s*\d+\)"
    r")$",
    re.IGNORECASE,
)
_DECIMAL_PATTERN = re.compile(r"^DECIMAL\(([1-9]\d*),\s*(\d+)\)$", re.IGNORECASE)
_HARD_CONFIG_MAXIMUMS = {
    "max_operations": 100,
    "max_columns": 1000,
    "max_items_per_operation": 500,
    "max_ast_nodes": 5000,
    "max_ast_depth": 50,
    "max_input_bytes": 100 * 1024**3,
    "max_output_bytes": 100 * 1024**3,
}


class TransformInputError(DataLayerError):
    """The requested transform is invalid or unsafe."""


class TransformExecutionError(DataLayerError):
    """A valid transform could not be evaluated against the source data."""


class TransformConfigError(ValueError):
    """A transform resource limit is invalid or exceeds the hard ceiling."""


class TransformBudgetError(DataLayerError):
    """Input, output, or plan shape exceeds an explicit resource budget."""

    def __init__(self, *, dimension: str, actual: int, limit: int) -> None:
        self.dimension = str(dimension)
        self.actual = int(actual)
        self.limit = int(limit)
        super().__init__(
            f"transform {self.dimension} budget exceeded: "
            f"actual={self.actual}, limit={self.limit}"
        )

    def to_detail(self) -> dict[str, object]:
        return {
            "kind": "transform_budget_exceeded",
            "dimension": self.dimension,
            "actual": self.actual,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class TransformConfig:
    """Hard-bounded execution and grammar budgets.

    Row count is deliberately not capped: DuckDB can stream and spill large
    Parquet inputs.  File bytes, schema width, operation count, per-operation
    fan-out, and expression complexity are bounded explicitly.
    """

    max_operations: int = 50
    max_columns: int = 500
    max_items_per_operation: int = 100
    max_ast_nodes: int = 1000
    max_ast_depth: int = 20
    max_input_bytes: int = 20 * 1024**3
    max_output_bytes: int = 20 * 1024**3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TransformConfigError(f"{name} must be a positive integer")
            maximum = _HARD_CONFIG_MAXIMUMS[name]
            if value > maximum:
                raise TransformConfigError(f"{name} must be at most {maximum}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class _Column:
    name: str
    duckdb_type: str


@dataclass(frozen=True)
class _PreparedOperation:
    op: str
    query: str
    parameters: tuple[object, ...]
    canonical: dict[str, object]
    context: dict[str, object]


def transform_parquet(
    input_path: Path,
    output_path: Path,
    *,
    temp_directory: Path,
    operations: Sequence[Mapping[str, object]],
    config: TransformConfig | None = None,
) -> dict[str, object]:
    """Apply a closed, ordered transform plan and atomically write Parquet.

    ``output_path`` must not already exist.  DuckDB writes to a sibling scratch
    file first; only a fully hashed, JSON-safe result is moved into place.  Any
    validation or execution failure removes the scratch file and leaves no
    output artifact.
    """

    source = Path(input_path)
    output = Path(output_path)
    temp_root = Path(temp_directory)
    effective_config = config or TransformConfig()
    if not isinstance(effective_config, TransformConfig):
        raise TransformConfigError("config must be a TransformConfig")
    operation_list = _validate_paths_and_plan(
        source,
        output,
        operations,
        config=effective_config,
    )
    configure_duckdb_defaults(temp_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    ast_budget = ExpressionAstBudget(
        maximum_nodes=effective_config.max_ast_nodes,
        maximum_depth=effective_config.max_ast_depth,
    )

    try:
        with connect_duckdb(temp_root) as conn:
            # Floating-point aggregates such as AVG are order-sensitive. DuckDB's
            # parallel aggregate merge order can change with the configured thread
            # count, producing a different imputation value (and therefore different
            # Parquet bytes) for the same transform identity. Keep this governed
            # kernel single-threaded regardless of the process-wide performance
            # setting; other data operations continue to use that configured value.
            conn.execute(f"SET threads={_DETERMINISTIC_DUCKDB_THREADS}")
            conn.execute("SET preserve_insertion_order=true")
            current_relation = parquet_rel(source)
            try:
                source_columns = _describe(conn, current_relation)
                source_rows = _row_count(conn, current_relation)
            except duckdb.Error as exc:
                raise TransformInputError(
                    "input is not a readable normalized Parquet dataset"
                ) from exc
            _enforce_budget(
                "columns", len(source_columns), effective_config.max_columns
            )
            if not source_columns:
                raise TransformInputError("input Parquet must contain at least one column")

            current_columns = source_columns
            current_rows = source_rows
            steps: list[dict[str, object]] = []
            canonical_operations: list[dict[str, object]] = []
            current_table: str | None = None
            for index, raw_operation in enumerate(operation_list):
                before_relation = current_relation
                before_columns = current_columns
                before_rows = current_rows
                try:
                    prepared = _prepare_operation(
                        conn,
                        before_relation,
                        before_columns,
                        raw_operation,
                        config=effective_config,
                        ast_budget=ast_budget,
                    )
                    next_table = f"__marvis_transform_step_{index + 1}"
                    next_relation = _internal_identifier(next_table)
                    conn.execute(
                        f"CREATE TEMP TABLE {next_relation} AS {prepared.query}",
                        list(prepared.parameters),
                    )
                    after_columns = _describe(conn, next_relation)
                    _enforce_budget(
                        "columns", len(after_columns), effective_config.max_columns
                    )
                    after_rows = _row_count(conn, next_relation)
                    impact = _operation_impact(
                        conn,
                        before_relation,
                        next_relation,
                        prepared,
                        before_rows=before_rows,
                        after_rows=after_rows,
                    )
                except (TransformInputError, TransformBudgetError):
                    raise
                except duckdb.Error as exc:
                    op_name = str(raw_operation.get("op") or "unknown")
                    raise TransformExecutionError(
                        f"transform step {index + 1} ({op_name}) failed deterministic execution"
                    ) from exc

                steps.append(
                    {
                        "step": index + 1,
                        "op": prepared.op,
                        "row_count_before": before_rows,
                        "row_count_after": after_rows,
                        "row_delta": after_rows - before_rows,
                        "columns_before": _schema_payload(before_columns),
                        "columns_after": _schema_payload(after_columns),
                        "impact": impact,
                    }
                )
                canonical_operations.append(prepared.canonical)
                if current_table is not None:
                    conn.execute(f"DROP TABLE {_internal_identifier(current_table)}")
                current_table = next_table
                current_relation = next_relation
                current_columns = after_columns
                current_rows = after_rows

            quoted_columns = ", ".join(
                _column_identifier(column.name, current_columns)
                for column in current_columns
            )
            conn.execute(
                f"COPY (SELECT {quoted_columns} FROM {current_relation}) "
                f"TO {sql_string_literal(scratch.as_posix())} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )

        size_bytes = scratch.stat().st_size
        _enforce_budget("output_bytes", size_bytes, effective_config.max_output_bytes)
        content_hash = sha256_file(scratch)
        result = {
            "schema_version": TRANSFORM_RESULT_SCHEMA_VERSION,
            "execution": {
                "mode": TRANSFORM_EXECUTION_MODE,
                "duckdb_threads": _DETERMINISTIC_DUCKDB_THREADS,
                "preserve_insertion_order": True,
            },
            "config": effective_config.to_dict(),
            "operations": canonical_operations,
            "steps": steps,
            "summary": {
                "row_count_before": source_rows,
                "row_count_after": steps[-1]["row_count_after"],
                "row_delta": int(steps[-1]["row_count_after"]) - source_rows,
                "column_count_before": len(source_columns),
                "column_count_after": len(current_columns),
                "operation_count": len(steps),
            },
            "source": {
                "format": "parquet",
                "size_bytes": source.stat().st_size,
                "columns": _schema_payload(source_columns),
            },
            "output": {
                "path": str(output),
                "format": "parquet",
                "size_bytes": size_bytes,
                "content_hash": content_hash,
                "hash_algorithm": "sha256",
                "row_count": int(steps[-1]["row_count_after"]),
                "column_count": len(current_columns),
                "columns": _schema_payload(current_columns),
            },
        }
        _assert_json_safe(result)
        if output.exists() or output.is_symlink():
            raise TransformInputError(f"output path already exists: {output}")
        scratch.replace(output)
        return result
    except Exception:
        scratch.unlink(missing_ok=True)
        raise


def _validate_paths_and_plan(
    source: Path,
    output: Path,
    operations: Sequence[Mapping[str, object]],
    *,
    config: TransformConfig,
) -> list[Mapping[str, object]]:
    if source.suffix.lower() != ".parquet":
        raise TransformInputError("transform input must be normalized Parquet")
    if not source.exists() or not source.is_file():
        raise TransformInputError(f"input Parquet does not exist: {source}")
    if source.stat().st_size > config.max_input_bytes:
        raise TransformBudgetError(
            dimension="input_bytes",
            actual=source.stat().st_size,
            limit=config.max_input_bytes,
        )
    try:
        same_path = source.resolve(strict=True) == output.resolve(strict=False)
    except OSError:
        same_path = source.absolute() == output.absolute()
    if same_path:
        raise TransformInputError("output path must be different from input path")
    if output.exists() or output.is_symlink():
        raise TransformInputError(f"output path already exists: {output}")
    if isinstance(operations, (str, bytes)) or not isinstance(operations, Sequence):
        raise TransformInputError("operations must be an ordered sequence")
    operation_list = list(operations)
    if not operation_list:
        raise TransformInputError("operations must contain at least one operation")
    _enforce_budget("operations", len(operation_list), config.max_operations)
    for index, operation in enumerate(operation_list, start=1):
        if not isinstance(operation, Mapping):
            raise TransformInputError(f"operation {index} must be an object")
    return operation_list


def _prepare_operation(
    conn: duckdb.DuckDBPyConnection,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    config: TransformConfig,
    ast_budget: ExpressionAstBudget,
) -> _PreparedOperation:
    op = raw.get("op")
    if not isinstance(op, str) or not op.strip():
        raise TransformInputError("operation op must be a non-empty string")
    dispatch = {
        "rename_columns": _prepare_rename,
        "drop_columns": _prepare_drop,
        "cast_columns": _prepare_cast,
        "fill_missing": _prepare_fill,
        "filter_rows": _prepare_filter,
        "derive_columns": _prepare_derive,
        "deduplicate": _prepare_deduplicate,
    }
    handler = dispatch.get(op)
    if handler is None:
        raise TransformInputError(f"unsupported transform operation: {op}")
    return handler(
        conn,
        relation,
        columns,
        raw,
        config=config,
        ast_budget=ast_budget,
    )


def _prepare_rename(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    **_kwargs,
) -> _PreparedOperation:
    del conn
    _strict_fields(raw, required={"op", "mapping"}, optional=set(), label="rename_columns")
    mapping_value = raw["mapping"]
    if not isinstance(mapping_value, Mapping) or not mapping_value:
        raise TransformInputError("rename_columns mapping must not be empty")
    existing = {column.name for column in columns}
    mapping: dict[str, str] = {}
    for source_name in [column.name for column in columns]:
        if source_name not in mapping_value:
            continue
        target_name = _new_column_name(mapping_value[source_name], label="rename target")
        if source_name == target_name:
            raise TransformInputError("rename_columns cannot rename a column to the same name")
        mapping[source_name] = target_name
    unknown = [key for key in mapping_value if not isinstance(key, str) or key not in existing]
    if unknown:
        raise TransformInputError("rename_columns references an unknown source column")
    final_names = [mapping.get(column.name, column.name) for column in columns]
    duplicate = _first_duplicate(final_names)
    if duplicate is not None:
        raise TransformInputError(f"duplicate output column after rename: {duplicate}")
    select_items = []
    for column, final_name in zip(columns, final_names, strict=True):
        source_sql = _column_identifier(column.name, columns)
        if final_name == column.name:
            select_items.append(source_sql)
        else:
            select_items.append(f"{source_sql} AS {_quote_new_identifier(final_name)}")
    return _PreparedOperation(
        op="rename_columns",
        query=f"SELECT {', '.join(select_items)} FROM {relation}",
        parameters=(),
        canonical={"op": "rename_columns", "mapping": mapping},
        context={"mapping": mapping},
    )


def _prepare_drop(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    config: TransformConfig,
    **_kwargs,
) -> _PreparedOperation:
    del conn
    _strict_fields(raw, required={"op", "columns"}, optional=set(), label="drop_columns")
    requested = _column_name_list(raw["columns"], label="drop_columns columns")
    _enforce_budget("operation_items", len(requested), config.max_items_per_operation)
    existing = {column.name for column in columns}
    unknown = [name for name in requested if name not in existing]
    if unknown:
        raise TransformInputError(f"drop_columns references unknown column: {unknown[0]}")
    requested_set = set(requested)
    kept = [column for column in columns if column.name not in requested_set]
    if not kept:
        raise TransformInputError("drop_columns cannot remove every column")
    canonical_columns = [column.name for column in columns if column.name in requested_set]
    select_sql = ", ".join(_column_identifier(column.name, columns) for column in kept)
    return _PreparedOperation(
        op="drop_columns",
        query=f"SELECT {select_sql} FROM {relation}",
        parameters=(),
        canonical={"op": "drop_columns", "columns": canonical_columns},
        context={"columns": canonical_columns},
    )


def _prepare_cast(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    config: TransformConfig,
    **_kwargs,
) -> _PreparedOperation:
    _strict_fields(raw, required={"op", "casts"}, optional=set(), label="cast_columns")
    items = _object_list(raw["casts"], label="cast_columns casts")
    _enforce_budget("operation_items", len(items), config.max_items_per_operation)
    by_name = {column.name: column for column in columns}
    normalized: dict[str, dict[str, str]] = {}
    for item in items:
        _strict_fields(
            item,
            required={"column", "to_type", "mode"},
            optional=set(),
            label="cast_columns item",
        )
        name = _existing_column_name(item["column"], by_name, label="cast column")
        if name in normalized:
            raise TransformInputError(f"cast_columns repeats column: {name}")
        target_type = _normalize_type(item["to_type"])
        mode = item["mode"]
        if mode not in {"strict", "try"}:
            raise TransformInputError("cast_columns mode must be strict or try")
        canonical_target = _canonical_duckdb_type(conn, target_type)
        if canonical_target.upper() == by_name[name].duckdb_type.upper():
            raise TransformInputError(f"cast_columns would not change type for column: {name}")
        normalized[name] = {
            "column": name,
            "to_type": canonical_target,
            "mode": str(mode),
        }

    canonical_items = [normalized[column.name] for column in columns if column.name in normalized]
    by_column: dict[str, dict[str, object]] = {}
    for item in canonical_items:
        name = item["column"]
        column_sql = _column_identifier(name, columns)
        target_type = item["to_type"]
        row = conn.execute(
            "SELECT "
            f"count(*) FILTER (WHERE {column_sql} IS NOT NULL), "
            f"count(*) FILTER (WHERE {column_sql} IS NOT NULL "
            f"AND TRY_CAST({column_sql} AS {target_type}) IS NULL) "
            f"FROM {relation}"
        ).fetchone()
        non_null = int(row[0]) if row else 0
        invalid = int(row[1]) if row else 0
        if item["mode"] == "strict" and invalid:
            raise TransformExecutionError(
                f"cast_columns strict conversion failed for {name}: "
                f"{invalid} non-null values are not convertible"
            )
        by_column[name] = {
            "mode": item["mode"],
            "non_null_input_count": non_null,
            "invalid_to_null_count": invalid if item["mode"] == "try" else 0,
        }

    select_items = []
    for column in columns:
        column_sql = _column_identifier(column.name, columns)
        item = normalized.get(column.name)
        if item is None:
            select_items.append(column_sql)
            continue
        function = "CAST" if item["mode"] == "strict" else "TRY_CAST"
        select_items.append(
            f"{function}({column_sql} AS {item['to_type']}) "
            f"AS {_quote_new_identifier(column.name)}"
        )
    return _PreparedOperation(
        op="cast_columns",
        query=f"SELECT {', '.join(select_items)} FROM {relation}",
        parameters=(),
        canonical={"op": "cast_columns", "casts": canonical_items},
        context={"by_column": by_column},
    )


def _prepare_fill(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    config: TransformConfig,
    **_kwargs,
) -> _PreparedOperation:
    _strict_fields(raw, required={"op", "fills"}, optional=set(), label="fill_missing")
    items = _object_list(raw["fills"], label="fill_missing fills")
    _enforce_budget("operation_items", len(items), config.max_items_per_operation)
    by_name = {column.name: column for column in columns}
    normalized: dict[str, dict[str, object]] = {}
    replacements: dict[str, tuple[object, str | None]] = {}
    missing_before: dict[str, int] = {}
    for item in items:
        name_value = item.get("column")
        name = _existing_column_name(name_value, by_name, label="fill column")
        if name in normalized:
            raise TransformInputError(f"fill_missing repeats column: {name}")
        method = item.get("method")
        column = by_name[name]
        column_sql = _column_identifier(name, columns)
        if method == "constant":
            _strict_fields(
                item,
                required={"column", "method", "value"},
                optional={"value_type"},
                label="fill_missing constant item",
            )
            value = _validate_literal_value(item["value"])
            if value is None:
                raise TransformInputError("fill_missing constant must not be null")
            value_type = (
                _normalize_type(item["value_type"])
                if "value_type" in item
                else None
            )
            _normalize_type(column.duckdb_type)
            canonical_item: dict[str, object] = {
                "column": name,
                "method": "constant",
                "value": value,
            }
            if value_type is not None:
                value_type = _canonical_duckdb_type(conn, value_type)
                canonical_item["value_type"] = value_type
            replacement = value
        elif method in {"mean", "median", "min", "max"}:
            _strict_fields(
                item,
                required={"column", "method"},
                optional=set(),
                label="fill_missing statistic item",
            )
            if not _is_numeric_type(column.duckdb_type):
                raise TransformInputError(
                    f"fill_missing {method} requires a numeric column: {name}"
                )
            _normalize_type(column.duckdb_type)
            function = {"mean": "avg", "median": "median", "min": "min", "max": "max"}[
                str(method)
            ]
            finite_clause = (
                f" FILTER (WHERE isfinite({column_sql}))"
                if _is_floating_type(column.duckdb_type)
                else ""
            )
            row = conn.execute(
                f"SELECT {function}({column_sql}){finite_clause} FROM {relation}"
            ).fetchone()
            replacement = row[0] if row else None
            if isinstance(replacement, float) and not math.isfinite(replacement):
                raise TransformExecutionError(
                    f"fill_missing {method} produced a non-finite statistic for {name}"
                )
            value_type = None
            canonical_item = {"column": name, "method": str(method)}
        else:
            raise TransformInputError(
                "fill_missing method must be constant, mean, median, min, or max"
            )
        row = conn.execute(
            f"SELECT count(*) FILTER (WHERE {column_sql} IS NULL) FROM {relation}"
        ).fetchone()
        missing_before[name] = int(row[0]) if row else 0
        normalized[name] = canonical_item
        replacements[name] = (replacement, value_type)

    canonical_items = [normalized[column.name] for column in columns if column.name in normalized]
    select_items: list[str] = []
    parameters: list[object] = []
    for column in columns:
        column_sql = _column_identifier(column.name, columns)
        replacement_info = replacements.get(column.name)
        if replacement_info is None:
            select_items.append(column_sql)
            continue
        replacement, value_type = replacement_info
        if replacement is None:
            replacement_sql = "NULL"
        else:
            replacement_sql = "?"
            parameters.append(replacement)
            if value_type is not None:
                replacement_sql = f"CAST({replacement_sql} AS {value_type})"
        select_items.append(
            f"CAST(COALESCE({column_sql}, {replacement_sql}) AS {column.duckdb_type}) "
            f"AS {_quote_new_identifier(column.name)}"
        )
    return _PreparedOperation(
        op="fill_missing",
        query=f"SELECT {', '.join(select_items)} FROM {relation}",
        parameters=tuple(parameters),
        canonical={"op": "fill_missing", "fills": canonical_items},
        context={"missing_before": missing_before},
    )


def _prepare_filter(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    ast_budget: ExpressionAstBudget,
    **_kwargs,
) -> _PreparedOperation:
    del conn
    _strict_fields(raw, required={"op", "predicate"}, optional=set(), label="filter_rows")
    compiled = _compile_expression(
        raw["predicate"],
        columns,
        ast_budget,
        depth=1,
        predicate=True,
    )
    select_sql = ", ".join(_column_identifier(column.name, columns) for column in columns)
    return _PreparedOperation(
        op="filter_rows",
        query=f"SELECT {select_sql} FROM {relation} WHERE {compiled.sql}",
        parameters=compiled.parameters,
        canonical={"op": "filter_rows", "predicate": compiled.canonical},
        context={},
    )


def _prepare_derive(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    config: TransformConfig,
    ast_budget: ExpressionAstBudget,
    **_kwargs,
) -> _PreparedOperation:
    _strict_fields(raw, required={"op", "derivations"}, optional=set(), label="derive_columns")
    items = _object_list(raw["derivations"], label="derive_columns derivations")
    _enforce_budget("operation_items", len(items), config.max_items_per_operation)
    existing = {column.name for column in columns}
    derived_names: set[str] = set()
    compiled_items: list[tuple[str, CompiledExpression, str | None]] = []
    canonical_items: list[dict[str, object]] = []
    for item in items:
        _strict_fields(
            item,
            required={"name", "expression"},
            optional={"to_type"},
            label="derive_columns item",
        )
        name = _new_column_name(item["name"], label="derived column")
        if name in existing:
            raise TransformInputError(f"derive_columns output already exists: {name}")
        if name in derived_names:
            raise TransformInputError(f"derive_columns repeats output name: {name}")
        derived_names.add(name)
        compiled = _compile_expression(
            item["expression"],
            columns,
            ast_budget,
            depth=1,
            predicate=False,
        )
        to_type = None
        if "to_type" in item:
            to_type = _canonical_duckdb_type(conn, _normalize_type(item["to_type"]))
        canonical_item: dict[str, object] = {
            "name": name,
            "expression": compiled.canonical,
        }
        if to_type is not None:
            canonical_item["to_type"] = to_type
        canonical_items.append(canonical_item)
        compiled_items.append((name, compiled, to_type))

    _enforce_budget("columns", len(columns) + len(compiled_items), config.max_columns)
    select_items = [_column_identifier(column.name, columns) for column in columns]
    parameters: list[object] = []
    for name, compiled, to_type in compiled_items:
        expression_sql = compiled.sql
        if to_type is not None:
            expression_sql = f"CAST({expression_sql} AS {to_type})"
        select_items.append(f"{expression_sql} AS {_quote_new_identifier(name)}")
        parameters.extend(compiled.parameters)
    return _PreparedOperation(
        op="derive_columns",
        query=f"SELECT {', '.join(select_items)} FROM {relation}",
        parameters=tuple(parameters),
        canonical={"op": "derive_columns", "derivations": canonical_items},
        context={"columns": [item[0] for item in compiled_items]},
    )


def _prepare_deduplicate(
    conn,
    relation: str,
    columns: list[_Column],
    raw: Mapping[str, object],
    *,
    config: TransformConfig,
    **_kwargs,
) -> _PreparedOperation:
    del conn
    _strict_fields(
        raw,
        required={"op", "keys", "order_by"},
        optional=set(),
        label="deduplicate",
    )
    keys = _column_name_list(raw["keys"], label="deduplicate keys")
    order_items = _object_list(raw["order_by"], label="deduplicate order_by")
    _enforce_budget(
        "operation_items", len(keys) + len(order_items), config.max_items_per_operation
    )
    existing = {column.name for column in columns}
    for name in keys:
        if name not in existing:
            raise TransformInputError(f"deduplicate references unknown key: {name}")
    normalized_order: list[dict[str, str]] = []
    ordered_names: set[str] = set()
    for item in order_items:
        _strict_fields(
            item,
            required={"column"},
            optional={"direction", "nulls"},
            label="deduplicate order item",
        )
        name = _existing_column_name(item["column"], existing, label="order column")
        if name in ordered_names:
            raise TransformInputError(f"deduplicate repeats order column: {name}")
        ordered_names.add(name)
        direction = str(item.get("direction", "asc")).lower()
        nulls = str(item.get("nulls", "last")).lower()
        if direction not in {"asc", "desc"}:
            raise TransformInputError("deduplicate direction must be asc or desc")
        if nulls not in {"first", "last"}:
            raise TransformInputError("deduplicate nulls must be first or last")
        normalized_order.append(
            {"column": name, "direction": direction, "nulls": nulls}
        )

    rank_name = _unique_internal_name("__marvis_dedup_rank", existing)
    key_sql = ", ".join(_column_identifier(name, columns) for name in keys)
    primary_order = [
        f"{_column_identifier(item['column'], columns)} {item['direction'].upper()} "
        f"NULLS {item['nulls'].upper()}"
        for item in normalized_order
    ]
    # Explicit business order remains primary.  Remaining columns are stable
    # tie-breakers; if every value ties, the rows are identical and either copy
    # is observationally equivalent.
    tie_order = [
        f"{_column_identifier(column.name, columns)} ASC NULLS FIRST"
        for column in columns
        if column.name not in ordered_names
    ]
    complete_order = primary_order + tie_order
    if not complete_order:  # Defensive: keys/order validation makes this unreachable.
        raise TransformInputError("deduplicate requires an explicit order")
    select_sql = ", ".join(_column_identifier(column.name, columns) for column in columns)
    rank_sql = _quote_new_identifier(rank_name)
    windowed = (
        f"SELECT {select_sql}, row_number() OVER (PARTITION BY {key_sql} "
        f"ORDER BY {', '.join(complete_order)}) AS {rank_sql} FROM {relation}"
    )
    global_order = [
        f"{_column_identifier(name, columns)} ASC NULLS FIRST" for name in keys
    ] + complete_order
    query = (
        f"SELECT {select_sql} FROM ({windowed}) AS __marvis_ranked "
        f"WHERE {rank_sql} = 1 ORDER BY {', '.join(global_order)}"
    )
    return _PreparedOperation(
        op="deduplicate",
        query=query,
        parameters=(),
        canonical={
            "op": "deduplicate",
            "keys": keys,
            "order_by": normalized_order,
        },
        context={"keys": keys, "order_by": normalized_order},
    )


def _compile_expression(
    value: object,
    columns: list[_Column],
    budget: ExpressionAstBudget,
    *,
    depth: int,
    predicate: bool,
) -> CompiledExpression:
    if depth != 1:
        raise TransformInputError("expression compilation must start at depth 1")
    try:
        return compile_expression(
            value,
            columns=[column.name for column in columns],
            predicate=predicate,
            budget=budget,
        )
    except PredicateAstBudgetError as exc:
        raise TransformBudgetError(
            dimension=exc.dimension,
            actual=exc.actual,
            limit=exc.limit,
        ) from exc
    except PredicateAstError as exc:
        raise TransformInputError(str(exc)) from exc


def _operation_impact(
    conn: duckdb.DuckDBPyConnection,
    before_relation: str,
    after_relation: str,
    prepared: _PreparedOperation,
    *,
    before_rows: int,
    after_rows: int,
) -> dict[str, object]:
    del before_relation
    if prepared.op == "rename_columns":
        mapping = dict(prepared.context["mapping"])
        return {"renamed_count": len(mapping), "mapping": mapping}
    if prepared.op == "drop_columns":
        names = list(prepared.context["columns"])
        return {"dropped_count": len(names), "columns": names}
    if prepared.op == "cast_columns":
        by_column = dict(prepared.context["by_column"])
        return {
            "columns": list(by_column),
            "mode_by_column": {
                name: str(detail["mode"]) for name, detail in by_column.items()
            },
            "non_null_input_count": sum(
                int(detail["non_null_input_count"]) for detail in by_column.values()
            ),
            "invalid_to_null_count": sum(
                int(detail["invalid_to_null_count"]) for detail in by_column.values()
            ),
            "by_column": by_column,
        }
    if prepared.op == "fill_missing":
        missing_before = dict(prepared.context["missing_before"])
        after_columns = _describe(conn, after_relation)
        missing_after: dict[str, int] = {}
        for name in missing_before:
            column_sql = _column_identifier(name, after_columns)
            row = conn.execute(
                f"SELECT count(*) FILTER (WHERE {column_sql} IS NULL) FROM {after_relation}"
            ).fetchone()
            missing_after[name] = int(row[0]) if row else 0
        by_column = {
            name: {
                "missing_before": int(missing_before[name]),
                "missing_after": int(missing_after[name]),
                "filled_count": int(missing_before[name]) - int(missing_after[name]),
            }
            for name in missing_before
        }
        return {
            "columns": list(missing_before),
            "filled_count": sum(int(item["filled_count"]) for item in by_column.values()),
            "by_column": by_column,
        }
    if prepared.op == "filter_rows":
        return {"kept_rows": after_rows, "removed_rows": before_rows - after_rows}
    if prepared.op == "derive_columns":
        names = list(prepared.context["columns"])
        after_columns = _describe(conn, after_relation)
        non_null: dict[str, int] = {}
        for name in names:
            column_sql = _column_identifier(name, after_columns)
            row = conn.execute(
                f"SELECT count(*) FILTER (WHERE {column_sql} IS NOT NULL) FROM {after_relation}"
            ).fetchone()
            non_null[name] = int(row[0]) if row else 0
        return {
            "derived_count": len(names),
            "columns": names,
            "non_null_count_by_column": non_null,
        }
    if prepared.op == "deduplicate":
        return {
            "keys": list(prepared.context["keys"]),
            "order_by": list(prepared.context["order_by"]),
            "kept_rows": after_rows,
            "removed_rows": before_rows - after_rows,
        }
    raise TransformExecutionError(f"unsupported impact operation: {prepared.op}")


def _describe(conn: duckdb.DuckDBPyConnection, relation: str) -> list[_Column]:
    rows = conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    return [_Column(name=str(row[0]), duckdb_type=str(row[1])) for row in rows]


def _row_count(conn: duckdb.DuckDBPyConnection, relation: str) -> int:
    row = conn.execute(f"SELECT count(*) FROM {relation}").fetchone()
    return int(row[0]) if row else 0


def _schema_payload(columns: list[_Column]) -> list[dict[str, str]]:
    return [
        {"name": column.name, "duckdb_type": column.duckdb_type}
        for column in columns
    ]


def _column_identifier(name: str, columns: list[_Column] | set[str]) -> str:
    allowed = (
        columns
        if isinstance(columns, set)
        else {column.name for column in columns}
    )
    return sql_identifier(name, set(allowed))


def _quote_new_identifier(name: str) -> str:
    return sql_identifier(name, {name})


def _internal_identifier(name: str) -> str:
    return sql_identifier(name, {name})


def _new_column_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TransformInputError(f"{label} must be a non-empty string without NUL")
    return value


def _existing_column_name(
    value: object,
    columns: Mapping[str, object] | set[str],
    *,
    label: str,
) -> str:
    if not isinstance(value, str):
        raise TransformInputError(f"{label} must be a string")
    if value not in columns:
        raise TransformInputError(f"{label} is unknown: {value}")
    return value


def _column_name_list(value: object, *, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TransformInputError(f"{label} must be a list")
    result = list(value)
    if not result:
        raise TransformInputError(f"{label} must not be empty")
    names = []
    for item in result:
        if not isinstance(item, str):
            raise TransformInputError(f"{label} entries must be strings")
        names.append(item)
    duplicate = _first_duplicate(names)
    if duplicate is not None:
        raise TransformInputError(f"{label} repeats column: {duplicate}")
    return names


def _object_list(value: object, *, label: str) -> list[Mapping[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TransformInputError(f"{label} must be a list")
    items = list(value)
    if not items:
        raise TransformInputError(f"{label} must not be empty")
    if not all(isinstance(item, Mapping) for item in items):
        raise TransformInputError(f"{label} entries must be objects")
    return items  # type: ignore[return-value]


def _strict_fields(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise TransformInputError(f"{label} missing required fields: {', '.join(sorted(missing))}")
    unexpected = set(value) - required - optional
    if unexpected:
        raise TransformInputError(
            f"{label} has unexpected fields: {', '.join(sorted(str(item) for item in unexpected))}"
        )


def _normalize_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransformInputError("type must be a non-empty string")
    normalized = " ".join(value.strip().upper().split())
    decimal_match = _DECIMAL_PATTERN.fullmatch(normalized)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        if precision > 38 or scale > precision:
            raise TransformInputError("DECIMAL type requires precision <= 38 and scale <= precision")
        return f"DECIMAL({precision},{scale})"
    if not _TYPE_PATTERN.fullmatch(normalized):
        raise TransformInputError(f"unsupported or unsafe DuckDB type: {value}")
    return normalized


def _canonical_duckdb_type(conn: duckdb.DuckDBPyConnection, type_name: str) -> str:
    try:
        row = conn.execute(f"DESCRIBE SELECT CAST(NULL AS {type_name}) AS value").fetchone()
    except duckdb.Error as exc:
        raise TransformInputError(f"unsupported DuckDB type: {type_name}") from exc
    if row is None:
        raise TransformInputError(f"unsupported DuckDB type: {type_name}")
    return str(row[1])


def _is_numeric_type(type_name: str) -> bool:
    base = type_name.upper().split("(", 1)[0].strip()
    return base in DUCKDB_NUMERIC_TYPES


def _is_floating_type(type_name: str) -> bool:
    return type_name.upper().split("(", 1)[0].strip() in {"REAL", "FLOAT", "DOUBLE"}


def _validate_literal_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise TransformInputError(
                "literal integer exceeds exact JSON range; use a decimal string with an explicit type"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TransformInputError("literal floats must be finite")
        return value
    raise TransformInputError("literal must be null, boolean, exact JSON integer, finite float, or string")


def _first_duplicate(values: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _unique_internal_name(prefix: str, existing: set[str]) -> str:
    candidate = prefix
    index = 1
    while candidate in existing:
        candidate = f"{prefix}_{index}"
        index += 1
    return candidate


def _enforce_budget(dimension: str, actual: int, limit: int) -> None:
    if actual > limit:
        raise TransformBudgetError(dimension=dimension, actual=actual, limit=limit)


def _assert_json_safe(value: object) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TransformExecutionError(
            "transform evidence contains a non-JSON or non-finite value"
        ) from exc


__all__ = [
    "TRANSFORM_EXECUTION_MODE",
    "TRANSFORM_RESULT_SCHEMA_VERSION",
    "TransformBudgetError",
    "TransformConfig",
    "TransformConfigError",
    "TransformExecutionError",
    "TransformInputError",
    "transform_parquet",
]

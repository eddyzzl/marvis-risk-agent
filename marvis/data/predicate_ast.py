"""Closed, parameterizable predicate AST shared by governed data workflows.

This module accepts data, never SQL or Python source.  The public predicate
surface is deliberately smaller than the transform expression grammar: only
column/literal leaves, comparisons, boolean composition, and null tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import re

import duckdb
import pandas as pd

from marvis.data.backend import sql_identifier


DEFAULT_MAX_AST_NODES = 1000
DEFAULT_MAX_AST_DEPTH = 20
HARD_MAX_AST_NODES = 5000
HARD_MAX_AST_DEPTH = 50
MAX_SAFE_JSON_INTEGER = 2**53 - 1

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

PREDICATE_OPERATIONS = frozenset(
    {
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "and",
        "or",
        "not",
        "is_null",
        "is_not_null",
    }
)


class PredicateAstError(ValueError):
    """The predicate is outside the closed, safe AST contract."""


class PredicateAstBudgetError(PredicateAstError):
    """The predicate exceeded its explicit node or depth budget."""

    def __init__(self, *, dimension: str, actual: int, limit: int) -> None:
        self.dimension = str(dimension)
        self.actual = int(actual)
        self.limit = int(limit)
        super().__init__(
            f"{self.dimension} budget exceeded: "
            f"actual={self.actual}, limit={self.limit}"
        )


class PredicateEvaluationError(PredicateAstError):
    """A valid predicate could not be evaluated against the supplied frame."""


@dataclass(frozen=True)
class CanonicalPredicate:
    canonical: dict[str, object]
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class CompiledPredicate:
    sql: str
    parameters: tuple[object, ...]
    canonical: dict[str, object]
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalExpression:
    canonical: dict[str, object]
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class CompiledExpression:
    sql: str
    parameters: tuple[object, ...]
    canonical: dict[str, object]
    required_columns: tuple[str, ...]


@dataclass
class ExpressionAstBudget:
    maximum_nodes: int
    maximum_depth: int
    nodes: int = 0

    def __post_init__(self) -> None:
        _validate_limit(
            "maximum_nodes",
            self.maximum_nodes,
            maximum=HARD_MAX_AST_NODES,
        )
        _validate_limit(
            "maximum_depth",
            self.maximum_depth,
            maximum=HARD_MAX_AST_DEPTH,
        )

    def consume(self, depth: int) -> None:
        if depth > self.maximum_depth:
            raise PredicateAstBudgetError(
                dimension="expression_depth",
                actual=depth,
                limit=self.maximum_depth,
            )
        self.nodes += 1
        if self.nodes > self.maximum_nodes:
            raise PredicateAstBudgetError(
                dimension="expression_nodes",
                actual=self.nodes,
                limit=self.maximum_nodes,
            )


def canonicalize_predicate(
    value: object,
    columns: Iterable[str],
    *,
    max_nodes: int = DEFAULT_MAX_AST_NODES,
    max_depth: int = DEFAULT_MAX_AST_DEPTH,
) -> CanonicalPredicate:
    """Validate and canonicalize the strict selector predicate grammar."""

    allowed_columns = _column_set(columns)
    budget = ExpressionAstBudget(
        maximum_nodes=max_nodes,
        maximum_depth=max_depth,
    )
    required_columns: set[str] = set()
    canonical = _canonicalize_expression_node(
        value,
        allowed_columns,
        budget,
        required_columns,
        depth=1,
        predicate=True,
        selector_subset=True,
    )
    return CanonicalPredicate(
        canonical=canonical,
        required_columns=tuple(sorted(required_columns)),
    )


def compile_predicate(
    value: object,
    columns: Iterable[str],
    *,
    max_nodes: int = DEFAULT_MAX_AST_NODES,
    max_depth: int = DEFAULT_MAX_AST_DEPTH,
) -> CompiledPredicate:
    """Compile a strict predicate to quoted SQL plus bound parameters."""

    normalized_columns = _column_names(columns)
    normalized = canonicalize_predicate(
        value,
        normalized_columns,
        max_nodes=max_nodes,
        max_depth=max_depth,
    )
    allowed_columns = set(normalized_columns)
    sql, parameters = _compile_expression_sql(
        normalized.canonical,
        allowed_columns,
    )
    return CompiledPredicate(
        sql=sql,
        parameters=parameters,
        canonical=normalized.canonical,
        required_columns=normalized.required_columns,
    )


def canonicalize_expression(
    value: object,
    columns: Iterable[str],
    *,
    predicate: bool,
    budget: ExpressionAstBudget | None = None,
    max_nodes: int = DEFAULT_MAX_AST_NODES,
    max_depth: int = DEFAULT_MAX_AST_DEPTH,
) -> CanonicalExpression:
    """Canonicalize the complete transform expression grammar."""

    allowed_columns = _column_set(columns)
    effective_budget = budget or ExpressionAstBudget(
        maximum_nodes=max_nodes,
        maximum_depth=max_depth,
    )
    required_columns: set[str] = set()
    canonical = _canonicalize_expression_node(
        value,
        allowed_columns,
        effective_budget,
        required_columns,
        depth=1,
        predicate=predicate,
        selector_subset=False,
    )
    return CanonicalExpression(
        canonical=canonical,
        required_columns=tuple(sorted(required_columns)),
    )


def compile_expression(
    value: object,
    columns: Iterable[str],
    *,
    predicate: bool,
    budget: ExpressionAstBudget | None = None,
    max_nodes: int = DEFAULT_MAX_AST_NODES,
    max_depth: int = DEFAULT_MAX_AST_DEPTH,
) -> CompiledExpression:
    """Compile the complete transform expression grammar to bound SQL."""

    normalized_columns = _column_names(columns)
    normalized = canonicalize_expression(
        value,
        normalized_columns,
        predicate=predicate,
        budget=budget,
        max_nodes=max_nodes,
        max_depth=max_depth,
    )
    sql, parameters = _compile_expression_sql(
        normalized.canonical,
        set(normalized_columns),
    )
    return CompiledExpression(
        sql=sql,
        parameters=parameters,
        canonical=normalized.canonical,
        required_columns=normalized.required_columns,
    )


def evaluate_predicate(
    frame: pd.DataFrame,
    value: object,
    *,
    max_nodes: int = DEFAULT_MAX_AST_NODES,
    max_depth: int = DEFAULT_MAX_AST_DEPTH,
) -> pd.Series:
    """Evaluate a strict predicate with SQL ``WHERE`` null semantics."""

    if not isinstance(frame, pd.DataFrame):
        raise PredicateAstError("predicate source must be a pandas DataFrame")
    compiled = compile_predicate(
        value,
        columns=list(frame.columns),
        max_nodes=max_nodes,
        max_depth=max_depth,
    )
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute("SET threads=1")
        conn.execute("SET preserve_insertion_order=true")
        conn.register("__marvis_predicate_frame", frame)
        rows = conn.execute(
            "SELECT COALESCE(" + compiled.sql + ", FALSE) AS __mask "
            "FROM __marvis_predicate_frame",
            list(compiled.parameters),
        ).fetchall()
    except duckdb.Error as exc:
        raise PredicateEvaluationError(
            "predicate failed deterministic frame evaluation"
        ) from exc
    finally:
        conn.close()
    return pd.Series(
        [bool(row[0]) for row in rows],
        index=frame.index,
        dtype=bool,
        name=None,
    )


def _canonicalize_expression_node(
    value: object,
    columns: set[str],
    budget: ExpressionAstBudget,
    required_columns: set[str],
    *,
    depth: int,
    predicate: bool,
    selector_subset: bool,
) -> dict[str, object]:
    budget.consume(depth)
    if not isinstance(value, Mapping):
        raise PredicateAstError("expression node must be an object")
    if "column" in value:
        _strict_fields(
            value,
            required={"column"},
            optional=set(),
            label="column expression",
        )
        if predicate:
            raise PredicateAstError(
                "predicate must use an explicit comparison or null test"
            )
        name = _existing_column(value["column"], columns)
        required_columns.add(name)
        return {"column": name}
    if "literal" in value:
        _strict_fields(
            value,
            required={"literal"},
            optional=set() if selector_subset else {"type"},
            label="literal expression",
        )
        if predicate:
            raise PredicateAstError("predicate must not be a bare literal")
        canonical: dict[str, object] = {
            "literal": _validate_literal(value["literal"])
        }
        if "type" in value:
            canonical["type"] = _normalize_type(value["type"])
        return canonical

    op = value.get("op")
    if not isinstance(op, str):
        raise PredicateAstError("expression must contain column, literal, or op")
    binary_arithmetic = {"add", "subtract", "multiply", "divide", "modulo"}
    comparisons = {"eq", "ne", "gt", "gte", "lt", "lte"}
    if op in comparisons or (not selector_subset and op in binary_arithmetic):
        _strict_fields(
            value,
            required={"op", "left", "right"},
            optional=set(),
            label=f"{op} expression",
        )
        if predicate and op not in comparisons:
            raise PredicateAstError("predicate root must be boolean")
        return {
            "op": op,
            "left": _canonicalize_expression_node(
                value["left"],
                columns,
                budget,
                required_columns,
                depth=depth + 1,
                predicate=False,
                selector_subset=selector_subset,
            ),
            "right": _canonicalize_expression_node(
                value["right"],
                columns,
                budget,
                required_columns,
                depth=depth + 1,
                predicate=False,
                selector_subset=selector_subset,
            ),
        }
    if op in {"and", "or"}:
        _strict_fields(
            value,
            required={"op", "args"},
            optional=set(),
            label=f"{op} expression",
        )
        args = _expression_args(value["args"], op=op)
        return {
            "op": op,
            "args": [
                _canonicalize_expression_node(
                    arg,
                    columns,
                    budget,
                    required_columns,
                    depth=depth + 1,
                    predicate=True,
                    selector_subset=selector_subset,
                )
                for arg in args
            ],
        }
    unary_operations = {"not", "is_null", "is_not_null"}
    if not selector_subset:
        unary_operations.add("negate")
    if op in unary_operations:
        _strict_fields(
            value,
            required={"op", "arg"},
            optional=set(),
            label=f"{op} expression",
        )
        if predicate and op == "negate":
            raise PredicateAstError("predicate root must be boolean")
        return {
            "op": op,
            "arg": _canonicalize_expression_node(
                value["arg"],
                columns,
                budget,
                required_columns,
                depth=depth + 1,
                predicate=op == "not",
                selector_subset=selector_subset,
            ),
        }
    if not selector_subset and op == "coalesce":
        _strict_fields(
            value,
            required={"op", "args"},
            optional=set(),
            label="coalesce expression",
        )
        if predicate:
            raise PredicateAstError("predicate root must be boolean")
        args = _expression_args(value["args"], op="coalesce")
        return {
            "op": "coalesce",
            "args": [
                _canonicalize_expression_node(
                    arg,
                    columns,
                    budget,
                    required_columns,
                    depth=depth + 1,
                    predicate=False,
                    selector_subset=False,
                )
                for arg in args
            ],
        }
    if not selector_subset and op == "case":
        _strict_fields(
            value,
            required={"op", "cases", "else"},
            optional=set(),
            label="case expression",
        )
        if predicate:
            raise PredicateAstError("predicate root must be boolean")
        cases = _object_list(value["cases"], label="case cases")
        canonical_cases = []
        for case in cases:
            _strict_fields(
                case,
                required={"when", "then"},
                optional=set(),
                label="case branch",
            )
            canonical_cases.append(
                {
                    "when": _canonicalize_expression_node(
                        case["when"],
                        columns,
                        budget,
                        required_columns,
                        depth=depth + 1,
                        predicate=True,
                        selector_subset=False,
                    ),
                    "then": _canonicalize_expression_node(
                        case["then"],
                        columns,
                        budget,
                        required_columns,
                        depth=depth + 1,
                        predicate=False,
                        selector_subset=False,
                    ),
                }
            )
        return {
            "op": "case",
            "cases": canonical_cases,
            "else": _canonicalize_expression_node(
                value["else"],
                columns,
                budget,
                required_columns,
                depth=depth + 1,
                predicate=False,
                selector_subset=False,
            ),
        }
    raise PredicateAstError(f"unsupported expression operation: {op}")


def _compile_expression_sql(
    value: Mapping[str, object],
    columns: set[str],
) -> tuple[str, tuple[object, ...]]:
    if "column" in value:
        return sql_identifier(str(value["column"]), columns), ()
    if "literal" in value:
        if "type" in value:
            return f"CAST(? AS {value['type']})", (value["literal"],)
        return "?", (value["literal"],)
    op = str(value["op"])
    if op in {"and", "or"}:
        compiled = [
            _compile_expression_sql(arg, columns)
            for arg in value["args"]  # type: ignore[union-attr]
        ]
        operator = f" {op.upper()} "
        return (
            "(" + operator.join(item[0] for item in compiled) + ")",
            tuple(
                parameter
                for _, parameters in compiled
                for parameter in parameters
            ),
        )
    if op in {"not", "is_null", "is_not_null", "negate"}:
        child_sql, child_parameters = _compile_expression_sql(
            value["arg"],  # type: ignore[arg-type]
            columns,
        )
        return (
            {
                "not": f"(NOT {child_sql})",
                "is_null": f"({child_sql} IS NULL)",
                "is_not_null": f"({child_sql} IS NOT NULL)",
                "negate": f"(-{child_sql})",
            }[op],
            child_parameters,
        )
    if op == "coalesce":
        compiled = [
            _compile_expression_sql(arg, columns)
            for arg in value["args"]  # type: ignore[union-attr]
        ]
        return (
            "COALESCE(" + ", ".join(item[0] for item in compiled) + ")",
            tuple(
                parameter
                for _, parameters in compiled
                for parameter in parameters
            ),
        )
    if op == "case":
        sql_parts = ["CASE"]
        parameters: list[object] = []
        for case in value["cases"]:  # type: ignore[union-attr]
            when_sql, when_parameters = _compile_expression_sql(
                case["when"],
                columns,
            )
            then_sql, then_parameters = _compile_expression_sql(
                case["then"],
                columns,
            )
            sql_parts.append(f"WHEN {when_sql} THEN {then_sql}")
            parameters.extend(when_parameters)
            parameters.extend(then_parameters)
        else_sql, else_parameters = _compile_expression_sql(
            value["else"],  # type: ignore[arg-type]
            columns,
        )
        sql_parts.append(f"ELSE {else_sql} END")
        parameters.extend(else_parameters)
        return " ".join(sql_parts), tuple(parameters)
    operators = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "modulo": "%",
        "eq": "=",
        "ne": "<>",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }
    left_sql, left_parameters = _compile_expression_sql(
        value["left"],  # type: ignore[arg-type]
        columns,
    )
    right_sql, right_parameters = _compile_expression_sql(
        value["right"],  # type: ignore[arg-type]
        columns,
    )
    return (
        f"({left_sql} {operators[op]} {right_sql})",
        left_parameters + right_parameters,
    )


def _column_set(columns: Iterable[str]) -> set[str]:
    return set(_column_names(columns))


def _column_names(columns: Iterable[str]) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)):
        raise PredicateAstError("columns must be a sequence of strings")
    try:
        normalized = list(columns)
    except TypeError as exc:
        raise PredicateAstError("columns must be a sequence of strings") from exc
    if any(not isinstance(column, str) for column in normalized):
        raise PredicateAstError("columns must be a sequence of strings")
    if len(normalized) != len(set(normalized)):
        raise PredicateAstError("columns must not contain duplicates")
    return tuple(normalized)


def _existing_column(value: object, columns: set[str]) -> str:
    if not isinstance(value, str):
        raise PredicateAstError("expression column must be a string")
    if value not in columns:
        raise PredicateAstError(f"expression column is unknown: {value}")
    return value


def _validate_literal(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise PredicateAstError(
                "literal integer exceeds exact JSON range; "
                "use a decimal string with an explicit type"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PredicateAstError("literal floats must be finite")
        return value
    raise PredicateAstError(
        "literal must be null, boolean, exact JSON integer, finite float, or string"
    )


def _normalize_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredicateAstError("type must be a non-empty string")
    normalized = " ".join(value.strip().upper().split())
    decimal_match = _DECIMAL_PATTERN.fullmatch(normalized)
    if decimal_match:
        precision = int(decimal_match.group(1))
        scale = int(decimal_match.group(2))
        if precision > 38 or scale > precision:
            raise PredicateAstError(
                "DECIMAL type requires precision <= 38 and scale <= precision"
            )
        return f"DECIMAL({precision},{scale})"
    if not _TYPE_PATTERN.fullmatch(normalized):
        raise PredicateAstError(f"unsupported or unsafe DuckDB type: {value}")
    return normalized


def _expression_args(value: object, *, op: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredicateAstError(f"{op} args must be a list")
    args = list(value)
    if len(args) < 2:
        raise PredicateAstError(f"{op} requires at least two arguments")
    return args


def _object_list(value: object, *, label: str) -> list[Mapping[object, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredicateAstError(f"{label} must be a list")
    items = list(value)
    if not items:
        raise PredicateAstError(f"{label} must not be empty")
    if not all(isinstance(item, Mapping) for item in items):
        raise PredicateAstError(f"{label} entries must be objects")
    return items  # type: ignore[return-value]


def _strict_fields(
    value: Mapping[object, object],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise PredicateAstError(
            f"{label} missing required fields: {', '.join(sorted(missing))}"
        )
    unexpected = set(value) - required - optional
    if unexpected:
        raise PredicateAstError(
            f"{label} has unexpected fields: "
            f"{', '.join(sorted(str(item) for item in unexpected))}"
        )


def _validate_limit(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PredicateAstError(f"{name} must be a positive integer")
    if value > maximum:
        raise PredicateAstError(f"{name} must be at most {maximum}")
    return value


__all__ = [
    "CanonicalPredicate",
    "CanonicalExpression",
    "CompiledExpression",
    "CompiledPredicate",
    "DEFAULT_MAX_AST_DEPTH",
    "DEFAULT_MAX_AST_NODES",
    "ExpressionAstBudget",
    "HARD_MAX_AST_DEPTH",
    "HARD_MAX_AST_NODES",
    "MAX_SAFE_JSON_INTEGER",
    "PREDICATE_OPERATIONS",
    "PredicateAstBudgetError",
    "PredicateAstError",
    "PredicateEvaluationError",
    "canonicalize_expression",
    "canonicalize_predicate",
    "compile_expression",
    "compile_predicate",
    "evaluate_predicate",
]

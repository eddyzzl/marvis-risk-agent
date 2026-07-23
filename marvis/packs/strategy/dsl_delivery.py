"""Deterministic, standalone delivery code for canonical Strategy DSL.

The public Python generator lowers first-match strategy rules into disjoint
routing leaves and reuses the same expression interpreter as automatic-tree
delivery.  The generated module contains no MARVIS import and exposes stable
``apply_row`` / ``apply_rows`` functions returning typed action evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from numbers import Integral, Real
import re
import types
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from marvis.packs.strategy.codegen import (
    AutomaticTreeCodegenError,
    _DuckDBExpressionCompiler,
    _expression_fields,
    _generate_python_source_from_leaf_rules,
    _quote_identifier,
    _sql_literal,
)
from marvis.packs.strategy.dsl import (
    StrategyAction,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_strategy_frame


class StrategyDeliveryError(StrategyError):
    """A canonical Strategy DSL delivery cannot be generated safely."""


_DUCKDB_SCALAR_TYPES = frozenset(
    {
        "BOOLEAN",
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
        "FLOAT",
        "DOUBLE",
        "VARCHAR",
    }
)
_MAX_EXACT_DOUBLE_INTEGER = 2**53
MAX_EQUIVALENCE_ROWS = 4096
_EQUIVALENCE_FIELDS = frozenset(
    {
        "schema_version",
        "strategy_spec_hash",
        "source_row_count",
        "sample_count",
        "sample_hash",
        "engines",
        "result_hashes",
        "matched",
        "bounded",
        "equivalence_id",
        "content_hash",
    }
)
_EQUIVALENCE_ENGINES = (
    "marvis_evaluator",
    "python",
    "duckdb_sql",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EQUIVALENCE_ID_RE = re.compile(
    r"^strategy-dsl-equivalence-[0-9a-f]{24}$"
)


def generate_strategy_python_source(
    spec: Mapping[str, Any],
) -> str:
    """Generate a standalone first-match Python implementation of ``spec``."""

    try:
        parsed = parse_strategy_spec(spec)
        if not parsed.rules:
            return _default_only_python_source(parsed.default_action)
        leaves: list[dict[str, Any]] = []
        prior_conditions: list[dict[str, Any]] = []
        results: dict[str, dict[str, Any]] = {}
        for index, rule in enumerate(parsed.rules):
            leaf_id = f"__marvis_strategy_action_{index}"
            condition = _remaining_condition(
                rule.condition,
                prior_conditions=prior_conditions,
            )
            leaves.append(
                {
                    "leaf_id": leaf_id,
                    "rule_id": f"__marvis_strategy_rule_{index}",
                    "condition": condition,
                }
            )
            results[leaf_id] = _result(
                matched_rule_id=rule.rule_id,
                action=rule.action,
            )
            prior_conditions.append(dict(rule.condition))

        default_leaf_id = "__marvis_strategy_default"
        leaves.append(
            {
                "leaf_id": default_leaf_id,
                "rule_id": "__marvis_strategy_default_rule",
                "condition": _default_condition(prior_conditions),
            }
        )
        results[default_leaf_id] = _result(
            matched_rule_id=None,
            action=parsed.default_action,
        )
        base = _generate_python_source_from_leaf_rules(leaves)
    except StrategyDeliveryError:
        raise
    except (AutomaticTreeCodegenError, StrategyError) as exc:
        raise StrategyDeliveryError(str(exc)) from exc
    return _wrap_routing_source(base, results)


def generate_strategy_duckdb_sql_source(
    spec: Mapping[str, Any],
) -> str:
    """Generate DuckDB SQL over the fixed, preflighted ``input_rows`` relation."""

    try:
        parsed = parse_strategy_spec(spec)
        fields = sorted(
            {
                field
                for rule in parsed.rules
                for field in _expression_fields(rule.condition)
            }
        )
        aliases = {
            field: f"__marvis_strategy_field_{index}"
            for index, field in enumerate(fields)
        }
        raw_aliases = {
            field: f"__marvis_strategy_raw_{index}"
            for index, field in enumerate(fields)
        }
        compiler = _DuckDBExpressionCompiler(aliases)
        prior_conditions: list[dict[str, Any]] = []
        conditions: list[str] = []
        for rule in parsed.rules:
            disjoint = _remaining_condition(
                rule.condition,
                prior_conditions=prior_conditions,
            )
            conditions.append(compiler.compile(disjoint))
            prior_conditions.append(dict(rule.condition))

        exact_columns = [
            '    CAST(ROW_NUMBER() OVER () - 1 AS BIGINT) AS "__marvis_row_ordinal"'
        ]
        exact_columns.extend(
            "    COLUMNS(c -> c = "
            f"{_sql_literal(field)}) AS {_quote_identifier(raw_aliases[field])}"
            for field in fields
        )
        source_columns = ['    "__marvis_row_ordinal"']
        source_columns.extend(
            f"    {_quote_identifier(raw_aliases[field])} AS "
            f"{_quote_identifier(aliases[field])}"
            for field in fields
        )
        flag_columns = ['    "__marvis_row_ordinal"']
        flag_columns.extend(
            f'    ({condition}) AS "__marvis_match_{index}"'
            for index, condition in enumerate(conditions)
        )
        match_names = [
            f'"__marvis_match_{index}"' for index in range(len(conditions))
        ]
        results = [
            _result(matched_rule_id=rule.rule_id, action=rule.action)
            for rule in parsed.rules
        ]
        default = _result(
            matched_rule_id=None,
            action=parsed.default_action,
        )
    except StrategyDeliveryError:
        raise
    except (AutomaticTreeCodegenError, StrategyError) as exc:
        raise StrategyDeliveryError(str(exc)) from exc

    projections = [
        _sql_case(
            match_names,
            [item["matched_rule_id"] for item in results],
            default["matched_rule_id"],
        )
        + ' AS "matched_rule_id"',
        _sql_case(
            match_names,
            [item["action_type"] for item in results],
            default["action_type"],
        )
        + ' AS "action_type"',
        _sql_case(
            match_names,
            [_json_scalar(item["action_value"]) for item in results],
            _json_scalar(default["action_value"]),
        )
        + ' AS "action_value_json"',
        _sql_case(
            match_names,
            [_json_scalar(item["decision"]) for item in results],
            _json_scalar(default["decision"]),
        )
        + ' AS "decision_json"',
        _sql_case(
            match_names,
            [item["reason_code"] for item in results],
            default["reason_code"],
        )
        + ' AS "reason_code"',
    ]
    lines = [
        "-- Generated by MARVIS Strategy DSL delivery. Do not edit.",
        "-- Mandatory: validate the same frame before registering input_rows.",
        "SET preserve_insertion_order = true;",
        "",
        'WITH "__marvis_exact_input" AS MATERIALIZED (',
        "  SELECT",
        ",\n".join(exact_columns),
        '  FROM "input_rows"',
        "),",
        '"__marvis_source" AS MATERIALIZED (',
        "  SELECT",
        ",\n".join(source_columns),
        '  FROM "__marvis_exact_input"',
        "),",
        '"__marvis_flags" AS MATERIALIZED (',
        "  SELECT",
        ",\n".join(flag_columns),
        '  FROM "__marvis_source"',
        ")",
        "SELECT",
        '  "__marvis_row_ordinal",',
        ",\n".join(f"  {item}" for item in projections),
        'FROM "__marvis_flags"',
        'ORDER BY "__marvis_row_ordinal";',
    ]
    return "\n".join(lines) + "\n"


def validate_strategy_duckdb_input_frame(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Fail closed before DuckDB registration can coerce Strategy DSL fields."""

    try:
        parsed = parse_strategy_spec(spec)
        fields = sorted(
            {
                field
                for rule in parsed.rules
                for field in _expression_fields(rule.condition)
            }
        )
        auto_numeric_fields = {
            field
            for rule in parsed.rules
            for field in _auto_numeric_fields(rule.condition)
        }
    except (AutomaticTreeCodegenError, StrategyError) as exc:
        raise StrategyDeliveryError(str(exc)) from exc
    if not isinstance(frame, pd.DataFrame):
        raise StrategyDeliveryError("DuckDB input must be a pandas DataFrame")
    columns = frame.columns.tolist()
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise StrategyDeliveryError(
            "DuckDB input frame has duplicate columns: "
            + ", ".join(str(item) for item in duplicates)
        )
    non_text = [item for item in columns if not isinstance(item, str)]
    if non_text:
        raise StrategyDeliveryError(
            "DuckDB input column names must all be text: "
            + ", ".join(repr(item) for item in non_text)
        )
    missing = [field for field in fields if field not in set(columns)]
    if missing:
        raise StrategyDeliveryError(
            "DuckDB input frame is missing exact required fields: "
            + ", ".join(missing)
        )
    folded: dict[str, str] = {}
    for column in columns:
        previous = folded.get(column.casefold())
        if previous is not None:
            raise StrategyDeliveryError(
                "DuckDB input frame has a case-insensitive column collision: "
                f"{previous}, {column}"
            )
        folded[column.casefold()] = column
    for field in fields:
        _validate_scalar_series(
            frame[field],
            field=field,
            auto_numeric=field in auto_numeric_fields,
        )

    relation_name = "__marvis_strategy_delivery_preflight"
    try:
        with duckdb.connect() as connection:
            connection.register(relation_name, frame)
            relation = connection.table(relation_name)
            if relation.columns != columns:
                raise StrategyDeliveryError(
                    "DuckDB registration changed input column names"
                )
            type_by_field = dict(
                zip(
                    relation.columns,
                    (str(item).upper() for item in relation.types),
                    strict=True,
                )
            )
            unsupported = [
                field
                for field in fields
                if type_by_field[field] not in _DUCKDB_SCALAR_TYPES
            ]
            if unsupported:
                raise StrategyDeliveryError(
                    "DuckDB input strategy fields require scalar "
                    "boolean/numeric/text physical types: "
                    + ", ".join(unsupported)
                )
    except StrategyDeliveryError:
        raise
    except Exception as exc:
        raise StrategyDeliveryError(
            f"DuckDB input schema registration failed: {exc}"
        ) from exc
    return frame


def verify_strategy_delivery_equivalence(
    spec: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    maximum_rows: int = MAX_EQUIVALENCE_ROWS,
) -> dict[str, Any]:
    """Reconcile MARVIS, standalone Python, and DuckDB SQL row by row."""

    if (
        isinstance(maximum_rows, bool)
        or not isinstance(maximum_rows, int)
        or maximum_rows < 1
        or maximum_rows > MAX_EQUIVALENCE_ROWS
    ):
        raise StrategyDeliveryError(
            f"maximum_rows must be between 1 and {MAX_EQUIVALENCE_ROWS}"
        )
    parsed = parse_strategy_spec(spec)
    validate_strategy_duckdb_input_frame(frame, parsed.to_dict())
    positions = _sample_positions(len(frame), maximum_rows=maximum_rows)
    sample = frame.iloc[positions].reset_index(drop=True)
    validate_strategy_duckdb_input_frame(sample, parsed.to_dict())

    reference_evaluation = evaluate_strategy_frame(sample, parsed)
    reference = [
        {
            "matched_rule_id": _json_value(
                reference_evaluation.matched_rule_id.iloc[index]
            ),
            "action_type": _json_value(
                reference_evaluation.action_type.iloc[index]
            ),
            "action_value": _json_value(
                reference_evaluation.action_values.iloc[index]
            ),
            "decision": _json_value(
                reference_evaluation.decisions.iloc[index]
            ),
            "reason_code": _json_value(
                reference_evaluation.reason_code.iloc[index]
            ),
        }
        for index in range(len(sample))
    ]
    python_results = _execute_python_delivery(
        generate_strategy_python_source(parsed.to_dict()),
        sample,
    )
    duckdb_results = _execute_duckdb_delivery(
        generate_strategy_duckdb_sql_source(parsed.to_dict()),
        sample,
    )
    results = {
        "marvis_evaluator": reference,
        "python": python_results,
        "duckdb_sql": duckdb_results,
    }
    result_hashes = {
        name: _sha256(_canonical_json(value))
        for name, value in results.items()
    }
    if len(set(result_hashes.values())) != 1 or any(
        value != reference for value in results.values()
    ):
        raise StrategyDeliveryError(
            "generated Strategy DSL delivery does not match the MARVIS evaluator"
        )
    body = {
        "schema_version": "strategy.dsl-delivery-equivalence.v1",
        "strategy_spec_hash": strategy_spec_hash(parsed),
        "source_row_count": len(frame),
        "sample_count": len(sample),
        "sample_hash": _sample_hash(
            sample,
            source_positions=positions,
            spec=parsed.to_dict(),
        ),
        "engines": ["marvis_evaluator", "python", "duckdb_sql"],
        "result_hashes": result_hashes,
        "matched": True,
        "bounded": len(sample) < len(frame),
    }
    equivalence_id = "strategy-dsl-equivalence-" + _sha256(
        _canonical_json(body)
    )[:24]
    document = {**body, "equivalence_id": equivalence_id}
    document["content_hash"] = _sha256(_canonical_json(document))
    return validate_strategy_delivery_equivalence(
        document,
        expected_strategy_spec_hash=document["strategy_spec_hash"],
        expected_sample_hash=document["sample_hash"],
        expected_content_hash=document["content_hash"],
    )


def validate_strategy_delivery_equivalence(
    value: object,
    *,
    expected_strategy_spec_hash: str,
    expected_sample_hash: str,
    expected_content_hash: str,
) -> dict[str, Any]:
    """Validate cached evidence against caller-authenticated source bindings."""

    if not isinstance(value, Mapping) or set(value) != _EQUIVALENCE_FIELDS:
        raise StrategyDeliveryError(
            "strategy delivery equivalence fields are invalid"
        )
    try:
        normalized = json.loads(_canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyDeliveryError(
            "strategy delivery equivalence must be canonical JSON"
        ) from exc
    if (
        normalized["schema_version"]
        != "strategy.dsl-delivery-equivalence.v1"
    ):
        raise StrategyDeliveryError(
            "strategy delivery equivalence schema_version is invalid"
        )
    if normalized["engines"] != list(_EQUIVALENCE_ENGINES):
        raise StrategyDeliveryError(
            "strategy delivery equivalence engines are invalid"
        )
    result_hashes = normalized["result_hashes"]
    if (
        not isinstance(result_hashes, dict)
        or set(result_hashes) != set(_EQUIVALENCE_ENGINES)
    ):
        raise StrategyDeliveryError(
            "strategy delivery equivalence result_hashes are invalid"
        )
    for field in (
        "strategy_spec_hash",
        "sample_hash",
        "content_hash",
        *(
            f"result_hashes.{engine}"
            for engine in _EQUIVALENCE_ENGINES
        ),
    ):
        observed = (
            result_hashes[field.split(".", 1)[1]]
            if field.startswith("result_hashes.")
            else normalized[field]
        )
        if not isinstance(observed, str) or _SHA256_RE.fullmatch(observed) is None:
            raise StrategyDeliveryError(
                f"strategy delivery equivalence {field} is invalid"
            )
    source_count = _non_negative_int(
        normalized["source_row_count"],
        "source_row_count",
    )
    sample_count = _non_negative_int(
        normalized["sample_count"],
        "sample_count",
    )
    if sample_count > source_count or (
        source_count > 0 and sample_count == 0
    ):
        raise StrategyDeliveryError(
            "strategy delivery equivalence sample counts are invalid"
        )
    if sample_count > MAX_EQUIVALENCE_ROWS:
        raise StrategyDeliveryError(
            "strategy delivery equivalence sample_count exceeds its budget"
        )
    if normalized["matched"] is not True:
        raise StrategyDeliveryError(
            "strategy delivery equivalence matched must be true"
        )
    if not isinstance(normalized["bounded"], bool):
        raise StrategyDeliveryError(
            "strategy delivery equivalence bounded must be boolean"
        )
    identifier = normalized["equivalence_id"]
    if (
        not isinstance(identifier, str)
        or _EQUIVALENCE_ID_RE.fullmatch(identifier) is None
    ):
        raise StrategyDeliveryError(
            "strategy delivery equivalence equivalence_id is invalid"
        )

    without_hash = {
        key: normalized[key]
        for key in normalized
        if key != "content_hash"
    }
    if normalized["content_hash"] != _sha256(_canonical_json(without_hash)):
        raise StrategyDeliveryError(
            "strategy delivery equivalence content_hash does not match content"
        )
    body = {
        key: normalized[key]
        for key in normalized
        if key not in {"equivalence_id", "content_hash"}
    }
    expected_id = "strategy-dsl-equivalence-" + _sha256(
        _canonical_json(body)
    )[:24]
    if identifier != expected_id:
        raise StrategyDeliveryError(
            "strategy delivery equivalence equivalence_id does not match content"
        )
    if len(set(result_hashes.values())) != 1:
        raise StrategyDeliveryError(
            "strategy delivery equivalence engine hashes disagree"
        )
    if normalized["bounded"] is not (sample_count < source_count):
        raise StrategyDeliveryError(
            "strategy delivery equivalence bounded flag is inconsistent"
        )
    trusted = {
        "strategy_spec_hash": _trusted_hash(
            expected_strategy_spec_hash,
            "expected_strategy_spec_hash",
        ),
        "sample_hash": _trusted_hash(
            expected_sample_hash,
            "expected_sample_hash",
        ),
        "content_hash": _trusted_hash(
            expected_content_hash,
            "expected_content_hash",
        ),
    }
    if any(normalized[field] != expected for field, expected in trusted.items()):
        raise StrategyDeliveryError(
            "strategy delivery equivalence does not match its trusted binding"
        )
    return normalized


def _trusted_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StrategyDeliveryError(
            f"strategy delivery equivalence {name} is invalid"
        )
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyDeliveryError(
            f"strategy delivery equivalence {name} must be non-negative"
        )
    return value


def _execute_python_delivery(
    source: str,
    sample: pd.DataFrame,
) -> list[dict[str, Any]]:
    module = types.ModuleType("__marvis_generated_strategy")
    try:
        exec(compile(source, "<generated-strategy>", "exec"), module.__dict__)
        raw = module.apply_rows(sample)
    except Exception as exc:
        raise StrategyDeliveryError(
            f"generated Python delivery failed: {exc}"
        ) from exc
    return [_canonical_result(item) for item in raw]


def _execute_duckdb_delivery(
    source: str,
    sample: pd.DataFrame,
) -> list[dict[str, Any]]:
    try:
        with duckdb.connect() as connection:
            connection.register("input_rows", sample)
            rows = connection.sql(source).fetchall()
    except Exception as exc:
        raise StrategyDeliveryError(
            f"generated DuckDB SQL delivery failed: {exc}"
        ) from exc
    return [
        _canonical_result(
            {
                "matched_rule_id": row[1],
                "action_type": row[2],
                "action_value": json.loads(row[3]),
                "decision": json.loads(row[4]),
                "reason_code": row[5],
            }
        )
        for row in rows
    ]


def _canonical_result(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "matched_rule_id",
        "action_type",
        "action_value",
        "decision",
        "reason_code",
    }:
        raise StrategyDeliveryError("generated delivery result shape is invalid")
    return {
        "matched_rule_id": _json_value(value["matched_rule_id"]),
        "action_type": _json_value(value["action_type"]),
        "action_value": _json_value(value["action_value"]),
        "decision": _json_value(value["decision"]),
        "reason_code": _json_value(value["reason_code"]),
    }


def _sample_positions(row_count: int, *, maximum_rows: int) -> list[int]:
    if row_count <= maximum_rows:
        return list(range(row_count))
    if maximum_rows == 1:
        return [row_count - 1]
    return [
        (index * (row_count - 1)) // (maximum_rows - 1)
        for index in range(maximum_rows)
    ]


def _sample_hash(
    sample: pd.DataFrame,
    *,
    source_positions: Sequence[int],
    spec: Mapping[str, Any],
) -> str:
    fields = sorted(
        {
            field
            for rule in parse_strategy_spec(spec).rules
            for field in _expression_fields(rule.condition)
        }
    )
    rows = [
        {
            "source_position": source_positions[index],
            "values": {
                field: _json_value(sample.iloc[index][field])
                for field in fields
            },
        }
        for index in range(len(sample))
    ]
    return _sha256(_canonical_json(rows))


def _json_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number):
            return number
        return {"__marvis_float__": "Infinity" if number > 0 else "-Infinity"}
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        return [_json_value(item) for item in value]
    raise StrategyDeliveryError("strategy delivery result is not canonical JSON")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _remaining_condition(
    condition: Mapping[str, Any],
    *,
    prior_conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    current = dict(condition)
    if not prior_conditions:
        return current
    prior = (
        prior_conditions[0]
        if len(prior_conditions) == 1
        else {"op": "or", "args": list(prior_conditions)}
    )
    return {
        "op": "and",
        "args": [
            {"op": "not", "arg": prior},
            current,
        ],
    }


def _default_condition(
    prior_conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    prior = (
        prior_conditions[0]
        if len(prior_conditions) == 1
        else {"op": "or", "args": list(prior_conditions)}
    )
    return {"op": "not", "arg": prior}


def _sql_case(
    conditions: Sequence[str],
    values: Sequence[str | None],
    default: str | None,
) -> str:
    if len(conditions) != len(values):
        raise StrategyDeliveryError("Strategy SQL routing values are inconsistent")
    if not conditions:
        return _nullable_sql_text(default)
    branches = " ".join(
        f"WHEN {condition} THEN {_nullable_sql_text(value)}"
        for condition, value in zip(conditions, values, strict=True)
    )
    return f"CASE {branches} ELSE {_nullable_sql_text(default)} END"


def _nullable_sql_text(value: str | None) -> str:
    return (
        "CAST(NULL AS VARCHAR)"
        if value is None
        else _sql_literal(value)
    )


def _json_scalar(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_scalar_series(
    series: pd.Series,
    *,
    field: str,
    auto_numeric: bool,
) -> None:
    kinds: set[str] = set()
    for value in series.tolist():
        if _is_missing(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            kinds.add("boolean")
            continue
        if isinstance(value, Integral):
            if not -_MAX_EXACT_DOUBLE_INTEGER <= int(value) <= _MAX_EXACT_DOUBLE_INTEGER:
                raise StrategyDeliveryError(
                    f"DuckDB input integer exceeds exact comparison range: {field}"
                )
            kinds.add("numeric")
            continue
        if isinstance(value, Real):
            kinds.add("numeric")
            continue
        if isinstance(value, str):
            if auto_numeric:
                _validate_auto_numeric_text(value, field=field)
            kinds.add("text")
            continue
        raise StrategyDeliveryError(
            f"DuckDB input strategy field contains an unsupported scalar: {field}"
        )
    if len(kinds) > 1:
        raise StrategyDeliveryError(
            "DuckDB input strategy field mixes scalar domains and could be "
            f"coerced during registration: {field}"
        )


def _validate_auto_numeric_text(value: str, *, field: str) -> None:
    try:
        numeric = Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise StrategyDeliveryError(
            "DuckDB input auto-numeric text is not numeric: " + field
        ) from exc
    if not numeric.is_finite():
        raise StrategyDeliveryError(
            "DuckDB input auto-numeric text must be finite: " + field
        )
    if (
        numeric == numeric.to_integral_value()
        and abs(numeric) > _MAX_EXACT_DOUBLE_INTEGER
    ):
        raise StrategyDeliveryError(
            f"DuckDB input integer exceeds exact comparison range: {field}"
        )
    try:
        converted = float(numeric)
    except (OverflowError, ValueError) as exc:
        raise StrategyDeliveryError(
            "DuckDB input auto-numeric text exceeds finite DOUBLE range: "
            + field
        ) from exc
    if not math.isfinite(converted):
        raise StrategyDeliveryError(
            "DuckDB input auto-numeric text exceeds finite DOUBLE range: "
            + field
        )


def _auto_numeric_fields(expression: Mapping[str, Any]) -> set[str]:
    op = expression["op"]
    if op == "compare":
        if expression.get("coercion", "auto") == "strict":
            return set()
        expected = expression["value"]
        if expression["operator"] in {"in", "not_in"}:
            candidates = list(expected)
            numeric = bool(candidates) and all(
                _numeric_literal(item) for item in candidates
            )
        else:
            numeric = _numeric_literal(expected)
        return {str(expression["field"])} if numeric else set()
    if op == "between":
        numeric = _numeric_literal(expression["lower"]) and _numeric_literal(
            expression["upper"]
        )
        return {str(expression["field"])} if numeric else set()
    if op in {"and", "or", "n_of_k"}:
        return {
            field
            for argument in expression["args"]
            for field in _auto_numeric_fields(argument)
        }
    if op == "not":
        return _auto_numeric_fields(expression["arg"])
    if op in {"is_null", "is_not_null"}:
        return set()
    raise StrategyDeliveryError(f"unsupported Strategy DSL expression op: {op}")


def _numeric_literal(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        marker = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(marker) if isinstance(marker, (bool, np.bool_)) else False


def _result(
    *,
    matched_rule_id: str | None,
    action: StrategyAction,
) -> dict[str, Any]:
    return {
        "matched_rule_id": matched_rule_id,
        "action_type": action.type,
        "action_value": action.value,
        "decision": action.decision_value,
        "reason_code": action.reason_code,
    }


def _wrap_routing_source(
    source: str,
    results: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = json.dumps(
        results,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload_literal = json.dumps(payload, ensure_ascii=True)
    header = "# Generated by MARVIS Strategy DSL delivery. Do not edit."
    _, separator, remainder = source.partition("\n")
    base = header + (separator + remainder if separator else "\n")
    wrapper = f'''

_MARVIS_DELIVERY_RESULTS = _json.loads({payload_literal})
_marvis_route_row = apply_row
_marvis_route_rows = apply_rows


def _marvis_delivery_result(routed: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_MARVIS_DELIVERY_RESULTS[routed["leaf_id"]])


def apply_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the canonical first-match strategy to one row."""

    return _marvis_delivery_result(_marvis_route_row(row))


def apply_rows(rows: Iterable[Mapping[str, Any]] | _pd.DataFrame) -> list[dict[str, Any]]:
    """Apply the canonical first-match strategy to a row batch."""

    return [_marvis_delivery_result(item) for item in _marvis_route_rows(rows)]


__all__ = ["apply_row", "apply_rows"]
'''
    return base.rstrip() + "\n" + wrapper.lstrip()


def _default_only_python_source(action: StrategyAction) -> str:
    result = json.dumps(
        _result(matched_rule_id=None, action=action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result_literal = json.dumps(result, ensure_ascii=True)
    return f'''# Generated by MARVIS Strategy DSL delivery. Do not edit.
from __future__ import annotations

import json as _json
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as _pd


_MARVIS_DEFAULT_RESULT = _json.loads({result_literal})


def _result() -> dict[str, Any]:
    return dict(_MARVIS_DEFAULT_RESULT)


def apply_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) and not isinstance(row, _pd.Series):
        raise ValueError("strategy row must be a mapping")
    return _result()


def apply_rows(rows: Iterable[Mapping[str, Any]] | _pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(rows, _pd.DataFrame):
        count = len(rows)
    else:
        try:
            materialized = list(rows)
        except TypeError as exc:
            raise ValueError("strategy rows must be iterable") from exc
        if any(not isinstance(row, Mapping) for row in materialized):
            raise ValueError("strategy row must be a mapping")
        count = len(materialized)
    return [_result() for _ in range(count)]


__all__ = ["apply_row", "apply_rows"]
'''


__all__ = [
    "StrategyDeliveryError",
    "generate_strategy_duckdb_sql_source",
    "generate_strategy_python_source",
    "validate_strategy_delivery_equivalence",
    "validate_strategy_duckdb_input_frame",
    "verify_strategy_delivery_equivalence",
]

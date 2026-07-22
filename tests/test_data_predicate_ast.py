from __future__ import annotations

import pandas as pd
import pytest

from marvis.data.predicate_ast import (
    PredicateAstBudgetError,
    PredicateAstError,
    canonicalize_predicate,
    compile_expression,
    compile_predicate,
    evaluate_predicate,
)


def test_predicate_canonicalization_reports_sorted_required_columns():
    predicate = {
        "op": "and",
        "args": [
            {
                "op": "gte",
                "left": {"column": "age"},
                "right": {"literal": 21},
            },
            {
                "op": "not",
                "arg": {
                    "op": "is_null",
                    "arg": {"column": "segment"},
                },
            },
        ],
    }

    result = canonicalize_predicate(
        predicate,
        columns=pd.Index(["segment", "age", "unused"]),
    )

    assert result.canonical == predicate
    assert result.required_columns == ("age", "segment")


def test_predicate_compilation_quotes_columns_and_binds_literals():
    literal = "x' OR TRUE --"
    predicate = {
        "op": "eq",
        "left": {"column": 'customer"segment'},
        "right": {"literal": literal},
    }

    compiled = compile_predicate(predicate, columns=['customer"segment'])

    assert compiled.sql == '("customer""segment" = ?)'
    assert compiled.parameters == (literal,)
    assert literal not in compiled.sql
    assert compiled.canonical == predicate
    assert compiled.required_columns == ('customer"segment',)


def test_predicate_evaluation_returns_deterministic_where_mask_with_nulls():
    frame = pd.DataFrame(
        {
            "age": [18.0, 25.0, 30.0, None],
            "segment": ["A", "B", None, "A"],
        },
        index=[11, 12, 13, 14],
    )
    predicate = {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [
                    {
                        "op": "gte",
                        "left": {"column": "age"},
                        "right": {"literal": 21},
                    },
                    {
                        "op": "ne",
                        "left": {"column": "segment"},
                        "right": {"literal": "B"},
                    },
                ],
            },
            {"op": "is_null", "arg": {"column": "age"}},
        ],
    }

    mask = evaluate_predicate(frame, predicate)

    assert mask.index.tolist() == [11, 12, 13, 14]
    assert mask.dtype == bool
    assert mask.tolist() == [False, False, False, True]


def test_full_transform_expression_compiles_typed_arithmetic_without_inline_values():
    expression = {
        "op": "add",
        "left": {"column": "amount"},
        "right": {"literal": "1.50", "type": "decimal(4, 2)"},
    }

    compiled = compile_expression(expression, columns=["amount"], predicate=False)

    assert compiled.sql == '("amount" + CAST(? AS DECIMAL(4,2)))'
    assert compiled.parameters == ("1.50",)
    assert compiled.canonical == {
        "op": "add",
        "left": {"column": "amount"},
        "right": {"literal": "1.50", "type": "DECIMAL(4,2)"},
    }
    assert compiled.required_columns == ("amount",)


def test_full_transform_expression_keeps_case_coalesce_and_unary_grammar():
    expression = {
        "op": "case",
        "cases": [
            {
                "when": {
                    "op": "and",
                    "args": [
                        {
                            "op": "gte",
                            "left": {"column": "amount"},
                            "right": {"literal": 20, "type": "double"},
                        },
                        {
                            "op": "is_not_null",
                            "arg": {"column": "segment"},
                        },
                    ],
                },
                "then": {
                    "op": "coalesce",
                    "args": [
                        {"column": "segment"},
                        {"literal": "UNKNOWN"},
                    ],
                },
            }
        ],
        "else": {"op": "negate", "arg": {"column": "amount"}},
    }

    compiled = compile_expression(
        expression,
        columns=["segment", "amount"],
        predicate=False,
    )

    assert compiled.sql == (
        'CASE WHEN (("amount" >= CAST(? AS DOUBLE)) AND '
        '("segment" IS NOT NULL)) THEN COALESCE("segment", ?) '
        'ELSE (-"amount") END'
    )
    assert compiled.parameters == (20, "UNKNOWN")
    assert compiled.canonical["cases"][0]["when"]["args"][0]["right"][  # type: ignore[index]
        "type"
    ] == "DOUBLE"
    assert compiled.required_columns == ("amount", "segment")


@pytest.mark.parametrize(
    "op,expected",
    [
        ("eq", [False, True, False]),
        ("ne", [True, False, False]),
        ("gt", [False, False, False]),
        ("gte", [False, True, False]),
        ("lt", [True, False, False]),
        ("lte", [True, True, False]),
    ],
)
def test_predicate_comparisons_share_sql_where_null_semantics(op, expected):
    frame = pd.DataFrame({"value": [1.0, 2.0, None]})
    predicate = {
        "op": op,
        "left": {"column": "value"},
        "right": {"literal": 2},
    }

    assert evaluate_predicate(frame, predicate).tolist() == expected


@pytest.mark.parametrize(
    "predicate,columns",
    [
        ({"sql": "TRUE; DROP TABLE samples"}, ["value"]),
        ({"op": "python", "source": "open('/tmp/x')"}, ["value"]),
        (
            {
                "op": "add",
                "left": {"column": "value"},
                "right": {"literal": 1},
            },
            ["value"],
        ),
        (
            {
                "op": "eq",
                "left": {"column": "missing"},
                "right": {"literal": 1},
            },
            ["value"],
        ),
        (
            {
                "op": "eq",
                "left": {"column": "value"},
                "right": {"literal": float("nan")},
            },
            ["value"],
        ),
        (
            {
                "op": "eq",
                "left": {"column": "value"},
                "right": {"literal": float("inf")},
            },
            ["value"],
        ),
        (
            {
                "op": "eq",
                "left": {"column": "value"},
                "right": {"literal": 2**53},
            },
            ["value"],
        ),
        (
            {
                "op": "eq",
                "left": {"column": "value"},
                "right": {"literal": "1", "type": "INTEGER"},
            },
            ["value"],
        ),
    ],
)
def test_selector_predicate_rejects_code_unknown_fields_and_unsafe_literals(
    predicate,
    columns,
):
    with pytest.raises(PredicateAstError):
        canonicalize_predicate(predicate, columns)


def test_predicate_node_and_depth_budgets_fail_closed():
    comparison = {
        "op": "eq",
        "left": {"column": "value"},
        "right": {"literal": 1},
    }
    with pytest.raises(PredicateAstBudgetError) as nodes:
        canonicalize_predicate(comparison, ["value"], max_nodes=2)
    assert (nodes.value.dimension, nodes.value.actual, nodes.value.limit) == (
        "expression_nodes",
        3,
        2,
    )

    with pytest.raises(PredicateAstBudgetError) as depth:
        canonicalize_predicate(
            {"op": "not", "arg": comparison},
            ["value"],
            max_depth=2,
        )
    assert (depth.value.dimension, depth.value.actual, depth.value.limit) == (
        "expression_depth",
        3,
        2,
    )

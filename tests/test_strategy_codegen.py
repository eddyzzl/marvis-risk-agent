from __future__ import annotations

import importlib.util
import types
from decimal import Decimal
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import (
    apply_weighted_rule_tree,
    build_weighted_rule_tree,
)
from marvis.packs.strategy.automatic_tree_asset import build_automatic_tree_asset
from marvis.packs.strategy.codegen import (
    AutomaticTreeCodegenError,
    _generate_duckdb_sql_from_leaf_rules,
    _generate_python_source_from_leaf_rules,
    generate_automatic_tree_duckdb_sql_source,
    generate_automatic_tree_python_source,
    validate_automatic_tree_duckdb_input_frame,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _tree(
    frame: pd.DataFrame | None = None,
    *,
    feature: str = 'select "中文 额度"',
) -> dict:
    source = (
        frame
        if frame is not None
        else pd.DataFrame(
            {
                feature: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "bad": [0, 0, 1, 0, 1, 1],
            }
        )
    )
    return build_weighted_rule_tree(
        source,
        feature_cols=[feature],
        target_col="bad",
        max_depth=2,
        min_leaf_count=1,
    )


def _asset(tree: dict) -> dict:
    return build_automatic_tree_asset(
        tree,
        task_id="task-codegen",
        dataset_id="dataset-codegen",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=["dataset:dataset-codegen"],
    )


def _load_generated_python(source: str, tmp_path: Path) -> types.ModuleType:
    path = tmp_path / "generated_automatic_tree.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generated_automatic_tree", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaf_rule_pairs(tree: dict) -> dict[str, str]:
    return {rule["leaf_id"]: rule["rule_id"] for rule in tree["rules"]}


def _duckdb_results(
    frame: pd.DataFrame,
    sql: str,
    *,
    initial_preserve_insertion_order: bool | None = None,
) -> list[dict[str, object]]:
    with duckdb.connect() as connection:
        if initial_preserve_insertion_order is not None:
            enabled = str(initial_preserve_insertion_order).lower()
            connection.execute(f"SET preserve_insertion_order = {enabled}")
        connection.register("input_rows", frame)
        result = connection.sql(sql).df()
        preserve_order = connection.sql(
            "SELECT current_setting('preserve_insertion_order')"
        ).fetchone()
        assert preserve_order == (True,)
    assert result.columns.tolist() == [
        "__marvis_row_ordinal",
        "leaf_id",
        "rule_id",
    ]
    return result.to_dict(orient="records")


def test_public_codegen_is_byte_deterministic_for_tree_and_committed_asset() -> None:
    tree = _tree()
    asset = _asset(tree)

    python_source = generate_automatic_tree_python_source(tree)
    sql_source = generate_automatic_tree_duckdb_sql_source(tree)

    assert python_source == generate_automatic_tree_python_source(tree)
    assert python_source == generate_automatic_tree_python_source(asset)
    assert sql_source == generate_automatic_tree_duckdb_sql_source(tree)
    assert sql_source == generate_automatic_tree_duckdb_sql_source(asset)
    assert python_source.endswith("\n")
    assert sql_source.endswith("\n")
    assert "eval(" not in python_source
    assert "approve" not in python_source.lower()
    assert "reject" not in python_source.lower()
    assert "approve" not in sql_source.lower()
    assert "reject" not in sql_source.lower()
    assert 'FROM "input_rows"' in sql_source
    assert "SET preserve_insertion_order = true;" in sql_source
    assert "COLUMNS(c -> c =" in sql_source
    assert "validate_automatic_tree_duckdb_input_frame" in sql_source
    assert "casefold collisions" in sql_source
    assert "byte-identical schema" in sql_source
    assert "LIMIT (SELECT CASE" in sql_source
    assert '"__marvis_row_ordinal"' in sql_source


def test_public_codegen_rejects_unknown_or_tampered_schema() -> None:
    with pytest.raises(AutomaticTreeCodegenError, match="schema_version"):
        generate_automatic_tree_python_source({"rules": []})

    tampered = _tree()
    tampered["rules"][0]["leaf_id"] = "leaf-tampered"
    with pytest.raises(AutomaticTreeCodegenError, match="weighted rule tree"):
        generate_automatic_tree_duckdb_sql_source(tampered)

    asset = _asset(_tree())
    asset["tree_result"]["rules"][0]["rule_id"] = "rule-tampered"
    with pytest.raises(AutomaticTreeCodegenError, match="automatic tree asset"):
        generate_automatic_tree_python_source(asset)


def test_generated_python_and_duckdb_match_weighted_tree_row_by_row(
    tmp_path: Path,
) -> None:
    feature = 'select "中文 额度"'
    training = pd.DataFrame(
        {
            feature: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )
    tree = _tree(training, feature=feature)
    scored = pd.DataFrame({feature: [None, np.nan, 0.0, 1.5, 2.0, 4.0, 99.0]})
    reference_leaf_ids = apply_weighted_rule_tree(scored, tree).tolist()
    rule_ids = _leaf_rule_pairs(tree)
    expected = [
        {
            "__marvis_row_ordinal": index,
            "leaf_id": leaf_id,
            "rule_id": rule_ids[leaf_id],
        }
        for index, leaf_id in enumerate(reference_leaf_ids)
    ]

    module = _load_generated_python(
        generate_automatic_tree_python_source(tree), tmp_path
    )
    python_rows = module.apply_rows(scored.to_dict(orient="records"))
    assert python_rows == [
        {"leaf_id": row["leaf_id"], "rule_id": row["rule_id"]} for row in expected
    ]
    assert validate_automatic_tree_duckdb_input_frame(scored, tree) is scored
    assert (
        _duckdb_results(scored, generate_automatic_tree_duckdb_sql_source(tree))
        == expected
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        "1",
        "1_0",
        Decimal("1.0"),
        float("inf"),
        float("-inf"),
        10**1000,
    ],
)
def test_public_python_rejects_values_outside_weighted_tree_domain(
    invalid_value: object,
    tmp_path: Path,
) -> None:
    tree = _tree(feature="x")
    module = _load_generated_python(
        generate_automatic_tree_python_source(tree), tmp_path
    )

    with pytest.raises(ValueError, match="weighted-tree numeric feature"):
        module.apply_rows([{"x": invalid_value}])


def test_public_python_rejects_any_duplicate_dataframe_column(tmp_path: Path) -> None:
    tree = _tree(feature="x")
    module = _load_generated_python(
        generate_automatic_tree_python_source(tree), tmp_path
    )
    duplicate = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        columns=["x", "duplicate", "duplicate"],
    )

    with pytest.raises(ValueError, match="duplicate columns"):
        module.apply_rows(duplicate)


@pytest.mark.parametrize(
    "feature",
    [
        "__MARVIS_FIELDS_JSON_LITERAL__",
        "__MARVIS_ENFORCE_WEIGHTED_TREE_DOMAIN__",
        "$marvis_rules_json_literal",
        "$marvis_fields_json_literal",
        "$marvis_enforce_weighted_tree_domain",
    ],
)
def test_generated_python_treats_template_tokens_as_field_data(
    feature: str,
    tmp_path: Path,
) -> None:
    tree = _tree(feature=feature)
    frame = pd.DataFrame({feature: [0.0, 4.0]})
    leaf_ids = apply_weighted_rule_tree(frame, tree).tolist()
    rule_ids = _leaf_rule_pairs(tree)
    module = _load_generated_python(
        generate_automatic_tree_python_source(tree), tmp_path
    )

    assert module.apply_rows(frame) == [
        {"leaf_id": leaf_id, "rule_id": rule_ids[leaf_id]} for leaf_id in leaf_ids
    ]


def test_duckdb_preflight_blocks_registration_synthesized_required_field(
    tmp_path: Path,
) -> None:
    tree = _tree(feature="score_1")
    frame = pd.DataFrame({"Score": [0.0], "score": [1.0]})
    module = _load_generated_python(
        generate_automatic_tree_python_source(tree), tmp_path
    )

    with duckdb.connect() as connection:
        connection.register("input_rows", frame)
        assert connection.table("input_rows").columns == ["Score", "score_1"]
    with pytest.raises(ValueError, match="unknown field: score_1"):
        module.apply_rows(frame)
    with pytest.raises(
        AutomaticTreeCodegenError,
        match="missing exact required fields: score_1",
    ):
        validate_automatic_tree_duckdb_input_frame(frame, tree)


@pytest.mark.parametrize(
    ("first", "second"),
    [("Score", "score"), ("Straße", "STRASSE")],
)
def test_duckdb_preflight_rejects_casefold_column_collisions(
    first: str,
    second: str,
) -> None:
    tree = _tree(feature=first)
    frame = pd.DataFrame({first: [0.0], second: [1.0]})

    with pytest.raises(AutomaticTreeCodegenError, match="case-insensitive"):
        validate_automatic_tree_duckdb_input_frame(frame, tree)


def test_duckdb_preflight_rejects_duplicate_or_canonicalized_columns() -> None:
    tree = _tree(feature="x")
    duplicate = pd.DataFrame([[0.0, 1.0]], columns=["x", "x"])
    non_text = pd.DataFrame({1: [0.0], "x": [1.0]})

    with pytest.raises(AutomaticTreeCodegenError, match="duplicate columns"):
        validate_automatic_tree_duckdb_input_frame(duplicate, tree)
    with pytest.raises(AutomaticTreeCodegenError, match="must all be text"):
        validate_automatic_tree_duckdb_input_frame(non_text, tree)


@pytest.mark.parametrize(
    "values",
    [
        ["1", "2"],
        ["1_0"],
        [True, False],
        [Decimal("1.0"), Decimal("2.0")],
        [True, 1, "2"],
    ],
)
def test_duckdb_rejects_non_integer_float_physical_types(values: list[object]) -> None:
    tree = _tree(feature="x")
    frame = pd.DataFrame({"x": pd.Series(values, dtype=object)})

    with pytest.raises(duckdb.Error, match="integer/float physical type"):
        _duckdb_results(frame, generate_automatic_tree_duckdb_sql_source(tree))


@pytest.mark.parametrize("dtype", ["string", "bool"])
def test_duckdb_rejects_invalid_physical_type_for_empty_input(dtype: str) -> None:
    tree = _tree(feature="x")
    frame = pd.DataFrame({"x": pd.Series([], dtype=dtype)})

    with pytest.raises(duckdb.Error, match="integer/float physical type"):
        _duckdb_results(frame, generate_automatic_tree_duckdb_sql_source(tree))


@pytest.mark.parametrize("invalid_value", [2**53 + 1, -(2**53) - 1, float("inf")])
def test_duckdb_rejects_lossy_integer_or_infinite_float(
    invalid_value: int | float,
) -> None:
    tree = _tree(feature="x")
    frame = pd.DataFrame({"x": [invalid_value]})

    with pytest.raises(duckdb.Error, match="finite|exact DOUBLE range"):
        _duckdb_results(frame, generate_automatic_tree_duckdb_sql_source(tree))


def test_duckdb_accepts_exact_double_integer_boundaries() -> None:
    tree = _tree(feature="x")
    frame = pd.DataFrame({"x": [-(2**53), 2**53]})
    leaf_ids = apply_weighted_rule_tree(frame, tree).tolist()
    rule_ids = _leaf_rule_pairs(tree)

    assert _duckdb_results(frame, generate_automatic_tree_duckdb_sql_source(tree)) == [
        {
            "__marvis_row_ordinal": index,
            "leaf_id": leaf_id,
            "rule_id": rule_ids[leaf_id],
        }
        for index, leaf_id in enumerate(leaf_ids)
    ]


def test_duckdb_projection_is_exact_case_and_python_preflight_matches(
    tmp_path: Path,
) -> None:
    tree = _tree(feature="Score")
    wrong_case = pd.DataFrame({"score": [1.0]})
    python_module = _load_generated_python(
        generate_automatic_tree_python_source(tree), tmp_path
    )

    with pytest.raises(ValueError, match="unknown field: Score"):
        python_module.apply_rows(wrong_case)
    with pytest.raises(duckdb.Error, match="empty set of columns"):
        _duckdb_results(
            wrong_case,
            generate_automatic_tree_duckdb_sql_source(tree),
        )


def test_duckdb_source_forces_preserved_input_order() -> None:
    tree = _tree(feature="x")
    frame = pd.DataFrame({"x": [4.0, 0.0, 3.0, 1.0]})
    leaf_ids = apply_weighted_rule_tree(frame, tree).tolist()
    rule_ids = _leaf_rule_pairs(tree)

    assert _duckdb_results(
        frame,
        generate_automatic_tree_duckdb_sql_source(tree),
        initial_preserve_insertion_order=False,
    ) == [
        {
            "__marvis_row_ordinal": index,
            "leaf_id": leaf_id,
            "rule_id": rule_ids[leaf_id],
        }
        for index, leaf_id in enumerate(leaf_ids)
    ]


def _rules_for_expression(expression: dict) -> list[dict]:
    canonical = canonicalize_expression(expression)
    inverse = canonicalize_expression({"op": "not", "arg": canonical})
    return [
        {
            "leaf_id": "leaf-match",
            "rule_id": "rule-match",
            "condition": canonical,
        },
        {
            "leaf_id": "leaf-no-match",
            "rule_id": "rule-no-match",
            "condition": inverse,
        },
    ]


def _assert_expression_equivalence(
    expression: dict,
    frame: pd.DataFrame,
    tmp_path: Path,
    *,
    check_sql: bool = True,
) -> None:
    rules = _rules_for_expression(expression)
    expected = []
    for index, row in frame.iterrows():
        matched = evaluate_expression(row, expression)
        suffix = "match" if matched else "no-match"
        expected.append(
            {
                "__marvis_row_ordinal": index,
                "leaf_id": f"leaf-{suffix}",
                "rule_id": f"rule-{suffix}",
            }
        )
    module = _load_generated_python(
        _generate_python_source_from_leaf_rules(rules), tmp_path
    )
    assert module.apply_rows(frame.to_dict(orient="records")) == [
        {"leaf_id": row["leaf_id"], "rule_id": row["rule_id"]} for row in expected
    ]
    if check_sql:
        assert (
            _duckdb_results(frame, _generate_duckdb_sql_from_leaf_rules(rules))
            == expected
        )


@pytest.mark.parametrize(
    ("expression", "frame"),
    [
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "in",
                "value": [1, 2],
                "missing": "no_match",
            },
            pd.DataFrame({"x": [" 1 ", "2", "3e0"]}),
        ),
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "not_in",
                "value": [1, 2],
                "missing": "no_match",
            },
            pd.DataFrame({"x": [" 1 ", "3", "+2"]}),
        ),
        (
            {
                "op": "between",
                "field": "x",
                "lower": 1,
                "upper": 3,
                "include_lower": False,
                "include_upper": True,
                "missing": "no_match",
            },
            pd.DataFrame({"x": [" 1 ", "2", "3"]}),
        ),
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "==",
                "value": 1,
                "missing": "no_match",
            },
            pd.DataFrame({"x": [False, True]}),
        ),
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "==",
                "value": 1,
                "coercion": "strict",
                "missing": "no_match",
            },
            pd.DataFrame({"x": [0, 1, 2]}),
        ),
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "==",
                "value": True,
                "coercion": "strict",
                "missing": "no_match",
            },
            pd.DataFrame({"x": [False, True]}),
        ),
        (
            {
                "op": "compare",
                "field": 'select "中 文"',
                "operator": "==",
                "value": '  中文 "额度"  ',
                "coercion": "strict",
                "missing": "no_match",
            },
            pd.DataFrame({'select "中 文"': ['  中文 "额度"  ', "其他"]}),
        ),
    ],
)
def test_generic_dsl_python_matches_reference(
    expression: dict,
    frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    _assert_expression_equivalence(expression, frame, tmp_path, check_sql=False)


@pytest.mark.parametrize(
    ("expression", "frame"),
    [
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "==",
                "value": 1,
                "missing": "match",
            },
            pd.DataFrame({"x": [None, np.nan, 1.0]}),
        ),
        (
            {"op": "is_null", "field": "x"},
            pd.DataFrame({"x": [None, np.nan, 1.0]}),
        ),
        (
            {"op": "is_not_null", "field": "x"},
            pd.DataFrame({"x": [None, np.nan, 1.0]}),
        ),
    ],
)
def test_missing_and_null_semantics_match_reference(
    expression: dict,
    frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    _assert_expression_equivalence(expression, frame, tmp_path)


def test_invalid_numeric_and_missing_error_fail_in_all_executors(
    tmp_path: Path,
) -> None:
    expressions_and_frames = [
        (
            {
                "op": "compare",
                "field": "x",
                "operator": ">",
                "value": 1,
                "missing": "no_match",
            },
            pd.DataFrame({"x": ["not-a-number"]}),
        ),
        (
            {
                "op": "compare",
                "field": "x",
                "operator": "==",
                "value": 1,
                "missing": "error",
            },
            pd.DataFrame({"x": [np.nan]}),
        ),
    ]
    for expression, frame in expressions_and_frames:
        with pytest.raises(StrategyError):
            evaluate_expression(frame.iloc[0], expression)
        rules = _rules_for_expression(expression)
        module = _load_generated_python(
            _generate_python_source_from_leaf_rules(rules), tmp_path
        )
        with pytest.raises(ValueError):
            module.apply_rows(frame.to_dict(orient="records"))
        with pytest.raises(duckdb.Error):
            _duckdb_results(frame, _generate_duckdb_sql_from_leaf_rules(rules))


def test_bool_number_and_string_strict_types_are_not_conflated(
    tmp_path: Path,
) -> None:
    strict_number = {
        "op": "compare",
        "field": "x",
        "operator": "==",
        "value": 1,
        "coercion": "strict",
        "missing": "no_match",
    }
    strict_bool = {
        **strict_number,
        "value": True,
    }
    strict_string = {
        **strict_number,
        "value": "1",
    }
    for expression, frame in (
        (strict_number, pd.DataFrame({"x": [True, False]})),
        (strict_number, pd.DataFrame({"x": ["1", "2"]})),
        (strict_bool, pd.DataFrame({"x": [1, 0]})),
        (strict_bool, pd.DataFrame({"x": ["true", "false"]})),
        (strict_string, pd.DataFrame({"x": [1, 2]})),
    ):
        _assert_expression_equivalence(expression, frame, tmp_path, check_sql=False)


@pytest.mark.parametrize(
    "expression",
    [
        {
            "op": "and",
            "args": [
                {
                    "op": "compare",
                    "field": "gate",
                    "operator": "==",
                    "value": False,
                    "coercion": "strict",
                    "missing": "no_match",
                },
                {
                    "op": "compare",
                    "field": "bad-number",
                    "operator": ">",
                    "value": 0,
                    "missing": "no_match",
                },
            ],
        },
        {
            "op": "or",
            "args": [
                {
                    "op": "compare",
                    "field": "gate",
                    "operator": "==",
                    "value": True,
                    "coercion": "strict",
                    "missing": "no_match",
                },
                {
                    "op": "compare",
                    "field": "bad-number",
                    "operator": ">",
                    "value": 0,
                    "missing": "no_match",
                },
            ],
        },
        {
            "op": "n_of_k",
            "n": 2,
            "args": [
                {"op": "is_not_null", "field": "first"},
                {"op": "not", "arg": {"op": "is_null", "field": "second"}},
                {
                    "op": "compare",
                    "field": "bad-number",
                    "operator": ">",
                    "value": 0,
                    "missing": "no_match",
                },
            ],
        },
    ],
)
def test_boolean_operators_short_circuit_runtime_errors(
    expression: dict,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "gate": [True],
            "first": [1],
            "second": [2],
            "bad-number": ["invalid"],
        }
    )
    _assert_expression_equivalence(expression, frame, tmp_path, check_sql=False)


@pytest.mark.parametrize(
    "expression",
    [
        {
            "op": "and",
            "args": [
                {
                    "op": "compare",
                    "field": "gate",
                    "operator": "==",
                    "value": 0,
                    "missing": "no_match",
                },
                {
                    "op": "compare",
                    "field": "later",
                    "operator": "==",
                    "value": 1,
                    "missing": "error",
                },
            ],
        },
        {
            "op": "or",
            "args": [
                {
                    "op": "compare",
                    "field": "gate",
                    "operator": "==",
                    "value": 1,
                    "missing": "no_match",
                },
                {
                    "op": "compare",
                    "field": "later",
                    "operator": "==",
                    "value": 1,
                    "missing": "error",
                },
            ],
        },
        {
            "op": "n_of_k",
            "n": 1,
            "args": [
                {
                    "op": "compare",
                    "field": "gate",
                    "operator": "==",
                    "value": 1,
                    "missing": "no_match",
                },
                {
                    "op": "compare",
                    "field": "later",
                    "operator": "==",
                    "value": 1,
                    "missing": "error",
                },
            ],
        },
    ],
)
def test_duckdb_boolean_operators_short_circuit_in_numeric_domain(
    expression: dict,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"gate": [1.0], "later": [np.nan]})

    _assert_expression_equivalence(expression, frame, tmp_path)


@pytest.mark.parametrize("invalid_value", [float("inf"), 2**53 + 1])
def test_duckdb_prevalidates_unreachable_numeric_feature(
    invalid_value: int | float,
) -> None:
    expression = {
        "op": "or",
        "args": [
            {
                "op": "compare",
                "field": "gate",
                "operator": "==",
                "value": 1,
                "missing": "no_match",
            },
            {
                "op": "compare",
                "field": "later",
                "operator": "==",
                "value": 1,
                "missing": "no_match",
            },
        ],
    }
    frame = pd.DataFrame({"gate": [1], "later": [invalid_value]})

    with pytest.raises(duckdb.Error, match="finite|exact DOUBLE range"):
        _duckdb_results(
            frame,
            _generate_duckdb_sql_from_leaf_rules(_rules_for_expression(expression)),
        )


def test_all_fields_are_prevalidated_before_short_circuit(tmp_path: Path) -> None:
    expression = {
        "op": "or",
        "args": [
            {"op": "is_not_null", "field": "present"},
            {"op": "is_not_null", "field": "unreachable_missing"},
        ],
    }
    rules = _rules_for_expression(expression)
    frame = pd.DataFrame({"present": [1]})

    with pytest.raises(StrategyError, match="unknown field: unreachable_missing"):
        evaluate_expression(frame.iloc[0], expression)
    module = _load_generated_python(
        _generate_python_source_from_leaf_rules(rules), tmp_path
    )
    with pytest.raises(ValueError, match="unknown field: unreachable_missing"):
        module.apply_rows(frame.to_dict(orient="records"))
    with pytest.raises(ValueError, match="unknown field: unreachable_missing"):
        module.apply_rows(frame.iloc[0:0])
    with pytest.raises(ValueError, match="unknown field: unreachable_missing"):
        module.apply_rows(
            [
                {"present": 1, "unreachable_missing": 2},
                {"present": 1},
            ]
        )
    with pytest.raises(duckdb.Error, match="unreachable_missing"):
        _duckdb_results(frame, _generate_duckdb_sql_from_leaf_rules(rules))


def test_generated_sources_treat_identifiers_and_values_as_data(
    tmp_path: Path,
) -> None:
    field = "x\" FROM input_rows; DROP TABLE input_rows; --\n'''"
    expression = {
        "op": "compare",
        "field": field,
        "operator": "==",
        "value": 1,
        "missing": "no_match",
    }
    sql = _generate_duckdb_sql_from_leaf_rules(_rules_for_expression(expression))
    frame = pd.DataFrame({field: [1, 2]})
    expected_identities = [
        {"leaf_id": "leaf-match", "rule_id": "rule-match"},
        {"leaf_id": "leaf-no-match", "rule_id": "rule-no-match"},
    ]

    module = _load_generated_python(
        _generate_python_source_from_leaf_rules(_rules_for_expression(expression)),
        tmp_path,
    )
    assert module.apply_rows(frame.to_dict(orient="records")) == expected_identities
    assert _duckdb_results(frame, sql) == [
        {
            "__marvis_row_ordinal": 0,
            **expected_identities[0],
        },
        {
            "__marvis_row_ordinal": 1,
            **expected_identities[1],
        },
    ]

from __future__ import annotations

import ast
import math

import pandas as pd
import pytest

from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    extract_safe_transformations,
    required_transformation_inputs,
    topologically_sorted_transformations,
    validate_transformation_plan,
)
from marvis.validation.input_contracts import TransformationSpec


def _spec(operation, output, inputs=(), params=None):
    return TransformationSpec(
        operation=operation,
        output_field=output,
        input_fields=tuple(inputs),
        params={} if params is None else params,
    )


def test_extracts_direct_and_astype_date_to_month_and_typed_constant_mapping():
    tree = ast.parse(
        "df['apply_month'] = df['ober_date'].str[:7]\n"
        "df['book_month'] = df['book_date'].astype(str).str[:7]\n"
        "df['model_flag'] = df['source_tag'].map({0: 'train', 1: 'oot'})\n"
    )

    specs = extract_safe_transformations(tree, cell_index=2)

    assert [(row.operation, row.output_field) for row in specs] == [
        ("date_to_month", "apply_month"),
        ("date_to_month", "book_month"),
        ("constant_mapping", "model_flag"),
    ]
    assert specs[0].params == {"mode": "direct_string_slice"}
    assert specs[1].params == {"mode": "astype_string_slice"}
    assert specs[2].params == {
        "mapping": [
            {"source": 0, "target": "train"},
            {"source": 1, "target": "oot"},
        ]
    }


def test_extracts_to_datetime_period_month_exact_shape():
    specs = extract_safe_transformations(
        ast.parse(
            "df['month'] = pd.to_datetime(df['apply_dt']).dt.to_period('M').astype(str)"
        ),
        cell_index=0,
    )

    assert specs == (
        _spec(
            "date_to_month",
            "month",
            ("apply_dt",),
            {"mode": "datetime_period"},
        ),
    )


def test_extracts_copy_assigned_rename_and_scalar_source_label():
    specs = extract_safe_transformations(
        ast.parse(
            "df['copied'] = df['raw']\n"
            "df = df.rename(columns={'old': 'renamed'})\n"
            "df['source'] = 'development'\n"
        ),
        cell_index=0,
    )

    assert specs == (
        _spec("copy", "copied", ("raw",)),
        _spec("rename", "renamed", ("old",)),
        _spec("constant_source_label", "source", params={"value": "development"}),
    )


def test_assigned_rename_requires_disjoint_unique_source_and_output_names():
    for source in (
        "df = df.rename(columns={'a': 'x', 'a': 'y'})",
        "df = df.rename(columns={'a': 'b', 'b': 'c'})",
        "df = df.rename(columns={'a': 'a'})",
    ):
        assert extract_safe_transformations(ast.parse(source), cell_index=0) == ()

    assert extract_safe_transformations(
        ast.parse("df = df.rename(columns={'a': 'x', 'b': 'y'})"), cell_index=0
    ) == (
        _spec("rename", "x", ("a",)),
        _spec("rename", "y", ("b",)),
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            "df['split'] = np.where(df['age'] >= 18, 'train', 'oot')",
            _spec(
                "constant_threshold",
                "split",
                ("age",),
                {
                    "operator": "ge",
                    "threshold": 18,
                    "true_value": "train",
                    "false_value": "oot",
                },
            ),
        ),
        (
            "df['split'] = df['age'].apply(lambda x: 'train' if x < 60 else 'oot')",
            _spec(
                "constant_threshold",
                "split",
                ("age",),
                {
                    "operator": "lt",
                    "threshold": 60,
                    "true_value": "train",
                    "false_value": "oot",
                },
            ),
        ),
    ],
)
def test_extracts_exact_constant_threshold_shapes(expression, expected):
    assert extract_safe_transformations(ast.parse(expression), cell_index=4) == (
        expected,
    )


@pytest.mark.parametrize(
    "source",
    [
        "df['x'] = private_package.make_feature(df)",
        "df['x'] = other['raw']",
        "df['x'] = df['raw'].apply(lambda x: choose(x))",
        "df['x'] = df['raw'].apply(lambda x: 'a' if x < cutoff else 'b')",
        "df['x'] = df['raw'].apply(lambda x: ('a', side_effect())[0])",
        "df['x'] = np.where(df['raw'] < threshold, 'a', 'b')",
        "df.rename(columns={'a': 'b'}, inplace=True)",
        "other = df.rename(columns={'a': 'b'})",
        "df = df.rename(columns={'a': 'same', 'b': 'same'})",
        "df['x'] = df[key]",
    ],
)
def test_ignores_dynamic_arbitrary_or_cross_frame_shapes(source):
    assert extract_safe_transformations(ast.parse(source), cell_index=0) == ()


def test_ignores_non_integer_slice_and_empty_mapping_shapes():
    assert extract_safe_transformations(
        ast.parse("df['month'] = df['date'].str[:7.0]"), cell_index=0
    ) == ()
    assert extract_safe_transformations(
        ast.parse("df['split'] = df['source'].map({})"), cell_index=0
    ) == ()


def test_rejects_safe_transformations_from_multiple_dataframe_roots():
    assert extract_safe_transformations(
        ast.parse(
            "left['month'] = left['date'].str[:7]\n"
            "right['split'] = 'oot'\n"
        ),
        cell_index=0,
    ) == ()


def test_topology_and_required_raw_inputs_preserve_stable_source_order():
    specs = (
        _spec("copy", "second", ("first",)),
        _spec("copy", "first", ("raw_b",)),
        _spec("copy", "other", ("raw_a",)),
    )

    ordered = topologically_sorted_transformations(specs)
    required = required_transformation_inputs(("second", "other", "second"), specs)

    assert [spec.output_field for spec in ordered] == ["first", "second", "other"]
    assert required == ("raw_b", "raw_a")


def test_validate_plan_rejects_duplicates_cycles_raw_overwrite_and_missing_raw_inputs():
    with pytest.raises(ValueError, match="duplicate transformation output"):
        validate_transformation_plan(
            (_spec("copy", "x", ("a",)), _spec("copy", "x", ("b",))),
            sample_columns=("a", "b"),
        )
    with pytest.raises(ValueError, match="cycle"):
        validate_transformation_plan(
            (_spec("copy", "a", ("b",)), _spec("copy", "b", ("a",))),
            sample_columns=("raw",),
        )
    with pytest.raises(ValueError, match="overwrites raw sample field"):
        validate_transformation_plan(
            (_spec("copy", "already_there", ("raw",)),),
            sample_columns=("raw", "already_there"),
        )
    with pytest.raises(ValueError, match="missing raw inputs: missing"):
        validate_transformation_plan(
            (_spec("copy", "derived", ("missing",)),),
            sample_columns=("raw",),
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (_spec("copy", "x", ()), "exactly one input"),
        (_spec("copy", "x", ("a",), {"extra": 1}), "parameter keys"),
        (
            _spec("date_to_month", "x", ("a",), {"mode": "unknown"}),
            "date_to_month mode",
        ),
        (
            _spec(
                "constant_threshold",
                "x",
                ("a",),
                {
                    "operator": "lt",
                    "threshold": math.inf,
                    "true_value": "a",
                    "false_value": "b",
                },
            ),
            "finite",
        ),
        (
            _spec(
                "constant_mapping",
                "x",
                ("a",),
                {"mapping": [{"source": [], "target": "bad"}]},
            ),
            "JSON scalar",
        ),
        (
            _spec("constant_source_label", "x", ("a",), {"value": "source"}),
            "no inputs",
        ),
    ],
)
def test_validate_plan_enforces_exact_arity_param_keys_types_and_finite_json(
    spec, message
):
    with pytest.raises(ValueError, match=message):
        validate_transformation_plan((spec,), sample_columns=("a",))


def test_numeric_mapping_round_trip_preserves_types_and_rejects_unmapped_values():
    spec = _spec(
        "constant_mapping",
        "split",
        ("source",),
        {
            "mapping": [
                {"source": 0, "target": "train"},
                {"source": 1.0, "target": "test"},
                {"source": "1", "target": "oot"},
            ]
        },
    )
    frame = pd.DataFrame({"source": pd.Series([0, 1.0, "1"], dtype="object")})

    result = apply_confirmed_transformations(frame, (spec,))

    assert result["split"].tolist() == ["train", "test", "oot"]
    assert frame.columns.tolist() == ["source"]
    with pytest.raises(ValueError, match="unmapped values"):
        apply_confirmed_transformations(
            pd.DataFrame({"source": pd.Series([2], dtype="object")}), (spec,)
        )


def test_mapping_runtime_matches_python_and_pandas_scalar_key_equality():
    pairs = [
        {"source": 0, "target": "zero"},
        {"source": 1, "target": "one"},
        {"source": "1", "target": "string-one"},
        {"source": None, "target": "missing"},
    ]
    spec = _spec(
        "constant_mapping", "mapped", ("source",), {"mapping": pairs}
    )
    source = pd.Series([0.0, 1.0, False, True, "1", None], dtype="object")
    expected = source.map({
        pair["source"]: pair["target"] for pair in pairs
    })

    result = apply_confirmed_transformations(
        pd.DataFrame({"source": source}), (spec,)
    )["mapped"]

    assert result.tolist() == expected.tolist()
    assert result.iloc[0] == "zero"
    assert result.iloc[2] == "zero"


def test_mapping_rejects_python_equal_duplicate_sources():
    spec = _spec(
        "constant_mapping",
        "mapped",
        ("source",),
        {
            "mapping": [
                {"source": 1, "target": "integer"},
                {"source": True, "target": "boolean"},
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate.*source"):
        validate_transformation_plan((spec,), sample_columns=("source",))


def test_mapping_rejects_duplicate_typed_sources_without_python_key_coercion():
    duplicated = _spec(
        "constant_mapping",
        "split",
        ("source",),
        {
            "mapping": [
                {"source": 1, "target": "train"},
                {"source": 1, "target": "oot"},
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate typed source"):
        validate_transformation_plan((duplicated,), sample_columns=("source",))


def test_mapping_targets_preserve_json_scalar_types():
    spec = _spec(
        "constant_mapping",
        "mapped",
        ("source",),
        {
            "mapping": [
                {"source": "integer", "target": 1},
                {"source": "float", "target": 1.0},
            ]
        },
    )

    result = apply_confirmed_transformations(
        pd.DataFrame({"source": ["integer", "float"]}), (spec,)
    )

    assert type(result["mapped"].iloc[0]) is int
    assert type(result["mapped"].iloc[1]) is float


def test_date_modes_are_deterministic_and_reproduce_the_recognized_expressions():
    frame = pd.DataFrame(
        {
            "direct": pd.Series(["2024/01/31", "2024-02-20"], dtype="string"),
            "mixed": [20240131, None],
            "parseable": ["2024-03-31", "2024-04-01"],
        }
    )
    specs = (
        _spec(
            "date_to_month",
            "direct_month",
            ("direct",),
            {"mode": "direct_string_slice"},
        ),
        _spec(
            "date_to_month",
            "mixed_month",
            ("mixed",),
            {"mode": "astype_string_slice"},
        ),
        _spec(
            "date_to_month",
            "period_month",
            ("parseable",),
            {"mode": "datetime_period"},
        ),
    )

    result = apply_confirmed_transformations(frame, specs)

    assert result["direct_month"].equals(frame["direct"].str[:7])
    assert result["mixed_month"].equals(frame["mixed"].astype(str).str[:7])
    expected_period = (
        pd.to_datetime(frame["parseable"]).dt.to_period("M").astype(str)
    )
    assert result["period_month"].equals(expected_period)


def test_lambda_threshold_materializes_mixed_scalar_outputs_as_object_values():
    specs = extract_safe_transformations(
        ast.parse("df['group'] = df['x'].apply(lambda x: 1 if x < 1 else 'high')"),
        cell_index=0,
    )

    result = apply_confirmed_transformations(pd.DataFrame({"x": [0, 2]}), specs)

    assert result["group"].tolist() == [1, "high"]
    assert type(result["group"].iloc[0]) is int
    assert type(result["group"].iloc[1]) is str


def test_apply_is_repeatable_and_preserves_equal_control_and_model_series_bytes():
    frame = pd.DataFrame(
        {
            "target_raw": [0, 1, 0],
            "source_raw": [0, 1, 2],
            "date_raw": ["2024-01-02", "2024-02-03", "2024-03-04"],
            "feature_raw": [1.5, 2.5, 3.5],
        }
    )
    specs = (
        _spec("copy", "target", ("target_raw",)),
        _spec(
            "constant_mapping",
            "split",
            ("source_raw",),
            {
                "mapping": [
                    {"source": 0, "target": "train"},
                    {"source": 1, "target": "test"},
                    {"source": 2, "target": "oot"},
                ]
            },
        ),
        _spec(
            "date_to_month",
            "month",
            ("date_raw",),
            {"mode": "astype_string_slice"},
        ),
        _spec("rename", "feature", ("feature_raw",)),
    )

    scoring = apply_confirmed_transformations(frame, specs)
    metrics = apply_confirmed_transformations(frame, specs)
    stress = apply_confirmed_transformations(frame, specs)
    selected = ["target", "split", "month", "feature"]

    assert scoring[selected].to_csv(index=False).encode() == metrics[selected].to_csv(
        index=False
    ).encode()
    assert scoring[selected].to_csv(index=False).encode() == stress[selected].to_csv(
        index=False
    ).encode()


def test_already_materialized_output_is_rejected_instead_of_overwritten():
    frame = pd.DataFrame({"raw": [1], "derived": [999]})
    spec = _spec("copy", "derived", ("raw",))

    with pytest.raises(ValueError, match="overwrites raw sample field"):
        validate_transformation_plan((spec,), sample_columns=frame.columns)
    with pytest.raises(ValueError, match="overwrites raw sample field"):
        apply_confirmed_transformations(frame, (spec,))
    assert frame["derived"].tolist() == [999]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (_spec(["copy"], "out", ("raw",)), "operation must be a string"),
        (
            _spec("date_to_month", "out", ("raw",), {"mode": ["bad"]}),
            "mode must be a string",
        ),
        (
            _spec(
                "constant_threshold",
                "out",
                ("raw",),
                {
                    "operator": ["lt"],
                    "threshold": 1,
                    "true_value": "a",
                    "false_value": "b",
                },
            ),
            "operator must be a string",
        ),
    ],
)
def test_malformed_runtime_specs_raise_controlled_value_errors(spec, message):
    with pytest.raises(ValueError, match=message):
        validate_transformation_plan((spec,), sample_columns=("raw",))


def test_deep_transformation_chain_is_iterative_and_stable_in_both_input_orders():
    expected_outputs = tuple(f"derived_{index}" for index in range(1_500))
    forward = tuple(
        _spec(
            "copy",
            output,
            ("raw" if index == 0 else expected_outputs[index - 1],),
        )
        for index, output in enumerate(expected_outputs)
    )

    for specs in (forward, tuple(reversed(forward))):
        ordered = topologically_sorted_transformations(specs)
        assert tuple(spec.output_field for spec in ordered) == expected_outputs
        assert required_transformation_inputs((expected_outputs[-1],), specs) == (
            "raw",
        )


def test_transformation_count_limit_is_controlled(monkeypatch):
    from marvis.validation import field_transformations

    monkeypatch.setattr(
        field_transformations, "MAX_TRANSFORMATIONS", 2, raising=False
    )
    specs = (
        _spec("copy", "a", ("raw",)),
        _spec("copy", "b", ("a",)),
        _spec("copy", "c", ("b",)),
    )

    with pytest.raises(ValueError, match="transformation count limit"):
        topologically_sorted_transformations(specs)

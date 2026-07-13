from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from marvis.files import sha256_file
from marvis.validation import input_confirmation
from marvis.validation.checks import validate_required_splits
from marvis.validation.config import ValidationConfig
from marvis.validation.field_transformations import apply_confirmed_transformations
from marvis.validation.input_confirmation import (
    inspect_confirmation_values,
    json_scalar_identity,
    validate_binary_labels,
    validate_confirmation_against_materials,
    validate_split_mapping,
)
from marvis.validation.input_contracts import (
    FieldCandidate,
    FieldEvidence,
    TransformationSpec,
)
from marvis.validation.sample_schema import inspect_sample_schema
from tests.validation_builders import (
    make_candidate_contract,
    make_validation_confirmation,
)
from tests.validation_material_builders import write_validation_material_bundle


def _metadata_candidate() -> FieldCandidate:
    return FieldCandidate(
        value={
            "metadata_sheet": None,
            "feature_col": "feature",
            "category_col": "category",
            "importance_col": "importance",
        },
        evidence=(FieldEvidence("feature_metadata", None, "metadata.csv", 1.0),),
    )


def _contract_for_bundle(bundle, *, transformations=(), sample_schema=None):
    contract = make_candidate_contract(
        material_hashes={
            "notebook": sha256_file(bundle.notebook_path),
            "sample": sha256_file(bundle.sample_path),
            "pmml": sha256_file(bundle.pmml_path),
            "dictionary": sha256_file(bundle.dictionary_path),
        }
    )
    return replace(
        contract,
        sample_schema=sample_schema or inspect_sample_schema(bundle.sample_path),
        candidates={
            **contract.candidates,
            "feature_metadata_selection": (_metadata_candidate(),),
        },
        transformations=tuple(transformations),
    )


def _request(**changes):
    request = replace(make_validation_confirmation(), metadata_sheet=None)
    return replace(request, **changes)


def _spec(operation, output, inputs=(), params=None):
    return TransformationSpec(
        operation=operation,
        output_field=output,
        input_fields=tuple(inputs),
        params={} if params is None else params,
    )


def test_material_builder_writes_real_four_materials_without_notebook_execution(
    tmp_path: Path,
) -> None:
    source = "RMC_TARGET_COL = 'y'"
    bundle = write_validation_material_bundle(
        tmp_path / "bundle", notebook_source=source
    )

    notebook = nbformat.read(bundle.notebook_path, as_version=4)

    assert notebook.cells[0].source == source
    assert pd.read_parquet(bundle.sample_path).shape == (4, 5)
    assert bundle.pmml_path.read_bytes() == (
        Path(__file__).parents[1] / "fixtures" / "min_lr.pmml"
    ).read_bytes()
    assert list(pd.read_csv(bundle.dictionary_path).columns) == [
        "feature",
        "category",
        "importance",
    ]


def test_confirmation_reads_real_control_columns_and_complete_metadata(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(
        tmp_path / "bundle", notebook_source="RMC_TARGET_COL='y'"
    )
    contract = _contract_for_bundle(bundle)

    validated = validate_confirmation_against_materials(
        contract=contract,
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(),
    )

    assert validated.sample_schema is contract.sample_schema
    assert validated.sample_schema.row_count == 4
    assert validated.feature_metadata.coverage.importance == 1.0
    assert validated.values.negative_label == 0


def test_confirmation_never_projects_all_pmml_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = write_validation_material_bundle(
        tmp_path / "bundle", notebook_source="RMC_TARGET_COL='y'"
    )
    contract = _contract_for_bundle(bundle)
    original = input_confirmation.iter_sample_projection
    projections: list[tuple[str, ...]] = []

    def capture_projection(path, *, columns, chunk_size, schema):
        projections.append(columns)
        yield from original(
            path, columns=columns, chunk_size=chunk_size, schema=schema
        )

    monkeypatch.setattr(
        input_confirmation, "iter_sample_projection", capture_projection
    )

    validate_confirmation_against_materials(
        contract=contract,
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(),
    )

    assert projections == [("y", "split", "apply_month")]
    assert "x1" not in projections[0]
    assert "x2" not in projections[0]


def test_confirmation_preserves_numeric_split_values(tmp_path: Path) -> None:
    sample = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": [0, 1, 0, 1],
            "split": [0, 1, 2, 2],
            "apply_month": ["202601", "202602", "202603", "202603"],
        }
    )
    bundle = write_validation_material_bundle(
        tmp_path / "numeric", notebook_source="", sample=sample
    )

    validated = validate_confirmation_against_materials(
        contract=_contract_for_bundle(bundle),
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(
            split_value_mapping={"train": 0, "test": 1, "oot": 2}
        ),
    )

    assert validated.values.split_value_mapping == {"train": 0, "test": 1, "oot": 2}
    assert type(validated.values.split_value_mapping["oot"]) is int


def test_negative_label_none_derives_and_persists_other_observed_value(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(tmp_path / "bundle", notebook_source="")

    validated = validate_confirmation_against_materials(
        contract=_contract_for_bundle(bundle),
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(negative_label=None),
    )

    assert validated.values.negative_label == 0
    assert type(validated.values.negative_label) is int


def test_unobserved_positive_and_nonbinary_target_are_blocking(tmp_path: Path) -> None:
    bundle = write_validation_material_bundle(tmp_path / "labels", notebook_source="")
    contract = _contract_for_bundle(bundle)

    with pytest.raises(ValueError, match="positive label"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(positive_label=99),
        )

    sample = pd.read_parquet(bundle.sample_path)
    sample["y"] = [0, 1, 2, 1]
    sample.to_parquet(bundle.sample_path, index=False)
    changed_contract = _contract_for_bundle(bundle)
    with pytest.raises(ValueError, match="exactly two"):
        validate_confirmation_against_materials(
            contract=changed_contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )


@pytest.mark.parametrize(
    ("bad_values", "message"),
    [
        ([0, 1, None, 1], "target contains null"),
        ([0.0, 1.0, np.inf, 1.0], "not finite"),
        ([[0], [1], [0], [1]], "JSON scalar"),
    ],
)
def test_target_null_nonfinite_and_nonjson_values_block(
    tmp_path: Path, bad_values, message: str
) -> None:
    sample = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": bad_values,
            "split": ["train", "test", "oot", "oot"],
            "apply_month": ["202601", "202602", "202603", "202603"],
        }
    )
    bundle = write_validation_material_bundle(
        tmp_path / "bad-target", notebook_source="", sample=sample
    )

    with pytest.raises(ValueError, match=message):
        validate_confirmation_against_materials(
            contract=_contract_for_bundle(bundle),
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )


def test_json_scalar_and_binary_labels_use_type_stable_identity() -> None:
    identities = {
        json_scalar_identity(value) for value in (1, 1.0, True, "1")
    }

    assert len(identities) == 4
    assert validate_binary_labels((1, 1.0), positive=1, negative=1.0) == 1.0
    assert input_confirmation.normalize_binary_target(
        pd.Series([1, 1.0], dtype="object"),
        positive=1,
        negative=1.0,
    ).tolist() == [1, 0]
    with pytest.raises(ValueError, match="positive label"):
        validate_binary_labels((1, 1.0), positive=True, negative=1.0)


def test_binary_labels_accept_unambiguous_integer_float_json_round_trip() -> None:
    observed = (0.0, 1.0)

    assert validate_binary_labels(observed, positive=1, negative=0) == 0.0
    normalized = input_confirmation.normalize_binary_target(
        pd.Series([0.0, 1.0, 1.0, 0.0]),
        positive=1,
        negative=0,
    )

    assert normalized.tolist() == [0, 1, 1, 0]


def test_binary_labels_do_not_coerce_strings_or_booleans_to_numbers() -> None:
    with pytest.raises(ValueError, match="positive label"):
        validate_binary_labels((0.0, 1.0), positive="1", negative=0)
    with pytest.raises(ValueError, match="positive label"):
        validate_binary_labels((0.0, 1.0), positive=True, negative=0)


def test_split_mapping_requires_exact_keys_typed_unique_values_and_full_coverage() -> None:
    observed = (1, 1.0, True)
    mapping = {"train": 1, "test": 1.0, "oot": True}

    assert validate_split_mapping(observed, mapping) == mapping
    with pytest.raises(ValueError, match="keys"):
        validate_split_mapping(observed, {"train": 1, "test": 1.0})
    with pytest.raises(ValueError, match="keys"):
        validate_split_mapping(
            observed, {**mapping, "other": "x"}
        )
    with pytest.raises(ValueError, match="type-stably unique"):
        validate_split_mapping(
            (1, 2), {"train": 1, "test": 1, "oot": 2}
        )
    with pytest.raises(ValueError, match="unmapped"):
        validate_split_mapping(
            (1, 2, 3, 4), {"train": 1, "test": 2, "oot": 3}
        )
    with pytest.raises(ValueError, match="no rows"):
        validate_split_mapping(
            (1, 2), {"train": 1, "test": 2, "oot": 3}
        )


def test_time_granularity_null_and_late_chunk_values_are_fully_validated(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(tmp_path / "time", notebook_source="")
    contract = _contract_for_bundle(bundle)

    with pytest.raises(ValueError, match="time_granularity"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(time_granularity="quarter"),
        )

    sample = pd.read_parquet(bundle.sample_path)
    sample["apply_month"] = ["202601", "202602", "202603", None]
    sample.to_parquet(bundle.sample_path, index=False)
    schema = inspect_sample_schema(bundle.sample_path)
    with pytest.raises(ValueError, match="time column"):
        inspect_confirmation_values(
            bundle.sample_path,
            columns=("y", "split", "apply_month"),
            transformations=(),
            target_col="y",
            split_col="split",
            time_col="apply_month",
            time_granularity="month",
            sample_schema=schema,
            chunk_size=2,
        )


def test_date_granularity_is_validated_and_preserved(tmp_path: Path) -> None:
    sample = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": [0, 1, 0, 1],
            "split": ["train", "test", "oot", "oot"],
            "apply_date": ["2026-01-01", "2026-02-02", "20260303", "20260304"],
        }
    )
    bundle = write_validation_material_bundle(
        tmp_path / "date-granularity", notebook_source="", sample=sample
    )

    validated = validate_confirmation_against_materials(
        contract=_contract_for_bundle(bundle),
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(time_col="apply_date", time_granularity="date"),
    )

    assert validated.values.time_granularity == "date"


def test_time_validation_errors_are_bounded_before_rendering_values(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(
        tmp_path / "long-time", notebook_source=""
    )
    sample = pd.read_parquet(bundle.sample_path)
    sample["apply_month"] = ["202601", "202602", "202603", "x" * 20_000]
    sample.to_parquet(bundle.sample_path, index=False)

    with pytest.raises(ValueError) as exc_info:
        validate_confirmation_against_materials(
            contract=_contract_for_bundle(bundle),
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )

    assert "scalar length limit" in str(exc_info.value)
    assert len(str(exc_info.value)) <= input_confirmation.MAX_CONFIRMATION_ERROR_CHARS


def test_empty_sample_and_schema_row_count_mismatch_block(tmp_path: Path) -> None:
    bundle = write_validation_material_bundle(tmp_path / "rows", notebook_source="")
    schema = inspect_sample_schema(bundle.sample_path)
    mismatched = replace(schema, row_count=5)
    with pytest.raises(ValueError, match="row count"):
        validate_confirmation_against_materials(
            contract=_contract_for_bundle(bundle, sample_schema=mismatched),
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )

    empty = pd.read_parquet(bundle.sample_path).iloc[0:0]
    empty.to_parquet(bundle.sample_path, index=False)
    with pytest.raises(ValueError, match="no rows"):
        validate_confirmation_against_materials(
            contract=_contract_for_bundle(bundle),
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )


def test_requested_transformations_must_be_exact_scanned_subset(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(tmp_path / "transform", notebook_source="")
    scanned = _spec("copy", "derived", ("y",))
    injected = _spec("copy", "other", ("y",))
    contract = _contract_for_bundle(bundle, transformations=(scanned,))

    with pytest.raises(ValueError, match="scanned transformation subset"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(transformations=(injected,)),
        )


def test_deep_transformation_params_raise_value_error_not_recursion_error(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(
        tmp_path / "deep-transform", notebook_source=""
    )
    nested: object = "leaf"
    for _ in range(70):
        nested = {"nested": nested}
    deep = _spec("copy", "derived", ("y",), {"nested": nested})

    with pytest.raises(ValueError, match="maximum depth"):
        validate_confirmation_against_materials(
            contract=_contract_for_bundle(bundle, transformations=(deep,)),
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(transformations=(deep,)),
        )


def test_derived_controls_apply_only_dependency_closure_and_not_unused_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "raw_y": [0, 1, 0, 1],
            "raw_split": [0, 1, 2, 2],
            "raw_date": ["202601", "202602", "202603", "202603"],
        }
    )
    bundle = write_validation_material_bundle(
        tmp_path / "derived", notebook_source="", sample=sample
    )
    target = _spec("copy", "derived_y", ("raw_y",))
    split = _spec(
        "constant_mapping",
        "derived_split",
        ("raw_split",),
        {
            "mapping": [
                {"source": 0, "target": "train"},
                {"source": 1, "target": "test"},
                {"source": 2, "target": "oot"},
            ]
        },
    )
    month = _spec(
        "date_to_month",
        "derived_month",
        ("raw_date",),
        {"mode": "direct_string_slice"},
    )
    unused = _spec("copy", "unused", ("x1",))
    contract = _contract_for_bundle(
        bundle, transformations=(target, split, month, unused)
    )
    original = input_confirmation.iter_sample_projection
    projections: list[tuple[str, ...]] = []

    def capture(path, *, columns, chunk_size, schema):
        projections.append(columns)
        yield from original(path, columns=columns, chunk_size=chunk_size, schema=schema)

    monkeypatch.setattr(input_confirmation, "iter_sample_projection", capture)

    validated = validate_confirmation_against_materials(
        contract=contract,
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(
            target_col="derived_y",
            split_col="derived_split",
            time_col="derived_month",
            split_value_mapping={
                "train": "train",
                "test": "test",
                "oot": "oot",
            },
            transformations=(target, split, month),
        ),
    )

    assert validated.values.transformations == (target, split, month)
    assert projections == [("raw_y", "raw_split", "raw_date")]
    assert "x1" not in projections[0]
    assert "x2" not in projections[0]


def test_transformed_pmml_input_is_static_only_not_confirmation_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = pd.DataFrame(
        {
            "raw_x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": [0, 1, 0, 1],
            "split": ["train", "test", "oot", "oot"],
            "apply_month": ["202601", "202602", "202603", "202603"],
        }
    )
    bundle = write_validation_material_bundle(
        tmp_path / "pmml-transform", notebook_source="", sample=sample
    )
    create_x1 = _spec("copy", "x1", ("raw_x1",))
    contract = _contract_for_bundle(bundle, transformations=(create_x1,))
    original = input_confirmation.iter_sample_projection
    projections: list[tuple[str, ...]] = []

    def capture(path, *, columns, chunk_size, schema):
        projections.append(columns)
        yield from original(path, columns=columns, chunk_size=chunk_size, schema=schema)

    monkeypatch.setattr(input_confirmation, "iter_sample_projection", capture)

    validate_confirmation_against_materials(
        contract=contract,
        sample_path=bundle.sample_path,
        dictionary_path=bundle.dictionary_path,
        requested=_request(transformations=(create_x1,)),
    )

    assert projections == [("y", "split", "apply_month")]


def test_missing_pmml_raw_input_and_reused_control_columns_block(tmp_path: Path) -> None:
    sample = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "y": [0, 1, 0, 1],
            "split": ["train", "test", "oot", "oot"],
            "apply_month": ["202601", "202602", "202603", "202603"],
        }
    )
    bundle = write_validation_material_bundle(
        tmp_path / "missing", notebook_source="", sample=sample
    )
    contract = _contract_for_bundle(bundle)

    with pytest.raises(ValueError, match="missing.*x2"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )
    with pytest.raises(ValueError, match="control columns must be distinct"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(split_col="y"),
        )


def test_metadata_selection_is_atomic_and_errors_have_material_prefix(
    tmp_path: Path,
) -> None:
    bundle = write_validation_material_bundle(tmp_path / "metadata", notebook_source="")
    contract = _contract_for_bundle(bundle)

    with pytest.raises(ValueError, match="feature metadata:.*scanned candidate"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(category_col="source"),
        )

    pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, None],
        }
    ).to_csv(bundle.dictionary_path, index=False)
    with pytest.raises(ValueError, match="feature metadata:.*importance"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(),
        )


def test_invalid_pmml_output_and_model_params_block(tmp_path: Path) -> None:
    bundle = write_validation_material_bundle(tmp_path / "output", notebook_source="")
    contract = _contract_for_bundle(bundle)

    with pytest.raises(ValueError, match="PMML output"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(pmml_output_field="missing"),
        )
    with pytest.raises(ValueError, match="model_params"):
        validate_confirmation_against_materials(
            contract=contract,
            sample_path=bundle.sample_path,
            dictionary_path=bundle.dictionary_path,
            requested=_request(model_params={"bad": np.inf}),
        )


def test_validation_config_and_required_split_checks_preserve_typed_scalars() -> None:
    config = ValidationConfig(
        target_col="y",
        score_col="score",
        split_col="split",
        time_col="apply_month",
        split_values={"train": 1, "test": 1.0, "oot": True},
    )
    assert config.split_values == {"train": 1, "test": 1.0, "oot": True}
    assert type(config.split_values["test"]) is float

    typed = pd.DataFrame(
        {"split": pd.Series([1, 1.0, True], dtype="object")}
    )
    validate_required_splits(
        typed, split_col="split", split_values=config.split_values
    )
    with pytest.raises(ValueError, match="test, oot"):
        validate_required_splits(
            pd.DataFrame({"split": pd.Series([1], dtype="object")}),
            split_col="split",
            split_values=config.split_values,
        )


def test_apply_confirmed_transformations_remains_legacy_compatible() -> None:
    frame = pd.DataFrame({"raw": [1, 2]})
    spec = _spec("copy", "derived", ("raw",))

    assert apply_confirmed_transformations(frame, (spec,))["derived"].tolist() == [1, 2]

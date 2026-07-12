from __future__ import annotations

from dataclasses import replace
import math

import pytest

from marvis.validation.input_contracts import (
    FieldCandidate,
    FieldEvidence,
    TransformationSpec,
    ValidationInputContract,
    input_contract_from_dict,
    input_contract_to_dict,
    transformation_spec_from_dict,
)
from tests.validation_builders import make_candidate_contract, make_ready_contract


def test_input_contract_round_trip_preserves_all_nested_types_and_tuples():
    contract = make_ready_contract()

    restored = input_contract_from_dict(input_contract_to_dict(contract))

    assert restored == contract
    assert isinstance(restored.candidates["target_col"], tuple)
    assert isinstance(restored.candidates["target_col"][0].evidence, tuple)
    assert isinstance(restored.pmml_manifest.stress_units, tuple)
    assert isinstance(restored.feature_metadata.rows, tuple)
    assert isinstance(restored.feature_metadata.per_category_raw_fields["内部"], tuple)


def test_input_contract_minimal_for_test_and_require_helpers():
    target = FieldCandidate(
        value="y",
        evidence=(FieldEvidence("rmc_literal", 4, "RMC_TARGET_COL = 'y'", 1.0),),
    )
    contract = ValidationInputContract.minimal_for_test(
        material_hashes={
            "notebook": "n",
            "sample": "s",
            "pmml": "p",
            "dictionary": "d",
        },
        target_col=target,
    )

    assert input_contract_from_dict(input_contract_to_dict(contract)) == contract
    with pytest.raises(ValueError, match="no PMML manifest"):
        contract.require_pmml_manifest()
    with pytest.raises(ValueError, match="no sample schema"):
        contract.require_sample_schema()
    with pytest.raises(ValueError, match="no resolved feature metadata"):
        contract.require_feature_metadata()
    with pytest.raises(ValueError, match="no confirmed PMML output field"):
        contract.require_output_field()
    with pytest.raises(ValueError, match="no confirmed model parameters"):
        contract.require_model_params()


def test_input_contract_rejects_unknown_schema_version():
    with pytest.raises(
        ValueError, match="unsupported validation input contract schema"
    ):
        input_contract_from_dict({"schema_version": "unknown"})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(extra=True), "validation input contract keys"),
        (lambda p: p["material_hashes"].update(extra="x"), "material hash roles"),
        (
            lambda p: p["candidates"]["target_col"][0].update(extra=True),
            "field candidate keys",
        ),
        (
            lambda p: p["candidates"]["target_col"][0]["evidence"][0].update(
                extra=True
            ),
            "field evidence keys",
        ),
        (lambda p: p["sample_schema"].update(extra=True), "sample schema keys"),
        (lambda p: p["pmml_manifest"].update(extra=True), "PMML input manifest keys"),
        (
            lambda p: p["pmml_manifest"]["stress_units"][0].update(extra=True),
            "stress unit keys",
        ),
        (lambda p: p["feature_metadata"].update(extra=True), "feature metadata keys"),
        (
            lambda p: p["feature_metadata"]["rows"][0].update(extra=True),
            "feature metadata row keys",
        ),
        (
            lambda p: p["feature_metadata"]["coverage"].update(extra=True),
            "metadata coverage keys",
        ),
    ],
)
def test_input_contract_rejects_unknown_keys_at_each_nested_level(mutation, message):
    payload = input_contract_to_dict(make_candidate_contract())
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        input_contract_from_dict(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.pop("conflicts"), "validation input contract keys"),
        (
            lambda p: p["candidates"]["target_col"][0]["evidence"][0].pop("confidence"),
            "field evidence keys",
        ),
    ],
)
def test_input_contract_rejects_missing_keys(mutation, message):
    payload = input_contract_to_dict(make_candidate_contract())
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        input_contract_from_dict(payload)


@pytest.mark.parametrize(
    "field_path",
    [
        ("candidates", "target_col", 0, "evidence", 0, "confidence"),
        ("feature_metadata", "rows", 0, "importance"),
        ("feature_metadata", "coverage", "feature"),
        ("confirmed", "model_params", "learning_rate"),
    ],
)
def test_input_contract_rejects_non_finite_numbers_recursively(field_path):
    payload = input_contract_to_dict(make_ready_contract())
    current = payload
    for key in field_path[:-1]:
        current = current[key]
    current[field_path[-1]] = math.nan

    with pytest.raises(ValueError, match="finite"):
        input_contract_from_dict(payload)


def test_input_contract_rejects_non_json_values_recursively():
    payload = input_contract_to_dict(make_ready_contract())
    payload["confirmed"]["model_params"] = {"bad": object()}

    with pytest.raises(ValueError, match="JSON"):
        input_contract_from_dict(payload)


def test_sample_schema_round_trip_preserves_selected_excel_sheet():
    contract = make_candidate_contract()
    schema = replace(contract.require_sample_schema(), sheet_name="有效样本")

    restored = input_contract_from_dict(
        input_contract_to_dict(replace(contract, sample_schema=schema))
    )

    assert restored.require_sample_schema().sheet_name == "有效样本"


def test_transformation_decoder_is_strict_and_validates_operations():
    payload = {
        "operation": "date_to_month",
        "output_field": "apply_month",
        "input_fields": ["apply_date"],
        "params": {"format": "%Y-%m"},
    }
    assert transformation_spec_from_dict(payload) == TransformationSpec(
        operation="date_to_month",
        output_field="apply_month",
        input_fields=("apply_date",),
        params={"format": "%Y-%m"},
    )
    with pytest.raises(ValueError, match="invalid transformation keys"):
        transformation_spec_from_dict({**payload, "extra": True})
    with pytest.raises(ValueError, match="unsupported transformation operation"):
        transformation_spec_from_dict({**payload, "operation": "execute_python"})
    with pytest.raises(ValueError, match="non-empty"):
        transformation_spec_from_dict({**payload, "output_field": " "})


def test_transformation_decoder_rejects_deep_params_without_recursion_error():
    nested: object = "leaf"
    for _ in range(70):
        nested = {"nested": nested}

    with pytest.raises(ValueError, match="maximum depth"):
        transformation_spec_from_dict(
            {
                "operation": "copy",
                "output_field": "derived",
                "input_fields": ["x1"],
                "params": {"nested": nested},
            }
        )


def test_constant_mapping_params_preserve_numeric_keys_as_typed_pairs():
    payload = {
        "operation": "constant_mapping",
        "output_field": "split",
        "input_fields": ["raw_split"],
        "params": {
            "pairs": [
                {"from": 1, "to": "train"},
                {"from": "1", "to": "test"},
            ]
        },
    }

    restored = transformation_spec_from_dict(payload)

    assert restored.params["pairs"][0]["from"] == 1
    assert type(restored.params["pairs"][0]["from"]) is int
    assert restored.params["pairs"][1]["from"] == "1"
    with pytest.raises(ValueError, match="JSON object with string keys"):
        transformation_spec_from_dict({**payload, "params": {1: "train"}})


def test_require_helpers_return_confirmed_allowlisted_values():
    contract = make_ready_contract()

    assert contract.require_output_field() == "probability_1"
    assert contract.require_algorithm() == "xgb"
    assert contract.require_model_params() == {}
    assert (
        replace(
            contract, confirmed={**contract.confirmed, "algorithm": "lgb"}
        ).require_algorithm()
        == "lgb"
    )

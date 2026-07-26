from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from marvis.packs.strategy.model_evidence_tools import (
    run_materialize_model_evidence_v2,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    run_materialize_sample_design_v2,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    run_materialize_sample_design_v2_native,
)
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from test_strategy_model_evidence_tool import _fixture as _model_evidence_fixture
from test_strategy_sample_design_v2_tool import _setup as _sample_design_fixture
from test_strategy_sample_design_v2_native_tool import (
    _setup_native as _native_sample_design_fixture,
)


def _manifest_tool(name: str):
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    assert manifest.version == "0.19.0"
    return next(tool for tool in manifest.tools if tool.name == name)


def test_sample_design_v2_manifest_accepts_real_envelope_and_rejects_drift(
    tmp_path: Path,
) -> None:
    fixture = _sample_design_fixture(tmp_path, target_bad_value=1)
    output = run_materialize_sample_design_v2(
        fixture["request"], fixture["ctx"], fixture["runtime"]
    )
    tool = _manifest_tool("materialize_sample_design_v2")

    validate_against_schema(
        fixture["request"], tool.input_schema, label="sample-design V2 input"
    )
    validate_against_schema(
        output, tool.output_schema, label="sample-design V2 output"
    )

    extra = deepcopy(fixture["request"])
    extra["dataset_id"] = fixture["dataset"].id
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            extra, tool.input_schema, label="sample-design V2 extra input"
        )

    omitted_nullable = deepcopy(fixture["request"])
    del omitted_nullable["field_bindings"]["weight_field"]
    with pytest.raises(SchemaValidationError, match="required property"):
        validate_against_schema(
            omitted_nullable,
            tool.input_schema,
            label="sample-design V2 omitted nullable input",
        )

    for role, field, value in (
        ("membership", "artifact_id", "a" * 64),
        ("membership", "content_hash", "b" * 64),
        ("membership", "download_url", "/forged-membership"),
        ("bundle", "artifact_id", "c" * 64),
        ("bundle", "download_url", "/forged-bundle"),
    ):
        old_artifact_shape = deepcopy(output)
        old_artifact_shape["artifacts"][role][field] = value
        with pytest.raises(SchemaValidationError, match="Additional properties"):
            validate_against_schema(
                old_artifact_shape,
                tool.output_schema,
                label="sample-design V2 old artifact output",
            )


def test_native_sample_design_v2_manifest_accepts_real_closed_envelope(
    tmp_path: Path,
) -> None:
    fixture = _native_sample_design_fixture(tmp_path, target_bad_value=1)
    output = run_materialize_sample_design_v2_native(
        fixture["request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    tool = _manifest_tool("materialize_sample_design_v2_native")

    validate_against_schema(
        fixture["request"],
        tool.input_schema,
        label="native sample-design V2 input",
    )
    validate_against_schema(
        output,
        tool.output_schema,
        label="native sample-design V2 output",
    )

    legacy_drift = deepcopy(fixture["request"])
    legacy_drift["legacy_sample_design_ref"] = {}
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            legacy_drift,
            tool.input_schema,
            label="native sample-design V2 legacy drift",
        )

    nested_extra = deepcopy(fixture["request"])
    nested_extra["approval_population"]["invented"] = True
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            nested_extra,
            tool.input_schema,
            label="native sample-design V2 nested drift",
        )

    malformed_predicate = deepcopy(fixture["request"])
    malformed_predicate["risk_population"]["inclusion"] = {
        "op": "and",
        "args": [
            {
                "op": "eq",
                "left": {"column": "channel"},
                "right": {"literal": "app"},
            }
        ],
    }
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            malformed_predicate,
            tool.input_schema,
            label="native sample-design V2 malformed predicate",
        )

    forged_membership = deepcopy(output)
    forged_membership["membership"]["invented"] = True
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            forged_membership,
            tool.output_schema,
            label="native sample-design V2 forged membership",
        )

    forged_bundle = deepcopy(output)
    forged_bundle["bundle"]["invented"] = True
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            forged_bundle,
            tool.output_schema,
            label="native sample-design V2 forged bundle",
        )

    forged_compatibility = deepcopy(output)
    forged_compatibility["bundle"]["sample_design"]["compatibility"][
        "invented"
    ] = True
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            forged_compatibility,
            tool.output_schema,
            label="native sample-design V2 forged compatibility",
        )

    for field, forged_value in (
        ("populations", [{"invented": True}, {"invented": True}]),
        ("historical_score", {"invented": True}),
        ("policy", {"invented": True}),
        ("diagnostics", [{"invented": True}]),
        ("metric_definitions", [{"invented": True}]),
        ("metric_observations", [{"invented": True}]),
    ):
        forged_child = deepcopy(output)
        forged_child["bundle"][field] = forged_value
        with pytest.raises(SchemaValidationError):
            validate_against_schema(
                forged_child,
                tool.output_schema,
                label=f"native sample-design V2 forged {field}",
            )


def test_model_evidence_v2_manifest_accepts_real_envelope_and_rejects_old_shape(
    tmp_path: Path,
) -> None:
    fixture = _model_evidence_fixture(tmp_path)
    output = run_materialize_model_evidence_v2(
        fixture["inputs"], fixture["ctx"], fixture["runtime"]
    )
    tool = _manifest_tool("materialize_model_evidence_v2")

    validate_against_schema(
        fixture["inputs"], tool.input_schema, label="model-evidence V2 input"
    )
    validate_against_schema(
        output, tool.output_schema, label="model-evidence V2 output"
    )

    extra = deepcopy(fixture["inputs"])
    extra["univariate_sources"][0]["candidate_id"] = fixture["candidate"][
        "candidate_id"
    ]
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            extra, tool.input_schema, label="model-evidence V2 extra input"
        )

    old_artifact_shape = deepcopy(output)
    old_artifact_shape["artifact"]["artifact_id"] = "a" * 64
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            old_artifact_shape,
            tool.output_schema,
            label="model-evidence V2 old artifact output",
        )

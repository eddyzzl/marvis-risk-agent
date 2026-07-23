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
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from test_strategy_model_evidence_tool import _fixture as _model_evidence_fixture
from test_strategy_sample_design_v2_tool import _setup as _sample_design_fixture


def _manifest_tool(name: str):
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
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

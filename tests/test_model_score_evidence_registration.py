from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from marvis.packs.modeling import evidence_tools
from marvis.packs.modeling import score_evidence_tools
from marvis.packs.modeling import tools as modeling_tools
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from tests.test_modeling_training_evidence_tool import (
    _binding,
    _fixture,
    _run as run_training,
)


TOOL_NAME = "materialize_model_score_evidence_v2"


def _manifest_tool():
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "modeling",
        builtin=True,
    )
    assert manifest.version == "0.6.0"
    return next(tool for tool in manifest.tools if tool.name == TOOL_NAME)


def _score_inputs(fixture: dict, training_output: dict) -> dict:
    return {
        "training_evidence_ref": evidence_tools.build_training_evidence_ref(
            _binding(fixture, training_output)
        )
    }


def test_model_score_evidence_v2_manifest_is_strict_and_governed() -> None:
    tool = _manifest_tool()

    assert tool.entrypoint == "tool_materialize_model_score_evidence_v2"
    assert (
        getattr(modeling_tools, tool.entrypoint)
        is score_evidence_tools.tool_materialize_model_score_evidence_v2
    )
    assert tool.determinism == "deterministic"
    assert tool.timeout_seconds == 1800
    assert tool.memory_limit_mb == 4096
    assert set(tool.side_effects) == {
        "read:task",
        "read:artifacts",
        "read:dataset",
        "read:experiment",
        "read:model",
        "write:artifact",
        "write:task",
    }
    assert tool.policy.human_decision_gate == "none"
    assert tool.policy.effect_authorization == "none"
    assert tool.input_schema["required"] == ["training_evidence_ref"]
    assert set(tool.input_schema["properties"]) == {"training_evidence_ref"}
    assert tool.input_schema["additionalProperties"] is False
    training_ref_schema = tool.input_schema["$defs"]["training_evidence_ref"]
    assert set(training_ref_schema["required"]) == {
        "sample_design_ref",
        "model_binary_artifact_id",
        "expected_model_binary_artifact_content_hash",
        "evidence_artifact_id",
        "expected_evidence_artifact_content_hash",
        "expected_experiment_id",
        "expected_model_artifact_id",
        "expected_evidence_id",
        "expected_evidence_content_hash",
    }
    assert training_ref_schema["additionalProperties"] is False
    assert (
        tool.output_schema["properties"]["schema_version"]["const"]
        == score_evidence_tools.MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION
    )
    assert tool.output_schema["additionalProperties"] is False


def test_registered_entrypoint_output_matches_runtime_validator_and_json_schema(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    training_output = run_training(fixture)
    inputs = _score_inputs(fixture, training_output)
    tool = _manifest_tool()

    output = getattr(modeling_tools, tool.entrypoint)(inputs, fixture["ctx"])

    validate_against_schema(
        inputs,
        tool.input_schema,
        label="model-score-evidence V2 input",
    )
    validate_against_schema(
        output,
        tool.output_schema,
        label="model-score-evidence V2 output",
    )
    assert (
        modeling_tools.validate_materialize_model_score_evidence_v2_tool_output(
            output,
            runtime=fixture["runtime"],
            task_id=fixture["task"].id,
        )
        == output
    )
    loaded = modeling_tools.load_model_score_evidence_artifacts(
        fixture["runtime"],
        task_id=fixture["task"].id,
        evidence_artifact_id=output["artifacts"]["score_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"][
            "score_evidence"
        ]["content_hash"],
        score_vector_artifact_id=output["artifacts"]["score_vector"]["artifact_id"],
        expected_score_vector_artifact_content_hash=output["artifacts"][
            "score_vector"
        ]["content_hash"],
    )
    assert isinstance(
        loaded,
        modeling_tools.ModelScoreEvidenceArtifactBinding,
    )

    missing_ref = deepcopy(inputs)
    del missing_ref["training_evidence_ref"]["expected_evidence_content_hash"]
    with pytest.raises(SchemaValidationError, match="required property"):
        validate_against_schema(
            missing_ref,
            tool.input_schema,
            label="incomplete model-score-evidence V2 input",
        )

    caller_metric = deepcopy(inputs)
    caller_metric["auc"] = 0.99
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            caller_metric,
            tool.input_schema,
            label="caller-owned model-score-evidence metric",
        )

    forged_output = deepcopy(output)
    forged_output["governance"]["not_selected"] = False
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            forged_output,
            tool.output_schema,
            label="forged model-score-evidence V2 output",
        )

    extra_output = deepcopy(output)
    extra_output["metrics"] = {"auc": 0.99}
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            extra_output,
            tool.output_schema,
            label="extra model-score-evidence V2 output",
        )

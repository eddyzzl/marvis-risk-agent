from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from marvis.packs.strategy import tools as strategy_tools
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from tests.test_model_score_evidence_tool import _run_score
from tests.test_modeling_training_evidence_tool import (
    _fixture,
    _run as run_training,
)


def _manifest_tool(name: str):
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    assert manifest.version == "0.19.0"
    return next(tool for tool in manifest.tools if tool.name == name)


def test_scorecard_candidate_tools_are_registered_without_strategy_effects() -> None:
    expected = {
        "build_scorecard_band_asset": {
            "entrypoint": "tool_build_scorecard_band_asset",
            "side_effects": {
                "read:task",
                "read:artifacts",
                "read:dataset",
                "read:experiment",
                "read:model",
                "write:artifact",
            },
        },
        "materialize_scorecard_cutoff_selection": {
            "entrypoint": "tool_materialize_scorecard_cutoff_selection",
            "side_effects": {
                "read:task",
                "read:artifacts",
                "read:dataset",
                "read:experiment",
                "read:model",
                "write:artifact",
            },
        },
    }

    for name, contract in expected.items():
        tool = _manifest_tool(name)
        assert tool.entrypoint == contract["entrypoint"]
        assert callable(getattr(strategy_tools, tool.entrypoint))
        assert tool.determinism == "deterministic"
        assert tool.failure_policy == "fail"
        assert tool.policy.human_decision_gate == "none"
        assert tool.policy.effect_authorization == "none"
        assert set(tool.side_effects) == contract["side_effects"]
        assert tool.input_schema["additionalProperties"] is False
        assert tool.output_schema["additionalProperties"] is False
        for boundary in (
            "not_admitted",
            "not_applied",
            "not_adopted",
            "not_deployed",
        ):
            assert tool.output_schema["properties"][boundary] == {"const": True}

    build = _manifest_tool("build_scorecard_band_asset")
    assert set(build.input_schema["required"]) == {
        "score_evidence_ref",
        "sample_design_ref",
    }
    assert set(build.input_schema["properties"]) == {
        "score_evidence_ref",
        "sample_design_ref",
        "banding",
        "raw_pd_band_edges",
    }
    assert build.input_schema["properties"]["banding"][
        "additionalProperties"
    ] is False
    selection = _manifest_tool("materialize_scorecard_cutoff_selection")
    assert set(selection.input_schema["required"]) == {
        "source_artifact_id",
        "expected_source_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "cutoff_id",
    }
    assert set(selection.input_schema["properties"]) == {
        "source_artifact_id",
        "expected_source_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "cutoff_id",
        "reason",
    }


@pytest.mark.slow
def test_registered_scorecard_entrypoints_accept_real_envelopes_and_reject_forgery(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["inputs"]["recipe"] = "scorecard"
    fixture["inputs"]["params"].update(
        {
            "max_iter": 200,
            "scorecard_max_bins": 3,
        }
    )
    training_output = run_training(fixture)
    score_output = _run_score(fixture, training_output)
    score_artifacts = score_output["artifacts"]
    build_inputs = {
        "score_evidence_ref": {
            "evidence_artifact_id": score_artifacts["score_evidence"][
                "artifact_id"
            ],
            "expected_evidence_artifact_content_hash": score_artifacts[
                "score_evidence"
            ]["content_hash"],
            "score_vector_artifact_id": score_artifacts["score_vector"][
                "artifact_id"
            ],
            "expected_score_vector_artifact_content_hash": score_artifacts[
                "score_vector"
            ]["content_hash"],
        },
        "sample_design_ref": dict(fixture["sample_ref"]),
        "banding": {
            "method": "equal_frequency",
            "bin_count": 3,
        },
    }
    build_tool = _manifest_tool("build_scorecard_band_asset")

    validate_against_schema(
        build_inputs,
        build_tool.input_schema,
        label="scorecard band build input",
    )
    build_output = getattr(strategy_tools, build_tool.entrypoint)(
        build_inputs,
        fixture["ctx"],
    )
    validate_against_schema(
        build_output,
        build_tool.output_schema,
        label="scorecard band build output",
    )

    manual_inputs = deepcopy(build_inputs)
    del manual_inputs["banding"]
    manual_inputs["raw_pd_band_edges"] = [0.0, 0.4, 0.7, 1.0]
    validate_against_schema(
        manual_inputs,
        build_tool.input_schema,
        label="manual scorecard band build input",
    )
    conflicting_banding = deepcopy(manual_inputs)
    conflicting_banding["banding"] = {
        "method": "equal_frequency",
        "bin_count": 3,
    }
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            conflicting_banding,
            build_tool.input_schema,
            label="conflicting scorecard banding input",
        )

    band_artifact = build_output["artifacts"][0]
    cutoff = build_output["scorecard_band_asset"]["cutoffs"][0]
    selection_inputs = {
        "source_artifact_id": band_artifact["artifact_id"],
        "expected_source_artifact_content_hash": band_artifact["content_hash"],
        "expected_asset_id": build_output["asset_id"],
        "expected_asset_hash": build_output["asset_hash"],
        "cutoff_id": cutoff["cutoff_id"],
        "reason": "人工确认评估该通过线",
    }
    selection_tool = _manifest_tool(
        "materialize_scorecard_cutoff_selection"
    )
    validate_against_schema(
        selection_inputs,
        selection_tool.input_schema,
        label="scorecard cutoff selection input",
    )
    selection_output = getattr(strategy_tools, selection_tool.entrypoint)(
        selection_inputs,
        fixture["ctx"],
    )
    validate_against_schema(
        selection_output,
        selection_tool.output_schema,
        label="scorecard cutoff selection output",
    )

    for tool, valid_input, forged_field in (
        (build_tool, build_inputs, "dataset_id"),
        (selection_tool, selection_inputs, "strategy_id"),
    ):
        forged = deepcopy(valid_input)
        forged[forged_field] = "caller-owned-platform-field"
        with pytest.raises(SchemaValidationError, match="Additional properties"):
            validate_against_schema(
                forged,
                tool.input_schema,
                label=f"forged {tool.name} input",
            )

    nested_forgery = deepcopy(build_inputs)
    nested_forgery["score_evidence_ref"]["score_product"] = "caller-owned"
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            nested_forgery,
            build_tool.input_schema,
            label="forged score-evidence ref",
        )

    nested_output_forgery = deepcopy(build_output)
    nested_output_forgery["scorecard_band_asset"]["governance"][
        "pool_entry_id"
    ] = "forged"
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            nested_output_forgery,
            build_tool.output_schema,
            label="forged scorecard band governance output",
        )

    forged_output = deepcopy(selection_output)
    forged_output["adopted"] = True
    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_against_schema(
            forged_output,
            selection_tool.output_schema,
            label="forged scorecard cutoff output",
        )

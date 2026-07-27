"""Manifest contract for exact Pool-to-draft Strategy materialization."""

from __future__ import annotations

import json
from pathlib import Path


def _tool() -> dict:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "marvis"
            / "packs"
            / "strategy"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    return next(
        tool
        for tool in manifest["tools"]
        if tool["name"] == "materialize_strategy_from_pool"
    )


def test_pool_materialize_manifest_freezes_exactly_six_inputs_without_effect_gate() -> None:
    tool = _tool()
    expected_inputs = {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "expected_pool_artifact_id",
        "expected_pool_artifact_content_hash",
        "expected_design_hash",
    }

    assert set(tool["input_schema"]["properties"]) == expected_inputs
    assert set(tool["input_schema"]["required"]) == expected_inputs
    assert tool["input_schema"]["additionalProperties"] is False
    assert tool["entrypoint"] == "tool_materialize_strategy_from_pool"
    assert tool["determinism"] == "deterministic"
    assert tool["policy"] == {
        "schema_version": "tool-policy.v1",
        "human_decision_gate": "none",
        "effect_authorization": "none",
    }
    assert "write:strategy" in tool["side_effects"]
    assert not {"write:deployment", "write:monitoring"} & set(tool["side_effects"])


def test_pool_materialize_manifest_exposes_lifecycle_and_readiness_truth() -> None:
    output = _tool()["output_schema"]

    assert output["additionalProperties"] is False
    assert set(output["required"]) == {
        "schema_version",
        "materialization_id",
        "strategy_ref",
        "pool_ref",
        "design_hash",
        "requirements",
        "lifecycle",
    }
    assert set(output["properties"]["requirements"]["required"]) == {
        "requirements_hash",
        "requirement_count",
        "virtual_fields",
        "runtime_requirements_supported",
        "blocker_code",
    }
    assert set(output["properties"]["lifecycle"]["required"]) == {
        "created_status",
        "created_asset_status",
        "current_status",
        "current_asset_status",
        "adopted_by_this_tool",
        "deployed_by_this_tool",
    }
    assert "draft" in _tool()["summary"]
    assert "without adopting or deploying" in _tool()["summary"]

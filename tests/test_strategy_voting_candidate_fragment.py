from __future__ import annotations

from copy import deepcopy

import pytest

from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
    validate_verified_candidate_fragment,
)
from marvis.packs.strategy.pool import (
    CandidatePoolError,
    add_verified_candidate_fragment,
    compile_strategy_pool,
)
from marvis.packs.strategy.voting_candidate import build_voting_candidate_asset
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    VotingCandidateFragmentError,
    voting_candidate_to_verified_fragment,
)


SAMPLE_DESIGN_REF = {
    "artifact_id": "1" * 64,
    "artifact_content_hash": "2" * 64,
    "sample_design_id": "sample-design-voting-fragment",
    "sample_design_content_hash": "3" * 64,
    "partition": "development",
}


def _source(index: int) -> dict:
    digest = str(index + 1) * 64
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{index}",
            "artifact_kind": "strategy_test_candidate_json",
            "artifact_schema_version": "strategy.test-candidate-artifact.v1",
            "artifact_content_hash": digest,
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "strategy.test-candidate.v1",
            "asset_id": f"source-asset-{index}",
            "asset_hash": digest,
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=f"source-rule-{index}",
        condition={
            "op": "compare",
            "field": f"score_{index}",
            "operator": ">=",
            "value": 1,
            "missing": "no_match",
        },
        requirements=[],
        effect_id=f"source-effect-{index}",
        evidence_id=f"source-evidence-{index}",
        evidence_hash=digest,
        evidence_identity={
            "dataset_id": "dataset-1",
            "dataset_content_hash": "a" * 64,
            "workspace_revision": 1,
            "workspace_generation": 1,
            "semantic_mapping_hash": "b" * 64,
            "sample_context_hash": "c" * 64,
        },
    )


def _asset() -> dict:
    pool = None
    for index in range(2):
        pool = add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action={"type": "approval", "value": "approve"},
            verified_candidate_fragment=_source(index),
            action={"type": "reject", "value": "reject"},
        )
    assert pool is not None
    return build_voting_candidate_asset(
        pool,
        selected_entry_ids=[entry["entry_id"] for entry in pool["entries"]],
        n=2,
        target_col="bad",
        sample_design_ref=SAMPLE_DESIGN_REF,
        effect={
            "population_count": 10,
            "labeled_count": 10,
            "matched_count": 2,
            "matched_rate": 0.2,
            "matched_bad_count": 1,
            "matched_bad_rate": 0.5,
            "unmatched_count": 8,
            "unmatched_bad_count": 1,
            "unmatched_bad_rate": 0.125,
            "bad_capture_rate": 0.5,
            "lift": 2.5,
        },
    )


def _binding(asset: dict) -> dict:
    return {
        "artifact_id": "artifact-voting",
        "task_id": "task-1",
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "content_hash": "d" * 64,
        "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
        "artifact_schema_version": VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
    }


def test_verified_voting_asset_projects_without_business_action() -> None:
    asset = _asset()
    fragment = voting_candidate_to_verified_fragment(
        asset,
        artifact_binding=_binding(asset),
    )

    assert validate_verified_candidate_fragment(fragment) == fragment
    assert fragment["fragment"]["condition"] == asset["rule"]["condition"]
    assert fragment["fragment"]["effect_id"] == asset["effect"]["effect_id"]
    assert fragment["asset"]["asset_id"] == asset["asset_id"]
    assert "action" not in fragment


def test_voting_fragment_rejects_forged_artifact_binding() -> None:
    asset = _asset()
    forged = deepcopy(_binding(asset))
    forged["task_id"] = "other-task"

    with pytest.raises(VotingCandidateFragmentError, match="task_id"):
        voting_candidate_to_verified_fragment(asset, artifact_binding=forged)


def test_voting_fragment_requires_governed_placement_and_preserves_earlier_rule() -> None:
    pool = None
    for index in range(3):
        pool = add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action={"type": "approval", "value": "approve"},
            verified_candidate_fragment=_source(index),
            action={"type": "reject", "value": "reject"},
        )
    assert pool is not None
    selected_ids = [entry["entry_id"] for entry in pool["entries"][1:]]
    asset = build_voting_candidate_asset(
        pool,
        selected_entry_ids=selected_ids,
        n=2,
        target_col="bad",
        sample_design_ref=SAMPLE_DESIGN_REF,
        effect={
            "population_count": 10,
            "labeled_count": 10,
            "matched_count": 2,
            "matched_rate": 0.2,
            "matched_bad_count": 1,
            "matched_bad_rate": 0.5,
            "unmatched_count": 8,
            "unmatched_bad_count": 1,
            "unmatched_bad_rate": 0.125,
            "bad_capture_rate": 0.5,
            "lift": 2.5,
        },
    )
    fragment = voting_candidate_to_verified_fragment(
        asset,
        artifact_binding=_binding(asset),
    )

    with pytest.raises(CandidatePoolError, match="cannot be appended"):
        add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action={"type": "approval", "value": "approve"},
            verified_candidate_fragment=fragment,
            action={"type": "review", "value": "review"},
        )
    with pytest.raises(CandidatePoolError, match="do not match"):
        add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action={"type": "approval", "value": "approve"},
            verified_candidate_fragment=fragment,
            action={"type": "review", "value": "review"},
            placement_mode="before_selected_members",
            selected_entry_ids=[
                pool["entries"][0]["entry_id"],
                pool["entries"][1]["entry_id"],
            ],
        )

    placed = add_verified_candidate_fragment(
        pool,
        task_id="task-1",
        strategy_type="approval",
        default_action={"type": "approval", "value": "approve"},
        verified_candidate_fragment=fragment,
        action={"type": "review", "value": "review"},
        placement_mode="before_selected_members",
        selected_entry_ids=selected_ids,
    )
    assert [entry["source"]["asset_type"] for entry in placed["entries"]] == [
        "test_candidate",
        "voting_n_of_k",
        "test_candidate",
        "test_candidate",
    ]
    assert compile_strategy_pool(placed)["strategy_spec"]["rules"][1][
        "condition"
    ]["op"] == "n_of_k"

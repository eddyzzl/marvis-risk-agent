from __future__ import annotations

from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from marvis.packs.strategy.pool_apply_tools import (
    RESULT_DATASET_ROLE,
    run_apply_strategy_pool,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    model_score_virtual_field,
)
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.voting_candidate_tools import (
    run_build_voting_candidate,
)
from marvis.repositories.strategy_pool import (
    StrategyCandidatePoolRepository,
)
from tests.test_strategy_voting_scorecard import (
    _two_scorecard_pool_entries,
    _voting_inputs,
)


def _apply_inputs(pool: dict) -> dict:
    return {
        "strategy_type": "approval",
        "expected_pool_revision": pool["revision"],
        "expected_pool_snapshot_hash": pool["snapshot_hash"],
    }


def _assert_governed_model_score_result(
    real: dict,
    result: dict,
    *,
    expected_virtual_field: str,
) -> None:
    runtime = real["runtime"]
    source_dataset = real["fx"]["dataset"]
    derived_dataset = runtime.registry.get(result["result"]["dataset_id"])

    assert result["activated"] is False
    assert result["adopted"] is False
    assert result["deployed"] is False
    assert result["requirements"]["virtual_fields"] == [
        expected_virtual_field
    ]
    assert result["source"]["dataset_id"] == source_dataset.id
    assert result["source"]["row_count"] == source_dataset.row_count
    assert result["result"]["row_count"] == source_dataset.row_count
    assert derived_dataset.task_id == real["fx"]["task"].id
    assert derived_dataset.role == RESULT_DATASET_ROLE
    assert derived_dataset.row_count == source_dataset.row_count

    source = runtime.backend.read_frame(
        runtime.registry.resolve_verified_path(source_dataset.id)
    )
    derived = runtime.backend.read_frame(
        runtime.registry.resolve_verified_path(derived_dataset.id)
    )
    assert_frame_equal(
        derived.loc[:, source.columns],
        source,
        check_dtype=True,
        check_exact=True,
    )
    assert expected_virtual_field not in source.columns
    assert expected_virtual_field not in derived.columns
    assert set(result["columns"].values()).isdisjoint(source.columns)
    assert derived.columns.tolist() == [
        *source.columns.tolist(),
        result["columns"]["action"],
        result["columns"]["value"],
        result["columns"]["value_type"],
        result["columns"]["rule_id"],
        result["columns"]["entry_id"],
        result["columns"]["reason_code"],
    ]

    row_count = source_dataset.row_count
    assert sum(result["action_counts"].values()) == row_count
    assert sum(result["rule_counts"].values()) + result["default_count"] == row_count
    assert (
        sum(result["entry_counts"].values()) + result["default_count"]
        == row_count
    )


@pytest.mark.slow
def test_current_pool_applies_real_scorecards_then_replaced_voting_without_writing_scores(
    tmp_path: Path,
) -> None:
    real = _two_scorecard_pool_entries(tmp_path)
    runtime = real["runtime"]
    task_id = real["fx"]["task"].id
    repository = StrategyCandidatePoolRepository(
        real["fx"]["settings"].db_path
    )
    source_workspace = runtime.data_workspaces.get_or_default(task_id)
    score_vector_id = real["band"]["scorecard_band_asset"]["source_refs"][
        "score_vector"
    ]["artifact_id"]
    virtual_field = model_score_virtual_field(score_vector_id)

    scorecard_pool = real["pool"]
    scorecard_rules = {
        entry["rule_id"] for entry in scorecard_pool["entries"]
    }
    assert all(
        requirement["virtual_field"] == virtual_field
        for entry in scorecard_pool["entries"]
        for requirement in entry["execution"]["requirements"]
    )

    scorecard_result = run_apply_strategy_pool(
        _apply_inputs(scorecard_pool),
        real["fx"]["ctx"],
        runtime,
    )

    _assert_governed_model_score_result(
        real,
        scorecard_result,
        expected_virtual_field=virtual_field,
    )
    assert set(scorecard_result["rule_counts"]).issubset(scorecard_rules)
    assert sum(scorecard_result["rule_counts"].values()) > 0
    assert repository.get_current(task_id, "approval") == scorecard_pool

    voting = run_build_voting_candidate(
        _voting_inputs(real),
        real["fx"]["ctx"],
        runtime,
    )
    descriptor = voting["artifacts"][0]
    admitted = run_add_candidate_to_pool(
        {
            "source_artifact_id": descriptor["artifact_id"],
            "expected_artifact_content_hash": descriptor["content_hash"],
            "expected_asset_id": voting["asset_id"],
            "expected_asset_hash": voting["asset_hash"],
            "strategy_type": "approval",
            "default_action": {
                "type": "approval",
                "value": "approve",
                "reason_code": None,
                "stop": True,
            },
            "action": {
                "type": "review",
                "value": "review",
                "reason_code": "SCORECARD_VOTING",
                "stop": True,
            },
            "expected_pool_revision": scorecard_pool["revision"],
            "expected_pool_snapshot_hash": scorecard_pool["snapshot_hash"],
            "placement_mode": "replace_selected_members",
        },
        real["fx"]["ctx"],
        runtime,
    )
    voting_pool = admitted["pool"]
    assert [entry["rule_id"] for entry in voting_pool["entries"]] == [
        voting["rule_id"]
    ]

    voting_result = run_apply_strategy_pool(
        _apply_inputs(voting_pool),
        real["fx"]["ctx"],
        runtime,
    )

    _assert_governed_model_score_result(
        real,
        voting_result,
        expected_virtual_field=virtual_field,
    )
    assert voting_result["rule_counts"][voting["rule_id"]] > 0
    [voting_entry] = voting_pool["entries"]
    assert voting_result["entry_counts"] == {
        voting_entry["entry_id"]: voting_result["rule_counts"][
            voting["rule_id"]
        ]
    }
    assert repository.get_current(task_id, "approval") == voting_pool
    assert runtime.data_workspaces.get_or_default(task_id) == source_workspace
    assert runtime.strategies.list_for_task(task_id) == []

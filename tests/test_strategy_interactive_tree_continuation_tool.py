from __future__ import annotations

import json
from pathlib import Path

import pytest

from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def _search_frontier(scenario) -> dict:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    parent = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": root_id,
            "operation": "prune_subtree",
            "reason": "Create an explicitly reviewed continuation frontier.",
        },
        scenario.ctx,
    )
    return strategy_tools.tool_search_interactive_tree_split_candidates(
        {
            "source_tree_id": parent["revision_id"],
            "node_id": root_id,
            "mode": "all_features",
            "max_thresholds_per_feature": 4,
            "max_row_evaluations": 10_000,
        },
        scenario.ctx,
    )


def _inputs(search: dict) -> dict:
    candidate = next(
        item
        for item in search["search_result"]["candidates"]
        if item["eligible"]
    )
    return {
        "search_id": search["search_id"],
        "candidate_id": candidate["candidate_id"],
        "max_additional_depth": 3,
        "min_gini_gain": 0.0,
        "max_generated_nodes": 15,
        "max_thresholds_per_feature": 4,
        "max_row_evaluations": 100_000,
        "objective": "max_gini_gain",
        "tie_break": "eligible_gain_feature_threshold_candidate_id",
        "reason": "Continue only from the reviewed seed candidate.",
    }


def test_auto_continue_tool_is_idempotent_governed_and_pool_neutral(
    scenario,
) -> None:
    search = _search_frontier(scenario)
    inputs = _inputs(search)

    first = strategy_tools.tool_auto_continue_interactive_tree(
        inputs,
        scenario.ctx,
    )
    repeated = strategy_tools.tool_auto_continue_interactive_tree(
        inputs,
        scenario.ctx,
    )

    assert repeated == first
    assert first["schema_version"] == (
        "strategy.auto-continue-interactive-tree-tool.v1"
    )
    assert first["search_id"] == search["search_id"]
    assert first["candidate_id"] == inputs["candidate_id"]
    assert first["edit"]["operation"] == "auto_continue_subtree"
    assert first["edit"]["controls"]["max_additional_depth"] == 3
    assert first["replay"]["exactly_once"] is True
    assert first["automatic_winner_selection"] is False
    assert first["pool_modified"] is False

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(
        item
        for item in manifest.tools
        if item.name == "auto_continue_interactive_tree"
    )
    validate_against_schema(
        first,
        tool.output_schema,
        label="automatic continuation output",
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        first["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    revision = json.loads(Path(record["path"]).read_text("utf-8"))
    assert revision["edit"]["search_hash"] == search["search_hash"]
    assert len(revision["tree"]["nodes"]) <= (
        first["visible_node_count"]
    )
    fragment = revision["fragments"][0]
    selection = (
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision["revision_id"],
                "source_node_id": fragment["source_node_id"],
                "selection_reason": "Use one reviewed generated frontier.",
            },
            scenario.ctx,
        )
    )
    added = strategy_tools.tool_add_candidate_to_pool(
        {
            "source_artifact_id": selection["artifacts"][0]["artifact_id"],
            "expected_artifact_content_hash": selection["artifacts"][0][
                "content_hash"
            ],
            "expected_asset_id": selection["semantic_tree_id"],
            "expected_asset_hash": selection["tree_hash"],
            "strategy_type": "approval",
            "default_action": {
                "type": "approval",
                "value": "approve",
                "reason_code": None,
                "stop": True,
            },
            "action": {
                "type": "reject",
                "value": "reject",
                "reason_code": "GENERATED_TREE_RISK",
                "stop": True,
            },
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        scenario.ctx,
    )
    compiled = strategy_tools.tool_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        scenario.ctx,
    )
    assert compiled["strategy_spec"]["rules"][0]["condition"] == (
        fragment["condition"]
    )


def test_auto_continue_requires_exact_eligible_candidate_and_policy(
    scenario,
) -> None:
    search = _search_frontier(scenario)
    inputs = _inputs(search)
    ineligible = next(
        (
            item
            for item in search["search_result"]["candidates"]
            if not item["eligible"]
        ),
        None,
    )
    if ineligible is not None:
        with pytest.raises(StrategyError, match="eligible"):
            strategy_tools.tool_auto_continue_interactive_tree(
                {**inputs, "candidate_id": ineligible["candidate_id"]},
                scenario.ctx,
            )
    with pytest.raises(StrategyError, match="objective|tie_break"):
        strategy_tools.tool_auto_continue_interactive_tree(
            {**inputs, "objective": "first_candidate"},
            scenario.ctx,
        )

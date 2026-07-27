from __future__ import annotations

import json
from pathlib import Path

import pytest

from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.candidate_lab_projection import (
    build_strategy_candidate_lab_projection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_split_search import (
    canonical_interactive_tree_split_search_json,
)
from marvis.packs.strategy.interactive_tree_split_search_tools import (
    INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
    INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL,
    canonical_interactive_tree_split_search_path,
    load_verified_interactive_tree_split_search,
)
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from tests.test_strategy_interactive_tree_tool import (
    _Scenario,
    _split_node_ids,
)

pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def test_search_all_features_is_task_local_canonical_and_idempotent(
    scenario: _Scenario,
) -> None:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    inputs = {
        "source_tree_id": scenario.source_asset["asset_id"],
        "node_id": root_id,
        "mode": "all_features",
        "max_thresholds_per_feature": 5,
        "max_row_evaluations": 2_000,
    }

    first = strategy_tools.tool_search_interactive_tree_split_candidates(
        inputs,
        scenario.ctx,
    )
    repeated = strategy_tools.tool_search_interactive_tree_split_candidates(
        inputs,
        scenario.ctx,
    )

    assert repeated == first
    assert first["feature_count"] == 2
    assert first["evaluated_candidates"] <= 10
    assert first["eligible_candidates"] > 0
    assert first["winner_selected"] is False
    assert first["tree_modified"] is False
    assert first["search_result"]["request"]["features"] == ["x", "z"]
    assert first["search_result"]["claims"] == {
        "rank_is_navigation_only": True,
        "winner_selected": False,
        "tree_modified": False,
    }
    [record] = _search_records(scenario)
    assert record["origin_tool"] == INTERACTIVE_TREE_SPLIT_SEARCH_ORIGIN_TOOL
    expected_path = canonical_interactive_tree_split_search_path(
        scenario.settings.tasks_dir,
        task_id=scenario.task.id,
        search_id=first["search_id"],
    )
    assert Path(record["path"]) == expected_path
    persisted = json.loads(expected_path.read_text("utf-8"))
    assert (
        expected_path.read_bytes()
        == canonical_interactive_tree_split_search_json(persisted).encode("utf-8")
    )
    verified = load_verified_interactive_tree_split_search(
        _runtime(scenario),
        task_id=scenario.task.id,
        search_id=first["search_id"],
    )
    assert verified.result == first["search_result"]
    assert verified.artifact_id == first["artifacts"][0]["artifact_id"]
    assert verified.source.source_tree_id == scenario.source_asset["asset_id"]
    projected = build_strategy_candidate_lab_projection(
        scenario.settings,
        scenario.task.id,
    )
    collection = projected["candidates"]["interactive_tree_split_search"]
    assert collection["total"] == 1
    [projected_search] = collection["all"]
    assert projected_search["search_id"] == first["search_id"]
    assert projected_search["claims"]["winner_selected"] is False
    assert projected_search["candidates"][0]["candidate_id"].startswith(
        "interactive-tree-split-candidate-"
    )

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(
        item
        for item in manifest.tools
        if item.name == "search_interactive_tree_split_candidates"
    )
    validate_against_schema(
        inputs,
        tool.input_schema,
        label="interactive-tree split search input",
    )
    validate_against_schema(
        first,
        tool.output_schema,
        label="interactive-tree split search output",
    )


def test_search_selected_feature_on_exact_revision_source(
    scenario: _Scenario,
) -> None:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    revision = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": root_id,
            "operation": "adjust_split_threshold",
            "threshold": 1.5,
            "reason": "Create an exact revised search source.",
        },
        scenario.ctx,
    )

    output = strategy_tools.tool_search_interactive_tree_split_candidates(
        {
            "source_tree_id": revision["revision_id"],
            "node_id": root_id,
            "mode": "selected_features",
            "features": ["x"],
            "max_thresholds_per_feature": 4,
            "max_row_evaluations": 1_000,
        },
        scenario.ctx,
    )

    assert output["source_tree_id"] == revision["revision_id"]
    assert output["search_result"]["request"]["features"] == ["x"]
    assert {
        candidate["feature"]
        for candidate in output["search_result"]["candidates"]
    } == {"x"}
    verified = load_verified_interactive_tree_split_search(
        _runtime(scenario),
        task_id=scenario.task.id,
        search_id=output["search_id"],
    )
    assert verified.source.parent_revision is not None
    assert (
        verified.source.parent_revision["revision_id"]
        == revision["revision_id"]
    )


def test_search_rejects_ungoverned_feature_hidden_node_and_bad_mode(
    scenario: _Scenario,
) -> None:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    with pytest.raises(StrategyError, match="feature universe"):
        strategy_tools.tool_search_interactive_tree_split_candidates(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": root_id,
                "mode": "selected_features",
                "features": ["customer_id"],
                "max_thresholds_per_feature": 4,
                "max_row_evaluations": 1_000,
            },
            scenario.ctx,
        )
    with pytest.raises(StrategyError, match="must not provide"):
        strategy_tools.tool_search_interactive_tree_split_candidates(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": root_id,
                "mode": "all_features",
                "features": ["x"],
                "max_thresholds_per_feature": 4,
                "max_row_evaluations": 1_000,
            },
            scenario.ctx,
        )
    deepest = _split_node_ids(scenario.source_asset)[-1]
    pruned = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": root_id,
            "operation": "prune_subtree",
            "reason": "Hide the descendant used by this negative test.",
        },
        scenario.ctx,
    )
    with pytest.raises(StrategyError, match="exact and visible"):
        strategy_tools.tool_search_interactive_tree_split_candidates(
            {
                "source_tree_id": pruned["revision_id"],
                "node_id": deepest,
                "mode": "all_features",
                "max_thresholds_per_feature": 4,
                "max_row_evaluations": 1_000,
            },
            scenario.ctx,
        )


def test_search_loader_rejects_file_tampering(scenario: _Scenario) -> None:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    output = strategy_tools.tool_search_interactive_tree_split_candidates(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": root_id,
            "mode": "all_features",
            "max_thresholds_per_feature": 3,
            "max_row_evaluations": 1_000,
        },
        scenario.ctx,
    )
    [record] = _search_records(scenario)
    path = Path(record["path"])
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(StrategyError, match="hash|content"):
        load_verified_interactive_tree_split_search(
            _runtime(scenario),
            task_id=scenario.task.id,
            search_id=output["search_id"],
        )


def _search_records(scenario: _Scenario) -> list[dict]:
    return [
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"] == INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND
    ]


def _runtime(scenario: _Scenario):
    return strategy_tools._runtime(scenario.ctx)

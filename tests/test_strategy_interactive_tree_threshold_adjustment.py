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


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {"approval": "approve", "reject": "reject"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _threshold_revision(scenario, *, threshold: float = 1.5) -> tuple[dict, dict]:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    result = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": root_id,
            "operation": "adjust_split_threshold",
            "threshold": threshold,
            "reason": "Reviewed threshold adjustment.",
        },
        scenario.ctx,
    )
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(
        item for item in manifest.tools if item.name == "revise_interactive_tree"
    )
    validate_against_schema(
        result,
        tool.output_schema,
        label="threshold-adjusted interactive-tree output",
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    return result, json.loads(Path(record["path"]).read_text("utf-8"))


def test_v2_threshold_frontier_selection_compiles_exact_effective_condition(
    scenario,
) -> None:
    _result, revision = _threshold_revision(scenario)
    fragment = revision["fragments"][0]
    selection = (
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision["revision_id"],
                "source_node_id": fragment["source_node_id"],
                "selection_reason": "Use the adjusted effective segment.",
            },
            scenario.ctx,
        )
    )
    assert selection["schema_version"] == (
        "strategy.materialize-interactive-tree-frontier-selection-tool.v2"
    )
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    validate_against_schema(
        selection,
        next(
            item
            for item in manifest.tools
            if item.name == "materialize_interactive_tree_frontier_selection"
        ).output_schema,
        label="v2 interactive-tree frontier selection output",
    )
    artifact = selection["artifacts"][0]
    added = strategy_tools.tool_add_candidate_to_pool(
        {
            "source_artifact_id": artifact["artifact_id"],
            "expected_artifact_content_hash": artifact["content_hash"],
            "expected_asset_id": selection["semantic_tree_id"],
            "expected_asset_hash": selection["tree_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject", reason="ADJUSTED_TREE_RISK"),
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
    [rule] = compiled["strategy_spec"]["rules"]
    assert rule["condition"] == fragment["condition"]
    assert rule["condition"]["value"] == 1.5
    applied = strategy_tools.tool_apply_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        scenario.ctx,
    )
    assert sum(applied["rule_counts"].values()) == 12
    assert applied["default_count"] == 12


def test_threshold_rejects_noop_nonfinite_hidden_frontier_and_min_leaf(
    scenario,
) -> None:
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    with pytest.raises(StrategyError, match="no-op"):
        strategy_tools.tool_revise_interactive_tree(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": root_id,
                "operation": "adjust_split_threshold",
                "threshold": 0.5,
            },
            scenario.ctx,
        )
    with pytest.raises(StrategyError, match="finite"):
        strategy_tools.tool_revise_interactive_tree(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": root_id,
                "operation": "adjust_split_threshold",
                "threshold": float("nan"),
            },
            scenario.ctx,
        )
    with pytest.raises(StrategyError, match="empty|min_leaf"):
        strategy_tools.tool_revise_interactive_tree(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": root_id,
                "operation": "adjust_split_threshold",
                "threshold": -1_000_000.0,
            },
            scenario.ctx,
        )


def test_threshold_allows_same_grouping_with_explicit_warning(scenario) -> None:
    result, revision = _threshold_revision(scenario, threshold=0.75)

    assert result["replay"]["grouping_unchanged"] is True
    assert result["replay"]["affected_row_count"] == 0
    assert result["replay"]["warning_codes"] == [
        "threshold_grouping_unchanged"
    ]
    assert revision["checks"]["warning_codes"] == [
        "threshold_grouping_unchanged"
    ]


def test_threshold_adjusts_nested_visible_split_without_changing_topology(
    scenario,
) -> None:
    split_nodes = [
        node
        for node in scenario.source_asset["tree_result"]["tree"]["nodes"]
        if node["kind"] == "split"
    ]
    nested = split_nodes[1]
    result = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": nested["node_id"],
            "operation": "adjust_split_threshold",
            "threshold": 3.5,
            "reason": "Move one development row across the nested split.",
        },
        scenario.ctx,
    )

    assert result["visible_node_count"] == len(
        scenario.source_asset["tree_result"]["tree"]["nodes"]
    )
    assert result["replay"]["affected_row_count"] > 0
    assert result["replay"]["grouping_unchanged"] is False


def test_v2_chain_keeps_threshold_when_pruned_and_rejects_hidden_adjustment(
    scenario,
) -> None:
    threshold_result, threshold_revision = _threshold_revision(scenario)
    split_nodes = [
        node
        for node in threshold_revision["tree"]["nodes"]
        if node["kind"] == "split"
    ]
    deepest = split_nodes[-1]
    pruned = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": threshold_result["revision_id"],
            "node_id": deepest["node_id"],
            "operation": "prune_subtree",
            "reason": "Prune after threshold review.",
        },
        scenario.ctx,
    )
    assert pruned["schema_version"] == "strategy.revise-interactive-tree-tool.v2"
    assert pruned["edit"]["operation"] == "prune_subtree"
    with pytest.raises(StrategyError, match="frontier|hidden"):
        strategy_tools.tool_revise_interactive_tree(
            {
                "source_tree_id": pruned["revision_id"],
                "node_id": deepest["node_id"],
                "operation": "adjust_split_threshold",
                "threshold": float(deepest["threshold"]) + 0.25,
            },
            scenario.ctx,
        )


def test_v2_threshold_frontier_group_compiles_exact_effective_or(
    scenario,
) -> None:
    _result, revision = _threshold_revision(scenario)
    selected = revision["fragments"][:2]
    group = (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision["revision_id"],
                "source_node_ids": [
                    item["source_node_id"] for item in selected
                ],
                "selection_reason": "Combine two adjusted frontier segments.",
            },
            scenario.ctx,
        )
    )
    assert group["schema_version"] == (
        "strategy.materialize-interactive-tree-frontier-group-selection-tool.v2"
    )
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    validate_against_schema(
        group,
        next(
            item
            for item in manifest.tools
            if item.name
            == "materialize_interactive_tree_frontier_group_selection"
        ).output_schema,
        label="v2 interactive-tree frontier group output",
    )
    artifact = group["artifacts"][0]
    added = strategy_tools.tool_add_candidate_to_pool(
        {
            "source_artifact_id": artifact["artifact_id"],
            "expected_artifact_content_hash": artifact["content_hash"],
            "expected_asset_id": group["semantic_tree_id"],
            "expected_asset_hash": group["tree_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject", reason="ADJUSTED_TREE_GROUP"),
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
    [rule] = compiled["strategy_spec"]["rules"]
    assert rule["condition"]["op"] == "or"
    assert len(rule["condition"]["args"]) == 2


def test_threshold_upgrade_from_v1_parent_preserves_pruned_frontier(
    scenario,
) -> None:
    split_ids = [
        node["node_id"]
        for node in scenario.source_asset["tree_result"]["tree"]["nodes"]
        if node["kind"] == "split"
    ]
    parent = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": split_ids[-1],
            "operation": "prune_subtree",
            "reason": "First preserve the reviewed v1 prune.",
        },
        scenario.ctx,
    )
    child = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": parent["revision_id"],
            "node_id": split_ids[0],
            "operation": "adjust_split_threshold",
            "threshold": 1.5,
            "reason": "Then upgrade the chain with an effective threshold.",
        },
        scenario.ctx,
    )

    assert child["schema_version"] == "strategy.revise-interactive-tree-tool.v2"
    assert child["parent_revision_id"] == parent["revision_id"]
    assert child["frontier_node_count"] == parent["frontier_node_count"]

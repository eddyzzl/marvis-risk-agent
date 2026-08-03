"""Turn binding for persisted frontier OR groups entering Strategy Pool."""

from __future__ import annotations

from types import SimpleNamespace

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import (
    _candidate_selection_artifact_slots,
    _strategy_pool_plan_slots,
)
from marvis.packs.strategy import tools as strategy_tools
from tests.test_strategy_interactive_tree_frontier_group_tool import _revision
from tests.test_strategy_interactive_tree_threshold_adjustment import (
    _threshold_revision,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def _draft(selection_id: str) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_add_candidate",
        workflow_inputs={
            "selection_id": selection_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
            "source_artifact_id": "forged-artifact",
            "expected_artifact_content_hash": "f" * 64,
            "expected_asset_id": "forged-tree",
            "expected_asset_hash": "e" * 64,
            "fragment_id": "forged-fragment",
        },
    )


def test_frontier_group_resolves_live_artifact_and_replayed_fragment(
    scenario,
) -> None:
    revision_result, revision = _revision(scenario)
    source_node_ids = revision["tree"]["frontier_node_ids"][:2]
    materialized = (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": list(reversed(source_node_ids)),
            },
            scenario.ctx,
        )
    )
    runtime = SimpleNamespace(settings=scenario.settings)
    descriptor = materialized["artifacts"][0]

    verified_slots, fragment_id = _candidate_selection_artifact_slots(
        runtime,
        task_id=scenario.task.id,
        selection_id=materialized["selection_id"],
    )
    plan_slots = _strategy_pool_plan_slots(
        runtime,
        scenario.task,
        _draft(materialized["selection_id"]),
    )

    assert verified_slots == {
        "source_artifact_id": descriptor["artifact_id"],
        "expected_artifact_content_hash": descriptor["content_hash"],
        "expected_asset_id": materialized["semantic_tree_id"],
        "expected_asset_hash": materialized["tree_hash"],
    }
    assert fragment_id == materialized["fragment_id"]
    assert {
        "source_artifact_id": plan_slots["source_artifact_id"],
        "expected_artifact_content_hash": plan_slots[
            "expected_artifact_content_hash"
        ],
        "expected_asset_id": plan_slots["expected_asset_id"],
        "expected_asset_hash": plan_slots["expected_asset_hash"],
    } == verified_slots
    assert "selection_id" not in plan_slots
    assert "fragment_id" not in plan_slots


def test_v2_frontier_group_resolves_into_pool_turn_slots(scenario) -> None:
    _result, revision = _threshold_revision(scenario)
    selected = revision["fragments"][:2]
    materialized = (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision["revision_id"],
                "source_node_ids": [
                    item["source_node_id"] for item in selected
                ],
                "selection_reason": "Combine reviewed v2 frontiers.",
            },
            scenario.ctx,
        )
    )

    verified_slots, fragment_id = _candidate_selection_artifact_slots(
        SimpleNamespace(settings=scenario.settings),
        task_id=scenario.task.id,
        selection_id=materialized["selection_id"],
    )

    assert verified_slots == {
        "source_artifact_id": materialized["artifacts"][0]["artifact_id"],
        "expected_artifact_content_hash": materialized["artifacts"][0][
            "content_hash"
        ],
        "expected_asset_id": materialized["semantic_tree_id"],
        "expected_asset_hash": materialized["tree_hash"],
    }
    assert fragment_id == materialized["fragment_id"]

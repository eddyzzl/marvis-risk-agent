"""Candidate Lab projections for governed interactive-tree topology."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.interactive_tree_tools import (
    INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
    INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
)
from tests.test_strategy_pool_interactive_tree_frontier import (
    _add_inputs as _interactive_pool_add_inputs,
    _materialize_frontier,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


NODE_FIELDS = {
    "node_id",
    "kind",
    "depth",
    "condition",
    "metrics",
    "is_visible",
    "is_frontier",
    "can_prune",
}
PRUNE_FIELDS = {"source_tree_id", "node_id", "operation"}
THRESHOLD_ADJUSTMENT_FIELDS = {
    "source_tree_id",
    "node_id",
    "operation",
    "feature",
    "current_threshold",
}
FEATURE_REPLACEMENT_FIELDS = {
    "source_tree_id",
    "node_id",
    "operation",
    "current_feature",
    "current_threshold",
}


def _revise(
    scenario,
    *,
    source_tree_id: str,
    node_id: str,
    reason: str,
) -> dict:
    return strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": source_tree_id,
            "node_id": node_id,
            "operation": "prune_subtree",
            "reason": reason,
        },
        scenario.ctx,
    )


def _revision_record_and_payload(scenario, result: Mapping) -> tuple[dict, dict]:
    artifact_id = result["artifacts"][0]["artifact_id"]
    record = scenario.repository.get_for_task(scenario.task.id, artifact_id)
    assert record is not None
    payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    return record, payload


def _automatic_item(projection: Mapping, asset_id: str) -> Mapping:
    collection = projection["candidates"]["automatic_tree"]
    return next(
        item
        for item in collection["all"]
        if item["detail"]["asset_id"] == asset_id
    )


def _assert_operable_topology(
    item: Mapping,
    *,
    source_tree_id: str,
    source_asset: Mapping,
    visible_node_ids: list[str],
    frontier_node_ids: list[str],
) -> None:
    assert item["detail"]["source_tree_id"] == source_tree_id
    expected_nodes = source_asset["tree_result"]["tree"]["nodes"]
    nodes = item["pointers"]["nodes"]
    assert len(nodes) == len(expected_nodes)
    assert len(nodes) <= 511
    assert [node["node_id"] for node in nodes] == [
        node["node_id"] for node in expected_nodes
    ]

    visible = set(visible_node_ids)
    frontier = set(frontier_node_ids)
    for actual, expected in zip(nodes, expected_nodes, strict=True):
        assert NODE_FIELDS <= set(actual)
        assert actual["node_id"] == expected["node_id"]
        assert actual["kind"] == expected["kind"]
        assert actual["depth"] == expected["depth"]
        assert isinstance(actual["condition"], Mapping)
        assert actual["metrics"] == expected["metrics"]
        assert actual["is_visible"] is (expected["node_id"] in visible)
        assert actual["is_frontier"] is (expected["node_id"] in frontier)
        assert actual["can_prune"] is (
            expected["kind"] == "split"
            and expected["node_id"] in visible
            and expected["node_id"] not in frontier
        )

    eligible_prunes = item["pointers"]["eligible_prunes"]
    assert all(set(pointer) == PRUNE_FIELDS for pointer in eligible_prunes)
    assert eligible_prunes == [
        {
            "source_tree_id": source_tree_id,
            "node_id": node["node_id"],
            "operation": "prune_subtree",
        }
        for node in nodes
        if node["can_prune"]
    ]
    eligible_threshold_adjustments = item["pointers"][
        "eligible_threshold_adjustments"
    ]
    assert all(
        set(pointer) == THRESHOLD_ADJUSTMENT_FIELDS
        for pointer in eligible_threshold_adjustments
    )
    assert eligible_threshold_adjustments == [
        {
            "source_tree_id": source_tree_id,
            "node_id": node["node_id"],
            "operation": "adjust_split_threshold",
            "feature": node["feature"],
            "current_threshold": node["threshold"],
        }
        for node in nodes
        if node["can_prune"]
    ]
    feature_universe = source_asset["tree_result"]["training"]["feature_order"]
    assert item["pointers"]["feature_universe"] == feature_universe
    eligible_feature_replacements = item["pointers"][
        "eligible_feature_replacements"
    ]
    assert all(
        set(pointer) == FEATURE_REPLACEMENT_FIELDS
        for pointer in eligible_feature_replacements
    )
    assert eligible_feature_replacements == [
        {
            "source_tree_id": source_tree_id,
            "node_id": node["node_id"],
            "operation": "replace_split_feature",
            "current_feature": node["feature"],
            "current_threshold": node["threshold"],
        }
        for node in nodes
        if (
            node["can_prune"]
            and any(feature != node["feature"] for feature in feature_universe)
        )
    ]


def _current_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            str(key)
            for key in value
            if str(key).startswith("current_")
        } | set().union(
            *(_current_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_current_keys(child) for child in value), set())
    return set()


def _hash_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            str(key)
            for key in value
            if str(key).endswith("_hash")
        } | set().union(
            *(_hash_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_hash_keys(child) for child in value), set())
    return set()


def test_candidate_lab_projects_each_tree_source_without_inventing_one_current_branch(
    scenario,
) -> None:
    client = TestClient(create_app(scenario.settings))
    asset = scenario.source_asset
    asset_id = asset["asset_id"]
    tree = asset["tree_result"]["tree"]
    base_visible = [node["node_id"] for node in tree["nodes"]]
    base_frontier = list(tree["leaf_ids"])

    before = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert before.status_code == 200, before.text
    before_projection = before.json()
    base_before = _automatic_item(before_projection, asset_id)
    _assert_operable_topology(
        base_before,
        source_tree_id=asset_id,
        source_asset=asset,
        visible_node_ids=base_visible,
        frontier_node_ids=base_frontier,
    )
    assert before_projection["candidates"]["interactive_tree_revision"] == {
        "latest": None,
        "all": [],
        "total": 0,
        "truncated": False,
    }

    split_ids = [
        node["node_id"] for node in tree["nodes"] if node["kind"] == "split"
    ]
    first = _revise(
        scenario,
        source_tree_id=asset_id,
        node_id=split_ids[-1],
        reason="Prune the deepest unstable branch.",
    )
    second = _revise(
        scenario,
        source_tree_id=first["revision_id"],
        node_id=split_ids[-2],
        reason="Broaden the first branch.",
    )
    sibling = _revise(
        scenario,
        source_tree_id=asset_id,
        node_id=tree["root_node_id"],
        reason="Independent root-level branch.",
    )
    expected = {}
    for result in (first, second, sibling):
        record, payload = _revision_record_and_payload(scenario, result)
        expected[result["revision_id"]] = (result, record, payload)

    response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    projection = response.json()
    base_after = _automatic_item(projection, asset_id)
    _assert_operable_topology(
        base_after,
        source_tree_id=asset_id,
        source_asset=asset,
        visible_node_ids=base_visible,
        frontier_node_ids=base_frontier,
    )

    revisions = projection["candidates"]["interactive_tree_revision"]
    assert revisions["total"] == 3
    assert revisions["truncated"] is False
    assert 0 < len(revisions["all"]) <= 20
    assert revisions["latest"] in revisions["all"]
    by_id = {
        item["detail"]["revision_id"]: item
        for item in revisions["all"]
    }
    assert set(by_id) == set(expected)

    for revision_id, (result, record, payload) in expected.items():
        item = by_id[revision_id]
        parent = payload["parent_revision"]
        assert item["kind"] == "interactive_tree_revision"
        assert item["candidate_id"] == payload["candidate_evidence"]["candidate_id"]
        assert item["artifact"]["artifact_id"] == record["id"]
        assert item["detail"]["revision_id"] == revision_id
        assert item["detail"]["source_tree_id"] == revision_id
        assert item["detail"]["derived_from_source_tree_id"] == result[
            "source_tree_id"
        ]
        assert item["detail"]["parent_revision_id"] == (
            None if parent is None else parent["revision_id"]
        )
        assert item["detail"]["base_asset_id"] == asset_id
        assert item["detail"]["edit"] == payload["edit"]
        assert item["pointers"]["frontier_node_ids"] == payload["tree"][
            "frontier_node_ids"
        ]
        assert item["pointers"]["visible_node_ids"] == payload["tree"][
            "visible_node_ids"
        ]
        _assert_operable_topology(
            item,
            source_tree_id=revision_id,
            source_asset=asset,
            visible_node_ids=payload["tree"]["visible_node_ids"],
            frontier_node_ids=payload["tree"]["frontier_node_ids"],
        )

    assert by_id[first["revision_id"]]["detail"]["parent_revision_id"] is None
    assert (
        by_id[second["revision_id"]]["detail"]["parent_revision_id"]
        == first["revision_id"]
    )
    assert by_id[sibling["revision_id"]]["detail"]["parent_revision_id"] is None
    assert (
        by_id[second["revision_id"]]["pointers"]["visible_node_ids"]
        != by_id[sibling["revision_id"]]["pointers"]["visible_node_ids"]
    )
    assert _current_keys(base_after) == {
        "current_feature",
        "current_threshold",
    }
    assert _current_keys(revisions) == {
        "current_feature",
        "current_threshold",
    }
    assert _hash_keys(base_after) == set()
    assert _hash_keys(revisions) == set()


def test_candidate_lab_projects_effective_threshold_revision_as_new_immutable_source(
    scenario,
) -> None:
    client = TestClient(create_app(scenario.settings))
    asset = scenario.source_asset
    asset_id = asset["asset_id"]
    root_id = asset["tree_result"]["tree"]["root_node_id"]

    before_response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )
    assert before_response.status_code == 200, before_response.text
    base_before = _automatic_item(before_response.json(), asset_id)
    base_adjustments_before = base_before["pointers"][
        "eligible_threshold_adjustments"
    ]
    base_root_before = next(
        pointer
        for pointer in base_adjustments_before
        if pointer["node_id"] == root_id
    )

    adjusted = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": asset_id,
            "node_id": root_id,
            "operation": "adjust_split_threshold",
            "threshold": 1.5,
            "reason": "Review the effective root split.",
        },
        scenario.ctx,
    )

    after_response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )
    assert after_response.status_code == 200, after_response.text
    projection = after_response.json()
    base_after = _automatic_item(projection, asset_id)
    assert (
        base_after["pointers"]["eligible_threshold_adjustments"]
        == base_adjustments_before
    )

    revision = next(
        item
        for item in projection["candidates"]["interactive_tree_revision"]["all"]
        if item["detail"]["revision_id"] == adjusted["revision_id"]
    )
    assert revision["detail"]["source_tree_id"] == adjusted["revision_id"]
    assert revision["detail"]["derived_from_source_tree_id"] == asset_id
    revision_adjustments = revision["pointers"][
        "eligible_threshold_adjustments"
    ]
    assert all(
        set(pointer) == THRESHOLD_ADJUSTMENT_FIELDS
        for pointer in revision_adjustments
    )
    revised_root = next(
        pointer
        for pointer in revision_adjustments
        if pointer["node_id"] == root_id
    )
    assert base_root_before["current_threshold"] == 0.5
    assert revised_root == {
        "source_tree_id": adjusted["revision_id"],
        "node_id": root_id,
        "operation": "adjust_split_threshold",
        "feature": base_root_before["feature"],
        "current_threshold": 1.5,
    }
    assert _current_keys(revision) == {
        "current_feature",
        "current_threshold",
    }
    assert _hash_keys(revision) == set()


def test_candidate_lab_replays_interactive_frontier_pool_sources(
    scenario,
) -> None:
    selection, revision = _materialize_frontier(scenario)
    added = strategy_tools.tool_add_candidate_to_pool(
        _interactive_pool_add_inputs(selection),
        scenario.ctx,
    )
    client = TestClient(create_app(scenario.settings))

    response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    projection = response.json()
    assert projection["pools"]["total"] == 1
    pool = projection["pools"]["latest"]
    assert pool["revision"] == added["revision"]
    [entry] = pool["entries"]
    fragment = next(
        item
        for item in revision["fragments"]
        if item["source_node_id"] == selection["source_node_id"]
    )
    assert entry["source"] == {
        "asset_id": revision["semantic_tree_id"],
        "asset_type": revision["asset_type"],
        "fragment_id": fragment["fragment_id"],
        "fragment_type": "strategy_rule",
        "effect_id": fragment["effect_id"],
        "evidence_id": revision["candidate_evidence"]["candidate_id"],
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }
    assert entry["execution"]["condition"] == fragment["condition"]
    assert _hash_keys(projection) == set()


def test_candidate_lab_frontier_pool_source_tampering_fails_closed(
    scenario,
) -> None:
    selection, _revision = _materialize_frontier(scenario)
    strategy_tools.tool_add_candidate_to_pool(
        _interactive_pool_add_inputs(selection),
        scenario.ctx,
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        selection["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    Path(record["path"]).write_text('{"forged":true}', encoding="utf-8")
    client = TestClient(create_app(scenario.settings))

    response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


@pytest.mark.parametrize("tamper", ["bytes", "provenance"])
def test_candidate_lab_interactive_revision_tampering_fails_closed(
    scenario,
    tamper: str,
) -> None:
    result = _revise(
        scenario,
        source_tree_id=scenario.source_asset["asset_id"],
        node_id=scenario.source_asset["tree_result"]["tree"]["root_node_id"],
        reason="Create evidence for fail-closed projection.",
    )
    record, _payload = _revision_record_and_payload(scenario, result)
    if tamper == "bytes":
        Path(record["path"]).write_text('{"forged":true}', encoding="utf-8")
    else:
        provenance = dict(record["provenance"])
        provenance["source_tree_id"] = "candidate-asset-" + "f" * 32
        with connect(scenario.settings.db_path) as conn:
            conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
            conn.execute(
                "UPDATE task_artifacts SET provenance_json = ? WHERE id = ?",
                (
                    json.dumps(
                        provenance,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    record["id"],
                ),
            )
    client = TestClient(create_app(scenario.settings))

    response = client.get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_rejects_foreign_registered_interactive_revision(
    scenario,
) -> None:
    result = _revise(
        scenario,
        source_tree_id=scenario.source_asset["asset_id"],
        node_id=scenario.source_asset["tree_result"]["tree"]["root_node_id"],
        reason="Task-owned revision.",
    )
    record, _payload = _revision_record_and_payload(scenario, result)
    scenario.repository.register(
        task_id=scenario.foreign_task.id,
        kind=INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        path=record["path"],
        content_hash=record["content_hash"],
        origin_tool=INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
        provenance=record["provenance"],
    )
    client = TestClient(create_app(scenario.settings))

    response = client.get(
        f"/api/tasks/{scenario.foreign_task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.packs.strategy import pool_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from tests.test_strategy_pool_interactive_tree_frontier import (
    _action,
    _add_inputs,
    _materialize_from_revision,
)
from tests.test_strategy_interactive_tree_frontier_tool import _revision


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def _materialize_group(
    scenario,
    revision: dict,
    indexes: list[int],
    *,
    reason: str | None = None,
) -> dict:
    inputs = {
        "revision_id": revision["revision_id"],
        "source_node_ids": [
            revision["tree"]["frontier_node_ids"][index]
            for index in indexes
        ],
    }
    if reason is not None:
        inputs["selection_reason"] = reason
    return (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            inputs,
            scenario.ctx,
        )
    )


def _revision_payload(scenario) -> dict:
    _result, revision = _revision(scenario)
    return revision


def test_frontier_group_adds_and_compiles_one_canonical_or_rule(
    scenario,
) -> None:
    revision = _revision_payload(scenario)
    selection = _materialize_group(
        scenario,
        revision,
        [1, 0],
        reason="Combine adjacent reviewed branches.",
    )
    action = _action("review", reason="GROUP_REVIEW")

    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(selection, action=action),
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

    [entry] = added["entries"]
    by_node_id = {
        item["source_node_id"]: item
        for item in revision["fragments"]
    }
    expected_condition = {
        "op": "or",
        "args": [
            by_node_id[node_id]["condition"]
            for node_id in selection["source_node_ids"]
        ],
    }
    assert entry["source"]["artifact_id"] == selection["artifacts"][0][
        "artifact_id"
    ]
    assert entry["source"]["fragment_id"] == selection["fragment_id"]
    assert entry["source"]["effect_id"] == selection["effect_id"]
    assert entry["execution"] == {
        "condition": expected_condition,
        "requirements": [],
    }
    [rule] = compiled["strategy_spec"]["rules"]
    assert rule["condition"] == expected_condition
    assert rule["action"] == action


def test_same_tree_rejects_singleton_group_overlap_but_allows_disjoint(
    scenario,
) -> None:
    revision = _revision_payload(scenario)
    assert len(revision["tree"]["frontier_node_ids"]) >= 3
    singleton = _materialize_from_revision(
        scenario,
        revision,
        fragment_index=0,
        reason="Singleton review.",
    )
    overlapping = _materialize_group(
        scenario,
        revision,
        [0, 1],
        reason="Overlaps singleton.",
    )
    disjoint = _materialize_group(
        scenario,
        revision,
        [1, 2],
        reason="Disjoint group.",
    )
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(singleton),
        scenario.ctx,
    )

    with pytest.raises(StrategyError, match="overlap|frontier"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                overlapping,
                revision=first["revision"],
                snapshot_hash=first["snapshot_hash"],
                action=_action("review", reason="OVERLAP"),
            ),
            scenario.ctx,
        )

    second = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            disjoint,
            revision=first["revision"],
            snapshot_hash=first["snapshot_hash"],
            action=_action("review", reason="DISJOINT"),
        ),
        scenario.ctx,
    )
    assert len(second["entries"]) == 2


def test_same_tree_rejects_group_group_overlap(
    scenario,
) -> None:
    revision = _revision_payload(scenario)
    assert len(revision["tree"]["frontier_node_ids"]) >= 3
    first_group = _materialize_group(scenario, revision, [0, 1])
    overlap_group = _materialize_group(scenario, revision, [1, 2])
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(first_group),
        scenario.ctx,
    )

    with pytest.raises(StrategyError, match="overlap|frontier"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                overlap_group,
                revision=first["revision"],
                snapshot_hash=first["snapshot_hash"],
            ),
            scenario.ctx,
        )


def test_group_lineage_is_reauthenticated_under_pool_writer_lock(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = _revision_payload(scenario)
    group = _materialize_group(scenario, revision, [0, 1])
    record = scenario.repository.get_for_task(
        scenario.task.id,
        group["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    original = pool_tools._require_lineage_on_connection
    changed = False

    def drift_then_verify(conn, lineage, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            path = Path(record["path"])
            path.write_bytes(path.read_bytes() + b"\n")
        return original(conn, lineage, **kwargs)

    monkeypatch.setattr(
        pool_tools,
        "_require_lineage_on_connection",
        drift_then_verify,
    )
    with pytest.raises(StrategyError, match="changed|hash|canonical"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(group),
            scenario.ctx,
        )
    assert (
        StrategyCandidatePoolRepository(
            scenario.settings.db_path
        ).get_current(scenario.task.id, "approval")
        is None
    )


def test_candidate_lab_replays_frontier_group_pool_source(scenario) -> None:
    revision = _revision_payload(scenario)
    group = _materialize_group(scenario, revision, [0, 1])
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(group),
        scenario.ctx,
    )
    response = TestClient(create_app(scenario.settings)).get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    pool = response.json()["pools"]["latest"]
    assert pool["revision"] == added["revision"]
    [entry] = pool["entries"]
    assert entry["source"]["fragment_id"] == group["fragment_id"]
    assert entry["source"]["effect_id"] == group["effect_id"]
    by_node_id = {
        item["source_node_id"]: item
        for item in revision["fragments"]
    }
    assert entry["execution"]["condition"] == {
        "op": "or",
        "args": [
            by_node_id[node_id]["condition"]
            for node_id in group["source_node_ids"]
        ],
    }


def test_candidate_lab_group_source_tampering_fails_closed(
    scenario,
) -> None:
    revision = _revision_payload(scenario)
    group = _materialize_group(scenario, revision, [0, 1])
    strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(group),
        scenario.ctx,
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        group["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    Path(record["path"]).write_text(
        json.dumps({"forged": True}),
        encoding="utf-8",
    )

    response = TestClient(create_app(scenario.settings)).get(
        f"/api/tasks/{scenario.task.id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )

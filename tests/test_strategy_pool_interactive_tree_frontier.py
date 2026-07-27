from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from marvis.db_schema import connect
from marvis.packs.strategy import interactive_tree_tools as revision_tools
from marvis.packs.strategy import pool_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
)
from marvis.packs.strategy.interactive_tree_tools import (
    INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
)
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.repositories.strategy_pool import (
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _materialize_frontier(
    scenario,
    *,
    fragment_index: int = 0,
    reason: str | None = None,
) -> tuple[dict, dict]:
    split_id = next(
        node["node_id"]
        for node in reversed(scenario.source_asset["tree_result"]["tree"]["nodes"])
        if node["kind"] == "split"
    )
    revision_result = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": split_id,
            "operation": "prune_subtree",
            "reason": "Create a Pool-ready reviewed frontier.",
        },
        scenario.ctx,
    )
    revision_record = scenario.repository.get_for_task(
        scenario.task.id,
        revision_result["artifacts"][0]["artifact_id"],
    )
    assert revision_record is not None
    revision = json.loads(Path(revision_record["path"]).read_text("utf-8"))
    selection = _materialize_from_revision(
        scenario,
        revision,
        fragment_index=fragment_index,
        reason=reason,
    )
    return selection, revision


def _materialize_from_revision(
    scenario,
    revision: dict,
    *,
    fragment_index: int = 0,
    reason: str | None = None,
) -> dict:
    fragment = revision["fragments"][fragment_index]
    inputs = {
        "revision_id": revision["revision_id"],
        "source_node_id": fragment["source_node_id"],
    }
    if reason is not None:
        inputs["selection_reason"] = reason
    return strategy_tools.tool_materialize_interactive_tree_frontier_selection(
        inputs,
        scenario.ctx,
    )


def _materialize_child_revision_frontier(
    scenario,
) -> tuple[dict, dict, dict]:
    split_ids = [
        node["node_id"]
        for node in scenario.source_asset["tree_result"]["tree"]["nodes"]
        if node["kind"] == "split"
    ]
    parent_result = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": split_ids[-1],
            "operation": "prune_subtree",
            "reason": "First reviewed prune.",
        },
        scenario.ctx,
    )
    child_result = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": parent_result["revision_id"],
            "node_id": split_ids[-2],
            "operation": "prune_subtree",
            "reason": "Second reviewed prune.",
        },
        scenario.ctx,
    )
    records = {
        record["provenance"]["revision_id"]: record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
    }
    child = json.loads(
        Path(records[child_result["revision_id"]]["path"]).read_text("utf-8")
    )
    selection = _materialize_from_revision(scenario, child)
    return selection, child, records[parent_result["revision_id"]]


def _add_inputs(
    selection: dict,
    *,
    revision: int = 0,
    snapshot_hash: str = ABSENT_POOL_SNAPSHOT_HASH,
    action: dict | None = None,
) -> dict:
    artifact = selection["artifacts"][0]
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": selection["semantic_tree_id"],
        "expected_asset_hash": selection["tree_hash"],
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": action or _action("reject", reason="INTERACTIVE_TREE_RISK"),
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": snapshot_hash,
    }


def test_interactive_tree_frontier_selection_adds_and_compiles_exact_revision_rule(
    scenario,
) -> None:
    selection, revision = _materialize_frontier(scenario)
    fragment = next(
        item
        for item in revision["fragments"]
        if item["source_node_id"] == selection["source_node_id"]
    )
    pool_action = _action("review", reason="POLICY_OWNER_REVIEW")

    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(selection, action=pool_action),
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
    assert entry["source"]["artifact_id"] == selection["artifacts"][0]["artifact_id"]
    assert entry["source"]["asset_id"] == revision["semantic_tree_id"]
    assert entry["source"]["asset_hash"] == revision["tree"]["tree_hash"]
    assert entry["source"]["fragment_id"] == fragment["fragment_id"]
    assert entry["source"]["effect_id"] == fragment["effect_id"]
    assert entry["execution"] == {
        "condition": fragment["condition"],
        "requirements": fragment["requirements"],
    }
    assert entry["action"] == pool_action
    [compiled_rule] = compiled["strategy_spec"]["rules"]
    assert compiled_rule["condition"] == fragment["condition"]
    assert compiled_rule["action"] == pool_action

    runtime = strategy_tools._runtime(scenario.ctx)
    pool_binding = pool_tools.load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=scenario.task.id,
        strategy_type="approval",
        expected_pool_revision=added["revision"],
        expected_pool_snapshot_hash=added["snapshot_hash"],
    )
    development = pool_tools.bind_strategy_pool_development_execution(
        runtime,
        pool_binding,
    )
    assert development.dataset.dataset_id == scenario.dataset.id
    assert development.evidence_identity == entry["source"]["evidence_identity"]
    assert development.sample_design.reference.to_ref_dict() == (
        scenario.sample_design_ref
    )


@pytest.mark.parametrize(
    "field",
    ["condition", "requirements", "metrics", "action_override", "fragment"],
)
def test_pool_rejects_caller_derived_frontier_evidence(
    scenario,
    field: str,
) -> None:
    selection, _revision = _materialize_frontier(scenario)

    with pytest.raises(StrategyError, match="unsupported"):
        strategy_tools.tool_add_candidate_to_pool(
            {
                **_add_inputs(selection),
                field: {"forged": True},
            },
            scenario.ctx,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "expected_asset_id",
            "interactive-tree-" + "e" * 32,
        ),
        ("expected_asset_hash", "e" * 64),
        ("expected_artifact_content_hash", "e" * 64),
    ],
)
def test_pool_rejects_selection_or_semantic_revision_identity_mismatch(
    scenario,
    field: str,
    value: str,
) -> None:
    selection, _revision = _materialize_frontier(scenario)

    with pytest.raises(StrategyError):
        strategy_tools.tool_add_candidate_to_pool(
            {
                **_add_inputs(selection),
                field: value,
            },
            scenario.ctx,
        )


def test_complete_interactive_tree_revision_cannot_enter_pool_directly(
    scenario,
) -> None:
    _selection, revision = _materialize_frontier(scenario)
    revision_record = next(
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
        and record["provenance"]["revision_id"] == revision["revision_id"]
    )
    inputs = {
        **_add_inputs(
            {
                "artifacts": [
                    {
                        "artifact_id": revision_record["id"],
                        "content_hash": revision_record["content_hash"],
                    }
                ],
                "semantic_tree_id": revision["semantic_tree_id"],
                "tree_hash": revision["tree"]["tree_hash"],
            }
        ),
    }

    with pytest.raises(
        StrategyError,
        match="complete interactive-tree revision.*frontier selection",
    ):
        strategy_tools.tool_add_candidate_to_pool(inputs, scenario.ctx)


def test_pool_caps_aggregate_interactive_revision_reads(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _revision = _materialize_frontier(scenario)
    monkeypatch.setattr(
        pool_tools,
        "_MAX_INTERACTIVE_TREE_LINEAGE_BYTES",
        1,
    )

    with pytest.raises(StrategyError, match="byte|budget|exceed"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(selection),
            scenario.ctx,
        )


def _update_artifact_row(
    scenario,
    artifact_id: str,
    **changes: object,
) -> None:
    assignments = ", ".join(f"{field} = ?" for field in changes)
    with connect(scenario.settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            f"UPDATE task_artifacts SET {assignments} WHERE id = ?",  # noqa: S608
            (*changes.values(), artifact_id),
        )


@pytest.mark.parametrize("mutation", ["kind", "origin", "schema"])
def test_frontier_selection_artifact_triple_is_authenticated(
    scenario,
    mutation: str,
) -> None:
    selection, _revision = _materialize_frontier(scenario)
    artifact_id = selection["artifacts"][0]["artifact_id"]
    record = scenario.repository.get_for_task(scenario.task.id, artifact_id)
    assert record is not None
    assert record["kind"] == INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND
    if mutation == "kind":
        changes = {"kind": "forged"}
    elif mutation == "origin":
        changes = {"origin_tool": "forged"}
    else:
        provenance = deepcopy(record["provenance"])
        provenance["schema_version"] = "forged"
        changes = {
            "provenance_json": json.dumps(
                provenance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        }
    _update_artifact_row(scenario, artifact_id, **changes)

    with pytest.raises(StrategyError, match="unsupported|changed|invalid"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(selection),
            scenario.ctx,
        )


@pytest.mark.parametrize("drift", ["selection", "revision", "base", "dataset"])
def test_frontier_lineage_is_reauthenticated_under_pool_writer_lock(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    selection, revision = _materialize_frontier(scenario)
    selection_record = scenario.repository.get_for_task(
        scenario.task.id,
        selection["artifacts"][0]["artifact_id"],
    )
    revision_record = next(
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
        and record["provenance"]["revision_id"] == revision["revision_id"]
    )
    assert selection_record is not None
    paths = {
        "selection": Path(selection_record["path"]),
        "revision": Path(revision_record["path"]),
        "base": Path(scenario.source_record["path"]),
    }
    original = pool_tools._require_lineage_on_connection
    changed = False

    def drift_then_verify(conn, lineage, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            if drift == "dataset":
                conn.execute(
                    "UPDATE datasets SET row_count = row_count + 1 WHERE id = ?",
                    (scenario.dataset.id,),
                )
            else:
                path = paths[drift]
                path.write_bytes(path.read_bytes() + b"\n")
        return original(conn, lineage, **kwargs)

    monkeypatch.setattr(
        pool_tools,
        "_require_lineage_on_connection",
        drift_then_verify,
    )
    with pytest.raises(
        StrategyError,
        match="changed|canonical|content hash|drift",
    ):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(selection),
            scenario.ctx,
        )

    assert (
        StrategyCandidatePoolRepository(
            scenario.settings.db_path
        ).get_current(scenario.task.id, "approval")
        is None
    )
    assert not [
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"] == POOL_ARTIFACT_KIND
    ]


def test_same_semantic_revision_accepts_distinct_frontiers_but_not_reason_alias(
    scenario,
) -> None:
    first_selection, revision = _materialize_frontier(
        scenario,
        reason="First review.",
    )
    assert len(revision["fragments"]) >= 2
    second_selection = _materialize_from_revision(
        scenario,
        revision,
        fragment_index=1,
        reason="Second frontier.",
    )
    alias_selection = _materialize_from_revision(
        scenario,
        revision,
        reason="Same frontier, another audit reason.",
    )

    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(first_selection),
        scenario.ctx,
    )
    second = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            second_selection,
            revision=first["revision"],
            snapshot_hash=first["snapshot_hash"],
            action=_action("review", reason="SECOND_FRONTIER"),
        ),
        scenario.ctx,
    )

    assert len(second["entries"]) == 2
    assert len({entry["source"]["asset_id"] for entry in second["entries"]}) == 1
    assert len({entry["source"]["fragment_id"] for entry in second["entries"]}) == 2
    with pytest.raises(StrategyError, match="duplicate asset fragment"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                alias_selection,
                revision=second["revision"],
                snapshot_hash=second["snapshot_hash"],
            ),
            scenario.ctx,
        )


def test_pool_reuses_one_automatic_source_read_per_verification_phase(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_selection, revision = _materialize_frontier(
        scenario,
        reason="First reviewed frontier.",
    )
    second_selection = _materialize_from_revision(
        scenario,
        revision,
        fragment_index=1,
        reason="Second reviewed frontier.",
    )
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(first_selection),
        scenario.ctx,
    )
    original = (
        revision_tools.load_verified_automatic_tree_source_artifact_on_connection
    )
    source_reads = 0

    def count_source_read(*args, **kwargs):
        nonlocal source_reads
        source_reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        revision_tools,
        "load_verified_automatic_tree_source_artifact_on_connection",
        count_source_read,
    )

    strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            second_selection,
            revision=first["revision"],
            snapshot_hash=first["snapshot_hash"],
            action=_action("review", reason="SECOND_FRONTIER"),
        ),
        scenario.ctx,
    )

    assert source_reads == 2


def test_parent_revision_is_reauthenticated_under_pool_writer_lock(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, _child, parent_record = _materialize_child_revision_frontier(
        scenario
    )
    original = pool_tools._require_lineage_on_connection
    changed = False

    def drift_parent_then_verify(conn, lineage, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            path = Path(parent_record["path"])
            path.write_bytes(path.read_bytes() + b"\n")
        return original(conn, lineage, **kwargs)

    monkeypatch.setattr(
        pool_tools,
        "_require_lineage_on_connection",
        drift_parent_then_verify,
    )
    with pytest.raises(StrategyError, match="content hash|canonical|changed"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(selection),
            scenario.ctx,
        )

    assert (
        StrategyCandidatePoolRepository(
            scenario.settings.db_path
        ).get_current(scenario.task.id, "approval")
        is None
    )

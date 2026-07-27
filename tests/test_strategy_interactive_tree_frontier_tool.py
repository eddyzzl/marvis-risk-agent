"""Governed persistence for explicit interactive-tree frontier selections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from marvis.packs.strategy import (
    interactive_tree_frontier_tools as frontier_tools,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    canonical_interactive_tree_frontier_selection_json,
    interactive_tree_frontier_selection_to_verified_candidate_fragment,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


TOOL_SCHEMA_VERSION = (
    "strategy.materialize-interactive-tree-frontier-selection-tool.v1"
)
SELECTION_DIRECTORY = "strategy_interactive_tree_frontier_selections"


def _revision(scenario) -> tuple[dict, dict]:
    split_id = next(
        node["node_id"]
        for node in reversed(scenario.source_asset["tree_result"]["tree"]["nodes"])
        if node["kind"] == "split"
    )
    result = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": split_id,
            "operation": "prune_subtree",
            "reason": "Create a reviewed frontier.",
        },
        scenario.ctx,
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    return result, json.loads(Path(record["path"]).read_text(encoding="utf-8"))


def test_materialize_frontier_selection_is_pointer_only_task_local_and_idempotent(
    scenario,
) -> None:
    revision_result, revision = _revision(scenario)
    fragment = revision["fragments"][0]
    inputs = {
        "revision_id": revision_result["revision_id"],
        "source_node_id": fragment["source_node_id"],
        "selection_reason": "  Reviewed by the policy owner.  ",
    }

    first = strategy_tools.tool_materialize_interactive_tree_frontier_selection(
        inputs,
        scenario.ctx,
    )
    repeated = strategy_tools.tool_materialize_interactive_tree_frontier_selection(
        inputs,
        scenario.ctx,
    )

    assert repeated == first
    assert first["schema_version"] == TOOL_SCHEMA_VERSION
    assert first["selection_id"].startswith(
        "interactive-tree-frontier-selection-"
    )
    assert len(first["selection_hash"]) == 64
    assert first["selection_reason"] == "Reviewed by the policy owner."
    assert first["revision_id"] == revision["revision_id"]
    assert first["semantic_tree_id"] == revision["semantic_tree_id"]
    assert first["tree_hash"] == revision["tree"]["tree_hash"]
    for field in (
        "source_node_id",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
    ):
        assert first[field] == fragment[field]
    assert len(first["artifacts"]) == 1
    assert "path" not in _recursive_keys(first)

    artifact = first["artifacts"][0]
    record = scenario.repository.get_for_task(
        scenario.task.id,
        artifact["artifact_id"],
    )
    assert record is not None
    expected_path = (
        Path(scenario.settings.tasks_dir)
        / scenario.task.id
        / SELECTION_DIRECTORY
        / f"{first['selection_id']}.json"
    )
    assert Path(record["path"]) == expected_path
    assert record["kind"] == INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND
    assert record["origin_tool"] == INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL
    assert record["content_hash"] == artifact["content_hash"]
    selection = json.loads(expected_path.read_text(encoding="utf-8"))
    assert selection["revision"]["revision_id"] == revision["revision_id"]
    assert selection["frontier"]["source_node_id"] == fragment["source_node_id"]
    assert {
        "condition",
        "requirements",
        "metrics",
        "action",
    }.isdisjoint(_recursive_keys(selection))
    assert expected_path.read_text(encoding="utf-8") == (
        canonical_interactive_tree_frontier_selection_json(selection)
    )
    assert record["provenance"] == (
        frontier_tools.interactive_tree_frontier_selection_provenance(
            selection
        )
    )


def test_verified_selection_reloads_revision_and_derives_exact_pool_fragment(
    scenario,
) -> None:
    result, selection, record = _materialized_selection(scenario)
    verified = (
        frontier_tools.load_verified_interactive_tree_frontier_selection_artifact(
            strategy_tools._runtime(scenario.ctx),
            task_id=scenario.task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )
    )

    fragment = (
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            verified.selection,
            verified.revision.revision,
            verified.revision.automatic_source.asset,
            selection_artifact_binding=verified.artifact_binding(),
            revision_artifact_binding=verified.revision.builder_binding(),
            parent_revision=(
                verified.revision.ancestor_revisions[0]
                if verified.revision.ancestor_revisions
                else None
            ),
            ancestor_revisions=verified.revision.ancestor_revisions[1:],
        )
    )

    assert fragment["asset"]["asset_id"] == result["semantic_tree_id"]
    assert fragment["asset"]["asset_hash"] == result["tree_hash"]
    assert fragment["fragment"]["fragment_id"] == result["fragment_id"]
    assert (
        verified.selection["frontier"]["fragment_hash"]
        == result["fragment_hash"]
    )
    assert len(fragment["fragment_hash"]) == 64
    assert fragment["artifact"]["artifact_id"] == record["id"]
    assert fragment["fragment"]["condition"]
    assert fragment["evidence"]["identity"]


@pytest.mark.parametrize(
    "field",
    [
        "condition",
        "requirements",
        "metrics",
        "action",
        "fragment",
        "selection_id",
        "selection_hash",
        "expected_asset_id",
        "expected_artifact_content_hash",
    ],
)
def test_materialize_input_rejects_caller_owned_derived_or_platform_fields(
    scenario,
    field: str,
) -> None:
    revision_result, revision = _revision(scenario)
    with pytest.raises(StrategyError, match="unsupported|invalid"):
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_id": revision["fragments"][0]["source_node_id"],
                field: "forged",
            },
            scenario.ctx,
        )


def test_materialize_rejects_a_node_outside_the_revision_frontier(
    scenario,
) -> None:
    revision_result, revision = _revision(scenario)
    frontier_ids = set(revision["tree"]["frontier_node_ids"])
    hidden_node_id = next(
        node["node_id"]
        for node in scenario.source_asset["tree_result"]["tree"]["nodes"]
        if node["node_id"] not in frontier_ids
    )

    with pytest.raises(StrategyError, match="frontier|invalid"):
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_id": hidden_node_id,
            },
            scenario.ctx,
        )


def test_materialize_caps_total_revision_chain_bytes(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_result, revision = _revision(scenario)
    monkeypatch.setattr(
        frontier_tools,
        "MAX_INTERACTIVE_TREE_FRONTIER_REVISION_BYTES",
        1,
    )

    with pytest.raises(StrategyError, match="byte|budget|exceed"):
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_id": revision["fragments"][0]["source_node_id"],
            },
            scenario.ctx,
        )


def test_selection_loader_rejects_file_and_registry_provenance_drift(
    scenario,
) -> None:
    result, _selection, record = _materialized_selection(scenario)
    runtime = strategy_tools._runtime(scenario.ctx)
    path = Path(record["path"])
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(StrategyError, match="hash|changed"):
        frontier_tools.load_verified_interactive_tree_frontier_selection_artifact(
            runtime,
            task_id=scenario.task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )

    path.write_bytes(original)
    with scenario.repository.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        stored = conn.execute(
            "SELECT provenance_json FROM task_artifacts WHERE id = ?",
            (record["id"],),
        ).fetchone()
        provenance = json.loads(stored["provenance_json"])
        provenance["fragment_hash"] = hashlib.sha256(b"forged").hexdigest()
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
        conn.commit()
    with pytest.raises(StrategyError, match="provenance|changed"):
        frontier_tools.load_verified_interactive_tree_frontier_selection_artifact(
            runtime,
            task_id=scenario.task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )


def test_selection_loader_rejects_cross_task_reuse(scenario) -> None:
    result, _selection, record = _materialized_selection(scenario)

    with pytest.raises(StrategyError, match="not found"):
        frontier_tools.load_verified_interactive_tree_frontier_selection_artifact(
            strategy_tools._runtime(scenario.ctx),
            task_id=scenario.foreign_task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )


def test_selection_loader_rejects_registry_primary_key_alias(
    scenario,
) -> None:
    result, _selection, record = _materialized_selection(scenario)
    alias = "a" * 64
    with scenario.repository.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET id = ? WHERE id = ?",
            (alias, record["id"]),
        )
        conn.commit()

    with pytest.raises(StrategyError, match="stable identity"):
        frontier_tools.load_verified_interactive_tree_frontier_selection_artifact(
            strategy_tools._runtime(scenario.ctx),
            task_id=scenario.task.id,
            artifact_id=alias,
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )


def test_materialize_idempotent_replay_rejects_registry_primary_key_alias(
    scenario,
) -> None:
    result, selection, record = _materialized_selection(scenario)
    alias = "a" * 64
    with scenario.repository.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET id = ? WHERE id = ?",
            (alias, record["id"]),
        )
        conn.commit()

    with pytest.raises(StrategyError, match="stable identity"):
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": result["revision_id"],
                "source_node_id": selection["frontier"]["source_node_id"],
                "selection_reason": selection["selection_reason"],
            },
            scenario.ctx,
        )


def test_registration_failure_rolls_back_selection_file_and_registry(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_result, revision = _revision(scenario)
    original = TaskArtifactRepository.register_on_connection

    def fail_selection_registration(self, conn, **kwargs):
        if (
            kwargs.get("kind")
            == INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND
        ):
            raise RuntimeError("injected frontier selection registration failure")
        return original(self, conn, **kwargs)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_selection_registration,
    )
    with pytest.raises(
        RuntimeError,
        match="injected frontier selection registration failure",
    ):
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_id": revision["fragments"][0]["source_node_id"],
            },
            scenario.ctx,
        )

    assert not [
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"]
        == INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND
    ]
    assert not list(
        (
            Path(scenario.settings.tasks_dir)
            / scenario.task.id
            / SELECTION_DIRECTORY
        ).glob("*.json")
    )


def _materialized_selection(scenario) -> tuple[dict, dict, dict]:
    revision_result, revision = _revision(scenario)
    result = strategy_tools.tool_materialize_interactive_tree_frontier_selection(
        {
            "revision_id": revision_result["revision_id"],
            "source_node_id": revision["fragments"][0]["source_node_id"],
            "selection_reason": "Ready for governed Pool admission.",
        },
        scenario.ctx,
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    selection = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    return result, selection, record


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(_recursive_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()

"""Governed persistence for explicit interactive-tree frontier OR groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from marvis.packs.strategy import (
    interactive_tree_frontier_group_tools as group_tools,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_frontier_group_selection import (
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
    canonical_interactive_tree_frontier_group_selection_json,
    interactive_tree_frontier_group_selection_to_verified_candidate_fragment,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_interactive_tree_frontier_tool import (
    _recursive_keys,
    _revision,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


TOOL_SCHEMA_VERSION = (
    "strategy.materialize-interactive-tree-frontier-group-selection-tool.v1"
)
SELECTION_DIRECTORY = (
    "strategy_interactive_tree_frontier_group_selections"
)
OUTPUT_FIELDS = {
    "schema_version",
    "selection_id",
    "selection_hash",
    "group_id",
    "selection_reason",
    "revision_id",
    "semantic_tree_id",
    "tree_hash",
    "source_node_ids",
    "member_count",
    "fragment_id",
    "rule_id",
    "effect_id",
    "artifacts",
}


def _materialized_group(scenario) -> tuple[dict, dict, dict, dict]:
    revision_result, revision = _revision(scenario)
    source_node_ids = revision["tree"]["frontier_node_ids"][:2]
    result = (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": list(reversed(source_node_ids)),
                "selection_reason": "  Ready for Pool admission.  ",
            },
            scenario.ctx,
        )
    )
    record = scenario.repository.get_for_task(
        scenario.task.id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    selection = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    return result, selection, record, revision


def test_materialize_frontier_group_is_pointer_only_task_local_and_idempotent(
    scenario,
) -> None:
    revision_result, revision = _revision(scenario)
    canonical_node_ids = revision["tree"]["frontier_node_ids"][:2]
    inputs = {
        "revision_id": revision_result["revision_id"],
        "source_node_ids": list(reversed(canonical_node_ids)),
        "selection_reason": "  Reviewed group.  ",
    }

    first = (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            inputs,
            scenario.ctx,
        )
    )
    repeated = (
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            inputs,
            scenario.ctx,
        )
    )

    assert repeated == first
    assert set(first) == OUTPUT_FIELDS
    assert first["schema_version"] == TOOL_SCHEMA_VERSION
    assert first["source_node_ids"] == canonical_node_ids
    assert first["member_count"] == 2
    assert first["selection_reason"] == "Reviewed group."
    assert first["group_id"].startswith("interactive-tree-frontier-group-")
    assert first["fragment_id"].startswith("candidate-fragment-")
    assert first["rule_id"].startswith("candidate-rule-")
    assert first["effect_id"].startswith("candidate-effect-")
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
    assert (
        record["kind"]
        == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND
    )
    assert (
        record["origin_tool"]
        == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL
    )
    selection = json.loads(expected_path.read_text(encoding="utf-8"))
    assert selection["source_node_ids"] == canonical_node_ids
    assert {
        "condition",
        "requirements",
        "metrics",
        "action",
        "fragment_id",
        "rule_id",
        "effect_id",
    }.isdisjoint(_recursive_keys(selection))
    assert expected_path.read_text(encoding="utf-8") == (
        canonical_interactive_tree_frontier_group_selection_json(selection)
    )
    assert record["provenance"] == (
        group_tools.interactive_tree_frontier_group_selection_provenance(
            selection
        )
    )


def test_verified_group_loader_replays_exact_or_fragment(scenario) -> None:
    result, _selection, record, _revision_payload = _materialized_group(
        scenario
    )
    verified = (
        group_tools.load_verified_interactive_tree_frontier_group_selection_artifact(
            strategy_tools._runtime(scenario.ctx),
            task_id=scenario.task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )
    )
    revision = verified.revision
    ancestry = revision.ancestor_revisions
    fragment = (
        interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
            verified.selection,
            revision.revision,
            revision.automatic_source.asset,
            selection_artifact_binding=verified.artifact_binding(),
            revision_artifact_binding=revision.builder_binding(),
            parent_revision=ancestry[0] if ancestry else None,
            ancestor_revisions=ancestry[1:],
        )
    )

    assert fragment["fragment"]["fragment_id"] == result["fragment_id"]
    assert fragment["fragment"]["rule_id"] == result["rule_id"]
    assert fragment["fragment"]["effect_id"] == result["effect_id"]
    selected_by_id = {
        item["source_node_id"]: item
        for item in revision.revision["fragments"]
    }
    assert fragment["fragment"]["condition"] == {
        "op": "or",
        "args": [
            selected_by_id[node_id]["condition"]
            for node_id in result["source_node_ids"]
        ],
    }


@pytest.mark.parametrize(
    "field",
    [
        "condition",
        "requirements",
        "metrics",
        "action",
        "group_id",
        "selection_id",
        "selection_hash",
        "expected_asset_id",
    ],
)
def test_materialize_group_rejects_caller_owned_platform_fields(
    scenario,
    field: str,
) -> None:
    revision_result, revision = _revision(scenario)
    with pytest.raises(StrategyError, match="unsupported|invalid"):
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": revision["tree"][
                    "frontier_node_ids"
                ][:2],
                field: "forged",
            },
            scenario.ctx,
        )


def test_materialize_group_rejects_duplicate_or_unknown_nodes(scenario) -> None:
    revision_result, revision = _revision(scenario)
    frontier = revision["tree"]["frontier_node_ids"]
    with pytest.raises(StrategyError, match="duplicate"):
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": [frontier[0], frontier[0]],
            },
            scenario.ctx,
        )

    hidden = next(
        node["node_id"]
        for node in scenario.source_asset["tree_result"]["tree"]["nodes"]
        if node["node_id"] not in set(frontier)
    )
    with pytest.raises(StrategyError, match="frontier|invalid"):
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": [frontier[0], hidden],
            },
            scenario.ctx,
        )


def test_group_loader_rejects_cross_task_file_and_provenance_drift(
    scenario,
) -> None:
    result, _selection, record, _revision_payload = _materialized_group(
        scenario
    )
    runtime = strategy_tools._runtime(scenario.ctx)
    with pytest.raises(StrategyError, match="not found"):
        group_tools.load_verified_interactive_tree_frontier_group_selection_artifact(
            runtime,
            task_id=scenario.foreign_task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )

    path = Path(record["path"])
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(StrategyError, match="hash|changed"):
        group_tools.load_verified_interactive_tree_frontier_group_selection_artifact(
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
        provenance["group_id"] = (
            "interactive-tree-frontier-group-" + "f" * 32
        )
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
        group_tools.load_verified_interactive_tree_frontier_group_selection_artifact(
            runtime,
            task_id=scenario.task.id,
            artifact_id=record["id"],
            expected_content_hash=record["content_hash"],
            expected_asset_id=result["semantic_tree_id"],
            expected_asset_hash=result["tree_hash"],
        )


def test_group_idempotent_replay_rejects_registry_primary_key_alias(
    scenario,
) -> None:
    result, selection, record, _revision_payload = _materialized_group(
        scenario
    )
    alias = "a" * 64
    with scenario.repository.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET id = ? WHERE id = ?",
            (alias, record["id"]),
        )
        conn.commit()

    with pytest.raises(StrategyError, match="stable identity"):
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": result["revision_id"],
                "source_node_ids": selection["source_node_ids"],
                "selection_reason": selection["selection_reason"],
            },
            scenario.ctx,
        )


def test_group_registration_failure_rolls_back_file_and_registry(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_result, revision = _revision(scenario)
    original = TaskArtifactRepository.register_on_connection

    def fail_group_registration(self, conn, **kwargs):
        if (
            kwargs.get("kind")
            == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND
        ):
            raise RuntimeError("injected group registration failure")
        return original(self, conn, **kwargs)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_group_registration,
    )
    with pytest.raises(RuntimeError, match="injected group"):
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": revision["tree"][
                    "frontier_node_ids"
                ][:2],
            },
            scenario.ctx,
        )

    assert not [
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"]
        == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND
    ]
    assert not list(
        (
            Path(scenario.settings.tasks_dir)
            / scenario.task.id
            / SELECTION_DIRECTORY
        ).glob("*.json")
    )


def test_group_registration_reauthenticates_revision_under_writer_lock(
    scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_result, revision = _revision(scenario)
    revision_record = scenario.repository.get_for_task(
        scenario.task.id,
        revision_result["artifacts"][0]["artifact_id"],
    )
    assert revision_record is not None
    original = (
        group_tools.load_verified_interactive_tree_revision_on_connection
    )
    changed = False

    def drift_then_verify(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            path = Path(revision_record["path"])
            path.write_bytes(path.read_bytes() + b"\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        group_tools,
        "load_verified_interactive_tree_revision_on_connection",
        drift_then_verify,
    )
    with pytest.raises(StrategyError, match="hash|changed|canonical"):
        strategy_tools.tool_materialize_interactive_tree_frontier_group_selection(
            {
                "revision_id": revision_result["revision_id"],
                "source_node_ids": revision["tree"][
                    "frontier_node_ids"
                ][:2],
            },
            scenario.ctx,
        )

    assert not [
        record
        for record in scenario.repository.list_for_task(scenario.task.id)
        if record["kind"]
        == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND
    ]


def test_group_provenance_content_hash_is_bound_to_canonical_bytes(
    scenario,
) -> None:
    _result, selection, record, _revision_payload = _materialized_group(
        scenario
    )
    canonical = canonical_interactive_tree_frontier_group_selection_json(
        selection
    ).encode("utf-8")
    assert record["content_hash"] == hashlib.sha256(canonical).hexdigest()

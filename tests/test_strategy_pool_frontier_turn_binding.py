"""Turn binding for persisted interactive-tree frontier selections entering Pool."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _candidate_selection_artifact_slots,
    _strategy_pool_plan_slots,
)
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.test_strategy_interactive_tree_threshold_adjustment import (
    _threshold_revision,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


SELECTION_ID = "interactive-tree-frontier-selection-" + "a" * 32
SEMANTIC_TREE_ID = "interactive-tree-" + "b" * 32
TREE_HASH = "c" * 64
FRAGMENT_ID = "candidate-rule-" + "d" * 32


def _task(repository: TaskRepository, name: str):
    return repository.create_task(
        TaskCreate(
            model_name=name,
            model_version="dev",
            validator="qa",
            source_dir=f"/tmp/{name}",
            task_type="strategy",
        )
    )


def _draft(selection_id: str) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_add_candidate",
        workflow_inputs={
            "selection_id": selection_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
            # Direct internal callers can construct a draft without the API
            # schema. The turn boundary must still replace forged platform data.
            "source_artifact_id": "forged-artifact",
            "expected_artifact_content_hash": "f" * 64,
            "expected_asset_id": "forged-tree",
            "expected_asset_hash": "e" * 64,
            "fragment_id": "forged-fragment",
        },
    )


@pytest.fixture
def binding_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = _task(tasks, "interactive-frontier-pool-turn")
    foreign_task = _task(tasks, "foreign-interactive-frontier-pool-turn")
    repository = TaskArtifactRepository(settings.db_path)
    path = settings.tasks_dir / task.id / "frontier-selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    provenance = {
        "schema_version": (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "selection_id": SELECTION_ID,
        "semantic_tree_id": SEMANTIC_TREE_ID,
        "tree_hash": TREE_HASH,
    }
    record = repository.register(
        task_id=task.id,
        kind=INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        path=str(path),
        content_hash="1" * 64,
        origin_tool=INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
        provenance=provenance,
    )
    calls: list[dict] = []

    def fake_loader(conn, **kwargs):
        calls.append(kwargs)
        assert kwargs["runtime"].settings.db_path == settings.db_path
        return SimpleNamespace(
            artifact_id=record["id"],
            content_hash=record["content_hash"],
            selection={
                "selection_id": SELECTION_ID,
                "revision": {
                    "semantic_tree_id": SEMANTIC_TREE_ID,
                    "tree_hash": TREE_HASH,
                },
                "frontier": {"fragment_id": FRAGMENT_ID},
            },
        )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers."
        "load_verified_interactive_tree_frontier_selection_artifact_on_connection",
        fake_loader,
    )
    return SimpleNamespace(
        settings=settings,
        runtime=SimpleNamespace(settings=settings),
        task=task,
        foreign_task=foreign_task,
        repository=repository,
        record=record,
        provenance=provenance,
        calls=calls,
    )


def test_frontier_selection_resolves_unique_registry_row_and_four_verified_slots(
    binding_fixture: SimpleNamespace,
) -> None:
    fx = binding_fixture

    verified_slots, fragment_id = _candidate_selection_artifact_slots(
        fx.runtime,
        task_id=fx.task.id,
        selection_id=SELECTION_ID,
    )
    slots = _strategy_pool_plan_slots(
        fx.runtime,
        fx.task,
        _draft(SELECTION_ID),
    )

    assert {
        "source_artifact_id": slots["source_artifact_id"],
        "expected_artifact_content_hash": slots["expected_artifact_content_hash"],
        "expected_asset_id": slots["expected_asset_id"],
        "expected_asset_hash": slots["expected_asset_hash"],
    } == {
        "source_artifact_id": fx.record["id"],
        "expected_artifact_content_hash": fx.record["content_hash"],
        "expected_asset_id": SEMANTIC_TREE_ID,
        "expected_asset_hash": TREE_HASH,
    }
    assert verified_slots == {
        "source_artifact_id": fx.record["id"],
        "expected_artifact_content_hash": fx.record["content_hash"],
        "expected_asset_id": SEMANTIC_TREE_ID,
        "expected_asset_hash": TREE_HASH,
    }
    assert fragment_id == FRAGMENT_ID
    assert "selection_id" not in slots
    assert "fragment_id" not in slots
    assert len(fx.calls) == 2
    assert fx.calls[0] == {
        "runtime": ANY,
        "task_id": fx.task.id,
        "artifact_id": fx.record["id"],
        "expected_content_hash": fx.record["content_hash"],
        "expected_asset_id": SEMANTIC_TREE_ID,
        "expected_asset_hash": TREE_HASH,
    }


def test_frontier_selection_rejects_cross_task_or_unknown_pointer(
    binding_fixture: SimpleNamespace,
) -> None:
    fx = binding_fixture

    with pytest.raises(StrategySetupError, match="当前任务没有.*frontier selection"):
        _strategy_pool_plan_slots(
            fx.runtime,
            fx.foreign_task,
            _draft(SELECTION_ID),
        )
    assert fx.calls == []


def test_frontier_selection_rejects_duplicate_task_registry_identity(
    binding_fixture: SimpleNamespace,
) -> None:
    fx = binding_fixture
    duplicate = Path(fx.record["path"]).with_name("duplicate-frontier.json")
    duplicate.write_text("{}", encoding="utf-8")
    fx.repository.register(
        task_id=fx.task.id,
        kind=INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        path=str(duplicate),
        content_hash=fx.record["content_hash"],
        origin_tool=INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
        provenance=json.loads(json.dumps(fx.provenance)),
    )

    with pytest.raises(StrategySetupError, match="多个.*frontier selection"):
        _strategy_pool_plan_slots(
            fx.runtime,
            fx.task,
            _draft(SELECTION_ID),
        )
    assert fx.calls == []


def test_v2_frontier_selection_resolves_into_pool_turn_slots(scenario) -> None:
    _result, revision = _threshold_revision(scenario)
    fragment = revision["fragments"][0]
    materialized = (
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision["revision_id"],
                "source_node_id": fragment["source_node_id"],
                "selection_reason": "Use reviewed v2 frontier.",
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

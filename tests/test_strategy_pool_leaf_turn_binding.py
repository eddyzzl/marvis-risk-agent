"""Turn binding for persisted automatic-tree leaf selections entering Pool."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import _strategy_pool_plan_slots
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import (
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
)
from marvis.packs.strategy import automatic_tree_leaf_tools as leaf_tools
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


class _PoolRepository:
    def __init__(self, current) -> None:
        self.current = current

    def get_current(self, task_id: str, strategy_type: str):
        assert strategy_type == "approval"
        return self.current


def _task(repository: TaskRepository, name: str):
    return repository.create_task(
        TaskCreate(
            model_name=name,
            model_version="dev",
            validator="qa",
            source_dir=f"/tmp/{name}",
            task_type="strategy",
            target_col="bad",
        )
    )


def _asset(task_id: str) -> dict:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )
    tree = build_weighted_rule_tree(
        frame,
        feature_cols=["x", "z"],
        target_col="bad",
        max_depth=2,
        min_leaf_count=1,
    )
    return build_automatic_tree_asset(
        tree,
        task_id=task_id,
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=[f"workspace:{task_id}:3", "dataset:dataset-labelled"],
    )


def _materialize_task_selections(settings, task, repository) -> tuple[dict, dict]:
    asset = _asset(task.id)
    content = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    source_path = leaf_tools.canonical_automatic_tree_source_path(
        settings.tasks_dir,
        task_id=task.id,
        asset_id=asset["asset_id"],
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(content)
    source = repository.register(
        task_id=task.id,
        kind=AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        path=str(source_path),
        content_hash=hashlib.sha256(content).hexdigest(),
        origin_tool=AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
        provenance=leaf_tools.automatic_tree_source_provenance_from_asset(asset),
    )
    runtime = SimpleNamespace(settings=settings, task_artifacts=repository)
    ctx = SimpleNamespace(task_id=task.id)
    bound = {
        "source_artifact_id": source["id"],
        "expected_artifact_content_hash": source["content_hash"],
        "expected_asset_id": asset["asset_id"],
        "expected_asset_hash": asset["asset_hash"],
        "expected_tree_result_hash": asset["tree_result"]["result_hash"],
    }
    selections = [
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {**bound, "leaf_id": fragment["leaf_id"]},
            ctx,
            runtime,
        )
        for fragment in asset["fragments"][:2]
    ]
    return selections[0], selections[1]


@pytest.fixture
def selection_fixture(tmp_path: Path) -> SimpleNamespace:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = _task(tasks, "leaf-pool-turn")
    foreign_task = _task(tasks, "foreign-leaf-pool-turn")
    repository = TaskArtifactRepository(settings.db_path)
    first, second = _materialize_task_selections(settings, task, repository)
    foreign, _ = _materialize_task_selections(settings, foreign_task, repository)
    return SimpleNamespace(
        settings=settings,
        runtime=SimpleNamespace(settings=settings),
        task=task,
        foreign_task=foreign_task,
        repository=repository,
        first=first,
        second=second,
        foreign=foreign,
    )


def _draft(selection_id: str) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_add_candidate",
        workflow_inputs={
            "selection_id": selection_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
            # These are deliberately forged caller claims.  The turn boundary
            # must ignore them and derive all lineage inputs from the registry.
            "expected_artifact_content_hash": "f" * 64,
            "expected_asset_id": "candidate-asset-" + "f" * 32,
            "expected_asset_hash": "e" * 64,
            "tree_asset_id": "candidate-asset-" + "d" * 32,
            "leaf_id": "leaf-" + "c" * 20,
        },
    )


def _selection_record(fx: SimpleNamespace, selected: dict) -> dict:
    record = fx.repository.get_for_task(
        fx.task.id,
        selected["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    return record


def test_selection_turn_derives_the_four_platform_slots_from_verified_artifact(
    selection_fixture: SimpleNamespace,
) -> None:
    fx = selection_fixture
    selected = fx.first
    record = _selection_record(fx, selected)

    slots = _strategy_pool_plan_slots(
        fx.runtime,
        fx.task,
        _draft(selected["selection_id"]),
    )

    assert {
        "source_artifact_id": slots["source_artifact_id"],
        "expected_artifact_content_hash": slots["expected_artifact_content_hash"],
        "expected_asset_id": slots["expected_asset_id"],
        "expected_asset_hash": slots["expected_asset_hash"],
    } == {
        "source_artifact_id": record["id"],
        "expected_artifact_content_hash": record["content_hash"],
        "expected_asset_id": selected["tree_asset_id"],
        "expected_asset_hash": selected["tree_asset_hash"],
    }


@pytest.mark.parametrize(
    "selection_id",
    [
        "automatic-tree-leaf-selection-" + "f" * 32,
        "foreign",
    ],
)
def test_selection_turn_rejects_unknown_or_cross_task_selection(
    selection_fixture: SimpleNamespace,
    selection_id: str,
) -> None:
    fx = selection_fixture
    requested = (
        fx.foreign["selection_id"] if selection_id == "foreign" else selection_id
    )

    with pytest.raises(StrategySetupError, match="当前任务没有.*selection"):
        _strategy_pool_plan_slots(fx.runtime, fx.task, _draft(requested))


def test_selection_turn_rejects_duplicate_registry_records(
    selection_fixture: SimpleNamespace,
) -> None:
    fx = selection_fixture
    selected = fx.first
    record = _selection_record(fx, selected)
    duplicate_path = Path(record["path"]).with_name("duplicate-selection.json")
    duplicate_path.write_bytes(Path(record["path"]).read_bytes())
    fx.repository.register(
        task_id=fx.task.id,
        kind=record["kind"],
        path=str(duplicate_path),
        content_hash=record["content_hash"],
        origin_tool=record["origin_tool"],
        provenance=record["provenance"],
    )

    with pytest.raises(StrategySetupError, match="多个.*selection artifact"):
        _strategy_pool_plan_slots(
            fx.runtime,
            fx.task,
            _draft(selected["selection_id"]),
        )


@pytest.mark.parametrize("drift", ["bytes", "provenance"])
def test_selection_turn_rejects_bytes_or_provenance_drift(
    selection_fixture: SimpleNamespace,
    drift: str,
) -> None:
    fx = selection_fixture
    selected = fx.first
    record = _selection_record(fx, selected)
    if drift == "bytes":
        Path(record["path"]).write_bytes(b"{}")
    else:
        provenance = deepcopy(record["provenance"])
        provenance["selection_hash"] = "f" * 64
        with fx.repository.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
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

    with pytest.raises(StrategySetupError, match="完整性校验"):
        _strategy_pool_plan_slots(
            fx.runtime,
            fx.task,
            _draft(selected["selection_id"]),
        )


def test_same_tree_different_leaf_is_allowed_but_same_fragment_is_rejected(
    selection_fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = selection_fixture
    first = fx.first
    current = {
        "revision": 1,
        "default_action": {"type": "approval"},
        "entries": [
            {
                "source": {
                    "asset_id": first["tree_asset_id"],
                    "fragment_id": first["fragment_id"],
                }
            }
        ],
    }
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(current),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda pool: "9" * 64,
    )

    second_slots = _strategy_pool_plan_slots(
        fx.runtime,
        fx.task,
        _draft(fx.second["selection_id"]),
    )
    assert second_slots["expected_asset_id"] == first["tree_asset_id"]

    with pytest.raises(StrategySetupError, match="资产.*片段.*已存在"):
        _strategy_pool_plan_slots(
            fx.runtime,
            fx.task,
            _draft(first["selection_id"]),
        )


def test_univariate_candidate_asset_duplicate_keeps_preflight_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_id = "candidate-asset-" + "a" * 32
    current = {
        "revision": 4,
        "default_action": {"type": "approval"},
        "entries": [
            {
                "source": {
                    "asset_id": asset_id,
                    "fragment_id": "candidate-rule-" + "b" * 32,
                }
            }
        ],
    }
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(current),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda pool: "9" * 64,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._candidate_asset_artifact_slots",
        lambda runtime, *, task_id, asset_id: {
            "source_artifact_id": "artifact-univariate",
            "expected_artifact_content_hash": "c" * 64,
            "expected_asset_id": asset_id,
            "expected_asset_hash": "d" * 64,
        },
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_add_candidate",
        workflow_inputs={
            "candidate_asset_id": asset_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    with pytest.raises(StrategySetupError, match="候选资产.*已存在"):
        _strategy_pool_plan_slots(
            SimpleNamespace(
                settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite")
            ),
            SimpleNamespace(id="task-1"),
            draft,
        )

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy import automatic_tree_leaf_tools as leaf_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.automatic_tree_asset import (
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
    AUTOMATIC_TREE_ASSET_ORIGIN_TOOL,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_revision import (
    canonical_interactive_tree_revision_json,
    validate_interactive_tree_revision,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


INTERACTIVE_TREE_TOOL_SCHEMA_VERSION = "strategy.revise-interactive-tree-tool.v1"
INTERACTIVE_TREE_REVISION_ARTIFACT_KIND = (
    "strategy_interactive_tree_revision_json"
)
INTERACTIVE_TREE_REVISION_ORIGIN_TOOL = "strategy.revise_interactive_tree"
INTERACTIVE_TREE_REVISION_DIRECTORY = "strategy_interactive_tree_revisions"


@dataclass(frozen=True)
class _Scenario:
    settings: object
    registry: DatasetRegistry
    repository: TaskArtifactRepository
    task: object
    foreign_task: object
    dataset: object
    workspace: object
    mapping: DataSemanticMapping
    sample_design_ref: dict[str, str]
    development_frame: pd.DataFrame
    source_record: dict
    source_asset: dict
    ctx: ToolContext


@pytest.fixture
def scenario(tmp_path: Path) -> _Scenario:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="interactive-tree-revision",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    foreign_task = tasks.create_task(
        TaskCreate(
            model_name="foreign-interactive-tree-revision",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    development_bad = [
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
    ]
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:03d}" for index in range(24)],
            "x": [float(index) for index in range(24)],
            "z": [float(index % 4) for index in range(24)],
            "weight": [1.0 + (index % 3) * 0.25 for index in range(24)],
            "loan_amount": [1000.0 + index * 50.0 for index in range(24)],
            "overdue_amount": [
                40.0 + index if value else 0.0
                for index, value in enumerate(
                    development_bad + [0, 1, 0, 1, 0, 1, 0, 1]
                )
            ],
            "bad": development_bad + [0, 1, 0, 1, 0, 1, 0, 1],
            "sample_split": (
                ["dev"] * 16 + ["validation"] * 4 + ["oot"] * 4
            ),
        }
    )
    source = tmp_path / "interactive-tree-source.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "customer_id": "id",
            "x": "numeric",
            "z": "numeric",
            "weight": "weight",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            "bad": "target",
            "sample_split": "segment",
        },
    )
    workspaces = DataWorkspaceRepository(settings.db_path)
    activated = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    workspace = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    ctx = _tool_context(settings, task)
    sample_design = strategy_tools.tool_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "weight_col": "weight",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "performance_window_status": "provided",
            "performance_window_days": 90,
            "observation_window_status": "provided",
            "observation_window_start": "2025-01-01",
            "observation_window_end": "2025-12-31",
            "maturity_status": "confirmed_matured",
            "drop_nan_labels": False,
            "split_col": "sample_split",
            "development_values": ["dev"],
            "validation_values": ["validation"],
            "oot_values": ["oot"],
        },
        ctx,
    )
    sample_design_ref = {
        "artifact_id": sample_design["artifact"]["artifact_id"],
        "artifact_content_hash": sample_design["artifact"]["content_hash"],
        "sample_design_id": sample_design["sample_design_id"],
        "sample_design_content_hash": sample_design["content_hash"],
        "partition": "development",
    }
    strategy_tools.tool_build_automatic_tree_candidate(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "features": ["x", "z"],
            "drop_nan_labels": False,
            "sample_weight_col": "weight",
            "directions": {"x": "increasing", "z": "increasing"},
            "max_depth": 3,
            "min_leaf_count": 1,
            "min_weight_fraction_leaf": 0.0,
            "seed": 20260719,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "budgets": {
                "max_rows": 100,
                "max_features": 5,
                "max_cells": 500,
                "max_nodes": 31,
                "max_cutpoint_evaluations": 1000,
            },
        },
        ctx,
    )
    repository = TaskArtifactRepository(settings.db_path)
    source_record = next(
        record
        for record in repository.list_for_task(task.id)
        if record["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    source_asset = json.loads(
        Path(source_record["path"]).read_text(encoding="utf-8")
    )
    assert len(_split_node_ids(source_asset)) >= 2
    return _Scenario(
        settings=settings,
        registry=registry,
        repository=repository,
        task=task,
        foreign_task=foreign_task,
        dataset=dataset,
        workspace=workspace,
        mapping=mapping,
        sample_design_ref=sample_design_ref,
        development_frame=frame.loc[frame["sample_split"] == "dev"].copy(),
        source_record=source_record,
        source_asset=source_asset,
        ctx=ctx,
    )


def test_revise_automatic_tree_is_canonical_task_local_and_idempotent(
    scenario: _Scenario,
) -> None:
    node_id = _split_node_ids(scenario.source_asset)[-1]
    inputs = {
        "source_tree_id": scenario.source_asset["asset_id"],
        "node_id": node_id,
        "operation": "prune_subtree",
        "reason": "  Analyst\tpruned an unstable subtree.  ",
    }

    first = _invoke(inputs, scenario.ctx)
    repeated = _invoke(inputs, scenario.ctx)

    assert repeated == first
    assert first["schema_version"] == INTERACTIVE_TREE_TOOL_SCHEMA_VERSION
    assert first["revision_id"].startswith("interactive-tree-revision-")
    assert len(first["revision_hash"]) == 64
    assert first["semantic_tree_id"].startswith("interactive-tree-")
    assert first["source_tree_id"] == scenario.source_asset["asset_id"]
    assert first["edit"] == {
        "operation": "prune_subtree",
        "node_id": node_id,
        "reason": "Analyst pruned an unstable subtree.",
    }
    assert first["replay"]["partition"] == "development"
    assert first["replay"]["source_row_count"] == 16
    assert first["replay"]["exactly_once"] is True
    assert first["replay"]["metrics_matched"] is True
    assert len(first["replay"]["result_hash"]) == 64
    assert len(first["artifacts"]) == 1
    assert "path" not in _recursive_keys(first)

    [record] = _revision_records(scenario)
    expected_path = (
        Path(scenario.settings.tasks_dir)
        / scenario.task.id
        / INTERACTIVE_TREE_REVISION_DIRECTORY
        / f"{first['revision_id']}.json"
    )
    assert Path(record["path"]) == expected_path
    assert record["origin_tool"] == INTERACTIVE_TREE_REVISION_ORIGIN_TOOL
    assert record["content_hash"] == first["artifacts"][0]["content_hash"]
    assert record["id"] == first["artifacts"][0]["artifact_id"]
    persisted_bytes = expected_path.read_bytes()
    persisted = json.loads(persisted_bytes.decode("utf-8"))
    assert persisted["revision_id"] == first["revision_id"]
    assert persisted["revision_hash"] == first["revision_hash"]
    assert persisted["semantic_tree_id"] == first["semantic_tree_id"]
    assert persisted["edit"] == first["edit"]
    assert persisted_bytes == canonical_interactive_tree_revision_json(
        persisted,
        scenario.source_asset,
    ).encode("utf-8")
    assert hashlib.sha256(persisted_bytes).hexdigest() == record["content_hash"]


def test_revise_interactive_tree_accepts_an_exact_parent_revision_source(
    scenario: _Scenario,
) -> None:
    split_ids = _split_node_ids(scenario.source_asset)
    deepest_id = split_ids[-1]
    middle_id = split_ids[-2]
    root_id = scenario.source_asset["tree_result"]["tree"]["root_node_id"]
    first_result = _invoke(
        {
            "source_tree_id": scenario.source_asset["asset_id"],
            "node_id": deepest_id,
            "operation": "prune_subtree",
            "reason": "First local merge.",
        },
        scenario.ctx,
    )
    second_result = _invoke(
        {
            "source_tree_id": first_result["revision_id"],
            "node_id": middle_id,
            "operation": "prune_subtree",
            "reason": "Broaden to the middle branch.",
        },
        scenario.ctx,
    )
    third_result = _invoke(
        {
            "source_tree_id": second_result["revision_id"],
            "node_id": root_id,
            "operation": "prune_subtree",
            "reason": "Broaden to the root.",
        },
        scenario.ctx,
    )

    assert second_result["source_tree_id"] == first_result["revision_id"]
    assert third_result["source_tree_id"] == second_result["revision_id"]
    records = _revision_records(scenario)
    assert len(records) == 3
    by_revision_id = {
        json.loads(Path(record["path"]).read_text(encoding="utf-8"))[
            "revision_id"
        ]: record
        for record in records
    }
    first = json.loads(
        Path(by_revision_id[first_result["revision_id"]]["path"]).read_text(
            encoding="utf-8"
        )
    )
    second = json.loads(
        Path(by_revision_id[second_result["revision_id"]]["path"]).read_text(
            encoding="utf-8"
        )
    )
    third = json.loads(
        Path(by_revision_id[third_result["revision_id"]]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert second["parent_revision"]["revision_id"] == first["revision_id"]
    assert third["parent_revision"]["revision_id"] == second["revision_id"]
    assert validate_interactive_tree_revision(
        second,
        scenario.source_asset,
        parent_revision=first,
    ) == second
    assert validate_interactive_tree_revision(
        third,
        scenario.source_asset,
        parent_revision=second,
        ancestor_revisions=(first,),
    ) == third


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "source_artifact_id",
        "expected_artifact_content_hash",
        "dataset_id",
        "frontier_node_ids",
        "metrics",
    ],
)
def test_revise_interactive_tree_rejects_caller_owned_platform_fields(
    scenario: _Scenario,
    forbidden_field: str,
) -> None:
    node_id = _split_node_ids(scenario.source_asset)[-1]
    with pytest.raises(StrategyError, match="unsupported|invalid"):
        _invoke(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": node_id,
                "operation": "prune_subtree",
                "reason": None,
                forbidden_field: "forged",
            },
            scenario.ctx,
        )

    _assert_no_revision_side_effects(scenario)


def test_source_tree_resolution_is_task_local(scenario: _Scenario) -> None:
    foreign_ctx = _tool_context(scenario.settings, scenario.foreign_task)

    with pytest.raises(StrategyError, match="source|task|not found"):
        _invoke(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": _split_node_ids(scenario.source_asset)[-1],
                "operation": "prune_subtree",
                "reason": None,
            },
            foreign_ctx,
        )

    assert _revision_records_for_task(
        scenario.repository,
        scenario.foreign_task.id,
    ) == []
    _assert_revision_directory_has_no_files(
        scenario.settings,
        scenario.foreign_task.id,
    )


@pytest.mark.parametrize(
    "drift_kind",
    ["dataset", "workspace", "sample_design"],
)
def test_current_governed_binding_drift_fails_without_any_revision(
    scenario: _Scenario,
    drift_kind: str,
) -> None:
    if drift_kind == "dataset":
        scenario.registry.resolve_path(scenario.dataset.id).write_bytes(
            b"physical dataset drift"
        )
    elif drift_kind == "workspace":
        changed_mapping = DataSemanticMapping(
            target_col="bad",
            field_roles=dict(scenario.mapping.field_roles),
            business_names={"x": "changed after automatic-tree creation"},
        )
        DataWorkspaceRepository(scenario.settings.db_path).save(
            scenario.task.id,
            DataWorkspaceDraft(
                active_dataset_id=scenario.dataset.id,
                active_dataset_content_hash=scenario.dataset.content_hash,
                semantic_mapping=changed_mapping,
            ),
            expected_revision=scenario.workspace.revision,
        )
    else:
        sample_record = scenario.repository.get_for_task(
            scenario.task.id,
            scenario.sample_design_ref["artifact_id"],
        )
        assert sample_record is not None
        Path(sample_record["path"]).write_bytes(b"sample-design drift")

    with pytest.raises(StrategyError, match="binding|dataset|workspace|sample"):
        _invoke(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": _split_node_ids(scenario.source_asset)[-1],
                "operation": "prune_subtree",
                "reason": None,
            },
            scenario.ctx,
        )

    _assert_no_revision_side_effects(scenario)


def test_development_replay_rejects_an_internally_valid_but_false_tree(
    scenario: _Scenario,
) -> None:
    forged_asset = _register_false_source_asset(scenario)

    with pytest.raises(StrategyError, match="replay|metric|development"):
        _invoke(
            {
                "source_tree_id": forged_asset["asset_id"],
                "node_id": _split_node_ids(forged_asset)[-1],
                "operation": "prune_subtree",
                "reason": "This must never persist.",
            },
            scenario.ctx,
        )

    _assert_no_revision_side_effects(scenario)


def test_registration_failure_rolls_back_promoted_revision_and_registry(
    scenario: _Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = TaskArtifactRepository.register_on_connection

    def fail_revision_registration(self, conn, **kwargs):
        if kwargs.get("kind") == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND:
            raise RuntimeError("injected interactive-tree registration failure")
        return original(self, conn, **kwargs)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_revision_registration,
    )

    with pytest.raises(
        RuntimeError,
        match="injected interactive-tree registration failure",
    ):
        _invoke(
            {
                "source_tree_id": scenario.source_asset["asset_id"],
                "node_id": _split_node_ids(scenario.source_asset)[-1],
                "operation": "prune_subtree",
                "reason": None,
            },
            scenario.ctx,
        )

    _assert_no_revision_side_effects(scenario)


def _invoke(inputs: dict, ctx: ToolContext) -> dict:
    return strategy_tools.tool_revise_interactive_tree(inputs, ctx)


def _tool_context(settings, task) -> ToolContext:
    return ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _split_node_ids(asset: dict) -> list[str]:
    return [
        node["node_id"]
        for node in asset["tree_result"]["tree"]["nodes"]
        if node["kind"] == "split"
    ]


def _revision_records(scenario: _Scenario) -> list[dict]:
    return _revision_records_for_task(scenario.repository, scenario.task.id)


def _revision_records_for_task(
    repository: TaskArtifactRepository,
    task_id: str,
) -> list[dict]:
    return [
        record
        for record in repository.list_for_task(task_id)
        if record["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
    ]


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(
            *(_recursive_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()


def _assert_no_revision_side_effects(scenario: _Scenario) -> None:
    assert _revision_records(scenario) == []
    _assert_revision_directory_has_no_files(
        scenario.settings,
        scenario.task.id,
    )


def _assert_revision_directory_has_no_files(settings, task_id: str) -> None:
    out_dir = (
        Path(settings.tasks_dir)
        / task_id
        / INTERACTIVE_TREE_REVISION_DIRECTORY
    )
    assert not out_dir.exists() or not any(
        path.is_file() or path.is_symlink()
        for path in out_dir.rglob("*")
    )


def _register_false_source_asset(scenario: _Scenario) -> dict:
    false_frame = scenario.development_frame.copy()
    false_frame["bad"] = 1 - false_frame["bad"]
    false_tree = build_weighted_rule_tree(
        false_frame,
        feature_cols=["x", "z"],
        target_col="bad",
        sample_weight_col="weight",
        directions={"x": "increasing", "z": "increasing"},
        max_depth=3,
        min_leaf_count=1,
        min_weight_fraction_leaf=0.0,
        seed=20260719,
        loan_amount_col="loan_amount",
        overdue_amount_col="overdue_amount",
    )
    identity = scenario.source_asset["identity"]
    forged_asset = build_automatic_tree_asset(
        false_tree,
        task_id=scenario.task.id,
        dataset_id=identity["dataset_id"],
        dataset_content_hash=identity["dataset_content_hash"],
        workspace_revision=identity["workspace_revision"],
        workspace_generation=identity["workspace_generation"],
        semantic_mapping_hash=identity["semantic_mapping_hash"],
        registry_metadata_hash=identity["registry_metadata_hash"],
        sample_context_hash=identity["sample_context_hash"],
        source_refs=deepcopy(scenario.source_asset["source_refs"]),
    )
    content = canonical_automatic_tree_asset_json(forged_asset).encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    path = leaf_tools.canonical_automatic_tree_source_path(
        scenario.settings.tasks_dir,
        task_id=scenario.task.id,
        asset_id=forged_asset["asset_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    scenario.repository.register(
        task_id=scenario.task.id,
        kind=AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
        path=str(path),
        content_hash=content_hash,
        origin_tool=AUTOMATIC_TREE_ASSET_ORIGIN_TOOL,
        provenance=leaf_tools.automatic_tree_source_provenance_from_asset(
            forged_asset
        ),
    )
    return forged_asset

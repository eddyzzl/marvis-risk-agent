from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
import threading

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.db_schema import connect
from marvis.files import sha256_file
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.data_transform import DataTransformRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
import marvis.packs.data_ops.tools as data_ops_tools


_CREATED_AT = "2026-07-19T05:00:00+00:00"


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    with connect(settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, task_type, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (
                'task-1', 'strategy', 'transform task', 'v1', 'tester',
                '/tmp/source', 'created', 'created', ?, ?
            )
            """,
            (_CREATED_AT, _CREATED_AT),
        )
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    return settings, runner, backend, registry


def _source_dataset(tmp_path, registry):
    source_csv = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D"],
            "bad": [0, 1, 0, 1],
            "score": [700.0, 620.0, 450.0, 680.0],
            "amount": [100.0, None, 200.0, 300.0],
        }
    ).to_csv(source_csv, index=False)
    return registry.register_from_upload("task-1", source_csv, role="strategy_sample")


def _workspace(settings, dataset):
    repo = DataWorkspaceRepository(settings.db_path)
    active = repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    return repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="semantics",
            selected_field="score",
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={
                    "customer_id": "id",
                    "bad": "target",
                    "score": "score",
                    "amount": "amount",
                },
                business_names={
                    "customer_id": "客户编号",
                    "bad": "风险标签",
                    "score": "模型分",
                    "amount": "申请金额",
                },
            ),
        ),
        expected_revision=active.revision,
    )


def _empty_workspace(settings, dataset):
    return DataWorkspaceRepository(settings.db_path).save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )


def _inputs(dataset, workspace, operations, **extra):
    return {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "analysis_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(
            workspace.semantic_mapping
        ),
        "operations": operations,
        **extra,
    }


def test_transform_dataset_tool_atomically_creates_lineage_evidence_and_workspace(
    tmp_path,
):
    settings, runner, backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _workspace(settings, source)
    source_path = registry.resolve_verified_path(source.id)
    source_hash_before = sha256_file(source_path)

    result = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [
                {
                    "op": "rename_columns",
                    "mapping": {"bad": "label", "score": "model_score"},
                },
                {
                    "op": "fill_missing",
                    "fills": [{"column": "amount", "method": "mean"}],
                },
                {
                    "op": "derive_columns",
                    "derivations": [
                        {
                            "name": "score_per_amount",
                            "expression": {
                                "op": "divide",
                                "left": {"column": "model_score"},
                                "right": {"column": "amount"},
                            },
                            "to_type": "DOUBLE",
                        }
                    ],
                },
                {
                    "op": "filter_rows",
                    "predicate": {
                        "op": "gte",
                        "left": {"column": "model_score"},
                        "right": {"literal": 500.0},
                    },
                },
            ],
        ),
        task_id="task-1",
    )

    assert result.ok is True, result.error
    output = result.output
    assert output["schema_version"] == "data-transform-tool-result.v1"
    assert output["source_dataset_id"] == source.id
    assert output["row_count_before"] == 4
    assert output["row_count_after"] == 3
    assert output["column_count_before"] == 4
    assert output["column_count_after"] == 5
    assert output["cached"] is False
    derived = registry.get(output["result_dataset_id"])
    assert derived.task_id == "task-1"
    assert derived.role == "derived"
    assert derived.has_target is True
    assert derived.target_col == "label"
    derived_path = registry.resolve_verified_path(derived.id)
    assert derived_path.parent.name == "transforms"
    assert sha256_file(source_path) == source_hash_before == source.content_hash
    frame = backend.read_frame(derived_path)
    assert list(frame.columns) == [
        "customer_id",
        "label",
        "model_score",
        "amount",
        "score_per_amount",
    ]
    assert frame["customer_id"].tolist() == ["A", "B", "D"]
    assert frame["amount"].isna().sum() == 0

    active = DataWorkspaceRepository(settings.db_path).get_or_default("task-1")
    assert active.active_dataset_id == derived.id
    assert active.active_dataset_content_hash == derived.content_hash
    assert active.revision == workspace.revision + 1
    assert active.analysis_generation == workspace.analysis_generation + 1
    assert active.page == "history"
    assert active.selected_field == "model_score"
    assert active.semantic_mapping.target_col == "label"
    assert active.semantic_mapping.field_roles["customer_id"] == "id"
    assert active.semantic_mapping.business_names["model_score"] == "模型分"

    runs = DataTransformRepository(settings.db_path)
    record = runs.get_for_task("task-1", output["run_id"])
    assert record is not None
    assert record.result_dataset_id == derived.id
    assert record.result_content_hash == derived.content_hash
    lineage = runs.list_lineage("task-1")
    assert [(row["parent_dataset_id"], row["child_dataset_id"]) for row in lineage] == [
        (source.id, derived.id)
    ]
    artifacts = TaskArtifactRepository(settings.db_path).list_for_task("task-1")
    assert len(artifacts) == 1
    assert artifacts[0]["id"] == output["evidence_artifact_id"]
    assert artifacts[0]["kind"] == "data_transform_evidence"
    evidence_path = Path(artifacts[0]["path"])
    assert evidence_path.is_file()
    assert sha256_file(evidence_path) == artifacts[0]["content_hash"]
    assert not list(derived_path.parent.rglob("*.tmp"))


def test_transform_dataset_target_or_key_drop_requires_explicit_confirmation(tmp_path):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _workspace(settings, source)

    refused = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "drop_columns", "columns": ["bad"]}],
        ),
        task_id="task-1",
    )

    assert refused.ok is False
    assert "explicit confirmation" in refused.error
    assert [item.id for item in registry.list_for_task("task-1")] == [source.id]
    assert DataTransformRepository(settings.db_path).list_lineage("task-1") == []
    assert TaskArtifactRepository(settings.db_path).list_for_task("task-1") == []
    assert DataWorkspaceRepository(settings.db_path).get_or_default("task-1") == workspace
    output_dir = settings.datasets_dir / "task-1" / "transforms"
    assert not list(output_dir.glob("*.parquet"))
    assert not list(output_dir.glob("*.json"))

    confirmed = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "drop_columns", "columns": ["bad"]}],
            confirm_protected_drop=True,
        ),
        task_id="task-1",
    )

    assert confirmed.ok is True, confirmed.error
    derived = registry.get(confirmed.output["result_dataset_id"])
    assert derived.has_target is False
    assert derived.target_col is None
    active = DataWorkspaceRepository(settings.db_path).get_or_default("task-1")
    assert active.semantic_mapping.target_col is None
    assert confirmed.output["semantic_migration"]["dropped_protected_fields"] == [
        "bad"
    ]


def test_transform_dataset_fails_closed_for_cross_task_dataset_reference(tmp_path):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _workspace(settings, source)
    with connect(settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, task_type, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (
                'task-2', 'strategy', 'foreign', 'v1', 'tester', '/tmp/source',
                'created', 'created', ?, ?
            )
            """,
            (_CREATED_AT, _CREATED_AT),
        )

    result = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "drop_columns", "columns": ["amount"]}],
        ),
        task_id="task-2",
    )

    assert result.ok is False
    assert "belongs to task task-1, not task-2" in result.error
    assert [item.id for item in registry.list_for_task("task-1")] == [source.id]


def test_transform_dataset_rejects_symlinked_dataset_output_directory(tmp_path):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _workspace(settings, source)
    outside = tmp_path / "outside"
    outside.mkdir()
    task_root = settings.datasets_dir / "task-1"
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "transforms").symlink_to(outside, target_is_directory=True)

    result = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "drop_columns", "columns": ["amount"]}],
        ),
        task_id="task-1",
    )

    assert result.ok is False
    assert "must not be a symlink" in result.error
    assert list(outside.iterdir()) == []
    assert [item.id for item in registry.list_for_task("task-1")] == [source.id]
    assert DataTransformRepository(settings.db_path).list_lineage("task-1") == []
    assert TaskArtifactRepository(settings.db_path).list_for_task("task-1") == []
    assert DataWorkspaceRepository(settings.db_path).get_or_default("task-1") == workspace


def test_transform_preserves_registered_target_when_workspace_semantics_are_empty(
    tmp_path,
):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    assert source.target_col == "bad"
    workspace = _empty_workspace(settings, source)

    result = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "rename_columns", "mapping": {"bad": "label"}}],
        ),
        task_id="task-1",
    )

    assert result.ok is True, result.error
    derived = registry.get(result.output["result_dataset_id"])
    assert derived.target_col == "label"
    assert derived.has_target is True
    active = DataWorkspaceRepository(settings.db_path).get_or_default("task-1")
    assert active.semantic_mapping.target_col == "label"
    assert active.semantic_mapping.field_roles["label"] == "target"


def test_transform_protects_registered_target_when_workspace_semantics_are_empty(
    tmp_path,
):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _empty_workspace(settings, source)

    result = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "drop_columns", "columns": ["bad"]}],
        ),
        task_id="task-1",
    )

    assert result.ok is False
    assert "explicit confirmation" in result.error
    assert DataWorkspaceRepository(settings.db_path).get_or_default("task-1") == workspace
    assert DataTransformRepository(settings.db_path).list_lineage("task-1") == []


def test_task_purge_counts_and_removes_transform_lineage_before_datasets(tmp_path):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _workspace(settings, source)
    transformed = runner.invoke(
        ToolRef("data_ops", "transform_dataset"),
        _inputs(
            source,
            workspace,
            [{"op": "drop_columns", "columns": ["amount"]}],
        ),
        task_id="task-1",
    )
    assert transformed.ok is True, transformed.error
    repo = TaskRepository(settings.db_path)

    preview = repo.purge_preview("task-1")
    assert preview["data_transform_runs"] == 1
    assert preview["dataset_lineage_edges"] == 1
    assert preview["datasets"] == 2

    summary = repo.purge_task("task-1")
    assert summary["data_transform_runs"] == 1
    assert summary["dataset_lineage_edges"] == 1
    with connect(settings.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM data_transform_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM dataset_lineage_edges").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0


def test_concurrent_identical_transform_attempts_cannot_corrupt_winner_artifacts(
    tmp_path,
    monkeypatch,
):
    settings, runner, _backend, registry = _runtime(tmp_path)
    source = _source_dataset(tmp_path, registry)
    workspace = _workspace(settings, source)
    inputs = _inputs(
        source,
        workspace,
        [{"op": "drop_columns", "columns": ["amount"]}],
    )
    original_transform = data_ops_tools.transform_parquet
    both_computed = threading.Barrier(2)

    def synchronized_transform(*args, **kwargs):
        result = original_transform(*args, **kwargs)
        both_computed.wait(timeout=15)
        return result

    monkeypatch.setattr(data_ops_tools, "transform_parquet", synchronized_transform)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                runner.invoke,
                ToolRef("data_ops", "transform_dataset"),
                inputs,
                task_id="task-1",
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=30) for future in futures]

    succeeded = [result for result in results if result.ok]
    assert succeeded, [result.error for result in results]
    repo = DataTransformRepository(settings.db_path)
    lineage = repo.list_lineage("task-1")
    assert len(lineage) == 1
    record = repo.get_for_task("task-1", lineage[0]["transform_run_id"])
    assert record is not None
    registry.resolve_verified_path(record.result_dataset_id)
    artifact = TaskArtifactRepository(settings.db_path).get_for_task(
        "task-1",
        record.result_artifact_id,
    )
    assert artifact is not None
    artifact_path = Path(artifact["path"])
    assert artifact_path.is_file()
    assert artifact_path.is_relative_to(settings.tasks_dir / "task-1")
    assert sha256_file(artifact_path) == artifact["content_hash"] == record.result_hash
    assert not list(settings.datasets_dir.rglob("*.bak"))
    assert not list(settings.tasks_dir.rglob("*.bak"))

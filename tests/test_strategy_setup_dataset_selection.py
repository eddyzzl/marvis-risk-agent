from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.agent import turn_handlers
from marvis.agent.strategy_setup import (
    StrategySetupError,
    build_strategy_dataset_context,
    preview_strategy_dataset_context,
)
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
)
from marvis.db import DatasetRepository, init_db
from marvis.db_schema import connect
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    now = datetime.now(UTC).isoformat()
    with connect(settings.db_path) as conn:
        for task_id in ("task-1", "task-2"):
            conn.execute(
                """
                INSERT INTO tasks(
                    id, task_type, model_name, model_version, validator,
                    source_dir, status, status_message, created_at, updated_at
                ) VALUES (?, 'strategy', 'strategy setup', 'v1', 'tester',
                          '/tmp/source', 'created', 'created', ?, ?)
                """,
                (task_id, now, now),
            )
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        backend,
        settings.datasets_dir,
    )
    return settings, backend, registry


def _dataset(tmp_path, registry, *, name: str, rows: int, role: str, task_id="task-1"):
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(rows)],
            "bad": [i % 2 for i in range(rows)],
            "score": [700 - i for i in range(rows)],
        }
    ).to_csv(path, index=False)
    return registry.register_from_upload(task_id, path, role=role)


def _activate(settings, dataset):
    repository = DataWorkspaceRepository(settings.db_path)
    activated = repository.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    return repository.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={"bad": "target", "score": "score"},
            ),
        ),
        expected_revision=activated.revision,
    )


def test_active_derived_dataset_wins_over_larger_legacy_sample(tmp_path):
    settings, backend, registry = _runtime(tmp_path)
    _dataset(tmp_path, registry, name="large-old", rows=20, role="sample")
    derived = _dataset(tmp_path, registry, name="derived", rows=4, role="derived")
    workspace = _activate(settings, derived)

    context = build_strategy_dataset_context(
        registry,
        backend,
        "task-1",
        source_dir=None,
    )
    preview = preview_strategy_dataset_context(
        registry,
        backend,
        "task-1",
        source_dir=None,
    )

    assert context.dataset_id == derived.id
    assert context.target_col == "bad"
    assert preview.dataset_id == derived.id
    assert preview.target_col == "bad"
    assert preview.identity["workspace_revision"] == workspace.revision
    assert preview.identity["analysis_generation"] == workspace.analysis_generation
    assert preview.identity["semantic_mapping_hash"] == context.semantic_mapping_hash


def test_workspace_hash_mismatch_fails_instead_of_falling_back(tmp_path):
    settings, backend, registry = _runtime(tmp_path)
    legacy = _dataset(tmp_path, registry, name="large-old", rows=20, role="sample")
    active = _dataset(tmp_path, registry, name="derived", rows=4, role="derived")
    _activate(settings, active)
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE data_workspaces SET active_dataset_content_hash = ? WHERE task_id = ?",
            (legacy.content_hash, "task-1"),
        )

    with pytest.raises(StrategySetupError, match="工作区绑定无效"):
        build_strategy_dataset_context(
            registry,
            backend,
            "task-1",
            source_dir=None,
        )


def test_workspace_cross_task_dataset_fails_instead_of_falling_back(tmp_path):
    settings, backend, registry = _runtime(tmp_path)
    _dataset(tmp_path, registry, name="legacy", rows=20, role="sample")
    foreign = _dataset(
        tmp_path,
        registry,
        name="foreign-derived",
        rows=4,
        role="derived",
        task_id="task-2",
    )
    own = _dataset(tmp_path, registry, name="own-derived", rows=4, role="derived")
    _activate(settings, own)
    with connect(settings.db_path) as conn:
        conn.execute(
            """
            UPDATE data_workspaces
               SET active_dataset_id = ?, active_dataset_content_hash = ?
             WHERE task_id = ?
            """,
            (foreign.id, foreign.content_hash, "task-1"),
        )

    with pytest.raises(StrategySetupError, match="工作区绑定无效"):
        preview_strategy_dataset_context(
            registry,
            backend,
            "task-1",
            source_dir=None,
        )


def test_active_dataset_physical_hash_drift_fails_instead_of_falling_back(tmp_path):
    settings, backend, registry = _runtime(tmp_path)
    _dataset(tmp_path, registry, name="legacy", rows=20, role="sample")
    active = _dataset(tmp_path, registry, name="derived", rows=4, role="derived")
    _activate(settings, active)
    with registry.resolve_path(active.id).open("ab") as handle:
        handle.write(b"out-of-band mutation")

    with pytest.raises(StrategySetupError, match="内容哈希校验"):
        build_strategy_dataset_context(
            registry,
            backend,
            "task-1",
            source_dir=None,
        )


def test_no_active_workspace_keeps_legacy_labeled_largest_sample_choice(tmp_path):
    _, backend, registry = _runtime(tmp_path)
    larger = _dataset(tmp_path, registry, name="large-old", rows=20, role="sample")
    _dataset(tmp_path, registry, name="small-new", rows=4, role="strategy_sample")

    context = build_strategy_dataset_context(
        registry,
        backend,
        "task-1",
        source_dir=None,
    )

    assert context.dataset_id == larger.id
    assert context.target_col == "bad"


def test_confirmation_binding_rejects_workspace_semantic_drift(monkeypatch):
    preview = SimpleNamespace(
        dataset_id="dataset-1",
        columns=("score", "bad"),
        target_col="bad",
        identity={
            "kind": "registered",
            "dataset_id": "dataset-1",
            "content_hash": "a" * 64,
            "workspace_revision": 1,
            "analysis_generation": 1,
            "semantic_mapping_hash": "b" * 64,
        },
    )
    context = SimpleNamespace(
        dataset_id="dataset-1",
        columns=("score", "bad"),
        target_col="bad",
        dataset_content_hash="a" * 64,
        workspace_revision=1,
        analysis_generation=1,
        semantic_mapping_hash="b" * 64,
    )
    refreshed = SimpleNamespace(
        dataset_id="dataset-1",
        columns=("score", "bad"),
        target_col="bad",
        identity={
            **preview.identity,
            "workspace_revision": 2,
            "semantic_mapping_hash": "c" * 64,
        },
    )
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_dataset_preview",
        lambda runtime, task: refreshed,
    )

    assert not turn_handlers._strategy_dataset_binding_matches(
        SimpleNamespace(),
        SimpleNamespace(),
        preview=preview,
        context=context,
    )

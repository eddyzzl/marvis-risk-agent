from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading

import pandas as pd
import pytest

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.automatic_tree_apply import AutomaticTreeApplyRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


@dataclass(frozen=True)
class _Scenario:
    settings: object
    runner: ToolRunner
    registry: DatasetRegistry
    task: object
    other_task: object
    dataset: object
    workspace: object
    mapping: DataSemanticMapping
    artifact: dict
    asset: dict
    ctx: ToolContext


@pytest.fixture
def scenario(tmp_path: Path) -> _Scenario:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="automatic-tree-apply",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    other_task = tasks.create_task(
        TaskCreate(
            model_name="foreign-automatic-tree-apply",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:03d}" for index in range(24)],
            "score": [360 + index * 20 for index in range(24)],
            "income": [3000 + (index % 8) * 800 for index in range(24)],
            "bad": [1 if index < 12 else 0 for index in range(24)],
        }
    )
    source = tmp_path / "automatic-tree-apply-source.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "customer_id": "id",
            "score": "score",
            "income": "numeric",
            "bad": "target",
        },
        business_names={"score": "信用分", "bad": "坏样本"},
    )
    workspaces = DataWorkspaceRepository(settings.db_path)
    reset = workspaces.save(
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
            selected_field="score",
            semantic_mapping=mapping,
        ),
        expected_revision=reset.revision,
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    sample_design = strategy_tools.tool_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 90,
            "observation_window_status": "provided",
            "observation_window_start": "2025-01-01",
            "observation_window_end": "2025-12-31",
            "maturity_status": "confirmed_matured",
            "drop_nan_labels": False,
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
            "features": ["score", "income"],
            "drop_nan_labels": False,
            "sample_weight_col": None,
            "directions": {"score": "decreasing", "income": "decreasing"},
            "max_depth": 2,
            "min_leaf_count": 2,
            "min_weight_fraction_leaf": 0.0,
            "seed": 20260719,
            "loan_amount_col": None,
            "overdue_amount_col": None,
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
    artifact = next(
        record
        for record in TaskArtifactRepository(settings.db_path).list_for_task(task.id)
        if record["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    asset = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    return _Scenario(
        settings=settings,
        runner=runner,
        registry=registry,
        task=task,
        other_task=other_task,
        dataset=dataset,
        workspace=workspace,
        mapping=mapping,
        artifact=artifact,
        asset=asset,
        ctx=ctx,
    )


def _inputs(scenario: _Scenario, **overrides) -> dict:
    inputs = {
        "source_artifact_id": scenario.artifact["id"],
        "expected_artifact_content_hash": scenario.artifact["content_hash"],
        "expected_asset_id": scenario.asset["asset_id"],
        "expected_asset_hash": scenario.asset["asset_hash"],
        "expected_tree_result_hash": scenario.asset["tree_result"]["result_hash"],
        "dataset_id": scenario.dataset.id,
        "expected_content_hash": scenario.dataset.content_hash,
        "workspace_revision": scenario.workspace.revision,
        "analysis_generation": scenario.workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(scenario.mapping),
        "activate_result": False,
    }
    inputs.update(overrides)
    return inputs


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(
            *(_mapping_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_mapping_keys(child) for child in value), set())
    return set()


def test_apply_false_registers_without_activating_and_is_path_free(
    scenario: _Scenario,
) -> None:
    result = strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)

    assert result["cached"] is False
    assert result["activated"] is False
    assert result["columns"] == {
        "leaf_id": "automatic_tree_leaf_id",
        "rule_id": "automatic_tree_rule_id",
    }
    assert sum(item["row_count"] for item in result["leaf_distribution"]) == 24
    assert len(result["result"]["result_hash"]) == 64
    assert not ({"path", "action", "rank", "recommendation"} & _mapping_keys(result))

    workspace = DataWorkspaceRepository(scenario.settings.db_path).get_or_default(
        scenario.task.id
    )
    assert workspace == scenario.workspace
    dataset = scenario.registry.get(result["result"]["dataset_id"])
    assert dataset.role == "strategy.automatic_tree.applied"
    assert dataset.target_col == "bad"
    assert Path(dataset.source_path) == Path(
        scenario.task.id,
        "strategy_automatic_tree_applies",
        f"{result['run_id']}.parquet",
    )
    evidence = TaskArtifactRepository(scenario.settings.db_path).get_for_task(
        scenario.task.id, result["evidence"]["artifact_id"]
    )
    assert evidence is not None
    assert evidence["kind"] == "strategy_automatic_tree_apply_evidence"
    assert evidence["origin_tool"] == "strategy.apply_automatic_tree"
    assert Path(evidence["path"]) == (
        scenario.settings.tasks_dir
        / scenario.task.id
        / "strategy_automatic_tree_applies"
        / f"{result['run_id']}.evidence.json"
    )


def test_apply_false_then_true_reuses_exact_facts_and_true_retry_is_noop(
    scenario: _Scenario,
) -> None:
    registered = strategy_tools.tool_apply_automatic_tree(
        _inputs(scenario), scenario.ctx
    )
    activated = strategy_tools.tool_apply_automatic_tree(
        _inputs(scenario, activate_result=True), scenario.ctx
    )
    revision = activated["workspace"]["result_revision"]
    generation = activated["workspace"]["result_analysis_generation"]
    retried = strategy_tools.tool_apply_automatic_tree(
        _inputs(scenario, activate_result=True), scenario.ctx
    )
    passive_replay = strategy_tools.tool_apply_automatic_tree(
        _inputs(scenario, activate_result=False), scenario.ctx
    )

    assert activated["cached"] is True
    assert activated["activated"] is True
    assert retried["cached"] is True
    assert retried["activated"] is True
    assert passive_replay["cached"] is True
    assert passive_replay["activated"] is True
    assert registered["run_id"] == activated["run_id"] == retried["run_id"]
    assert registered["result"] == activated["result"] == retried["result"]
    assert registered["evidence"] == activated["evidence"] == retried["evidence"]
    assert retried["workspace"]["result_revision"] == revision
    assert retried["workspace"]["result_analysis_generation"] == generation
    assert passive_replay["workspace"]["result_revision"] == revision
    workspace = DataWorkspaceRepository(scenario.settings.db_path).get_or_default(
        scenario.task.id
    )
    assert workspace.revision == scenario.workspace.revision + 1
    assert workspace.analysis_generation == scenario.workspace.analysis_generation + 1
    assert workspace.page == "history"
    assert workspace.selected_field == "score"
    assert dict(workspace.semantic_mapping.business_names) == dict(
        scenario.mapping.business_names
    )
    assert workspace.semantic_mapping.field_roles["automatic_tree_leaf_id"] == (
        "rule_node"
    )
    assert workspace.semantic_mapping.field_roles["automatic_tree_rule_id"] == (
        "segment"
    )


@pytest.mark.parametrize(
    "caller_field",
    ["path", "result", "result_dataset_id", "action", "selected_action", "rank"],
)
def test_apply_rejects_caller_result_path_and_action_fields(
    scenario: _Scenario,
    caller_field: str,
) -> None:
    with pytest.raises(StrategyError):
        strategy_tools.tool_apply_automatic_tree(
            _inputs(scenario, **{caller_field: "caller-owned"}),
            scenario.ctx,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("expected_content_hash", "f" * 64),
        ("workspace_revision", 99),
        ("analysis_generation", 99),
        ("semantic_mapping_hash", "f" * 64),
        ("expected_asset_hash", "f" * 64),
        ("expected_tree_result_hash", "f" * 64),
    ],
)
def test_apply_rejects_stale_or_mismatched_bindings(
    scenario: _Scenario,
    field: str,
    replacement: object,
) -> None:
    with pytest.raises(StrategyError):
        strategy_tools.tool_apply_automatic_tree(
            _inputs(scenario, **{field: replacement}), scenario.ctx
        )


def test_apply_rejects_case_insensitive_output_column_collisions(
    scenario: _Scenario,
) -> None:
    with pytest.raises(StrategyError):
        strategy_tools.tool_apply_automatic_tree(
            _inputs(scenario, leaf_id_column="SCORE"), scenario.ctx
        )


def test_apply_concurrent_same_input_commits_one_facts_set(
    scenario: _Scenario,
) -> None:
    barrier = threading.Barrier(2)
    outputs: list[dict] = []
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            barrier.wait(timeout=10)
            outputs.append(
                strategy_tools.tool_apply_automatic_tree(
                    _inputs(scenario), scenario.ctx
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(outputs) == 2
    assert {result["run_id"] for result in outputs} == {outputs[0]["run_id"]}
    assert {result["result"]["dataset_id"] for result in outputs} == {
        outputs[0]["result"]["dataset_id"]
    }
    assert {result["evidence"]["artifact_id"] for result in outputs} == {
        outputs[0]["evidence"]["artifact_id"]
    }
    assert (
        len(
            AutomaticTreeApplyRepository(scenario.settings.db_path)
            .get_by_id(scenario.task.id, outputs[0]["run_id"])
            .result_payload["output"]["leaf_distribution"]
        )
        >= 1
    )


def test_apply_rejects_foreign_dataset_without_leaking_it(
    scenario: _Scenario,
    tmp_path: Path,
) -> None:
    foreign_source = tmp_path / "foreign-source.parquet"
    pd.DataFrame({"score": [1, 2], "income": [3, 4], "bad": [0, 1]}).to_parquet(
        foreign_source, index=False
    )
    foreign = scenario.registry.register_existing(
        foreign_source,
        task_id=scenario.other_task.id,
        role="derived",
    )

    with pytest.raises(StrategyError):
        strategy_tools.tool_apply_automatic_tree(
            _inputs(
                scenario,
                dataset_id=foreign.id,
                expected_content_hash=foreign.content_hash,
            ),
            scenario.ctx,
        )


def test_cached_activation_rejects_a_third_active_dataset(
    scenario: _Scenario,
    tmp_path: Path,
) -> None:
    strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)
    replacement_path = tmp_path / "replacement.parquet"
    pd.read_parquet(
        scenario.registry.resolve_verified_path(scenario.dataset.id)
    ).to_parquet(replacement_path, index=False)
    replacement = scenario.registry.register_existing(
        replacement_path,
        task_id=scenario.task.id,
        role="derived",
    )
    DataWorkspaceRepository(scenario.settings.db_path).save(
        scenario.task.id,
        DataWorkspaceDraft(
            active_dataset_id=replacement.id,
            active_dataset_content_hash=replacement.content_hash,
        ),
        expected_revision=scenario.workspace.revision,
    )

    with pytest.raises(StrategyError):
        strategy_tools.tool_apply_automatic_tree(
            _inputs(scenario, activate_result=True), scenario.ctx
        )


@pytest.mark.parametrize("drift_target", ["source", "result", "evidence"])
def test_cached_apply_revalidates_all_physical_bytes(
    scenario: _Scenario,
    drift_target: str,
) -> None:
    result = strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)
    if drift_target == "source":
        path = scenario.registry.resolve_path(scenario.dataset.id)
    elif drift_target == "result":
        path = scenario.registry.resolve_path(result["result"]["dataset_id"])
    else:
        artifact = TaskArtifactRepository(scenario.settings.db_path).get_for_task(
            scenario.task.id, result["evidence"]["artifact_id"]
        )
        assert artifact is not None
        path = Path(artifact["path"])
    path.write_bytes(path.read_bytes() + b"drift")

    with pytest.raises(Exception):
        strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)


def test_cached_apply_rejects_evidence_provenance_drift(
    scenario: _Scenario,
) -> None:
    result = strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)
    with connect(scenario.settings.db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        row = conn.execute(
            "SELECT provenance_json FROM task_artifacts WHERE id = ?",
            (result["evidence"]["artifact_id"],),
        ).fetchone()
        provenance = json.loads(row["provenance_json"])
        provenance["result_hash"] = "f" * 64
        conn.execute(
            "UPDATE task_artifacts SET provenance_json = ? WHERE id = ?",
            (
                json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                result["evidence"]["artifact_id"],
            ),
        )

    with pytest.raises(StrategyError):
        strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)


def test_failed_commit_rolls_back_files_rows_and_workspace(
    scenario: _Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_record(*_args, **_kwargs):
        raise RuntimeError("injected run registration failure")

    monkeypatch.setattr(
        AutomaticTreeApplyRepository,
        "record_succeeded_on_connection",
        fail_record,
    )

    with pytest.raises(RuntimeError, match="injected run registration failure"):
        strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)

    assert (
        DataWorkspaceRepository(scenario.settings.db_path).get_or_default(
            scenario.task.id
        )
        == scenario.workspace
    )
    assert not [
        dataset
        for dataset in scenario.registry.list_for_task(scenario.task.id)
        if dataset.role == "strategy.automatic_tree.applied"
    ]
    assert not [
        artifact
        for artifact in TaskArtifactRepository(scenario.settings.db_path).list_for_task(
            scenario.task.id
        )
        if artifact["kind"] == "strategy_automatic_tree_apply_evidence"
    ]
    result_dir = (
        scenario.settings.datasets_dir
        / scenario.task.id
        / "strategy_automatic_tree_applies"
    )
    evidence_dir = (
        scenario.settings.tasks_dir
        / scenario.task.id
        / "strategy_automatic_tree_applies"
    )
    assert not list(result_dir.glob("*.parquet"))
    assert not list(evidence_dir.glob("*.json"))


def test_post_database_commit_cleanup_failure_keeps_durable_facts(
    scenario: _Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_commit = ArtifactUnitOfWork.commit

    def fail_cleanup(_self):
        raise RuntimeError("injected post-commit cleanup failure")

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", fail_cleanup)
    with pytest.raises(RuntimeError, match="injected post-commit cleanup failure"):
        strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)

    with connect(scenario.settings.db_path) as conn:
        run = conn.execute(
            "SELECT id, result_dataset_id, evidence_artifact_id "
            "FROM strategy_automatic_tree_apply_runs WHERE task_id = ?",
            (scenario.task.id,),
        ).fetchone()
    assert run is not None
    result_dataset = scenario.registry.get(run["result_dataset_id"])
    evidence = TaskArtifactRepository(scenario.settings.db_path).get_for_task(
        scenario.task.id, run["evidence_artifact_id"]
    )
    assert scenario.registry.resolve_verified_path(result_dataset.id).is_file()
    assert evidence is not None and Path(evidence["path"]).is_file()

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", original_commit)
    replay = strategy_tools.tool_apply_automatic_tree(_inputs(scenario), scenario.ctx)
    assert replay["cached"] is True
    assert replay["run_id"] == run["id"]


def test_manifest_runner_exposes_exact_apply_contract(scenario: _Scenario) -> None:
    result = scenario.runner.invoke(
        ToolRef("strategy", "apply_automatic_tree"),
        _inputs(scenario),
        task_id=scenario.task.id,
    )

    assert result.ok, result.error
    assert result.output["schema_version"] == ("strategy.apply-automatic-tree-tool.v1")

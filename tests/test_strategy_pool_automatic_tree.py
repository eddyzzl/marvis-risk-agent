from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy import pool_tools
from marvis.packs.strategy.automatic_tree_asset import (
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
    canonical_automatic_tree_leaf_fragment_json,
    validate_automatic_tree_leaf_fragment,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    automatic_tree_leaf_selection_provenance,
    automatic_tree_source_provenance_from_asset,
    canonical_automatic_tree_leaf_selection_path,
    canonical_automatic_tree_source_path,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentExecutionBinding,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.strategy_pool import (
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.test_strategy_automatic_tree_tool import (
    _inputs as _native_tree_inputs,
    _materialize_native_sample_design_ref as _native_tree_sample_ref,
    _runtime as _native_tree_runtime,
    _tool_context as _native_tree_context,
)


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _setup(tmp_path: Path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="automatic-tree-pool",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "score": [360 + index * 20 for index in range(24)],
            "income": [3000 + (index % 8) * 800 for index in range(24)],
            "loan_amount": [1000.0 + index * 50 for index in range(24)],
            "overdue_amount": [
                0.0 if index >= 12 else 50.0 + index for index in range(24)
            ],
            "bad": [1 if index < 12 else 0 for index in range(24)],
        }
    )
    source = tmp_path / "automatic-tree-pool.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    workspaces = DataWorkspaceRepository(settings.db_path)
    activated = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "score": "score",
            "income": "numeric",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            "bad": "target",
        },
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
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    runtime = strategy_tools._runtime(ctx)
    sample = strategy_tools.tool_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-01-31",
            "maturity_status": "confirmed_matured",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": False,
        },
        ctx,
    )
    sample_design_ref = {
        "artifact_id": sample["artifact"]["artifact_id"],
        "artifact_content_hash": sample["artifact"]["content_hash"],
        "sample_design_id": sample["sample_design_id"],
        "sample_design_content_hash": sample["content_hash"],
        "partition": "development",
    }
    tree = strategy_tools.tool_build_automatic_tree_candidate(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "features": ["score", "income"],
            "directions": {"score": "decreasing", "income": "decreasing"},
            "max_depth": 2,
            "min_leaf_count": 2,
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
    return {
        "settings": settings,
        "task": task,
        "ctx": ctx,
        "runtime": runtime,
        "dataset": dataset,
        "workspace": workspace,
        "mapping": mapping,
        "sample_design_ref": sample_design_ref,
        "tree": tree,
    }


def _materialize(fx: dict, leaf_index: int = 0, *, reason: str | None = None) -> dict:
    tree = fx["tree"]
    source = next(
        item
        for item in tree["artifacts"]
        if item["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    inputs = {
        "source_artifact_id": source["artifact_id"],
        "expected_artifact_content_hash": source["content_hash"],
        "expected_asset_id": tree["summary"]["asset_id"],
        "expected_asset_hash": tree["summary"]["asset_hash"],
        "expected_tree_result_hash": tree["summary"]["tree_result_hash"],
        "leaf_id": tree["leaf_index"][leaf_index]["leaf_id"],
    }
    if reason is not None:
        inputs["selection_reason"] = reason
    return strategy_tools.tool_materialize_automatic_tree_leaf_fragment(
        inputs,
        fx["ctx"],
    )


def _add_inputs(
    candidate: dict,
    *,
    revision: int,
    snapshot_hash: str,
    action: dict | None = None,
) -> dict:
    artifact = candidate["artifacts"][0]
    asset_id = candidate.get("tree_asset_id", candidate.get("asset_id"))
    asset_hash = candidate.get("tree_asset_hash", candidate.get("asset_hash"))
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": asset_id,
        "expected_asset_hash": asset_hash,
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": action or _action("reject", reason="TREE_RISK"),
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": snapshot_hash,
    }


def test_automatic_tree_leaf_materialize_add_compile_full_chain(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    selected = _materialize(fx)
    pool_action = _action("reject", reason="POOL_ONLY")

    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            action=pool_action,
        ),
        fx["ctx"],
    )
    compiled = strategy_tools.tool_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        fx["ctx"],
    )

    source_record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id,
        next(
            item["artifact_id"]
            for item in fx["tree"]["artifacts"]
            if item["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
        ),
    )
    assert source_record is not None
    tree_asset = json.loads(Path(source_record["path"]).read_text("utf-8"))
    tree_leaf = next(
        item
        for item in tree_asset["fragments"]
        if item["leaf_id"] == selected["leaf_id"]
    )
    [entry] = added["entries"]
    assert entry["source"]["artifact_id"] == selected["artifacts"][0]["artifact_id"]
    assert entry["source"]["asset_id"] == tree_asset["asset_id"]
    assert entry["source"]["fragment_id"] == tree_leaf["fragment_id"]
    assert entry["source"]["effect_id"] == tree_leaf["effect_id"]
    assert entry["source"]["fragment_hash"] != tree_leaf["fragment_hash"]
    assert entry["execution"] == {
        "condition": tree_leaf["condition"],
        "requirements": tree_leaf["requirements"],
    }
    assert entry["action"] == pool_action
    assert "metrics" not in entry
    [compiled_rule] = compiled["strategy_spec"]["rules"]
    assert compiled_rule["condition"] == tree_leaf["condition"]
    assert compiled_rule["action"] == pool_action


def test_native_automatic_tree_leaf_materializes_current_pool_development(
    tmp_path: Path,
) -> None:
    (
        settings,
        _runner,
        _registry,
        task,
        _other,
        dataset,
        workspace,
        mapping,
    ) = _native_tree_runtime(
        tmp_path,
        target_bad_value=0,
        with_split=True,
    )
    ctx = _native_tree_context(settings, task)
    runtime = strategy_tools._runtime(ctx)
    native_ref = _native_tree_sample_ref(
        settings,
        task,
        dataset,
        workspace,
        mapping,
        target_bad_value=0,
    )
    tree = strategy_tools.tool_build_automatic_tree_candidate(
        _native_tree_inputs(dataset, workspace, mapping, native_ref),
        ctx,
    )
    source = next(
        artifact
        for artifact in tree["artifacts"]
        if artifact["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    selected = strategy_tools.tool_materialize_automatic_tree_leaf_fragment(
        {
            "source_artifact_id": source["artifact_id"],
            "expected_artifact_content_hash": source["content_hash"],
            "expected_asset_id": tree["summary"]["asset_id"],
            "expected_asset_hash": tree["summary"]["asset_hash"],
            "expected_tree_result_hash": tree["summary"]["tree_result_hash"],
            "leaf_id": tree["leaf_index"][0]["leaf_id"],
        },
        ctx,
    )
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        ctx,
    )
    compiled = strategy_tools.tool_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        ctx,
    )
    current = pool_tools.load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task.id,
        strategy_type="approval",
        expected_pool_revision=added["revision"],
        expected_pool_snapshot_hash=added["snapshot_hash"],
    )
    development = pool_tools.bind_strategy_pool_development_execution(
        runtime,
        current,
    )

    assert compiled["strategy_spec"]["rules"]
    assert isinstance(
        development.sample_design,
        StrategyRiskDevelopmentExecutionBinding,
    )
    assert development.sample_design.source_mode == "native_active_dataset"
    assert development.sample_design.to_ref_dict() == native_ref
    assert development.sample_design.reference.partition == "risk/development"


def test_automatic_tree_pool_impact_uses_exact_governed_sample_design(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    selected = _materialize(fx)
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx["ctx"],
    )

    output = strategy_tools.tool_measure_pool_impact(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
            "dataset_id": fx["dataset"].id,
            "expected_dataset_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "workspace_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(fx["mapping"]),
            "target_col": "bad",
            "sample_design_ref": fx["sample_design_ref"],
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "comparison_mode": "absolute",
            "drop_nan_labels": False,
        },
        fx["ctx"],
    )

    assert output["assessment"]["bindings"]["sample_design_ref"] == fx[
        "sample_design_ref"
    ]
    assert all(
        row["source_ref"]["sample_design_ref"] == fx["sample_design_ref"]
        for row in output["assessment"]["waterfall"]
    )


def _univariate_candidate(fx: dict) -> dict:
    source = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "analysis_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(fx["mapping"]),
            "target_col": "bad",
            "sample_design_ref": fx["sample_design_ref"],
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fx["ctx"],
    )
    report = next(
        item
        for item in source["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    method = source["candidate_evidence"]["analysis"]["features"][0]["methods"][0]
    return strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": report["artifact_id"],
            "expected_artifact_content_hash": report["content_hash"],
            "expected_candidate_id": source["candidate_id"],
            "expected_evidence_hash": source["evidence_hash"],
            "feature": "score",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {"source_bin_ids": [method["bins"][0]["id"]]},
        },
        fx["ctx"],
    )


@pytest.mark.parametrize("tree_first", [True, False])
def test_mixed_univariate_and_tree_pool_succeeds_in_both_orders(
    tmp_path: Path,
    tree_first: bool,
) -> None:
    fx = _setup(tmp_path)
    tree = _materialize(fx)
    univariate = _univariate_candidate(fx)
    ordered = [tree, univariate] if tree_first else [univariate, tree]
    revision = 0
    snapshot_hash = ABSENT_POOL_SNAPSHOT_HASH
    added = None
    for index, candidate in enumerate(ordered):
        added = strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                candidate,
                revision=revision,
                snapshot_hash=snapshot_hash,
                action=_action("reject", reason=f"MIXED_{index}"),
            ),
            fx["ctx"],
        )
        revision = added["revision"]
        snapshot_hash = added["snapshot_hash"]
    assert added is not None
    assert len(added["entries"]) == 2
    assert {entry["source"]["artifact_kind"] for entry in added["entries"]} == {
        "strategy_candidate_asset_json",
        "strategy_automatic_tree_leaf_fragment_json",
    }
    compiled = strategy_tools.tool_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": revision,
            "expected_pool_snapshot_hash": snapshot_hash,
        },
        fx["ctx"],
    )
    assert len(compiled["strategy_spec"]["rules"]) == 2


def _materialize_mismatched_sample_asset(fx: dict) -> dict:
    repository = TaskArtifactRepository(fx["settings"].db_path)
    original_record = next(
        record
        for record in repository.list_for_task(fx["task"].id)
        if record["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    original = json.loads(Path(original_record["path"]).read_text("utf-8"))
    identity = original["identity"]
    changed = build_automatic_tree_asset(
        original["tree_result"],
        task_id=fx["task"].id,
        dataset_id=identity["dataset_id"],
        dataset_content_hash=identity["dataset_content_hash"],
        workspace_revision=identity["workspace_revision"],
        workspace_generation=identity["workspace_generation"],
        semantic_mapping_hash=identity["semantic_mapping_hash"],
        registry_metadata_hash=identity["registry_metadata_hash"],
        sample_context_hash="e" * 64,
        source_refs=original["source_refs"],
    )
    content = canonical_automatic_tree_asset_json(changed).encode("utf-8")
    path = canonical_automatic_tree_source_path(
        fx["settings"].tasks_dir,
        task_id=fx["task"].id,
        asset_id=changed["asset_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = repository.register(
        task_id=fx["task"].id,
        kind=AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
        path=str(path),
        content_hash=hashlib.sha256(content).hexdigest(),
        origin_tool=AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
        provenance=automatic_tree_source_provenance_from_asset(changed),
    )
    return strategy_tools.tool_materialize_automatic_tree_leaf_fragment(
        {
            "source_artifact_id": record["id"],
            "expected_artifact_content_hash": record["content_hash"],
            "expected_asset_id": changed["asset_id"],
            "expected_asset_hash": changed["asset_hash"],
            "expected_tree_result_hash": changed["tree_result"]["result_hash"],
            "leaf_id": changed["fragments"][1]["leaf_id"],
        },
        fx["ctx"],
    )


def test_sample_context_mismatch_fails_without_pool_mutation(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            _materialize(fx),
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx["ctx"],
    )
    mismatch = _materialize_mismatched_sample_asset(fx)

    with pytest.raises(StrategyError, match="evidence identity"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                mismatch,
                revision=first["revision"],
                snapshot_hash=first["snapshot_hash"],
            ),
            fx["ctx"],
        )
    current = StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
        fx["task"].id,
        "approval",
    )
    assert current == first["pool"]
    with connect(fx["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind LIKE 'strategy.pool.%'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("drift", ["selection", "tree", "dataset", "dataset_path"])
def test_under_lock_lineage_drift_is_zero_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    repository = TaskArtifactRepository(fx["settings"].db_path)
    selection_record = repository.get_for_task(
        fx["task"].id,
        selection["artifacts"][0]["artifact_id"],
    )
    tree_record = next(
        record
        for record in repository.list_for_task(fx["task"].id)
        if record["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    assert selection_record is not None
    paths = {
        "selection": Path(selection_record["path"]),
        "tree": Path(tree_record["path"]),
        "dataset": Path(fx["runtime"].registry.resolve_verified_path(fx["dataset"].id)),
    }
    original = pool_tools._require_lineage_on_connection
    changed = False

    def drift_then_verify(conn, lineage, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            if drift == "dataset_path":
                conn.execute(
                    "UPDATE datasets SET source_path = ? WHERE id = ?",
                    ("forged/path.parquet", fx["dataset"].id),
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
    with pytest.raises(StrategyError, match="drift|changed|canonical|hash"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selection,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )
    assert (
        StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
            fx["task"].id, "approval"
        )
        is None
    )
    assert not [
        record
        for record in repository.list_for_task(fx["task"].id)
        if record["kind"] == POOL_ARTIFACT_KIND
    ]
    with connect(fx["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind LIKE 'strategy.pool.%'"
            ).fetchone()[0]
            == 0
        )


def test_same_tree_two_distinct_leaves_succeed(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    first_leaf = _materialize(fx, 0)
    second_leaf = _materialize(fx, 1)
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            first_leaf,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx["ctx"],
    )
    second = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            second_leaf,
            revision=first["revision"],
            snapshot_hash=first["snapshot_hash"],
            action=_action("review", reason="SECOND_LEAF"),
        ),
        fx["ctx"],
    )

    assert len(second["entries"]) == 2
    assert len({entry["source"]["asset_id"] for entry in second["entries"]}) == 1
    assert len({entry["source"]["fragment_id"] for entry in second["entries"]}) == 2


def test_same_leaf_different_selection_reason_is_duplicate_asset_fragment(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first_selection = _materialize(fx, reason="first review")
    second_selection = _materialize(fx, reason="second review")
    assert (
        first_selection["artifacts"][0]["artifact_id"]
        != second_selection["artifacts"][0]["artifact_id"]
    )
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            first_selection,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx["ctx"],
    )

    with pytest.raises(StrategyError, match="duplicate asset fragment"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                second_selection,
                revision=first["revision"],
                snapshot_hash=first["snapshot_hash"],
            ),
            fx["ctx"],
        )
    assert (
        StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
            fx["task"].id,
            "approval",
        )
        == first["pool"]
    )


def _drift_tree_asset(fx: dict) -> None:
    repository = TaskArtifactRepository(fx["settings"].db_path)
    record = next(
        item
        for item in repository.list_for_task(fx["task"].id)
        if item["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    path = Path(record["path"])
    path.write_bytes(path.read_bytes() + b"\n")


@pytest.mark.parametrize("operation", ["compile", "set", "reorder", "remove"])
def test_existing_pool_operations_fail_closed_after_tree_drift(
    tmp_path: Path,
    operation: str,
) -> None:
    fx = _setup(tmp_path)
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            _materialize(fx),
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx["ctx"],
    )
    [rule_id] = [entry["rule_id"] for entry in added["entries"]]
    _drift_tree_asset(fx)
    common = {
        "strategy_type": "approval",
        "expected_pool_revision": added["revision"],
        "expected_pool_snapshot_hash": added["snapshot_hash"],
    }

    with pytest.raises(StrategyError, match="drift|changed|canonical|hash"):
        if operation == "compile":
            strategy_tools.tool_compile_strategy_pool(common, fx["ctx"])
        elif operation == "set":
            strategy_tools.tool_set_pool_entry_action(
                {
                    **common,
                    "rule_id": rule_id,
                    "action": _action("review", reason="DRIFT"),
                },
                fx["ctx"],
            )
        elif operation == "reorder":
            strategy_tools.tool_reorder_strategy_pool(
                {**common, "ordered_rule_ids": [rule_id]},
                fx["ctx"],
            )
        else:
            strategy_tools.tool_remove_pool_entry(
                {**common, "rule_id": rule_id},
                fx["ctx"],
            )
    current = StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
        fx["task"].id,
        "approval",
    )
    assert current == added["pool"]


def test_identical_automatic_leaf_add_retry_is_one_revision_artifact_and_audit(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    inputs = _add_inputs(
        _materialize(fx),
        revision=0,
        snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    first = strategy_tools.tool_add_candidate_to_pool(inputs, fx["ctx"])
    replay = strategy_tools.tool_add_candidate_to_pool(inputs, fx["ctx"])

    assert replay == first
    repository = TaskArtifactRepository(fx["settings"].db_path)
    assert (
        len(
            [
                record
                for record in repository.list_for_task(fx["task"].id)
                if record["kind"] == POOL_ARTIFACT_KIND
            ]
        )
        == 1
    )
    with connect(fx["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = 'strategy.pool.add_candidate'"
            ).fetchone()[0]
            == 1
        )


def test_concurrent_different_leaves_same_absent_cas_has_one_winner(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    candidates = [_materialize(fx, 0), _materialize(fx, 1)]

    def add(candidate: dict):
        return strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                candidate,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )

    successes: list[dict] = []
    failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(add, candidate) for candidate in candidates]
        for future in futures:
            try:
                successes.append(future.result(timeout=30))
            except BaseException as exc:
                failures.append(exc)
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], StrategyError)
    with connect(fx["settings"].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "field",
    ["source_type", "leaf_id", "tree_artifact_id", "tree_path", "provenance"],
)
def test_add_input_shape_rejects_caller_tree_bindings(
    tmp_path: Path,
    field: str,
) -> None:
    fx = _setup(tmp_path)
    inputs = _add_inputs(
        _materialize(fx),
        revision=0,
        snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    with pytest.raises(StrategyError, match="unsupported"):
        strategy_tools.tool_add_candidate_to_pool(
            {**inputs, field: "forged"},
            fx["ctx"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_asset_id", "candidate-asset-" + "e" * 32),
        ("expected_asset_hash", "e" * 64),
        ("expected_artifact_content_hash", "e" * 64),
    ],
)
def test_add_rejects_expected_selection_or_tree_asset_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    fx = _setup(tmp_path)
    inputs = _add_inputs(
        _materialize(fx),
        revision=0,
        snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    with pytest.raises(StrategyError):
        strategy_tools.tool_add_candidate_to_pool(
            {**inputs, field: value},
            fx["ctx"],
        )


def _update_artifact_row(fx: dict, artifact_id: str, **changes: object) -> None:
    assignments = ", ".join(f"{field} = ?" for field in changes)
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            f"UPDATE task_artifacts SET {assignments} WHERE id = ?",  # noqa: S608
            (*changes.values(), artifact_id),
        )


@pytest.mark.parametrize("mutation", ["kind", "origin", "schema", "path"])
def test_selection_registry_triple_schema_and_path_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    artifact_id = selection["artifacts"][0]["artifact_id"]
    record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id,
        artifact_id,
    )
    assert record is not None
    if mutation == "kind":
        changes = {"kind": "forged"}
    elif mutation == "origin":
        changes = {"origin_tool": "forged"}
    elif mutation == "path":
        changes = {"path": "relative/selection.json"}
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
    _update_artifact_row(fx, artifact_id, **changes)

    with pytest.raises(StrategyError, match="unsupported|canonical|invalid"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selection,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


def test_missing_selection_row_fails_closed(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    artifact_id = selection["artifacts"][0]["artifact_id"]
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM task_artifacts WHERE id = ?", (artifact_id,))
    with pytest.raises(StrategyError, match="not found"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selection,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


def test_noncanonical_selection_bytes_fail_closed_even_with_matching_row_hash(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    candidate = deepcopy(selection)
    artifact = candidate["artifacts"][0]
    record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id,
        artifact["artifact_id"],
    )
    assert record is not None
    path = Path(record["path"])
    changed = path.read_bytes() + b"\n"
    path.write_bytes(changed)
    changed_hash = hashlib.sha256(changed).hexdigest()
    _update_artifact_row(
        fx,
        artifact["artifact_id"],
        content_hash=changed_hash,
    )
    artifact["content_hash"] = changed_hash

    with pytest.raises(StrategyError, match="canonical"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                candidate,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


@pytest.mark.parametrize("mutation", ["kind", "origin", "schema", "path", "hash"])
def test_live_tree_row_contract_drift_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    repository = TaskArtifactRepository(fx["settings"].db_path)
    record = next(
        item
        for item in repository.list_for_task(fx["task"].id)
        if item["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    if mutation == "kind":
        changes = {"kind": "forged"}
    elif mutation == "origin":
        changes = {"origin_tool": "forged"}
    elif mutation == "path":
        changes = {"path": "relative/tree.json"}
    elif mutation == "hash":
        changes = {"content_hash": "e" * 64}
    else:
        provenance = deepcopy(record["provenance"])
        provenance["asset_hash"] = "e" * 64
        changes = {
            "provenance_json": json.dumps(
                provenance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        }
    _update_artifact_row(fx, record["id"], **changes)

    with pytest.raises(StrategyError):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selection,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def test_self_authenticating_but_forged_tree_pointer_fails_live_replay(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    original = _materialize(fx)
    repository = TaskArtifactRepository(fx["settings"].db_path)
    record = repository.get_for_task(
        fx["task"].id,
        original["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    forged = json.loads(Path(record["path"]).read_text("utf-8"))
    forged["tree_artifact"]["path"] = "/forged/tree.json"
    body = {
        key: value
        for key, value in forged.items()
        if key not in {"selection_id", "selection_hash"}
    }
    forged["selection_id"] = (
        "automatic-tree-leaf-selection-"
        + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:32]
    )
    without_hash = {
        key: value for key, value in forged.items() if key != "selection_hash"
    }
    forged["selection_hash"] = hashlib.sha256(
        _canonical_json(without_hash).encode("utf-8")
    ).hexdigest()
    forged = validate_automatic_tree_leaf_fragment(forged)
    content = canonical_automatic_tree_leaf_fragment_json(forged).encode("utf-8")
    path = canonical_automatic_tree_leaf_selection_path(
        fx["settings"].tasks_dir,
        task_id=fx["task"].id,
        selection_id=forged["selection_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    forged_record = repository.register(
        task_id=fx["task"].id,
        kind=AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        path=str(path),
        content_hash=hashlib.sha256(content).hexdigest(),
        origin_tool=AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
        provenance=automatic_tree_leaf_selection_provenance(forged),
    )
    candidate = {
        **original,
        "selection_id": forged["selection_id"],
        "selection_hash": forged["selection_hash"],
        "artifacts": [
            {
                **original["artifacts"][0],
                "artifact_id": forged_record["id"],
                "filename": path.name,
                "content_hash": forged_record["content_hash"],
            }
        ],
    }

    with pytest.raises(StrategyError, match="path|pointer|binding"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                candidate,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


@pytest.mark.parametrize("artifact", ["selection", "tree"])
def test_selection_or_tree_symlink_ancestor_fails_closed(
    tmp_path: Path,
    artifact: str,
) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    repository = TaskArtifactRepository(fx["settings"].db_path)
    if artifact == "selection":
        record = repository.get_for_task(
            fx["task"].id,
            selection["artifacts"][0]["artifact_id"],
        )
        assert record is not None
    else:
        record = next(
            item
            for item in repository.list_for_task(fx["task"].id)
            if item["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
        )
    directory = Path(record["path"]).parent
    real_directory = directory.with_name(f"real-{directory.name}")
    directory.rename(real_directory)
    directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(StrategyError, match="symlink"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selection,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


def test_tree_dataset_registry_metadata_drift_fails_closed(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    selection = _materialize(fx)
    with connect(fx["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE datasets SET row_count = row_count + 1 WHERE id = ?",
            (fx["dataset"].id,),
        )

    with pytest.raises(StrategyError, match="registry metadata"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selection,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx["ctx"],
        )


def test_add_wrapper_documents_all_supported_candidate_sources() -> None:
    doc = strategy_tools.tool_add_candidate_to_pool.__doc__ or ""
    assert "univariate asset" in doc
    assert "automatic-tree leaf" in doc
    assert "Cross Matrix cell selection" in doc
    assert "Voting candidate" in doc

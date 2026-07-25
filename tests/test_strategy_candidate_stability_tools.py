from __future__ import annotations

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
from marvis.files import sha256_file
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy import candidate_stability_tools
from marvis.packs.strategy.candidate_stability import (
    validate_candidate_stability_artifact,
)
from marvis.packs.strategy.candidate_stability_tools import (
    ARTIFACT_KIND,
    resolve_candidate_monthly_stability_inputs,
    run_measure_candidate_monthly_stability,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _context(settings, task_id: str) -> ToolContext:
    return ToolContext(
        task_id=task_id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {
        "approval": "approve",
        "reject": "reject",
        "review": "review",
    }
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _setup(tmp_path: Path, *, bind_month: bool = True) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="candidate-stability",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "score": list(range(120)),
            "month": ["2026-01"] * 40 + ["2026-02"] * 40 + ["2026-03"] * 40,
            "bad": [index % 2 for index in range(120)],
        }
    )
    source_path = tmp_path / "candidate-stability.parquet"
    frame.to_parquet(source_path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(
        source_path,
        task_id=task.id,
        role="derived",
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
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "score": "score",
            "month": "month",
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
    ctx = _context(settings, task.id)
    runtime = strategy_tools._runtime(ctx)
    sample_request = {
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
        "observation_window_end": "2026-03-31",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": False,
    }
    if bind_month:
        sample_request["month_col"] = "month"
    sample = strategy_tools.tool_materialize_sample_design(sample_request, ctx)
    sample_ref = {
        "artifact_id": sample["artifact"]["artifact_id"],
        "artifact_content_hash": sample["artifact"]["content_hash"],
        "sample_design_id": sample["sample_design_id"],
        "sample_design_content_hash": sample["content_hash"],
        "partition": "development",
    }
    source = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "sample_design_ref": sample_ref,
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
        },
        ctx,
    )
    report = next(
        artifact
        for artifact in source["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    method = source["candidate_evidence"]["analysis"]["features"][0]["methods"][0]

    def refine(bin_index: int) -> dict:
        return strategy_tools.tool_refine_univariate_candidate(
            {
                "source_artifact_id": report["artifact_id"],
                "expected_artifact_content_hash": report["content_hash"],
                "expected_candidate_id": source["candidate_id"],
                "expected_evidence_hash": source["evidence_hash"],
                "feature": "score",
                "method": "equal_width",
                "merge_groups": [],
                "selection": {
                    "source_bin_ids": [method["bins"][bin_index]["id"]]
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
        "first": refine(0),
        "refine": refine,
    }


def _pool_add_inputs(
    candidate: dict,
    *,
    expected_revision: int,
    expected_hash: str,
) -> dict:
    artifact = candidate["artifacts"][0]
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": candidate["asset_id"],
        "expected_asset_hash": candidate["asset_hash"],
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": _action("reject", reason="RISK"),
        "expected_pool_revision": expected_revision,
        "expected_pool_snapshot_hash": expected_hash,
    }


def test_asset_stability_preflight_recovers_all_governed_bindings_and_persists(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    runtime = fixture["runtime"]
    candidate = fixture["first"]
    resolved = resolve_candidate_monthly_stability_inputs(
        runtime,
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "univariate_asset",
            "asset_id": candidate["asset_id"],
        },
    )

    assert set(resolved) == {
        "source_kind",
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
    }
    assert not {
        "dataset_id",
        "target_col",
        "month_col",
        "sample_design_ref",
    } & set(resolved)

    first = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        runtime,
    )
    replay = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        runtime,
    )

    assert replay == first
    assert first["basis"] == "asset_rule_hit"
    assert first["source_kind"] == "univariate_asset"
    assert first["month_col"] == "month"
    assert first["population_count"] == 120
    assert first["month_count"] == 3
    assert first["warnings"] == []
    assert first["not_created_strategy"] is True
    assert first["not_adopted"] is True
    assert first["not_deployed"] is True
    stability = validate_candidate_stability_artifact(first["stability"])
    assert [row["sample_count"] for row in stability["monthly"]] == [40, 40, 40]
    assert sum(row["hit_count"] for row in stability["monthly"]) == stability[
        "baseline"
    ]["hit_count"]

    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        first["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["kind"] == ARTIFACT_KIND
    assert sha256_file(Path(record["path"])) == record["content_hash"]
    assert json.loads(Path(record["path"]).read_text("utf-8")) == stability


def test_pool_entry_stability_uses_incremental_first_match_and_exact_pool_cas(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    second = fixture["refine"](1)
    first_pool = run_add_candidate_to_pool(
        _pool_add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    current = run_add_candidate_to_pool(
        _pool_add_inputs(
            second,
            expected_revision=first_pool["revision"],
            expected_hash=first_pool["snapshot_hash"],
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    entry_id = current["entries"][1]["entry_id"]
    resolved = resolve_candidate_monthly_stability_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "pool_entry",
            "strategy_type": "approval",
            "entry_id": entry_id,
        },
    )
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["basis"] == "pool_entry_incremental_first_match"
    assert output["source_kind"] == "pool_entry"
    stability = output["stability"]
    assert stability["source_ref"]["revision"] == current["revision"]
    assert stability["source_ref"]["entry_id"] == entry_id
    assert [row["hit_count"] for row in stability["monthly"]] == [0, 40, 0]

    with pytest.raises(StrategyError, match="stale strategy candidate pool"):
        run_measure_candidate_monthly_stability(
            {
                **resolved,
                "expected_pool_snapshot_hash": "0" * 64,
            },
            fixture["ctx"],
            fixture["runtime"],
        )


def test_stability_rejects_caller_metric_or_dataset_injection(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    artifact = fixture["first"]["artifacts"][0]
    inputs = {
        "source_kind": "univariate_asset",
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": fixture["first"]["asset_id"],
        "expected_asset_hash": fixture["first"]["asset_hash"],
    }
    for field, value in (
        ("dataset_id", fixture["dataset"].id),
        ("month_col", "month"),
        ("metrics", {"psi": 0}),
    ):
        with pytest.raises(StrategyError, match="unsupported"):
            run_measure_candidate_monthly_stability(
                {**inputs, field: value},
                fixture["ctx"],
                fixture["runtime"],
            )


def test_preflight_refuses_to_plan_without_governed_month_binding(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, bind_month=False)

    with pytest.raises(StrategyError, match="requires a month field"):
        resolve_candidate_monthly_stability_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_pointer={
                "source_kind": "univariate_asset",
                "asset_id": fixture["first"]["asset_id"],
            },
        )


def test_execution_rejects_the_row_budget_before_reading_the_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    resolved = resolve_candidate_monthly_stability_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "univariate_asset",
            "asset_id": fixture["first"]["asset_id"],
        },
    )
    monkeypatch.setattr(
        candidate_stability_tools,
        "CANDIDATE_STABILITY_MAX_ROWS",
        100,
    )

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("dataset must not be read after the row budget fails")

    monkeypatch.setattr(
        candidate_stability_tools,
        "_read_authenticated_parquet_snapshot",
        unexpected_read,
    )

    with pytest.raises(StrategyError, match="row read budget"):
        run_measure_candidate_monthly_stability(
            resolved,
            fixture["ctx"],
            fixture["runtime"],
        )


def test_execution_never_persists_evidence_from_a_restored_live_dataset_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    resolved = resolve_candidate_monthly_stability_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "univariate_asset",
            "asset_id": fixture["first"]["asset_id"],
        },
    )
    dataset_path = (
        fixture["settings"].datasets_dir / fixture["dataset"].source_path
    )
    original_bytes = dataset_path.read_bytes()
    forged = pd.read_parquet(dataset_path)
    forged["score"] = 0
    forged_path = tmp_path / "forged-candidate-stability.parquet"
    forged.to_parquet(forged_path, index=False)
    forged_bytes = forged_path.read_bytes()
    read_sources: list[object] = []
    original_read_parquet = candidate_stability_tools.pd.read_parquet

    def tamper_live_dataset_during_read(source, *args, **kwargs):
        read_sources.append(source)
        dataset_path.write_bytes(forged_bytes)
        try:
            return original_read_parquet(source, *args, **kwargs)
        finally:
            dataset_path.write_bytes(original_bytes)

    monkeypatch.setattr(
        candidate_stability_tools.pd,
        "read_parquet",
        tamper_live_dataset_during_read,
    )

    with pytest.raises(StrategyError, match="dataset changed during replay"):
        run_measure_candidate_monthly_stability(
            resolved,
            fixture["ctx"],
            fixture["runtime"],
        )

    assert len(read_sources) == 1
    assert not isinstance(read_sources[0], (str, Path))
    assert dataset_path.read_bytes() == original_bytes
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    assert not [
        record
        for record in repository.list_for_task(fixture["task"].id)
        if record["kind"] == ARTIFACT_KIND
    ]


def test_execution_rejects_candidate_artifact_drift_after_preflight(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    resolved = resolve_candidate_monthly_stability_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "univariate_asset",
            "asset_id": fixture["first"]["asset_id"],
        },
    )
    source = repository.get_for_task(
        fixture["task"].id,
        resolved["source_artifact_id"],
    )
    assert source is not None
    Path(source["path"]).write_bytes(b"{}")

    with pytest.raises(StrategyError, match="content hash drifted"):
        run_measure_candidate_monthly_stability(
            resolved,
            fixture["ctx"],
            fixture["runtime"],
        )
    assert not [
        record
        for record in repository.list_for_task(fixture["task"].id)
        if record["kind"] == ARTIFACT_KIND
    ]


def test_registration_rejects_workspace_drift_after_preflight(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    resolved = resolve_candidate_monthly_stability_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "univariate_asset",
            "asset_id": fixture["first"]["asset_id"],
        },
    )
    DataWorkspaceRepository(fixture["settings"].db_path).save(
        fixture["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=fixture["dataset"].id,
            active_dataset_content_hash=fixture["dataset"].content_hash,
            semantic_mapping=fixture["mapping"],
            page="fields",
        ),
        expected_revision=fixture["workspace"].revision,
    )

    with pytest.raises(StrategyError, match="DataWorkspace binding changed"):
        run_measure_candidate_monthly_stability(
            resolved,
            fixture["ctx"],
            fixture["runtime"],
        )
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    assert not [
        record
        for record in repository.list_for_task(fixture["task"].id)
        if record["kind"] == ARTIFACT_KIND
    ]

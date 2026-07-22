from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from marvis.agent.renderers import render_tool_output
from marvis.data.backend import DataBackend
from marvis.data.errors import NanLabelNotConfirmedError
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
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_impact import canonical_strategy_pool_impact_json
from marvis.packs.strategy.pool_impact_tools import (
    POOL_IMPACT_ARTIFACT_KIND,
    POOL_IMPACT_TOOL_SCHEMA_VERSION,
    run_measure_pool_impact,
    validate_measure_pool_impact_tool_output,
)
import marvis.packs.strategy.pool_impact_tools as impact_tools
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.strategy import build_strategy
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _action(action_type: str) -> dict:
    return {
        "type": action_type,
        "value": "approve" if action_type == "approval" else action_type,
        "reason_code": None if action_type == "approval" else "RISK",
        "stop": True,
    }


def _context(settings, task_id: str) -> ToolContext:
    return ToolContext(
        task_id=task_id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _setup(tmp_path: Path, *, candidate_target_col: str = "bad") -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="pool-impact",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "score": [100, 150, 200, 250, 300, 350, 400, 450],
            "apply_month": [
                "202601",
                "202601",
                "202601",
                "202601",
                "202602",
                "202602",
                "202602",
                "202602",
            ],
            "loan_amount": [100, 120, None, 160, 180, 200, 220, 240],
            "overdue_amount": [0, 5, 0, 10, 0, None, 20, 25],
            "bad": [0, 1, 0, 1, 0, None, 1, 1],
            "alt_bad": [1, 0, 1, 0, 1, 0, 0, 1],
        }
    )
    source = tmp_path / "impact.parquet"
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
            "apply_month": "month",
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
    ctx = _context(settings, task.id)
    runtime = strategy_tools._runtime(ctx)
    analysis = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": candidate_target_col,
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "drop_nan_labels": True,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        ctx,
    )
    report = next(
        artifact
        for artifact in analysis["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    method = analysis["candidate_evidence"]["analysis"]["features"][0]["methods"][0]
    candidate = strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": report["artifact_id"],
            "expected_artifact_content_hash": report["content_hash"],
            "expected_candidate_id": analysis["candidate_id"],
            "expected_evidence_hash": analysis["evidence_hash"],
            "feature": "score",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {"source_bin_ids": [method["bins"][0]["id"]]},
        },
        ctx,
    )
    artifact = candidate["artifacts"][0]
    added = run_add_candidate_to_pool(
        {
            "source_artifact_id": artifact["artifact_id"],
            "expected_artifact_content_hash": artifact["content_hash"],
            "expected_asset_id": candidate["asset_id"],
            "expected_asset_hash": candidate["asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject"),
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        ctx,
        runtime,
    )
    request = {
        "strategy_type": "approval",
        "expected_pool_revision": added["revision"],
        "expected_pool_snapshot_hash": added["snapshot_hash"],
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "month_col": "apply_month",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "comparison_mode": "absolute",
        "drop_nan_labels": True,
    }
    return {
        "settings": settings,
        "task": task,
        "dataset": dataset,
        "workspace": workspace,
        "mapping": mapping,
        "ctx": ctx,
        "runtime": runtime,
        "pool": added["pool"],
        "request": request,
    }


def test_measure_pool_impact_rejects_candidate_target_that_differs_from_workspace(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path, candidate_target_col="alt_bad")

    with pytest.raises(StrategyError, match="candidate target.*workspace target"):
        run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])
    assert not [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]


def test_measure_pool_impact_publishes_direct_canonical_artifact_idempotently(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])
    second = run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])

    assert first == second
    assert first["schema_version"] == POOL_IMPACT_TOOL_SCHEMA_VERSION
    assert first["monthly_status"] == "available"
    assert first["population_count"] == 8
    assert first["labeled_count"] == 7
    assert first["nan_labels_excluded"] == 1
    assert first["not_created_strategy"] is True
    assert first["not_adopted"] is True
    assert first["not_deployed"] is True
    assert len(first["artifacts"]) == 1
    assert validate_measure_pool_impact_tool_output(first) == first
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tool = next(item for item in manifest.tools if item.name == "measure_pool_impact")
    validate_against_schema(
        fx["request"], tool.input_schema, label="Pool impact input"
    )
    validate_against_schema(first, tool.output_schema, label="Pool impact output")
    record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id, first["artifacts"][0]["artifact_id"]
    )
    assert record is not None
    path = Path(record["path"])
    assert path.name == f"{first['assessment_id']}.json"
    assert path.read_text("utf-8") == canonical_strategy_pool_impact_json(
        first["assessment"]
    )
    assert sha256_file(path) == first["artifacts"][0]["content_hash"]
    assert first["content_hash"] != first["artifacts"][0]["content_hash"]
    artifacts = [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]
    assert len(artifacts) == 1
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_measure_pool_impact_cached_output_tamper_fails_closed_in_renderer(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])
    tampered = copy.deepcopy(output)
    tampered["assessment"]["population"]["population_count"] = 999999

    with pytest.raises(StrategyError, match="content_hash"):
        validate_measure_pool_impact_tool_output(tampered)
    text, tables = render_tool_output("measure_pool_impact", tampered)

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []


def test_measure_pool_impact_deep_cache_payload_never_falls_back_to_scalars(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])
    hostile = copy.deepcopy(output)
    hostile["population_count"] = 999999
    nested: dict = {}
    cursor = nested
    for _ in range(20_000):
        child: dict = {}
        cursor["nested"] = child
        cursor = child
    hostile["assessment"]["identity"]["pool_id"] = nested

    with pytest.raises(StrategyError, match="canonical JSON"):
        validate_measure_pool_impact_tool_output(hostile)
    text, tables = render_tool_output("measure_pool_impact", hostile)

    assert "结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []


def test_measure_pool_impact_renderer_shows_action_amounts_overall_and_monthly(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])

    _text, tables = render_tool_output("measure_pool_impact", output)

    overall = next(
        table for table in tables if table["title"] == "总体动作金额影响"
    )
    monthly = next(
        table for table in tables if table["title"] == "逐月动作金额影响"
    )
    approve_loan = next(
        row
        for row in overall["rows"]
        if row[0] == "approve" and row[1] == "loan_amount"
    )
    reject_loan = next(
        row
        for row in overall["rows"]
        if row[0] == "reject" and row[1] == "loan_amount"
    )
    assert approve_loan[5] != reject_loan[5]
    assert {row[0] for row in monthly["rows"]} == {"202601", "202602"}


def test_measure_pool_impact_requires_missing_label_confirmation(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = {**fx["request"], "drop_nan_labels": False}

    with pytest.raises(NanLabelNotConfirmedError) as exc_info:
        run_measure_pool_impact(request, fx["ctx"], fx["runtime"])
    assert exc_info.value.to_detail()["kind"] == "nan_label_not_confirmed"
    assert not [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]


def test_measure_pool_impact_rejects_stale_or_injected_inputs(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    with pytest.raises(StrategyError, match="stale"):
        run_measure_pool_impact(
            {**fx["request"], "expected_pool_revision": 999},
            fx["ctx"],
            fx["runtime"],
        )
    with pytest.raises(StrategyError, match="unsupported"):
        run_measure_pool_impact(
            {**fx["request"], "strategy_spec": {"rules": []}},
            fx["ctx"],
            fx["runtime"],
        )


def test_measure_pool_impact_rejects_workspace_and_dataset_drift(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = {**fx["request"], "workspace_revision": 999}
    with pytest.raises(StrategyError, match="Pool sample workspace_revision"):
        run_measure_pool_impact(request, fx["ctx"], fx["runtime"])

    registered = Path(fx["runtime"].registry.resolve_path(fx["dataset"].id))
    registered.write_bytes(registered.read_bytes() + b"drift")
    with pytest.raises(StrategyError, match="drifted|changed|hash verification"):
        run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])


def test_measure_pool_impact_supports_same_task_same_type_baseline(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    baseline = build_strategy(
        "approval",
        [
            {
                "condition": "loan_amount < 150",
                "decision": "reject",
                "value": None,
            }
        ],
        score_col="score",
        default_decision="approve",
        description="baseline",
    )
    fx["runtime"].strategies.create_strategy(fx["task"].id, baseline)
    output = run_measure_pool_impact(
        {
            **fx["request"],
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": baseline.id,
        },
        fx["ctx"],
        fx["runtime"],
    )

    assert output["assessment"]["baseline"]["status"] == "available"
    assert output["assessment"]["baseline"]["binding"]["strategy_id"] == baseline.id


def test_measure_pool_impact_missing_baseline_field_writes_nothing(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    baseline = build_strategy(
        "approval",
        [
            {
                "condition": "missing_baseline_feature < 1",
                "decision": "reject",
                "value": None,
            }
        ],
        score_col="score",
        default_decision="approve",
        description="invalid baseline field",
    )
    fx["runtime"].strategies.create_strategy(fx["task"].id, baseline)

    with pytest.raises(StrategyError, match="missing columns"):
        run_measure_pool_impact(
            {
                **fx["request"],
                "comparison_mode": "vs_baseline",
                "baseline_strategy_id": baseline.id,
            },
            fx["ctx"],
            fx["runtime"],
        )

    assert not [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]


def test_measure_pool_impact_rejects_registry_path_drift_before_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fx = _setup(tmp_path)
    original_persist = impact_tools._persist_assessment

    def drift_then_persist(*args, **kwargs):
        with sqlite3.connect(fx["settings"].db_path) as conn:
            conn.execute(
                "UPDATE datasets SET source_path = ? WHERE id = ?",
                ("drifted/location.parquet", fx["dataset"].id),
            )
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(impact_tools, "_persist_assessment", drift_then_persist)

    with pytest.raises(StrategyError, match="registry path changed"):
        run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])

    assert not [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"] == POOL_IMPACT_ARTIFACT_KIND
    ]
    assert not list(
        (
            fx["settings"].tasks_dir
            / fx["task"].id
            / "strategy_pool_impacts"
        ).glob("*.json")
    )


def test_measure_pool_impact_artifact_registry_hash_is_file_sha(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    output = run_measure_pool_impact(fx["request"], fx["ctx"], fx["runtime"])
    artifact = output["artifacts"][0]
    record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id, artifact["artifact_id"]
    )
    assert record is not None
    raw = Path(record["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record["content_hash"]
    assert json.loads(raw)["content_hash"] == output["content_hash"]

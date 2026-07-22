from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.errors import NanLabelNotConfirmedError
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.output.automatic_tree_report import render_automatic_tree_report_xlsx
from marvis.output.automatic_tree_visual import (
    render_automatic_tree_png,
    render_automatic_tree_svg,
)
from marvis.packs.strategy import automatic_tree_tools as auto_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.automatic_tree_asset import (
    canonical_automatic_tree_asset_json,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
    AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION,
    build_automatic_tree_leaf_fragment,
)
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.codegen import (
    generate_automatic_tree_duckdb_sql_source,
    generate_automatic_tree_python_source,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_tools import SAMPLE_DESIGN_ARTIFACT_KIND
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


_SAMPLE_DESIGN_REFS: dict[str, dict[str, str]] = {}


def _runtime(
    tmp_path: Path,
    *,
    target_bad_value: int = 1,
    with_split: bool = False,
    include_optional_fields: bool = True,
    weight_col: str = "weight",
    loan_amount_col: str = "loan_amount",
    overdue_amount_col: str = "overdue_amount",
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(
        plugin_registry,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
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
    task_repo = TaskRepository(settings.db_path)
    task = task_repo.create_task(
        TaskCreate(
            model_name="automatic-tree-candidate",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    other_task = task_repo.create_task(
        TaskCreate(
            model_name="foreign-automatic-tree",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    bad_one_labels = [1 if index < 12 else 0 for index in range(24)]
    frame_data = {
        "customer_id": [f"C{index:03d}" for index in range(24)],
        "phone": [f"1380000{index:04d}" for index in range(24)],
        "score": [360 + index * 20 for index in range(24)],
        "income": [3000 + (index % 8) * 800 for index in range(24)],
        weight_col: [1.0 + (index % 3) * 0.25 for index in range(24)],
        loan_amount_col: [1000.0 + index * 50 for index in range(24)],
        overdue_amount_col: [
            0.0 if index >= 12 else 50.0 + index for index in range(24)
        ],
        "ignore_me": [index for index in range(24)],
        "unused_text": [f"unused-{index}" for index in range(24)],
        "bad": (
            bad_one_labels
            if target_bad_value == 1
            else [1 - value for value in bad_one_labels]
        ),
    }
    if with_split:
        frame_data["sample_split"] = ["dev"] * 16 + ["validation"] * 4 + ["oot"] * 4
    frame = pd.DataFrame(frame_data)
    source = tmp_path / "automatic-tree.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    workspace_repo = DataWorkspaceRepository(settings.db_path)
    activated = workspace_repo.save(
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
            "customer_id": "id",
            "phone": "phone",
            "score": "score",
            "income": "numeric",
            weight_col: "weight",
            loan_amount_col: "loan_amount",
            overdue_amount_col: "overdue_amount",
            "ignore_me": "ignore",
            "bad": "target",
            **({"sample_split": "segment"} if with_split else {}),
        },
    )
    workspace = workspace_repo.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    sample_design_ref = _materialize_sample_design_ref(
        settings,
        task,
        dataset,
        workspace,
        mapping,
        target_bad_value=target_bad_value,
        with_split=with_split,
        include_optional_fields=include_optional_fields,
        weight_col=weight_col,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
    )
    _SAMPLE_DESIGN_REFS[dataset.id] = sample_design_ref
    return settings, runner, registry, task, other_task, dataset, workspace, mapping


def _inputs(dataset, workspace, mapping, sample_design_ref=None) -> dict:
    if sample_design_ref is None:
        sample_design_ref = _SAMPLE_DESIGN_REFS[dataset.id]
    return {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "analysis_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "sample_design_ref": sample_design_ref,
        "features": ["score", "income"],
        "drop_nan_labels": False,
        "sample_weight_col": "weight",
        "directions": {"score": "decreasing", "income": "decreasing"},
        "max_depth": 2,
        "min_leaf_count": 2,
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
    }


def _materialize_sample_design_ref(
    settings,
    task,
    dataset,
    workspace,
    mapping,
    *,
    target_bad_value: int = 1,
    with_split: bool = False,
    include_optional_fields: bool = True,
    drop_nan_labels: bool = False,
    weight_col: str = "weight",
    loan_amount_col: str = "loan_amount",
    overdue_amount_col: str = "overdue_amount",
) -> dict[str, str]:
    request = {
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "target_bad_value": target_bad_value,
        "performance_window_status": "provided",
        "performance_window_days": 30,
        "observation_window_status": "provided",
        "observation_window_start": "2026-01-01",
        "observation_window_end": "2026-01-31",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": drop_nan_labels,
    }
    if include_optional_fields:
        request.update(
            {
                "weight_col": weight_col,
                "loan_amount_col": loan_amount_col,
                "overdue_amount_col": overdue_amount_col,
            }
        )
    if with_split:
        request.update(
            {
                "split_col": "sample_split",
                "development_values": ["dev"],
                "validation_values": ["validation"],
                "oot_values": ["oot"],
            }
        )
    output = strategy_tools.tool_materialize_sample_design(
        request,
        _tool_context(settings, task),
    )
    return {
        "artifact_id": output["artifact"]["artifact_id"],
        "artifact_content_hash": output["artifact"]["content_hash"],
        "sample_design_id": output["sample_design_id"],
        "sample_design_content_hash": output["content_hash"],
        "partition": "development",
    }


def _tool_context(settings, task) -> ToolContext:
    return ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _activate_frame(
    tmp_path: Path,
    *,
    settings,
    registry: DatasetRegistry,
    task,
    current_workspace,
    frame: pd.DataFrame,
    mapping: DataSemanticMapping,
):
    source = tmp_path / f"replacement-{current_workspace.revision}.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    repository = DataWorkspaceRepository(settings.db_path)
    reset = repository.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=current_workspace.revision,
    )
    workspace = repository.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=reset.revision,
    )
    return dataset, workspace


def _record_by_kind(settings, task) -> dict[str, dict]:
    return {
        record["kind"]: record for record in _automatic_tree_records(settings, task)
    }


def _automatic_tree_records(settings, task) -> list[dict]:
    return [
        record
        for record in TaskArtifactRepository(settings.db_path).list_for_task(task.id)
        if record["origin_tool"] == "strategy.build_automatic_tree_candidate"
    ]


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key).lower() for key in value)
        for child in value.values():
            keys.update(_all_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_mapping_keys(child))
    return keys


def test_automatic_tree_tool_happy_path_is_six_artifact_idempotent(
    tmp_path: Path,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = _inputs(dataset, workspace, mapping)
    ctx = _tool_context(settings, task)

    first = strategy_tools.tool_build_automatic_tree_candidate(inputs, ctx)
    repeated = strategy_tools.tool_build_automatic_tree_candidate(inputs, ctx)

    assert first == repeated
    assert set(first) == {
        "schema_version",
        "summary",
        "leaf_index",
        "report_info_gaps",
        "red_flags",
        "equivalence",
        "artifacts",
    }
    assert first["schema_version"] == "strategy.automatic-tree-candidate-tool.v2"
    assert first["summary"]["candidate_stage"] == "development"
    assert first["summary"]["observation_stage"] == "backtested"
    assert first["summary"]["validation_status"] == "unvalidated"
    assert first["summary"]["sample_design_ref"] == _SAMPLE_DESIGN_REFS[dataset.id]
    assert first["equivalence"]["matched"] is True
    assert first["equivalence"]["sample_count"] == 24
    assert (
        len(
            {
                first["equivalence"]["reference_result_hash"],
                first["equivalence"]["python_result_hash"],
                first["equivalence"]["duckdb_sql_result_hash"],
            }
        )
        == 1
    )
    assert len(first["leaf_index"]) == first["summary"]["leaf_count"]
    assert first["report_info_gaps"] == []
    assert len(first["artifacts"]) == 6
    assert all("path" not in artifact for artifact in first["artifacts"])
    assert all(
        set(artifact)
        == {
            "artifact_id",
            "kind",
            "format",
            "filename",
            "content_hash",
            "download_url",
        }
        for artifact in first["artifacts"]
    )
    records = _automatic_tree_records(settings, task)
    assert len(records) == 6
    assert {record["id"] for record in records} == {
        artifact["artifact_id"] for artifact in first["artifacts"]
    }
    for record in records:
        assert Path(record["path"]).is_file()

    by_kind = _record_by_kind(settings, task)
    json_record = by_kind[AUTOMATIC_TREE_ASSET_ARTIFACT_KIND]
    asset = validate_automatic_tree_asset(
        json.loads(Path(json_record["path"]).read_text(encoding="utf-8"))
    )
    assert first["leaf_index"] == [
        {
            "leaf_id": fragment["leaf_id"],
            "fragment_id": fragment["fragment_id"],
            "fragment_hash": fragment["fragment_hash"],
            "rule_id": fragment["rule_id"],
            "effect_id": fragment["effect_id"],
            "condition": fragment["condition"],
            "requirements": fragment["requirements"],
            "metric_basis": {
                "primary": "weighted",
                "sample_weight": {"status": "available", "column": "weight"},
            },
            "measurements": fragment["metrics"],
        }
        for fragment in asset["fragments"]
    ]
    expected_bytes = {
        AUTOMATIC_TREE_ASSET_ARTIFACT_KIND: canonical_automatic_tree_asset_json(
            asset
        ).encode("utf-8"),
        "strategy_automatic_tree_python": generate_automatic_tree_python_source(
            asset
        ).encode("utf-8"),
        "strategy_automatic_tree_duckdb_sql": (
            generate_automatic_tree_duckdb_sql_source(asset).encode("utf-8")
        ),
        "strategy_automatic_tree_svg": render_automatic_tree_svg(asset),
        "strategy_automatic_tree_png": render_automatic_tree_png(asset),
        "strategy_automatic_tree_xlsx": render_automatic_tree_report_xlsx(asset),
    }
    assert set(by_kind) == set(expected_bytes)
    for kind, expected in expected_bytes.items():
        record = by_kind[kind]
        persisted = Path(record["path"]).read_bytes()
        assert persisted == expected
        assert hashlib.sha256(persisted).hexdigest() == record["content_hash"]
        assert record["origin_tool"] == "strategy.build_automatic_tree_candidate"

    assert set(json_record["provenance"]) == auto_tools.FULL_TREE_PROVENANCE_FIELDS
    assert json_record["provenance"]["schema_version"] == (
        AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION
    )
    assert json_record["provenance"]["asset_hash"] == asset["asset_hash"]
    assert (
        json_record["provenance"]["sample_design_ref"]
        == _SAMPLE_DESIGN_REFS[dataset.id]
    )
    assert (
        auto_tools.strategy_sample_design_ref_from_source_refs(asset["source_refs"])
        == _SAMPLE_DESIGN_REFS[dataset.id]
    )
    selected_leaf = build_automatic_tree_leaf_fragment(
        asset,
        tree_artifact_binding={
            "artifact_id": json_record["id"],
            "task_id": task.id,
            "kind": json_record["kind"],
            "artifact_schema_version": AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION,
            "content_hash": json_record["content_hash"],
            "origin_tool": json_record["origin_tool"],
            "path": json_record["path"],
            "provenance": json_record["provenance"],
            "canonical_bytes": expected_bytes[AUTOMATIC_TREE_ASSET_ARTIFACT_KIND],
        },
        leaf_id=asset["fragments"][0]["leaf_id"],
    )
    assert selected_leaf["leaf"]["fragment_id"] == asset["fragments"][0]["fragment_id"]
    canonical_hash = hashlib.sha256(
        expected_bytes[AUTOMATIC_TREE_ASSET_ARTIFACT_KIND]
    ).hexdigest()
    for kind, record in by_kind.items():
        if kind == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND:
            continue
        provenance = record["provenance"]
        assert set(provenance) == auto_tools.DELIVERY_PROVENANCE_FIELDS
        assert provenance["asset_id"] == asset["asset_id"]
        assert provenance["asset_hash"] == asset["asset_hash"]
        assert provenance["tree_result_hash"] == asset["tree_result"]["result_hash"]
        assert provenance["canonical_asset_content_hash"] == canonical_hash
        assert (
            provenance["equivalence_sample_hash"] == first["equivalence"]["sample_hash"]
        )
        assert provenance["equivalence_sample_count"] == 24

    client = TestClient(create_app(settings))
    expected_media_types = {
        AUTOMATIC_TREE_ASSET_ARTIFACT_KIND: "application/json",
        "strategy_automatic_tree_python": "text/x-python",
        "strategy_automatic_tree_duckdb_sql": "application/sql",
        "strategy_automatic_tree_svg": "image/svg+xml",
        "strategy_automatic_tree_png": "image/png",
        "strategy_automatic_tree_xlsx": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    }
    for kind, record in by_kind.items():
        downloaded = client.get(
            f"/api/tasks/{task.id}/task-artifacts/{record['id']}/download"
        )
        foreign = client.get(
            f"/api/tasks/{_other.id}/task-artifacts/{record['id']}/download"
        )

        assert downloaded.status_code == 200
        assert downloaded.content == expected_bytes[kind]
        assert (
            downloaded.headers["content-type"].split(";", 1)[0]
            == (expected_media_types[kind])
        )
        assert downloaded.headers["content-length"] == str(len(expected_bytes[kind]))
        assert downloaded.headers["content-disposition"].startswith("attachment;")
        assert Path(record["path"]).name in downloaded.headers["content-disposition"]
        if kind == "strategy_automatic_tree_svg":
            assert downloaded.headers["x-content-type-options"] == "nosniff"
            relative_path = Path(record["path"]).relative_to(settings.workspace)
            generic_download = client.get(f"/api/artifacts/{relative_path.as_posix()}")
            assert generic_download.status_code == 200
            assert generic_download.content == expected_bytes[kind]
            assert generic_download.headers["content-disposition"].startswith(
                "attachment;"
            )
            assert generic_download.headers["x-content-type-options"] == "nosniff"
        assert foreign.status_code == 404

    for record in by_kind.values():
        path = Path(record["path"])
        path.write_bytes(path.read_bytes() + b"drift")
    for record in by_kind.values():
        drifted = client.get(
            f"/api/tasks/{task.id}/task-artifacts/{record['id']}/download"
        )
        assert drifted.status_code == 409
        assert drifted.json()["detail"] == "task artifact integrity check failed"

    forbidden = {
        "action",
        "adoption",
        "pool",
        "recommendation",
        "rank",
        "rules",
        "tree",
        "result",
        "metrics",
        "path",
    }
    assert not (_all_mapping_keys(first) & forbidden)


def test_automatic_tree_inherits_custom_optional_columns_from_sample_design(
    tmp_path: Path,
) -> None:
    optional = {
        "sample_weight_col": "development_exposure_weight",
        "loan_amount_col": "booked_principal_balance",
        "overdue_amount_col": "observed_delinquent_balance",
    }
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path,
        weight_col=optional["sample_weight_col"],
        loan_amount_col=optional["loan_amount_col"],
        overdue_amount_col=optional["overdue_amount_col"],
    )
    inputs = _inputs(dataset, workspace, mapping)
    for field in optional:
        inputs.pop(field)

    with pytest.raises(StrategyError, match="weight_col"):
        strategy_tools.tool_build_automatic_tree_candidate(
            {**inputs, "sample_weight_col": "score"},
            _tool_context(settings, task),
        )

    output = strategy_tools.tool_build_automatic_tree_candidate(
        inputs,
        _tool_context(settings, task),
    )

    assert output["report_info_gaps"] == []
    record = _record_by_kind(settings, task)[AUTOMATIC_TREE_ASSET_ARTIFACT_KIND]
    asset = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    training = asset["tree_result"]["training"]
    assert training["sample_weight"] == {
        "status": "available",
        "column": optional["sample_weight_col"],
    }
    assert training["loan_amount_col"] == optional["loan_amount_col"]
    assert training["overdue_amount_col"] == optional["overdue_amount_col"]


def test_automatic_tree_tool_reports_only_nonblocking_optional_context_gaps(
    tmp_path: Path,
) -> None:
    _settings, runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path,
        include_optional_fields=False,
    )
    inputs = {
        **_inputs(dataset, workspace, mapping),
        "sample_weight_col": None,
        "loan_amount_col": None,
        "overdue_amount_col": None,
    }

    result = runner.invoke(
        ToolRef("strategy", "build_automatic_tree_candidate"),
        inputs,
        task_id=task.id,
    )
    assert result.ok, result.error
    output = result.output

    assert output["report_info_gaps"] == [
        {
            "code": "sample_weight_not_provided",
            "context": "sample_weight",
            "blocking": False,
        },
        {
            "code": "loan_amount_not_provided",
            "context": "loan_amount",
            "blocking": False,
        },
        {
            "code": "overdue_amount_not_provided",
            "context": "overdue_amount",
            "blocking": False,
        },
    ]
    for leaf in output["leaf_index"]:
        assert leaf["metric_basis"] == {
            "primary": "unweighted",
            "sample_weight": {"status": "not_applicable"},
        }
        assert leaf["measurements"]["weighted"] == {"status": "not_applicable"}
        assert set(leaf["measurements"]["unweighted"]) == {
            "total",
            "good",
            "bad",
            "bad_rate",
            "share",
            "bad_capture",
            "lift",
        }


def test_automatic_tree_tool_projects_only_required_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original = DataBackend.read_frame
    projections: list[list[str] | None] = []

    def tracked_read(self, path, *, columns=None, nrows=None):
        projections.append(None if columns is None else list(columns))
        return original(self, path, columns=columns, nrows=nrows)

    monkeypatch.setattr(DataBackend, "read_frame", tracked_read)

    strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset, workspace, mapping),
        _tool_context(settings, task),
    )

    assert projections == [
        ["income", "score", "weight", "loan_amount", "overdue_amount", "bad"]
    ]


def test_manifest_runner_accepts_the_exact_automatic_tree_contract(
    tmp_path: Path,
) -> None:
    _settings, runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )

    result = runner.invoke(
        ToolRef("strategy", "build_automatic_tree_candidate"),
        _inputs(dataset, workspace, mapping),
        task_id=task.id,
    )

    assert result.ok, result.error
    assert result.output["schema_version"] == auto_tools.TOOL_SCHEMA_VERSION
    assert len(result.output["artifacts"]) == 6


def test_automatic_tree_tool_requires_exact_sample_design_ref(tmp_path: Path) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = _inputs(dataset, workspace, mapping)
    del inputs["sample_design_ref"]

    with pytest.raises(StrategyError, match="sample_design_ref"):
        strategy_tools.tool_build_automatic_tree_candidate(
            inputs,
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("artifact_id", "0" * 64),
        ("artifact_content_hash", "0" * 64),
        ("sample_design_id", "strategy-sample-design-forged"),
        ("sample_design_content_hash", "0" * 64),
        ("partition", "validation"),
    ],
)
def test_automatic_tree_tool_rejects_sample_design_reference_tamper(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    reference = _SAMPLE_DESIGN_REFS[dataset.id]

    with pytest.raises(StrategyError, match="sample.design|artifact"):
        strategy_tools.tool_build_automatic_tree_candidate(
            {
                **_inputs(dataset, workspace, mapping),
                "sample_design_ref": {**reference, field: replacement},
            },
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_automatic_tree_tool_rejects_sample_design_artifact_drift(
    tmp_path: Path,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    record = next(
        item
        for item in TaskArtifactRepository(settings.db_path).list_for_task(task.id)
        if item["kind"] == SAMPLE_DESIGN_ARTIFACT_KIND
    )
    path = Path(record["path"])
    path.write_bytes(path.read_bytes() + b"drift")

    with pytest.raises(StrategyError, match="content hash|binding changed"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("expected_content_hash", "f" * 64),
        ("workspace_revision", -1),
        ("analysis_generation", 99),
        ("semantic_mapping_hash", "e" * 64),
    ],
)
def test_automatic_tree_tool_rejects_stale_confirmed_binding(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = {**_inputs(dataset, workspace, mapping), field: replacement}

    with pytest.raises(StrategyError, match="binding changed|non-negative"):
        strategy_tools.tool_build_automatic_tree_candidate(
            inputs,
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_automatic_tree_tool_rejects_foreign_and_nonactive_datasets(
    tmp_path: Path,
) -> None:
    settings, _runner, registry, task, other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = _inputs(dataset, workspace, mapping)

    with pytest.raises(StrategyError, match="dataset not found"):
        strategy_tools.tool_build_automatic_tree_candidate(
            inputs,
            _tool_context(settings, other),
        )

    replacement_frame = pd.read_parquet(registry.resolve_path(dataset.id)).assign(
        score=lambda frame: frame["score"] + 1
    )
    replacement, _replacement_workspace = _activate_frame(
        tmp_path,
        settings=settings,
        registry=registry,
        task=task,
        current_workspace=workspace,
        frame=replacement_frame,
        mapping=mapping,
    )
    assert replacement.id != dataset.id
    with pytest.raises(StrategyError, match="current active"):
        strategy_tools.tool_build_automatic_tree_candidate(
            inputs,
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_automatic_tree_tool_rejects_physical_content_drift(
    tmp_path: Path,
) -> None:
    settings, _runner, registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    registry.resolve_path(dataset.id).write_bytes(b"out-of-band drift")

    with pytest.raises(StrategyError, match="immutable hash verification"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_automatic_tree_tool_enforces_nan_label_confirmation(
    tmp_path: Path,
) -> None:
    settings, _runner, registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    frame = pd.read_parquet(registry.resolve_path(dataset.id))
    frame.loc[3, "bad"] = float("nan")
    nan_dataset, nan_workspace = _activate_frame(
        tmp_path,
        settings=settings,
        registry=registry,
        task=task,
        current_workspace=workspace,
        frame=frame,
        mapping=mapping,
    )
    with pytest.raises(NanLabelNotConfirmedError):
        _materialize_sample_design_ref(
            settings,
            task,
            nan_dataset,
            nan_workspace,
            mapping,
            drop_nan_labels=False,
        )
    sample_design_ref = _materialize_sample_design_ref(
        settings,
        task,
        nan_dataset,
        nan_workspace,
        mapping,
        drop_nan_labels=True,
    )
    inputs = _inputs(nan_dataset, nan_workspace, mapping, sample_design_ref)

    output = strategy_tools.tool_build_automatic_tree_candidate(
        {**inputs, "drop_nan_labels": True},
        _tool_context(settings, task),
    )
    assert output["summary"]["nan_labels_dropped"] == 1
    assert output["summary"]["training_row_count"] == 23


@pytest.mark.parametrize("feature", ["customer_id", "phone", "ignore_me"])
def test_automatic_tree_tool_rejects_sensitive_or_ignored_features(
    tmp_path: Path,
    feature: str,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )

    with pytest.raises(StrategyError, match="personal-data|ignored"):
        strategy_tools.tool_build_automatic_tree_candidate(
            {
                **_inputs(dataset, workspace, mapping),
                "features": [feature],
                "directions": {feature: "unordered"},
            },
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_registry_sensitive_role_cannot_be_downgraded_by_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original_get = DatasetRegistry.get

    def registry_marks_customer_id_sensitive(self, dataset_id):
        loaded = original_get(self, dataset_id)
        return replace(
            loaded,
            columns=tuple(
                replace(profile, semantic_role="id")
                if profile.name == "customer_id"
                else profile
                for profile in loaded.columns
            ),
        )

    monkeypatch.setattr(DatasetRegistry, "get", registry_marks_customer_id_sensitive)
    downgraded_roles = dict(mapping.field_roles)
    downgraded_roles["customer_id"] = "numeric"
    downgraded = DataSemanticMapping(
        target_col="bad",
        field_roles=downgraded_roles,
    )
    changed = DataWorkspaceRepository(settings.db_path).save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=downgraded,
        ),
        expected_revision=workspace.revision,
    )
    sample_design_ref = _materialize_sample_design_ref(
        settings,
        task,
        dataset,
        changed,
        downgraded,
    )

    with pytest.raises(StrategyError, match="personal-data"):
        strategy_tools.tool_build_automatic_tree_candidate(
            {
                **_inputs(dataset, changed, downgraded, sample_design_ref),
                "features": ["customer_id"],
                "directions": {"customer_id": "unordered"},
            },
            _tool_context(settings, task),
        )


def test_automatic_tree_tool_requires_workspace_target_match(
    tmp_path: Path,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = {**_inputs(dataset, workspace, mapping), "target_col": "score"}

    with pytest.raises(
        StrategyError, match="sample-design.*target_col|semantic target"
    ):
        strategy_tools.tool_build_automatic_tree_candidate(
            inputs,
            _tool_context(settings, task),
        )


@pytest.mark.parametrize(
    "injected",
    ["metrics", "rules", "tree", "result", "action", "adoption", "pool"],
)
def test_automatic_tree_tool_rejects_caller_supplied_results_and_actions(
    tmp_path: Path,
    injected: str,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )

    with pytest.raises(StrategyError, match="caller cannot supply"):
        strategy_tools.tool_build_automatic_tree_candidate(
            {**_inputs(dataset, workspace, mapping), injected: {}},
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_duckdb_preflight_is_immediately_before_same_frame_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original_connect = auto_tools.duckdb.connect
    event: dict[str, object] = {}

    def tracked_preflight(frame, asset, *, additional_feature_fields=None):
        assert isinstance(frame, pd.DataFrame)
        if additional_feature_fields is not None:
            assert additional_feature_fields == ["income", "score"]
            assert len(frame) == 24
            assert frame.columns.tolist() == ["income", "score"]
            event["full_frame_validated"] = True
            return frame
        assert event.get("full_frame_validated") is True
        event["frame"] = frame
        event["columns"] = frame.columns.tolist()
        event["event"] = "preflight"
        return frame

    class TrackedConnection:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def register(self, name, frame):
            if name == "input_rows":
                assert event.get("event") == "preflight"
                assert frame is event["frame"]
                assert frame.columns.tolist() == event["columns"]
                event["event"] = "registered"
            return self.inner.register(name, frame)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def tracked_connect(*args, **kwargs):
        return TrackedConnection(original_connect(*args, **kwargs))

    monkeypatch.setattr(
        auto_tools,
        "validate_automatic_tree_duckdb_input_frame",
        tracked_preflight,
    )
    monkeypatch.setattr(auto_tools.duckdb, "connect", tracked_connect)

    output = strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset, workspace, mapping),
        _tool_context(settings, task),
    )

    assert output["equivalence"]["matched"] is True
    assert event["event"] == "registered"


def test_generated_python_mismatch_fails_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original = auto_tools.generate_automatic_tree_python_source

    def mismatched_source(asset):
        return original(asset) + (
            "\n_marvis_original_apply_rows = apply_rows\n"
            "def apply_rows(rows):\n"
            "    result = _marvis_original_apply_rows(rows)\n"
            "    if result:\n"
            "        result[0] = {'leaf_id': 'leaf-mismatch', "
            "'rule_id': 'rule-mismatch'}\n"
            "    return result\n"
        )

    monkeypatch.setattr(
        auto_tools,
        "generate_automatic_tree_python_source",
        mismatched_source,
    )

    with pytest.raises(StrategyError, match="Python does not match"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []
    output_dir = settings.tasks_dir / task.id / "strategy_automatic_trees"
    assert not output_dir.exists() or not any(output_dir.rglob("*"))


def test_generated_sql_mismatch_fails_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original = auto_tools.generate_automatic_tree_duckdb_sql_source

    def mismatched_source(asset):
        source = original(asset)
        first_rule_id = asset["tree_result"]["rules"][0]["rule_id"]
        return source.replace(first_rule_id, "rule-mismatch", 1)

    monkeypatch.setattr(
        auto_tools,
        "generate_automatic_tree_duckdb_sql_source",
        mismatched_source,
    )

    with pytest.raises(StrategyError, match="DuckDB SQL does not match"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_render_failure_occurs_before_staging_or_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )

    def fail_render(_asset):
        raise RuntimeError("injected PNG render failure")

    monkeypatch.setattr(auto_tools, "render_automatic_tree_png", fail_render)

    with pytest.raises(RuntimeError, match="injected PNG render failure"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []
    output_dir = settings.tasks_dir / task.id / "strategy_automatic_trees"
    assert not output_dir.exists()


def test_automatic_tree_artifact_directory_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    task_root = settings.tasks_dir / task.id
    task_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (task_root / "strategy_automatic_trees").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(StrategyError, match="must not be a symlink"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []
    assert list(outside.iterdir()) == []


def test_source_drift_after_compute_is_rejected_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original = auto_tools.render_automatic_tree_report_xlsx

    def mutate_after_render(asset):
        rendered = original(asset)
        registry.resolve_path(dataset.id).write_bytes(b"post-compute drift")
        return rendered

    monkeypatch.setattr(
        auto_tools,
        "render_automatic_tree_report_xlsx",
        mutate_after_render,
    )

    with pytest.raises(StrategyError, match="source dataset changed"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []


def test_workspace_change_after_compute_is_preserved_and_blocks_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original = auto_tools.render_automatic_tree_report_xlsx
    repository = DataWorkspaceRepository(settings.db_path)
    changed = None

    def mutate_after_render(asset):
        nonlocal changed
        rendered = original(asset)
        changed_mapping = DataSemanticMapping(
            target_col="bad",
            field_roles=dict(mapping.field_roles),
            business_names={"score": "updated score semantics"},
        )
        changed = repository.save(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=dataset.id,
                active_dataset_content_hash=dataset.content_hash,
                semantic_mapping=changed_mapping,
            ),
            expected_revision=workspace.revision,
        )
        return rendered

    monkeypatch.setattr(
        auto_tools,
        "render_automatic_tree_report_xlsx",
        mutate_after_render,
    )

    with pytest.raises(StrategyError, match="workspace changed"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert changed is not None
    assert repository.get_or_default(task.id).revision == changed.revision
    assert _automatic_tree_records(settings, task) == []


@pytest.mark.parametrize("failed_registration", [2, 3, 4, 5, 6])
def test_any_late_registration_failure_rolls_back_all_six_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_registration: int,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    original = TaskArtifactRepository.register_on_connection
    calls = 0

    def fail_selected(self, conn, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_registration:
            raise RuntimeError(f"injected registration {failed_registration} failure")
        return original(self, conn, **kwargs)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_selected,
    )

    with pytest.raises(RuntimeError, match="injected registration"):
        strategy_tools.tool_build_automatic_tree_candidate(
            _inputs(dataset, workspace, mapping),
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []
    output_dir = settings.tasks_dir / task.id / "strategy_automatic_trees"
    assert not output_dir.exists() or not any(output_dir.rglob("*"))


def test_concurrent_identical_writers_leave_exactly_six_intact_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = _inputs(dataset, workspace, mapping)
    original = auto_tools.render_automatic_tree_report_xlsx
    render_barrier = threading.Barrier(2)

    def synchronized_render(asset):
        rendered = original(asset)
        render_barrier.wait(timeout=15)
        return rendered

    monkeypatch.setattr(
        auto_tools,
        "render_automatic_tree_report_xlsx",
        synchronized_render,
    )
    outputs: dict[str, dict] = {}
    failures: dict[str, BaseException] = {}

    def invoke(name: str) -> None:
        try:
            outputs[name] = strategy_tools.tool_build_automatic_tree_candidate(
                inputs,
                _tool_context(settings, task),
            )
        except BaseException as exc:  # captured for main-thread assertions
            failures[name] = exc

    writers = [
        threading.Thread(target=invoke, args=(name,), name=f"writer-{name}")
        for name in ("a", "b")
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=30)

    assert all(not writer.is_alive() for writer in writers)
    assert failures == {}
    assert outputs["a"] == outputs["b"]
    records = _automatic_tree_records(settings, task)
    assert len(records) == 6
    assert len({record["path"] for record in records}) == 6
    for record in records:
        assert sha256_file(Path(record["path"])) == record["content_hash"]


def test_failed_writer_rolls_back_before_identical_peer_can_promote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = _inputs(dataset, workspace, mapping)
    failed_writer_exited_db = threading.Event()
    release_failed_writer = threading.Event()
    original_register = TaskArtifactRepository.register_on_connection
    original_transaction = TaskArtifactRepository.transaction
    failing_calls = 0

    def fail_fourth_for_first_writer(self, conn, **kwargs):
        nonlocal failing_calls
        if threading.current_thread().name == "failing-writer":
            failing_calls += 1
            if failing_calls == 4:
                raise RuntimeError("injected post-promotion registration failure")
        return original_register(self, conn, **kwargs)

    @contextmanager
    def pause_failed_writer_after_db_exit(self):
        try:
            with original_transaction(self) as conn:
                yield conn
        finally:
            if threading.current_thread().name == "failing-writer":
                failed_writer_exited_db.set()
                if not release_failed_writer.wait(timeout=15):
                    raise RuntimeError("timed out waiting to release failed writer")

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_fourth_for_first_writer,
    )
    monkeypatch.setattr(
        TaskArtifactRepository,
        "transaction",
        pause_failed_writer_after_db_exit,
    )
    outputs: dict[str, dict] = {}
    failures: dict[str, BaseException] = {}

    def invoke(name: str) -> None:
        try:
            outputs[name] = strategy_tools.tool_build_automatic_tree_candidate(
                inputs,
                _tool_context(settings, task),
            )
        except BaseException as exc:  # captured for main-thread assertions
            failures[name] = exc

    failing = threading.Thread(
        target=invoke,
        args=("failing",),
        name="failing-writer",
    )
    peer = threading.Thread(target=invoke, args=("peer",), name="peer-writer")
    failing.start()
    assert failed_writer_exited_db.wait(timeout=20)
    peer.start()
    peer.join(timeout=30)
    assert not peer.is_alive()
    assert "peer" not in failures
    release_failed_writer.set()
    failing.join(timeout=30)

    assert not failing.is_alive()
    assert isinstance(failures.get("failing"), RuntimeError)
    assert outputs["peer"]["summary"]["validation_status"] == "unvalidated"
    records = _automatic_tree_records(settings, task)
    assert len(records) == 6
    for record in records:
        path = Path(record["path"])
        assert path.is_file()
        assert sha256_file(path) == record["content_hash"]


def test_writer_lock_precedes_promotion_for_identical_final_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    inputs = _inputs(dataset, workspace, mapping)
    late_writer_holds_lock = threading.Event()
    release_late_writer = threading.Event()
    peer_promoted = threading.Event()
    original_require = auto_tools._require_binding_on_connection
    original_promote = ArtifactUnitOfWork.promote_all

    def gated_require(conn, **kwargs):
        if threading.current_thread().name == "late-writer" and conn.in_transaction:
            late_writer_holds_lock.set()
            if not release_late_writer.wait(timeout=15):
                raise RuntimeError("timed out waiting to release late writer")
            raise StrategyError("injected late binding failure")
        return original_require(conn, **kwargs)

    def tracked_promote(self):
        if threading.current_thread().name == "peer-writer":
            peer_promoted.set()
        return original_promote(self)

    monkeypatch.setattr(auto_tools, "_require_binding_on_connection", gated_require)
    monkeypatch.setattr(ArtifactUnitOfWork, "promote_all", tracked_promote)
    failures: dict[str, BaseException] = {}
    outputs: dict[str, dict] = {}

    def invoke(name: str) -> None:
        try:
            outputs[name] = strategy_tools.tool_build_automatic_tree_candidate(
                inputs,
                _tool_context(settings, task),
            )
        except BaseException as exc:  # captured for main-thread assertions
            failures[name] = exc

    late = threading.Thread(target=invoke, args=("late",), name="late-writer")
    peer = threading.Thread(target=invoke, args=("peer",), name="peer-writer")
    late.start()
    assert late_writer_holds_lock.wait(timeout=15)
    peer.start()
    assert not peer_promoted.wait(timeout=1)
    release_late_writer.set()
    late.join(timeout=30)
    peer.join(timeout=30)

    assert not late.is_alive()
    assert not peer.is_alive()
    assert isinstance(failures.get("late"), StrategyError)
    assert "peer" not in failures
    assert outputs["peer"]["summary"]["validation_status"] == "unvalidated"
    assert len(_automatic_tree_records(settings, task)) == 6


def test_sample_context_hash_matches_actual_univariate_candidate_same_sample(
    tmp_path: Path,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    ctx = _tool_context(settings, task)
    automatic = strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset, workspace, mapping),
        ctx,
    )
    univariate = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "sample_design_ref": _SAMPLE_DESIGN_REFS[dataset.id],
            "drop_nan_labels": False,
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "min_bin_pct": 0.02,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "sentinel_values": [],
        },
        ctx,
    )

    assert automatic["summary"]["sample_context_hash"] == (
        sample_context_hash_from_candidate_evidence(univariate["candidate_evidence"])
    )


def test_automatic_tree_normalizes_bad_zero_to_same_measured_tree(
    tmp_path: Path,
) -> None:
    first = _runtime(tmp_path / "bad-one", target_bad_value=1)
    second = _runtime(tmp_path / "bad-zero", target_bad_value=0)
    (
        settings_one,
        _runner,
        _registry,
        task_one,
        _other,
        dataset_one,
        workspace_one,
        mapping_one,
    ) = first
    (
        settings_zero,
        _runner,
        _registry,
        task_zero,
        _other,
        dataset_zero,
        workspace_zero,
        mapping_zero,
    ) = second

    output_one = strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset_one, workspace_one, mapping_one),
        _tool_context(settings_one, task_one),
    )
    output_zero = strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset_zero, workspace_zero, mapping_zero),
        _tool_context(settings_zero, task_zero),
    )

    measured_one = [
        (item["condition"], item["measurements"], item["metric_basis"])
        for item in output_one["leaf_index"]
    ]
    measured_zero = [
        (item["condition"], item["measurements"], item["metric_basis"])
        for item in output_zero["leaf_index"]
    ]
    assert measured_zero == measured_one
    assert output_zero["summary"]["training_row_count"] == 24


def test_automatic_tree_uses_only_sample_design_development_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path,
        with_split=True,
    )
    original = DataBackend.read_frame
    projections: list[list[str] | None] = []

    def tracked_read(self, path, *, columns=None, nrows=None):
        projections.append(None if columns is None else list(columns))
        return original(self, path, columns=columns, nrows=nrows)

    monkeypatch.setattr(DataBackend, "read_frame", tracked_read)
    output = strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset, workspace, mapping),
        _tool_context(settings, task),
    )

    assert projections == [
        [
            "income",
            "score",
            "weight",
            "loan_amount",
            "overdue_amount",
            "bad",
            "sample_split",
        ]
    ]
    assert output["summary"]["training_row_count"] == 16
    assert output["equivalence"]["source_row_count"] == 16


def test_automatic_tree_rejects_split_column_as_feature(tmp_path: Path) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path,
        with_split=True,
    )

    with pytest.raises(StrategyError, match="split column"):
        strategy_tools.tool_build_automatic_tree_candidate(
            {
                **_inputs(dataset, workspace, mapping),
                "features": ["sample_split"],
                "directions": {"sample_split": "unordered"},
            },
            _tool_context(settings, task),
        )


def test_sample_context_hash_excludes_tree_generation_choices(
    tmp_path: Path,
) -> None:
    settings, _runner, _registry, task, _other, dataset, workspace, mapping = _runtime(
        tmp_path
    )
    ctx = _tool_context(settings, task)
    first = strategy_tools.tool_build_automatic_tree_candidate(
        _inputs(dataset, workspace, mapping),
        ctx,
    )
    second = strategy_tools.tool_build_automatic_tree_candidate(
        {
            **_inputs(dataset, workspace, mapping),
            "features": ["score"],
            "directions": {"score": "unordered"},
            "max_depth": 1,
            "seed": 7,
            "budgets": {
                "max_rows": 50,
                "max_features": 2,
                "max_cells": 100,
                "max_nodes": 7,
                "max_cutpoint_evaluations": 100,
            },
        },
        ctx,
    )

    assert first["summary"]["asset_id"] != second["summary"]["asset_id"]
    assert (
        first["summary"]["sample_context_hash"]
        == second["summary"]["sample_context_hash"]
    )


def test_equivalence_sample_is_deterministic_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "x": [float(index) for index in range(10_005)],
            "unused": [index % 3 for index in range(10_005)],
        }
    )

    first, first_evidence = auto_tools._equivalence_sample(frame, features=["x"])
    second, second_evidence = auto_tools._equivalence_sample(frame, features=["x"])

    assert first_evidence == second_evidence
    assert first.equals(second)
    assert first.columns.tolist() == ["x"]
    assert len(first) == auto_tools.MAX_EQUIVALENCE_SAMPLE_ROWS
    assert first.iloc[0, 0] == 0.0
    assert first.iloc[-1, 0] == 10_004.0
    assert first_evidence["selection_rule"] == (
        "evenly_spaced_source_positions_including_endpoints"
    )


def test_full_duckdb_domain_validation_catches_value_omitted_by_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, _runner, registry, task, _other, _dataset, workspace, _mapping = _runtime(
        tmp_path
    )
    row_count = auto_tools.MAX_EQUIVALENCE_SAMPLE_ROWS + 1
    sampled_positions = {
        (index * (row_count - 1)) // (auto_tools.MAX_EQUIVALENCE_SAMPLE_ROWS - 1)
        for index in range(auto_tools.MAX_EQUIVALENCE_SAMPLE_ROWS)
    }
    omitted_position = next(
        index for index in range(row_count) if index not in sampled_positions
    )
    scores = list(range(row_count))
    offending_value = 2**53 + 1
    scores[omitted_position] = offending_value
    frame = pd.DataFrame(
        {
            "score": scores,
            "bad": [1 if index < row_count // 2 else 0 for index in range(row_count)],
        }
    )
    bounded_sample, _evidence = auto_tools._equivalence_sample(
        frame,
        features=["score"],
    )
    assert offending_value not in bounded_sample["score"].tolist()
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={"score": "numeric", "bad": "target"},
    )
    dataset, active = _activate_frame(
        tmp_path,
        settings=settings,
        registry=registry,
        task=task,
        current_workspace=workspace,
        frame=frame,
        mapping=mapping,
    )
    sample_design_ref = _materialize_sample_design_ref(
        settings,
        task,
        dataset,
        active,
        mapping,
        include_optional_fields=False,
    )

    def bounded_sampling_must_not_start(*_args, **_kwargs):
        pytest.fail("bounded equivalence sampling must follow full-frame validation")

    monkeypatch.setattr(
        auto_tools,
        "_equivalence_sample",
        bounded_sampling_must_not_start,
    )
    inputs = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": active.revision,
        "analysis_generation": active.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "sample_design_ref": sample_design_ref,
        "features": ["score"],
        "directions": {"score": "unordered"},
        "max_depth": 2,
        "min_leaf_count": 100,
        "budgets": {
            "max_rows": 20_000,
            "max_features": 1,
            "max_cells": 20_000,
            "max_nodes": 31,
            "max_cutpoint_evaluations": 30_000,
        },
    }

    with pytest.raises(StrategyError, match="exact DOUBLE range"):
        strategy_tools.tool_build_automatic_tree_candidate(
            inputs,
            _tool_context(settings, task),
        )
    assert _automatic_tree_records(settings, task) == []
    output_dir = settings.tasks_dir / task.id / "strategy_automatic_trees"
    assert not output_dir.exists()


def test_valid_large_frame_keeps_equivalence_bounded(
    tmp_path: Path,
) -> None:
    settings, _runner, registry, task, _other, _dataset, workspace, _mapping = _runtime(
        tmp_path
    )
    row_count = auto_tools.MAX_EQUIVALENCE_SAMPLE_ROWS + 1
    frame = pd.DataFrame(
        {
            "score": list(range(row_count)),
            "bad": [1 if index < row_count // 2 else 0 for index in range(row_count)],
        }
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={"score": "numeric", "bad": "target"},
    )
    dataset, active = _activate_frame(
        tmp_path,
        settings=settings,
        registry=registry,
        task=task,
        current_workspace=workspace,
        frame=frame,
        mapping=mapping,
    )
    sample_design_ref = _materialize_sample_design_ref(
        settings,
        task,
        dataset,
        active,
        mapping,
        include_optional_fields=False,
    )

    output = strategy_tools.tool_build_automatic_tree_candidate(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": active.revision,
            "analysis_generation": active.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "features": ["score"],
            "directions": {"score": "unordered"},
            "max_depth": 2,
            "min_leaf_count": 100,
            "budgets": {
                "max_rows": 20_000,
                "max_features": 1,
                "max_cells": 20_000,
                "max_nodes": 31,
                "max_cutpoint_evaluations": 30_000,
            },
        },
        _tool_context(settings, task),
    )

    assert output["equivalence"]["source_row_count"] == row_count
    assert output["equivalence"]["sample_count"] == (
        auto_tools.MAX_EQUIVALENCE_SAMPLE_ROWS
    )
    assert len(output["artifacts"]) == 6

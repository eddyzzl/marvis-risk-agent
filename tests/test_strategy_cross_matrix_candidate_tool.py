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
from marvis.packs.strategy import cross_matrix_candidate_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.cross_matrix_candidate import (
    parse_cross_matrix_candidate_asset_json,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    ASSET_ARTIFACT_KIND,
    run_build_cross_matrix_candidate,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _setup(
    tmp_path: Path,
    *,
    drop_one_nan_label: bool = False,
    include_amount_columns: bool = True,
) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="cross-matrix",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "unused": [f"row-{index}" for index in range(12)],
            "age": [20, 22, 24, 26, 40, 42, 44, 46, 60, 62, 64, 66],
            "score": [100, 110, 300, 310, 120, 130, 320, 330, 140, 150, 340, 350],
            "loan_amount": [100, 120, None, 160, 180, 200, 220, 240, 260, 280, 300, 320],
            "overdue_amount": [0, 0, 3, None, 0, 10, 0, 15, 20, 25, 30, 40],
            "bad": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    if drop_one_nan_label:
        frame.loc[len(frame) - 1, "bad"] = None
    source_path = tmp_path / "cross-matrix.parquet"
    frame.to_parquet(source_path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source_path, task_id=task.id, role="derived")
    workspaces = DataWorkspaceRepository(settings.db_path)
    active = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    field_roles = {
        "unused": "id",
        "age": "feature",
        "score": "score",
        "bad": "target",
    }
    if include_amount_columns:
        field_roles.update(
            {
                "loan_amount": "loan_amount",
                "overdue_amount": "overdue_amount",
            }
        )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles=field_roles,
    )
    workspace = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=active.revision,
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    runtime = strategy_tools._runtime(ctx)
    source_inputs = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "analysis_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "drop_nan_labels": drop_one_nan_label,
        "features": ["age", "score"],
        "methods": ["equal_width"],
        "bin_count": 3,
    }
    if include_amount_columns:
        source_inputs.update(
            {
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
            }
        )
    source = strategy_tools.tool_analyze_univariate_candidates(source_inputs, ctx)
    source_artifact = next(
        artifact
        for artifact in source["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    inputs = {
        "source_artifact_id": source_artifact["artifact_id"],
        "expected_artifact_content_hash": source_artifact["content_hash"],
        "expected_candidate_id": source["candidate_id"],
        "expected_evidence_hash": source["evidence_hash"],
        "x_feature": "age",
        "x_method": "equal_width",
        "y_feature": "score",
        "y_method": "equal_width",
    }
    return {
        "settings": settings,
        "task": task,
        "runtime": runtime,
        "ctx": ctx,
        "source": source,
        "dataset": dataset,
        "inputs": inputs,
    }


def test_cross_matrix_tool_reads_once_replays_each_bin_and_persists_complete_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    runtime = fixture["runtime"]
    reads: list[list[str] | None] = []
    evaluations: list[dict] = []
    original_read = runtime.backend.read_frame
    original_evaluate = cross_matrix_candidate_tools.evaluate_expression_frame

    def tracked_read(path, *, columns=None, nrows=None):
        reads.append(None if columns is None else list(columns))
        return original_read(path, columns=columns, nrows=nrows)

    def tracked_evaluate(frame, condition):
        evaluations.append(condition)
        return original_evaluate(frame, condition)

    monkeypatch.setattr(runtime.backend, "read_frame", tracked_read)
    monkeypatch.setattr(
        cross_matrix_candidate_tools,
        "evaluate_expression_frame",
        tracked_evaluate,
    )

    result = run_build_cross_matrix_candidate(
        fixture["inputs"], fixture["ctx"], runtime
    )

    evidence = fixture["source"]["candidate_evidence"]
    methods = {
        feature["feature"]: feature["methods"][0]
        for feature in evidence["analysis"]["features"]
    }
    expected_cells = len(methods["age"]["bins"]) * len(methods["score"]["bins"])
    assert reads == [["age", "score", "bad", "loan_amount", "overdue_amount"]]
    assert len(evaluations) == len(methods["age"]["bins"]) + len(
        methods["score"]["bins"]
    )
    assert result["schema_version"] == (
        "strategy.build-cross-matrix-candidate-tool.v1"
    )
    assert result["parent_candidate_id"] == evidence["candidate_id"]
    assert result["parent_evidence_hash"] == evidence["evidence_hash"]
    assert result["candidate_id"] == result["cross_matrix_candidate"][
        "candidate_evidence"
    ]["candidate_id"]
    assert result["evidence_hash"] == result["cross_matrix_candidate"][
        "candidate_evidence"
    ]["evidence_hash"]
    assert result["dataset_id"] == evidence["identity"]["dataset_id"]
    assert result["target_col"] == "bad"
    assert result["population_count"] == 12
    assert result["labeled_count"] == 12
    assert result["drop_nan_labels"] is False
    assert result["nan_labels_dropped"] == 0
    assert result["row_axis"] == {
        "feature": "age",
        "method": "equal_width",
        "bin_count": len(methods["age"]["bins"]),
    }
    assert result["column_axis"] == {
        "feature": "score",
        "method": "equal_width",
        "bin_count": len(methods["score"]["bins"]),
    }
    assert result["cell_count"] == expected_cells
    assert result["candidate_stage"] == "development"
    assert result["observation_stage"] == "backtested"
    assert result["validation_status"] == "unvalidated"
    assert result["not_selected"] is True
    assert result["not_admitted"] is True
    assert result["not_applied"] is True
    assert result["not_adopted"] is True
    assert result["not_deployed"] is True
    assert len(result["artifacts"]) == 1
    assert result["artifacts"][0]["kind"] == ASSET_ARTIFACT_KIND
    assert result["artifacts"][0]["format"] == "json"
    assert "path" not in result["artifacts"][0]

    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    artifact_path = Path(record["path"])
    assert sha256_file(artifact_path) == record["content_hash"]
    assert parse_cross_matrix_candidate_asset_json(artifact_path.read_bytes()) == result[
        "cross_matrix_candidate"
    ]
    provenance = record["provenance"]
    assert set(provenance) == cross_matrix_candidate_tools._ASSET_PROVENANCE_FIELDS
    assert provenance["producer_version"] == result["cross_matrix_candidate"][
        "producer_version"
    ]
    assert provenance["asset_schema_version"] == result[
        "cross_matrix_candidate"
    ]["schema_version"]
    assert provenance["asset_type"] == "cross_matrix"
    assert provenance["parent_candidate_id"] == evidence["candidate_id"]
    assert provenance["parent_evidence_hash"] == evidence["evidence_hash"]
    assert provenance["candidate_id"] == result["candidate_id"]
    assert provenance["evidence_hash"] == result["evidence_hash"]
    assert provenance["task_id"] == fixture["task"].id
    assert provenance["workspace_revision"] == evidence["identity"][
        "workspace_revision"
    ]
    assert provenance["workspace_generation"] == evidence["identity"][
        "workspace_generation"
    ]
    assert provenance["semantic_mapping_hash"] == evidence["identity"][
        "semantic_mapping_hash"
    ]
    assert provenance["sample_context_hash"] == result["cross_matrix_candidate"][
        "sample_identity"
    ]["sample_context_hash"]
    assert provenance["target_col"] == "bad"
    assert provenance["labeled_row_count"] == 12
    assert provenance["candidate_stage"] == "development"
    assert provenance["observation_stage"] == "backtested"
    assert provenance["validation_status"] == "unvalidated"
    assert provenance["budget"] == cross_matrix_candidate_tools.CROSS_MATRIX_MAX_CELLS
    assert provenance["truncated"] is False

    asset = result["cross_matrix_candidate"]
    matrix_cells = asset["matrix"]["cells"]
    assert len(matrix_cells) == expected_cells
    assert sum(cell["effect"]["count"] for cell in matrix_cells) == 12
    assert any(cell["effect"]["count"] == 0 for cell in matrix_cells)
    empty = next(cell for cell in asset["measurement"]["cells"] if cell["count"] == 0)
    assert empty["good"] == empty["bad"] == 0
    assert empty["amounts"]["loan_amount"] == {
        "status": "available",
        "covered_count": 0,
        "value": 0.0,
    }
    assert asset["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }


@pytest.mark.parametrize("forbidden", ["max_cells", "budget", "cells", "metrics"])
def test_cross_matrix_tool_rejects_platform_or_result_injection(
    tmp_path: Path,
    forbidden: str,
) -> None:
    fixture = _setup(tmp_path)
    with pytest.raises(StrategyError, match=f"unsupported: {forbidden}"):
        run_build_cross_matrix_candidate(
            {**fixture["inputs"], forbidden: 1},
            fixture["ctx"],
            fixture["runtime"],
        )


def test_cross_matrix_summary_distinguishes_population_and_labeled_rows(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, drop_one_nan_label=True)
    result = run_build_cross_matrix_candidate(
        fixture["inputs"], fixture["ctx"], fixture["runtime"]
    )

    assert result["population_count"] == 12
    assert result["labeled_count"] == 11
    assert result["drop_nan_labels"] is True
    assert result["nan_labels_dropped"] == 1


def test_cross_matrix_tool_succeeds_without_optional_amount_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path, include_amount_columns=False)
    runtime = fixture["runtime"]
    reads: list[list[str] | None] = []
    original_read = runtime.backend.read_frame

    def tracked_read(path, *, columns=None, nrows=None):
        reads.append(None if columns is None else list(columns))
        return original_read(path, columns=columns, nrows=nrows)

    monkeypatch.setattr(runtime.backend, "read_frame", tracked_read)
    result = run_build_cross_matrix_candidate(
        fixture["inputs"], fixture["ctx"], runtime
    )

    assert reads == [["age", "score", "bad"]]
    assert len(result["artifacts"]) == 1
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert Path(record["path"]).is_file()
    assert sha256_file(Path(record["path"])) == record["content_hash"]

    asset = result["cross_matrix_candidate"]
    summary_amounts = asset["summary"]["amount_metrics"]
    assert summary_amounts["loan_amount"] == {
        "status": "unavailable",
        "covered_count": None,
        "coverage_rate": None,
        "value": None,
        "reason": "column_unavailable",
    }
    assert summary_amounts["overdue_amount"] == {
        "status": "unavailable",
        "covered_count": None,
        "coverage_rate": None,
        "value": None,
        "reason": "column_unavailable",
    }
    assert summary_amounts["overdue_rate"] == {
        "status": "unavailable",
        "covered_count": None,
        "coverage_rate": None,
        "value": None,
        "reason": "columns_unavailable",
    }
    assert all(
        cell["amounts"] == {
            "loan_amount": {
                "status": "unavailable",
                "covered_count": None,
                "value": None,
            },
            "overdue_amount": {
                "status": "unavailable",
                "covered_count": None,
                "value": None,
            },
            "paired": {
                "status": "unavailable",
                "covered_count": None,
                "loan_value": None,
                "overdue_value": None,
            },
        }
        for cell in asset["measurement"]["cells"]
    )
    assert all(
        cell["effect"]["amount_metrics"]["loan_amount"]["status"]
        == "unavailable"
        and cell["effect"]["amount_metrics"]["overdue_amount"]["status"]
        == "unavailable"
        and cell["effect"]["amount_metrics"]["overdue_rate"]["status"]
        == "unavailable"
        for cell in asset["matrix"]["cells"]
    )
    assert result["not_selected"] is True
    assert result["not_applied"] is True
    assert result["not_adopted"] is True


def test_cross_matrix_tool_gates_budget_before_reading_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    monkeypatch.setattr(cross_matrix_candidate_tools, "CROSS_MATRIX_MAX_CELLS", 1)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("frame must not be materialized before the budget gate")

    monkeypatch.setattr(fixture["runtime"].backend, "read_frame", forbidden_read)
    with pytest.raises(StrategyError, match="platform cell budget"):
        run_build_cross_matrix_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )


def test_cross_matrix_tool_requires_distinct_available_axis_features(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    with pytest.raises(StrategyError, match="features must be distinct"):
        run_build_cross_matrix_candidate(
            {
                **fixture["inputs"],
                "y_feature": "age",
                "y_method": "equal_width",
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    # The API boundary rejects the same feature even when the requested methods
    # differ; the upstream Agent compiler is not a security boundary.
    with pytest.raises(StrategyError, match="features must be distinct"):
        run_build_cross_matrix_candidate(
            {
                **fixture["inputs"],
                "y_feature": "age",
                "y_method": "equal_frequency",
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    with pytest.raises(StrategyError, match="available Cross Matrix y method not found"):
        run_build_cross_matrix_candidate(
            {**fixture["inputs"], "y_method": "tree"},
            fixture["ctx"],
            fixture["runtime"],
        )


def test_cross_matrix_tool_rejects_bound_dataset_byte_drift(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    dataset_path = fixture["runtime"].registry.resolve_path(fixture["dataset"].id)
    dataset_path.write_bytes(dataset_path.read_bytes() + b"drift")

    with pytest.raises(StrategyError, match="dataset failed hash verification"):
        run_build_cross_matrix_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )
    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assert all(record["kind"] != ASSET_ARTIFACT_KIND for record in records)


def test_cross_matrix_atomic_failure_removes_staged_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)

    def fail_under_lock(conn, dataset):
        raise StrategyError("injected Cross Matrix registry drift")

    monkeypatch.setattr(
        cross_matrix_candidate_tools.candidate_asset_tools,
        "_require_dataset_on_connection",
        fail_under_lock,
    )
    with pytest.raises(StrategyError, match="injected Cross Matrix registry drift"):
        run_build_cross_matrix_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )

    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assert all(record["kind"] != ASSET_ARTIFACT_KIND for record in records)
    output_dir = (
        fixture["settings"].tasks_dir
        / fixture["task"].id
        / "strategy_cross_matrix_candidates"
    )
    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_cross_matrix_persisted_artifact_is_canonical_json(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    result = run_build_cross_matrix_candidate(
        fixture["inputs"], fixture["ctx"], fixture["runtime"]
    )
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        result["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    raw = Path(record["path"]).read_text("utf-8")
    assert json.dumps(
        json.loads(raw),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == raw

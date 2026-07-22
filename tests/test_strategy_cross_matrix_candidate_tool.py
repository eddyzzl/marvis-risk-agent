from __future__ import annotations

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
from marvis.files import sha256_file
from marvis.packs.strategy import cross_matrix_candidate_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.output.strategy_candidate_report import (
    canonical_strategy_candidate_report_json,
)
from marvis.packs.strategy.candidate_evidence import build_candidate_evidence
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
    with_split: bool = False,
    target_bad_value: int = 1,
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
    normalized_target = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    frame_data = {
        "unused": [f"row-{index}" for index in range(12)],
        "age": [20, 22, 24, 26, 40, 42, 44, 46, 60, 62, 64, 66],
        "score": [100, 110, 300, 310, 120, 130, 320, 330, 140, 150, 340, 350],
        "loan_amount": [100, 120, None, 160, 180, 200, 220, 240, 260, 280, 300, 320],
        "overdue_amount": [0, 0, 3, None, 0, 10, 0, 15, 20, 25, 30, 40],
        "bad": (
            normalized_target
            if target_bad_value == 1
            else [1 - value for value in normalized_target]
        ),
    }
    if with_split:
        frame_data["sample_split"] = ["dev"] * 8 + ["validation"] * 2 + ["oot"] * 2
    frame = pd.DataFrame(frame_data)
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
        **({"sample_split": "segment"} if with_split else {}),
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
    sample_request = {
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "target_bad_value": target_bad_value,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_window_start": "2026-01-01",
        "observation_window_end": "2026-06-30",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": drop_one_nan_label,
    }
    if include_amount_columns:
        sample_request.update(
            {
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
            }
        )
    if with_split:
        sample_request.update(
            {
                "split_col": "sample_split",
                "development_values": ["dev"],
                "validation_values": ["validation"],
                "oot_values": ["oot"],
            }
        )
    sample_output = strategy_tools.tool_materialize_sample_design(sample_request, ctx)
    sample_design_ref = {
        "artifact_id": sample_output["artifact"]["artifact_id"],
        "artifact_content_hash": sample_output["artifact"]["content_hash"],
        "sample_design_id": sample_output["sample_design_id"],
        "sample_design_content_hash": sample_output["content_hash"],
        "partition": "development",
    }
    source_inputs = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "analysis_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "sample_design_ref": sample_design_ref,
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
        "sample_design_ref": sample_design_ref,
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


def test_cross_matrix_tool_isolates_the_bound_development_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
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

    assert reads == [
        [
            "age",
            "score",
            "bad",
            "loan_amount",
            "overdue_amount",
            "sample_split",
        ]
    ]
    assert result["population_count"] == 12
    assert result["labeled_count"] == 8
    measurement = result["cross_matrix_candidate"]["measurement"]
    assert measurement["population_count"] == 8
    assert sum(cell["count"] for cell in measurement["cells"]) == 8


def test_cross_matrix_tool_normalizes_bad_zero_to_the_internal_bad_one_contract(
    tmp_path: Path,
) -> None:
    bad_one = _setup(tmp_path / "bad-one", target_bad_value=1)
    bad_zero = _setup(tmp_path / "bad-zero", target_bad_value=0)

    one = run_build_cross_matrix_candidate(
        bad_one["inputs"], bad_one["ctx"], bad_one["runtime"]
    )
    zero = run_build_cross_matrix_candidate(
        bad_zero["inputs"], bad_zero["ctx"], bad_zero["runtime"]
    )

    assert bad_one["source"]["candidate_evidence"]["analysis"] == bad_zero[
        "source"
    ]["candidate_evidence"]["analysis"]
    one_measurement = one["cross_matrix_candidate"]["measurement"]
    zero_measurement = zero["cross_matrix_candidate"]["measurement"]
    assert (one_measurement["good"], one_measurement["bad"]) == (
        zero_measurement["good"],
        zero_measurement["bad"],
    )
    assert [
        (cell["count"], cell["good"], cell["bad"], cell["amounts"])
        for cell in one_measurement["cells"]
    ] == [
        (cell["count"], cell["good"], cell["bad"], cell["amounts"])
        for cell in zero_measurement["cells"]
    ]


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


def test_cross_matrix_tool_rejects_legacy_unbound_source_before_frame_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    evidence = fixture["source"]["candidate_evidence"]
    parameters = dict(evidence["generation"]["parameters"])
    del parameters["sample_design_ref"]
    legacy_evidence = build_candidate_evidence(
        task_id=evidence["identity"]["task_id"],
        dataset_id=evidence["identity"]["dataset_id"],
        dataset_content_hash=evidence["identity"]["dataset_content_hash"],
        workspace_revision=evidence["identity"]["workspace_revision"],
        workspace_generation=evidence["identity"]["workspace_generation"],
        semantic_mapping_hash=evidence["identity"]["semantic_mapping_hash"],
        generation_parameters=parameters,
        seed=evidence["generation"]["seed"],
        budget=evidence["generation"]["budget"],
        truncated=evidence["generation"]["truncated"],
        analysis=evidence["analysis"],
        metrics=evidence["metrics"],
        source_refs=[
            ref
            for ref in evidence["source_refs"]
            if not ref.startswith("strategy-sample-design:")
        ],
        red_flags=evidence["red_flags"],
        producer_version=evidence["producer_version"],
    )
    content = canonical_strategy_candidate_report_json(
        legacy_evidence,
        evidence["analysis"],
    )
    content_hash = hashlib.sha256(content).hexdigest()
    out_dir = fixture["settings"].tasks_dir / fixture["task"].id / "strategy_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"{legacy_evidence['candidate_id']}_{content_hash[:12]}.json"
    )
    path.write_bytes(content)

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    current = repository.get_for_task(
        fixture["task"].id,
        fixture["inputs"]["source_artifact_id"],
    )
    assert current is not None
    provenance = {
        **current["provenance"],
        "candidate_id": legacy_evidence["candidate_id"],
        "evidence_hash": legacy_evidence["evidence_hash"],
        "generation_parameters": parameters,
    }
    legacy = repository.register(
        task_id=fixture["task"].id,
        kind="strategy_candidate_json",
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.analyze_univariate_candidates",
        provenance=provenance,
    )

    def forbidden_read(*args, **kwargs):
        raise AssertionError("legacy unbound source must fail before frame read")

    monkeypatch.setattr(fixture["runtime"].backend, "read_frame", forbidden_read)
    with pytest.raises(StrategyError, match="sample_design_ref"):
        run_build_cross_matrix_candidate(
            {
                **fixture["inputs"],
                "source_artifact_id": legacy["id"],
                "expected_artifact_content_hash": content_hash,
                "expected_candidate_id": legacy_evidence["candidate_id"],
                "expected_evidence_hash": legacy_evidence["evidence_hash"],
            },
            fixture["ctx"],
            fixture["runtime"],
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


def test_cross_matrix_sample_binding_deletion_under_lock_rolls_back_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    original_require = (
        cross_matrix_candidate_tools.require_strategy_sample_design_execution_binding_on_connection
    )

    def delete_then_require(conn, binding):
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (binding.task_id, binding.reference.artifact_id),
        )
        return original_require(conn, binding)

    monkeypatch.setattr(
        cross_matrix_candidate_tools,
        "require_strategy_sample_design_execution_binding_on_connection",
        delete_then_require,
    )

    with pytest.raises(StrategyError, match="sample-design artifact disappeared"):
        run_build_cross_matrix_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    assert (
        repository.get_for_task(
            fixture["task"].id,
            fixture["sample_design_ref"]["artifact_id"],
        )
        is not None
    )
    assert all(
        record["kind"] != ASSET_ARTIFACT_KIND
        for record in repository.list_for_task(fixture["task"].id)
    )
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

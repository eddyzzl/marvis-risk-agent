from __future__ import annotations

import json
from pathlib import Path
import threading

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository, connect, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy import candidate_asset_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.candidate_asset import validate_candidate_asset
from marvis.packs.strategy.candidate_asset_tools import (
    ASSET_ARTIFACT_KIND,
    run_refine_univariate_candidate,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _setup(tmp_path: Path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="candidate-asset",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    other_task = tasks.create_task(
        TaskCreate(
            model_name="foreign",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "unused": [f"row-{index}" for index in range(12)],
            "score": [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430],
            "loan_amount": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320],
            "overdue_amount": [0, 0, 0, 5, 0, 10, 0, 15, 20, 25, 30, 40],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    source_path = tmp_path / "candidate.parquet"
    frame.to_parquet(source_path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source_path, task_id=task.id, role="derived")
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
            "unused": "id",
            "score": "score",
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
    source_output = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        ctx,
    )
    json_artifact = next(
        artifact
        for artifact in source_output["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    xlsx_artifact = next(
        artifact
        for artifact in source_output["artifacts"]
        if artifact["kind"] == "strategy_candidate_xlsx"
    )
    method = source_output["candidate_evidence"]["analysis"]["features"][0]["methods"][
        0
    ]
    source_bin_id = method["bins"][0]["id"]
    inputs = {
        "source_artifact_id": json_artifact["artifact_id"],
        "expected_artifact_content_hash": json_artifact["content_hash"],
        "expected_candidate_id": source_output["candidate_id"],
        "expected_evidence_hash": source_output["evidence_hash"],
        "feature": "score",
        "method": "equal_width",
        "merge_groups": [],
        "selection": {"source_bin_ids": [source_bin_id]},
        "selection_reason": "manual risk review",
    }
    return {
        "settings": settings,
        "task": task,
        "other_task": other_task,
        "runtime": runtime,
        "ctx": ctx,
        "dataset": dataset,
        "source_output": source_output,
        "json_artifact": json_artifact,
        "xlsx_artifact": xlsx_artifact,
        "inputs": inputs,
    }


def _context(settings, task_id: str) -> ToolContext:
    return ToolContext(
        task_id=task_id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def test_refine_tool_projects_bound_columns_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    runtime = fixture["runtime"]
    projections: list[list[str] | None] = []
    original_read = runtime.backend.read_frame

    def tracked_read(path, *, columns=None, nrows=None):
        projections.append(None if columns is None else list(columns))
        return original_read(path, columns=columns, nrows=nrows)

    monkeypatch.setattr(runtime.backend, "read_frame", tracked_read)

    first = run_refine_univariate_candidate(fixture["inputs"], fixture["ctx"], runtime)
    repeated = run_refine_univariate_candidate(
        fixture["inputs"], fixture["ctx"], runtime
    )

    assert first == repeated
    assert projections == [
        ["score", "bad", "loan_amount", "overdue_amount"],
        ["score", "bad", "loan_amount", "overdue_amount"],
    ]
    assert first["schema_version"] == ("strategy.refine-univariate-candidate-tool.v1")
    assert first["effect_stage"] == "development"
    assert first["validation_status"] == "unvalidated"
    assert first["effect_id"] == first["effect"]["effect_id"]
    assert first["rule"] == first["candidate_asset"]["rule"]
    assert first["asset_hash"] == first["candidate_asset"]["asset_hash"]
    assert len(first["artifacts"]) == 1
    artifact = first["artifacts"][0]
    assert artifact["kind"] == ASSET_ARTIFACT_KIND
    assert "path" not in artifact

    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    asset_records = [row for row in records if row["kind"] == ASSET_ARTIFACT_KIND]
    assert len(asset_records) == 1
    asset_record = asset_records[0]
    assert sha256_file(Path(asset_record["path"])) == asset_record["content_hash"]
    assert (
        validate_candidate_asset(
            json.loads(Path(asset_record["path"]).read_text("utf-8"))
        )
        == first["candidate_asset"]
    )


def test_refine_tool_rejects_result_injection_and_invalid_source_bindings(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    inputs = fixture["inputs"]
    runtime = fixture["runtime"]

    with pytest.raises(StrategyError, match="unsupported: metrics"):
        run_refine_univariate_candidate(
            {**inputs, "metrics": [{"iv": 1}]}, fixture["ctx"], runtime
        )

    other_ctx = _context(fixture["settings"], fixture["other_task"].id)
    with pytest.raises(StrategyError, match="artifact not found"):
        run_refine_univariate_candidate(
            inputs,
            other_ctx,
            strategy_tools._runtime(other_ctx),
        )

    with pytest.raises(StrategyError, match="strategy_candidate_json"):
        run_refine_univariate_candidate(
            {
                **inputs,
                "source_artifact_id": fixture["xlsx_artifact"]["artifact_id"],
                "expected_artifact_content_hash": fixture["xlsx_artifact"][
                    "content_hash"
                ],
            },
            fixture["ctx"],
            runtime,
        )

    with pytest.raises(StrategyError, match="content hash changed"):
        run_refine_univariate_candidate(
            {**inputs, "expected_artifact_content_hash": "0" * 64},
            fixture["ctx"],
            runtime,
        )

    with pytest.raises(StrategyError, match="evidence_hash does not match"):
        run_refine_univariate_candidate(
            {**inputs, "expected_evidence_hash": "0" * 64},
            fixture["ctx"],
            runtime,
        )

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    source_record = repository.get_for_task(
        fixture["task"].id,
        fixture["json_artifact"]["artifact_id"],
    )
    assert source_record is not None
    source_bytes = Path(source_record["path"]).read_bytes()
    wrong_origin_path = tmp_path / "wrong-origin.json"
    wrong_origin_path.write_bytes(source_bytes)
    wrong_origin = repository.register(
        task_id=fixture["task"].id,
        kind="strategy_candidate_json",
        path=str(wrong_origin_path),
        content_hash=source_record["content_hash"],
        origin_tool="strategy.forge_candidate",
        provenance=source_record["provenance"],
    )
    with pytest.raises(StrategyError, match="origin_tool is invalid"):
        run_refine_univariate_candidate(
            {**inputs, "source_artifact_id": wrong_origin["id"]},
            fixture["ctx"],
            runtime,
        )

    wrong_provenance_path = tmp_path / "wrong-provenance.json"
    wrong_provenance_path.write_bytes(source_bytes)
    wrong_provenance = repository.register(
        task_id=fixture["task"].id,
        kind="strategy_candidate_json",
        path=str(wrong_provenance_path),
        content_hash=source_record["content_hash"],
        origin_tool=source_record["origin_tool"],
        provenance={**source_record["provenance"], "metrics": []},
    )
    with pytest.raises(StrategyError, match="provenance fields are invalid"):
        run_refine_univariate_candidate(
            {**inputs, "source_artifact_id": wrong_provenance["id"]},
            fixture["ctx"],
            runtime,
        )

    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assert all(row["kind"] != ASSET_ARTIFACT_KIND for row in records)


def test_refine_tool_rejects_artifact_path_and_byte_drift(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    source_record = repository.get_for_task(
        fixture["task"].id,
        fixture["json_artifact"]["artifact_id"],
    )
    assert source_record is not None
    rogue_path = tmp_path / "rogue-candidate.json"
    rogue_path.write_bytes(Path(source_record["path"]).read_bytes())
    rogue = repository.register(
        task_id=fixture["task"].id,
        kind="strategy_candidate_json",
        path=str(rogue_path),
        content_hash=source_record["content_hash"],
        origin_tool=source_record["origin_tool"],
        provenance=source_record["provenance"],
    )
    with pytest.raises(StrategyError, match="path is not canonical"):
        run_refine_univariate_candidate(
            {**fixture["inputs"], "source_artifact_id": rogue["id"]},
            fixture["ctx"],
            fixture["runtime"],
        )

    source_path = Path(source_record["path"])
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(StrategyError, match="content hash drifted"):
        run_refine_univariate_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )


def test_refine_tool_rejects_bound_dataset_byte_drift(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    dataset_path = fixture["runtime"].registry.resolve_path(fixture["dataset"].id)
    dataset_path.write_bytes(dataset_path.read_bytes() + b"drift")

    with pytest.raises(StrategyError, match="dataset failed hash verification"):
        run_refine_univariate_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )

    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assert all(row["kind"] != ASSET_ARTIFACT_KIND for row in records)


def test_refine_tool_rejects_under_lock_dataset_registry_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    original_source_path = fixture["dataset"].source_path
    original_require = candidate_asset_tools._require_dataset_on_connection
    changed = False

    def drift_then_require(conn, dataset):
        nonlocal changed
        if not changed:
            changed = True
            conn.execute(
                "UPDATE datasets SET source_path = ? WHERE id = ?",
                ("forged/path.parquet", fixture["dataset"].id),
            )
        return original_require(conn, dataset)

    monkeypatch.setattr(
        candidate_asset_tools,
        "_require_dataset_on_connection",
        drift_then_require,
    )
    with pytest.raises(StrategyError, match="registry path changed"):
        run_refine_univariate_candidate(
            fixture["inputs"], fixture["ctx"], fixture["runtime"]
        )

    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assert all(row["kind"] != ASSET_ARTIFACT_KIND for row in records)
    with connect(fixture["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT source_path FROM datasets WHERE id = ?",
            (fixture["dataset"].id,),
        ).fetchone()
    assert row is not None
    assert str(row["source_path"]) == original_source_path


def test_identical_refine_writers_lock_before_artifact_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    late_writer_entered = threading.Event()
    release_late_writer = threading.Event()
    committing_writer_promoted = threading.Event()
    original_require = candidate_asset_tools._require_source_on_connection
    original_promote = candidate_asset_tools.ArtifactUnitOfWork.promote_all

    def gated_require(conn, source):
        if threading.current_thread().name == "late-writer":
            late_writer_entered.set()
            if not release_late_writer.wait(timeout=10):
                raise RuntimeError("timed out waiting to release late writer")
            raise StrategyError("injected late source binding failure")
        return original_require(conn, source)

    def tracked_promote(self):
        result = original_promote(self)
        if threading.current_thread().name == "committing-writer":
            committing_writer_promoted.set()
        return result

    monkeypatch.setattr(
        candidate_asset_tools,
        "_require_source_on_connection",
        gated_require,
    )
    monkeypatch.setattr(
        candidate_asset_tools.ArtifactUnitOfWork,
        "promote_all",
        tracked_promote,
    )
    failures: dict[str, BaseException] = {}
    outputs: dict[str, dict] = {}

    def invoke(name: str) -> None:
        try:
            outputs[name] = run_refine_univariate_candidate(
                fixture["inputs"], fixture["ctx"], fixture["runtime"]
            )
        except BaseException as exc:  # asserted by the main test thread
            failures[name] = exc

    late = threading.Thread(target=invoke, args=("late",), name="late-writer")
    committing = threading.Thread(
        target=invoke,
        args=("committing",),
        name="committing-writer",
    )
    late.start()
    assert late_writer_entered.wait(timeout=10)
    committing.start()
    assert not committing_writer_promoted.wait(timeout=1)
    release_late_writer.set()
    late.join(timeout=10)
    committing.join(timeout=10)

    assert not late.is_alive()
    assert not committing.is_alive()
    assert isinstance(failures.get("late"), StrategyError)
    assert "committing" not in failures
    assert outputs["committing"]["validation_status"] == "unvalidated"
    records = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assets = [row for row in records if row["kind"] == ASSET_ARTIFACT_KIND]
    assert len(assets) == 1
    assert Path(assets[0]["path"]).is_file()
    assert sha256_file(Path(assets[0]["path"])) == assets[0]["content_hash"]

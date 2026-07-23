from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.domain import TaskCreate
from marvis.feature.univariate import analyze_univariate
from marvis.output.strategy_candidate_report import render_strategy_candidate_bundle
from marvis.orchestrator.contracts import Plan, PlanStatus
from marvis.packs.strategy import candidate_lab_projection
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    refine_univariate_candidate,
)
from marvis.packs.strategy.candidate_fragment import (
    sample_context_hash_from_candidate_evidence,
    univariate_asset_to_verified_fragment,
)
from marvis.packs.strategy.candidate_evidence import build_candidate_evidence
from marvis.packs.strategy.cross_matrix_candidate import (
    CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION,
    build_cross_matrix_candidate_asset,
    canonical_cross_matrix_candidate_asset_json,
)
from marvis.packs.strategy.evaluator import evaluate_expression_frame
from marvis.packs.strategy.pool import add_verified_candidate_fragment
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
    canonical_strategy_pool_snapshot_json,
    strategy_pool_artifact_content_hash,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _strategy_task(app) -> str:
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="Candidate Lab projection",
            model_version="v2",
            validator="owner",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )
    return task.id


def _register_univariate_candidate(
    app,
    task_id: str,
    *,
    seed: int = 0,
    created_at: str | None = None,
) -> tuple[dict, Path]:
    frame = pd.DataFrame(
        {
            "score": [300, 350, 400, 450, 500, 550, 600, 650],
            "bad": [1, 1, 0, 1, 0, 0, 0, 1],
        }
    )
    analysis = analyze_univariate(
        frame,
        features=["score"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
        min_bin_pct=0,
    )
    generation_parameters = {
        "analysis_schema_version": analysis["schema_version"],
        "features": ["score"],
        "methods": ["equal_width"],
        "bin_count": 3,
    }
    evidence = build_candidate_evidence(
        task_id=task_id,
        dataset_id="dataset-1",
        dataset_content_hash=HASH_A,
        workspace_revision=2,
        workspace_generation=3,
        semantic_mapping_hash=HASH_B,
        generation_parameters=generation_parameters,
        seed=seed,
        budget=10_000,
        truncated=False,
        analysis=analysis,
        metrics=[
            {
                "metric_name": "univariate.iv",
                "dimension": "count",
                "status": "observed",
                "value": analysis["rankings"][0]["iv"],
            },
            {
                "metric_name": "univariate.iv",
                "dimension": "loan_amount",
                "status": "unavailable",
                "value": None,
            },
            {
                "metric_name": "univariate.iv",
                "dimension": "overdue_amount",
                "status": "unavailable",
                "value": None,
            },
        ],
        source_refs=["dataset:dataset-1"],
        red_flags=["test_warning"],
        producer_version="strategy.univariate-candidate/1",
    )
    content = render_strategy_candidate_bundle(evidence, analysis)["json"]
    content_hash = hashlib.sha256(content).hexdigest()
    path = (
        app.state.settings.tasks_dir
        / task_id
        / "strategy_candidates"
        / f"{evidence['candidate_id']}_{content_hash[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = TaskArtifactRepository(app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_candidate_json",
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.analyze_univariate_candidates",
        provenance={
            "schema_version": "strategy.univariate-candidate-artifact.v1",
            "producer_version": evidence["producer_version"],
            "candidate_id": evidence["candidate_id"],
            "evidence_hash": evidence["evidence_hash"],
            "dataset_id": evidence["identity"]["dataset_id"],
            "dataset_content_hash": evidence["identity"]["dataset_content_hash"],
            "registry_metadata_hash": HASH_C,
            "workspace_revision": evidence["identity"]["workspace_revision"],
            "workspace_generation": evidence["identity"]["workspace_generation"],
            "semantic_mapping_hash": evidence["identity"]["semantic_mapping_hash"],
            "generation_parameters": generation_parameters,
            "format": "json",
        },
        created_at=created_at,
    )
    return record, path


def _cross_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52],
            "score": [300, 310, 320, 400, 410, 420, 500, 510, 520, 600, 610, 620],
            "bad": [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            "loan": [100.0, 120, 140, 200, 220, 240, 300, 320, 340, 400, 420, 440],
            "overdue": [0.0, 0, 10, 0, 20, 30, 0, 0, 40, 50, 60, 70],
        }
    )


def _register_cross_source(
    app,
    task_id: str,
    *,
    seed: int = 0,
    directory_name: str = "strategy_candidates",
    provenance_candidate_id: str | None = None,
    created_at: str | None = None,
) -> tuple[dict, Path, dict]:
    frame = _cross_frame()
    analysis = analyze_univariate(
        frame,
        features=["age", "score"],
        target="bad",
        methods=["equal_width"],
        bin_count=4,
        min_bin_pct=0,
        loan_amount="loan",
        overdue_amount="overdue",
    )
    generation_parameters = {
        "analysis_schema_version": analysis["schema_version"],
        "features": ["age", "score"],
        "methods": ["equal_width"],
        "bin_count": 4,
        "sample_design_ref": {
            "artifact_id": HASH_A,
            "artifact_content_hash": HASH_B,
            "sample_design_id": f"strategy-sample-design-cross-{seed}",
            "sample_design_content_hash": HASH_C,
            "partition": "development",
        },
    }
    evidence = build_candidate_evidence(
        task_id=task_id,
        dataset_id="dataset-cross-1",
        dataset_content_hash=HASH_A,
        workspace_revision=2,
        workspace_generation=3,
        semantic_mapping_hash=HASH_B,
        generation_parameters=generation_parameters,
        seed=seed,
        budget=100_000,
        truncated=False,
        analysis=analysis,
        metrics=[
            {
                "metric_name": "univariate.iv",
                "dimension": "count",
                "status": "observed",
                "value": analysis["rankings"][0]["iv"],
            },
            {
                "metric_name": "univariate.iv",
                "dimension": "loan_amount",
                "status": "unavailable",
                "value": None,
            },
            {
                "metric_name": "univariate.iv",
                "dimension": "overdue_amount",
                "status": "unavailable",
                "value": None,
            },
        ],
        source_refs=["dataset:dataset-cross-1"],
        producer_version="strategy.univariate-candidate/1",
    )
    content = render_strategy_candidate_bundle(evidence, analysis)["json"]
    content_hash = hashlib.sha256(content).hexdigest()
    path = (
        app.state.settings.tasks_dir
        / task_id
        / directory_name
        / f"{evidence['candidate_id']}_{content_hash[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = TaskArtifactRepository(app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_candidate_json",
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.analyze_univariate_candidates",
        provenance={
            "schema_version": "strategy.univariate-candidate-artifact.v1",
            "producer_version": evidence["producer_version"],
            "candidate_id": provenance_candidate_id or evidence["candidate_id"],
            "evidence_hash": evidence["evidence_hash"],
            "dataset_id": evidence["identity"]["dataset_id"],
            "dataset_content_hash": evidence["identity"]["dataset_content_hash"],
            "registry_metadata_hash": HASH_C,
            "workspace_revision": evidence["identity"]["workspace_revision"],
            "workspace_generation": evidence["identity"]["workspace_generation"],
            "semantic_mapping_hash": evidence["identity"]["semantic_mapping_hash"],
            "generation_parameters": generation_parameters,
            "format": "json",
        },
        created_at=created_at,
    )
    return record, path, evidence


def _cross_measurement(evidence: dict) -> dict:
    frame = _cross_frame()
    feature_methods = {
        row["feature"]: row["methods"][0]
        for row in evidence["analysis"]["features"]
    }
    cells = []
    for row_bin in feature_methods["age"]["bins"]:
        row_mask = evaluate_expression_frame(frame, row_bin["condition"])
        for column_bin in feature_methods["score"]["bins"]:
            mask = row_mask & evaluate_expression_frame(
                frame,
                column_bin["condition"],
            )
            selected = frame.loc[mask]
            cells.append(
                {
                    "row_source_bin_id": row_bin["id"],
                    "column_source_bin_id": column_bin["id"],
                    "count": len(selected),
                    "good": int((selected["bad"] == 0).sum()),
                    "bad": int((selected["bad"] == 1).sum()),
                    "amounts": {
                        "loan_amount": {
                            "status": "available",
                            "covered_count": len(selected),
                            "value": float(selected["loan"].sum()),
                        },
                        "overdue_amount": {
                            "status": "available",
                            "covered_count": len(selected),
                            "value": float(selected["overdue"].sum()),
                        },
                        "paired": {
                            "status": "available",
                            "covered_count": len(selected),
                            "loan_value": float(selected["loan"].sum()),
                            "overdue_value": float(selected["overdue"].sum()),
                        },
                    },
                }
            )
    return {
        "schema_version": CROSS_MATRIX_MEASUREMENT_SCHEMA_VERSION,
        "sample_context_hash": sample_context_hash_from_candidate_evidence(evidence),
        "population_count": len(frame),
        "good": int((frame["bad"] == 0).sum()),
        "bad": int((frame["bad"] == 1).sum()),
        "cells": cells,
    }


def _register_cross_candidate(
    app,
    task_id: str,
    *,
    evidence: dict,
    source_record: dict,
) -> tuple[dict, Path, dict]:
    sample_identity = {
        **evidence["identity"],
        "sample_context_hash": sample_context_hash_from_candidate_evidence(evidence),
        "target_col": evidence["analysis"]["target"],
        "row_count": evidence["analysis"]["row_count"],
    }
    asset = build_cross_matrix_candidate_asset(
        evidence,
        row_axis={"feature": "age", "method": "equal_width"},
        column_axis={"feature": "score", "method": "equal_width"},
        sample_identity=sample_identity,
        measurement=_cross_measurement(evidence),
        budget=100,
    )
    content = canonical_cross_matrix_candidate_asset_json(asset).encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    path = (
        app.state.settings.tasks_dir
        / task_id
        / "strategy_cross_matrix_candidates"
        / f"{asset['asset_id']}_{content_hash[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    identity = asset["parent"]["identity"]
    sample = asset["sample_identity"]
    lifecycle = asset["lifecycle"]
    record = TaskArtifactRepository(app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_cross_matrix_candidate_json",
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.build_cross_matrix_candidate",
        provenance={
            "schema_version": "strategy.cross-matrix-candidate-artifact.v1",
            "producer_version": asset["producer_version"],
            "asset_schema_version": asset["schema_version"],
            "asset_type": asset["asset_type"],
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "parent_candidate_id": asset["parent"]["candidate_id"],
            "parent_evidence_hash": asset["parent"]["evidence_hash"],
            "candidate_id": asset["candidate_evidence"]["candidate_id"],
            "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
            "source_artifact_id": source_record["id"],
            "source_artifact_content_hash": source_record["content_hash"],
            "task_id": identity["task_id"],
            "dataset_id": sample["dataset_id"],
            "dataset_content_hash": sample["dataset_content_hash"],
            "registry_metadata_hash": source_record["provenance"][
                "registry_metadata_hash"
            ],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "sample_context_hash": sample["sample_context_hash"],
            "target_col": sample["target_col"],
            "labeled_row_count": sample["row_count"],
            "row_axis": {"feature": "age", "method": "equal_width"},
            "column_axis": {"feature": "score", "method": "equal_width"},
            "cell_count": asset["matrix"]["cell_count"],
            "candidate_stage": lifecycle["candidate_stage"],
            "observation_stage": lifecycle["observation_stage"],
            "validation_status": lifecycle["validation_status"],
            "budget": asset["budget"]["limit"],
            "truncated": asset["budget"]["truncated"],
        },
    )
    return record, path, asset


def _hide_univariate_artifacts_from_candidate_window(
    monkeypatch: pytest.MonkeyPatch,
    artifact_ids: set[str],
) -> None:
    original = TaskArtifactRepository.list_recent_for_task_kind_with_count

    def bounded_without_hidden(self, task_id, kind, *, limit):
        records, total = original(self, task_id, kind, limit=limit)
        if kind == "strategy_candidate_json":
            records = [
                record for record in records if record["id"] not in artifact_ids
            ]
        return records, total

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_recent_for_task_kind_with_count",
        bounded_without_hidden,
    )


def _register_refined_asset(
    app,
    task_id: str,
    *,
    drift_provenance: bool = False,
) -> tuple[dict, Path, dict, dict]:
    parent_record, _path, evidence = _register_cross_source(app, task_id)
    source_bin_id = evidence["analysis"]["features"][0]["methods"][0]["bins"][0]["id"]
    asset = refine_univariate_candidate(
        evidence,
        _cross_frame(),
        source_evidence={
            "artifact_id": parent_record["id"],
            "kind": parent_record["kind"],
            "content_hash": parent_record["content_hash"],
        },
        feature="age",
        method="equal_width",
        merge_groups=[],
        selection={"source_bin_ids": [source_bin_id]},
        selection_reason="candidate lab test",
    )
    content = canonical_candidate_asset_json(asset).encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    path = (
        app.state.settings.tasks_dir
        / task_id
        / "strategy_candidate_assets"
        / f"{asset['asset_id']}_{content_hash[:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    identity = evidence["identity"]
    provenance = {
        "schema_version": "strategy.candidate-asset-artifact.v1",
        "producer_version": asset["producer_version"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_id": evidence["candidate_id"],
        "evidence_hash": evidence["evidence_hash"],
        "source_artifact_id": parent_record["id"],
        "source_artifact_content_hash": parent_record["content_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "feature": asset["feature"],
        "method": asset["method"],
    }
    if drift_provenance:
        provenance["candidate_id"] = "candidate-" + "f" * 32
    record = TaskArtifactRepository(app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_candidate_asset_json",
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.refine_univariate_candidate",
        provenance=provenance,
    )
    source_binding = {
        "artifact_id": record["id"],
        "kind": record["kind"],
        "content_hash": record["content_hash"],
        "origin_tool": record["origin_tool"],
        "artifact_schema_version": "strategy.candidate-asset-artifact.v1",
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_kind": asset["asset_type"],
        "fragment_id": asset["rule"]["rule_id"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_stage": asset["effect_stage"],
        "validation_status": asset["validation_status"],
        "parent_candidate_id": evidence["candidate_id"],
        "parent_evidence_hash": evidence["evidence_hash"],
        "evidence_identity": {
            key: identity[key]
            for key in (
                "dataset_id",
                "dataset_content_hash",
                "workspace_revision",
                "workspace_generation",
                "semantic_mapping_hash",
            )
        },
    }
    fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=source_binding,
        candidate_evidence=evidence,
    )
    return record, path, fragment, evidence


def _persist_initial_pool(
    app,
    task_id: str,
    *,
    strategy_type: str,
    fragment: dict,
) -> dict:
    snapshot = add_verified_candidate_fragment(
        None,
        task_id=task_id,
        strategy_type=strategy_type,
        default_action={
            "type": "approval",
            "value": "approve",
            "reason_code": None,
            "stop": True,
        },
        verified_candidate_fragment=fragment,
        action={
            "type": "reject",
            "value": "reject",
            "reason_code": "TEST",
            "stop": True,
        },
        reason="candidate lab test",
    )
    content = canonical_strategy_pool_snapshot_json(snapshot).encode("utf-8")
    content_hash = strategy_pool_artifact_content_hash(snapshot)
    path = (
        app.state.settings.tasks_dir
        / task_id
        / "strategy_candidate_pools"
        / (
            f"{snapshot['pool_id']}_r{snapshot['revision']}_"
            f"{snapshot['snapshot_hash'][:12]}.json"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    entries = snapshot["entries"]
    record = TaskArtifactRepository(app.state.settings.db_path).register(
        task_id=task_id,
        kind=POOL_ARTIFACT_KIND,
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.add_candidate_to_pool",
        provenance={
            "schema_version": "strategy.candidate-pool-artifact.v2",
            "producer_version": "strategy.candidate-pool/2",
            "pool_id": snapshot["pool_id"],
            "strategy_type": snapshot["strategy_type"],
            "revision": snapshot["revision"],
            "revision_id": snapshot["revision_id"],
            "parent_revision_id": snapshot["parent_revision_id"],
            "snapshot_hash": snapshot["snapshot_hash"],
            "operation_kind": snapshot["operation"]["kind"],
            "source_artifact_ids": [
                entry["source"]["artifact_id"] for entry in entries
            ],
            "evidence_identity": entries[0]["source"]["evidence_identity"],
        },
    )
    StrategyCandidatePoolRepository(app.state.settings.db_path).apply_snapshot(
        snapshot=snapshot,
        expected_revision=ABSENT_POOL_REVISION,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=record["id"],
        artifact_content_hash=record["content_hash"],
        audit={
            "kind": "strategy.pool.add_candidate",
            "target_ref": snapshot["revision_id"],
            "inputs_hash": snapshot["operation"]["operation_hash"],
            "outcome": "succeeded",
            "detail": {"entry_count": len(entries)},
        },
    )
    return snapshot


def test_candidate_lab_empty_projection_is_task_scoped_and_bounded(tmp_path: Path) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema_version": "strategy.candidate-lab-projection.v1",
        "task_id": task_id,
        "can_start": True,
        "blocked_reason": None,
        "active_plan": None,
        "open_gate": None,
        "candidates": {
            kind: {"latest": None, "all": [], "total": 0, "truncated": False}
            for kind in ("univariate", "cross_matrix", "automatic_tree")
        },
        "pools": {"latest": None, "all": [], "total": 0, "truncated": False},
    }


def test_task_artifact_candidate_lookup_is_exact_and_capped_at_two(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    task_id = _strategy_task(app)
    repository = TaskArtifactRepository(app.state.settings.db_path)
    candidate_id = "candidate-" + "a" * 32
    expected_ids = []
    for index in range(3):
        record = repository.register(
            task_id=task_id,
            kind="strategy_candidate_json",
            path=str(tmp_path / f"candidate-{index}.json"),
            content_hash=HASH_A,
            origin_tool="strategy.analyze_univariate_candidates",
            provenance={"candidate_id": candidate_id},
            created_at=f"2026-07-24T00:00:0{index}+00:00",
        )
        expected_ids.append(record["id"])
    repository.register(
        task_id=task_id,
        kind="strategy_candidate_json",
        path=str(tmp_path / "numeric-candidate.json"),
        content_hash=HASH_A,
        origin_tool="strategy.analyze_univariate_candidates",
        provenance={"candidate_id": 123},
    )

    matches = (
        repository.find_for_task_kind_origin_by_provenance_candidate_id(
            task_id,
            "strategy_candidate_json",
            "strategy.analyze_univariate_candidates",
            candidate_id,
        )
    )

    assert [record["id"] for record in matches] == list(
        reversed(expected_ids[-2:])
    )


def test_candidate_lab_returns_verified_ui_safe_univariate_projection(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    record, _path = _register_univariate_candidate(app, task_id)

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    collection = response.json()["candidates"]["univariate"]
    assert collection["total"] == 1
    assert collection["truncated"] is False
    assert collection["latest"] == collection["all"][0]
    item = collection["latest"]
    assert item["kind"] == "univariate"
    assert item["artifact"] == {
        "artifact_id": record["id"],
        "created_at": record["created_at"],
        "download_url": (
            f"/api/tasks/{task_id}/task-artifacts/{record['id']}/download"
        ),
    }
    assert item["candidate_id"].startswith("candidate-")
    assert item["detail"]["rankings"][0]["feature"] == "score"
    assert item["risks"] == {
        "red_flags": ["test_warning"],
        "report_info_gaps": [],
    }
    assert item["pointers"]["bins"]
    assert set(item["pointers"]["bins"][0]) == {
        "feature",
        "method",
        "bin_id",
        "condition",
        "metrics",
    }
    assert "path" not in str(response.json())
    assert "source_refs" not in str(response.json())


def test_candidate_lab_fails_closed_when_registered_bytes_drift(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _record, path = _register_univariate_candidate(app, task_id)
    path.write_text('{"forged":true}', encoding="utf-8")

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == "strategy candidate lab evidence verification failed"


def test_candidate_lab_cross_rejects_source_at_noncanonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    source, _path, evidence = _register_cross_source(
        app,
        task_id,
        directory_name="strategy_candidates_shadow",
    )
    _register_cross_candidate(
        app,
        task_id,
        evidence=evidence,
        source_record=source,
    )
    _hide_univariate_artifacts_from_candidate_window(
        monkeypatch,
        {source["id"]},
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409


def test_candidate_lab_cross_rejects_unrelated_valid_source_report(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _expected_source, _path, evidence = _register_cross_source(
        app,
        task_id,
        seed=1,
    )
    unrelated_source, _path, _unrelated_evidence = _register_cross_source(
        app,
        task_id,
        seed=2,
    )
    _register_cross_candidate(
        app,
        task_id,
        evidence=evidence,
        source_record=unrelated_source,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409


def test_candidate_lab_cross_rejects_source_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    source, _path, evidence = _register_cross_source(
        app,
        task_id,
        provenance_candidate_id="candidate-" + "f" * 32,
    )
    _register_cross_candidate(
        app,
        task_id,
        evidence=evidence,
        source_record=source,
    )
    _hide_univariate_artifacts_from_candidate_window(
        monkeypatch,
        {source["id"]},
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409


def test_candidate_lab_pool_replays_source_and_rejects_provenance_drift(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _record, _path, fragment, _evidence = _register_refined_asset(
        app,
        task_id,
        drift_provenance=True,
    )
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=fragment,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409


def test_candidate_lab_reads_duplicate_pool_source_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _record, asset_path, fragment, _evidence = _register_refined_asset(
        app,
        task_id,
    )
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=fragment,
    )
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="reject",
        fragment=fragment,
    )
    original = candidate_lab_projection._read_regular_file
    reads = 0

    def counting_read(path, **kwargs):
        nonlocal reads
        if path == asset_path:
            reads += 1
        return original(path, **kwargs)

    monkeypatch.setattr(
        candidate_lab_projection,
        "_read_regular_file",
        counting_read,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    assert response.json()["pools"]["total"] == 2
    assert reads == 1


def test_candidate_lab_bounds_deep_validation_to_latest_twenty_candidates(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    oldest, oldest_path = _register_univariate_candidate(
        app,
        task_id,
        seed=0,
        created_at="2026-07-24T00:00:00+00:00",
    )
    for seed in range(1, 21):
        _register_univariate_candidate(
            app,
            task_id,
            seed=seed,
            created_at=f"2026-07-24T00:00:{seed:02d}+00:00",
        )
    artifacts = TaskArtifactRepository(app.state.settings.db_path)
    for index in range(200):
        artifacts.register(
            task_id=task_id,
            kind="strategy_candidate_json",
            path=str(
                app.state.settings.tasks_dir
                / task_id
                / "historical_invalid_candidates"
                / f"{index}.json"
            ),
            content_hash=HASH_A,
            origin_tool="historical.invalid_origin",
            provenance={"candidate_id": f"historical-{index}"},
            created_at=f"2020-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
        )
    # Old registry rows and bytes remain outside the newest-first SQL window;
    # the exact COUNT still reports them without decoding the full history.
    oldest_path.write_text('{"historical_bytes_are_outside_window":true}', "utf-8")

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    collection = response.json()["candidates"]["univariate"]
    assert collection["total"] == 221
    assert collection["truncated"] is True
    assert len(collection["all"]) == 20
    assert all(
        item["artifact"]["artifact_id"] != oldest["id"]
        for item in collection["all"]
    )


def test_candidate_lab_never_reads_unbounded_artifact_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _register_univariate_candidate(app, task_id)

    def reject_unbounded_history(*_args, **_kwargs):
        raise AssertionError("Candidate Lab must not call list_for_task")

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_for_task",
        reject_unbounded_history,
    )
    monkeypatch.setattr(
        PlanRepository,
        "list_plans_for_task",
        reject_unbounded_history,
    )
    monkeypatch.setattr(
        TaskRepository,
        "list_agent_messages",
        reject_unbounded_history,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    assert response.json()["candidates"]["univariate"]["total"] == 1


def test_candidate_lab_enforces_one_aggregate_artifact_byte_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _register_univariate_candidate(app, task_id)
    monkeypatch.setattr(candidate_lab_projection, "_MAX_PROJECTION_BYTES", 1)

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_response_omits_platform_bindings_and_hashes(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _register_univariate_candidate(app, task_id)
    source, _path, evidence = _register_cross_source(app, task_id, seed=99)
    _register_cross_candidate(
        app,
        task_id,
        evidence=evidence,
        source_record=source,
    )
    _record, _asset_path, fragment, _evidence = _register_refined_asset(
        app,
        task_id,
    )
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=fragment,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    forbidden_keys = {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "registry_metadata_hash",
        "semantic_mapping_hash",
        "sample_context_hash",
        "sample_design_ref",
        "provenance",
        "source_refs",
    }

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in forbidden_keys
                assert not key.endswith("_hash")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(response.json())


def test_candidate_lab_queries_only_latest_nonterminal_plan_summary(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    repository = PlanRepository(app.state.settings.db_path)
    repository.create_plan(
        Plan(
            id="terminal-plan",
            task_id=task_id,
            goal="done",
            source="agent",
            template_id=None,
            steps=[],
            autonomy_level=1,
            status=PlanStatus.DONE,
            created_at="2026-07-24T00:00:00+00:00",
        )
    )
    repository.create_plan(
        Plan(
            id="active-plan",
            task_id=task_id,
            goal="active",
            source="agent",
            template_id=None,
            steps=[],
            autonomy_level=1,
            status=PlanStatus.DRAFT,
            created_at="2026-07-24T00:00:01+00:00",
        )
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["can_start"] is False
    assert payload["blocked_reason"] == "active_plan"
    assert payload["active_plan"] == {
        "plan_id": "active-plan",
        "status": "draft",
    }


def test_candidate_lab_blocks_start_while_confirmation_gate_is_open(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    repository = TaskRepository(app.state.settings.db_path)
    task_id = _strategy_task(app)
    message = repository.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="confirm",
        metadata={"kind": "gate", "step_id": "strategy-step-1"},
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["can_start"] is False
    assert payload["blocked_reason"] == "open_gate"
    assert payload["active_plan"] is None
    assert payload["open_gate"] == {
        "message_id": message["id"],
        "kind": "gate",
        "step_id": "strategy-step-1",
    }


def test_candidate_lab_rejects_non_strategy_tasks(tmp_path: Path) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="not strategy",
            model_version="v2",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="validation",
        )
    )

    response = client.get(f"/api/tasks/{task.id}/strategy-candidate-lab")

    assert response.status_code == 422

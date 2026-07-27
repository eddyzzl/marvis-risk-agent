from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from marvis.api_schemas import ManualStrategyRequest
from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.feature.univariate import analyze_univariate
from marvis.output.strategy_candidate_report import render_strategy_candidate_bundle
from marvis.orchestrator.contracts import Plan, PlanStatus
from marvis.packs.strategy import (
    candidate_lab_projection,
    voting_candidate_search_tools,
    voting_candidate_tools,
)
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
)
from marvis.packs.modeling.evidence_tools import (
    TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL,
)
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION,
)
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
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import add_verified_candidate_fragment
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.pool_impact_tools import run_measure_pool_impact
from marvis.packs.strategy.pool_stability_tools import (
    run_measure_strategy_pool_stability,
)
from marvis.packs.strategy.pool_validation_tools import (
    run_measure_strategy_pool_validation,
)
from marvis.packs.strategy.scorecard_candidate import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    build_scorecard_band_asset,
    build_scorecard_cutoff_selection,
    canonical_scorecard_band_asset_json,
    canonical_scorecard_cutoff_selection_json,
    scorecard_cutoff_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.voting_candidate import build_voting_candidate_asset
from marvis.packs.strategy.voting_candidate_search import (
    canonical_voting_candidate_search_result_json,
    search_voting_candidate_combinations,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    voting_candidate_to_verified_fragment,
)
from marvis.packs.strategy.voting_candidate_tools import (
    build_voting_candidate_artifact_document,
    canonical_voting_candidate_artifact_json,
    canonical_voting_candidate_path,
    voting_candidate_artifact_provenance,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_ORIGIN_TOOL,
)
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
    canonical_strategy_pool_snapshot_json,
    strategy_pool_artifact_content_hash,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.strategy_reports import StrategyReportRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.test_strategy_candidate_stability_tools import (
    _pool_add_inputs,
    _setup,
)
from tests.test_strategy_pool_impact_tools import (
    _setup as _pool_impact_setup,
)
from tests.test_strategy_pool_stability_tools import (
    _setup as _pool_stability_setup,
)
from tests.test_strategy_report_repository import (
    _bundle as _report_bundle,
    _register_outputs as _register_report_outputs,
    _seed_strategy_task as _seed_report_strategy_task,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


@pytest.fixture(autouse=True)
def _fast_scorecard_live_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Candidate Lab tests fast; authoritative loader has its own suite."""

    def load(runtime, **request):
        record = runtime.task_artifacts.get_for_task(
            request["task_id"],
            request["artifact_id"],
        )
        assert record is not None
        raw = Path(record["path"]).read_bytes()
        asset = candidate_lab_projection.validate_scorecard_band_asset(
            json.loads(raw)
        )
        assert record["content_hash"] == request[
            "expected_artifact_content_hash"
        ]
        assert asset["asset_id"] == request["expected_asset_id"]
        assert asset["asset_hash"] == request["expected_asset_hash"]
        return SimpleNamespace(asset=asset, canonical_bytes=raw)

    monkeypatch.setattr(
        candidate_lab_projection,
        "load_scorecard_band_asset_artifact",
        load,
    )


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


def _searched_candidate_lab_fixture(tmp_path: Path) -> dict:
    fixture = _setup(tmp_path)
    pool = None
    for candidate in (
        fixture["first"],
        fixture["refine"](1),
        fixture["refine"](2),
    ):
        added = run_add_candidate_to_pool(
            _pool_add_inputs(
                candidate,
                expected_revision=0 if pool is None else pool["revision"],
                expected_hash=(
                    ABSENT_POOL_SNAPSHOT_HASH
                    if pool is None
                    else pool["snapshot_hash"]
                ),
            ),
            fixture["ctx"],
            fixture["runtime"],
        )
        pool = added["pool"]
    assert pool is not None
    controls = {
        "strategy_type": "approval",
        "member_count": 2,
        "n": 1,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
        "constraints": [
            {"metric": "hit_share", "operator": "gte", "value": 0.05}
        ],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 100,
    }
    inputs = (
        voting_candidate_search_tools.resolve_voting_candidate_search_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_controls=controls,
        )
    )
    search = voting_candidate_search_tools.run_search_voting_candidates(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    app = create_app(fixture["settings"])
    return {
        **fixture,
        "app": app,
        "client": TestClient(app),
        "pool": pool,
        "controls": controls,
        "search": search,
    }


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
    sample_design_ref = {
        "artifact_id": HASH_A,
        "artifact_content_hash": HASH_B,
        "sample_design_id": f"strategy-sample-design-cross-{seed}",
        "sample_design_content_hash": HASH_C,
        "partition": "development",
    }
    generation_parameters = {
        "analysis_schema_version": analysis["schema_version"],
        "features": ["age", "score"],
        "methods": ["equal_width"],
        "bin_count": 4,
        "sample_design_ref": sample_design_ref,
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
        source_refs=[
            "dataset:dataset-cross-1",
            "strategy-sample-design:"
            + json.dumps(
                {
                    "kind": "strategy_sample_design",
                    **sample_design_ref,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        ],
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
    _persist_pool_snapshot(
        app,
        task_id,
        snapshot=snapshot,
        expected_revision=ABSENT_POOL_REVISION,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    return snapshot


def _persist_pool_snapshot(
    app,
    task_id: str,
    *,
    snapshot: dict,
    expected_revision: int,
    expected_snapshot_hash: str,
) -> dict:
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
        expected_revision=expected_revision,
        expected_snapshot_hash=expected_snapshot_hash,
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
    return record


def _scorecard_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _scorecard_scale() -> dict[str, float | int]:
    factor = 50.0 / math.log(2.0)
    return {
        "base_score": 600,
        "pdo": 50,
        "base_odds": 50.0,
        "factor": factor,
        "offset": 600.0 - factor * math.log(50.0),
    }


def _scorecard_table() -> list[dict[str, object]]:
    scale = _scorecard_scale()
    return [
        {
            "feature": "__base__",
            "bin_index": -999,
            "bin_label": "base_points",
            "lower": None,
            "upper": None,
            "count": None,
            "bad_count": None,
            "good_count": None,
            "bad_rate": None,
            "woe": None,
            "iv_contribution": None,
            "coefficient": None,
            "monotonic_direction": None,
            "points": 320.0,
            **scale,
        },
        {
            "feature": "income",
            "bin_index": 0,
            "bin_label": "[-inf, 10)",
            "lower": None,
            "upper": 10.0,
            "count": 3,
            "bad_count": 2,
            "good_count": 1,
            "bad_rate": 2.0 / 3.0,
            "woe": 0.4,
            "iv_contribution": 0.08,
            "coefficient": 0.5,
            "monotonic_direction": "increasing",
            "points": -14.0,
        },
        {
            "feature": "income",
            "bin_index": 1,
            "bin_label": "[10, inf)",
            "lower": 10.0,
            "upper": None,
            "count": 3,
            "bad_count": 0,
            "good_count": 3,
            "bad_rate": 0.0,
            "woe": -0.4,
            "iv_contribution": 0.08,
            "coefficient": 0.5,
            "monotonic_direction": "increasing",
            "points": 14.0,
        },
    ]


def _large_scorecard_table(row_count: int) -> list[dict[str, object]]:
    base = _scorecard_table()[0]
    return [
        base,
        *[
            {
                "feature": f"feature_{index:05d}",
                "bin_index": 0,
                "bin_label": "all",
                "lower": None,
                "upper": None,
                "count": 6,
                "bad_count": 2,
                "good_count": 4,
                "bad_rate": 1.0 / 3.0,
                "woe": 0.0,
                "iv_contribution": 0.0,
                "coefficient": 0.0,
                "monotonic_direction": None,
                "points": float(index % 100),
            }
            for index in range(row_count - 1)
        ],
    ]


def _scorecard_band_provenance(asset: dict) -> dict:
    identity = asset["identity"]
    refs = asset["source_refs"]
    return {
        "schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        "asset_schema_version": asset["schema_version"],
        "producer_version": asset["producer_version"],
        "task_id": identity["task_id"],
        "asset_type": asset["asset_type"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "sample_context_hash": identity["sample_context_hash"],
        "sample_design_ref": asset["sample_design_ref"],
        "training_evidence_ref": refs["training_evidence"],
        "score_evidence_ref": refs["score_evidence"],
        "score_vector_ref": refs["score_vector"],
        "score_product": asset["score_contract"]["score_product"],
        "scorecard_table_hash": asset["score_contract"]["scorecard_table_hash"],
        "raw_pd_internal_edges": [
            band["upper_bound"] for band in asset["bands"][:-1]
        ],
        "band_count": len(asset["bands"]),
        "cutoff_count": len(asset["cutoffs"]),
    }


def _scorecard_selection_provenance(selection: dict) -> dict:
    source = selection["source_asset_ref"]
    return {
        "schema_version": SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
        "selection_schema_version": selection["schema_version"],
        "producer_version": selection["producer_version"],
        "task_id": source["task_id"],
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "cutoff_id": selection["cutoff_id"],
        "selection_reason": selection["selection_reason"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_content_hash": source["artifact_content_hash"],
        "source_asset_id": source["asset_id"],
        "source_asset_hash": source["asset_hash"],
    }


def _register_scorecard_candidate(
    app,
    task_id: str,
    *,
    variant: int = 0,
    created_at: str | None = None,
    band_directory_name: str = "strategy_scorecard_candidates",
    drift_band_provenance: bool = False,
    scorecard_rows: list[dict[str, object]] | None = None,
) -> tuple[dict, Path, dict, dict, Path, dict, dict]:
    artifacts = TaskArtifactRepository(app.state.settings.db_path)
    task_root = app.state.settings.tasks_dir / task_id
    identity = {
        "task_id": task_id,
        "dataset_id": f"dataset-scorecard-{variant}",
        "dataset_content_hash": _scorecard_hash(f"dataset-{variant}"),
        "workspace_revision": variant + 1,
        "workspace_generation": 1,
        "semantic_mapping_hash": _scorecard_hash(f"semantics-{variant}"),
        "sample_context_hash": _scorecard_hash(f"sample-context-{variant}"),
    }
    membership_id = f"sample-membership-{variant}"
    membership_content_hash = _scorecard_hash(
        f"membership-logical-content-{variant}"
    )
    membership_path = (
        task_root / "strategy_sample_designs_v2" / f"{membership_id}.bin"
    )
    membership_bytes = f"membership-source-{variant}".encode()
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    membership_path.write_bytes(membership_bytes)
    membership_record = artifacts.register(
        task_id=task_id,
        kind=SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        path=str(membership_path),
        content_hash=hashlib.sha256(membership_bytes).hexdigest(),
        origin_tool=SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        provenance={
            "schema_version": SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
            "task_id": task_id,
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "format": "binary",
            "artifact_role": "membership",
            "membership_id": membership_id,
            "membership_content_hash": membership_content_hash,
            "membership_artifact_content_hash": hashlib.sha256(
                membership_bytes
            ).hexdigest(),
        },
    )
    bundle_id = f"strategy-sample-design-bundle-{variant}"
    sample_design_id = f"strategy-sample-design-{variant}"
    sample_design_content_hash = _scorecard_hash(f"sample-design-{variant}")
    bundle_path = (
        task_root / "strategy_sample_designs_v2" / f"{bundle_id}.json"
    )
    bundle_bytes = f'{{"bundle_fixture":{variant}}}'.encode()
    bundle_path.write_bytes(bundle_bytes)
    bundle_record = artifacts.register(
        task_id=task_id,
        kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        path=str(bundle_path),
        content_hash=hashlib.sha256(bundle_bytes).hexdigest(),
        origin_tool=SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        provenance={
            "schema_version": SAMPLE_DESIGN_V2_ARTIFACT_SCHEMA_VERSION,
            "task_id": task_id,
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "format": "json",
            "artifact_role": "bundle",
            "membership_id": membership_id,
            "membership_content_hash": membership_content_hash,
            "membership_artifact_id": membership_record["id"],
            "membership_artifact_content_hash": membership_record[
                "content_hash"
            ],
            "bundle_id": bundle_id,
            "bundle_artifact_content_hash": hashlib.sha256(
                bundle_bytes
            ).hexdigest(),
            "sample_design_id": sample_design_id,
            "sample_design_content_hash": sample_design_content_hash,
        },
    )
    sample_design_ref = {
        "membership_artifact_id": membership_record["id"],
        "expected_membership_artifact_content_hash": membership_record[
            "content_hash"
        ],
        "bundle_artifact_id": bundle_record["id"],
        "expected_bundle_artifact_content_hash": bundle_record["content_hash"],
        "expected_bundle_id": bundle_id,
        "expected_sample_design_id": sample_design_id,
        "expected_sample_design_content_hash": sample_design_content_hash,
    }
    training_evidence_id = f"training-evidence-{variant}"
    training_evidence_content_hash = _scorecard_hash(
        f"training-evidence-logical-{variant}"
    )
    training_path = (
        task_root
        / "modeling_artifacts"
        / f"{training_evidence_id}.training_evidence.json"
    )
    training_bytes = f'{{"training_fixture":{variant}}}'.encode()
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_bytes(training_bytes)
    training_record = artifacts.register(
        task_id=task_id,
        kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        path=str(training_path),
        content_hash=hashlib.sha256(training_bytes).hexdigest(),
        origin_tool=TRAIN_MODEL_WITH_EVIDENCE_V2_ORIGIN_TOOL,
        provenance={
            "schema_version": TRAINING_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
            "format": "json",
            "artifact_role": "training_evidence",
            "task_id": task_id,
            "evidence_id": training_evidence_id,
            "evidence_content_hash": training_evidence_content_hash,
            "evidence_artifact_content_hash": hashlib.sha256(
                training_bytes
            ).hexdigest(),
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "sample_design_id": sample_design_id,
            "sample_design_content_hash": sample_design_content_hash,
            "sample_membership_id": membership_id,
            "sample_membership_content_hash": membership_content_hash,
            "sample_membership_artifact_id": membership_record["id"],
            "sample_membership_artifact_content_hash": membership_record[
                "content_hash"
            ],
            "sample_bundle_artifact_id": bundle_record["id"],
            "sample_bundle_artifact_content_hash": bundle_record[
                "content_hash"
            ],
        },
    )
    training_evidence_ref = {
        "artifact_id": training_record["id"],
        "artifact_content_hash": training_record["content_hash"],
        "evidence_id": training_evidence_id,
        "evidence_content_hash": training_evidence_content_hash,
    }
    score_request_hash = _scorecard_hash(f"score-request-{variant}")
    score_dir = task_root / "model_score_evidence"
    score_dir.mkdir(parents=True, exist_ok=True)
    vector_path = score_dir / f"{score_request_hash}.scores.parquet"
    vector_bytes = f"score-vector-fixture-{variant}".encode()
    vector_path.write_bytes(vector_bytes)
    score_lineage = {
        "schema_version": (
            MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_TOOL_SCHEMA_VERSION
        ),
        "task_id": task_id,
        "request_hash": score_request_hash,
        "training_evidence_id": training_evidence_id,
        "training_evidence_content_hash": training_evidence_content_hash,
        "training_evidence_artifact_id": training_record["id"],
        "training_evidence_artifact_content_hash": training_record[
            "content_hash"
        ],
        "sample_design_id": sample_design_id,
        "sample_design_content_hash": sample_design_content_hash,
        "sample_membership_id": membership_id,
        "sample_membership_content_hash": membership_content_hash,
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "score_product": "raw_native_uncalibrated_bad_probability",
    }
    vector_record = artifacts.register(
        task_id=task_id,
        kind=MODEL_SCORE_VECTOR_ARTIFACT_KIND,
        path=str(vector_path),
        content_hash=hashlib.sha256(vector_bytes).hexdigest(),
        origin_tool=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
        provenance={
            **score_lineage,
            "format": "parquet",
            "artifact_role": "model_score_vector",
            "row_count": 6,
        },
    )
    score_evidence_id = f"score-evidence-{variant}"
    score_evidence_content_hash = _scorecard_hash(
        f"score-evidence-logical-{variant}"
    )
    score_path = (
        score_dir / f"{score_request_hash}.model_score_evidence.json"
    )
    score_bytes = f'{{"score_fixture":{variant}}}'.encode()
    score_path.write_bytes(score_bytes)
    score_record = artifacts.register(
        task_id=task_id,
        kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        path=str(score_path),
        content_hash=hashlib.sha256(score_bytes).hexdigest(),
        origin_tool=MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
        provenance={
            **score_lineage,
            "format": "json",
            "artifact_role": "model_score_evidence",
            "score_vector_artifact_id": vector_record["id"],
            "score_vector_artifact_content_hash": vector_record[
                "content_hash"
            ],
            "score_evidence_id": score_evidence_id,
            "score_evidence_content_hash": score_evidence_content_hash,
            "score_evidence_artifact_content_hash": hashlib.sha256(
                score_bytes
            ).hexdigest(),
        },
    )
    score_evidence_ref = {
        "artifact_id": score_record["id"],
        "artifact_content_hash": score_record["content_hash"],
        "evidence_id": score_evidence_id,
        "evidence_content_hash": score_evidence_content_hash,
    }
    score_vector_ref = {
        "artifact_id": vector_record["id"],
        "artifact_content_hash": vector_record["content_hash"],
    }
    asset = build_scorecard_band_asset(
        identity=identity,
        sample_design_ref=sample_design_ref,
        training_evidence_ref=training_evidence_ref,
        score_evidence_ref=score_evidence_ref,
        score_vector_ref=score_vector_ref,
        score_product="raw_native_uncalibrated_bad_probability",
        score_direction="higher_is_riskier",
        points_direction="higher_is_better",
        scorecard_scale=_scorecard_scale(),
        scorecard_table=(
            _scorecard_table()
            if scorecard_rows is None
            else scorecard_rows
        ),
        raw_pd=np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
        risk_development_mask=np.ones(6, dtype=np.bool_),
        labels=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, np.nan]),
        score_bins=[
            {
                "ordinal": 0,
                "bin_id": "score-bin-00",
                "lower_bound": None,
                "upper_bound": 0.3,
                "lower_inclusive": False,
                "upper_inclusive": False,
            },
            {
                "ordinal": 1,
                "bin_id": "score-bin-01",
                "lower_bound": 0.3,
                "upper_bound": 0.7,
                "lower_inclusive": True,
                "upper_inclusive": False,
            },
            {
                "ordinal": 2,
                "bin_id": "score-bin-02",
                "lower_bound": 0.7,
                "upper_bound": None,
                "lower_inclusive": True,
                "upper_inclusive": False,
            },
        ],
    )
    band_bytes = canonical_scorecard_band_asset_json(asset).encode("utf-8")
    band_path = (
        app.state.settings.tasks_dir
        / task_id
        / band_directory_name
        / f"{asset['asset_id']}.json"
    )
    band_path.parent.mkdir(parents=True, exist_ok=True)
    band_path.write_bytes(band_bytes)
    band_provenance = _scorecard_band_provenance(asset)
    if drift_band_provenance:
        band_provenance["score_vector_ref"] = {
            **band_provenance["score_vector_ref"],
            "artifact_content_hash": "f" * 64,
        }
    band_record = artifacts.register(
        task_id=task_id,
        kind=SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        path=str(band_path),
        content_hash=hashlib.sha256(band_bytes).hexdigest(),
        origin_tool=SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        provenance=band_provenance,
        created_at=created_at,
    )
    selection_record, selection_path, fragment = (
        _register_scorecard_selection(
            app,
            task_id,
            asset=asset,
            band_record=band_record,
            cutoff_ordinal=0,
            selection_reason="候选实验室明确选择",
            created_at=created_at,
        )
    )
    return (
        band_record,
        band_path,
        asset,
        selection_record,
        selection_path,
        fragment,
        {
            "membership": (membership_record, membership_path),
            "bundle": (bundle_record, bundle_path),
            "training": (training_record, training_path),
            "score": (score_record, score_path),
            "vector": (vector_record, vector_path),
        },
    )


def _register_scorecard_selection(
    app,
    task_id: str,
    *,
    asset: dict,
    band_record: dict,
    cutoff_ordinal: int,
    selection_reason: str,
    created_at: str | None = None,
) -> tuple[dict, Path, dict]:
    band_bytes = canonical_scorecard_band_asset_json(asset).encode("utf-8")
    band_binding = {
        "artifact_id": band_record["id"],
        "task_id": task_id,
        "kind": SCORECARD_BAND_ASSET_ARTIFACT_KIND,
        "artifact_schema_version": SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
        "content_hash": band_record["content_hash"],
        "origin_tool": SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        "canonical_bytes": band_bytes,
    }
    selection = build_scorecard_cutoff_selection(
        asset,
        source_artifact_binding=band_binding,
        cutoff_id=asset["cutoffs"][cutoff_ordinal]["cutoff_id"],
        selection_reason=selection_reason,
    )
    selection_bytes = canonical_scorecard_cutoff_selection_json(selection).encode(
        "utf-8"
    )
    selection_path = (
        app.state.settings.tasks_dir
        / task_id
        / "strategy_scorecard_candidates"
        / f"{selection['selection_id']}.json"
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_bytes(selection_bytes)
    selection_record = TaskArtifactRepository(
        app.state.settings.db_path
    ).register(
        task_id=task_id,
        kind=SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        path=str(selection_path),
        content_hash=hashlib.sha256(selection_bytes).hexdigest(),
        origin_tool=SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        provenance=_scorecard_selection_provenance(selection),
        created_at=created_at,
    )
    selection_binding = {
        "artifact_id": selection_record["id"],
        "task_id": task_id,
        "kind": SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
        "artifact_schema_version": (
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "content_hash": selection_record["content_hash"],
        "origin_tool": SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        "canonical_bytes": selection_bytes,
    }
    fragment = scorecard_cutoff_selection_to_verified_candidate_fragment(
        selection,
        asset,
        selection_artifact_binding=selection_binding,
        source_artifact_binding=band_binding,
    )
    return selection_record, selection_path, fragment


def test_candidate_lab_replays_scorecard_pool_and_projects_safe_evidence(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        _band_record,
        _band_path,
        asset,
        _selection_record,
        _selection_path,
        fragment,
        _sources,
    ) = _register_scorecard_candidate(app, task_id)
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=fragment,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "strategy.candidate-lab-projection.v4"
    band = body["candidates"]["scorecard_band"]["latest"]
    assert band["detail"]["asset_id"] == asset["asset_id"]
    assert band["detail"]["performance"] == {"auc": 1.0, "ks": 1.0}
    assert band["detail"]["directions"] == {
        "raw_pd": {
            "direction": "higher_is_riskier",
            "meaning": "higher_raw_pd_means_higher_risk",
        },
        "scorecard_points": {
            "direction": "higher_is_better",
            "meaning": "higher_points_mean_safer",
        },
    }
    assert len(band["pointers"]["bands"]) == 3
    assert len(band["pointers"]["cutoffs"]) == 2
    assert band["pointers"]["scorecard_points"] == _scorecard_table()
    selection = body["candidates"]["scorecard_cutoff_selection"]["latest"]
    assert selection["detail"] == {
        "selection_id": selection["candidate_id"],
        "asset_id": asset["asset_id"],
        "cutoff_id": asset["cutoffs"][0]["cutoff_id"],
        "reason": "候选实验室明确选择",
        "directions": band["detail"]["directions"],
        "effect": band["pointers"]["cutoffs"][0],
    }
    assert body["pools"]["total"] == 1
    assert body["pools"]["latest"]["entries"][0]["source"]["asset_type"] == (
        "scorecard_band"
    )


def test_candidate_lab_bounds_scorecard_point_rows(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    cap = candidate_lab_projection._MAX_SCORECARD_POINT_POINTERS
    scorecard_rows = _large_scorecard_table(cap + 1)
    (
        _band_record,
        _band_path,
        asset,
        _selection_record,
        _selection_path,
        _fragment,
        _sources,
    ) = _register_scorecard_candidate(
        app,
        task_id,
        scorecard_rows=scorecard_rows,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    projected = response.json()["candidates"]["scorecard_band"]["latest"]
    points = projected["pointers"]["scorecard_points"]
    assert len(points) == cap
    assert projected["total"] == (
        len(asset["bands"]) + len(asset["cutoffs"]) + len(scorecard_rows)
    )
    assert projected["truncated"] is True
    assert points[0]["feature"] == "__base__"
    assert points[-1]["feature"] == f"feature_{cap - 2:05d}"
    assert {
        "feature",
        "bin_index",
        "bin_label",
        "woe",
        "coefficient",
        "points",
    } <= set(points[-1])
    assert not any("hash" in key or "ref" in key for key in points[-1])


def test_candidate_lab_replays_voting_with_nested_scorecard_requirements(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        band_record,
        _band_path,
        asset,
        _first_selection_record,
        first_selection_path,
        first_fragment,
        sources,
    ) = _register_scorecard_candidate(app, task_id)
    _second_selection, _second_path, second_fragment = (
        _register_scorecard_selection(
            app,
            task_id,
            asset=asset,
            band_record=band_record,
            cutoff_ordinal=1,
            selection_reason="Voting 第二个阈值",
        )
    )
    first_pool = _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=first_fragment,
    )
    parent_pool = add_verified_candidate_fragment(
        first_pool,
        task_id=task_id,
        strategy_type="approval",
        default_action={
            "type": "approval",
            "value": "approve",
            "reason_code": None,
            "stop": True,
        },
        verified_candidate_fragment=second_fragment,
        action={
            "type": "reject",
            "value": "reject",
            "reason_code": "TEST",
            "stop": True,
        },
        reason="candidate lab voting parent",
    )
    parent_record = _persist_pool_snapshot(
        app,
        task_id,
        snapshot=parent_pool,
        expected_revision=first_pool["revision"],
        expected_snapshot_hash=first_pool["snapshot_hash"],
    )
    bundle_record, _bundle_path = sources["bundle"]
    sample_ref = {
        "artifact_id": bundle_record["id"],
        "artifact_content_hash": bundle_record["content_hash"],
        "sample_design_id": asset["sample_design_ref"][
            "expected_sample_design_id"
        ],
        "sample_design_content_hash": asset["sample_design_ref"][
            "expected_sample_design_content_hash"
        ],
        "partition": "development",
    }
    hit_count = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    target = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    voting_mask = hit_count >= 2
    effect = voting_candidate_tools._effect_from_mask(
        voting_mask,
        target=target,
        population_count=6,
    )
    distribution = voting_candidate_tools._hit_distribution(
        hit_count,
        target=target,
        k=2,
    )
    observations = voting_candidate_tools._metric_observations(
        voting_mask,
        hit_count=hit_count,
        target=target,
        amount_values={"loan_amount": None, "overdue_amount": None},
        k=2,
    )
    voting_asset = build_voting_candidate_asset(
        parent_pool,
        selected_entry_ids=[
            entry["entry_id"] for entry in parent_pool["entries"]
        ],
        n=2,
        target_col="bad",
        sample_design_ref=sample_ref,
        effect=effect,
    )
    voting_document = build_voting_candidate_artifact_document(
        voting_asset,
        target_col="bad",
        drop_nan_labels=True,
        nan_labels_dropped=1,
        population_count=6,
        labeled_count=5,
        hit_distribution=distribution,
        metric_observations=observations,
    )
    voting_bytes = canonical_voting_candidate_artifact_json(
        voting_document
    ).encode("utf-8")
    voting_path = canonical_voting_candidate_path(
        app.state.settings.tasks_dir,
        task_id=task_id,
        asset_id=voting_asset["asset_id"],
    )
    voting_path.parent.mkdir(parents=True, exist_ok=True)
    voting_path.write_bytes(voting_bytes)
    voting_record = TaskArtifactRepository(
        app.state.settings.db_path
    ).register(
        task_id=task_id,
        kind=VOTING_CANDIDATE_ARTIFACT_KIND,
        path=str(voting_path),
        content_hash=hashlib.sha256(voting_bytes).hexdigest(),
        origin_tool=VOTING_CANDIDATE_ORIGIN_TOOL,
        provenance=voting_candidate_artifact_provenance(
            voting_document,
            task_id=task_id,
            pool_artifact={
                "id": parent_record["id"],
                "content_hash": parent_record["content_hash"],
            },
        ),
    )
    voting_fragment = voting_candidate_to_verified_fragment(
        voting_asset,
        artifact_binding={
            "artifact_id": voting_record["id"],
            "task_id": task_id,
            "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
            "content_hash": voting_record["content_hash"],
            "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
            "artifact_schema_version": (
                VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
            ),
            "asset_id": voting_asset["asset_id"],
            "asset_hash": voting_asset["asset_hash"],
        },
    )
    current_pool = add_verified_candidate_fragment(
        parent_pool,
        task_id=task_id,
        strategy_type="approval",
        default_action={
            "type": "approval",
            "value": "approve",
            "reason_code": None,
            "stop": True,
        },
        verified_candidate_fragment=voting_fragment,
        action={
            "type": "reject",
            "value": "reject",
            "reason_code": "TEST",
            "stop": True,
        },
        placement_mode="replace_selected_members",
        selected_entry_ids=[
            entry["entry_id"] for entry in parent_pool["entries"]
        ],
        reason="candidate lab voting replacement",
    )
    _persist_pool_snapshot(
        app,
        task_id,
        snapshot=current_pool,
        expected_revision=parent_pool["revision"],
        expected_snapshot_hash=parent_pool["snapshot_hash"],
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    voting_pool = next(
        pool
        for pool in response.json()["pools"]["all"]
        if pool["strategy_type"] == "approval"
    )
    execution = voting_pool["entries"][0]["execution"]
    assert execution["requirement_types"] == [
        "model_score_vector.v1",
        "model_score_vector.v1",
    ]
    assert execution["condition"]["op"] == "n_of_k"
    assert {
        condition["field"] for condition in execution["condition"]["args"]
    } == {"scorecard_raw_pd"}

    for index in range(3):
        _register_scorecard_selection(
            app,
            task_id,
            asset=asset,
            band_record=band_record,
            cutoff_ordinal=0,
            selection_reason=f"Voting 历史窗口占位 {index}",
        )
    first_selection_path.write_bytes(b"nested-scorecard-selection-drift")

    drifted = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert drifted.status_code == 409
    assert drifted.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_scorecard_projection_omits_private_bindings_and_hashes(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        _band_record,
        _band_path,
        asset,
        _selection_record,
        _selection_path,
        fragment,
        _sources,
    ) = _register_scorecard_candidate(app, task_id)
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=fragment,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text

    def all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key
                for nested in value.values()
                for key in all_keys(nested)
            }
        if isinstance(value, list):
            return {
                key
                for nested in value
                for key in all_keys(nested)
            }
        return set()

    keys = all_keys(response.json())
    forbidden = {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
        "source_refs",
        "sample_design_ref",
        "asset_hash",
        "selection_hash",
        "content_hash",
        "artifact_content_hash",
        "raw_pd_content_hash",
        "scorecard_table_hash",
        "score_evidence_artifact_content_hash",
        "score_vector_artifact_content_hash",
    }
    assert forbidden & keys == set()
    private_virtual_field = (
        "__marvis_model_pd_"
        + asset["source_refs"]["score_vector"]["artifact_id"][:16]
    )
    assert private_virtual_field not in str(response.json())


@pytest.mark.parametrize(
    "mutation",
    ("bytes", "path", "provenance", "missing_source"),
)
def test_candidate_lab_scorecard_lineage_tampering_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        band_record,
        band_path,
        _asset,
        _selection_record,
        _selection_path,
        fragment,
        _sources,
    ) = _register_scorecard_candidate(
        app,
        task_id,
        band_directory_name=(
            "strategy_scorecard_candidates_shadow"
            if mutation == "path"
            else "strategy_scorecard_candidates"
        ),
        drift_band_provenance=mutation == "provenance",
    )
    _persist_initial_pool(
        app,
        task_id,
        strategy_type="approval",
        fragment=fragment,
    )
    if mutation == "bytes":
        band_path.write_bytes(b"{}")
    elif mutation == "missing_source":
        with TaskArtifactRepository(
            app.state.settings.db_path
        ).transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM task_artifacts WHERE id = ?",
                (band_record["id"],),
            )
            conn.commit()

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


@pytest.mark.parametrize(
    "source_name",
    ("training", "score", "vector", "membership", "bundle"),
)
def test_candidate_lab_scorecard_upstream_source_drift_fails_closed(
    tmp_path: Path,
    source_name: str,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        _band_record,
        _band_path,
        _asset,
        _selection_record,
        _selection_path,
        _fragment,
        sources,
    ) = _register_scorecard_candidate(app, task_id)
    _record, source_path = sources[source_name]
    source_path.write_bytes(b"upstream-source-drift")

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_authoritative_scorecard_replay_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    _register_scorecard_candidate(app, task_id)

    def reject_live_sources(*_args, **_kwargs):
        raise StrategyError("live scorecard evidence replay failed")

    monkeypatch.setattr(
        candidate_lab_projection,
        "load_scorecard_band_asset_artifact",
        reject_live_sources,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_bounds_scorecard_history_before_deep_replay(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        oldest_band,
        oldest_band_path,
        _asset,
        oldest_selection,
        oldest_selection_path,
        _fragment,
        _sources,
    ) = _register_scorecard_candidate(
        app,
        task_id,
        created_at="2026-07-24T00:00:00+00:00",
    )
    oldest_band_path.write_bytes(b"not-read-old-band")
    oldest_selection_path.write_bytes(b"not-read-old-selection")
    for variant in range(1, 4):
        _register_scorecard_candidate(
            app,
            task_id,
            variant=variant,
            created_at=f"2026-07-24T00:00:{variant:02d}+00:00",
        )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    candidates = response.json()["candidates"]
    for kind, oldest in (
        ("scorecard_band", oldest_band),
        ("scorecard_cutoff_selection", oldest_selection),
    ):
        collection = candidates[kind]
        assert collection["total"] == 4
        assert collection["truncated"] is True
        assert len(collection["all"]) == 3
        assert oldest["id"] not in {
            item["artifact"]["artifact_id"] for item in collection["all"]
        }


def test_candidate_lab_reuses_scorecard_verification_across_history_and_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    (
        _band_record,
        band_path,
        _asset,
        _selection_record,
        selection_path,
        fragment,
        _sources,
    ) = _register_scorecard_candidate(app, task_id)
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
    reads = {band_path: 0, selection_path: 0}

    def counting_read(path, **kwargs):
        if path in reads:
            reads[path] += 1
        return original(path, **kwargs)

    monkeypatch.setattr(
        candidate_lab_projection,
        "_read_regular_file",
        counting_read,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    assert response.json()["pools"]["total"] == 2
    assert reads == {band_path: 1, selection_path: 1}


def test_candidate_lab_empty_projection_is_task_scoped_and_bounded(tmp_path: Path) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema_version": "strategy.candidate-lab-projection.v4",
        "task_id": task_id,
        "can_start": True,
        "blocked_reason": None,
        "active_plan": None,
        "open_gate": None,
        "candidates": {
            kind: {"latest": None, "all": [], "total": 0, "truncated": False}
            for kind in (
                "univariate",
                "cross_matrix",
                "automatic_tree",
                "interactive_tree_revision",
                "scorecard_band",
                "scorecard_cutoff_selection",
                "voting_search",
            )
        },
        "pool_add_sources": {
            "latest": None,
            "all": [],
            "total": 0,
            "truncated": False,
        },
        "pools": {"latest": None, "all": [], "total": 0, "truncated": False},
        "workflow": {
            "sample_design": None,
            "latest_evidence": {
                "pool_stability": None,
                "pool_impact": None,
                "impact_cube": None,
                "pool_validation": {
                    "validation": None,
                    "oot": None,
                },
            },
            "report": None,
            "stages": [
                {
                    "id": "current_context",
                    "label": "项目现状",
                    "status": "missing",
                },
                {
                    "id": "history",
                    "label": "历史版本",
                    "status": "missing",
                },
                {
                    "id": "sample_design",
                    "label": "样本设计",
                    "status": "missing",
                },
                {
                    "id": "candidate_analysis",
                    "label": "单变量/模型",
                    "status": "missing",
                },
                {
                    "id": "strategy_combination",
                    "label": "交叉组合/策略",
                    "status": "missing",
                },
                {
                    "id": "impact",
                    "label": "影响测算",
                    "status": "missing",
                },
                {
                    "id": "report",
                    "label": "形成报告",
                    "status": "missing",
                },
            ],
        },
    }


def test_candidate_lab_v4_projects_authenticated_native_dual_population_sample(
    tmp_path: Path,
) -> None:
    fixture = _setup(
        tmp_path,
        native_sample=True,
        target_bad_value=0,
    )
    app = create_app(fixture["settings"])
    response = TestClient(app).get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "strategy.candidate-lab-projection.v4"
    sample = body["workflow"]["sample_design"]
    assert sample["source_mode"] == "native_active_dataset"
    assert sample["relationship"] == "parallel_time_cohorts"
    assert sample["freshness"] == "current"
    assert sample["target"] == {
        "column": "bad",
        "good_value": 1,
        "bad_value": 0,
        "missing_policy": "keep",
    }
    assert sample["populations"] == {
        "approval": {
            "total_count": 40,
            "partitions": {
                "development": 30,
                "validation": 5,
                "oot": 5,
            },
            "maturity": {
                "status": "not_applicable",
                "performance_window_days": None,
                "cutoff_date": None,
                "eligible_count": None,
                "labeled_count": None,
                "reason": "Approval population does not require outcome maturity.",
            },
        },
        "risk": {
            "total_count": 80,
            "partitions": {
                "development": 60,
                "validation": 10,
                "oot": 10,
            },
            "maturity": {
                "status": "confirmed_matured",
                "performance_window_days": 30,
                "cutoff_date": "2026-05-31",
                "eligible_count": 80,
                "labeled_count": 80,
                "reason": None,
            },
        },
    }
    assert sample["relationship_counts"] == {
        "approval_and_risk": 0,
        "approval_only": 40,
        "risk_only": 80,
        "neither": 0,
    }
    assert sample["diagnostics"]["overall_status"] == "unavailable"
    assert body["workflow"]["stages"][2]["status"] == "complete"
    assert "membership" not in sample
    assert "masks" not in str(sample)


def test_candidate_lab_v4_projects_latest_authenticated_strategy_evidence(
    tmp_path: Path,
) -> None:
    fixture = _pool_stability_setup(tmp_path)
    stability_output = run_measure_strategy_pool_stability(
        fixture["stability_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    validation_output = run_measure_strategy_pool_validation(
        fixture["validation_request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    response = TestClient(create_app(fixture["settings"])).get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    evidence = response.json()["workflow"]["latest_evidence"]
    impact_cube = evidence["impact_cube"]
    assert impact_cube["cube_id"] == fixture["impact_output"]["cube_id"]
    assert impact_cube["strategy_type"] == "approval"
    assert impact_cube["partitions"] == ["development", "validation"]
    assert impact_cube["freshness"] == "current"
    stability = evidence["pool_stability"]
    assert stability["stability_id"] == stability_output["stability_id"]
    assert stability["comparison_partitions"] == ["validation"]
    assert stability["freshness"] == "current"
    validation = evidence["pool_validation"]["validation"]
    assert validation["evidence_id"] == validation_output["evidence_id"]
    assert validation["partition"] == "validation"
    assert validation["lifecycle"]["validation_status"] == (
        "independent_evidence"
    )
    assert validation["freshness"] == "current"
    assert evidence["pool_validation"]["oot"] is None
    assert response.json()["workflow"]["stages"][5]["status"] == "complete"


def test_candidate_lab_v4_projects_latest_authenticated_legacy_pool_impact(
    tmp_path: Path,
) -> None:
    fixture = _pool_impact_setup(tmp_path)
    output = run_measure_pool_impact(
        fixture["request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    response = TestClient(create_app(fixture["settings"])).get(
        f"/api/tasks/{fixture['task'].id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    impact = response.json()["workflow"]["latest_evidence"]["pool_impact"]
    assert impact["assessment_id"] == output["assessment_id"]
    assert impact["strategy_type"] == "approval"
    assert impact["population_count"] == output["population_count"]
    assert impact["labeled_count"] == output["labeled_count"]
    assert impact["monthly_status"] == output["monthly_status"]
    assert impact["freshness"] == "current"


def test_candidate_lab_v4_projects_latest_authenticated_four_format_report(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path / "workspace")
    task_id, strategy_id = _seed_report_strategy_task(settings.db_path)
    with connect(settings.db_path) as connection:
        connection.execute(
            "UPDATE tasks SET status = 'created' WHERE id = ?",
            (task_id,),
        )
    bundle = _report_bundle(task_id=task_id, strategy_id=strategy_id)
    artifacts = _register_report_outputs(settings.db_path, bundle)
    StrategyReportRepository(settings.db_path).publish(
        bundle=bundle,
        artifacts=artifacts,
        expected_revision=0,
        expected_report_id=None,
        expected_content_hash=None,
    )

    response = TestClient(create_app(settings)).get(
        f"/api/tasks/{task_id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    report = response.json()["workflow"]["report"]
    assert report["report_id"] == bundle["report_id"]
    assert report["revision"] == 1
    assert report["status"] == "partial"
    assert report["title"] == "策略迭代报告"
    assert report["freshness"] == "current"
    assert set(report["artifacts"]) == {"json", "markdown", "xlsx", "docx"}
    for output_format, artifact in report["artifacts"].items():
        assert artifact["artifact_id"] == artifacts[output_format]["id"]
        assert artifact["download_url"].endswith(
            f"/{artifacts[output_format]['id']}/download"
        )
    assert response.json()["workflow"]["stages"][6]["status"] == "complete"


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "pricing",
                "partitions": ["development", "validation", "oot"],
                "month_col": "apply_month",
            },
        ),
        (
            "strategy_report_bundle_v2",
            {"title": "策略迭代评审报告", "status": "partial"},
        ),
    ],
)
def test_manual_strategy_request_accepts_v2_workflow_spine_entries(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": workflow,
            "workflow_inputs": workflow_inputs,
        },
        strict=True,
    )

    assert request.workflow == workflow
    assert request.workflow_inputs == workflow_inputs


def test_manual_strategy_request_rejects_impact_platform_bindings() -> None:
    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted|platform-owned",
    ):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_impact_cube",
                "workflow_inputs": {
                    "strategy_type": "approval",
                    "artifact_id": "a" * 64,
                },
            },
            strict=True,
        )


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


def test_candidate_lab_projects_authenticated_voting_search_as_ui_safe_evidence(
    tmp_path: Path,
) -> None:
    fixture = _searched_candidate_lab_fixture(tmp_path)
    task_id = fixture["task"].id
    search = fixture["search"]
    [descriptor] = search["artifacts"]

    response = fixture["client"].get(
        f"/api/tasks/{task_id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    collection = response.json()["candidates"]["voting_search"]
    assert collection["total"] == 1
    assert collection["truncated"] is False
    assert collection["latest"] == collection["all"][0]
    item = collection["latest"]
    assert set(item) == {
        "search_id",
        "strategy_type",
        "pool_revision",
        "member_count",
        "n",
        "objective",
        "constraints",
        "include_rule_ids",
        "exclude_rule_ids",
        "max_combinations",
        "search_space",
        "evaluated",
        "eligible",
        "truncated",
        "combinations",
        "artifact",
    }
    assert item == {
        "search_id": search["search_id"],
        "strategy_type": "approval",
        "pool_revision": fixture["pool"]["revision"],
        "member_count": fixture["controls"]["member_count"],
        "n": fixture["controls"]["n"],
        "objective": fixture["controls"]["objective"],
        "constraints": fixture["controls"]["constraints"],
        "include_rule_ids": fixture["controls"]["include_rule_ids"],
        "exclude_rule_ids": fixture["controls"]["exclude_rule_ids"],
        "max_combinations": fixture["controls"]["max_combinations"],
        "search_space": search["search_space"],
        "evaluated": search["evaluated"],
        "eligible": search["search_result"]["eligible"],
        "truncated": search["truncated"],
        "combinations": [
            {
                "combo_id": combo["combo_id"],
                "members": combo["member_ids"],
                "eligible": combo["eligible"],
                "failures": combo["constraint_failures"],
                "metrics": combo["metrics"],
            }
            for combo in search["search_result"]["combinations"][:20]
        ],
        "artifact": {
            "artifact_id": descriptor["artifact_id"],
            "created_at": TaskArtifactRepository(
                fixture["settings"].db_path
            ).get_for_task(task_id, descriptor["artifact_id"])["created_at"],
            "download_url": (
                f"/api/tasks/{task_id}/task-artifacts/"
                f"{descriptor['artifact_id']}/download"
            ),
        },
    }
    forbidden_keys = {
        "candidate_ids",
        "content_hash",
        "dataset_binding",
        "hit_count_distribution",
        "hit_matrix",
        "labels",
        "lifecycle",
        "objective_value",
        "path",
        "population",
        "provenance",
        "rank",
        "request_hash",
        "score_vector",
        "selected",
        "target",
        "weights",
    }

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert key not in forbidden_keys
                assert not key.endswith("_hash")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(item)
    serialized = json.dumps(item, ensure_ascii=False).lower()
    assert "champion" not in serialized
    assert '"best"' not in serialized


def test_candidate_lab_bounds_voting_search_history_before_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _searched_candidate_lab_fixture(tmp_path)
    task_id = fixture["task"].id
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    [oldest_descriptor] = fixture["search"]["artifacts"]
    oldest_record = repository.get_for_task(
        task_id,
        oldest_descriptor["artifact_id"],
    )
    assert oldest_record is not None
    for index in range(21):
        threshold = index / 100
        if threshold == 0.05:
            continue
        controls = {
            **fixture["controls"],
            "constraints": [
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "value": threshold,
                }
            ],
        }
        inputs = (
            voting_candidate_search_tools.resolve_voting_candidate_search_inputs(
                fixture["runtime"],
                task_id=task_id,
                user_controls=controls,
            )
        )
        voting_candidate_search_tools.run_search_voting_candidates(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )
    Path(oldest_record["path"]).write_text(
        '{"old_search_outside_the_projection_window":true}',
        encoding="utf-8",
    )

    def reject_unbounded_history(*_args, **_kwargs):
        raise AssertionError("Candidate Lab must not call list_for_task")

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_for_task",
        reject_unbounded_history,
    )

    response = fixture["client"].get(
        f"/api/tasks/{task_id}/strategy-candidate-lab"
    )

    assert response.status_code == 200, response.text
    collection = response.json()["candidates"]["voting_search"]
    assert collection["total"] == 21
    assert collection["truncated"] is True
    assert len(collection["all"]) == 20
    assert all(
        item["artifact"]["artifact_id"] != oldest_descriptor["artifact_id"]
        for item in collection["all"]
    )


def test_candidate_lab_fails_closed_when_latest_voting_search_is_corrupt(
    tmp_path: Path,
) -> None:
    fixture = _searched_candidate_lab_fixture(tmp_path)
    task_id = fixture["task"].id
    [descriptor] = fixture["search"]["artifacts"]
    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(task_id, descriptor["artifact_id"])
    assert record is not None
    Path(record["path"]).write_text('{"forged":true}', encoding="utf-8")

    response = fixture["client"].get(
        f"/api/tasks/{task_id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_requires_authoritative_voting_search_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _searched_candidate_lab_fixture(tmp_path)
    task_id = fixture["task"].id

    def reject_live_sources(*_args, **_kwargs):
        raise StrategyError("historical Voting search replay failed")

    monkeypatch.setattr(
        candidate_lab_projection,
        "load_historical_voting_candidate_search_artifact",
        reject_live_sources,
    )

    response = fixture["client"].get(
        f"/api/tasks/{task_id}/strategy-candidate-lab"
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "strategy candidate lab evidence verification failed"
    )


def test_candidate_lab_caps_projected_voting_combinations_at_twenty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _strategy_task(app)
    candidate_ids = [f"rule-{index}" for index in range(7)]
    target = [index % 2 for index in range(20)]
    result = search_voting_candidate_combinations(
        {
            "schema_version": "strategy.voting-candidate-search-request.v1",
            "candidate_ids": candidate_ids,
            "hit_matrix": [
                [
                    (row_index + candidate_index) % (candidate_index + 2) == 0
                    for row_index in range(20)
                ]
                for candidate_index in range(7)
            ],
            "target": target,
            "weights": None,
            "amounts": None,
            "member_count": 3,
            "n": 2,
            "objective": {
                "metric": "bad_capture_rate",
                "direction": "maximize",
            },
            "constraints": [
                {"metric": "hit_share", "operator": "gte", "value": 0.01}
            ],
            "include": [],
            "exclude": [],
            "max_combinations": 35,
        }
    )
    raw = canonical_voting_candidate_search_result_json(result).encode("utf-8")
    content_hash = hashlib.sha256(raw).hexdigest()
    path = (
        app.state.settings.tasks_dir
        / task_id
        / "strategy_voting_candidate_searches"
        / f"{result['search_id']}-{'d' * 16}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    record = TaskArtifactRepository(app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_voting_candidate_search_json",
        path=str(path),
        content_hash=content_hash,
        origin_tool="strategy.search_voting_candidates",
        provenance={},
    )

    def load(_runtime, **request):
        assert request == {
            "task_id": task_id,
            "artifact_id": record["id"],
            "expected_artifact_content_hash": content_hash,
        }
        return SimpleNamespace(
            result=result,
            pool_development=SimpleNamespace(
                pool=SimpleNamespace(
                    pool={"strategy_type": "approval", "revision": 4}
                )
            ),
        )

    monkeypatch.setattr(
        candidate_lab_projection,
        "load_historical_voting_candidate_search_artifact",
        load,
    )

    response = client.get(f"/api/tasks/{task_id}/strategy-candidate-lab")

    assert response.status_code == 200, response.text
    item = response.json()["candidates"]["voting_search"]["latest"]
    assert result["evaluated"] == 35
    assert item["evaluated"] == 35
    assert item["truncated"] is False
    assert len(item["combinations"]) == 20
    assert [combo["combo_id"] for combo in item["combinations"]] == [
        combo["combo_id"] for combo in result["combinations"][:20]
    ]


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

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
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy import candidate_stability_tools
from marvis.packs.strategy.candidate_stability import (
    validate_candidate_stability_artifact,
)
from marvis.packs.strategy.candidate_stability_tools import (
    ARTIFACT_KIND,
    StrategyCandidateStabilityArtifactBinding,
    load_candidate_stability_artifact,
    require_candidate_stability_artifact_binding_on_connection,
    resolve_candidate_monthly_stability_inputs,
    run_measure_candidate_monthly_stability,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _governed_model_score_requirement() -> dict:
    return {
        "rule_id": "governed-model-score-rule",
        "fragment_id": "governed-model-score-fragment",
        "requirement": {
            "type": "model_score_vector.v1",
            "virtual_field": "__marvis_model_pd_0123456789abcdef",
            "score_product": "raw_native_uncalibrated_bad_probability",
            "score_evidence_artifact_id": "1" * 64,
            "score_evidence_artifact_content_hash": "2" * 64,
            "score_vector_artifact_id": "0123456789abcdef" + "3" * 48,
            "score_vector_artifact_content_hash": "4" * 64,
        },
    }


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


def _setup(
    tmp_path: Path,
    *,
    bind_month: bool = True,
    native_sample: bool = False,
    target_bad_value: int = 1,
) -> dict:
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
    frame_data = {
        "score": list(range(120)),
        "age": [20 + (index % 60) for index in range(120)],
        "month": ["2026-01"] * 40 + ["2026-02"] * 40 + ["2026-03"] * 40,
        "bad": [index % 2 for index in range(120)],
    }
    if native_sample:
        frame_data.update(
            {
                "bad": [1 if index % 4 == 0 else 0 for index in range(120)],
                "customer_id": [f"C{index:03d}" for index in range(120)],
                "apply_date": (
                    pd.date_range("2026-01-01", periods=120, freq="D")
                    .strftime("%Y-%m-%d")
                    .tolist()
                ),
                "segment": [["A", "B", "C"][index % 3] for index in range(120)],
                "sample_split": (
                    ["dev"] * 90 + ["valid"] * 15 + ["oot"] * 15
                ),
            }
        )
    frame = pd.DataFrame(frame_data)
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
            "age": "feature",
            "month": "month",
            "bad": "target",
            **(
                {
                    "customer_id": "id",
                    "apply_date": "date",
                    "segment": "categorical",
                    "sample_split": "segment",
                }
                if native_sample
                else {}
            ),
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
    if native_sample:
        def eq(column: str, value: object) -> dict:
            return {
                "op": "eq",
                "left": {"column": column},
                "right": {"literal": value},
            }

        def any_of(*predicates: dict) -> dict:
            return {"op": "or", "args": list(predicates)}

        sample = strategy_tools.tool_materialize_sample_design_v2_native(
            {
                "source_mode": "native_active_dataset",
                "dataset_id": dataset.id,
                "expected_dataset_content_hash": dataset.content_hash,
                "workspace_revision": workspace.revision,
                "workspace_generation": workspace.analysis_generation,
                "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
                "target_col": "bad",
                "target_bad_value": target_bad_value,
                "drop_nan_labels": False,
                "relationship": "parallel_time_cohorts",
                "scope": "strategy_development",
                "approval_population": {
                    "inclusion": eq("segment", "A"),
                    "exclusion": None,
                },
                "risk_population": {
                    "inclusion": any_of(
                        eq("segment", "B"),
                        eq("segment", "C"),
                    ),
                    "exclusion": None,
                },
                "partitioning": {
                    "method": "predicate_ast",
                    "selectors": {
                        "development": eq("sample_split", "dev"),
                        "validation": eq("sample_split", "valid"),
                        "oot": eq("sample_split", "oot"),
                    },
                },
                "maturity": {
                    "status": "confirmed_matured",
                    "performance_window_days": 30,
                    "cutoff_date": "2026-05-31",
                    "reason": None,
                },
                "performance_window": {"status": "provided", "days": 30},
                "observation_window": {
                    "status": "provided",
                    "start": "2026-01-01",
                    "end": "2026-05-31",
                },
                "field_bindings": {
                    "entity_field": "customer_id",
                    "time_field": "apply_date",
                    "group_field": "segment",
                    "month_field": "month" if bind_month else None,
                    "weight_field": None,
                    "loan_amount_field": None,
                    "overdue_amount_field": None,
                },
                "historical_score": {
                    "status": "unavailable",
                    "column": None,
                    "direction": None,
                    "reason": "not supplied for stability test",
                },
                "policy": {
                    "minimum_partition_count": 1,
                    "minimum_bad_count": 1,
                    "minimum_label_coverage": 1.0,
                    "minimum_historical_score_coverage": 0.0,
                    "maximum_group_coverage_gap": 1.0,
                    "diagnostic_severities": {
                        "entity_overlap": "warn",
                        "temporal_oot": "warn",
                        "risk_outside_approval": "warn",
                        "maturity": "fail",
                        "label_coverage": "fail",
                        "historical_score_coverage": "warn",
                        "group_coverage_gap": "warn",
                        "sufficiency": "fail",
                    },
                },
            },
            ctx,
        )
        bundle_record = next(
            record
            for record in TaskArtifactRepository(
                settings.db_path
            ).list_for_task(task.id)
            if record["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
            and record["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
        )
        sample_ref = {
            "artifact_id": bundle_record["id"],
            "artifact_content_hash": bundle_record["content_hash"],
            "sample_design_id": sample["sample_design_id"],
            "sample_design_content_hash": sample[
                "sample_design_content_hash"
            ],
            "partition": "risk/development",
        }
    else:
        sample_request = {
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
            "features": ["score", "age"],
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
        "source": source,
        "source_report": report,
        "sample_ref": sample_ref,
        "frame": frame,
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


def test_native_pool_entry_stability_uses_risk_development_mask_and_bad_zero(
    tmp_path: Path,
) -> None:
    fixture = _setup(
        tmp_path,
        native_sample=True,
        target_bad_value=0,
    )
    current = run_add_candidate_to_pool(
        _pool_add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    [entry] = current["entries"]

    output = run_measure_candidate_monthly_stability(
        resolve_candidate_monthly_stability_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_pointer={
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": entry["entry_id"],
            },
        ),
        fixture["ctx"],
        fixture["runtime"],
    )

    stability = output["stability"]
    assert stability["sample_design_ref"] == fixture["sample_ref"]
    assert stability["sample_design_ref"]["partition"] == "risk/development"
    assert stability["bindings"]["target_bad_value"] == 1
    assert stability["summary"]["population_count"] == 60
    assert [row["sample_count"] for row in stability["monthly"]] == [26, 27, 7]
    assert stability["baseline"]["hit_count"] == 20
    assert stability["baseline"]["hit_bad_count"] == 15
    assert stability["baseline"]["hit_bad_rate"] == 0.75


def test_native_pool_stability_requirement_receives_exact_physical_v2_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(
        tmp_path,
        native_sample=True,
        target_bad_value=0,
    )
    current = run_add_candidate_to_pool(
        _pool_add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    [entry] = current["entries"]
    request = resolve_candidate_monthly_stability_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_pointer={
            "source_kind": "pool_entry",
            "strategy_type": "approval",
            "entry_id": entry["entry_id"],
        },
    )
    requirement = _governed_model_score_requirement()
    resolved = object()
    captured = {}

    monkeypatch.setattr(
        candidate_stability_tools,
        "project_pool_entry_requirements",
        lambda _entries: (requirement,),
    )

    def resolve_requirements(
        runtime,
        *,
        task_id,
        compiled_design,
        sample_design,
    ):
        captured.update(
            runtime=runtime,
            task_id=task_id,
            compiled_design=compiled_design,
            sample_design=sample_design,
        )
        return resolved

    monkeypatch.setattr(
        candidate_stability_tools,
        "resolve_pool_requirements",
        resolve_requirements,
    )

    binding = candidate_stability_tools._load_execution_binding(
        fixture["runtime"],
        task_id=fixture["task"].id,
        request=request,
    )

    physical = captured["sample_design"]
    assert binding.resolved_requirements is resolved
    assert captured["task_id"] == fixture["task"].id
    assert captured["compiled_design"]["requirements"] == [requirement]
    assert physical.bundle_artifact_id == fixture["sample_ref"]["artifact_id"]
    assert physical.bundle_artifact_content_hash == fixture["sample_ref"][
        "artifact_content_hash"
    ]
    assert physical.bundle["sample_design"]["content_hash"] == fixture[
        "sample_ref"
    ]["sample_design_content_hash"]
    assert physical.membership["header"]["counts"]["approval"]["development"] == 30
    assert physical.membership["header"]["counts"]["risk"]["development"] == 60


def test_cross_matrix_pool_entry_stability_uses_full_waterfall_first_match(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    matrix_output = strategy_tools.tool_build_cross_matrix_candidate(
        {
            "source_artifact_id": fixture["source_report"]["artifact_id"],
            "expected_artifact_content_hash": fixture["source_report"][
                "content_hash"
            ],
            "expected_candidate_id": fixture["source"]["candidate_id"],
            "expected_evidence_hash": fixture["source"]["evidence_hash"],
            "x_feature": "age",
            "x_method": "equal_width",
            "y_feature": "score",
            "y_method": "equal_width",
        },
        fixture["ctx"],
    )
    matrix = matrix_output["cross_matrix_candidate"]
    populated = [
        cell for cell in matrix["matrix"]["cells"] if cell["effect"]["count"] > 0
    ]
    selection = strategy_tools.tool_materialize_cross_matrix_cell_selection(
        {
            "source_artifact_id": matrix_output["artifacts"][0]["artifact_id"],
            "expected_artifact_content_hash": matrix_output["artifacts"][0][
                "content_hash"
            ],
            "expected_asset_id": matrix["asset_id"],
            "expected_asset_hash": matrix["asset_hash"],
            "expected_candidate_id": matrix["candidate_evidence"]["candidate_id"],
            "expected_evidence_hash": matrix["candidate_evidence"]["evidence_hash"],
            "cell_ids": [populated[0]["cell_id"]],
        },
        fixture["ctx"],
    )
    selection_artifact = selection["artifacts"][0]
    pool = run_add_candidate_to_pool(
        {
            "source_artifact_id": selection_artifact["artifact_id"],
            "expected_artifact_content_hash": selection_artifact["content_hash"],
            "expected_asset_id": selection["source_asset_id"],
            "expected_asset_hash": selection["source_asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject", reason="CROSS_RISK"),
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    [entry] = pool["entries"]

    output = run_measure_candidate_monthly_stability(
        resolve_candidate_monthly_stability_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_pointer={
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": entry["entry_id"],
            },
        ),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["basis"] == "pool_entry_incremental_first_match"
    assert output["source_kind"] == "pool_entry"
    assert output["stability"]["source_ref"]["entry_id"] == entry["entry_id"]
    assert sum(
        row["hit_count"] for row in output["stability"]["monthly"]
    ) == populated[0]["effect"]["count"]


def test_automatic_tree_pool_entry_stability_uses_governed_month_sample(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    tree = strategy_tools.tool_build_automatic_tree_candidate(
        {
            "dataset_id": fixture["dataset"].id,
            "expected_content_hash": fixture["dataset"].content_hash,
            "workspace_revision": fixture["workspace"].revision,
            "analysis_generation": fixture["workspace"].analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                fixture["mapping"]
            ),
            "target_col": "bad",
            "sample_design_ref": fixture["sample_ref"],
            "features": ["score", "age"],
            "directions": {"score": "increasing", "age": "increasing"},
            "max_depth": 2,
            "min_leaf_count": 5,
            "budgets": {
                "max_rows": 200,
                "max_features": 5,
                "max_cells": 1_000,
                "max_nodes": 31,
                "max_cutpoint_evaluations": 2_000,
            },
        },
        fixture["ctx"],
    )
    tree_artifact = next(
        artifact
        for artifact in tree["artifacts"]
        if artifact["kind"] == "strategy_automatic_tree_asset_json"
    )
    leaf = tree["leaf_index"][0]
    selection = strategy_tools.tool_materialize_automatic_tree_leaf_fragment(
        {
            "source_artifact_id": tree_artifact["artifact_id"],
            "expected_artifact_content_hash": tree_artifact["content_hash"],
            "expected_asset_id": tree["summary"]["asset_id"],
            "expected_asset_hash": tree["summary"]["asset_hash"],
            "expected_tree_result_hash": tree["summary"]["tree_result_hash"],
            "leaf_id": leaf["leaf_id"],
        },
        fixture["ctx"],
    )
    selection_artifact = selection["artifacts"][0]
    pool = run_add_candidate_to_pool(
        {
            "source_artifact_id": selection_artifact["artifact_id"],
            "expected_artifact_content_hash": selection_artifact["content_hash"],
            "expected_asset_id": selection["tree_asset_id"],
            "expected_asset_hash": selection["tree_asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject", reason="TREE_RISK"),
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    [entry] = pool["entries"]

    output = run_measure_candidate_monthly_stability(
        resolve_candidate_monthly_stability_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_pointer={
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": entry["entry_id"],
            },
        ),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["month_col"] == "month"
    assert sum(
        row["hit_count"] for row in output["stability"]["monthly"]
    ) == leaf["measurements"]["unweighted"]["total"]


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


def test_report_consumer_loads_and_revalidates_authenticated_stability(
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
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]

    binding = load_candidate_stability_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_stability_id=output["stability_id"],
        expected_stability_content_hash=output["content_hash"],
    )

    assert isinstance(binding, StrategyCandidateStabilityArtifactBinding)
    assert binding.task_id == fixture["task"].id
    assert binding.artifact_id == artifact["artifact_id"]
    assert binding.artifact_content_hash == artifact["content_hash"]
    assert binding.stability == output["stability"]
    assert binding.artifact_path == Path(
        fixture["settings"].tasks_dir,
        fixture["task"].id,
        "strategy_candidate_stability",
        f"{output['stability_id']}.json",
    )
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_candidate_stability_artifact_binding_on_connection(
            conn,
            binding,
        )
        conn.commit()


def test_report_consumer_rejects_cross_task_registry_lookup(
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
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    foreign_task = TaskRepository(fixture["settings"].db_path).create_task(
        TaskCreate(
            model_name="foreign-candidate-stability",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign-source"),
            task_type="strategy",
            target_col="bad",
        )
    )

    with pytest.raises(StrategyError, match="registry row is invalid"):
        load_candidate_stability_artifact(
            fixture["runtime"],
            task_id=foreign_task.id,
            artifact_id=artifact["artifact_id"],
            expected_artifact_content_hash=artifact["content_hash"],
            expected_stability_id=output["stability_id"],
            expected_stability_content_hash=output["content_hash"],
        )


@pytest.mark.parametrize("tamper", ["file_bytes", "provenance_encoding"])
def test_report_consumer_rejects_persisted_artifact_tamper(
    tmp_path: Path,
    tamper: str,
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
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        artifact["artifact_id"],
    )
    assert record is not None
    if tamper == "file_bytes":
        Path(record["path"]).write_bytes(b"{}")
        expected_error = "content hash drifted"
    else:
        with fixture["runtime"].task_artifacts.transaction() as conn:
            conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
            row = conn.execute(
                "SELECT provenance_json FROM task_artifacts WHERE id = ?",
                (artifact["artifact_id"],),
            ).fetchone()
            assert row is not None
            conn.execute(
                "UPDATE task_artifacts SET provenance_json = ? WHERE id = ?",
                (" " + str(row["provenance_json"]), artifact["artifact_id"]),
            )
            conn.commit()
        expected_error = "registry binding changed"

    with pytest.raises(StrategyError, match=expected_error):
        load_candidate_stability_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=artifact["artifact_id"],
            expected_artifact_content_hash=artifact["content_hash"],
            expected_stability_id=output["stability_id"],
            expected_stability_content_hash=output["content_hash"],
        )


@pytest.mark.parametrize(
    "invalid_json",
    ["duplicate_key", "non_finite", "invalid_utf8", "oversized"],
)
def test_report_consumer_rejects_non_strict_json(
    tmp_path: Path,
    invalid_json: str,
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
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        artifact["artifact_id"],
    )
    assert record is not None
    if invalid_json == "duplicate_key":
        raw = Path(record["path"]).read_text("utf-8")
        encoded = (
            '{"schema_version":'
            + json.dumps(output["stability"]["schema_version"])
            + ","
            + raw[1:]
        ).encode("utf-8")
        expected_error = "duplicate key"
    elif invalid_json == "non_finite":
        forged = json.loads(Path(record["path"]).read_text("utf-8"))
        forged["summary"]["max_psi"] = float("nan")
        encoded = json.dumps(
            forged,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_error = "non-finite"
    elif invalid_json == "invalid_utf8":
        encoded = b"\xff"
        expected_error = "JSON is invalid"
    else:
        encoded = b" " * (2 * 1024 * 1024)
        expected_error = "JSON byte budget"
    forged_artifact_hash = hashlib.sha256(encoded).hexdigest()
    Path(record["path"]).write_bytes(encoded)
    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
            (forged_artifact_hash, artifact["artifact_id"]),
        )
        conn.commit()

    with pytest.raises(StrategyError, match=expected_error):
        load_candidate_stability_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=artifact["artifact_id"],
            expected_artifact_content_hash=forged_artifact_hash,
            expected_stability_id=output["stability_id"],
            expected_stability_content_hash=output["content_hash"],
        )


def test_report_consumer_rejects_symlinked_canonical_artifact(
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
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        artifact["artifact_id"],
    )
    assert record is not None
    artifact_path = Path(record["path"])
    retained = tmp_path / "retained-candidate-stability.json"
    artifact_path.rename(retained)
    artifact_path.symlink_to(retained)

    with pytest.raises(StrategyError, match="regular file"):
        load_candidate_stability_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=artifact["artifact_id"],
            expected_artifact_content_hash=artifact["content_hash"],
            expected_stability_id=output["stability_id"],
            expected_stability_content_hash=output["content_hash"],
        )


def test_report_consumer_binding_enforces_transaction_database_and_row_cas(
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
    output = run_measure_candidate_monthly_stability(
        resolved,
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    binding = load_candidate_stability_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_stability_id=output["stability_id"],
        expected_stability_content_hash=output["content_hash"],
    )

    with fixture["runtime"].task_artifacts.transaction() as conn:
        with pytest.raises(StrategyError, match="caller-owned transaction"):
            require_candidate_stability_artifact_binding_on_connection(
                conn,
                binding,
            )

    foreign_settings = build_settings(tmp_path / "foreign-workspace")
    init_db(foreign_settings.db_path)
    with TaskArtifactRepository(foreign_settings.db_path).transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="database changed"):
            require_candidate_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()

    with fixture["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET origin_tool = ? WHERE id = ?",
            ("forged.origin", binding.artifact_id),
        )
        with pytest.raises(StrategyError, match="registry binding changed"):
            require_candidate_stability_artifact_binding_on_connection(
                conn,
                binding,
            )
        conn.rollback()

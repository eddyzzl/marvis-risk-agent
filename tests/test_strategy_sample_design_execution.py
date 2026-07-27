from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3

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
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_execution import (
    bind_strategy_risk_development_frame,
    load_historical_strategy_risk_development_execution_binding,
    load_strategy_risk_development_execution_binding,
    require_historical_strategy_risk_development_execution_binding_on_connection,
    require_strategy_risk_development_execution_binding_on_connection,
    revalidate_historical_strategy_risk_development_execution_binding,
    revalidate_strategy_risk_development_execution_binding,
)
from marvis.packs.strategy.sample_design_binding import (
    bind_strategy_development_frame,
    load_strategy_sample_design_execution_binding,
    sample_design_ref_hash,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
    run_materialize_sample_design,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    run_materialize_sample_design_v2_native,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.packs.strategy.sample_membership import (
    decode_sample_membership,
    encode_sample_membership,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _parallel_native_setup(tmp_path: Path) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="native-execution",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "row_id": [
                "approval-only",
                "risk-only",
                "both",
                "neither",
                "valid-both",
                "oot-both",
            ],
            "sample_split": ["dev", "dev", "dev", "dev", "valid", "oot"],
            "approval_flag": [1, 0, 1, 0, 1, 1],
            "risk_flag": [0, 1, 1, 0, 1, 1],
            "apply_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-02-01",
                "2026-03-01",
            ],
            "apply_month": [
                "202601",
                "202601",
                "202601",
                "202601",
                "202602",
                "202603",
            ],
            "customer_id": ["a", "b", "c", "d", "e", "f"],
            "channel": ["app", "web", "app", "web", "app", "web"],
            "bad": [1, 0, 1, 0, 1, 0],
            "feature": [10, 20, 30, 40, 50, 60],
        }
    )
    source = tmp_path / "parallel.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(
        source,
        task_id=task.id,
        role="derived",
    )
    workspaces = DataWorkspaceRepository(settings.db_path)
    active = workspaces.save(
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
            "row_id": "id",
            "sample_split": "segment",
            "approval_flag": "categorical",
            "risk_flag": "categorical",
            "apply_date": "date",
            "apply_month": "month",
            "customer_id": "id",
            "channel": "categorical",
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
        expected_revision=active.revision,
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    runtime = strategy_tools._runtime(ctx)
    request = {
        "source_mode": "native_active_dataset",
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "target_bad_value": 0,
        "drop_nan_labels": True,
        "relationship": "parallel_time_cohorts",
        "scope": "strategy_development",
        "approval_population": {
            "inclusion": _eq("approval_flag", 1),
            "exclusion": None,
        },
        "risk_population": {
            "inclusion": _eq("risk_flag", 1),
            "exclusion": None,
        },
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_split", "dev"),
                "validation": _eq("sample_split", "valid"),
                "oot": _eq("sample_split", "oot"),
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-04-30",
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-04-30",
        },
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": "channel",
            "month_field": "apply_month",
            "weight_field": None,
            "loan_amount_field": None,
            "overdue_amount_field": None,
        },
        "historical_score": {
            "status": "not_applicable",
            "column": None,
            "direction": None,
            "reason": "not supplied",
        },
        "policy": {
            "minimum_partition_count": 1,
            "minimum_bad_count": 0,
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
                "sufficiency": "warn",
            },
        },
    }
    output = run_materialize_sample_design_v2_native(
        request,
        ctx,
        runtime,
    )
    records = [
        item
        for item in TaskArtifactRepository(settings.db_path).list_for_task(
            task.id
        )
        if item["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
    ]
    membership = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    )
    sample_ref = {
        "artifact_id": bundle["id"],
        "artifact_content_hash": bundle["content_hash"],
        "sample_design_id": output["sample_design_id"],
        "sample_design_content_hash": output["sample_design_content_hash"],
        "partition": "risk/development",
    }
    return {
        "settings": settings,
        "task": task,
        "dataset": dataset,
        "workspace": workspace,
        "mapping": mapping,
        "runtime": runtime,
        "frame": frame,
        "output": output,
        "membership": membership,
        "bundle": bundle,
        "sample_ref": sample_ref,
    }


def _load_execution(fx: dict, *, historical: bool = False):
    loader = (
        load_historical_strategy_risk_development_execution_binding
        if historical
        else load_strategy_risk_development_execution_binding
    )
    return loader(
        fx["runtime"],
        task_id=fx["task"].id,
        sample_design_ref=fx["sample_ref"],
        dataset_id=fx["dataset"].id,
        dataset_content_hash=fx["dataset"].content_hash,
        workspace_revision=fx["workspace"].revision,
        workspace_generation=fx["workspace"].analysis_generation,
        semantic_mapping_hash=data_semantic_mapping_hash(fx["mapping"]),
        target_col="bad",
        drop_nan_labels=True,
        month_col="apply_month",
    )


def _materialize_legacy_ref(fx: dict) -> dict[str, str]:
    output = run_materialize_sample_design(
        {
            "dataset_id": fx["dataset"].id,
            "expected_dataset_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "workspace_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(fx["mapping"]),
            "target_col": "bad",
            "target_bad_value": 0,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-04-30",
            "maturity_status": "confirmed_matured",
            "split_col": "sample_split",
            "development_values": ["dev"],
            "validation_values": ["valid"],
            "oot_values": ["oot"],
            "month_col": "apply_month",
            "drop_nan_labels": True,
        },
        ToolContext(
            task_id=fx["task"].id,
            seed=0,
            datasets_root=fx["settings"].datasets_dir,
            workspace=fx["settings"].workspace,
        ),
        fx["runtime"],
    )
    return {
        "artifact_id": output["artifact"]["artifact_id"],
        "artifact_content_hash": output["artifact"]["content_hash"],
        "sample_design_id": output["sample_design_id"],
        "sample_design_content_hash": output["content_hash"],
        "partition": "development",
    }


def test_native_execution_selects_persisted_risk_development_in_source_order(
    tmp_path: Path,
) -> None:
    fx = _parallel_native_setup(tmp_path)

    assert fx["output"]["membership"]["counts"]["approval"]["development"] == 2
    assert fx["output"]["membership"]["counts"]["risk"]["development"] == 2
    binding = _load_execution(fx)
    selected = bind_strategy_risk_development_frame(
        fx["frame"],
        binding=binding,
    )

    assert binding.source_mode == "native_active_dataset"
    assert binding.to_ref_dict() == fx["sample_ref"]
    assert binding.partition_columns == ("sample_split",)
    assert binding.population_filter_columns == (
        "approval_flag",
        "risk_flag",
    )
    assert binding.excluded_feature_columns == (
        "approval_flag",
        "bad",
        "risk_flag",
        "sample_split",
    )
    assert selected["row_id"].tolist() == ["risk-only", "both"]
    assert selected["bad"].tolist() == [1, 0]
    assert revalidate_strategy_risk_development_execution_binding(
        fx["runtime"],
        binding,
    ).to_ref_dict() == fx["sample_ref"]


def test_historical_native_execution_replays_after_workspace_switch(
    tmp_path: Path,
) -> None:
    fx = _parallel_native_setup(tmp_path)
    replacement_path = tmp_path / "replacement.parquet"
    fx["frame"].iloc[:3].to_parquet(replacement_path, index=False)
    replacement = fx["runtime"].registry.register_existing(
        replacement_path,
        task_id=fx["task"].id,
        role="derived",
    )
    workspaces = DataWorkspaceRepository(fx["settings"].db_path)
    workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=replacement.id,
            active_dataset_content_hash=replacement.content_hash,
            semantic_mapping=DataSemanticMapping(),
        ),
        expected_revision=fx["workspace"].revision,
    )

    with pytest.raises(StrategyError, match="DataWorkspace binding changed"):
        _load_execution(fx)
    binding = _load_execution(fx, historical=True)
    selected = bind_strategy_risk_development_frame(
        fx["frame"],
        binding=binding,
    )

    assert binding.to_ref_dict() == fx["sample_ref"]
    assert selected["row_id"].tolist() == ["risk-only", "both"]
    assert selected["bad"].tolist() == [1, 0]
    assert revalidate_historical_strategy_risk_development_execution_binding(
        fx["runtime"],
        binding,
    ).to_ref_dict() == fx["sample_ref"]


def test_native_execution_current_and_historical_writer_lock_revalidation(
    tmp_path: Path,
) -> None:
    fx = _parallel_native_setup(tmp_path)
    current = _load_execution(fx)
    historical = _load_execution(fx, historical=True)
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_risk_development_execution_binding_on_connection(
            conn,
            current,
        )
        require_historical_strategy_risk_development_execution_binding_on_connection(
            conn,
            historical,
        )

    with sqlite3.connect(fx["settings"].db_path) as conn:
        conn.execute(
            """
            UPDATE data_workspaces
               SET revision = revision + 1
             WHERE task_id = ?
            """,
            (fx["task"].id,),
        )
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="DataWorkspace binding changed"):
            require_strategy_risk_development_execution_binding_on_connection(
                conn,
                current,
            )
        require_historical_strategy_risk_development_execution_binding_on_connection(
            conn,
            historical,
        )

    membership_path = Path(fx["membership"]["path"])
    membership = decode_sample_membership(membership_path.read_bytes())
    membership["masks"]["risk/development"] = membership["masks"][
        "approval/development"
    ].copy()
    membership_path.write_bytes(
        encode_sample_membership(
            task_id=membership["header"]["task_id"],
            dataset_id=membership["header"]["dataset_ref"]["dataset_id"],
            dataset_content_hash=membership["header"]["dataset_ref"][
                "content_hash"
            ],
            masks=membership["masks"],
        )
    )
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="artifact"):
            require_historical_strategy_risk_development_execution_binding_on_connection(
                conn,
                historical,
            )


@pytest.mark.parametrize(
    "case",
    [
        "legacy_partition_on_native_bundle",
        "native_membership_as_bundle_ref",
        "wrong_bundle_kind",
        "wrong_bundle_origin",
    ],
)
def test_native_execution_rejects_crossed_ref_kind_origin_and_partition(
    tmp_path: Path,
    case: str,
) -> None:
    fx = _parallel_native_setup(tmp_path / case)
    ref = deepcopy(fx["sample_ref"])
    if case == "legacy_partition_on_native_bundle":
        ref["partition"] = "development"
    elif case == "native_membership_as_bundle_ref":
        ref["artifact_id"] = fx["membership"]["id"]
        ref["artifact_content_hash"] = fx["membership"]["content_hash"]
    else:
        forged_path = tmp_path / case / f"{case}.json"
        forged_path.write_bytes(Path(fx["bundle"]["path"]).read_bytes())
        forged = fx["runtime"].task_artifacts.register(
            task_id=fx["task"].id,
            kind=(
                SAMPLE_DESIGN_ARTIFACT_KIND
                if case == "wrong_bundle_kind"
                else SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
            ),
            path=str(forged_path),
            content_hash=fx["bundle"]["content_hash"],
            origin_tool=(
                SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
                if case == "wrong_bundle_kind"
                else SAMPLE_DESIGN_ORIGIN_TOOL
            ),
            provenance=fx["bundle"]["provenance"],
        )
        ref["artifact_id"] = forged["id"]
        ref["artifact_content_hash"] = forged["content_hash"]

    with pytest.raises(StrategyError, match="sample_design_ref"):
        load_strategy_risk_development_execution_binding(
            fx["runtime"],
            task_id=fx["task"].id,
            sample_design_ref=ref,
            dataset_id=fx["dataset"].id,
            dataset_content_hash=fx["dataset"].content_hash,
            workspace_revision=fx["workspace"].revision,
            workspace_generation=fx["workspace"].analysis_generation,
            semantic_mapping_hash=data_semantic_mapping_hash(fx["mapping"]),
            target_col="bad",
            drop_nan_labels=True,
            month_col="apply_month",
        )


def test_legacy_execution_ref_token_and_frame_identity_are_unchanged(
    tmp_path: Path,
) -> None:
    fx = _parallel_native_setup(tmp_path)
    legacy_ref = _materialize_legacy_ref(fx)
    kwargs = {
        "task_id": fx["task"].id,
        "sample_design_ref": legacy_ref,
        "dataset_id": fx["dataset"].id,
        "dataset_content_hash": fx["dataset"].content_hash,
        "workspace_revision": fx["workspace"].revision,
        "workspace_generation": fx["workspace"].analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(fx["mapping"]),
        "target_col": "bad",
        "drop_nan_labels": True,
        "month_col": "apply_month",
    }
    legacy = load_strategy_sample_design_execution_binding(
        fx["runtime"],
        **kwargs,
    )
    generic = load_strategy_risk_development_execution_binding(
        fx["runtime"],
        **kwargs,
    )

    assert generic.source_mode == "legacy_anchored"
    assert generic.to_ref_dict() == legacy.to_ref_dict() == legacy_ref
    assert generic.source_ref == legacy.source_ref
    assert generic.source_ref_token == legacy.source_ref_token
    assert sample_design_ref_hash(generic.to_ref_dict()) == (
        sample_design_ref_hash(legacy.to_ref_dict())
    )
    assert bind_strategy_risk_development_frame(
        fx["frame"],
        binding=generic,
    ).equals(
        bind_strategy_development_frame(
            fx["frame"],
            binding=legacy,
        )
    )


@pytest.mark.parametrize(
    "drift",
    ["registry_path", "registry_metadata", "dataset_bytes"],
)
def test_historical_native_execution_rejects_dataset_source_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    fx = _parallel_native_setup(tmp_path / drift)
    historical = _load_execution(fx, historical=True)
    if drift == "dataset_bytes":
        dataset_path = Path(
            fx["runtime"].registry.resolve_path(fx["dataset"].id)
        )
        dataset_path.write_bytes(dataset_path.read_bytes() + b"drift")
    else:
        with sqlite3.connect(fx["settings"].db_path) as conn:
            if drift == "registry_path":
                conn.execute(
                    "UPDATE datasets SET source_path = ? WHERE id = ?",
                    (str(tmp_path / "moved.parquet"), fx["dataset"].id),
                )
            else:
                conn.execute(
                    "UPDATE datasets SET row_count = row_count + 1 "
                    "WHERE id = ?",
                    (fx["dataset"].id,),
                )

    with pytest.raises(StrategyError, match="historical dataset"):
        _load_execution(fx, historical=True)
    with pytest.raises(StrategyError, match="historical dataset"):
        revalidate_historical_strategy_risk_development_execution_binding(
            fx["runtime"],
            historical,
        )

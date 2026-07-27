from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

import marvis.data.registry as registry_module
from marvis.artifacts import ArtifactUnitOfWork
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
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
    SAMPLE_DESIGN_V2_NATIVE_TOOL_SCHEMA_VERSION,
    load_native_strategy_sample_design_v2_artifacts,
    run_materialize_sample_design_v2_native,
    validate_materialize_sample_design_v2_native_tool_output,
)
import marvis.packs.strategy.sample_design_v2_native_tools as native_tools
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    load_any_strategy_sample_design_v2_artifacts,
    require_any_strategy_sample_design_v2_artifact_binding_on_connection,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.task_artifacts import TaskArtifactDataError
from marvis.settings import build_settings


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _setup_native(tmp_path: Path, *, target_bad_value: int = 1) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="sample-v2-native",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "sample_split": ["dev", "dev", "valid", "valid", "oot", "oot"],
            "apply_date": [
                "2026-01-01",
                "2026-01-10",
                "2026-02-01",
                "2026-02-10",
                "2026-03-01",
                "2026-03-10",
            ],
            "apply_month": [
                "202601",
                "202601",
                "202602",
                "202602",
                "202603",
                "202603",
            ],
            "customer_id": ["a", "b", "c", "d", "e", "f"],
            "channel": ["app", "web", "app", "web", "app", "web"],
            "legacy_score": [100.0, 200.0, 120.0, None, 140.0, 240.0],
            "weight": [1.0] * 6,
            "loan_amount": [100.0, 200.0, 150.0, 180.0, 300.0, 250.0],
            "overdue_amount": [0.0, 20.0, 0.0, 10.0, 0.0, 30.0],
            "bad": [0, 1, 0, 1, None, 1],
            "unused_feature": [10, 20, 30, 40, 50, 60],
        }
    )
    if target_bad_value == 0:
        frame["bad"] = frame["bad"].map({0.0: 1, 1.0: 0})
    source = tmp_path / "sample.parquet"
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
            "sample_split": "segment",
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
        expected_revision=activated.revision,
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
        "target_bad_value": target_bad_value,
        "drop_nan_labels": True,
        "relationship": "nested_same_cohort",
        "scope": "strategy_development",
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
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
            "weight_field": "weight",
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "historical_score": {
            "status": "available",
            "column": "legacy_score",
            "direction": "higher_is_riskier",
            "reason": None,
        },
        "policy": {
            "minimum_partition_count": 1,
            "minimum_bad_count": 1,
            "minimum_label_coverage": 0.8,
            "minimum_historical_score_coverage": 0.8,
            "maximum_group_coverage_gap": 0.2,
            "diagnostic_severities": {
                "entity_overlap": "fail",
                "temporal_oot": "fail",
                "risk_outside_approval": "fail",
                "maturity": "fail",
                "label_coverage": "fail",
                "historical_score_coverage": "warn",
                "group_coverage_gap": "warn",
                "sufficiency": "fail",
            },
        },
    }
    return {
        "settings": settings,
        "task": task,
        "dataset": dataset,
        "workspace": workspace,
        "ctx": ctx,
        "runtime": runtime,
        "request": request,
    }


def _native_artifacts(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
    ]


def test_native_v2_materializes_replays_and_loads_without_legacy_anchor(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)

    first = run_materialize_sample_design_v2_native(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    second = run_materialize_sample_design_v2_native(
        deepcopy(fx["request"]), fx["ctx"], fx["runtime"]
    )

    assert first == second
    assert first["schema_version"] == SAMPLE_DESIGN_V2_NATIVE_TOOL_SCHEMA_VERSION
    assert (
        validate_materialize_sample_design_v2_native_tool_output(first)
        == first
    )
    assert first["bundle"]["sample_design"]["compatibility"] == {
        "source_mode": "native_active_dataset",
        "development_partition": "risk/development",
    }
    assert "legacy_mapping" not in first
    assert first["source_binding"]["source_mode"] == "native_active_dataset"
    assert first["source_binding"]["target_selector"] == {
        "column": "bad",
        "bad_value": 1,
        "drop_missing": True,
    }
    records = _native_artifacts(fx)
    assert {item["kind"] for item in records} == {
        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    }
    assert all(
        "legacy_sample_design_ref" not in item["provenance"]
        for item in records
    )
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
    loaded = load_native_strategy_sample_design_v2_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        membership_artifact_id=membership["id"],
        expected_membership_artifact_content_hash=membership["content_hash"],
        bundle_artifact_id=bundle["id"],
        expected_bundle_artifact_content_hash=bundle["content_hash"],
        expected_bundle_id=first["bundle_id"],
        expected_sample_design_id=first["sample_design_id"],
        expected_sample_design_content_hash=first[
            "sample_design_content_hash"
        ],
    )
    assert loaded.bundle == first["bundle"]
    assert loaded.membership["header"] == first["membership"]
    assert loaded.source_binding.target_col == "bad"
    any_loaded = load_any_strategy_sample_design_v2_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        membership_artifact_id=membership["id"],
        expected_membership_artifact_content_hash=membership[
            "content_hash"
        ],
        bundle_artifact_id=bundle["id"],
        expected_bundle_artifact_content_hash=bundle["content_hash"],
        expected_bundle_id=first["bundle_id"],
        expected_sample_design_id=first["sample_design_id"],
        expected_sample_design_content_hash=first[
            "sample_design_content_hash"
        ],
    )
    assert any_loaded.membership_artifact_id == loaded.membership_artifact_id
    assert any_loaded.bundle == loaded.bundle
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_any_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            any_loaded,
        )


def test_native_v2_supports_independent_filters_for_both_relationships(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    request = deepcopy(fx["request"])
    request["approval_population"]["exclusion"] = _eq(
        "customer_id",
        "a",
    )

    with pytest.raises(StrategyError, match="nested_same_cohort"):
        run_materialize_sample_design_v2_native(
            request,
            fx["ctx"],
            fx["runtime"],
        )

    request["relationship"] = "parallel_time_cohorts"
    output = run_materialize_sample_design_v2_native(
        request,
        fx["ctx"],
        fx["runtime"],
    )
    counts = output["membership"]["counts"]
    assert counts["approval"]["development"] == 1
    assert counts["risk"]["development"] == 2
    statuses = {
        item["code"]: item["status"]
        for item in output["bundle"]["diagnostics"]
    }
    assert statuses["risk_outside_approval"] == "not_applicable"


def test_native_v2_membership_registry_identity_includes_target_binding(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    risk_bad_one = run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    inverted = deepcopy(fx["request"])
    inverted["target_bad_value"] = 0

    risk_bad_zero = run_materialize_sample_design_v2_native(
        inverted,
        fx["ctx"],
        fx["runtime"],
    )

    assert risk_bad_zero["membership_id"] == risk_bad_one["membership_id"]
    assert (
        risk_bad_zero["membership_content_hash"]
        == risk_bad_one["membership_content_hash"]
    )
    membership_filenames = {
        risk_bad_one["artifacts"]["membership"]["filename"],
        risk_bad_zero["artifacts"]["membership"]["filename"],
    }
    assert len(membership_filenames) == 2
    assert all(
        filename.startswith(f"{risk_bad_one['membership_id']}-")
        and filename.endswith(".bin")
        for filename in membership_filenames
    )
    memberships = [
        item
        for item in _native_artifacts(fx)
        if item["kind"] == SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
    ]
    assert len(memberships) == 2
    assert {item["provenance"]["target_bad_value"] for item in memberships} == {
        0,
        1,
    }


def test_native_v2_concurrent_distinct_target_bindings_do_not_collide(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    bad_one = deepcopy(fx["request"])
    bad_zero = deepcopy(fx["request"])
    bad_zero["target_bad_value"] = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_materialize_sample_design_v2_native,
                request,
                fx["ctx"],
                fx["runtime"],
            )
            for request in (bad_one, bad_zero)
        ]
        outputs = [future.result() for future in futures]

    assert outputs[0]["membership_id"] == outputs[1]["membership_id"]
    assert (
        outputs[0]["artifacts"]["membership"]["filename"]
        != outputs[1]["artifacts"]["membership"]["filename"]
    )
    records = _native_artifacts(fx)
    assert [
        item["kind"] for item in records
    ].count(SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND) == 2
    assert [
        item["kind"] for item in records
    ].count(SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND) == 2


def test_native_v2_concurrent_exact_replays_share_one_registry_pair(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_materialize_sample_design_v2_native,
                deepcopy(fx["request"]),
                fx["ctx"],
                fx["runtime"],
            )
            for _ in range(2)
        ]
        outputs = [future.result() for future in futures]

    assert outputs[0] == outputs[1]
    records = _native_artifacts(fx)
    assert [
        item["kind"] for item in records
    ].count(SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND) == 1
    assert [
        item["kind"] for item in records
    ].count(SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND) == 1


def test_native_v2_membership_filename_binds_full_registry_identity(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    output = run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    membership = next(
        item
        for item in _native_artifacts(fx)
        if item["kind"] == SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
    )

    identity_hash = (
        native_tools.native_sample_design_v2_membership_registry_identity_hash(
            membership["provenance"]
        )
    )
    assert output["source_binding"][
        "membership_registry_identity_hash"
    ] == identity_hash
    assert output["artifacts"]["membership"]["filename"] == (
        f"{output['membership_id']}-{identity_hash[:24]}.bin"
    )

    source = {
        key: membership["provenance"][key]
        for key in native_tools._SOURCE_PROVENANCE_FIELDS
    }
    bumped_schema = {
        **source,
        "schema_version": source["schema_version"] + ".next",
    }
    bumped_producer = {
        **source,
        "producer_version": source["producer_version"] + ".next",
    }
    assert len(
        {
            identity_hash,
            native_tools.native_sample_design_v2_membership_registry_identity_hash(
                bumped_schema
            ),
            native_tools.native_sample_design_v2_membership_registry_identity_hash(
                bumped_producer
            ),
        }
    ) == 3


def test_native_v2_output_recomputes_full_registry_identity_hash(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    output = run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    forged = deepcopy(output)
    identity_hash = forged["source_binding"][
        "membership_registry_identity_hash"
    ]
    forged["source_binding"]["membership_registry_identity_hash"] = (
        identity_hash[:24] + ("0" if identity_hash[24] != "0" else "1") * 40
    )
    body = {
        key: value
        for key, value in forged.items()
        if key != "content_hash"
    }
    forged["content_hash"] = hashlib.sha256(
        native_tools.common._canonical_json(body).encode("utf-8")
    ).hexdigest()

    with pytest.raises(StrategyError, match="registry identity"):
        validate_materialize_sample_design_v2_native_tool_output(forged)


@pytest.mark.parametrize(
    ("budget_name", "budget_value", "message"),
    [
        ("MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_ROWS", 5, "source_rows"),
        ("MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_COLUMNS", 1, "required_columns"),
        ("MAX_NATIVE_SAMPLE_DESIGN_V2_REQUIRED_CELLS", 5, "required_cells"),
        ("MAX_NATIVE_SAMPLE_DESIGN_V2_SOURCE_BYTES", 1, "source_bytes"),
    ],
)
def test_native_v2_resource_preflight_fails_before_frame_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_name: str,
    budget_value: int,
    message: str,
) -> None:
    fx = _setup_native(tmp_path)
    monkeypatch.setattr(native_tools, budget_name, budget_value)
    calls = 0

    def forbidden_read_frame(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("frame read must happen after native preflight")

    monkeypatch.setattr(
        fx["runtime"].backend,
        "read_frame",
        forbidden_read_frame,
    )

    with pytest.raises(StrategyError, match=message):
        run_materialize_sample_design_v2_native(
            fx["request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert calls == 0


def test_native_v2_reads_only_columns_required_by_the_governed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup_native(tmp_path)
    original = fx["runtime"].backend.read_frame
    projected: list[tuple[str, ...]] = []

    def capture_projection(path, *, columns=None, nrows=None):
        projected.append(tuple(columns or ()))
        return original(path, columns=columns, nrows=nrows)

    monkeypatch.setattr(
        fx["runtime"].backend,
        "read_frame",
        capture_projection,
    )
    run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )

    assert projected == [
        (
            "apply_date",
            "apply_month",
            "bad",
            "channel",
            "customer_id",
            "legacy_score",
            "loan_amount",
            "overdue_amount",
            "sample_split",
            "weight",
        )
    ]
    assert "unused_feature" not in projected[0]


def test_native_v2_execution_hashes_the_full_source_at_most_three_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup_native(tmp_path)
    original = native_tools.sha256_file
    calls: list[Path] = []

    def counted(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(native_tools, "sha256_file", counted)
    monkeypatch.setattr(registry_module, "sha256_file", counted)

    run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )

    dataset_path = Path(
        fx["runtime"].registry.resolve_path(fx["dataset"].id)
    )
    assert calls == [dataset_path, dataset_path, dataset_path]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request: request["approval_population"].update(
            {"inclusion": _eq("bad", 1)}
        ),
        lambda request: request["risk_population"].update(
            {"exclusion": _eq("bad", 0)}
        ),
        lambda request: request["partitioning"]["selectors"].update(
            {"development": _eq("bad", 0)}
        ),
        *[
            lambda request, field=field: request["field_bindings"].update(
                {field: "bad"}
            )
            for field in (
                "entity_field",
                "time_field",
                "group_field",
                "month_field",
                "weight_field",
                "loan_amount_field",
                "overdue_amount_field",
            )
        ],
        lambda request: request["historical_score"].update(
            {
                "status": "available",
                "column": "bad",
                "direction": "higher_is_riskier",
                "reason": None,
            }
        ),
    ],
)
def test_native_v2_rejects_target_leakage_before_artifact_write(
    tmp_path: Path,
    mutate,
) -> None:
    fx = _setup_native(tmp_path)
    request = deepcopy(fx["request"])
    mutate(request)

    with pytest.raises(StrategyError, match="target column"):
        run_materialize_sample_design_v2_native(
            request,
            fx["ctx"],
            fx["runtime"],
        )
    assert _native_artifacts(fx) == []


def test_native_v2_rejects_target_as_time_range_partition_column(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    request = deepcopy(fx["request"])
    request["field_bindings"]["time_field"] = "bad"
    request["partitioning"] = {
        "method": "time_ranges",
        "column": "bad",
        "ranges": {
            "development": {"start": None, "end": "2026-01-31"},
            "validation": {
                "start": "2026-02-01",
                "end": "2026-02-28",
            },
            "oot": {"start": "2026-03-01", "end": None},
        },
    }

    with pytest.raises(StrategyError, match="target column"):
        run_materialize_sample_design_v2_native(
            request,
            fx["ctx"],
            fx["runtime"],
        )
    assert _native_artifacts(fx) == []


def test_native_v2_requires_time_range_column_to_match_bound_time_field(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    request = deepcopy(fx["request"])
    request["field_bindings"]["time_field"] = "apply_month"
    request["partitioning"] = {
        "method": "time_ranges",
        "column": "apply_date",
        "ranges": {
            "development": {"start": None, "end": "2026-01-31"},
            "validation": {
                "start": "2026-02-01",
                "end": "2026-02-28",
            },
            "oot": {"start": "2026-03-01", "end": None},
        },
    }

    with pytest.raises(
        StrategyError,
        match="time_ranges column must equal field_bindings.time_field",
    ):
        run_materialize_sample_design_v2_native(
            request,
            fx["ctx"],
            fx["runtime"],
        )
    assert _native_artifacts(fx) == []


def test_native_v2_loader_rejects_workspace_drift_after_replay(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    output = run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    records = _native_artifacts(fx)
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
    with sqlite3.connect(fx["settings"].db_path) as conn:
        conn.execute(
            """
            UPDATE data_workspaces
               SET revision = revision + 1
             WHERE task_id = ?
            """,
            (fx["task"].id,),
        )

    with pytest.raises(StrategyError, match="DataWorkspace binding changed"):
        load_native_strategy_sample_design_v2_artifacts(
            fx["runtime"],
            task_id=fx["task"].id,
            membership_artifact_id=membership["id"],
            expected_membership_artifact_content_hash=membership[
                "content_hash"
            ],
            bundle_artifact_id=bundle["id"],
            expected_bundle_artifact_content_hash=bundle["content_hash"],
            expected_bundle_id=output["bundle_id"],
            expected_sample_design_id=output["sample_design_id"],
            expected_sample_design_content_hash=output[
                "sample_design_content_hash"
            ],
        )


def test_native_v2_bundle_record_authentication_is_independent_of_workspace_head(
    tmp_path: Path,
) -> None:
    fx = _setup_native(tmp_path)
    output = run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    bundle_record = next(
        item
        for item in _native_artifacts(fx)
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
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

    authenticated = (
        native_tools.authenticate_native_strategy_sample_design_v2_bundle_record(
            fx["runtime"],
            task_id=fx["task"].id,
            record=bundle_record,
        )
    )

    assert authenticated.bundle["bundle_id"] == output["bundle_id"]
    assert authenticated.source_provenance["dataset_id"] == fx["dataset"].id
    assert authenticated.source_provenance["workspace_revision"] == (
        fx["workspace"].revision
    )


def test_native_v2_rechecks_workspace_under_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup_native(tmp_path)
    original = native_tools._persist_native_pair

    def drift_before_persist(*args, **kwargs):
        with sqlite3.connect(fx["settings"].db_path) as conn:
            conn.execute(
                """
                UPDATE data_workspaces
                   SET revision = revision + 1
                 WHERE task_id = ?
                """,
                (fx["task"].id,),
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        native_tools,
        "_persist_native_pair",
        drift_before_persist,
    )
    with pytest.raises(StrategyError, match="DataWorkspace binding changed"):
        run_materialize_sample_design_v2_native(
            fx["request"],
            fx["ctx"],
            fx["runtime"],
        )
    assert _native_artifacts(fx) == []


def test_native_v2_registration_failure_rolls_back_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup_native(tmp_path)
    original = fx["runtime"].task_artifacts.register_on_connection
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TaskArtifactDataError("injected native registration failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "register_on_connection",
        fail_second,
    )
    with pytest.raises(StrategyError, match="injected native registration failure"):
        run_materialize_sample_design_v2_native(
            fx["request"],
            fx["ctx"],
            fx["runtime"],
        )

    assert _native_artifacts(fx) == []
    out_dir = (
        fx["settings"].tasks_dir
        / fx["task"].id
        / "strategy_sample_designs_v2"
    )
    assert not list(out_dir.glob("*.bin"))
    assert not list(out_dir.glob("*.json"))


def test_native_v2_post_commit_cleanup_failure_keeps_pair_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup_native(tmp_path)
    original_commit = ArtifactUnitOfWork.commit

    def fail_cleanup(_self):
        raise RuntimeError("injected native post-commit cleanup failure")

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", fail_cleanup)
    with pytest.raises(
        RuntimeError,
        match="injected native post-commit cleanup failure",
    ):
        run_materialize_sample_design_v2_native(
            fx["request"],
            fx["ctx"],
            fx["runtime"],
        )

    records = _native_artifacts(fx)
    assert {record["kind"] for record in records} == {
        SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    }
    assert all(Path(record["path"]).is_file() for record in records)

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", original_commit)
    replay = run_materialize_sample_design_v2_native(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    assert (
        validate_materialize_sample_design_v2_native_tool_output(replay)
        == replay
    )
    assert {record["id"] for record in _native_artifacts(fx)} == {
        record["id"] for record in records
    }

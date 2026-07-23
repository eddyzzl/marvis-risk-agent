from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_tools import (
    load_strategy_sample_design_artifact,
    run_materialize_sample_design,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_TOOL_SCHEMA_VERSION,
    load_strategy_sample_design_v2_artifacts,
    require_strategy_sample_design_v2_artifact_binding_on_connection,
    run_materialize_sample_design_v2,
    validate_materialize_sample_design_v2_tool_output,
)
import marvis.packs.strategy.sample_design_v2_tools as v2_tools
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


def _setup(tmp_path: Path, *, target_bad_value: int = 1) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="sample-v2",
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
            "apply_month": ["202601", "202601", "202602", "202602", "202603", "202603"],
            "customer_id": ["a", "b", "c", "d", "e", "f"],
            "channel": ["app", "web", "app", "web", "app", "web"],
            "legacy_score": [100.0, 200.0, 120.0, None, 140.0, 240.0],
            "weight": [1.0] * 6,
            "loan_amount": [100.0, 200.0, 150.0, 180.0, 300.0, 250.0],
            "overdue_amount": [0.0, 20.0, 0.0, 10.0, 0.0, 30.0],
            "bad": [0, 1, 0, 1, None, 1],
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
    legacy = run_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": v2_tools.data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": target_bad_value,
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
            "weight_col": "weight",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": True,
        },
        ctx,
        runtime,
    )
    legacy_ref = {
        "artifact_id": legacy["artifact"]["artifact_id"],
        "artifact_content_hash": legacy["artifact"]["content_hash"],
        "sample_design_id": legacy["sample_design_id"],
        "sample_design_content_hash": legacy["content_hash"],
        "partition": "development",
    }
    request = {
        "legacy_sample_design_ref": legacy_ref,
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


def _v2_artifacts(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(fx["task"].id)
        if item["kind"] in {
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        }
    ]


def _live_v2_artifact(fx: dict, kind: str) -> dict:
    records = [item for item in _v2_artifacts(fx) if item["kind"] == kind]
    assert len(records) == 1
    return records[0]


def _load_output(fx: dict, output: dict):
    membership = _live_v2_artifact(
        fx, SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle = _live_v2_artifact(fx, SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND)
    return load_strategy_sample_design_v2_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        membership_artifact_id=membership["id"],
        expected_membership_artifact_content_hash=membership["content_hash"],
        bundle_artifact_id=bundle["id"],
        expected_bundle_artifact_content_hash=bundle["content_hash"],
        expected_bundle_id=output["bundle_id"],
        expected_sample_design_id=output["sample_design_id"],
        expected_sample_design_content_hash=output["sample_design_content_hash"],
    )


@pytest.mark.parametrize("target_bad_value", [0, 1])
def test_materialize_v2_is_idempotent_strict_and_verified(
    tmp_path: Path, target_bad_value: int
) -> None:
    fx = _setup(tmp_path, target_bad_value=target_bad_value)

    first = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    second = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])

    assert first == second
    assert first["schema_version"] == SAMPLE_DESIGN_V2_TOOL_SCHEMA_VERSION
    assert first["schema_version"] == "strategy.materialize-sample-design-v2-tool.v2"
    assert validate_materialize_sample_design_v2_tool_output(first) == first
    assert set(first["artifacts"]["membership"]) == {"kind", "format", "filename"}
    assert set(first["artifacts"]["bundle"]) == {
        "kind",
        "format",
        "filename",
        "content_hash",
    }
    assert len(first["bundle"]["metric_observations"]) == 40
    definitions = {
        item["metric_definition_id"]: item["metric_key"]
        for item in first["bundle"]["metric_definitions"]
    }
    assert {
        item["status"]
        for item in first["bundle"]["metric_observations"]
        if item["population"] == "approval"
        and definitions[item["metric_definition_ref"]["metric_definition_id"]]
        != "population_count"
    } == {"not_applicable"}
    assert len(_v2_artifacts(fx)) == 2
    loaded = _load_output(fx, first)
    assert loaded.bundle == first["bundle"]
    assert loaded.membership["header"] == first["membership"]
    assert fx["runtime"].strategies.list_for_task(fx["task"].id) == []


def test_v2_cached_artifact_summaries_reject_unbound_identity_fields_and_swaps(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    assert len(output["content_hash"]) == 64

    for role in ("membership", "bundle"):
        for field, forged in (
            ("artifact_id", "f" * 64),
            ("download_url", "/api/tasks/forged/task-artifacts/forged/download"),
        ):
            tampered = deepcopy(output)
            tampered["artifacts"][role][field] = forged
            body = {key: value for key, value in tampered.items() if key != "content_hash"}
            tampered["content_hash"] = v2_tools.hashlib.sha256(
                v2_tools._canonical_json(body).encode("utf-8")
            ).hexdigest()
            with pytest.raises(StrategyError, match=f"unsupported: {field}"):
                validate_materialize_sample_design_v2_tool_output(tampered)

    membership_hash = deepcopy(output)
    membership_hash["artifacts"]["membership"]["content_hash"] = "f" * 64
    membership_body = {
        key: value for key, value in membership_hash.items() if key != "content_hash"
    }
    membership_hash["content_hash"] = v2_tools.hashlib.sha256(
        v2_tools._canonical_json(membership_body).encode("utf-8")
    ).hexdigest()
    with pytest.raises(StrategyError, match="unsupported: content_hash"):
        validate_materialize_sample_design_v2_tool_output(membership_hash)

    forged_bundle_hash = deepcopy(output)
    forged_bundle_hash["artifacts"]["bundle"]["content_hash"] = "f" * 64
    forged_bundle_body = {
        key: value
        for key, value in forged_bundle_hash.items()
        if key != "content_hash"
    }
    forged_bundle_hash["content_hash"] = v2_tools.hashlib.sha256(
        v2_tools._canonical_json(forged_bundle_body).encode("utf-8")
    ).hexdigest()
    with pytest.raises(StrategyError, match="artifact drifted"):
        validate_materialize_sample_design_v2_tool_output(forged_bundle_hash)

    swapped = deepcopy(output)
    swapped["artifacts"]["membership"] = deepcopy(output["artifacts"]["bundle"])
    swapped_body = {key: value for key, value in swapped.items() if key != "content_hash"}
    swapped["content_hash"] = v2_tools.hashlib.sha256(
        v2_tools._canonical_json(swapped_body).encode("utf-8")
    ).hexdigest()
    with pytest.raises(StrategyError, match="artifact drifted|unsupported"):
        validate_materialize_sample_design_v2_tool_output(swapped)


def test_v2_reuses_membership_across_policy_specific_bundles(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    first = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    changed = deepcopy(fx["request"])
    changed["policy"]["minimum_bad_count"] = 2

    second = run_materialize_sample_design_v2(changed, fx["ctx"], fx["runtime"])

    assert second["membership_id"] == first["membership_id"]
    assert second["artifacts"]["membership"] == first["artifacts"]["membership"]
    assert second["bundle_id"] != first["bundle_id"]
    assert (
        second["artifacts"]["bundle"]["content_hash"]
        != first["artifacts"]["bundle"]["content_hash"]
    )
    artifacts = _v2_artifacts(fx)
    assert [item["kind"] for item in artifacts].count(
        SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    ) == 1
    assert [item["kind"] for item in artifacts].count(
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    ) == 2
    membership_record = next(
        item
        for item in artifacts
        if item["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    assert "request" not in membership_record["provenance"]
    assert "bundle_id" not in membership_record["provenance"]
    for bundle_record in (
        item for item in artifacts if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    ):
        assert bundle_record["provenance"]["membership_artifact_id"] == (
            membership_record["id"]
        )


def test_v2_not_matured_is_typed_and_cannot_claim_development_scope(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["scope"] = "exploration_only"
    request["maturity"] = {
        "status": "not_matured",
        "performance_window_days": 30,
        "cutoff_date": "2026-02-15",
        "reason": "Only early cohorts are mature.",
    }

    output = run_materialize_sample_design_v2(request, fx["ctx"], fx["runtime"])

    assert output["bundle"]["populations"][1]["maturity_evidence"]["status"] == "not_matured"
    risk_bad_statuses = {
        item["status"]
        for item in output["bundle"]["metric_observations"]
        if item["population"] == "risk"
        and next(
            definition["metric_key"]
            for definition in output["bundle"]["metric_definitions"]
            if definition["metric_definition_id"]
            == item["metric_definition_ref"]["metric_definition_id"]
        )
        in {"bad_count", "bad_rate"}
    }
    assert risk_bad_statuses == {"not_matured"}


def test_v2_not_matured_must_match_the_deterministic_cutoff(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["scope"] = "exploration_only"
    request["maturity"] = {
        "status": "not_matured",
        "performance_window_days": 30,
        "cutoff_date": "2026-12-31",
        "reason": "Claimed immature even though every row is eligible.",
    }

    with pytest.raises(StrategyError, match="at least one risk row outside"):
        run_materialize_sample_design_v2(
            request, fx["ctx"], fx["runtime"]
        )


@pytest.mark.parametrize("partitioning_method", ["predicate_ast", "time_ranges"])
def test_v2_observation_window_rejects_selected_rows_outside_it(
    tmp_path: Path,
    partitioning_method: str,
) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["observation_window"]["start"] = "2026-02-01"
    if partitioning_method == "time_ranges":
        request["partitioning"] = {
            "method": "time_ranges",
            "column": "apply_date",
            "ranges": {
                "development": {
                    "start": "2026-01-01",
                    "end": "2026-01-31",
                },
                "validation": {
                    "start": "2026-02-01",
                    "end": "2026-02-28",
                },
                "oot": {
                    "start": "2026-03-01",
                    "end": "2026-03-31",
                },
            },
        }

    with pytest.raises(StrategyError, match="outside the observation window"):
        run_materialize_sample_design_v2(
            request, fx["ctx"], fx["runtime"]
        )


def test_v2_time_field_requires_date_semantics_and_rejects_numeric_epoch(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["field_bindings"]["time_field"] = "weight"

    with pytest.raises(StrategyError, match="date semantic role"):
        run_materialize_sample_design_v2(
            request, fx["ctx"], fx["runtime"]
        )
    with pytest.raises(StrategyError, match="numeric epoch-style"):
        v2_tools._date_series(pd.Series([20261231]), "apply_date")


def test_v2_time_ranges_resolve_the_same_exact_legacy_development_rows(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["partitioning"] = {
        "method": "time_ranges",
        "column": "apply_date",
        "ranges": {
            "development": {"start": "2026-01-01", "end": "2026-01-31"},
            "validation": {"start": "2026-02-01", "end": "2026-02-28"},
            "oot": {"start": "2026-03-01", "end": "2026-03-31"},
        },
    }

    output = run_materialize_sample_design_v2(request, fx["ctx"], fx["runtime"])

    split = output["bundle"]["sample_design"]["sample_semantics"]["split_definition"]
    assert split["method"] == "time_ranges"
    assert output["legacy_mapping"]["row_equal"] is True


def test_v2_parallel_allows_independent_populations_but_nested_rejects(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["approval_population"]["exclusion"] = _eq("customer_id", "a")
    with pytest.raises(StrategyError, match="nested_same_cohort"):
        run_materialize_sample_design_v2(request, fx["ctx"], fx["runtime"])

    request["relationship"] = "parallel_time_cohorts"
    output = run_materialize_sample_design_v2(request, fx["ctx"], fx["runtime"])
    statuses = {item["code"]: item["status"] for item in output["bundle"]["diagnostics"]}
    assert statuses["risk_outside_approval"] == "not_applicable"


def test_v2_rejects_predicate_budget_and_legacy_row_inequality(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    request = deepcopy(fx["request"])
    request["risk_population"]["exclusion"] = _eq("customer_id", "a")
    with pytest.raises(StrategyError, match="do not equal.*legacy"):
        run_materialize_sample_design_v2(request, fx["ctx"], fx["runtime"])

    deep = _eq("sample_split", "dev")
    for _ in range(20):
        deep = {"op": "not", "arg": deep}
    request = deepcopy(fx["request"])
    request["approval_population"]["inclusion"] = deep
    with pytest.raises(StrategyError, match="depth budget"):
        run_materialize_sample_design_v2(request, fx["ctx"], fx["runtime"])


def test_v2_rejects_workspace_dataset_and_artifact_drift(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    membership = next(
        item for item in _v2_artifacts(fx) if item["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    membership_path = Path(membership["path"])
    original_membership = membership_path.read_bytes()
    membership_path.write_bytes(b"tampered")
    with pytest.raises(StrategyError, match="content hash|bytes changed"):
        run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    membership_path.write_bytes(original_membership)

    with sqlite3.connect(fx["settings"].db_path) as conn:
        conn.execute(
            "UPDATE data_workspaces SET revision = revision + 1 WHERE task_id = ?",
            (fx["task"].id,),
        )
    membership_record = _live_v2_artifact(
        fx, SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle_record = _live_v2_artifact(fx, SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND)
    with pytest.raises(StrategyError, match="workspace_revision|binding|DataWorkspace"):
        load_strategy_sample_design_v2_artifacts(
            fx["runtime"],
            task_id=fx["task"].id,
            membership_artifact_id=membership_record["id"],
            expected_membership_artifact_content_hash=membership_record[
                "content_hash"
            ],
            bundle_artifact_id=bundle_record["id"],
            expected_bundle_artifact_content_hash=bundle_record["content_hash"],
            expected_bundle_id=output["bundle_id"],
            expected_sample_design_id=output["sample_design_id"],
            expected_sample_design_content_hash=output["sample_design_content_hash"],
        )


def test_v2_registration_failure_rolls_back_pair(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path)
    original = fx["runtime"].task_artifacts.register_on_connection
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TaskArtifactDataError("injected registration failure")
        return original(*args, **kwargs)

    from marvis.repositories.task_artifacts import TaskArtifactDataError

    monkeypatch.setattr(fx["runtime"].task_artifacts, "register_on_connection", fail_second)
    with pytest.raises(StrategyError, match="injected registration failure"):
        run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    assert _v2_artifacts(fx) == []
    out_dir = fx["settings"].tasks_dir / fx["task"].id / "strategy_sample_designs_v2"
    assert not list(out_dir.glob("*.bin"))
    assert not list(out_dir.glob("*.json"))


def test_v2_post_database_commit_cleanup_failure_keeps_pair_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    original_commit = ArtifactUnitOfWork.commit

    def fail_cleanup(_self):
        raise RuntimeError("injected post-commit cleanup failure")

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", fail_cleanup)
    with pytest.raises(RuntimeError, match="injected post-commit cleanup failure"):
        run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])

    records = _v2_artifacts(fx)
    assert {record["kind"] for record in records} == {
        SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
        SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    }
    assert all(Path(record["path"]).is_file() for record in records)

    monkeypatch.setattr(ArtifactUnitOfWork, "commit", original_commit)
    replay = run_materialize_sample_design_v2(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    assert validate_materialize_sample_design_v2_tool_output(replay) == replay
    loaded = _load_output(fx, replay)
    assert loaded.bundle == replay["bundle"]
    assert loaded.membership["header"] == replay["membership"]
    assert {record["id"] for record in _v2_artifacts(fx)} == {
        record["id"] for record in records
    }


def test_v2_rechecks_workspace_under_writer_lock(tmp_path: Path, monkeypatch) -> None:
    fx = _setup(tmp_path)
    original = v2_tools._persist_pair

    def drift_before_persist(*args, **kwargs):
        with sqlite3.connect(fx["settings"].db_path) as conn:
            conn.execute(
                "UPDATE data_workspaces SET revision = revision + 1 WHERE task_id = ?",
                (fx["task"].id,),
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_tools, "_persist_pair", drift_before_persist)
    with pytest.raises(StrategyError, match="DataWorkspace|binding changed"):
        run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    assert _v2_artifacts(fx) == []


@pytest.mark.parametrize("drift", ["row_count", "dataset_bytes"])
def test_v2_loader_rechecks_frame_and_dataset_after_read(
    tmp_path: Path,
    monkeypatch,
    drift: str,
) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    original_read = fx["runtime"].backend.read_frame

    def drifting_read(path, columns=None):
        frame = original_read(path, columns=columns)
        if drift == "row_count":
            return frame.iloc[:-1].copy()
        Path(path).write_bytes(Path(path).read_bytes() + b"drift-after-read")
        return frame

    monkeypatch.setattr(fx["runtime"].backend, "read_frame", drifting_read)
    expected = "row count changed" if drift == "row_count" else "bytes changed"
    with pytest.raises(StrategyError, match=expected):
        _load_output(fx, output)


def test_v2_loaded_binding_can_be_revalidated_under_downstream_writer_lock(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    loaded = _load_output(fx, output)

    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_sample_design_v2_artifact_binding_on_connection(conn, loaded)

    membership_path = loaded.membership_path
    original = membership_path.read_bytes()
    membership_path.write_bytes(b"tampered")
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="content hash"):
            require_strategy_sample_design_v2_artifact_binding_on_connection(
                conn, loaded
            )
    membership_path.write_bytes(original)


def test_v2_symlink_and_v1_loader_fail_closed(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    output = run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])
    bundle = next(item for item in _v2_artifacts(fx) if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND)
    path = Path(bundle["path"])
    backup = path.with_suffix(".saved")
    path.rename(backup)
    path.symlink_to(backup)
    with pytest.raises(StrategyError, match="regular file|symlink"):
        run_materialize_sample_design_v2(fx["request"], fx["ctx"], fx["runtime"])

    with pytest.raises(StrategyError, match="registry binding"):
        load_strategy_sample_design_artifact(
            fx["runtime"],
            task_id=fx["task"].id,
            artifact_id=bundle["id"],
            expected_artifact_content_hash=bundle["content_hash"],
            expected_sample_design_id=output["sample_design_id"],
            expected_sample_design_content_hash=output["sample_design_content_hash"],
        )

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.orchestrator.templates.monitoring import STRATEGY_MONITORING
from marvis.packs.strategy import monitor_tools as monitor_tools_module
from marvis.packs.strategy import tools as strategy_tools_module
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.monitor_tools import (
    rerun_strategy_monitoring_with_candidate_plan,
    tool_run_strategy_monitoring,
)
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    load_monitoring_plan,
    save_monitoring_plan,
)
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.plugins.contracts import ToolContext
from marvis.repositories.audit import _list_audit_rows
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository
from marvis.settings import build_settings


def _spec(strategy_type: str) -> dict:
    values = {
        "limit": (1000.0, 2000.0),
        "pricing": (0.10, 0.20),
        "segmentation": ("base", "high"),
    }
    default_value, matched_value = values[strategy_type]
    action_type = "segment" if strategy_type == "segmentation" else strategy_type
    return {
        "strategy_type": strategy_type,
        "default_action": {"type": action_type, "value": default_value},
        "rules": [
            {
                "rule_id": "x-positive",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "x",
                    "operator": ">",
                    "value": 0,
                },
                "action": {"type": action_type, "value": matched_value},
            }
        ],
    }


def _runtime(tmp_path: Path, strategy_type: str):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name=f"{strategy_type} monitoring ledger",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
        )
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    strategy = build_strategy_from_spec(_spec(strategy_type))
    strategies = StrategyRepository(settings.db_path)
    strategies.create_strategy(task.id, strategy)
    strategies.adopt_strategy_with_audit(
        strategy.id,
        reason="monitoring runtime fixture",
        audit={
            "kind": "strategy.adopt.fixture",
            "target_ref": strategy.id,
            "outcome": "succeeded",
            "detail": {"task_id": task.id},
        },
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    return settings, task, registry, strategy, strategies, ctx


def _register(
    registry: DatasetRegistry,
    tmp_path: Path,
    *,
    task_id: str,
    name: str,
    frame: pd.DataFrame,
):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="monitoring")


def _thresholds(strategy_type: str) -> dict:
    if strategy_type == "limit":
        return {
            "mean_limit": {
                "metric": "mean_limit",
                "direction": "max",
                "warn": 10_000.0,
                "fail": 20_000.0,
            },
            "expected_loss": {
                "metric": "expected_loss",
                "direction": "max",
                "warn": 10_000.0,
                "fail": 20_000.0,
            },
        }
    return {
        metric: {
            "metric": metric,
            "direction": "max",
            "warn": 10_000.0,
            "fail": 20_000.0,
        }
        for metric in ("mean_rate", "expected_loss", "profit", "roa")
    }


def _ledger_plan(
    settings,
    strategy,
    strategies: StrategyRepository,
    *,
    bindings: dict,
    thresholds: dict | None = None,
):
    effect_hash = strategies.get_strategy_spec_hash(strategy.id)
    assert effect_hash is not None
    plan = MonitoringPlan(
        strategy_id=strategy.id,
        version=1,
        thresholds=thresholds or _thresholds(strategy.strategy_type),
        expectation_baseline={"strategy_effect_hash": effect_hash},
        economics_bindings=bindings,
    )
    return StrategyMonitoringRepository(settings.db_path).create_plan(
        plan, expected_revision=0
    )


@pytest.mark.parametrize(
    "binding_mode, bindings, frame, expected_loss",
    [
        (
            "scalar",
            {
                "pd": {"kind": "scalar", "value": 0.10},
                "lgd": {"kind": "scalar", "value": 0.50},
                "utilization": {"kind": "scalar", "value": 0.50},
            },
            pd.DataFrame({"x": [0, 1]}),
            75.0,
        ),
        (
            "column",
            {
                "pd": {"kind": "column", "column": "fresh_pd"},
                "lgd": {"kind": "column", "column": "fresh_lgd"},
                "utilization": {"kind": "column", "column": "fresh_utilization"},
            },
            pd.DataFrame(
                {
                    "x": [0, 1],
                    "fresh_pd": [0.10, 0.20],
                    "fresh_lgd": [0.50, 0.40],
                    "fresh_utilization": [0.60, 0.70],
                }
            ),
            142.0,
        ),
    ],
)
def test_limit_monitoring_recomputes_scalar_and_column_economics(
    tmp_path: Path,
    binding_mode: str,
    bindings: dict,
    frame: pd.DataFrame,
    expected_loss: float,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    plan = _ledger_plan(
        settings,
        strategy,
        strategies,
        bindings=bindings,
    )
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name=f"limit-{binding_mode}",
        frame=frame,
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
    )

    checks = {row["id"]: row for row in output["checks"]}
    assert checks["expected_loss"]["value"] == pytest.approx(expected_loss)
    assert output["economics"]["expected_loss"] == pytest.approx(expected_loss)
    assert output["monitoring_plan_id"] == plan.id
    assert output["monitoring_plan_revision"] == 1
    assert output["monitoring_plan_hash"] == plan.payload_hash
    assert output["plan_source"] == "ledger"
    assert output["monitoring_run_id"]
    assert output["plan_updated"] is False
    registered = DatasetRepository(settings.db_path).get_dataset(dataset.id)
    assert registered is not None
    assert output["monitoring_evidence"]["dataset_content_hash"] == (
        registered.content_hash
    )

    stored = StrategyMonitoringRepository(settings.db_path).get_run(
        output["monitoring_run_id"]
    )
    assert stored is not None
    assert stored.result["economics"]["expected_loss"] == pytest.approx(
        expected_loss
    )
    assert "by_row" not in json.dumps(stored.result, ensure_ascii=False)
    assert stored.result_hash == output["monitoring_evidence"]["result_hash"]


def test_pricing_monitoring_recomputes_mixed_economics_without_raw_rows(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "pricing"
    )
    _ledger_plan(
        settings,
        strategy,
        strategies,
        bindings={
            "ead": {"kind": "column", "column": "fresh_ead"},
            "pd": {"kind": "column", "column": "fresh_pd"},
            "lgd": {"kind": "scalar", "value": 0.50},
            "funding_rate": {"kind": "scalar", "value": 0.03},
            "term_months": {"kind": "scalar", "value": 12.0},
            "operating_cost_per_loan": {"kind": "scalar", "value": 10.0},
        },
    )
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="pricing-mixed",
        frame=pd.DataFrame(
            {
                "x": [0, 1],
                "fresh_ead": [1000.0, 2000.0],
                "fresh_pd": [0.10, 0.20],
            }
        ),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
    )

    assert output["economics"]["expected_loss"] == pytest.approx(250.0)
    assert output["economics"]["profit"] == pytest.approx(140.0)
    assert output["economics"]["roa"] == pytest.approx(140.0 / 3000.0)
    assert "by_row" not in output["economics"]
    checks = {row["id"]: row for row in output["checks"]}
    assert checks["profit"]["value"] == pytest.approx(140.0)
    assert checks["roa"]["value"] == pytest.approx(140.0 / 3000.0)


def test_missing_compatibility_column_rolls_back_plan_import_and_run(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    effect_hash = strategies.get_strategy_spec_hash(strategy.id)
    artifact_plan = MonitoringPlan(
        strategy_id=strategy.id,
        version=1,
        thresholds=_thresholds("limit"),
        expectation_baseline={"strategy_effect_hash": effect_hash},
        economics_bindings={
            "pd": {"kind": "column", "column": "missing_pd"},
            "lgd": {"kind": "scalar", "value": 0.50},
            "utilization": {"kind": "scalar", "value": 0.60},
        },
    )
    artifact_path = tmp_path / "compatibility-plan.json"
    save_monitoring_plan(artifact_path, artifact_plan)
    strategies.save_strategy_artifact(
        strategy.id, kind="monitoring_plan_json", path=str(artifact_path)
    )
    before = artifact_path.read_bytes()
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="missing-column",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    with pytest.raises(StrategyError, match="missing economics column"):
        tool_run_strategy_monitoring(
            {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
        )

    ledger = StrategyMonitoringRepository(settings.db_path)
    assert ledger.list_plans(strategy.id) == []
    assert ledger.list_runs(strategy.id) == []
    assert artifact_path.read_bytes() == before


def test_invalid_fresh_economics_column_does_not_create_run(tmp_path: Path) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    _ledger_plan(
        settings,
        strategy,
        strategies,
        bindings={
            "pd": {"kind": "column", "column": "fresh_pd"},
            "lgd": {"kind": "scalar", "value": 0.50},
            "utilization": {"kind": "scalar", "value": 0.60},
        },
    )
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="invalid-column",
        frame=pd.DataFrame({"x": [0, 1], "fresh_pd": ["0.10", "invalid"]}),
    )

    with pytest.raises(StrategyError, match="pd must contain numeric values"):
        tool_run_strategy_monitoring(
            {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
        )

    assert StrategyMonitoringRepository(settings.db_path).list_runs(
        strategy.id
    ) == []
    assert not _list_audit_rows(
        settings.db_path, kind="strategy.monitor", target_ref=strategy.id
    )


def test_v2_ledger_plan_without_economics_bindings_fails_closed(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    _ledger_plan(settings, strategy, strategies, bindings={})
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="missing-v2-bindings",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    with pytest.raises(StrategyError, match="requires economics_bindings"):
        tool_run_strategy_monitoring(
            {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
        )

    assert StrategyMonitoringRepository(settings.db_path).list_runs(
        strategy.id
    ) == []


def test_unledgered_v2_plan_without_economics_bindings_fails_before_import(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    artifact = MonitoringPlan(
        strategy_id=strategy.id,
        version=1,
        thresholds=_thresholds("limit"),
        expectation_baseline={
            "strategy_effect_hash": strategies.get_strategy_spec_hash(strategy.id)
        },
    )
    artifact_path = tmp_path / "unledgered-v2-missing-bindings.json"
    save_monitoring_plan(artifact_path, artifact)
    strategies.save_strategy_artifact(
        strategy.id, kind="monitoring_plan_json", path=str(artifact_path)
    )
    before = artifact_path.read_bytes()
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="missing-unledgered-v2-bindings",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    with pytest.raises(StrategyError, match="requires economics_bindings"):
        tool_run_strategy_monitoring(
            {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
        )

    ledger = StrategyMonitoringRepository(settings.db_path)
    assert ledger.list_plans(strategy.id) == []
    assert ledger.list_runs(strategy.id) == []
    assert artifact_path.read_bytes() == before
    assert not _list_audit_rows(
        settings.db_path, kind="strategy.monitor", target_ref=strategy.id
    )


def test_unledgered_v2_plan_is_imported_with_explicit_source_after_success(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    artifact = MonitoringPlan(
        strategy_id=strategy.id,
        version=1,
        thresholds=_thresholds("limit"),
        expectation_baseline={
            "strategy_effect_hash": strategies.get_strategy_spec_hash(strategy.id)
        },
        economics_bindings={
            "pd": {"kind": "scalar", "value": 0.10},
            "lgd": {"kind": "scalar", "value": 0.50},
            "utilization": {"kind": "scalar", "value": 0.50},
        },
    )
    artifact_path = tmp_path / "unledgered-v2.json"
    save_monitoring_plan(artifact_path, artifact)
    strategies.save_strategy_artifact(
        strategy.id, kind="monitoring_plan_json", path=str(artifact_path)
    )
    before = artifact_path.read_bytes()
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="compatibility-import-success",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
    )

    assert output["plan_source"] == "compatibility_import"
    assert output["monitoring_evidence"]["plan_source"] == "compatibility_import"
    assert artifact_path.read_bytes() == before
    ledger = StrategyMonitoringRepository(settings.db_path)
    plans = ledger.list_plans(strategy.id)
    assert len(plans) == 1
    assert plans[0].plan.plan_version == 2
    assert len(ledger.list_runs(strategy.id)) == 1


def test_ledger_plan_wins_and_registered_plan_file_is_not_rewritten(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    ledger_plan = _ledger_plan(
        settings,
        strategy,
        strategies,
        bindings={
            "pd": {"kind": "scalar", "value": 0.10},
            "lgd": {"kind": "scalar", "value": 0.50},
            "utilization": {"kind": "scalar", "value": 0.50},
        },
    )
    artifact_plan = MonitoringPlan(
        strategy_id=strategy.id,
        version=1,
        plan_version=1,
        thresholds={
            "mean_limit": {
                "metric": "mean_limit",
                "direction": "max",
                "warn": 1.0,
                "fail": 2.0,
            }
        },
    )
    artifact_path = tmp_path / "stale-artifact-plan.json"
    save_monitoring_plan(artifact_path, artifact_plan)
    strategies.save_strategy_artifact(
        strategy.id, kind="monitoring_plan_json", path=str(artifact_path)
    )
    before = artifact_path.read_bytes()
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="ledger-wins",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
    )

    assert output["plan_source"] == "ledger"
    assert output["monitoring_plan_id"] == ledger_plan.id
    assert output["monitoring_plan_hash"] == ledger_plan.payload_hash
    assert {row["id"] for row in output["checks"]} == {
        "mean_limit",
        "expected_loss",
    }
    assert artifact_path.read_bytes() == before


def test_legacy_v1_without_economics_remains_explicit_na_and_is_imported_atomically(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    legacy = MonitoringPlan(
        strategy_id=strategy.id,
        version=1,
        plan_version=1,
        thresholds=_thresholds("limit"),
        expectation_baseline={
            "strategy_effect_hash": strategies.get_strategy_spec_hash(strategy.id)
        },
    )
    artifact_path = tmp_path / "legacy-v1.json"
    save_monitoring_plan(artifact_path, legacy)
    strategies.save_strategy_artifact(
        strategy.id, kind="monitoring_plan_json", path=str(artifact_path)
    )
    before = artifact_path.read_bytes()
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="legacy-v1-fresh",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": dataset.id}, ctx
    )

    checks = {row["id"]: row for row in output["checks"]}
    assert checks["expected_loss"]["value"] is None
    assert checks["expected_loss"]["level"] == "n/a"
    assert output["plan_source"] == "legacy_v1"
    assert output["economics"] == {}
    assert artifact_path.read_bytes() == before
    ledger = StrategyMonitoringRepository(settings.db_path)
    assert len(ledger.list_plans(strategy.id)) == 1
    assert len(ledger.list_runs(strategy.id)) == 1


def test_cross_task_dataset_is_rejected_without_monitoring_run(tmp_path: Path) -> None:
    settings, _task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    _ledger_plan(
        settings,
        strategy,
        strategies,
        bindings={
            "pd": {"kind": "scalar", "value": 0.10},
            "lgd": {"kind": "scalar", "value": 0.50},
            "utilization": {"kind": "scalar", "value": 0.50},
        },
    )
    foreign_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="foreign dataset",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
        )
    )
    foreign = _register(
        registry,
        tmp_path,
        task_id=foreign_task.id,
        name="foreign-monitoring",
        frame=pd.DataFrame({"x": [0, 1]}),
    )

    with pytest.raises(StrategyError, match="dataset not found"):
        tool_run_strategy_monitoring(
            {"strategy_id": strategy.id, "dataset_id": foreign.id}, ctx
        )

    assert StrategyMonitoringRepository(settings.db_path).list_runs(
        strategy.id
    ) == []


def _red_limit_source_run(tmp_path: Path):
    settings, task, registry, strategy, strategies, ctx = _runtime(
        tmp_path, "limit"
    )
    plan = _ledger_plan(
        settings,
        strategy,
        strategies,
        bindings={
            "pd": {"kind": "scalar", "value": 0.10},
            "lgd": {"kind": "scalar", "value": 0.50},
            "utilization": {"kind": "scalar", "value": 0.50},
        },
        thresholds={
            "mean_limit": {
                "metric": "mean_limit",
                "direction": "max",
                "warn": 1_250.0,
                "fail": 1_500.0,
            },
            "expected_loss": {
                "metric": "expected_loss",
                "direction": "max",
                "warn": 10_000.0,
                "fail": 20_000.0,
            },
        },
    )
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="adjust-threshold-source",
        frame=pd.DataFrame({"x": [1, 1], "bad": [0, 1]}),
    )
    source = tool_run_strategy_monitoring(
        {
            "strategy_id": strategy.id,
            "dataset_id": dataset.id,
            "target_col": "bad",
        },
        ctx,
    )
    assert source["overall_level"] == "red"
    stored_source = StrategyMonitoringRepository(settings.db_path).get_run(
        source["monitoring_run_id"]
    )
    assert stored_source is not None
    assert stored_source.result["monitoring_inputs"] == {
        "target_col": "bad",
        "score_col": None,
        "dataset_mode": "raw",
    }
    return settings, task, strategy, plan, dataset, source, ctx


def _adjusted_limit_plan(plan, *, plan_id: str = "adjusted-plan") -> MonitoringPlan:
    thresholds = {name: dict(spec) for name, spec in plan.plan.thresholds.items()}
    thresholds["mean_limit"].update({"warn": 2_500.0, "fail": 3_000.0})
    return replace(
        plan.plan,
        monitoring_plan_id=plan_id,
        revision=plan.revision + 1,
        supersedes_plan_id=plan.id,
        thresholds=thresholds,
    )


def test_adjusted_plan_rerun_is_one_immutable_artifact_ledger_transaction(
    tmp_path: Path,
) -> None:
    settings, task, strategy, plan, dataset, source, ctx = _red_limit_source_run(
        tmp_path
    )
    candidate = _adjusted_limit_plan(plan)
    patch = {"mean_limit": {"warn": 2_500.0, "fail": 3_000.0}}

    output = rerun_strategy_monitoring_with_candidate_plan(
        ctx=ctx,
        strategy_id=strategy.id,
        source_monitoring_run_id=source["monitoring_run_id"],
        expected_latest_plan_id=plan.id,
        expected_latest_plan_revision=plan.revision,
        expected_latest_plan_hash=plan.payload_hash,
        candidate_plan=candidate,
        reason="risk owner approved wider operational bands",
        threshold_patch=patch,
    )

    assert output["overall_level"] == "green"
    assert output["monitoring_plan_id"] == "adjusted-plan"
    assert output["monitoring_plan_revision"] == 2
    assert output["monitoring_run_id"] != source["monitoring_run_id"]
    assert output["dataset_id"] == dataset.id
    artifact_path = Path(output["plan_artifact_path"])
    assert artifact_path.is_file()
    assert load_monitoring_plan(artifact_path) == candidate

    ledger = StrategyMonitoringRepository(settings.db_path)
    assert [row.revision for row in ledger.list_plans(strategy.id)] == [1, 2]
    assert len(ledger.list_runs(strategy.id)) == 2
    adjusted_run = ledger.get_run(output["monitoring_run_id"])
    assert adjusted_run is not None
    source_run = ledger.get_run(source["monitoring_run_id"])
    assert source_run is not None
    assert adjusted_run.result["monitoring_inputs"] == source_run.result[
        "monitoring_inputs"
    ]
    assert adjusted_run.dataset_content_hash == source["monitoring_evidence"][
        "dataset_content_hash"
    ]
    artifacts = StrategyRepository(settings.db_path).list_strategy_artifacts(
        strategy.id
    )
    adjusted_artifact = next(
        row
        for row in artifacts
        if row["kind"] == "monitoring_plan_json"
        and row["path"] == str(artifact_path)
    )
    assert adjusted_artifact["integrity_status"] == "verified"
    assert adjusted_artifact["content_hash"] == sha256_file(artifact_path)
    assert adjusted_artifact["content_size"] == artifact_path.stat().st_size
    provenance = adjusted_artifact["provenance"]
    assert provenance["schema_version"] == "strategy-artifact-provenance.v1"
    assert provenance["producer_version"] == "strategy.monitoring.adjust_plan.v1"
    assert provenance["task_id"] == task.id
    assert provenance["strategy_id"] == strategy.id
    assert provenance["kind"] == "monitoring_plan_json"
    assert provenance["evidence"] == {
        "operation": "strategy.monitoring.adjust_threshold",
        "source_monitoring_run_id": source["monitoring_run_id"],
        "source_monitoring_run_hash": source_run.result_hash,
        "dataset_id": dataset.id,
        "dataset_content_hash": source_run.dataset_content_hash,
        "strategy_effect_hash": source_run.strategy_effect_hash,
        "economics_binding_hash": source_run.economics_binding_hash,
        "old_monitoring_plan_id": plan.id,
        "old_monitoring_plan_revision": plan.revision,
        "old_monitoring_plan_hash": plan.payload_hash,
        "monitoring_plan_id": output["monitoring_plan_id"],
        "monitoring_plan_revision": output["monitoring_plan_revision"],
        "monitoring_plan_hash": output["monitoring_plan_hash"],
        "monitoring_run_id": output["monitoring_run_id"],
        "monitoring_run_hash": adjusted_run.result_hash,
        "threshold_patch": patch,
    }
    disposition = _list_audit_rows(
        settings.db_path,
        kind="strategy.monitoring.disposition",
        target_ref=source["monitoring_run_id"],
    )
    assert len(disposition) == 1
    detail = disposition[0]["detail"]
    assert detail["disposition"] == "adjust_threshold"
    assert detail["source_monitoring_run_id"] == source["monitoring_run_id"]
    assert detail["old_monitoring_plan_id"] == plan.id
    assert detail["new_monitoring_plan_id"] == output["monitoring_plan_id"]
    assert detail["new_monitoring_run_id"] == output["monitoring_run_id"]
    assert detail["reason"] == "risk owner approved wider operational bands"
    assert detail["threshold_patch"] == patch
    assert "metrics" not in detail
    assert "checks" not in detail

    with pytest.raises(StrategyError, match="already has a disposition"):
        rerun_strategy_monitoring_with_candidate_plan(
            ctx=ctx,
            strategy_id=strategy.id,
            source_monitoring_run_id=source["monitoring_run_id"],
            expected_latest_plan_id=plan.id,
            expected_latest_plan_revision=plan.revision,
            expected_latest_plan_hash=plan.payload_hash,
            candidate_plan=replace(candidate, monitoring_plan_id="replay-plan"),
            reason="replay",
            threshold_patch=patch,
        )


def test_adjusted_plan_audit_failure_rolls_back_plan_run_artifact_and_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, task, strategy, plan, _dataset, source, ctx = _red_limit_source_run(
        tmp_path
    )
    candidate = _adjusted_limit_plan(plan, plan_id="rollback-plan")
    original_write_audit = monitor_tools_module._write_audit_row

    def fail_disposition_audit(conn, *, kind, **kwargs):
        if kind == "strategy.monitoring.disposition":
            raise RuntimeError("injected disposition audit failure")
        return original_write_audit(conn, kind=kind, **kwargs)

    monkeypatch.setattr(
        monitor_tools_module,
        "_write_audit_row",
        fail_disposition_audit,
    )
    strategy_dir = Path(settings.tasks_dir) / task.id / "strategy"

    with pytest.raises(RuntimeError, match="injected disposition audit failure"):
        rerun_strategy_monitoring_with_candidate_plan(
            ctx=ctx,
            strategy_id=strategy.id,
            source_monitoring_run_id=source["monitoring_run_id"],
            expected_latest_plan_id=plan.id,
            expected_latest_plan_revision=plan.revision,
            expected_latest_plan_hash=plan.payload_hash,
            candidate_plan=candidate,
            reason="must rollback",
            threshold_patch={
                "mean_limit": {"warn": 2_500.0, "fail": 3_000.0}
            },
        )

    ledger = StrategyMonitoringRepository(settings.db_path)
    assert [row.id for row in ledger.list_plans(strategy.id)] == [plan.id]
    assert [row.id for row in ledger.list_runs(strategy.id)] == [
        source["monitoring_run_id"]
    ]
    assert StrategyRepository(settings.db_path).list_strategy_artifacts(
        strategy.id
    ) == []
    assert not list(strategy_dir.glob("monitoring_plan_*_r2_*.json"))
    assert not _list_audit_rows(
        settings.db_path,
        kind="strategy.monitoring.disposition",
        target_ref=source["monitoring_run_id"],
    )


def test_adjusted_plan_revalidates_latest_source_run_after_calculation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, task, strategy, plan, _dataset, source, ctx = _red_limit_source_run(
        tmp_path
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    interloper = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="newer-monitoring-run-race",
        frame=pd.DataFrame({"x": [0, 1], "bad": [0, 1]}),
    )
    original_calculate = monitor_tools_module._calculate_strategy_monitoring
    inserted = False

    def calculate_then_insert_newer_run(**kwargs):
        nonlocal inserted
        calculation = original_calculate(**kwargs)
        if not inserted:
            inserted = True
            newer = tool_run_strategy_monitoring(
                {
                    "strategy_id": strategy.id,
                    "dataset_id": interloper.id,
                    "target_col": "bad",
                },
                ctx,
            )
            assert newer["monitoring_run_id"] != source["monitoring_run_id"]
        return calculation

    monkeypatch.setattr(
        monitor_tools_module,
        "_calculate_strategy_monitoring",
        calculate_then_insert_newer_run,
    )

    with pytest.raises(StrategyError, match="latest monitoring run"):
        rerun_strategy_monitoring_with_candidate_plan(
            ctx=ctx,
            strategy_id=strategy.id,
            source_monitoring_run_id=source["monitoring_run_id"],
            expected_latest_plan_id=plan.id,
            expected_latest_plan_revision=plan.revision,
            expected_latest_plan_hash=plan.payload_hash,
            candidate_plan=_adjusted_limit_plan(plan, plan_id="raced-plan"),
            reason="must lose the race",
            threshold_patch={
                "mean_limit": {"warn": 2_500.0, "fail": 3_000.0}
            },
        )

    ledger = StrategyMonitoringRepository(settings.db_path)
    assert [row.id for row in ledger.list_plans(strategy.id)] == [plan.id]
    assert len(ledger.list_runs(strategy.id)) == 2
    assert StrategyRepository(settings.db_path).list_strategy_artifacts(
        strategy.id
    ) == []
    assert not _list_audit_rows(
        settings.db_path,
        kind="strategy.monitoring.disposition",
        target_ref=source["monitoring_run_id"],
    )


@pytest.mark.parametrize(
    ("disposition_result", "expected_action", "receipt"),
    (
        (
            {
                "status": "new_version_created",
                "disposition": "new_version",
                "new_task_id": "task-next",
                "new_strategy_id": "strategy-next",
                "new_dataset_id": "dataset-next",
                "overall_level": "red",
                "checks": [],
            },
            "new_version",
            "task-next",
        ),
        (
            {
                "status": "thresholds_adjusted",
                "disposition": "adjust_threshold",
                "monitoring_plan_id": "plan-r2",
                "monitoring_plan_revision": 2,
                "monitoring_plan_hash": "a" * 64,
                "resolved_monitoring_run_id": "run-r2",
                "overall_level": "green",
                "checks": [],
            },
            "adjust_threshold",
            "run-r2",
        ),
        (
            {
                "status": "observed",
                "disposition": "observe",
                "source_monitoring_run_id": "run-red",
                "resolved_monitoring_run_id": "run-red",
                "overall_level": "red",
                "checks": [],
            },
            "observe",
            "run-red",
        ),
        (
            {
                "status": "acknowledged",
                "disposition": None,
                "source_monitoring_run_id": "run-green",
                "resolved_monitoring_run_id": "run-green",
                "overall_level": "green",
                "checks": [],
            },
            "acknowledge",
            "run-green",
        ),
    ),
)
def test_executed_disposition_projection_never_suggests_an_unexecuted_workflow(
    disposition_result: dict,
    expected_action: str,
    receipt: str,
) -> None:
    action = monitor_tools_module._executed_disposition_next_action(
        disposition_result,
        strategy_id="strategy-source",
    )

    assert action["action"] == expected_action
    assert action["kind"] != "suggest_template"
    assert receipt in json.dumps(action, ensure_ascii=False)
    assert "template_id" not in action


def test_strategy_monitoring_template_requires_ledger_receipts() -> None:
    run_step = STRATEGY_MONITORING.steps[0]
    postcheck_fields = {
        check.spec.get("field")
        for check in run_step.post_checks
        if check.kind == "nonempty"
    }

    assert {"monitoring_run_id", "monitoring_plan_id"} <= postcheck_fields
    report_step = STRATEGY_MONITORING.steps[-1]
    assert report_step.inputs_template == {
        "strategy_id": "{slot:strategy_id}",
        "source_monitoring_run_id": (
            "$ref:处置监控结果.output.source_monitoring_run_id"
        ),
    }
    assert {
        check.spec.get("field")
        for check in report_step.post_checks
        if check.kind == "nonempty"
    } == {"report_path", "artifact_id"}


@pytest.mark.parametrize(
    ("kernel_result", "message"),
    (
        (
            {
                "checks": [{"id": "score_psi", "level": "green"}],
                "top_drifted_features": [],
            },
            "missing or unsupported overall level",
        ),
        (
            {
                "overall_level": "green",
                "checks": ["not-evidence"],
                "top_drifted_features": [],
            },
            "invalid check evidence",
        ),
        (
            {
                "overall_level": "green",
                "checks": [{"id": "score_psi", "level": "red"}],
                "top_drifted_features": [],
            },
            "conflicts with its checks",
        ),
    ),
)
def test_model_monitoring_fails_closed_on_invalid_kernel_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kernel_result: dict,
    message: str,
) -> None:
    _settings, _task, _registry, _strategy, _strategies, ctx = _runtime(
        tmp_path,
        "limit",
    )
    plan = MonitoringPlan(
        strategy_id="strategy-with-model",
        version=1,
        experiment_id="experiment-1",
    )

    def invalid_model_monitor(_inputs, _ctx):
        return kernel_result

    from marvis.packs.modeling import monitor_tools as modeling_monitor_tools

    monkeypatch.setattr(
        modeling_monitor_tools,
        "_calculate_monitor_run",
        invalid_model_monitor,
    )

    with pytest.raises(StrategyError, match=message):
        monitor_tools_module._run_model_monitoring(
            {"dataset_id": "dataset-1"},
            ctx,
            plan,
        )


def test_model_monitoring_thresholds_are_materialized_and_routed_by_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marvis.packs.modeling import monitor_tools as modeling_monitor_tools

    strategy_thresholds = {
        "approval_floor": {
            "metric": "approval_rate",
            "direction": "min",
            "warn": 0.70,
            "fail": 0.60,
        }
    }
    pure_rule = strategy_tools_module._with_model_monitoring_thresholds(
        strategy_thresholds,
        experiment_id=None,
    )
    assert pure_rule == strategy_thresholds

    combined = strategy_tools_module._with_model_monitoring_thresholds(
        strategy_thresholds,
        experiment_id="experiment-1",
    )
    assert set(combined) == {
        "approval_floor",
        *modeling_monitor_tools.MONITOR_RUN_THRESHOLDS,
    }
    assert combined["score_psi"] == modeling_monitor_tools.MONITOR_RUN_THRESHOLDS[
        "score_psi"
    ]

    plan = MonitoringPlan(
        strategy_id="strategy-with-model",
        version=1,
        experiment_id="experiment-1",
        thresholds=combined,
    )
    assert [
        check_id
        for check_id, _spec in monitor_tools_module._strategy_monitoring_threshold_items(
            plan
        )
    ] == ["approval_floor"]

    captured = {}

    def valid_model_monitor(inputs, _ctx):
        captured.update(inputs)
        return {
            "overall_level": "green",
            "checks": [{"id": "score_psi", "level": "green"}],
            "top_drifted_features": [],
        }

    monkeypatch.setattr(
        modeling_monitor_tools,
        "_calculate_monitor_run",
        valid_model_monitor,
    )
    checks, _drifted, level = monitor_tools_module._run_model_monitoring(
        {"dataset_id": "dataset-1"},
        object(),
        plan,
    )

    assert checks == [{"id": "score_psi", "level": "green"}]
    assert level == "green"
    assert captured["monitoring_policy"]["thresholds"] == {
        check_id: combined[check_id]
        for check_id in modeling_monitor_tools.MONITOR_RUN_THRESHOLDS
    }


def test_pure_rule_model_named_threshold_remains_a_strategy_check() -> None:
    plan = MonitoringPlan(
        strategy_id="pure-rule",
        version=1,
        thresholds={
            "score_psi": {
                "metric": "custom_strategy_metric",
                "direction": "max",
                "warn": 0.10,
                "fail": 0.20,
            }
        },
    )

    assert monitor_tools_module._model_monitoring_thresholds(plan) == {}
    assert monitor_tools_module._strategy_monitoring_threshold_items(plan) == [
        (
            "score_psi",
            {
                "metric": "custom_strategy_metric",
                "direction": "max",
                "warn": 0.10,
                "fail": 0.20,
            },
        )
    ]


def test_imported_model_plan_materializes_effective_default_thresholds() -> None:
    from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS

    legacy = MonitoringPlan(
        strategy_id="legacy-model-strategy",
        version=1,
        plan_version=1,
        experiment_id="experiment-1",
        thresholds={
            "approval_floor": {
                "metric": "approval_rate",
                "direction": "min",
                "warn": 0.70,
                "fail": 0.60,
            },
            "score_psi": {"warn": 0.12},
        },
    )

    imported = monitor_tools_module._materialize_imported_model_thresholds(legacy)

    assert legacy.thresholds["score_psi"] == {"warn": 0.12}
    assert set(imported.thresholds) == {
        "approval_floor",
        *MONITOR_RUN_THRESHOLDS,
    }
    assert imported.thresholds["score_psi"] == {
        **MONITOR_RUN_THRESHOLDS["score_psi"],
        "warn": 0.12,
    }

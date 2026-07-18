from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.monitor_tools import tool_run_strategy_monitoring
from marvis.packs.strategy.monitoring_plan import (
    load_monitoring_plan,
    save_monitoring_plan,
)
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.tools import tool_adopt_strategy, tool_backtest_strategy
from marvis.plugins.contracts import ToolContext
from marvis.repositories.audit import _list_audit_rows
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository
from marvis.settings import build_settings


def _spec(strategy_type: str) -> dict:
    actions = {
        "approval": (
            {"type": "approval"},
            {"type": "reject"},
        ),
        "reject": (
            {"type": "approval"},
            {"type": "reject"},
        ),
        "limit": (
            {"type": "limit", "value": 1000},
            {"type": "limit", "value": 2000},
        ),
        "pricing": (
            {"type": "pricing", "value": 0.10},
            {"type": "pricing", "value": 0.20},
        ),
        "segmentation": (
            {"type": "segment", "value": "base"},
            {"type": "segment", "value": "high"},
        ),
    }
    default_action, matched_action = actions[strategy_type]
    return {
        "strategy_type": strategy_type,
        "default_action": default_action,
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
                "action": matched_action,
            }
        ],
    }


def _economics_inputs(strategy_type: str) -> dict | None:
    if strategy_type == "limit":
        return {
            "pd_col": "pd",
            "lgd_value": 0.5,
            "utilization_value": 0.6,
        }
    if strategy_type == "pricing":
        return {
            "ead_col": "ead",
            "pd_col": "pd",
            "lgd_value": 0.5,
            "funding_rate_value": 0.03,
            "term_months_value": 12,
            "operating_cost_per_loan_value": 10,
        }
    return None


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
    return registry.register_existing(
        path,
        task_id=task_id,
        role="strategy_sample",
    )


def _adopt_typed_strategy(tmp_path: Path, strategy_type: str):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name=f"{strategy_type} typed monitoring",
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
    # The adoption distribution is 75% default action and 25% matched action.
    # That gives the numeric strategies room to cross their emitted fail bands,
    # and gives segmentation a stable baseline share distribution for PSI.
    baseline = pd.DataFrame(
        {
            "x": [0, 0, 0, 1],
            "bad": [0, 0, 0, 1],
            "ead": [1000.0, 1000.0, 1000.0, 2000.0],
            "pd": [0.05, 0.05, 0.05, 0.20],
        }
    )
    dataset = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name=f"{strategy_type}-baseline",
        frame=baseline,
    )
    strategy = build_strategy_from_spec(_spec(strategy_type))
    strategies = StrategyRepository(settings.db_path)
    strategies.create_strategy(task.id, strategy)
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    backtest_inputs = {
        "dataset_id": dataset.id,
        "strategy_id": strategy.id,
        "target_col": "bad",
    }
    economics_inputs = _economics_inputs(strategy_type)
    if economics_inputs is not None:
        backtest_inputs["economics_inputs"] = economics_inputs
    backtest = tool_backtest_strategy(backtest_inputs, ctx)
    adopted = tool_adopt_strategy(
        {
            "strategy_id": strategy.id,
            "backtest_id": backtest["backtest_id"],
            "adoption_reason": "typed monitoring plan consumption test",
        },
        ctx,
    )
    plan_path = Path(
        next(
            artifact["path"]
            for artifact in adopted["artifacts"]
            if artifact["kind"] == "monitoring_plan_json"
        )
    )
    return settings, task, registry, strategy, ctx, plan_path


def _assert_immutable_plan_run_and_audited(
    *,
    settings,
    strategy_id: str,
    plan_path: Path,
    plan_bytes_before: bytes,
    output: dict,
    expected_plan_source: str = "ledger",
) -> None:
    assert output["plan_updated"] is False
    assert output["plan_source"] == expected_plan_source
    assert plan_path.read_bytes() == plan_bytes_before
    assert load_monitoring_plan(plan_path).last_run_at is None

    ledger = StrategyMonitoringRepository(settings.db_path)
    plan = ledger.get_plan(output["monitoring_plan_id"])
    assert plan is not None
    assert plan.strategy_id == strategy_id
    assert plan.revision == output["monitoring_plan_revision"]
    assert plan.payload_hash == output["monitoring_plan_hash"]
    run = ledger.get_run(output["monitoring_run_id"])
    assert run is not None
    assert run.strategy_id == strategy_id
    assert run.monitoring_plan_id == plan.id
    assert run.result_hash == output["monitoring_evidence"]["result_hash"]
    assert run.dataset_content_hash == output["monitoring_evidence"][
        "dataset_content_hash"
    ]
    assert len(ledger.list_runs(strategy_id)) == 1

    audit = _list_audit_rows(
        settings.db_path,
        kind="strategy.monitor",
        target_ref=strategy_id,
    )
    assert len(audit) == 1
    assert audit[0]["detail"]["overall_level"] == output["overall_level"]
    assert audit[0]["detail"]["monitoring_plan_id"] == plan.id
    assert audit[0]["detail"]["monitoring_run_id"] == run.id


def test_limit_monitoring_recomputes_bound_economics_from_fresh_columns(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "limit"
    )
    plan_before = json.loads(plan_path.read_text(encoding="utf-8"))
    assert set(plan_before["thresholds"]) == {"mean_limit", "expected_loss"}
    plan_bytes_before = plan_path.read_bytes()

    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="limit-fresh",
        frame=pd.DataFrame(
            {
                "x": [1, 1, 1, 1],
                "pd": [0.10, 0.10, 0.10, 0.10],
            }
        ),
    )
    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": fresh.id},
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    assert checks["mean_limit"]["value"] == pytest.approx(2000.0)
    assert checks["mean_limit"]["level"] == "red"
    assert checks["mean_limit"]["fail"] == pytest.approx(
        plan_before["thresholds"]["mean_limit"]["fail"]
    )
    assert checks["expected_loss"]["value"] == pytest.approx(240.0)
    assert checks["expected_loss"]["level"] != "n/a"
    assert output["metrics"]["expected_ead"] == pytest.approx(4800.0)
    assert output["metrics"]["expected_loss"] == pytest.approx(240.0)
    assert output["economics"] == {
        "expected_ead": pytest.approx(4800.0),
        "expected_loss": pytest.approx(240.0),
    }
    assert output["overall_level"] == "red"
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
    )


def test_pricing_monitoring_recomputes_bound_economics_from_fresh_columns(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "pricing"
    )
    plan_before = json.loads(plan_path.read_text(encoding="utf-8"))
    assert set(plan_before["thresholds"]) == {
        "mean_rate",
        "expected_loss",
        "profit",
        "roa",
    }
    plan_bytes_before = plan_path.read_bytes()

    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="pricing-fresh",
        frame=pd.DataFrame(
            {
                "x": [1, 1, 1, 1],
                "ead": [1000.0, 1000.0, 1000.0, 1000.0],
                "pd": [0.10, 0.10, 0.10, 0.10],
            }
        ),
    )
    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": fresh.id},
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    assert checks["mean_rate"]["value"] == pytest.approx(0.20)
    assert checks["mean_rate"]["level"] == "red"
    assert checks["mean_rate"]["warn"] == pytest.approx(
        plan_before["thresholds"]["mean_rate"]["warn"]
    )
    assert checks["expected_loss"]["value"] == pytest.approx(200.0)
    assert checks["profit"]["value"] == pytest.approx(440.0)
    assert checks["roa"]["value"] == pytest.approx(0.11)
    assert output["economics"]["expected_loss"] == pytest.approx(200.0)
    assert output["economics"]["profit"] == pytest.approx(440.0)
    assert output["economics"]["roa"] == pytest.approx(0.11)
    assert "by_row" not in output["economics"]
    assert output["overall_level"] == "red"
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
    )


def test_segmentation_monitoring_consumes_bad_rate_and_share_psi_plan(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "segmentation"
    )
    plan_before = json.loads(plan_path.read_text(encoding="utf-8"))
    assert set(plan_before["thresholds"]) == {
        "overall_bad_rate",
        "segment_share_psi",
    }
    assert [row["share"] for row in plan_before["expectation_baseline"]["breakdown"]] == [
        0.75,
        0.25,
    ]
    plan_bytes_before = plan_path.read_bytes()

    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="segmentation-fresh",
        frame=pd.DataFrame(
            {
                "x": [1, 1, 1, 1],
                "bad": [1, 1, 0, 0],
            }
        ),
    )
    output = tool_run_strategy_monitoring(
        {
            "strategy_id": strategy.id,
            "dataset_id": fresh.id,
            "target_col": "bad",
        },
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    assert checks["overall_bad_rate"]["value"] == pytest.approx(0.5)
    assert checks["overall_bad_rate"]["level"] == "red"
    assert checks["segment_share_psi"]["value"] > 0.25
    assert checks["segment_share_psi"]["level"] == "red"
    assert checks["segment_share_psi"]["fail"] == pytest.approx(
        plan_before["thresholds"]["segment_share_psi"]["fail"]
    )
    assert output["overall_level"] == "red"
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
    )


def test_segmentation_monitoring_without_target_keeps_bad_rate_na(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "segmentation"
    )
    plan_bytes_before = plan_path.read_bytes()
    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="segmentation-unlabeled-fresh",
        frame=pd.DataFrame({"x": [0, 0, 0, 1]}),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": fresh.id},
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    assert checks["overall_bad_rate"]["value"] is None
    assert checks["overall_bad_rate"]["level"] == "n/a"
    assert checks["segment_share_psi"]["value"] == pytest.approx(0.0)
    assert checks["segment_share_psi"]["level"] == "green"
    assert output["overall_level"] == "green"
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
    )


def test_typed_monitoring_rejects_a_dataset_owned_by_another_task(
    tmp_path: Path,
) -> None:
    settings, _task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "limit"
    )
    plan_bytes_before = plan_path.read_bytes()
    foreign_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="foreign monitoring sample",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
        )
    )
    foreign_dataset = _register(
        registry,
        tmp_path,
        task_id=foreign_task.id,
        name="foreign-limit-fresh",
        frame=pd.DataFrame({"x": [1, 1]}),
    )

    with pytest.raises(StrategyError, match="dataset not found"):
        tool_run_strategy_monitoring(
            {
                "strategy_id": strategy.id,
                "dataset_id": foreign_dataset.id,
            },
            ctx,
        )

    assert plan_path.read_bytes() == plan_bytes_before
    assert load_monitoring_plan(plan_path).last_run_at is None
    assert StrategyMonitoringRepository(settings.db_path).list_runs(strategy.id) == []
    assert not _list_audit_rows(
        settings.db_path,
        kind="strategy.monitor",
        target_ref=strategy.id,
    )


def test_approval_monitoring_uses_versioned_plan_thresholds(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "approval"
    )
    ledger = StrategyMonitoringRepository(settings.db_path)
    current = ledger.latest_plan(strategy.id)
    assert current is not None
    thresholds = {
        name: dict(spec) for name, spec in current.plan.thresholds.items()
    }
    thresholds["approval_rate"].update({"warn": -1.0, "fail": -2.0})
    revision = ledger.create_plan(
        replace(
            current.plan,
            monitoring_plan_id=None,
            revision=2,
            supersedes_plan_id=current.id,
            thresholds=thresholds,
        ),
        expected_revision=current.revision,
        expected_payload_hash=current.payload_hash,
    )
    assert revision.revision == 2
    plan_bytes_before = plan_path.read_bytes()
    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="approval-custom-threshold",
        frame=pd.DataFrame({"x": [1, 1], "bad": [0, 1]}),
    )

    output = tool_run_strategy_monitoring(
        {
            "strategy_id": strategy.id,
            "dataset_id": fresh.id,
            "target_col": "bad",
        },
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    approval = checks["approval_rate_drift"]
    assert approval["actual"] == pytest.approx(0.0)
    assert approval["warn"] == pytest.approx(-1.0)
    assert approval["fail"] == pytest.approx(-2.0)
    assert approval["level"] == "green"
    assert checks["approved_bad_rate_drift"]["level"] == "n/a"
    assert output["overall_level"] == "green"
    assert output["monitoring_plan_id"] == revision.id
    assert output["monitoring_plan_revision"] == 2
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
    )


def test_reject_monitoring_executes_capture_and_good_reject_thresholds(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "reject"
    )
    plan_bytes_before = plan_path.read_bytes()
    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="reject-fresh",
        frame=pd.DataFrame(
            {
                "x": [1, 1, 0, 0],
                "bad": [1, 0, 1, 0],
            }
        ),
    )

    output = tool_run_strategy_monitoring(
        {
            "strategy_id": strategy.id,
            "dataset_id": fresh.id,
            "target_col": "bad",
        },
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    assert checks["bad_capture_rate"]["value"] == pytest.approx(0.5)
    assert checks["good_reject_rate"]["value"] == pytest.approx(0.5)
    assert checks["bad_capture_rate"]["level"] == "red"
    assert checks["good_reject_rate"]["level"] == "red"
    assert output["overall_level"] == "red"
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
    )


@pytest.mark.parametrize(
    ("strategy_type", "missing_column"),
    (("limit", "pd"), ("pricing", "ead")),
)
def test_v2_typed_monitoring_missing_economics_column_fails_closed(
    tmp_path: Path,
    strategy_type: str,
    missing_column: str,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, strategy_type
    )
    plan_bytes_before = plan_path.read_bytes()
    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name=f"{strategy_type}-missing-economics-column",
        frame=pd.DataFrame({"x": [1, 1]}),
    )

    with pytest.raises(
        StrategyError,
        match=rf"missing economics column: {missing_column}",
    ):
        tool_run_strategy_monitoring(
            {"strategy_id": strategy.id, "dataset_id": fresh.id},
            ctx,
        )

    ledger = StrategyMonitoringRepository(settings.db_path)
    assert len(ledger.list_plans(strategy.id)) == 1
    assert ledger.list_runs(strategy.id) == []
    assert plan_path.read_bytes() == plan_bytes_before
    assert not _list_audit_rows(
        settings.db_path,
        kind="strategy.monitor",
        target_ref=strategy.id,
    )


def test_explicit_v1_plan_without_economics_keeps_economic_metrics_na(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "pricing"
    )
    ledger = StrategyMonitoringRepository(settings.db_path)
    adopted_plan = ledger.latest_plan(strategy.id)
    assert adopted_plan is not None
    with connect(settings.db_path) as conn:
        conn.execute(
            "DELETE FROM strategy_monitoring_plans WHERE id = ?",
            (adopted_plan.id,),
        )
    legacy_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    legacy_payload.update(
        {
            "plan_version": 1,
            "monitoring_plan_id": None,
            "revision": 1,
            "supersedes_plan_id": None,
            "last_run_at": None,
            "economics_bindings": {},
        }
    )
    save_monitoring_plan(plan_path, legacy_payload)
    plan_bytes_before = plan_path.read_bytes()

    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="pricing-explicit-v1",
        frame=pd.DataFrame({"x": [1, 1]}),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": fresh.id},
        ctx,
    )

    checks = {check["id"]: check for check in output["checks"]}
    assert checks["mean_rate"]["value"] == pytest.approx(0.20)
    for metric in ("expected_loss", "profit", "roa"):
        assert checks[metric]["value"] is None
        assert checks[metric]["level"] == "n/a"
    assert output["economics"] == {}
    assert output["plan_source"] == "legacy_v1"
    imported = ledger.latest_plan(strategy.id)
    assert imported is not None
    assert imported.plan.plan_version == 1
    _assert_immutable_plan_run_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        plan_bytes_before=plan_bytes_before,
        output=output,
        expected_plan_source="legacy_v1",
    )

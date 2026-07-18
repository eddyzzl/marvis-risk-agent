from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.monitor_tools import tool_run_strategy_monitoring
from marvis.packs.strategy.monitoring_plan import load_monitoring_plan
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.tools import tool_adopt_strategy, tool_backtest_strategy
from marvis.plugins.contracts import ToolContext
from marvis.repositories.audit import _list_audit_rows
from marvis.repositories.strategy import StrategyRepository
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


def _assert_plan_updated_and_audited(
    *,
    settings,
    strategy_id: str,
    plan_path: Path,
    output: dict,
) -> None:
    assert output["plan_updated"] is True
    assert load_monitoring_plan(plan_path).last_run_at == output["last_run_at"]
    audit = _list_audit_rows(
        settings.db_path,
        kind="strategy.monitor",
        target_ref=strategy_id,
    )
    assert len(audit) == 1
    assert audit[0]["detail"]["overall_level"] == output["overall_level"]


def test_limit_monitoring_consumes_mean_limit_plan_and_marks_economics_na(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "limit"
    )
    plan_before = json.loads(plan_path.read_text(encoding="utf-8"))
    assert set(plan_before["thresholds"]) == {"mean_limit", "expected_loss"}

    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="limit-fresh",
        frame=pd.DataFrame({"x": [1, 1, 1, 1]}),
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
    assert checks["expected_loss"]["value"] is None
    assert checks["expected_loss"]["level"] == "n/a"
    assert "未提供" in checks["expected_loss"]["message"]
    assert output["overall_level"] == "red"
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        output=output,
    )


def test_pricing_monitoring_consumes_mean_rate_plan_and_marks_economics_na(
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

    fresh = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="pricing-fresh",
        frame=pd.DataFrame({"x": [1, 1, 1, 1]}),
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
    for metric in ("expected_loss", "profit", "roa"):
        assert checks[metric]["value"] is None
        assert checks[metric]["level"] == "n/a"
        assert "未提供" in checks[metric]["message"]
    assert output["overall_level"] == "red"
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
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
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        output=output,
    )


def test_segmentation_monitoring_without_target_keeps_bad_rate_na(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "segmentation"
    )
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
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        output=output,
    )


def test_typed_monitoring_rejects_a_dataset_owned_by_another_task(
    tmp_path: Path,
) -> None:
    settings, _task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "limit"
    )
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

    assert load_monitoring_plan(plan_path).last_run_at is None
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
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["thresholds"]["approval_rate"].update(
        {"warn": -1.0, "fail": -2.0}
    )
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
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
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        output=output,
    )


def test_reject_monitoring_executes_capture_and_good_reject_thresholds(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "reject"
    )
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
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        output=output,
    )


def test_all_unavailable_monitoring_metrics_are_not_reported_green(
    tmp_path: Path,
) -> None:
    settings, task, registry, strategy, ctx, plan_path = _adopt_typed_strategy(
        tmp_path, "pricing"
    )
    empty = _register(
        registry,
        tmp_path,
        task_id=task.id,
        name="pricing-empty",
        frame=pd.DataFrame({"x": pd.Series(dtype="float64")}),
    )

    output = tool_run_strategy_monitoring(
        {"strategy_id": strategy.id, "dataset_id": empty.id},
        ctx,
    )

    assert output["checks"]
    assert {check["level"] for check in output["checks"]} == {"n/a"}
    assert output["overall_level"] == "n/a"
    _assert_plan_updated_and_audited(
        settings=settings,
        strategy_id=strategy.id,
        plan_path=plan_path,
        output=output,
    )

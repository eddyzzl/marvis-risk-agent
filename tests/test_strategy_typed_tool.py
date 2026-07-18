from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.tools import tool_backtest_strategy
from marvis.packs.strategy.typed_backtest import StrategyBacktestResult
from marvis.plugins.contracts import ToolContext
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings


def _runtime_fixture(tmp_path, strategy_type: str):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name=f"{strategy_type} typed backtest",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "bad": [0, 1, None],
            "ead": [1000.0, 2000.0, 1500.0],
            "pd": [0.1, 0.2, 0.15],
        }
    )
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(
        path,
        task_id=task.id,
        role="strategy_sample",
    )
    strategy = build_strategy_from_spec(_spec(strategy_type))
    StrategyRepository(settings.db_path).create_strategy(task.id, strategy)
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    return settings, task, dataset, strategy, ctx


def _spec(strategy_type: str) -> dict:
    action_by_type = {
        "approval": ({"type": "approval"}, {"type": "reject"}),
        "reject": ({"type": "approval"}, {"type": "reject"}),
        "limit": ({"type": "limit", "value": 1000}, {"type": "limit", "value": 2000}),
        "pricing": (
            {"type": "pricing", "value": 0.1},
            {"type": "pricing", "value": 0.2},
        ),
        "segmentation": (
            {"type": "segment", "value": "base"},
            {"type": "segment", "value": "high"},
        ),
    }
    default_action, matched_action = action_by_type[strategy_type]
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


def _economics(strategy_type: str) -> dict | None:
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


@pytest.mark.parametrize(
    ("strategy_type", "metric_name"),
    [
        ("approval", "approve_rate"),
        ("reject", "bad_capture_rate"),
        ("limit", "total_limit"),
        ("pricing", "mean_rate"),
        ("segmentation", "segment_count"),
    ],
)
def test_backtest_tool_executes_and_persists_all_strategy_types(
    tmp_path,
    strategy_type: str,
    metric_name: str,
) -> None:
    settings, task, dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        strategy_type,
    )
    inputs = {
        "dataset_id": dataset.id,
        "strategy_id": strategy.id,
        "target_col": "bad",
        "drop_nan_labels": True,
    }
    economics = _economics(strategy_type)
    if economics is not None:
        inputs["economics_inputs"] = economics

    output = tool_backtest_strategy(inputs, ctx)

    assert output["schema_version"] == "strategy.backtest.v2"
    assert output["strategy_type"] == strategy_type
    assert output["population_count"] == 3
    assert output["labeled_count"] == 2
    assert output["label_coverage"] == pytest.approx(2 / 3)
    assert output["nan_labels_dropped"] == 1
    assert metric_name in output["metrics"]
    if strategy_type in {"approval", "reject"}:
        assert output["approval_rate"] == output["metrics"]["approve_rate"]
        assert output["expected_profit"] == 0.0
        assert output["economics"] == {}
    else:
        assert "approval_rate" not in output
        assert "expected_profit" not in output

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    schema = next(
        tool.output_schema for tool in manifest.tools if tool.name == "backtest_strategy"
    )
    validate_against_schema(output, schema, label=f"{strategy_type} typed backtest")
    invalid_metrics = dict(output)
    invalid_metrics["metrics"] = {**output["metrics"], "unexpected_metric": 1}
    with pytest.raises(SchemaValidationError):
        validate_against_schema(
            invalid_metrics,
            schema,
            label=f"{strategy_type} invalid metrics",
        )
    if strategy_type == "approval":
        invalid_reject_metrics = dict(output)
        invalid_reject_metrics["metrics"] = {
            **output["metrics"],
            "bad_capture_rate": 0.5,
            "good_reject_rate": 0.1,
        }
        with pytest.raises(SchemaValidationError):
            validate_against_schema(
                invalid_reject_metrics,
                schema,
                label="approval with reject metrics",
            )
    if strategy_type not in {"approval", "reject"}:
        invalid_alias = {**output, "approved_bad_rate": 0.1}
        with pytest.raises(SchemaValidationError):
            validate_against_schema(
                invalid_alias,
                schema,
                label=f"{strategy_type} approval alias",
            )
    stored = StrategyRepository(settings.db_path).get_backtest(output["backtest_id"])
    assert isinstance(stored, StrategyBacktestResult)
    assert stored.to_dict() == {
        key: output[key]
        for key in (
            "schema_version",
            "strategy_id",
            "strategy_type",
            "population_count",
            "labeled_count",
            "label_coverage",
            "metrics",
            "breakdown",
            "transitions",
            "economics",
            "warnings",
            "normalized_input",
        )
    }
    audit = next(
        row
        for row in PluginRepository(settings.db_path).list_audit(
            kind="strategy.backtest"
        )
        if row["target_ref"] == output["backtest_id"]
    )
    summary_key = {
        "approval": "approve_rate",
        "reject": "approve_rate",
        "limit": "total_limit",
        "pricing": "mean_rate",
        "segmentation": "segment_count",
    }[strategy_type]
    assert audit["detail"][summary_key] == output["metrics"][summary_key]


def test_backtest_manifest_rejects_partial_or_ambiguous_economics_inputs() -> None:
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    schema = next(
        tool.input_schema for tool in manifest.tools if tool.name == "backtest_strategy"
    )
    base = {
        "dataset_id": "dataset-1",
        "strategy_id": "strategy-1",
        "target_col": "bad",
    }

    for economics_inputs in (
        {},
        {"pd_value": 0.1, "lgd_value": 0.5},
        {
            "pd_col": "pd",
            "pd_value": 0.1,
            "lgd_value": 0.5,
            "utilization_value": 0.8,
        },
    ):
        with pytest.raises(SchemaValidationError):
            validate_against_schema(
                {**base, "economics_inputs": economics_inputs},
                schema,
                label="invalid economics inputs",
            )


def test_backtest_tool_rejects_dataset_owned_by_another_task(tmp_path) -> None:
    settings, task, _dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    foreign_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="foreign strategy data",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
        )
    )
    foreign_path = tmp_path / "foreign.parquet"
    pd.DataFrame({"x": [0, 1], "bad": [0, 1]}).to_parquet(
        foreign_path,
        index=False,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    foreign_dataset = registry.register_existing(
        foreign_path,
        task_id=foreign_task.id,
        role="strategy_sample",
    )

    with pytest.raises(StrategyError, match="dataset not found"):
        tool_backtest_strategy(
            {
                "dataset_id": foreign_dataset.id,
                "strategy_id": strategy.id,
                "target_col": "bad",
                "drop_nan_labels": True,
            },
            ctx,
        )

    assert task.id != foreign_task.id
    assert StrategyRepository(settings.db_path).list_backtests(strategy.id) == []

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import marvis.packs.strategy.tools as strategy_tools
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.tools import (
    tool_backtest_strategy,
    tool_materialize_sample_design,
)
from marvis.packs.strategy.typed_backtest import StrategyBacktestResult
from marvis.plugins.contracts import ToolContext
from marvis.plugins.errors import SchemaValidationError
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _runtime_fixture(
    tmp_path,
    strategy_type: str,
    *,
    target_bad_value: int = 1,
    with_split: bool = False,
):
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
    target = [0, 1, None]
    if target_bad_value == 0:
        target = [1 if value == 0 else 0 if value == 1 else None for value in target]
    frame_data = {
        "x": [0, 1, 2],
        "bad": target,
        "ead": [1000.0, 2000.0, 1500.0],
        "pd": [0.1, 0.2, 0.15],
    }
    if with_split:
        frame_data["sample_role"] = ["dev", "dev", "oot"]
    frame = pd.DataFrame(frame_data)
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
    workspace_repo = DataWorkspaceRepository(settings.db_path)
    activated = workspace_repo.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    workspace = workspace_repo.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={
                    "bad": "target",
                    **({"sample_role": "segment"} if with_split else {}),
                },
            ),
        ),
        expected_revision=activated.revision,
    )
    sample = tool_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            "target_col": "bad",
            "target_bad_value": target_bad_value,
            "performance_window_status": "provided",
            "performance_window_days": 90,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-06-30",
            "maturity_status": "confirmed_matured",
            "drop_nan_labels": True,
            **(
                {
                    "split_col": "sample_role",
                    "development_values": ["dev"],
                    "validation_values": [],
                    "oot_values": ["oot"],
                }
                if with_split
                else {}
            ),
        },
        ctx,
    )
    sample_design_ref = {
        "artifact_id": sample["artifact"]["artifact_id"],
        "artifact_content_hash": sample["artifact"]["content_hash"],
        "sample_design_id": sample["sample_design_id"],
        "sample_design_content_hash": sample["content_hash"],
        "partition": "development",
    }
    return settings, task, dataset, strategy, ctx, sample_design_ref


def test_backtest_tool_uses_development_partition_and_source_bad_polarity(
    tmp_path,
) -> None:
    first = _runtime_fixture(
        tmp_path / "bad-one",
        "reject",
        target_bad_value=1,
        with_split=True,
    )
    second = _runtime_fixture(
        tmp_path / "bad-zero",
        "reject",
        target_bad_value=0,
        with_split=True,
    )

    outputs = []
    for _settings, _task, dataset, strategy, ctx, sample_ref in (first, second):
        outputs.append(
            tool_backtest_strategy(
                {
                    "dataset_id": dataset.id,
                    "strategy_id": strategy.id,
                    "target_col": "bad",
                    "sample_design_ref": sample_ref,
                    "drop_nan_labels": True,
                },
                ctx,
            )
        )

    assert outputs[0]["population_count"] == 2
    assert outputs[1]["population_count"] == 2
    assert outputs[0]["metrics"] == outputs[1]["metrics"]
    assert outputs[0]["breakdown"] == outputs[1]["breakdown"]
    assert outputs[0]["normalized_input"]["target_encoding"] == {
        "good": 0,
        "bad": 1,
    }
    assert outputs[1]["normalized_input"]["target_encoding"] == {
        "good": 1,
        "bad": 0,
    }


def test_backtest_writer_reauthenticates_sample_design_inside_transaction(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, task, dataset, strategy, ctx, sample_design_ref = _runtime_fixture(
        tmp_path,
        "approval",
    )
    original = (
        strategy_tools.require_strategy_sample_design_execution_binding_on_connection
    )

    def remove_sample_design_before_persistence(conn, binding):
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (task.id, sample_design_ref["artifact_id"]),
        )
        original(conn, binding)

    monkeypatch.setattr(
        strategy_tools,
        "require_strategy_sample_design_execution_binding_on_connection",
        remove_sample_design_before_persistence,
    )

    with pytest.raises(StrategyError, match="sample-design artifact disappeared"):
        strategy_tools.tool_backtest_strategy(
            {
                "dataset_id": dataset.id,
                "strategy_id": strategy.id,
                "target_col": "bad",
                "sample_design_ref": sample_design_ref,
                "drop_nan_labels": True,
            },
            ctx,
        )

    assert StrategyRepository(settings.db_path).list_backtests(strategy.id) == []
    assert PluginRepository(settings.db_path).list_audit(
        kind="strategy.backtest"
    ) == []
    # The injected deletion shares the writer transaction and must roll back
    # together with the blocked backtest/audit insert.
    assert any(
        record["id"] == sample_design_ref["artifact_id"]
        for record in TaskArtifactRepository(settings.db_path).list_for_task(task.id)
    )


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
    settings, task, dataset, strategy, ctx, sample_design_ref = _runtime_fixture(
        tmp_path,
        strategy_type,
    )
    inputs = {
        "dataset_id": dataset.id,
        "strategy_id": strategy.id,
        "target_col": "bad",
        "sample_design_ref": sample_design_ref,
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
        "sample_design_ref": {
            "artifact_id": "a" * 64,
            "artifact_content_hash": "b" * 64,
            "sample_design_id": "strategy-sample-design-1",
            "sample_design_content_hash": "c" * 64,
            "partition": "development",
        },
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
    settings, task, _dataset, strategy, ctx, sample_design_ref = _runtime_fixture(
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
                "sample_design_ref": sample_design_ref,
                "drop_nan_labels": True,
            },
            ctx,
        )

    assert task.id != foreign_task.id
    assert StrategyRepository(settings.db_path).list_backtests(strategy.id) == []

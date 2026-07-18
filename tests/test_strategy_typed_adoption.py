from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import (
    DatasetRepository,
    PluginRepository,
    TaskRepository,
    connect,
    init_db,
)
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy.contracts import BacktestResult
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.tools import tool_adopt_strategy, tool_backtest_strategy
from marvis.packs.strategy.typed_backtest import (
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings


_ADOPTION_EVIDENCE_VERSION = "strategy.adoption-evidence.v1"


def _spec(strategy_type: str) -> dict:
    actions = {
        "approval": ({"type": "approval"}, {"type": "reject"}),
        "reject": ({"type": "approval"}, {"type": "reject"}),
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
                "action": {
                    **matched_action,
                    "reason_code": "positive-value",
                },
            }
        ],
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0, 1, 2],
            "bad": [0, 1, 0],
            "ead": [1000.0, 2000.0, 1500.0],
            "pd": [0.1, 0.2, 0.15],
        }
    )


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


def _task(task_repo: TaskRepository, tmp_path: Path, name: str):
    return task_repo.create_task(
        TaskCreate(
            model_name=name,
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
        )
    )


def _register_frame(
    registry: DatasetRegistry,
    tmp_path: Path,
    *,
    task_id: str,
    name: str,
    frame: pd.DataFrame | None = None,
):
    path = tmp_path / f"{name}.parquet"
    (frame if frame is not None else _frame()).to_parquet(path, index=False)
    return registry.register_existing(
        path,
        task_id=task_id,
        role="strategy_sample",
    )


def _runtime_fixture(tmp_path: Path, strategy_type: str):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task_repo = TaskRepository(settings.db_path)
    task = _task(task_repo, tmp_path, f"{strategy_type} typed adoption")
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = _register_frame(
        registry,
        tmp_path,
        task_id=task.id,
        name=f"{strategy_type}-source",
    )
    strategy = build_strategy_from_spec(_spec(strategy_type))
    strategy_repo = StrategyRepository(settings.db_path)
    strategy_repo.create_strategy(task.id, strategy)
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    return settings, task_repo, task, registry, dataset, strategy, strategy_repo, ctx


def _backtest(dataset_id: str, strategy_id: str, strategy_type: str, ctx) -> dict:
    inputs = {
        "dataset_id": dataset_id,
        "strategy_id": strategy_id,
        "target_col": "bad",
    }
    economics = _economics(strategy_type)
    if economics is not None:
        inputs["economics_inputs"] = economics
    return tool_backtest_strategy(inputs, ctx)


def _legacy_backtest(strategy_id: str) -> BacktestResult:
    return BacktestResult(
        strategy_id=strategy_id,
        approval_rate=0.7,
        approved_count=70,
        approved_bad_rate=0.04,
        rejected_bad_rate=0.22,
        expected_profit=2300.0,
        swap_in_count=5,
        swap_out_count=8,
        swap_in_bad_rate=0.12,
        swap_out_bad_rate=0.01,
        by_segment=(),
        rejected_count=30,
    )


@pytest.mark.parametrize(
    (
        "strategy_type",
        "evidence_metric",
        "threshold_metric",
        "decision",
        "decision_value",
        "default_decision",
        "default_value",
    ),
    [
        (
            "approval",
            "approve_bad_rate",
            "approved_bad_rate",
            "reject",
            "",
            "approve",
            "",
        ),
        (
            "reject",
            "bad_capture_rate",
            "bad_capture_rate",
            "reject",
            "",
            "approve",
            "",
        ),
        ("limit", "mean_limit", "mean_limit", "limit", "2000", "limit", "1000"),
        ("pricing", "mean_rate", "mean_rate", "price", "0.2", "price", "0.1"),
        (
            "segmentation",
            "overall_bad_rate",
            "overall_bad_rate",
            "segment",
            "high",
            "segment",
            "base",
        ),
    ],
)
def test_adopt_strategy_emits_typed_evidence_and_deliverables_for_all_types(
    tmp_path: Path,
    strategy_type: str,
    evidence_metric: str,
    threshold_metric: str,
    decision: str,
    decision_value: str,
    default_decision: str,
    default_value: str,
) -> None:
    (
        settings,
        _task_repo,
        task,
        registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, strategy_type)
    backtest = _backtest(dataset.id, strategy.id, strategy_type, ctx)

    output = tool_adopt_strategy(
        {
            "strategy_id": strategy.id,
            "backtest_id": backtest["backtest_id"],
            "adoption_reason": f"risk committee approved {strategy_type} policy",
            "band_stats": {
                "bands": [
                    {
                        "lo": -999,
                        "hi": 999,
                        "pop_pct": 0.99,
                        "bad_rate": 0.99,
                        "expected_profit": 999999,
                    }
                ]
            },
        },
        ctx,
    )

    assert output["status"] == "adopted"
    assert output["strategy_type"] == strategy_type
    assert output["backtest_id"] == backtest["backtest_id"]
    evidence = output["adoption_evidence"]
    assert evidence["schema_version"] == _ADOPTION_EVIDENCE_VERSION
    assert evidence["backtest_schema_version"] == "strategy.backtest.v2"
    assert evidence["strategy_id"] == strategy.id
    assert evidence["strategy_type"] == strategy_type
    assert evidence["source_dataset_id"] == dataset.id
    assert evidence["source_dataset_content_hash"] == sha256_file(
        registry.resolve_path(dataset.id)
    )
    assert evidence["strategy_effect_hash"] == strategy_spec_hash(strategy.spec)
    assert evidence["baseline_effect_hash"] == backtest["normalized_input"][
        "baseline_effect_hash"
    ]
    assert evidence["target_col"] == "bad"
    assert evidence["population_count"] == 3
    assert evidence["labeled_count"] == 3
    assert evidence_metric in evidence["metrics"]
    assert evidence["transitions"] == backtest["transitions"]
    assert "by_row" not in evidence["economics"]
    assert evidence["economics_input_evidence"] == backtest["normalized_input"][
        "economics_input_evidence"
    ]

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    adopt_tool = next(tool for tool in manifest.tools if tool.name == "adopt_strategy")
    validate_against_schema(output, adopt_tool.output_schema, label="typed adoption")

    artifacts = {row["kind"]: Path(row["path"]) for row in output["artifacts"]}
    assert set(artifacts) == {"decision_table_csv", "monitoring_plan_json"}
    decision_rows = list(
        csv.DictReader(artifacts["decision_table_csv"].read_text(encoding="utf-8").splitlines())
    )
    assert len(decision_rows) == 2
    assert decision_rows[0]["决策"] == decision
    assert decision_rows[0]["取值"] == decision_value
    assert decision_rows[0]["样本占比"] == ""
    assert decision_rows[0]["坏率"] == ""
    assert decision_rows[0]["预期利润"] == ""
    assert decision_rows[1]["条件"] == "未命中任何规则（默认动作）"
    assert decision_rows[1]["决策"] == default_decision
    assert decision_rows[1]["取值"] == default_value

    plan = json.loads(
        artifacts["monitoring_plan_json"].read_text(encoding="utf-8")
    )
    baseline = plan["expectation_baseline"]
    assert baseline["strategy_type"] == strategy_type
    assert baseline["backtest_schema_version"] == "strategy.backtest.v2"
    assert baseline["strategy_effect_hash"] == strategy_spec_hash(strategy.spec)
    assert baseline["source_dataset_content_hash"] == evidence[
        "source_dataset_content_hash"
    ]
    assert baseline["baseline_effect_hash"] == evidence["baseline_effect_hash"]
    assert baseline["source_backtest_id"] == backtest["backtest_id"]
    assert baseline["metrics"][evidence_metric] == evidence["metrics"][evidence_metric]
    assert threshold_metric in plan["thresholds"]

    adopt_audits = PluginRepository(settings.db_path).list_audit(
        kind="strategy.adopt"
    )
    assert len(adopt_audits) == 1
    assert adopt_audits[0]["detail"]["strategy_type"] == strategy_type
    assert adopt_audits[0]["detail"]["adoption_evidence"] == evidence
    assert strategy_repo.get_strategy_meta(strategy.id)["status"] == "adopted"


@pytest.mark.parametrize("strategy_type", ["approval", "reject"])
def test_adopt_strategy_preserves_legacy_action_backtest_compatibility(
    tmp_path: Path,
    strategy_type: str,
) -> None:
    (
        _settings,
        _task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, strategy_type)
    strategy_repo.save_backtest(
        "legacy-action-backtest",
        strategy.id,
        dataset.id,
        _legacy_backtest(strategy.id),
    )

    output = tool_adopt_strategy(
        {
            "strategy_id": strategy.id,
            "backtest_id": "legacy-action-backtest",
            "adoption_reason": "committee preserves the validated V1 migration path",
        },
        ctx,
    )

    assert output["status"] == "adopted"
    assert output["strategy_type"] == strategy_type
    evidence = output["adoption_evidence"]
    assert evidence["backtest_schema_version"] == "strategy.backtest.v1"
    assert evidence["population_count"] is None
    assert evidence["labeled_count"] is None
    assert evidence["baseline_effect_hash"] is None
    assert evidence["transitions"] == []
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    schema = next(
        tool.output_schema for tool in manifest.tools if tool.name == "adopt_strategy"
    )
    validate_against_schema(output, schema, label="legacy action adoption")


def test_reject_adoption_does_not_require_an_approved_labeled_group(
    tmp_path: Path,
) -> None:
    (
        _settings,
        _task_repo,
        task,
        registry,
        _dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "reject")
    frame = _frame()
    frame.loc[0, "bad"] = None
    dataset = _register_frame(
        registry,
        tmp_path,
        task_id=task.id,
        name="reject-with-unlabeled-approved-group",
        frame=frame,
    )
    backtest = tool_backtest_strategy(
        {
            "dataset_id": dataset.id,
            "strategy_id": strategy.id,
            "target_col": "bad",
            "drop_nan_labels": True,
        },
        ctx,
    )
    assert backtest["metrics"]["approve_bad_rate"] is None
    assert backtest["metrics"]["bad_capture_rate"] == 1.0
    assert backtest["metrics"]["good_reject_rate"] == 1.0

    output = tool_adopt_strategy(
        {
            "strategy_id": strategy.id,
            "backtest_id": backtest["backtest_id"],
            "adoption_reason": "committee approved measurable reject-quality evidence",
        },
        ctx,
    )

    assert output["status"] == "adopted"
    assert output["adoption_evidence"]["metrics"]["bad_capture_rate"] == 1.0
    plan_path = next(
        Path(row["path"])
        for row in output["artifacts"]
        if row["kind"] == "monitoring_plan_json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert "approved_bad_rate" not in plan["thresholds"]
    assert plan["expectation_baseline"]["approved_bad_rate"] is None


@pytest.mark.parametrize("strategy_type", ["limit", "pricing", "segmentation"])
def test_non_action_adoption_rejects_legacy_flat_backtests(
    tmp_path: Path,
    strategy_type: str,
) -> None:
    (
        settings,
        _task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, strategy_type)
    strategy_repo.save_backtest(
        "legacy-wrong-type",
        strategy.id,
        dataset.id,
        _legacy_backtest(strategy.id),
    )

    with pytest.raises(StrategyError, match="typed StrategyBacktestResult"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "legacy-wrong-type",
                "adoption_reason": "committee rejects untyped evidence",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def test_segmentation_adoption_rejects_unlabeled_evidence(
    tmp_path: Path,
) -> None:
    (
        settings,
        _task_repo,
        task,
        registry,
        _dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "segmentation")
    frame = _frame()
    frame["bad"] = None
    dataset = _register_frame(
        registry,
        tmp_path,
        task_id=task.id,
        name="unlabeled-segmentation",
        frame=frame,
    )
    backtest = run_typed_backtest(
        frame,
        strategy.spec,
        target_col="bad",
        strategy_id=strategy.id,
    )
    assert backtest.labeled_count == 0
    strategy_repo.save_backtest(
        "unlabeled-segmentation",
        strategy.id,
        dataset.id,
        backtest,
    )

    with pytest.raises(StrategyError, match="without labeled"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "unlabeled-segmentation",
                "adoption_reason": "committee requires measurable segment risk",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


@pytest.mark.parametrize("strategy_type", ["limit", "pricing"])
def test_adopt_strategy_rejects_missing_required_economics_without_mutation(
    tmp_path: Path,
    strategy_type: str,
) -> None:
    (
        settings,
        _task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, strategy_type)
    backtest = tool_backtest_strategy(
        {
            "dataset_id": dataset.id,
            "strategy_id": strategy.id,
            "target_col": "bad",
        },
        ctx,
    )
    assert backtest["economics"] == {}

    with pytest.raises(StrategyError, match="complete .* economics evidence"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": backtest["backtest_id"],
                "adoption_reason": "committee requires complete economics evidence",
            },
            ctx,
        )

    assert strategy_repo.get_strategy_meta(strategy.id)["status"] == "draft"
    assert strategy_repo.list_strategy_artifacts(strategy.id) == []
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.adopt") == []


def test_adopt_strategy_rejects_typed_strategy_type_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    (
        settings,
        _task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "approval")
    mismatched = run_typed_backtest(
        _frame(),
        _spec("pricing"),
        target_col="bad",
        strategy_id=strategy.id,
    )
    strategy_repo.save_backtest(
        "typed-type-mismatch",
        strategy.id,
        dataset.id,
        mismatched,
    )

    with pytest.raises(StrategyError, match="strategy_type"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "typed-type-mismatch",
                "adoption_reason": "committee evidence must match the policy type",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def test_adopt_strategy_rejects_tampered_effect_hash_without_mutation(
    tmp_path: Path,
) -> None:
    (
        settings,
        _task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "approval")
    valid = run_typed_backtest(
        _frame(),
        strategy.spec,
        target_col="bad",
        strategy_id=strategy.id,
    )
    payload = valid.to_dict()
    payload["normalized_input"] = {
        **payload["normalized_input"],
        "strategy_effect_hash": "0" * 64,
    }
    tampered = StrategyBacktestResult.from_dict(payload)
    strategy_repo.save_backtest(
        "typed-effect-mismatch",
        strategy.id,
        dataset.id,
        tampered,
    )

    with pytest.raises(StrategyError, match="strategy effect hash"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "typed-effect-mismatch",
                "adoption_reason": "committee requires evidence for this exact policy",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def test_adopt_strategy_rejects_a_dataset_mutated_after_backtest(
    tmp_path: Path,
) -> None:
    (
        settings,
        _task_repo,
        _task_row,
        registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "approval")
    backtest = _backtest(dataset.id, strategy.id, "approval", ctx)
    mutated = _frame()
    mutated["bad"] = [1, 0, 1]
    mutated.to_parquet(registry.resolve_path(dataset.id), index=False)
    assert sha256_file(registry.resolve_path(dataset.id)) != backtest[
        "source_dataset_content_hash"
    ]

    with pytest.raises(StrategyError, match="content hash no longer matches"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": backtest["backtest_id"],
                "adoption_reason": "committee requires immutable source evidence",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def test_adopt_strategy_rejects_foreign_task_dataset_evidence_without_mutation(
    tmp_path: Path,
) -> None:
    (
        settings,
        task_repo,
        _task_row,
        registry,
        _dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "approval")
    foreign_task = _task(task_repo, tmp_path, "foreign strategy evidence")
    foreign_dataset = _register_frame(
        registry,
        tmp_path,
        task_id=foreign_task.id,
        name="foreign-source",
    )
    foreign_evidence = run_typed_backtest(
        _frame(),
        strategy.spec,
        target_col="bad",
        strategy_id=strategy.id,
    )
    strategy_repo.save_backtest(
        "typed-foreign-dataset",
        strategy.id,
        foreign_dataset.id,
        foreign_evidence,
    )

    with pytest.raises(StrategyError, match="same task"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "typed-foreign-dataset",
                "adoption_reason": "committee requires task-bound evidence",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def test_legacy_adoption_rejects_a_registered_foreign_task_dataset(
    tmp_path: Path,
) -> None:
    (
        settings,
        task_repo,
        _task_row,
        registry,
        _dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "approval")
    foreign_task = _task(task_repo, tmp_path, "foreign legacy evidence")
    foreign_dataset = _register_frame(
        registry,
        tmp_path,
        task_id=foreign_task.id,
        name="foreign-legacy-source",
    )
    strategy_repo.save_backtest(
        "legacy-foreign-dataset",
        strategy.id,
        foreign_dataset.id,
        _legacy_backtest(strategy.id),
    )

    with pytest.raises(StrategyError, match="same task"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "legacy-foreign-dataset",
                "adoption_reason": "committee requires task-bound legacy evidence",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def test_adoption_rejects_a_foreign_monitoring_experiment(
    tmp_path: Path,
) -> None:
    (
        settings,
        task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "approval")
    foreign_task = _task(task_repo, tmp_path, "foreign monitoring experiment")
    with connect(settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiments(
                id, task_id, recipe_id, config_json, metrics_json,
                artifact_id, status, created_at
            ) VALUES (?, ?, ?, '{}', NULL, NULL, 'trained', ?)
            """,
            (
                "foreign-experiment",
                foreign_task.id,
                "recipe-foreign",
                "2026-07-18T00:00:00+00:00",
            ),
        )
    backtest = _backtest(dataset.id, strategy.id, "approval", ctx)

    with pytest.raises(StrategyError, match="same task"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": backtest["backtest_id"],
                "adoption_reason": "committee requires a task-bound monitor",
                "experiment_id": "foreign-experiment",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)


def _assert_adoption_rejected_without_mutation(
    settings,
    strategy_repo: StrategyRepository,
    strategy_id: str,
) -> None:
    assert strategy_repo.get_strategy_meta(strategy_id)["status"] == "draft"
    assert strategy_repo.list_strategy_artifacts(strategy_id) == []
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.adopt") == []

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
from marvis.packs.strategy.monitoring_plan import canonical_monitoring_plan_hash
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
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository
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

    artifact_outputs = {row["kind"]: row for row in output["artifacts"]}
    artifacts = {
        kind: Path(row["path"]) for kind, row in artifact_outputs.items()
    }
    assert set(artifacts) == {"decision_table_csv", "monitoring_plan_json"}
    registered_artifacts = {
        row["kind"]: row
        for row in strategy_repo.list_strategy_artifacts(strategy.id)
    }
    assert set(registered_artifacts) == set(artifacts)
    expected_producers = {
        "decision_table_csv": "strategy.adopt.decision_table.v1",
        "monitoring_plan_json": "strategy.adopt.monitoring_plan.v1",
    }
    for kind, path in artifacts.items():
        output_artifact = artifact_outputs[kind]
        registered = registered_artifacts[kind]
        assert output_artifact["artifact_id"] == registered["id"]
        assert output_artifact["content_hash"] == sha256_file(path)
        assert output_artifact["content_size"] == path.stat().st_size
        assert registered["integrity_status"] == "verified"
        assert registered["content_hash"] == output_artifact["content_hash"]
        assert registered["content_size"] == output_artifact["content_size"]
        provenance = registered["provenance"]
        assert provenance["schema_version"] == "strategy-artifact-provenance.v1"
        assert provenance["producer_version"] == expected_producers[kind]
        assert provenance["task_id"] == task.id
        assert provenance["strategy_id"] == strategy.id
        assert provenance["kind"] == kind
        assert provenance["evidence"]["operation"] == "strategy.adopt"
        assert provenance["evidence"]["backtest_id"] == backtest["backtest_id"]
        assert provenance["evidence"]["strategy_effect_hash"] == evidence[
            "strategy_effect_hash"
        ]
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
    assert plan["plan_version"] == 2
    assert plan["monitoring_plan_id"] == output["monitoring_plan_id"]
    assert plan["revision"] == output["monitoring_plan_revision"] == 1
    assert plan["supersedes_plan_id"] is None
    assert plan["last_run_at"] is None
    assert canonical_monitoring_plan_hash(plan) == output["monitoring_plan_hash"]
    plan_provenance = registered_artifacts["monitoring_plan_json"]["provenance"]
    assert plan_provenance["evidence"]["monitoring_plan_id"] == output[
        "monitoring_plan_id"
    ]
    assert plan_provenance["evidence"]["monitoring_plan_revision"] == 1
    assert plan_provenance["evidence"]["monitoring_plan_hash"] == output[
        "monitoring_plan_hash"
    ]
    expected_bindings = {
        "limit": {
            "lgd": {"kind": "scalar", "value": 0.5},
            "pd": {"kind": "column", "column": "pd"},
            "utilization": {"kind": "scalar", "value": 0.6},
        },
        "pricing": {
            "ead": {"kind": "column", "column": "ead"},
            "funding_rate": {"kind": "scalar", "value": 0.03},
            "lgd": {"kind": "scalar", "value": 0.5},
            "operating_cost_per_loan": {"kind": "scalar", "value": 10.0},
            "pd": {"kind": "column", "column": "pd"},
            "term_months": {"kind": "scalar", "value": 12.0},
        },
    }.get(strategy_type, {})
    assert plan["economics_bindings"] == expected_bindings
    assert all(
        "content_hash" not in binding and "row_count" not in binding
        for binding in plan["economics_bindings"].values()
    )
    plan_record = StrategyMonitoringRepository(settings.db_path).get_plan(
        output["monitoring_plan_id"]
    )
    assert plan_record is not None
    assert plan_record.plan.to_dict() == plan
    assert plan_record.payload_hash == output["monitoring_plan_hash"]
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
    assert adopt_audits[0]["detail"]["monitoring_plan_id"] == plan_record.id
    assert adopt_audits[0]["detail"]["monitoring_plan_revision"] == 1
    assert adopt_audits[0]["detail"]["monitoring_plan_hash"] == plan_record.payload_hash
    assert strategy_repo.get_strategy_meta(strategy.id)["status"] == "adopted"


@pytest.mark.parametrize("strategy_type", ["approval", "reject"])
def test_adopt_strategy_preserves_legacy_action_backtest_compatibility(
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
    plan_record = StrategyMonitoringRepository(settings.db_path).latest_plan(
        strategy.id
    )
    assert plan_record is not None
    assert plan_record.revision == 1
    assert plan_record.plan.plan_version == 2
    assert plan_record.plan.last_run_at is None
    assert plan_record.plan.economics_bindings == {}
    assert output["monitoring_plan_id"] == plan_record.id
    assert output["monitoring_plan_hash"] == plan_record.payload_hash
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


def test_limit_adoption_ledgers_an_all_scalar_economics_binding(
    tmp_path: Path,
) -> None:
    (
        settings,
        _task_repo,
        _task_row,
        _registry,
        dataset,
        strategy,
        _strategy_repo,
        ctx,
    ) = _runtime_fixture(tmp_path, "limit")
    backtest = tool_backtest_strategy(
        {
            "dataset_id": dataset.id,
            "strategy_id": strategy.id,
            "target_col": "bad",
            "economics_inputs": {
                "pd_value": 0.2,
                "lgd_value": 0.5,
                "utilization_value": 0.6,
            },
        },
        ctx,
    )

    output = tool_adopt_strategy(
        {
            "strategy_id": strategy.id,
            "backtest_id": backtest["backtest_id"],
            "adoption_reason": "committee approved scalar economics assumptions",
        },
        ctx,
    )

    plan = StrategyMonitoringRepository(settings.db_path).get_plan(
        output["monitoring_plan_id"]
    )
    assert plan is not None
    assert plan.plan.economics_bindings == {
        "lgd": {"kind": "scalar", "value": 0.5},
        "pd": {"kind": "scalar", "value": 0.2},
        "utilization": {"kind": "scalar", "value": 0.6},
    }


def test_adoption_rejects_unnamed_required_economics_series_without_mutation(
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
    ) = _runtime_fixture(tmp_path, "limit")
    result = run_typed_backtest(
        _frame(),
        strategy.spec,
        target_col="bad",
        strategy_id=strategy.id,
        economics_inputs={
            "pd": pd.Series([0.1, 0.2, 0.15]),
            "lgd": 0.5,
            "utilization": 0.6,
        },
    )
    assert result.normalized_input["economics_input_evidence"]["pd"]["name"] is None
    strategy_repo.save_backtest_with_audit(
        "unnamed-economics-series",
        strategy.id,
        dataset.id,
        result,
        audit={
            "kind": "strategy.backtest",
            "target_ref": "unnamed-economics-series",
            "outcome": "succeeded",
            "detail": {
                "task_id": str(ctx.task_id),
                "source_dataset_content_hash": sha256_file(
                    registry.resolve_path(dataset.id)
                ),
            },
        },
    )

    with pytest.raises(StrategyError, match="stable column name"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": "unnamed-economics-series",
                "adoption_reason": "committee requires reproducible monitoring inputs",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)
    assert StrategyMonitoringRepository(settings.db_path).latest_plan(strategy.id) is None
    strategy_dir = settings.tasks_dir / str(ctx.task_id) / "strategy"
    assert not list(strategy_dir.glob("decision_table_*.csv"))
    assert not list(strategy_dir.glob("monitoring_plan_*.json"))


def test_adoption_rolls_back_files_lifecycle_audit_and_effect_when_plan_ledger_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    backtest = _backtest(dataset.id, strategy.id, "approval", ctx)

    def fail_plan_write(*_args, **_kwargs):
        raise RuntimeError("injected monitoring ledger failure")

    monkeypatch.setattr(
        StrategyMonitoringRepository,
        "create_plan_on_connection",
        fail_plan_write,
    )

    with pytest.raises(RuntimeError, match="injected monitoring ledger failure"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": backtest["backtest_id"],
                "adoption_reason": "committee requires one atomic adoption receipt",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)
    assert StrategyMonitoringRepository(settings.db_path).latest_plan(strategy.id) is None
    strategy_dir = settings.tasks_dir / str(ctx.task_id) / "strategy"
    assert not list(strategy_dir.glob("decision_table_*.csv"))
    assert not list(strategy_dir.glob("monitoring_plan_*.json"))


def test_adoption_rolls_back_first_verified_artifact_when_second_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    backtest = _backtest(dataset.id, strategy.id, "approval", ctx)
    original_register = (
        StrategyRepository.register_verified_strategy_artifact_with_audit_on_connection
    )

    def fail_second_registration(self, conn, strategy_id, *, kind, **kwargs):
        if kind == "monitoring_plan_json":
            raise RuntimeError("injected verified artifact registration failure")
        return original_register(
            self,
            conn,
            strategy_id,
            kind=kind,
            **kwargs,
        )

    monkeypatch.setattr(
        StrategyRepository,
        "register_verified_strategy_artifact_with_audit_on_connection",
        fail_second_registration,
    )

    with pytest.raises(RuntimeError, match="verified artifact registration"):
        tool_adopt_strategy(
            {
                "strategy_id": strategy.id,
                "backtest_id": backtest["backtest_id"],
                "adoption_reason": "committee requires atomic verified deliverables",
            },
            ctx,
        )

    _assert_adoption_rejected_without_mutation(settings, strategy_repo, strategy.id)
    assert StrategyMonitoringRepository(settings.db_path).latest_plan(strategy.id) is None
    strategy_dir = settings.tasks_dir / str(ctx.task_id) / "strategy"
    assert not list(strategy_dir.glob("decision_table_*.csv"))
    assert not list(strategy_dir.glob("monitoring_plan_*.json"))


def _assert_adoption_rejected_without_mutation(
    settings,
    strategy_repo: StrategyRepository,
    strategy_id: str,
) -> None:
    assert strategy_repo.get_strategy_meta(strategy_id)["status"] == "draft"
    assert strategy_repo.list_strategy_artifacts(strategy_id) == []
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.adopt") == []

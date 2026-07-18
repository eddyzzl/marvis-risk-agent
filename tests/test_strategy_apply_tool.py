from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.tools import tool_apply_strategy
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_manifest
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings


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
            {"type": "segment", "value": 7},
        ),
    }
    default_action, matched_action = actions[strategy_type]
    return {
        "strategy_type": strategy_type,
        "default_action": default_action,
        "rules": [
            {
                "rule_id": "positive",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "x",
                    "operator": ">",
                    "value": 0,
                },
                "action": {**matched_action, "reason_code": "positive_value"},
            }
        ],
    }


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
        )
    )


def _runtime_fixture(tmp_path: Path, strategy_type: str):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _task(TaskRepository(settings.db_path), tmp_path, strategy_type)
    source = tmp_path / f"{strategy_type}.parquet"
    pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "x": [0, 1, 2],
            "keep_me": [10.0, 20.0, 30.0],
        }
    ).to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(
        source,
        task_id=task.id,
        role="strategy_sample",
    )
    strategy = build_strategy_from_spec(_spec(strategy_type))
    StrategyRepository(settings.db_path).create_strategy(task.id, strategy)
    ctx = ToolContext(
        task_id=task.id,
        seed=17,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    return settings, task, registry, dataset, strategy, ctx


@pytest.mark.parametrize(
    ("strategy_type", "expected_actions", "expected_values", "expected_value_types"),
    [
        (
            "approval",
            ["approval", "reject", "reject"],
            ["approve", "reject", "reject"],
            ["string", "string", "string"],
        ),
        (
            "reject",
            ["approval", "reject", "reject"],
            ["approve", "reject", "reject"],
            ["string", "string", "string"],
        ),
        (
            "limit",
            ["limit", "limit", "limit"],
            [1000, 2000, 2000],
            ["integer", "integer", "integer"],
        ),
        (
            "pricing",
            ["pricing", "pricing", "pricing"],
            [0.1, 0.2, 0.2],
            ["number", "number", "number"],
        ),
        (
            "segmentation",
            ["segment", "segment", "segment"],
            ["base", "7", "7"],
            ["string", "integer", "integer"],
        ),
    ],
)
def test_apply_strategy_executes_all_typed_channels_and_registers_evidence(
    tmp_path: Path,
    strategy_type: str,
    expected_actions: list[str],
    expected_values: list,
    expected_value_types: list[str],
) -> None:
    settings, task, registry, dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        strategy_type,
    )

    result = tool_apply_strategy(
        {"dataset_id": dataset.id, "strategy_id": strategy.id},
        ctx,
    )

    assert result["schema_version"] == "strategy.apply.v1"
    assert result["strategy_type"] == strategy_type
    assert result["population_count"] == 3
    assert result["rule_counts"] == {"positive": 2}
    assert result["default_count"] == 1
    assert sum(result["rule_counts"].values()) + result["default_count"] == 3
    assert sum(result["action_counts"].values()) == 3
    derived_dataset = registry.get(result["result_dataset_id"])
    assert derived_dataset.task_id == task.id
    assert derived_dataset.role == "strategy.applied"
    derived_path = registry.resolve_path(derived_dataset.id)
    derived = DataBackend(settings.datasets_dir).read_frame(derived_path)
    assert derived.columns[:3].tolist() == ["customer_id", "x", "keep_me"]
    columns = result["output_columns"]
    assert derived[columns["action"]].tolist() == expected_actions
    assert derived[columns["value"]].tolist() == expected_values
    assert derived[columns["value_type"]].tolist() == expected_value_types
    rule_ids = derived[columns["rule_id"]].tolist()
    assert pd.isna(rule_ids[0])
    assert rule_ids[1:] == ["positive", "positive"]
    reason_codes = derived[columns["reason_code"]].tolist()
    assert pd.isna(reason_codes[0])
    assert reason_codes[1:] == ["positive_value", "positive_value"]
    source_path = registry.resolve_path(dataset.id)
    assert result["evidence"] == {
        "source_dataset_content_hash": sha256_file(source_path),
        "strategy_effect_hash": strategy_spec_hash(strategy.spec),
        "result_dataset_content_hash": sha256_file(derived_path),
    }

    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    apply_tool = next(tool for tool in manifest.tools if tool.name == "apply_strategy")
    validate_against_schema(result, apply_tool.output_schema, label="strategy apply")
    assert apply_tool.entrypoint == "tool_apply_strategy"
    assert set(apply_tool.side_effects) == {
        "read:dataset",
        "read:strategy",
        "write:dataset",
    }

    audit = PluginRepository(settings.db_path).list_audit(kind="strategy.apply")
    assert len(audit) == 1
    assert audit[0]["target_ref"] == derived_dataset.id
    assert audit[0]["detail"]["task_id"] == task.id
    assert audit[0]["detail"]["strategy_id"] == strategy.id
    assert audit[0]["detail"]["evidence"] == result["evidence"]


def test_apply_strategy_supports_safe_custom_output_names(tmp_path: Path) -> None:
    settings, _task_row, registry, dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )

    result = tool_apply_strategy(
        {
            "dataset_id": dataset.id,
            "strategy_id": strategy.id,
            "output_columns": {"action": "decision_action"},
        },
        ctx,
    )

    assert result["output_columns"]["action"] == "decision_action"
    derived = DataBackend(settings.datasets_dir).read_frame(
        registry.resolve_path(result["result_dataset_id"])
    )
    assert "decision_action" in derived.columns
    assert "strategy_value" in derived.columns


def test_apply_strategy_preserves_legacy_row_output_alias(tmp_path: Path) -> None:
    settings, task, registry, dataset, _strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    aliased = build_strategy_from_spec(
        {
            "strategy_type": "approval",
            "default_action": {"type": "approval", "output_value": "pass"},
            "rules": [],
        }
    )
    StrategyRepository(settings.db_path).create_strategy(task.id, aliased)

    result = tool_apply_strategy(
        {"dataset_id": dataset.id, "strategy_id": aliased.id},
        ctx,
    )

    derived = DataBackend(settings.datasets_dir).read_frame(
        registry.resolve_path(result["result_dataset_id"])
    )
    columns = result["output_columns"]
    assert derived[columns["action"]].tolist() == ["approval"] * 3
    assert derived[columns["value"]].tolist() == ["pass"] * 3
    assert derived[columns["value_type"]].tolist() == ["string"] * 3


def test_apply_strategy_serializes_mixed_value_aliases_without_parquet_type_loss(
    tmp_path: Path,
) -> None:
    settings, task, registry, dataset, _strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    mixed = build_strategy_from_spec(
        {
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "rules": [
                {
                    "rule_id": "manual",
                    "priority": 1,
                    "condition": {
                        "op": "compare",
                        "field": "x",
                        "operator": ">",
                        "value": 0,
                    },
                    "action": {
                        "type": "approval",
                        "output_value": 123,
                    },
                }
            ],
        }
    )
    StrategyRepository(settings.db_path).create_strategy(task.id, mixed)

    result = tool_apply_strategy(
        {"dataset_id": dataset.id, "strategy_id": mixed.id},
        ctx,
    )

    derived = DataBackend(settings.datasets_dir).read_frame(
        registry.resolve_path(result["result_dataset_id"])
    )
    columns = result["output_columns"]
    assert derived[columns["value"]].tolist() == ["approve", "123", "123"]
    assert derived[columns["value_type"]].tolist() == [
        "string",
        "integer",
        "integer",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_prefix": "../escape"},
        {"output_prefix": 123},
        {
            "output_prefix": "decision_",
            "output_columns": {"action": "decision"},
        },
        {
            "output_columns": {
                "action": "Same_Name",
                "value": "same_name",
            }
        },
    ],
)
def test_apply_strategy_rejects_unsafe_or_ambiguous_output_names(
    tmp_path: Path,
    overrides: dict,
) -> None:
    settings, task, registry, dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    before = {item.id for item in registry.list_for_task(task.id)}

    with pytest.raises(StrategyError):
        tool_apply_strategy(
            {
                "dataset_id": dataset.id,
                "strategy_id": strategy.id,
                **overrides,
            },
            ctx,
        )

    assert {item.id for item in registry.list_for_task(task.id)} == before
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.apply") == []


def test_apply_strategy_rejects_output_column_collision(tmp_path: Path) -> None:
    settings, task, registry, _dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    source = tmp_path / "collision.parquet"
    pd.DataFrame({"x": [1], "Strategy_Action": ["existing"]}).to_parquet(
        source,
        index=False,
    )
    colliding = registry.register_existing(
        source,
        task_id=task.id,
        role="strategy_sample",
    )
    before = {item.id for item in registry.list_for_task(task.id)}

    with pytest.raises(StrategyError, match="already exist"):
        tool_apply_strategy(
            {"dataset_id": colliding.id, "strategy_id": strategy.id},
            ctx,
        )

    assert {item.id for item in registry.list_for_task(task.id)} == before
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.apply") == []
    assert not (settings.datasets_dir / task.id / "strategy_apply").exists()


def test_apply_strategy_fails_closed_for_cross_task_dataset_or_strategy(
    tmp_path: Path,
) -> None:
    settings, owner, registry, owner_dataset, strategy, owner_ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    foreign = _task(TaskRepository(settings.db_path), tmp_path, "foreign")
    foreign_path = tmp_path / "foreign.parquet"
    pd.DataFrame({"x": [0, 1]}).to_parquet(foreign_path, index=False)
    foreign_dataset = registry.register_existing(
        foreign_path,
        task_id=foreign.id,
        role="strategy_sample",
    )
    foreign_ctx = ToolContext(
        task_id=foreign.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )

    with pytest.raises(StrategyError, match="dataset not found"):
        tool_apply_strategy(
            {"dataset_id": foreign_dataset.id, "strategy_id": strategy.id},
            owner_ctx,
        )
    with pytest.raises(StrategyError, match="strategy not found"):
        tool_apply_strategy(
            {"dataset_id": owner_dataset.id, "strategy_id": strategy.id},
            foreign_ctx,
        )

    assert owner.id != foreign.id
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.apply") == []


def test_apply_strategy_rejects_source_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, task, registry, dataset, strategy, ctx = _runtime_fixture(
        tmp_path,
        "approval",
    )
    original_read = DataBackend.read_frame

    def read_then_mutate(backend, path, columns=None):
        frame = original_read(backend, path, columns=columns)
        mutated = frame.copy()
        mutated.loc[0, "x"] = 999
        mutated.to_parquet(path, index=False)
        return frame

    monkeypatch.setattr(DataBackend, "read_frame", read_then_mutate)

    with pytest.raises(StrategyError, match="changed while.*applied"):
        tool_apply_strategy(
            {"dataset_id": dataset.id, "strategy_id": strategy.id},
            ctx,
        )

    assert {item.role for item in registry.list_for_task(task.id)} == {
        "strategy_sample"
    }
    assert PluginRepository(settings.db_path).list_audit(kind="strategy.apply") == []

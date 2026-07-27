from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import (
    DatasetRepository,
    ModelingRepository,
    PluginRepository,
    TaskRepository,
    init_db,
)
from marvis.domain import TaskCreate
from marvis.feature.preprocessing import read_preprocessing_chain
from marvis.files import sha256_file
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="特殊值治理",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            target_col="y",
            split_col="split",
            feature_columns=["x1", "x2"],
        )
    )
    return runner, registry, settings, task


def _sample(registry, tmp_path, task_id):
    rows = 240
    frame = pd.DataFrame({
        "x1": [
            -999.0 if i % 12 == 0 else float((i * 37) % 101)
            for i in range(rows)
        ],
        "x2": [
            9999.0 if i % 15 == 0 else float((i * 17) % 89)
            for i in range(rows)
        ],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })
    path = tmp_path / "special_values.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="modeling_sample")


def _sentinels():
    return {
        "x1": [[-999.0, 20 / 240]],
        "x2": [[9999.0, 16 / 240]],
    }


def test_special_value_gate_is_noop_without_detected_values(tmp_path):
    runner, registry, _settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)

    result = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "sentinel_columns": {},
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["result_dataset_id"] == dataset.id
    assert result.output["selected"] == ["x1", "x2"]
    assert result.output["governance"] == {}


def test_special_value_gate_requires_complete_typed_decisions(tmp_path):
    runner, registry, _settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)

    result = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "sentinel_columns": _sentinels(),
            "decisions": {"x1": {"action": "mask"}},
        },
        task_id=task.id,
    )

    assert result.ok is False
    assert result.error_kind == "special_value_decision_required"
    assert result.error_detail["columns"] == ["x2"]
    assert result.error_detail["problems"]["x2"] == "missing_decision"


def test_retain_requires_confirmation_and_reason(tmp_path):
    runner, registry, _settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)

    result = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1"],
            "sentinel_columns": {"x1": _sentinels()["x1"]},
            "decisions": {
                "x1": {
                    "action": "retain",
                    "confirmed": False,
                    "reason": "业务编码",
                }
            },
        },
        task_id=task.id,
    )

    assert result.ok is False
    assert result.error_kind == "special_value_decision_required"
    assert (
        result.error_detail["problems"]["x1"]
        == "retain_requires_explicit_confirmation"
    )


def test_mask_writes_real_derived_dataset_and_exact_replay_step(tmp_path):
    runner, registry, _settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)

    result = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "sentinel_columns": _sentinels(),
            "decisions": {
                "x1": {"action": "mask"},
                "x2": {"action": "drop"},
            },
            "seed": 17,
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["result_dataset_id"] != dataset.id
    assert result.output["selected"] == ["x1"]
    assert result.output["masked"] == ["x1"]
    assert result.output["dropped"] == ["x2"]
    derived_path = registry.resolve_path(result.output["result_dataset_id"])
    derived = pd.read_parquet(derived_path)
    assert int(derived["x1"].isna().sum()) == 20
    assert not bool((derived["x1"] == -999.0).any())
    assert bool((derived["x2"] == 9999.0).any())
    chain = read_preprocessing_chain(derived_path)
    assert chain[-1] == {
        "kind": "sentinel",
        "columns": ["x1"],
        "params": {"x1": [-999.0]},
    }
    assert result.output["governance"]["x1"]["action"] == "mask"
    assert result.output["governance"]["x1"]["fingerprint"].startswith("sha256:")


def test_explicit_retain_freezes_fingerprint_without_rewriting_dataset(tmp_path):
    runner, registry, _settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)

    result = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1"],
            "sentinel_columns": {"x1": _sentinels()["x1"]},
            "decisions": {
                "x1": {
                    "action": "retain",
                    "confirmed": True,
                    "reason": "-999 是已确认的无征信记录业务分层",
                }
            },
        },
        task_id=task.id,
    )

    assert result.ok is True, result.error
    assert result.output["result_dataset_id"] == dataset.id
    evidence = result.output["governance"]["x1"]
    assert evidence["action"] == "retain"
    assert evidence["confirmed"] is True
    assert evidence["source_dataset_content_hash"] == sha256_file(
        registry.resolve_path(dataset.id)
    )
    assert evidence["fingerprint"].startswith("sha256:")


def test_train_blocks_forged_lineage_and_persists_real_governance(tmp_path):
    runner, registry, settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)
    sentinel_columns = {"x1": _sentinels()["x1"]}
    from marvis.packs.modeling.special_value_tools import (
        SPECIAL_VALUE_POLICY_VERSION,
        special_value_decision_fingerprint,
    )

    mask_evidence = {
        "policy_version": SPECIAL_VALUE_POLICY_VERSION,
        "column": "x1",
        "action": "mask",
        "detected_values": [-999.0],
        "confirmed": False,
        "reason": "",
        "source_dataset_id": dataset.id,
        "source_dataset_content_hash": sha256_file(
            registry.resolve_path(dataset.id)
        ),
        "resolved_dataset_id": dataset.id,
    }
    mask_fingerprint = special_value_decision_fingerprint(mask_evidence)
    mask_evidence["fingerprint"] = mask_fingerprint
    mask_evidence["decision_fingerprint"] = mask_fingerprint

    blocked = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "sentinel_columns": sentinel_columns,
            "special_value_governance": {"x1": mask_evidence},
            "seed": 17,
        },
        task_id=task.id,
    )
    assert blocked.ok is False
    assert blocked.error_kind == "special_value_decision_required"
    assert (
        blocked.error_detail["problems"]["x1"]
        == "missing_exact_sentinel_preprocessing_step"
    )

    resolved = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "sentinel_columns": sentinel_columns,
            "decisions": {"x1": {"action": "mask"}},
        },
        task_id=task.id,
    )
    assert resolved.ok is True, resolved.error
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": resolved.output["result_dataset_id"],
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "sentinel_columns": sentinel_columns,
            "special_value_governance": resolved.output["governance"],
            "seed": 17,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error
    artifact = ModelingRepository(settings.db_path).get_model_artifact(
        trained.output["artifact_id"]
    )
    assert artifact is not None
    assert artifact.params["special_value_governance"]["x1"]["action"] == "mask"
    assert artifact.params["preprocessing_steps"][-1]["params"] == {"x1": [-999.0]}


def test_train_rejects_forged_or_cross_dataset_retain_evidence(tmp_path):
    runner, registry, _settings, task = _runtime(tmp_path)
    dataset = _sample(registry, tmp_path, task.id)
    sentinel_columns = {"x1": _sentinels()["x1"]}
    resolved = runner.invoke(
        ToolRef("modeling", "resolve_special_values"),
        {
            "dataset_id": dataset.id,
            "features": ["x1", "x2"],
            "sentinel_columns": sentinel_columns,
            "decisions": {
                "x1": {
                    "action": "retain",
                    "confirmed": True,
                    "reason": "已核验为业务特殊编码",
                }
            },
        },
        task_id=task.id,
    )
    assert resolved.ok is True, resolved.error

    forged = {
        "x1": {
            **resolved.output["governance"]["x1"],
            "fingerprint": "sha256:forged",
            "decision_fingerprint": "sha256:forged",
        }
    }
    common = {
        "recipe": "lr",
        "features": ["x1", "x2"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "sentinel_columns": sentinel_columns,
        "seed": 17,
    }
    blocked = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            **common,
            "dataset_id": dataset.id,
            "special_value_governance": forged,
        },
        task_id=task.id,
    )
    assert blocked.ok is False
    assert blocked.error_detail["problems"]["x1"] == "invalid_decision_fingerprint"

    from marvis.packs.modeling.special_value_tools import (
        special_value_decision_fingerprint,
    )

    empty_reason_row = {
        **resolved.output["governance"]["x1"],
        "reason": "",
    }
    empty_reason_fingerprint = special_value_decision_fingerprint(empty_reason_row)
    empty_reason_row["fingerprint"] = empty_reason_fingerprint
    empty_reason_row["decision_fingerprint"] = empty_reason_fingerprint
    missing_reason = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            **common,
            "dataset_id": dataset.id,
            "special_value_governance": {"x1": empty_reason_row},
        },
        task_id=task.id,
    )
    assert missing_reason.ok is False
    assert missing_reason.error_detail["problems"]["x1"] == "retain_requires_reason"

    copied_path = tmp_path / "same_content_different_dataset.parquet"
    pd.read_parquet(registry.resolve_path(dataset.id)).to_parquet(
        copied_path,
        index=False,
    )
    copied = registry.register_existing(
        copied_path,
        task_id=task.id,
        role="modeling_sample",
    )
    reused = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            **common,
            "dataset_id": copied.id,
            "special_value_governance": resolved.output["governance"],
        },
        task_id=task.id,
    )
    assert reused.ok is False
    assert reused.error_detail["problems"]["x1"] in {
        "governance_resolved_dataset_mismatch",
        "retain_source_dataset_mismatch",
    }

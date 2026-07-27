from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from marvis.artifacts.transactional import StagedArtifact
from marvis.packs.modeling import train_tools
from marvis.packs.modeling import tune_checkpoint
from marvis.packs.modeling.tune_checkpoint import (
    TuneCheckpointStore,
    build_tune_checkpoint_identity,
    dataset_content_identity,
)


def _identity(**overrides) -> dict:
    values = {
        "task_id": "task-a",
        "dataset_id": "dataset-1",
        "dataset_content_hash": "sha256:" + "a" * 64,
        "features": ["x2", "x1"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "sample_weight_col": "weight",
        "drop_nan_labels": True,
        "recipe": "lgb",
        "n_trials": 40,
        "seed": 123,
        "cv_folds": 3,
        "early_stopping_rounds": 100,
        "max_boost_round": 3000,
        "overfit_penalty": 0.5,
        "base_params": {"learning_rate": 0.1, "sample_weight_col": "weight"},
        "control_params": {"sample_weight_col": "weight"},
    }
    values.update(overrides)
    return build_tune_checkpoint_identity(**values)


def _result(*, recipe: str = "lgb", n_trials: int = 1) -> dict:
    trials = [
        {
            "params": {"depth": index + 2},
            "score": 0.2 + index / 100,
            "test_ks": 0.3 + index / 100,
            "search_stage": "coarse" if index == 0 else "fine",
        }
        for index in range(n_trials)
    ]
    return {
        "best_params": {"recipe": recipe, "depth": n_trials + 1},
        "best_metrics": {"test_ks": trials[-1]["test_ks"]},
        "n_trials": n_trials,
        "trials": trials,
        "nan_labels_dropped": 0,
    }


def test_checkpoint_identity_is_per_recipe_complete_and_order_sensitive():
    identity = _identity()

    assert identity["task_id"] == "task-a"
    assert identity["dataset"] == {
        "id": "dataset-1",
        "content_hash": "sha256:" + "a" * 64,
    }
    assert identity["features"] == ["x2", "x1"]
    assert identity["recipe"] == "lgb"
    assert identity["runtime_fingerprint"]["pack_version"]
    assert identity["runtime_fingerprint"]["implementation_sha256"].startswith("sha256:")
    assert identity["runtime_fingerprint"]["dependencies"]["lightgbm"]
    assert identity["runtime_fingerprint"]["dependencies"]["pyarrow"]
    assert identity["runtime_fingerprint"]["dependencies"]["scipy"]
    assert identity["runtime_fingerprint"]["platform"]["system"]
    assert identity["runtime_fingerprint"]["cpu"]["logical_count"] >= 1
    assert (
        identity["runtime_fingerprint"]["threading"]["default_tune_num_threads"]
        == 0
    )
    assert identity["runtime_fingerprint"]["fingerprint"].startswith("sha256:")
    assert identity["n_trials"] == 40
    assert identity["seed"] == 123
    assert identity["base_params"]["learning_rate"] == 0.1
    assert identity["control_params"] == {"sample_weight_col": "weight"}
    assert "recipes" not in identity
    assert identity != _identity(features=["x1", "x2"])


def test_runtime_dependency_version_changes_checkpoint_identity(monkeypatch):
    versions = {
        "numpy": "1",
        "pandas": "2",
        "pyarrow": "3",
        "scipy": "4",
        "lightgbm": "3",
    }
    monkeypatch.setattr(
        tune_checkpoint,
        "_dependency_version",
        lambda distribution: versions[distribution],
    )
    first = _identity()
    versions["lightgbm"] = "4"
    second = _identity()

    assert first != second
    assert first["runtime_fingerprint"]["dependencies"]["lightgbm"] == "3"
    assert second["runtime_fingerprint"]["dependencies"]["lightgbm"] == "4"


def test_runtime_thread_environment_changes_checkpoint_identity(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    first = _identity()
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    second = _identity()

    assert first != second
    assert first["runtime_fingerprint"]["threading"]["environment"]["OMP_NUM_THREADS"] is None
    assert second["runtime_fingerprint"]["threading"]["environment"]["OMP_NUM_THREADS"] == "3"


def test_checkpoint_implementation_hash_covers_defaults_and_backend():
    assert "defaults.py" in tune_checkpoint._IMPLEMENTATION_FILES
    assert "../../data/backend.py" in tune_checkpoint._IMPLEMENTATION_FILES


def test_dataset_identity_hashes_current_bytes_even_when_registry_hash_is_stale(tmp_path):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"current bytes")
    stale = SimpleNamespace(content_hash=hashlib.sha256(b"old bytes").hexdigest())

    assert dataset_content_identity(stale, path) == (
        "sha256:" + hashlib.sha256(b"current bytes").hexdigest()
    )


def test_checkpoint_load_silently_misses_on_identity_or_checksum_damage(tmp_path):
    store = TuneCheckpointStore(tmp_path / "checkpoints")
    identity = _identity()
    result = _result()
    path = store.save("lgb", identity, result)

    assert store.load("lgb", identity) == result
    assert store.load("lgb", _identity(n_trials=41)) is None

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["result"]["best_metrics"]["test_ks"] = 0.999
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert store.load("lgb", identity) is None

    path.write_text("{broken", encoding="utf-8")
    assert store.load("lgb", identity) is None


def test_checkpoint_save_failure_leaves_previous_file_intact_and_no_stage(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoints"
    store = TuneCheckpointStore(root)
    identity = _identity()
    original = _result(n_trials=1)
    store.save("lgb", identity, original)

    def fail_promote(_self):
        raise OSError("simulated promote failure")

    monkeypatch.setattr(StagedArtifact, "promote", fail_promote)
    with pytest.raises(OSError, match="simulated promote failure"):
        store.save("lgb", identity, _result(n_trials=2))

    assert store.load("lgb", identity) == original
    staging = root / ".staging"
    assert not staging.exists() or not list(staging.iterdir())


class _Registry:
    def __init__(self, path: Path):
        self.path = path
        self.dataset = SimpleNamespace(
            id="dataset-1",
            task_id="task-a",
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def get(self, _dataset_id: str):
        return self.dataset

    def resolve_path(self, _dataset_id: str):
        return self.path


def _tuning_inputs(*, xgb_trials: int = 1, recipes: list[str] | None = None) -> dict:
    selected = recipes or ["lgb", "xgb"]
    return {
        "dataset_id": "dataset-1",
        "recipes": selected,
        "features": ["x2", "x1"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "n_trials_by_recipe": {
            recipe: xgb_trials if recipe == "xgb" else 1 for recipe in selected
        },
        "seed": 7,
        "early_stopping_rounds": 9,
        "max_boost_round": 30,
        "overfit_penalty": 0.25,
    }


def _runtime(tmp_path: Path):
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"stable dataset bytes")
    return SimpleNamespace(
        registry=_Registry(dataset_path),
        backend=object(),
        settings=SimpleNamespace(tasks_dir=tmp_path / "tasks"),
    )


def _patch_isolated_tuner(monkeypatch, fake_tune):
    """Adapt the historical in-process fake to the recipe-worker seam."""

    def fake_isolated(recipe_inputs, *, ctx, progress_callback):
        del ctx
        result = fake_tune(
            None,
            None,
            **recipe_inputs,
            progress_callback=progress_callback,
        )
        return {
            "best_params": result.best_params,
            "best_metrics": result.best_metrics,
            "n_trials": result.n_trials,
            "trials": list(result.trials),
            "nan_labels_dropped": result.nan_labels_dropped,
        }

    monkeypatch.setattr(
        train_tools,
        "_run_tuning_recipe_isolated",
        fake_isolated,
    )


def test_retry_reuses_completed_recipe_and_publishes_equivalent_progress(
    tmp_path, monkeypatch
):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(train_tools, "_runtime", lambda _ctx: runtime)
    calls = []
    fail_xgb = True

    def fake_tune(_backend, _path, **kwargs):
        nonlocal fail_xgb
        recipe = kwargs["recipe"]
        calls.append(recipe)
        if recipe == "xgb" and fail_xgb:
            raise TimeoutError("simulated timeout")
        result = _result(recipe=recipe, n_trials=max(1, int(kwargs["n_trials"])))
        last = result["trials"][-1]
        kwargs["progress_callback"](
            {
                "kind": "model_tuning",
                "algorithm": recipe,
                "trial": result["n_trials"],
                "trial_total": result["n_trials"],
                "stage": last["search_stage"],
                "selection_score": last["score"],
                "test_ks": last["test_ks"],
                "best_selection_score": last["score"],
                "best_test_ks": last["test_ks"],
            }
        )
        return SimpleNamespace(
            best_params=result["best_params"],
            best_metrics=result["best_metrics"],
            n_trials=result["n_trials"],
            trials=tuple(result["trials"]),
            nan_labels_dropped=0,
        )

    _patch_isolated_tuner(monkeypatch, fake_tune)
    ctx = SimpleNamespace(task_id="task-a", seed=7, report_progress=lambda _event: None)

    with pytest.raises(TimeoutError, match="simulated timeout"):
        train_tools.tool_tune_hyperparameters(_tuning_inputs(), ctx)
    assert calls == ["lgb", "xgb"]

    fail_xgb = False
    progress = []
    ctx.report_progress = progress.append
    recovered = train_tools.tool_tune_hyperparameters(_tuning_inputs(), ctx)
    assert calls == ["lgb", "xgb", "xgb"]
    hit = next(event for event in progress if event.get("cache_hit"))
    assert hit["algorithm"] == "lgb"
    assert hit["completed_trials"] == 1
    assert hit["total_trials"] == 2

    calls.clear()
    cached = train_tools.tool_tune_hyperparameters(_tuning_inputs(), ctx)
    assert calls == []
    assert cached == recovered
    assert set(cached) == {
        "best_params",
        "best_metrics",
        "n_trials",
        "trials",
        "nan_labels_dropped",
        "per_recipe",
    }


def test_checkpoint_miss_is_per_recipe_and_task_scoped(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(train_tools, "_runtime", lambda _ctx: runtime)
    calls = []

    def fake_tune(_backend, _path, **kwargs):
        recipe = kwargs["recipe"]
        calls.append(recipe)
        result = _result(recipe=recipe, n_trials=max(1, int(kwargs["n_trials"])))
        return SimpleNamespace(
            best_params=result["best_params"],
            best_metrics=result["best_metrics"],
            n_trials=result["n_trials"],
            trials=tuple(result["trials"]),
            nan_labels_dropped=0,
        )

    _patch_isolated_tuner(monkeypatch, fake_tune)
    task_a = SimpleNamespace(task_id="task-a", seed=7)
    task_b = SimpleNamespace(task_id="task-b", seed=7)

    train_tools.tool_tune_hyperparameters(_tuning_inputs(), task_a)
    assert calls == ["lgb", "xgb"]

    calls.clear()
    train_tools.tool_tune_hyperparameters(_tuning_inputs(xgb_trials=2), task_a)
    assert calls == ["xgb"]

    calls.clear()
    runtime.registry.dataset.task_id = "task-b"
    train_tools.tool_tune_hyperparameters(_tuning_inputs(xgb_trials=2), task_b)
    assert calls == ["lgb", "xgb"]
    assert (tmp_path / "tasks" / "task-a" / "modeling_artifacts" / "tuning_checkpoints" / "lgb.json").is_file()
    assert (tmp_path / "tasks" / "task-b" / "modeling_artifacts" / "tuning_checkpoints" / "lgb.json").is_file()


def test_current_dataset_byte_change_invalidates_every_recipe(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(train_tools, "_runtime", lambda _ctx: runtime)
    calls = []

    def fake_tune(_backend, _path, **kwargs):
        recipe = kwargs["recipe"]
        calls.append(recipe)
        result = _result(recipe=recipe)
        return SimpleNamespace(
            best_params=result["best_params"],
            best_metrics=result["best_metrics"],
            n_trials=1,
            trials=tuple(result["trials"]),
            nan_labels_dropped=0,
        )

    _patch_isolated_tuner(monkeypatch, fake_tune)
    ctx = SimpleNamespace(task_id="task-a", seed=7)
    inputs = _tuning_inputs()
    train_tools.tool_tune_hyperparameters(inputs, ctx)
    assert calls == ["lgb", "xgb"]

    calls.clear()
    runtime.registry.path.write_bytes(b"changed without updating registry content_hash")
    train_tools.tool_tune_hyperparameters(inputs, ctx)
    assert calls == ["lgb", "xgb"]


def test_single_recipe_output_shape_is_identical_on_checkpoint_hit(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(train_tools, "_runtime", lambda _ctx: runtime)
    calls = []

    def fake_tune(_backend, _path, **kwargs):
        calls.append(kwargs["recipe"])
        result = _result(recipe=kwargs["recipe"])
        return SimpleNamespace(
            best_params=result["best_params"],
            best_metrics=result["best_metrics"],
            n_trials=1,
            trials=tuple(result["trials"]),
            nan_labels_dropped=0,
        )

    _patch_isolated_tuner(monkeypatch, fake_tune)
    ctx = SimpleNamespace(task_id="task-a", seed=7)
    inputs = _tuning_inputs(recipes=["lgb"])

    cold = train_tools.tool_tune_hyperparameters(inputs, ctx)
    hit = train_tools.tool_tune_hyperparameters(inputs, ctx)
    forced = train_tools.tool_tune_hyperparameters(
        {**inputs, "force_recompute": True},
        ctx,
    )

    assert calls == ["lgb", "lgb"]
    assert hit == cold
    assert forced == cold
    assert isinstance(hit["best_params"], dict)
    assert hit["best_params"]["recipe"] == "lgb"


def test_multi_recipe_tuning_uses_one_isolated_worker_per_recipe(
    tmp_path, monkeypatch
):
    """Regression: native learner RSS must not accumulate across recipes.

    The aggregate tool worker owns progress/checkpoints only.  Every cache miss
    is delegated to a short-lived recipe worker; calling the in-process tuner
    here would retain LightGBM/XGBoost native allocator high-water marks until
    CatBoost starts.
    """

    runtime = _runtime(tmp_path)
    monkeypatch.setattr(train_tools, "_runtime", lambda _ctx: runtime)
    calls = []

    def fake_isolated(recipe_inputs, *, ctx, progress_callback):
        assert ctx.task_id == "task-a"
        recipe = recipe_inputs["recipe"]
        calls.append(recipe)
        result = _result(recipe=recipe)
        last = result["trials"][-1]
        progress_callback(
            {
                "kind": "model_tuning",
                "algorithm": recipe,
                "trial": 1,
                "trial_total": 1,
                "stage": last["search_stage"],
                "selection_score": last["score"],
                "test_ks": last["test_ks"],
                "best_selection_score": last["score"],
                "best_test_ks": last["test_ks"],
            }
        )
        return result

    monkeypatch.setattr(train_tools, "_run_tuning_recipe_isolated", fake_isolated)
    progress = []
    ctx = SimpleNamespace(task_id="task-a", seed=7, report_progress=progress.append)

    output = train_tools.tool_tune_hyperparameters(_tuning_inputs(), ctx)

    assert "tune_hyperparameters" not in train_tools.__dict__
    assert calls == ["lgb", "xgb"]
    assert list(output["per_recipe"]) == ["lgb", "xgb"]
    assert [
        event["algorithm"]
        for event in progress
        if event.get("trial") == 1 and event.get("stage") != "checkpoint_saved"
    ] == [
        "lgb",
        "xgb",
    ]


def test_checkpoint_save_and_hit_have_formal_correlatable_audits(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    audits = []
    runtime.repo = SimpleNamespace(write_audit=lambda **payload: audits.append(payload))
    monkeypatch.setattr(train_tools, "_runtime", lambda _ctx: runtime)

    def fake_isolated(recipe_inputs, *, ctx, progress_callback):
        del ctx, progress_callback
        return _result(recipe=recipe_inputs["recipe"])

    monkeypatch.setattr(train_tools, "_run_tuning_recipe_isolated", fake_isolated)
    ctx = SimpleNamespace(task_id="task-a", seed=7, report_progress=lambda _event: None)
    inputs = _tuning_inputs(recipes=["lgb"])

    train_tools.tool_tune_hyperparameters(inputs, ctx)
    train_tools.tool_tune_hyperparameters(inputs, ctx)

    assert [audit["kind"] for audit in audits] == [
        "modeling.tuning_checkpoint.saved",
        "modeling.tuning_checkpoint.hit",
    ]
    saved, hit = audits
    assert saved["target_ref"] == hit["target_ref"] == "task-a:lgb"
    assert saved["detail"]["checkpoint_identity_hash"] == hit["detail"][
        "checkpoint_identity_hash"
    ]
    assert saved["detail"]["cache_hit"] is False
    assert hit["detail"]["cache_hit"] is True
    assert saved["detail"]["dataset_content_hash"].startswith("sha256:")
    assert saved["detail"]["runtime_fingerprint"].startswith("sha256:")


def test_importing_tuning_aggregator_does_not_load_native_learner_runtimes():
    script = """
import json
import psutil
import sys
import marvis.packs.modeling.tools
print(json.dumps({
    "native": [name for name in ("lightgbm", "xgboost", "catboost") if name in sys.modules],
    "rss": psutil.Process().memory_info().rss,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONHASHSEED": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["native"] == []
    # The parent is a checkpoint/progress aggregator, not a hidden learner
    # host.  Keep a generous ceiling for pandas/pyarrow variation while still
    # catching the former eager three-runtime import regression.
    assert int(payload["rss"]) < 400 * 1024 * 1024


def test_tuning_manifest_allows_checkpoint_writes_and_twelve_hours():
    manifest_path = Path("marvis/packs/modeling/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tool = next(item for item in manifest["tools"] if item["name"] == "tune_hyperparameters")

    assert "write:artifact" in manifest["permissions"]
    assert "read:artifacts" in manifest["permissions"]
    assert "write:artifact" in tool["side_effects"]
    assert "read:artifacts" in tool["side_effects"]
    assert tool["input_schema"]["properties"]["force_recompute"] == {
        "type": "boolean",
        "default": False,
        "description": "Ignore valid per-recipe tuning checkpoints for this run and atomically replace them with newly computed results.",
    }
    assert tool["timeout_seconds"] >= 12 * 60 * 60

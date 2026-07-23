from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.modeling._common import _jsonable
from marvis.packs.modeling.tune import _cv_folds, tune_hyperparameters
from marvis.packs.modeling.tune_isolation import (
    IsolatedRecipeTuningError,
    _RECIPE_WORKER_TIMEOUT_SECONDS,
    run_tuning_recipe_isolated,
)
from marvis.packs.modeling.recipes.common import _group_ids, carve_early_stop_fold
from marvis.packs.modeling.training_dataset import TrainingDataset
from marvis.plugins.contracts import PROTOCOL_VERSION, ToolContext, WORKER_RESULT_SENTINEL
from marvis.plugins.runner import WorkerResourceLimitExceeded
from marvis.settings import build_settings


def test_recipe_worker_result_is_equivalent_and_process_has_exited(tmp_path):
    """One real recipe crosses the worker protocol without semantic drift.

    The PID assertion is the memory regression contract: native allocations
    cannot accumulate into the next recipe because the interpreter that owned
    them no longer exists when ``run_tuning_recipe_isolated`` returns.
    """

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="tune-isolation-test",
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
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        backend,
        settings.datasets_dir,
    )
    rows = 120
    frame = pd.DataFrame(
        {
            "x1": [((index * 37) % 101) / 100 for index in range(rows)],
            "x2": [((index * 17) % 89) / 100 for index in range(rows)],
            "y": [1 if index % 5 in {0, 1} else 0 for index in range(rows)],
            "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
        }
    )
    source = tmp_path / "sample.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(
        source,
        task_id=task.id,
        role="modeling_sample",
    )
    dataset_path = registry.resolve_path(dataset.id)
    tune_kwargs = {
        "features": ["x1", "x2"],
        "target_col": "y",
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "recipe": "lr",
        "n_trials": 2,
        "seed": 19,
        "early_stopping_rounds": 5,
        "max_boost_round": 30,
        "overfit_penalty": 0.25,
        "sample_weight_col": "",
        "base_params": {},
        "drop_nan_labels": False,
        "cv_folds": None,
    }
    direct = tune_hyperparameters(backend, dataset_path, **tune_kwargs)
    progress = []
    isolated = run_tuning_recipe_isolated(
        {"dataset_id": dataset.id, **tune_kwargs},
        ctx=ToolContext(
            task_id=task.id,
            seed=19,
            datasets_root=settings.datasets_dir,
            workspace=settings.workspace,
        ),
        progress_callback=progress.append,
    )
    worker_pid = int(isolated.pop("_worker_pid"))
    expected = {
        "best_params": _jsonable(direct.best_params),
        "best_metrics": _jsonable(direct.best_metrics),
        "n_trials": direct.n_trials,
        "trials": _jsonable(direct.trials),
        "nan_labels_dropped": direct.nan_labels_dropped,
    }

    assert isolated == expected
    assert [event["trial"] for event in progress] == [1, 2]
    assert worker_pid != os.getpid()
    assert not psutil.pid_exists(worker_pid)


def test_catboost_isolated_tuning_preserves_string_features_from_parquet(tmp_path):
    """The real compact reader must not coerce CatBoost categories to float32."""

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="catboost-string-tune",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="catboost",
            run_mode="agent",
            target_col="y",
            split_col="split",
            feature_columns=["channel", "x1"],
        )
    )
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        backend,
        settings.datasets_dir,
    )
    rows = 150
    channel = (["app", "web", "branch"] * (rows // 3 + 1))[:rows]
    channel[7] = None
    channel[93] = None
    segment = pd.Series(
        (["prime", "mass", None] * (rows // 3 + 1))[:rows],
        dtype="category",
    )
    frame = pd.DataFrame({
        "channel": channel,
        "segment": segment,
        "x1": [((index * 31) % 97) / 100 for index in range(rows)],
        "y": [1 if index % 5 in {0, 1} else 0 for index in range(rows)],
        "split": ["train"] * 90 + ["test"] * 40 + ["oot"] * 20,
    })
    source = tmp_path / "catboost_string.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(
        source,
        task_id=task.id,
        role="modeling_sample",
    )

    isolated = run_tuning_recipe_isolated(
        {
            "dataset_id": dataset.id,
            "features": ["channel", "segment", "x1"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "recipe": "catboost",
            "n_trials": 1,
            "seed": 23,
            "early_stopping_rounds": 3,
            "max_boost_round": 8,
            "overfit_penalty": 0.25,
            "sample_weight_col": "",
            "base_params": {},
            "drop_nan_labels": False,
            "cv_folds": None,
        },
        ctx=ToolContext(
            task_id=task.id,
            seed=23,
            datasets_root=settings.datasets_dir,
            workspace=settings.workspace,
        ),
        progress_callback=None,
    )
    worker_pid = int(isolated.pop("_worker_pid"))

    assert isolated["n_trials"] == 1
    assert isolated["trials"][0]["params"]["cat_features"] == ["channel", "segment"]
    assert not psutil.pid_exists(worker_pid)


def test_grouped_valid_fold_treats_null_identity_as_a_stable_group():
    """Nullable identity columns must never produce a float NaN group id."""

    frame = pd.DataFrame(
        {
            "identity": pd.Series(
                [9_007_199_254_740_991] * 2
                + [None] * 2
                + [9_007_199_254_740_993] * 2
                + [9_007_199_254_740_995] * 2,
                dtype="Int64",
            ),
            "x": range(8),
        }
    )

    fit_train, valid = carve_early_stop_fold(
        frame,
        seed=19,
        group_cols=["identity"],
        valid_fraction=0.25,
    )

    side_by_group = {}
    for side, part in (("fit", fit_train), ("valid", valid)):
        for value in part["identity"].astype("string").fillna("<missing>"):
            side_by_group.setdefault(str(value), set()).add(side)
    assert all(len(sides) == 1 for sides in side_by_group.values())
    groups = _group_ids(frame, ["identity"])
    assert groups.dtype.kind in {"i", "u"}
    assert groups[2] == groups[3]


def test_grouped_cv_keeps_nullable_identity_whole_and_covers_every_row():
    frame = pd.DataFrame(
        {
            "identity": pd.Series(
                [9_007_199_254_740_991] * 2
                + [None] * 2
                + [9_007_199_254_740_993] * 2
                + [9_007_199_254_740_995] * 2
                + [9_007_199_254_740_997] * 2,
                dtype="Int64",
            ),
            "x": range(10),
        }
    )

    folds = _cv_folds(
        frame,
        cv_folds=3,
        seed=31,
        group_cols=["identity"],
    )

    assert sorted(np.concatenate(folds).tolist()) == list(range(len(frame)))
    fold_by_row = {
        int(row): fold
        for fold, indices in enumerate(folds)
        for row in indices.tolist()
    }
    for _, rows in frame.groupby("identity", dropna=False, sort=False).groups.items():
        assert len({fold_by_row[int(row)] for row in rows}) == 1


def test_tuning_preserves_group_feature_large_integer_identity(tmp_path, monkeypatch):
    identity = [
        9_007_199_254_740_991,
        9_007_199_254_740_993,
        9_007_199_254_740_995,
    ]
    rows = 90
    frame = pd.DataFrame(
        {
            "identity": [identity[index % len(identity)] for index in range(rows)],
            "x": [((index * 17) % 89) / 100 for index in range(rows)],
            "y": [index % 2 for index in range(rows)],
            "split": ["train"] * 54 + ["test"] * 24 + ["oot"] * 12,
        }
    )
    path = tmp_path / "large_identity.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    original = TrainingDataset.load_compact.__func__
    captured = {}

    def capture_load(cls, backend_arg, path_arg, **kwargs):
        captured["features"] = list(kwargs["features"])
        captured["extra_columns"] = list(kwargs["extra_columns"])
        loaded = original(cls, backend_arg, path_arg, **kwargs)
        captured["dtype"] = loaded.frame["identity"].dtype
        captured["identity"] = loaded.frame["identity"].drop_duplicates().tolist()
        return loaded

    monkeypatch.setattr(TrainingDataset, "load_compact", classmethod(capture_load))

    tune_hyperparameters(
        backend,
        path,
        features=["identity", "x"],
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        recipe="lr",
        n_trials=1,
        seed=31,
        base_params={"valid_group_cols": ["identity"]},
    )

    assert captured["features"] == ["x"]
    assert "identity" in captured["extra_columns"]
    assert captured["dtype"] == frame["identity"].dtype
    assert captured["identity"] == identity


def test_tuning_timeout_budget_allows_three_recipe_timeouts_to_surface():
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "marvis"
            / "packs"
            / "modeling"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    tool = next(
        item for item in manifest["tools"] if item["name"] == "tune_hyperparameters"
    )

    assert tool["timeout_seconds"] > 3 * _RECIPE_WORKER_TIMEOUT_SECONDS


def test_nested_recipe_worker_stays_in_parent_process_group(tmp_path, monkeypatch):
    """The launch contract prevents an orphan after outer-worker host kill."""

    from marvis.packs.modeling import tune_isolation

    captured = {}

    def fake_run_worker(_python, _job, **kwargs):
        captured.update(kwargs)
        return type(
            "Completed",
            (),
            {
                "stdout": (
                    '@@MARVIS_PLUGIN_RESULT@@{"ok":true,"output":{},'
                    '"worker_protocol_version":3}'
                ),
                "stderr": "",
                "returncode": 0,
            },
        )()

    monkeypatch.setattr(tune_isolation, "_run_worker", fake_run_worker)
    monkeypatch.setattr(tune_isolation, "_try_new_progress_path", lambda _path: None)
    context = ToolContext(
        task_id="task",
        seed=1,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path,
    )

    tune_isolation.run_tuning_recipe_isolated(
        {"seed": 1},
        ctx=context,
        progress_callback=None,
    )

    assert captured["start_new_session"] is False


def _minimal_context(tmp_path) -> ToolContext:
    return ToolContext(
        task_id="task",
        seed=1,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path,
    )


def test_progress_journal_forwards_rapid_events_exactly_once(tmp_path):
    from marvis.packs.modeling import tune_isolation

    path = tmp_path / ".runtime" / "progress" / "events.jsonl"
    seen = []
    watcher = tune_isolation._RecipeProgressJournalWatcher(path, seen.append)
    context = SimpleNamespace(workspace=tmp_path)

    watcher.start()
    for trial in range(1, 101):
        tune_isolation._append_progress_journal(
            context,
            path,
            {"kind": "model_tuning", "trial": trial},
        )
    watcher.close()
    # A repeated close/delivery pass must not replay already-consumed lines.
    watcher.close()

    assert [event["trial"] for event in seen] == list(range(1, 101))


def test_progress_journal_has_a_total_size_cap(tmp_path, monkeypatch):
    from marvis.packs.modeling import tune_isolation

    path = tmp_path / ".runtime" / "progress" / "events.jsonl"
    context = SimpleNamespace(workspace=tmp_path)
    event = {"kind": "model_tuning", "trial": 1}
    encoded = json.dumps(
        event,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    monkeypatch.setattr(
        tune_isolation,
        "MAX_RECIPE_PROGRESS_JOURNAL_BYTES",
        len(encoded) + 1,
    )

    tune_isolation._append_progress_journal(context, path, event)
    tune_isolation._append_progress_journal(context, path, {**event, "trial": 2})

    assert path.read_bytes() == encoded


def test_isolated_timeout_preserves_typed_redacted_diagnostics(tmp_path, monkeypatch):
    from marvis.packs.modeling import tune_isolation

    monkeypatch.setattr(
        tune_isolation,
        "_run_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(
                ["python"],
                17,
                output="token=supersecret123 contact eddy@example.com",
                stderr="Bearer abcdefghijklmnop",
            )
        ),
    )

    with pytest.raises(IsolatedRecipeTuningError) as raised:
        run_tuning_recipe_isolated(
            {"seed": 1},
            ctx=_minimal_context(tmp_path),
            progress_callback=None,
        )

    detail = raised.value.to_detail()
    assert detail["kind"] == "timeout"
    assert detail["timeout_seconds"] == 17
    assert "supersecret123" not in json.dumps(detail)
    assert "eddy@example.com" not in json.dumps(detail)
    assert "[REDACTED" in json.dumps(detail)


def test_isolated_rss_limit_preserves_resource_limits(tmp_path, monkeypatch):
    from marvis.packs.modeling import tune_isolation

    limits = {
        "memory_limit_mb": 4096,
        "peak_rss_mb": 4101.25,
        "memory_limit_exceeded": True,
    }
    monkeypatch.setattr(
        tune_isolation,
        "_run_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            WorkerResourceLimitExceeded(limits)
        ),
    )

    with pytest.raises(IsolatedRecipeTuningError) as raised:
        run_tuning_recipe_isolated(
            {"seed": 1},
            ctx=_minimal_context(tmp_path),
            progress_callback=None,
        )

    assert raised.value.to_detail() == {
        "kind": "resource_limit",
        "subkind": "isolated_recipe_rss_limit",
        "resource_limits": limits,
        "child_error_detail": {"kind": "worker_rss_limit", **limits},
    }


@pytest.mark.skipif(not hasattr(signal, "SIGXCPU"), reason="SIGXCPU is POSIX-only")
def test_isolated_sigxcpu_is_not_misclassified_as_protocol(tmp_path, monkeypatch):
    from marvis.packs.modeling import tune_isolation

    monkeypatch.setattr(
        tune_isolation,
        "_run_worker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["python"],
            -int(signal.SIGXCPU),
            stdout="",
            stderr="token=supersecret123",
        ),
    )

    with pytest.raises(IsolatedRecipeTuningError) as raised:
        run_tuning_recipe_isolated(
            {"seed": 1},
            ctx=_minimal_context(tmp_path),
            progress_callback=None,
        )

    detail = raised.value.to_detail()
    assert detail["kind"] == "resource_limit"
    assert detail["subkind"] == "isolated_recipe_cpu_limit"
    assert detail["resource_limits"]["termination_signal"] == "SIGXCPU"
    assert "supersecret123" not in json.dumps(detail)


def test_child_error_detail_is_bounded_and_sensitive_keys_are_redacted(
    tmp_path,
    monkeypatch,
):
    from marvis.packs.modeling import tune_isolation

    protocol = {
        "ok": False,
        "error_kind": "execution",
        "error": "token=supersecret123",
        "error_detail": {
            "kind": "execution",
            "password": "plain-secret-value",
            "traceback": "contact eddy@example.com " + "x" * 8_000,
        },
        "worker_protocol_version": PROTOCOL_VERSION,
    }
    monkeypatch.setattr(
        tune_isolation,
        "_run_worker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["python"],
            1,
            stdout=WORKER_RESULT_SENTINEL + json.dumps(protocol),
            stderr="",
        ),
    )

    with pytest.raises(IsolatedRecipeTuningError) as raised:
        run_tuning_recipe_isolated(
            {"seed": 1},
            ctx=_minimal_context(tmp_path),
            progress_callback=None,
        )

    detail = raised.value.to_detail()
    encoded = json.dumps(detail)
    assert detail["password"] == "[REDACTED]"
    assert "eddy@example.com" not in encoded
    assert len(detail["traceback"]) <= 4_000

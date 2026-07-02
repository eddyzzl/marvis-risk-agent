"""S1b regression tests: monitor_run (marvis/packs/modeling/tools.py::tool_monitor_run).

Covers: (c) a drift-injected new dataset trips score PSI over the fail threshold
and lands red with gate-ready wording; (d) an unlabeled monitoring run reports
KS/AUC as explicit n/a, never a fabricated value; (e) an artifact trained before
S1b (no baseline_distributions) fails with a clear "no baseline" error.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from marvis.db import ModelingRepository, PluginRepository
from marvis.plugins.manifest import ToolRef

from test_modeling_pack import _runtime


def _linearly_separable_frame(rows: int = 400) -> pd.DataFrame:
    x1 = [((i * 37) % 101) / 100 for i in range(rows)]
    x2 = [((i * 17) % 89) / 100 for i in range(rows)]
    y = [1 if (x1[i] + x2[i]) > 1.0 else 0 for i in range(rows)]
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y,
        "split": ["train"] * 240 + ["test"] * 100 + ["oot"] * 60,
    })


def _train_lr_experiment(runner, registry, tmp_path, task):
    frame = _linearly_separable_frame()
    path = tmp_path / "modeling_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error
    return trained, frame


def test_monitor_run_on_stable_new_data_is_green(tmp_path):
    """A new dataset drawn from the same distribution as training must judge
    green on every check (score PSI, feature CSI, KS/AUC drop all within
    thresholds)."""
    runner, _pr, registry, _backend, task_settings, task = _runtime(tmp_path)
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)

    new_rows = 200
    new_frame = pd.DataFrame({
        "x1": [((i * 37 + 5) % 101) / 100 for i in range(new_rows)],
        "x2": [((i * 17 + 3) % 89) / 100 for i in range(new_rows)],
    })
    new_frame["y"] = [1 if (new_frame["x1"][i] + new_frame["x2"][i]) > 1.0 else 0 for i in range(new_rows)]
    new_path = tmp_path / "new_data.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    monitored = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {
            "experiment_id": trained.output["experiment_id"],
            "dataset_id": new_dataset.id,
            "target_col": "y",
        },
        task_id=task.id,
    )
    assert monitored.ok is True, monitored.error
    assert monitored.output["overall_level"] == "green"
    for check in monitored.output["checks"]:
        assert check["level"] == "green", check


def test_monitor_run_drift_injected_data_trips_psi_over_fail_threshold_and_is_red(tmp_path):
    """(c) Construct a new dataset whose feature distribution is deliberately
    shifted far from the training distribution (a distribution-shift injection).
    Score PSI must exceed the fail threshold (0.25) and the overall verdict must
    be red, with the gate-ready message naming the fail threshold."""
    runner, _pr, registry, _backend, task_settings, task = _runtime(tmp_path)
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)

    drift_rows = 200
    drift_frame = pd.DataFrame({
        "x1": [min(1.0, ((i * 37) % 101) / 100 + 0.6) for i in range(drift_rows)],
        "x2": [min(1.0, ((i * 17) % 89) / 100 + 0.6) for i in range(drift_rows)],
    })
    drift_path = tmp_path / "drift_data.parquet"
    drift_frame.to_parquet(drift_path, index=False)
    drift_dataset = registry.register_existing(drift_path, task_id=task.id, role="scoring_input")

    monitored = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": drift_dataset.id},
        task_id=task.id,
    )
    assert monitored.ok is True, monitored.error
    assert monitored.output["overall_level"] == "red"

    score_check = next(check for check in monitored.output["checks"] if check["id"] == "score_psi")
    assert score_check["level"] == "red"
    assert score_check["status"] == "fail"
    assert score_check["value"] > 0.25
    assert "fail" in score_check["message"] or "0.25" in score_check["message"]

    csi_check = next(check for check in monitored.output["checks"] if check["id"] == "feature_csi_max")
    assert csi_check["level"] == "red"
    assert csi_check["value"] > 0.25

    assert len(monitored.output["top_drifted_features"]) > 0
    assert monitored.output["top_drifted_features"][0]["feature"] in ("x1", "x2")
    # Sorted descending by CSI.
    csis = [row["csi"] for row in monitored.output["top_drifted_features"]]
    assert csis == sorted(csis, reverse=True)

    monitor_audit = PluginRepository(task_settings.db_path).list_audit(kind="modeling.monitor.run")
    assert len(monitor_audit) == 1
    assert monitor_audit[0]["detail"]["overall_level"] == "red"


def test_monitor_run_without_labels_reports_ks_auc_as_explicit_na(tmp_path):
    """(d) A monitoring run against unlabeled new data (the normal case -- labels
    mature months later) must report ks_drop/auc_drop as explicit n/a rows, never
    a fabricated 0.0/0.5-derived value, while score_psi/feature_csi are still
    computed normally."""
    runner, _pr, registry, _backend, task_settings, task = _runtime(tmp_path)
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)

    new_rows = 100
    new_frame = pd.DataFrame({
        "x1": [((i * 37 + 5) % 101) / 100 for i in range(new_rows)],
        "x2": [((i * 17 + 3) % 89) / 100 for i in range(new_rows)],
    })
    new_path = tmp_path / "unlabeled_new_data.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    # No target_col passed at all -- the dataset genuinely has no label column.
    monitored = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert monitored.ok is True, monitored.error

    ks_check = next(check for check in monitored.output["checks"] if check["id"] == "ks_drop")
    auc_check = next(check for check in monitored.output["checks"] if check["id"] == "auc_drop")
    assert ks_check["level"] == "n/a"
    assert ks_check["value"] is None
    assert auc_check["level"] == "n/a"
    assert auc_check["value"] is None

    # PSI/CSI are unaffected by the missing label -- still real computed values.
    score_check = next(check for check in monitored.output["checks"] if check["id"] == "score_psi")
    assert score_check["value"] is not None
    assert score_check["level"] in ("green", "amber", "red")

    # n/a must never contribute a red/amber to the overall verdict on its own --
    # stable score/feature distributions still yield an overall green.
    assert monitored.output["overall_level"] == "green"


def test_monitor_run_on_artifact_without_baseline_raises_clear_error(tmp_path):
    """(e) An artifact trained before S1b (or with its baseline manually cleared,
    simulating a pre-migration DB row) has no reference distribution to compare
    against -- monitor_run must fail with an explicit, actionable message, never
    fabricate a baseline."""
    runner, _pr, registry, _backend, task_settings, task = _runtime(tmp_path)
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)

    # Simulate a pre-S1b artifact: clear the baseline_distributions_json column
    # directly, exactly like an ALTER TABLE ADD COLUMN migration leaves historical
    # rows NULL (mirrors test_modeling_artifact_score_direction.py's own pattern
    # for the score_direction/points_direction NULL-tolerance regression).
    conn = sqlite3.connect(task_settings.db_path)
    conn.execute(
        "UPDATE model_artifacts SET baseline_distributions_json = NULL WHERE id = ?",
        (trained.output["artifact_id"],),
    )
    conn.commit()
    conn.close()

    artifact = ModelingRepository(task_settings.db_path).get_model_artifact(trained.output["artifact_id"])
    assert artifact.baseline_distributions is None

    new_frame = frame[["x1", "x2"]].iloc[:20].reset_index(drop=True)
    new_path = tmp_path / "new_data.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    monitored = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert monitored.ok is False
    assert "baseline" in monitored.error.lower()
    assert "retrain" in monitored.error.lower() or "重训" in monitored.error


def test_monitor_run_accepts_a_pre_scored_dataset(tmp_path):
    """monitor_run against a dataset already scored by score_dataset (scored_dataset_id
    + score_col) must produce the same PSI as scoring internally from dataset_id."""
    runner, _pr, registry, _backend, task_settings, task = _runtime(tmp_path)
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)

    new_rows = 150
    new_frame = pd.DataFrame({
        "x1": [((i * 37 + 5) % 101) / 100 for i in range(new_rows)],
        "x2": [((i * 17 + 3) % 89) / 100 for i in range(new_rows)],
    })
    new_path = tmp_path / "new_data.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    scored = runner.invoke(
        ToolRef("modeling", "score_dataset"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert scored.ok is True, scored.error

    via_scored = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {
            "experiment_id": trained.output["experiment_id"],
            "scored_dataset_id": scored.output["result_dataset_id"],
            "score_col": scored.output["score_col"],
        },
        task_id=task.id,
    )
    via_raw = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert via_scored.ok is True, via_scored.error
    assert via_raw.ok is True, via_raw.error

    scored_psi = next(c for c in via_scored.output["checks"] if c["id"] == "score_psi")["value"]
    raw_psi = next(c for c in via_raw.output["checks"] if c["id"] == "score_psi")["value"]
    assert scored_psi == pytest.approx(raw_psi)


def test_monitor_run_requires_dataset_id_or_scored_dataset_id(tmp_path):
    runner, _pr, registry, _backend, task_settings, task = _runtime(tmp_path)
    trained, _frame = _train_lr_experiment(runner, registry, tmp_path, task)

    result = runner.invoke(
        ToolRef("modeling", "monitor_run"),
        {"experiment_id": trained.output["experiment_id"]},
        task_id=task.id,
    )
    assert result.ok is False
    assert "dataset_id" in result.error

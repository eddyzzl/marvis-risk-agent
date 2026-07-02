"""S1b regression tests: the score_dataset tool
(marvis/packs/modeling/tools.py::tool_score_dataset).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.db import ModelingRepository, PluginRepository
from marvis.plugins.manifest import ToolRef

from test_modeling_pack import _runtime


def _linearly_separable_frame(rows: int = 240) -> pd.DataFrame:
    x1 = [((i * 37) % 101) / 100 for i in range(rows)]
    x2 = [((i * 17) % 89) / 100 for i in range(rows)]
    y = [1 if (x1[i] + x2[i]) > 1.0 else 0 for i in range(rows)]
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y": y,
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })


def test_score_dataset_matches_training_time_scorer_consistency(tmp_path):
    """(b) score_dataset end-to-end, including a preprocessing chain (WOE): scoring
    a *raw* dataset through the tool must produce exactly the same PD values as
    calling _ModelArtifactScorer directly with replay_preprocessing=True on the
    same raw frame -- score_dataset must not silently diverge from the platform's
    own scorer."""
    runner, _pr, registry, backend, settings, task = _runtime(tmp_path)
    rows = 240
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })
    path = tmp_path / "raw_for_woe.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="raw_sample")

    woe = runner.invoke(
        ToolRef("feature", "woe_encode"),
        {"dataset_id": dataset.id, "features": ["x1"], "target_col": "y", "split_col": "split"},
        task_id=task.id,
    )
    assert woe.ok is True, woe.error

    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": woe.output["result_dataset_id"],
            "recipe": "lr",
            "features": ["x1_woe", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error

    # New raw dataset (pre-WOE raw columns, exactly like x1/x2 would look for a
    # genuinely new month of applications).
    new_rows = 80
    new_frame = pd.DataFrame({
        "x1": [((i * 37 + 5) % 101) / 100 for i in range(new_rows)],
        "x2": [((i * 17 + 3) % 89) / 100 for i in range(new_rows)],
    })
    new_path = tmp_path / "new_raw.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    scored = runner.invoke(
        ToolRef("modeling", "score_dataset"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert scored.ok is True, scored.error
    assert scored.output["score_col"] == "model_score"
    assert scored.output["points_col"] is None
    assert scored.output["score_direction"] == "higher_is_riskier"
    assert scored.output["row_count"] == new_rows
    assert scored.output["score_missing_rate"] == 0.0

    tool_scored_frame = backend.read_frame(registry.resolve_path(scored.output["result_dataset_id"]))
    tool_scores = tool_scored_frame["model_score"].to_numpy(dtype=float)

    from marvis.packs.modeling.tools import _ModelArtifactScorer

    artifact = ModelingRepository(settings.db_path).get_model_artifact(trained.output["artifact_id"])
    base_dir = Path(settings.tasks_dir) / task.id / "modeling_artifacts"
    direct_scorer = _ModelArtifactScorer(artifact, base_dir=base_dir, replay_preprocessing=True)
    direct_scores = np.asarray(direct_scorer.score(new_frame), dtype=float)

    assert tool_scores.tolist() == pytest.approx(direct_scores.tolist())


def test_score_dataset_registers_derived_dataset_with_direction_metadata_and_audit(tmp_path):
    """score_dataset registers a modeling.dataset.scored audit entry carrying the
    artifact's own direction metadata verbatim (never re-inferred)."""
    runner, _pr, registry, _backend, settings, task = _runtime(tmp_path)
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

    new_frame = frame[["x1", "x2"]].iloc[:50].reset_index(drop=True)
    new_path = tmp_path / "new_data.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    scored = runner.invoke(
        ToolRef("modeling", "score_dataset"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert scored.ok is True, scored.error

    audit_rows = PluginRepository(settings.db_path).list_audit(kind="modeling.dataset.scored")
    assert len(audit_rows) == 1
    entry = audit_rows[0]
    assert entry["target_ref"] == scored.output["result_dataset_id"]
    assert entry["detail"]["source_dataset_id"] == new_dataset.id
    assert entry["detail"]["experiment_id"] == trained.output["experiment_id"]
    assert entry["detail"]["artifact_id"] == trained.output["artifact_id"]
    assert entry["detail"]["score_direction"] == "higher_is_riskier"
    assert entry["detail"]["points_direction"] is None
    assert entry["detail"]["row_count"] == 50


def test_score_dataset_writes_scorecard_points_column(tmp_path):
    """A scorecard artifact's score_dataset run also writes a points column
    (higher_is_better direction), alongside the PD column."""
    runner, _pr, registry, settings_backend, settings, task = _runtime(tmp_path)
    frame = _linearly_separable_frame()
    path = tmp_path / "modeling_sample.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")

    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "scorecard",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error

    new_frame = frame[["x1", "x2"]].iloc[:50].reset_index(drop=True)
    new_path = tmp_path / "new_data.parquet"
    new_frame.to_parquet(new_path, index=False)
    new_dataset = registry.register_existing(new_path, task_id=task.id, role="scoring_input")

    scored = runner.invoke(
        ToolRef("modeling", "score_dataset"),
        {"experiment_id": trained.output["experiment_id"], "dataset_id": new_dataset.id},
        task_id=task.id,
    )
    assert scored.ok is True, scored.error
    assert scored.output["points_col"] == "scorecard_points"
    assert scored.output["points_direction"] == "higher_is_better"
    assert scored.output["points_missing_rate"] == 0.0

    scored_frame = settings_backend.read_frame(registry.resolve_path(scored.output["result_dataset_id"]))
    assert "scorecard_points" in scored_frame.columns
    assert scored_frame["scorecard_points"].notna().all()

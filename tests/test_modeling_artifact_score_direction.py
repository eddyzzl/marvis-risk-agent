"""Regression tests for the S1a ModelArtifact direction metadata (score_direction /
points_direction).

Direction is a compile-time constant of the implementation (see
marvis/packs/modeling/artifact.py::score_direction_for_algorithm /
points_direction_for_algorithm), not a statistical inference -- these tests assert
the constant is correct per algorithm and that it survives save_model() and the
SQLite round trip, including tolerating pre-migration rows with NULL columns.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from marvis.db import ModelingRepository, init_db
from marvis.packs.modeling import Experiment, ModelArtifact, TrainConfig
from marvis.packs.modeling.artifact import (
    points_direction_for_algorithm,
    save_model,
    score_direction_for_algorithm,
)


@pytest.mark.parametrize(
    ("algorithm", "expected_score_direction", "expected_points_direction"),
    [
        ("lgb", "higher_is_riskier", None),
        ("xgb", "higher_is_riskier", None),
        ("lr", "higher_is_riskier", None),
        ("scorecard", "higher_is_riskier", "higher_is_better"),
        ("catboost", "higher_is_riskier", None),
        ("mlp", "higher_is_riskier", None),
        ("lgb_regressor", "higher_is_riskier", None),
        ("lgb_multiclass", "higher_is_riskier", None),
        ("ensemble", "higher_is_riskier", None),
    ],
)
def test_create_artifact_sets_score_direction_per_algorithm(
    algorithm, expected_score_direction, expected_points_direction
):
    assert score_direction_for_algorithm(algorithm) == expected_score_direction
    assert points_direction_for_algorithm(algorithm) == expected_points_direction


def test_save_model_writes_direction_fields_to_artifact_and_model_meta_json(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    model = LogisticRegression().fit(frame[["x1"]], frame["y"])

    artifact = save_model(model, "lr", tmp_path, feature_list=("x1",), params={"C": 1.0})

    assert artifact.score_direction == "higher_is_riskier"
    assert artifact.points_direction is None
    meta = json.loads((tmp_path / f"{artifact.id}.model_meta.json").read_text(encoding="utf-8"))
    assert meta["score_direction"] == "higher_is_riskier"
    assert meta["points_direction"] is None


def test_save_scorecard_model_writes_higher_is_better_points_direction(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    model = LogisticRegression().fit(frame[["x1"]], frame["y"])

    artifact = save_model(
        model,
        "scorecard",
        tmp_path,
        feature_list=("x1",),
        params={"base_score": 600},
        woe_maps={},
        scorecard_table=[],
    )

    assert artifact.score_direction == "higher_is_riskier"
    assert artifact.points_direction == "higher_is_better"
    meta = json.loads((tmp_path / f"{artifact.id}.model_meta.json").read_text(encoding="utf-8"))
    assert meta["score_direction"] == "higher_is_riskier"
    assert meta["points_direction"] == "higher_is_better"


def _train_config() -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("x1",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={},
        seed=1,
        early_stopping_rounds=None,
    )


def _persisted_artifact(*, score_direction, points_direction) -> ModelArtifact:
    return ModelArtifact(
        id="artifact-direction-1",
        experiment_id="experiment-direction-1",
        algorithm="scorecard",
        model_path="models/artifact-direction-1/model.joblib",
        pmml_path=None,
        feature_list=("x1",),
        params={},
        woe_maps=None,
        created_at="2026-07-02T00:00:00Z",
        score_direction=score_direction,
        points_direction=points_direction,
    )


def test_model_artifact_round_trips_direction_fields_through_sqlite(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-direction-1",
        task_id="task-1",
        recipe_id="scorecard",
        config=_train_config(),
        metrics=None,
        artifact_id=None,
        status="trained",
        created_at="2026-07-02T00:00:00Z",
    )
    repo.create_experiment(experiment)
    artifact = _persisted_artifact(score_direction="higher_is_riskier", points_direction="higher_is_better")

    repo.create_model_artifact(artifact)

    fetched = repo.get_model_artifact(artifact.id)
    assert fetched.score_direction == "higher_is_riskier"
    assert fetched.points_direction == "higher_is_better"
    listed = repo.list_model_artifacts(experiment_id=experiment.id)
    assert listed == [fetched]


def test_model_artifact_from_row_tolerates_missing_direction_columns(tmp_path):
    """Old artifacts trained before this migration have NULL score_direction /
    points_direction columns (ALTER TABLE ADD COLUMN does not backfill historical
    rows) -- readers must fall back to None, not raise or fabricate a value."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-direction-2",
        task_id="task-1",
        recipe_id="scorecard",
        config=_train_config(),
        metrics=None,
        artifact_id=None,
        status="trained",
        created_at="2026-07-02T00:00:00Z",
    )
    repo.create_experiment(experiment)
    artifact = ModelArtifact(
        id="artifact-direction-2",
        experiment_id=experiment.id,
        algorithm="scorecard",
        model_path="models/artifact-direction-2/model.joblib",
        pmml_path=None,
        feature_list=("x1",),
        params={},
        woe_maps=None,
        created_at="2026-07-02T00:00:00Z",
        score_direction="higher_is_riskier",
        points_direction="higher_is_better",
    )
    repo.create_model_artifact(artifact)

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE model_artifacts SET score_direction = NULL, points_direction = NULL WHERE id = ?",
        (artifact.id,),
    )
    conn.commit()
    conn.close()

    legacy = repo.get_model_artifact(artifact.id)
    assert legacy.score_direction is None
    assert legacy.points_direction is None

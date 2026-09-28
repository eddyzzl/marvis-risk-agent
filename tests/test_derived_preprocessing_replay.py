"""Derivation must behave identically for training batches and single raw rows."""

import json

import numpy as np
import pandas as pd
import pytest

from marvis.data.preprocessing_evidence import (
    load_preprocessing_state,
    training_preprocessing_state,
)
from marvis.feature.derived_preprocessing import derive_with_parameters
from marvis.feature.errors import FeatureError
from marvis.feature.preprocessing import apply_preprocessing_steps
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime
from tests.test_feature_preprocessing_provenance import scenario


def source_frame():
    return pd.DataFrame(
        {
            "x": [10.0, 10.0, 20.0, 30.0, 900.0, 2000.0],
            "d": [2.0] * 6,
            "g": ["a", "a", None, "b", "a", "unseen"],
            "when": ["2025-01-02"] * 6,
            "split": ["train"] * 4 + ["test", "oot"],
            "y": [0, 1, 0, 1, 0, 1],
        },
        index=[8, 4, 9, 1, 6, 2],
    )


def recipes():
    return [
        {"kind": "ratio", "num": "x", "den": "d"},
        {
            "kind": "transform",
            "col": "x_ratio_d",
            "ops": ["log1p", "rank"],
            "split_col": "split",
        },
        {
            "kind": "agg",
            "group": "g",
            "value": "x",
            "aggs": ["mean", "count", "std"],
            "min_group_size": 1,
            "split_col": "split",
        },
        {"kind": "month", "col": "when"},
    ]


def test_frozen_derivation_replays_each_row_and_preserves_ordinal_membership():
    frame = source_frame()
    derived, columns, steps, fits = derive_with_parameters(
        frame, recipes(), dataset_id="sample", target_col="y"
    )
    assert derived.index.tolist() == frame.index.tolist()
    assert len(fits) == 2
    reloaded = json.loads(json.dumps(steps))
    replayed = pd.concat(
        [
            apply_preprocessing_steps(frame.iloc[[i]], reloaded)
            for i in range(len(frame))
        ]
    )
    pd.testing.assert_frame_equal(derived, replayed)
    assert derived.loc[6, "x_by_g_mean"] == 10.0
    assert derived.loc[2, "x_by_g_mean"] == 17.5
    assert derived.loc[6, "x_ratio_d__rank"] == 1.0
    assert set(columns) <= set(replayed.columns)


def test_evaluation_distribution_cannot_change_rank_or_aggregation_parameters():
    frame = source_frame()
    before = derive_with_parameters(frame, recipes(), dataset_id="sample")[2]
    frame.loc[frame["split"] != "train", ["x", "g"]] = [999999.0, "another_group"]
    after = derive_with_parameters(frame, recipes(), dataset_id="sample")[2]
    assert before == after


@pytest.mark.parametrize(
    "recipe",
    [
        {"kind": "transform", "col": "x", "ops": ["rank"]},
        {"kind": "agg", "group": "g", "value": "x", "aggs": ["mean"]},
    ],
)
def test_population_dependent_recipes_require_explicit_fit_membership(recipe):
    with pytest.raises(FeatureError):
        derive_with_parameters(source_frame(), [recipe], dataset_id="sample")


@pytest.mark.parametrize(
    "recipe",
    [
        {"kind": "ratio", "num": "y", "den": "d"},
        {"kind": "cross", "a": "x", "b": "y", "ops": ["mul"]},
        {"kind": "transform", "col": "y", "ops": ["log1p"]},
        {
            "kind": "agg",
            "group": "y",
            "value": "x",
            "aggs": ["mean"],
            "split_col": "split",
        },
    ],
)
def test_registered_target_cannot_become_a_derived_feature(recipe):
    with pytest.raises(FeatureError, match="target"):
        derive_with_parameters(
            source_frame(), [recipe], dataset_id="sample", target_col="y"
        )


def test_derived_tool_keeps_prior_normalization_and_resplit_protection(tmp_path):
    runner, registry, _, _, transformed = scenario(tmp_path)
    result = runner.invoke(
        ToolRef("feature", "cross_features"),
        {
            "dataset_id": transformed.id,
            "recipe": [
                {
                    "kind": "transform",
                    "col": "x",
                    "ops": ["log1p", "rank"],
                    "split_col": "split",
                }
            ],
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    state = load_preprocessing_state(registry, result.output["result_dataset_id"])
    assert state.assurance == "training_only"
    assert [step["kind"] for step in state.steps] == [
        "normalize",
        "derive",
        "fitted_rank",
    ]
    checked = training_preprocessing_state(
        registry,
        result.output["result_dataset_id"],
        split_col="split",
        train_values="train",
    )
    assert checked == state


def test_any_full_pool_fit_in_recipe_blocks_independent_evaluation(tmp_path):
    runner, registry, _, _ = _runtime(tmp_path)
    frame = source_frame().reset_index(drop=True)
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    mixed = [
        {"kind": "transform", "col": "x", "ops": ["rank"], "split_col": "split"},
        {
            "kind": "agg",
            "group": "g",
            "value": "x",
            "aggs": ["mean"],
            "allow_full_fit": True,
        },
    ]
    result = runner.invoke(
        ToolRef("feature", "cross_features"),
        {"dataset_id": source.id, "recipe": mixed},
        task_id="task-feature",
    )
    assert result.ok, result.error
    assert (
        load_preprocessing_state(registry, result.output["result_dataset_id"]).assurance
        == "exploration"
    )
    with pytest.raises(FeatureError, match="evaluation rows"):
        training_preprocessing_state(
            registry,
            result.output["result_dataset_id"],
            split_col="split",
            train_values="train",
        )


def test_fitted_rank_preserves_training_ties_and_missing_values():
    frame = source_frame()
    frame.loc[6, "x"] = np.nan
    derived, _, steps, _ = derive_with_parameters(
        frame,
        [{"kind": "transform", "col": "x", "ops": ["rank"], "split_col": "split"}],
        dataset_id="sample",
    )
    assert derived["x__rank"].iloc[:4].tolist() == [0.375, 0.375, 0.75, 1.0]
    assert np.isnan(derived.loc[6, "x__rank"])
    pd.testing.assert_frame_equal(derived, apply_preprocessing_steps(frame, steps))


def test_registered_derivation_trains_and_scores_raw_rows_identically(tmp_path):
    from marvis.packs.modeling.scoring import _ModelArtifactScorer
    from marvis.repositories.modeling import ModelingRepository
    from tests.test_modeling_pack import _runtime as modeling_runtime

    runner, _, registry, _, settings, task = modeling_runtime(tmp_path)
    frame = pd.concat([source_frame()] * 20, ignore_index=True)
    path = tmp_path / "training.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="sample")
    derived = runner.invoke(
        ToolRef("feature", "cross_features"),
        {
            "dataset_id": dataset.id,
            "recipe": recipes()[:-1],
        },
        task_id=task.id,
    )
    assert derived.ok, derived.error
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": derived.output["result_dataset_id"],
            "recipe": "lr",
            "features": ["x_ratio_d", "x_ratio_d__rank", "x_by_g_mean"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 7,
        },
        task_id=task.id,
    )
    assert trained.ok, trained.error
    artifact = ModelingRepository(settings.db_path).get_model_artifact(
        trained.output["artifact_id"]
    )
    base = settings.tasks_dir / task.id / "modeling_artifacts"
    raw = _ModelArtifactScorer(artifact, base_dir=base, replay_preprocessing=True)
    batch = _ModelArtifactScorer(artifact, base_dir=base)
    materialized = registry.read_authenticated_parquet_snapshot(
        derived.output["result_dataset_id"]
    )
    expected = batch.score(materialized)
    assert raw.score(frame) == pytest.approx(expected)
    assert [raw.score(frame.iloc[[i]])[0] for i in range(6)] == pytest.approx(
        expected[:6]
    )


def test_aggregate_does_not_merge_large_integer_group_identities():
    from marvis.feature.derived_parameters import (
        fit_aggregate_parameters,
        apply_aggregate_parameters,
    )

    frame = pd.DataFrame({"g": [2**53, 2**53 + 1], "x": [10.0, 99.0]})
    params = fit_aggregate_parameters(frame, "g", "x", ["mean"], min_group_size=1)
    restored = json.loads(json.dumps(params))
    result, _ = apply_aggregate_parameters(frame, restored)
    assert result["x_by_g_mean"].tolist() == [10.0, 99.0]

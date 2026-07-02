import json

import numpy as np
import pandas as pd
import pytest

from marvis.agent.modeling_setup import build_modeling_proposal
from marvis.data.backend import DataBackend
from marvis.data.errors import NanLabelNotConfirmedError
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling import ModelingError
from marvis.packs.modeling.artifact import load_model
from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.recipes import get_recipe, list_recipes
from marvis.packs.modeling.recipes.common import (
    cat_feature_indices,
    compute_model_metrics,
    compute_multiclass_metrics,
    resolve_auto_scale_pos_weight,
    sample_weight_values,
    split_modeling_frame,
)
from marvis.packs.modeling.recipes.catboost import train_catboost
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lgb_multiclass import train_lgb_multiclass
from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.mlp import train_mlp
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
from marvis.packs.modeling.tools import _ModelArtifactScorer
from marvis.packs.modeling.tune import _lgb_base_params, _trial_score, tune_hyperparameters
from marvis.settings import build_settings


def _config() -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("score",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={},
        seed=13,
        early_stopping_rounds=None,
    )


def test_recipe_registry_exposes_builtin_classification_and_regression_recipes():
    recipes = list_recipes()

    assert [recipe.id for recipe in recipes] == [
        "lgb", "xgb", "catboost", "lr", "scorecard", "lgb_regressor", "mlp", "lgb_multiclass",
    ]
    assert get_recipe("catboost").algorithm == "catboost"
    assert get_recipe("scorecard").requires_woe is True
    assert get_recipe("lgb_regressor").algorithm == "lgb_regressor"
    assert get_recipe("lgb_multiclass").algorithm == "lgb_multiclass"
    assert get_recipe("lgb_multiclass").requires_woe is False
    assert get_recipe("lr").algorithm == "lr"
    with pytest.raises(KeyError):
        get_recipe("unknown")


def test_split_modeling_frame_requires_train_and_test_splits():
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.9],
        "y": [0, 0, 1],
        "split": ["train", "train", "oot"],
    })

    with pytest.raises(ModelingError, match="missing test split"):
        split_modeling_frame(frame, _config())


def test_compute_model_metrics_uses_platform_feature_metrics_and_overfitting():
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.25, 0.75, 0.3, 0.7, 0.35, 0.65],
        "y": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "split": ["train"] * 4 + ["test"] * 4 + ["oot"] * 4,
    })
    train, test, oot = split_modeling_frame(frame, _config())

    metrics = compute_model_metrics(
        lambda data: data["score"].to_numpy(dtype=float),
        train,
        test,
        oot,
        _config(),
    )

    train_scores = train["score"].to_numpy(dtype=float)
    train_target = train["y"].to_numpy(dtype=int)
    assert metrics.train_ks == pytest.approx(feature_ks(train_scores, train_target))
    assert metrics.train_auc == pytest.approx(feature_auc(train_scores, train_target))
    assert metrics.psi_test_vs_train is not None
    assert metrics.psi_oot_vs_train is not None
    assert metrics.overfit_flag is False


def test_compute_model_metrics_reports_weighted_binary_metrics():
    frame = pd.DataFrame({
        "score": [0.3, 0.4, 0.8, 0.9, 0.3, 0.4, 0.8, 0.9, 0.2, 0.7, 0.6, 0.95],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "weight": [1, 10, 10, 1, 1, 10, 10, 1, 1, 2, 3, 4],
        "split": ["train"] * 4 + ["test"] * 4 + ["oot"] * 4,
    })
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("score",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"sample_weight_col": "weight"},
        seed=13,
        early_stopping_rounds=None,
    )
    train, test, oot = split_modeling_frame(frame, config)

    metrics = compute_model_metrics(
        lambda data: data["score"].to_numpy(dtype=float),
        train,
        test,
        oot,
        config,
    )

    assert metrics.weighted_train_auc is not None
    assert metrics.weighted_test_auc is not None
    assert metrics.weighted_oot_auc is not None
    assert metrics.weighted_psi_test_vs_train is not None
    assert metrics.weighted_psi_oot_vs_train is not None
    assert metrics.weighted_test_auc != pytest.approx(metrics.test_auc)


def test_weighted_binary_metrics_have_formula_level_values_and_weight_validation():
    frame = pd.DataFrame({
        "score": [0.1, 0.4, 0.4, 0.9, 0.1, 0.4, 0.4, 0.9, 0.3, 0.8],
        "y": [0, 0, 1, 1, 0, 0, 1, 1, np.nan, np.nan],
        "weight": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "split": ["train"] * 4 + ["test"] * 4 + ["oot"] * 2,
    })
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("score",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"sample_weight_col": "weight"},
        seed=13,
        early_stopping_rounds=None,
    )
    train, test, oot = split_modeling_frame(frame, config)

    metrics = compute_model_metrics(
        lambda data: data["score"].to_numpy(dtype=float),
        train,
        test,
        oot,
        config,
        oot_has_labels=False,
    )

    assert metrics.weighted_train_auc == pytest.approx(12 / 14)
    assert metrics.weighted_train_ks == pytest.approx(4 / 7)
    assert metrics.weighted_test_auc == pytest.approx(12 / 14)
    assert metrics.weighted_psi_test_vs_train == pytest.approx(0.0)
    assert metrics.weighted_oot_ks is None
    assert metrics.weighted_oot_auc is None

    bad_weights = train.copy()
    bad_weights.loc[bad_weights.index[0], "weight"] = 0
    with pytest.raises(ModelingError, match="non-positive"):
        sample_weight_values(bad_weights, config)


def test_auto_scale_pos_weight_uses_effective_train_weights():
    frame = pd.DataFrame({
        "score": [0.1, 0.2, 0.8, 0.9, 0.3],
        "y": [0, 0, 0, 1, 1],
        "weight": [1.0, 2.0, 3.0, 2.0, 4.0],
        "split": ["train", "train", "train", "train", "test"],
    })
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("score",),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test"},
        params={"scale_pos_weight": "auto", "sample_weight_col": "weight"},
        seed=13,
        early_stopping_rounds=None,
    )
    train, _test, _oot = split_modeling_frame(frame, config)

    params = resolve_auto_scale_pos_weight({"scale_pos_weight": "auto"}, train, config)

    assert params["scale_pos_weight"] == pytest.approx(3.0)
    assert _lgb_base_params({"scale_pos_weight": "auto"}, pos_weight_hint=7.5)["scale_pos_weight"] == 7.5


def test_tune_trial_score_does_not_use_oot_holdout_for_selection():
    stable = _trial_score(
        train_ks=0.45,
        test_ks=0.40,
        overfit_penalty=0.5,
    )
    fragile = _trial_score(
        train_ks=0.45,
        test_ks=0.40,
        overfit_penalty=0.5,
    )

    assert stable == pytest.approx(0.40 - 0.5 * 0.05)
    assert fragile == stable


def test_train_lr_writes_artifact_and_is_seed_reproducible(tmp_path):
    rows = 180
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 100 + ["test"] * 50 + ["oot"] * 30,
    })
    path = tmp_path / "lr_sample.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"C": 0.7},
        seed=23,
        early_stopping_rounds=None,
    )

    backend = DataBackend(tmp_path)
    first = train_lr(backend, path, config, out_dir=tmp_path / "models_a")
    second = train_lr(backend, path, config, out_dir=tmp_path / "models_b")

    assert (tmp_path / "models_a" / first.artifact.model_path).exists()
    meta = json.loads((tmp_path / "models_a" / f"{first.artifact.id}.model_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 23
    assert meta["dataset_id"] == "dataset-1"
    assert meta["target_col"] == "y"
    assert meta["split_col"] == "split"
    assert meta["split_values"] == {"train": "train", "test": "test", "oot": "oot"}
    assert meta["feature_list"] == ["x1", "x2"]
    assert first.artifact.algorithm == "lr"
    assert first.artifact.params["C"] == 0.7
    assert first.artifact.feature_list == ("x1", "x2")
    assert first.metrics.test_auc == pytest.approx(second.metrics.test_auc)
    assert [item[0] for item in first.feature_importance] == [item[0] for item in second.feature_importance]
    assert [item[1] for item in first.feature_importance] == pytest.approx(
        [item[1] for item in second.feature_importance],
    )


def test_train_mlp_writes_artifact_and_is_seed_reproducible(tmp_path):
    rows = 180
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 100 + ["test"] * 50 + ["oot"] * 30,
    })
    path = tmp_path / "mlp_sample.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"hidden_layer_sizes": [8], "max_iter": 60},  # small/fast for the test
        seed=23,
        early_stopping_rounds=None,
    )

    backend = DataBackend(tmp_path)
    first = train_mlp(backend, path, config, out_dir=tmp_path / "mlp_a")
    second = train_mlp(backend, path, config, out_dir=tmp_path / "mlp_b")

    assert (tmp_path / "mlp_a" / first.artifact.model_path).exists()
    assert first.artifact.algorithm == "mlp"
    assert first.artifact.feature_list == ("x1", "x2")
    assert first.metrics.test_auc == pytest.approx(second.metrics.test_auc)  # seed-reproducible
    assert first.feature_importance == ()  # MLP exposes no native per-feature importance


def test_train_lgb_and_xgb_write_artifacts_and_are_seed_reproducible(tmp_path):
    rows = 240
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })
    path = tmp_path / "tree_sample.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)

    lgb_config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 6, "learning_rate": 0.1, "num_leaves": 4},
        seed=31,
        early_stopping_rounds=None,
    )
    xgb_config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 6, "max_depth": 2, "eta": 0.1},
        seed=31,
        early_stopping_rounds=None,
    )

    first_lgb = train_lgb(backend, path, lgb_config, out_dir=tmp_path / "lgb_a")
    second_lgb = train_lgb(backend, path, lgb_config, out_dir=tmp_path / "lgb_b")
    first_xgb = train_xgb(backend, path, xgb_config, out_dir=tmp_path / "xgb_a")
    second_xgb = train_xgb(backend, path, xgb_config, out_dir=tmp_path / "xgb_b")

    assert (tmp_path / "lgb_a" / first_lgb.artifact.model_path).exists()
    assert (tmp_path / "xgb_a" / first_xgb.artifact.model_path).exists()
    assert not (tmp_path / "lgb_a" / ".staging").exists()
    assert not (tmp_path / "xgb_a" / ".staging").exists()
    assert first_lgb.artifact.algorithm == "lgb"
    assert first_xgb.artifact.algorithm == "xgb"
    assert first_lgb.metrics.test_auc == pytest.approx(second_lgb.metrics.test_auc)
    assert first_xgb.metrics.test_auc == pytest.approx(second_xgb.metrics.test_auc)
    assert [item[0] for item in first_lgb.feature_importance] == ["x1", "x2"]
    assert [item[0] for item in first_xgb.feature_importance] == ["x1", "x2"]


def test_train_catboost_uses_native_categorical_features_without_float_coercion(tmp_path):
    """PREP-3/FS-3: catboost must consume string columns natively (no strong float cast,
    no crash) and auto-detect them as cat_features when the caller doesn't override."""
    rows = 180
    frame = pd.DataFrame({
        "chan": (["A", "B", "C"] * (rows // 3 + 1))[:rows],
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "y": [1 if i % 5 == 0 else 0 for i in range(rows)],
        "split": ["train"] * 100 + ["test"] * 50 + ["oot"] * 30,
    })
    path = tmp_path / "catboost_categorical.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("chan", "x1"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"iterations": 6},
        seed=11,
        early_stopping_rounds=None,
    )

    result = train_catboost(backend, path, config, out_dir=tmp_path / "catboost_native")

    assert (tmp_path / "catboost_native" / result.artifact.model_path).exists()
    assert result.artifact.params["cat_features"] == ["chan"]
    assert [name for name, _ in result.feature_importance] == ["chan", "x1"]
    assert 0.0 <= result.metrics.test_ks <= 1.0

    # scoring the persisted model on new rows works without any manual encoding
    model = load_model(result.artifact, base_dir=tmp_path / "catboost_native")
    scored = model.predict_proba(frame[["chan", "x1"]].head(10))
    assert scored.shape == (10, 2)


def test_train_catboost_accepts_explicit_cat_features_override(tmp_path):
    rows = 120
    frame = pd.DataFrame({
        "chan": (["A", "B"] * (rows // 2 + 1))[:rows],
        "x1": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 4 == 0 else 0 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "catboost_explicit.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("chan", "x1"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"iterations": 5, "cat_features": ["chan"]},
        seed=5,
        early_stopping_rounds=None,
    )

    result = train_catboost(backend, path, config, out_dir=tmp_path / "catboost_explicit")

    assert result.artifact.params["cat_features"] == ["chan"]


def test_cat_feature_indices_auto_detects_non_numeric_columns_and_resolves_explicit_names():
    frame = pd.DataFrame({"chan": ["A", "B"], "region": ["east", "west"], "x1": [1.0, 2.0]})
    features = ["chan", "region", "x1"]

    assert cat_feature_indices(frame, features, None) == [0, 1]
    assert cat_feature_indices(frame, features, ["region"]) == [1]
    assert cat_feature_indices(frame, features, [0, 2]) == [0, 2]

    with pytest.raises(ModelingError, match="cat_features column not in features"):
        cat_feature_indices(frame, features, ["missing_col"])


def test_train_lgb_resolves_auto_scale_pos_weight_before_fit(tmp_path):
    rows = 180
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 9 == 0 else 0 for i in range(rows)],
        "split": ["train"] * 100 + ["test"] * 50 + ["oot"] * 30,
    })
    path = tmp_path / "lgb_auto_weight.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 3, "learning_rate": 0.1, "num_leaves": 4, "scale_pos_weight": "auto"},
        seed=31,
        early_stopping_rounds=None,
    )

    result = train_lgb(DataBackend(tmp_path), path, config, out_dir=tmp_path / "lgb_auto")

    assert result.artifact.params["scale_pos_weight"] == pytest.approx(88 / 12)
    assert result.artifact.params["num_boost_round"] == 3


def test_train_lgb_uses_shared_nan_label_gate(tmp_path):
    rows = 160
    labels = [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)]
    labels[3] = np.nan
    labels[112] = np.nan
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": labels,
        "split": ["train"] * 100 + ["test"] * 40 + ["oot"] * 20,
    })
    path = tmp_path / "lgb_nan_labels.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 4, "learning_rate": 0.1, "num_leaves": 4},
        seed=37,
        early_stopping_rounds=None,
        drop_nan_labels=True,
    )

    result = train_lgb(DataBackend(tmp_path), path, config, out_dir=tmp_path / "lgb_nan")

    assert result.nan_labels_dropped == 2
    assert result.metrics.test_auc is not None


def test_train_lgb_blocks_train_test_nan_labels_without_confirmation(tmp_path):
    rows = 160
    labels = [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)]
    labels[3] = np.nan
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": labels,
        "split": ["train"] * 100 + ["test"] * 40 + ["oot"] * 20,
    })
    path = tmp_path / "lgb_nan_unconfirmed.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 4, "learning_rate": 0.1, "num_leaves": 4},
        seed=37,
        early_stopping_rounds=None,
        drop_nan_labels=False,
    )

    with pytest.raises(NanLabelNotConfirmedError):
        train_lgb(DataBackend(tmp_path), path, config, out_dir=tmp_path / "lgb_nan_unconfirmed")


def test_tune_hyperparameters_gates_nan_train_label(tmp_path):
    """DOM-1: tune_hyperparameters must not silently tune on NaN-polluted labels — the
    shared NaN-label confirmation gate applies here exactly as it does in train_lgb."""
    rows = 160
    labels = [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)]
    labels[3] = np.nan
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": labels,
        "split": ["train"] * 100 + ["test"] * 40 + ["oot"] * 20,
    })
    path = tmp_path / "tune_nan_labels.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    tune_kwargs = dict(
        features=["x1", "x2"],
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        n_trials=2,
        seed=37,
        early_stopping_rounds=5,
        max_boost_round=10,
    )

    with pytest.raises(NanLabelNotConfirmedError):
        tune_hyperparameters(backend, path, drop_nan_labels=False, **tune_kwargs)

    result = tune_hyperparameters(backend, path, drop_nan_labels=True, **tune_kwargs)

    assert result.nan_labels_dropped == 1
    assert result.n_trials == 2
    assert result.best_metrics["test_ks"] is not None


def _tune_fixture_frame(rows: int = 400) -> pd.DataFrame:
    """A modestly-sized, deterministic frame for exercising the two-stage search."""
    return pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "x3": [((i * 53) % 97) / 100 for i in range(rows)],
        "y": [1 if (i * 37 + i * 17) % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * int(rows * 0.6) + ["test"] * int(rows * 0.3) + ["oot"] * (
            rows - int(rows * 0.6) - int(rows * 0.3)
        ),
    })


def _tune_kwargs(**overrides) -> dict:
    base = dict(
        features=["x1", "x2", "x3"],
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        n_trials=10,
        seed=7,
        early_stopping_rounds=5,
        max_boost_round=50,
    )
    base.update(overrides)
    return base


def test_tune_hyperparameters_is_deterministic_across_runs(tmp_path):
    """TUNE-2 regression (a): same seed -> identical trial sequence and results,
    across the whole two-stage search (coarse + fine)."""
    frame = _tune_fixture_frame()
    path = tmp_path / "tune_determinism.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    kwargs = _tune_kwargs()

    first = tune_hyperparameters(backend, path, **kwargs)
    second = tune_hyperparameters(backend, path, **kwargs)

    assert first.n_trials == second.n_trials == 10
    assert [t["search_stage"] for t in first.trials] == [t["search_stage"] for t in second.trials]
    assert [t["params"] for t in first.trials] == [t["params"] for t in second.trials]
    assert [t["score"] for t in first.trials] == pytest.approx([t["score"] for t in second.trials])
    assert first.best_params == second.best_params
    assert first.best_metrics == second.best_metrics


def test_tune_hyperparameters_lgb_default_recipe_matches_historical_contract(tmp_path):
    """TUNE-1 regression (d): generalising tune.py to every recipe family must not
    change the lgb single-recipe contract — recipe defaults to "lgb", the returned
    TuneResult keeps the exact historical flat shape (best_params/best_metrics/
    trials/n_trials, no per-recipe nesting leaking into the dataclass), and every
    trial record still carries the original key set (plus the new recipe-agnostic
    best_iteration key only where lgb already exposed the concept implicitly via
    num_boost_round)."""
    frame = _tune_fixture_frame()
    path = tmp_path / "tune_lgb_contract.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)

    result = tune_hyperparameters(backend, path, **_tune_kwargs(n_trials=6))

    assert result.recipe == "lgb"
    assert isinstance(result.best_params, dict)
    assert isinstance(result.best_metrics, dict)
    assert isinstance(result.trials, tuple)
    assert result.n_trials == 6
    # historical lgb best_metrics keys (no xgb/catboost-only additions leak in
    # when best_iteration is absent for lgb — lgb reports round count via params).
    assert {"train_ks", "test_ks", "oot_ks", "overfit_gap", "overfit_gap_oot",
            "oot_stability_gap", "train_auc", "test_auc", "oot_auc"} <= set(result.best_metrics.keys())
    # lgb has always resolved a best_iteration internally (booster.best_iteration
    # feeds num_boost_round); the generalized engine now surfaces it explicitly
    # alongside the historical keys, matching the other tree recipes' evidence.
    assert "best_iteration" in result.best_metrics
    for trial in result.trials:
        assert {"params", "train_ks", "test_ks", "oot_ks", "score", "train_auc",
                "test_auc", "oot_auc", "search_stage"} <= set(trial.keys())
        assert "best_iteration" in trial  # lgb trials always resolve a best_iteration
        assert "num_boost_round" in trial["params"]


def test_tune_hyperparameters_runs_two_search_stages_with_fine_near_best_coarse(tmp_path):
    """TUNE-2 regression (b): trials carry a coarse/fine search_stage label, both
    stages actually run under the default 60/40 split, and every fine-stage trial's
    numeric params land inside the shrunk neighbourhood of the best coarse trial
    (log-neighbourhood for log-scaled params, linear-neighbourhood for linear ones) —
    proof the second stage is a genuine local refinement, not another full-space draw."""
    frame = _tune_fixture_frame()
    path = tmp_path / "tune_two_stage.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    result = tune_hyperparameters(backend, path, **_tune_kwargs(n_trials=10))

    stages = [t["search_stage"] for t in result.trials]
    assert set(stages) == {"coarse", "fine"}
    assert stages.count("coarse") == 6  # round(10 * 0.6)
    assert stages.count("fine") == 4

    coarse_trials = [t for t in result.trials if t["search_stage"] == "coarse"]
    fine_trials = [t for t in result.trials if t["search_stage"] == "fine"]
    best_coarse = max(coarse_trials, key=lambda t: t["score"])
    anchor = best_coarse["params"]

    from marvis.packs.modeling.tune import _FINE_LINEAR_SHRINK, _FINE_LOG_SHRINK, _LINEAR_SPACE_BOUNDS, _LOG_SPACE_BOUNDS

    for trial in fine_trials:
        params = trial["params"]
        # categorical-ish params stay pinned to the coarse anchor
        assert params["max_depth"] == anchor["max_depth"]
        assert params["scale_pos_weight"] == anchor["scale_pos_weight"]
        for name, (low, high) in _LOG_SPACE_BOUNDS.items():
            log_low, log_high = np.log10(low), np.log10(high)
            log_center = np.log10(max(anchor[name], low))
            span = (log_high - log_low) * _FINE_LOG_SHRINK
            lo_bound = max(log_low, log_center - span)
            hi_bound = min(log_high, log_center + span)
            assert lo_bound - 1e-9 <= np.log10(params[name]) <= hi_bound + 1e-9, (name, params[name])
        for name, (low, high) in _LINEAR_SPACE_BOUNDS.items():
            span = (high - low) * _FINE_LINEAR_SHRINK
            lo_bound = max(low, anchor[name] - span)
            hi_bound = min(high, anchor[name] + span)
            assert lo_bound - 1e-9 <= params[name] <= hi_bound + 1e-9, (name, params[name])


def test_tune_hyperparameters_lambda_sampling_spans_multiple_orders_of_magnitude(tmp_path):
    """TUNE-2 regression (c): lambda_l1/lambda_l2 are drawn log-uniformly over
    1e-8..10, not linearly over 0..20 — evidence is that repeated trials cover
    several orders of magnitude instead of clustering in the >10 tail."""
    import math

    frame = _tune_fixture_frame()
    path = tmp_path / "tune_lambda_space.parquet"
    frame.to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    result = tune_hyperparameters(backend, path, **_tune_kwargs(n_trials=20, seed=3))

    lambda_l1_values = [t["params"]["lambda_l1"] for t in result.trials]
    lambda_l2_values = [t["params"]["lambda_l2"] for t in result.trials]

    assert all(0.0 <= v <= 10.0 for v in lambda_l1_values + lambda_l2_values)
    orders_seen = {math.floor(math.log10(v)) for v in lambda_l1_values + lambda_l2_values if v > 0}
    assert len(orders_seen) >= 3, f"expected log-uniform spread across orders of magnitude, got {orders_seen}"

    learning_rates = [t["params"]["learning_rate"] for t in result.trials]
    assert all(0.01 <= lr <= 0.3 for lr in learning_rates)
    assert max(learning_rates) > 0.08  # old space capped at ~0.08; new upper bound is 0.3


def test_tune_hyperparameters_default_n_trials_resolves_per_recipe():
    """TUNE-1: n_trials defaults to None and resolves via DEFAULT_TRIAL_BUDGET
    per recipe (lgb keeps its historical 40; lr/scorecard/mlp get a smaller
    12-trial budget, tree challengers xgb/catboost also get 40)."""
    import inspect

    from marvis.packs.modeling.tune import DEFAULT_TRIAL_BUDGET

    sig = inspect.signature(tune_hyperparameters)
    assert sig.parameters["n_trials"].default is None
    assert sig.parameters["recipe"].default == "lgb"
    assert DEFAULT_TRIAL_BUDGET == {
        "lgb": 40, "xgb": 40, "catboost": 40, "lr": 12, "scorecard": 12, "mlp": 12,
    }


def test_train_lgb_all_nan_oot_is_scoring_only(tmp_path):
    rows = 160
    labels = [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)]
    for index in range(140, rows):
        labels[index] = np.nan
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": labels,
        "split": ["train"] * 100 + ["test"] * 40 + ["oot"] * 20,
    })
    path = tmp_path / "lgb_scoring_oot.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 4, "learning_rate": 0.1, "num_leaves": 4},
        seed=41,
        early_stopping_rounds=None,
        drop_nan_labels=False,
    )

    result = train_lgb(DataBackend(tmp_path), path, config, out_dir=tmp_path / "lgb_scoring_oot")

    assert result.nan_labels_dropped == 0
    assert result.metrics.oot_ks is None
    assert result.metrics.oot_auc is None


def test_train_scorecard_writes_woe_artifact_and_is_seed_reproducible(tmp_path):
    rows = 240
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 7 in {0, 1, 2} else 0 for i in range(rows)],
        "split": ["train"] * 140 + ["test"] * 60 + ["oot"] * 40,
    })
    path = tmp_path / "scorecard_sample.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"C": 0.8, "max_depth": 3, "scorecard_max_bins": 4},
        seed=41,
        early_stopping_rounds=None,
    )
    backend = DataBackend(tmp_path)

    first = train_scorecard(backend, path, config, out_dir=tmp_path / "scorecard_a")
    second = train_scorecard(backend, path, config, out_dir=tmp_path / "scorecard_b")

    assert (tmp_path / "scorecard_a" / first.artifact.model_path).exists()
    assert not (tmp_path / "scorecard_a" / ".staging").exists()
    assert first.artifact.algorithm == "scorecard"
    assert set(first.artifact.woe_maps or {}) == {"x1", "x2"}
    assert first.artifact.params["base_score"] == 600
    assert first.artifact.params["scorecard_base_points"] == pytest.approx(
        second.artifact.params["scorecard_base_points"]
    )
    assert first.artifact.params["enforce_monotonic"] is True
    assert set(first.artifact.params["monotonic_directions"]) == {"x1", "x2"}
    assert "max_depth" not in first.artifact.params
    assert first.artifact.scorecard_table
    assert first.artifact.scorecard_table[0]["feature"] == "__base__"
    assert first.artifact.scorecard_table[0]["points"] == pytest.approx(
        first.artifact.params["scorecard_base_points"]
    )
    detail_rows = [row for row in first.artifact.scorecard_table if row["feature"] != "__base__"]
    assert {"feature", "bin_label", "points"} <= set(detail_rows[0])
    assert detail_rows[0]["monotonic_direction"] in {"increasing", "decreasing"}
    assert max(abs(float(row["points"])) for row in first.artifact.scorecard_table) > 0
    assert {row["feature"] for row in detail_rows} == {"x1", "x2"}
    for feature in ("x1", "x2"):
        rows = sorted(
            (row for row in detail_rows if row["feature"] == feature and row["bin_index"] >= 0),
            key=lambda row: row["bin_index"],
        )
        bad_rates = [float(row["bad_rate"]) for row in rows]
        assert (
            all(left <= right for left, right in zip(bad_rates, bad_rates[1:]))
            or all(left >= right for left, right in zip(bad_rates, bad_rates[1:]))
        )
    assert first.metrics.test_auc == pytest.approx(second.metrics.test_auc)
    assert [item[0] for item in first.feature_importance] == [item[0] for item in second.feature_importance]
    assert [item[1] for item in first.feature_importance] == pytest.approx(
        [item[1] for item in second.feature_importance],
    )
    meta = json.loads((tmp_path / "scorecard_a" / f"{first.artifact.id}.model_meta.json").read_text(encoding="utf-8"))
    assert meta["scorecard_table"][0]["feature"] == "__base__"
    loaded = load_model(first.artifact, base_dir=tmp_path / "scorecard_a")
    assert loaded["scorecard_table"] == list(first.artifact.scorecard_table)
    scorer = _ModelArtifactScorer(first.artifact, base_dir=tmp_path / "scorecard_a")
    pd_scores = scorer.score(frame.head(10))
    assert all(0.0 <= score <= 1.0 for score in pd_scores)
    point_scores = scorer.scorecard_points(frame.head(10))
    assert point_scores is not None
    assert min(point_scores) > 100
    assert max(point_scores) < 1000
    first_row = frame.head(1).iloc[0]
    manual_points = float(first.artifact.scorecard_table[0]["points"])
    for feature in ("x1", "x2"):
        value = float(first_row[feature])
        match = next(
            row for row in detail_rows
            if row["feature"] == feature
            and (row["lower"] is None or value >= float(row["lower"]))
            and (row["upper"] is None or value < float(row["upper"]))
        )
        manual_points += float(match["points"])
    assert manual_points == pytest.approx(point_scores[0])


def test_train_lgb_regressor_writes_artifact_and_computes_regression_metrics(tmp_path):
    rows = 180
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "income": [3500 + (((i * 37) % 101) * 38) + (((i * 17) % 89) * 9) for i in range(rows)],
        "split": ["train"] * 100 + ["test"] * 50 + ["oot"] * 30,
    })
    path = tmp_path / "income_sample.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="income",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 8, "learning_rate": 0.1, "num_leaves": 4},
        seed=47,
        early_stopping_rounds=None,
        recipe_id="lgb_regressor",
        scenario_id="income",
        target_type="continuous",
        eval_metric="rmse_mae",
    )
    backend = DataBackend(tmp_path)

    first = train_lgb_regressor(backend, path, config, out_dir=tmp_path / "income_a")
    second = train_lgb_regressor(backend, path, config, out_dir=tmp_path / "income_b")

    assert (tmp_path / "income_a" / first.artifact.model_path).exists()
    assert first.artifact.algorithm == "lgb_regressor"
    assert first.metrics.train_ks is None
    assert first.metrics.test_auc is None
    assert first.metrics.test_rmse is not None
    assert first.metrics.test_rmse > 0
    assert first.metrics.test_mae is not None
    assert -1.0 <= first.metrics.test_r2 <= 1.0
    assert first.metrics.test_rmse == pytest.approx(second.metrics.test_rmse)
    assert [item[0] for item in first.feature_importance] == ["x1", "x2"]


def test_train_lgb_multiclass_writes_artifact_and_computes_multiclass_metrics(tmp_path):
    rows = 300
    grade = [0 if ((i * 37) % 101) < 33 else (1 if ((i * 37) % 101) < 67 else 2) for i in range(rows)]
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "grade": grade,
        "split": ["train"] * 180 + ["test"] * 70 + ["oot"] * 50,
    })
    path = tmp_path / "grade_sample.parquet"
    frame.to_parquet(path, index=False)
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="grade",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_boost_round": 10, "learning_rate": 0.2, "num_leaves": 8},
        seed=47,
        early_stopping_rounds=None,
        recipe_id="lgb_multiclass",
        target_type="multiclass",
    )
    backend = DataBackend(tmp_path)

    first = train_lgb_multiclass(backend, path, config, out_dir=tmp_path / "grade_a")
    second = train_lgb_multiclass(backend, path, config, out_dir=tmp_path / "grade_b")

    assert (tmp_path / "grade_a" / first.artifact.model_path).exists()
    assert first.artifact.algorithm == "lgb_multiclass"
    assert first.artifact.params["classes"] == [0, 1, 2]
    # multiclass metrics are populated; binary / regression fields stay None
    assert first.metrics.test_macro_auc is not None
    assert 0.0 <= first.metrics.test_accuracy <= 1.0
    assert first.metrics.test_logloss is not None
    assert first.metrics.test_ks is None
    assert first.metrics.test_auc is None
    assert first.metrics.test_rmse is None
    assert first.metrics.oot_macro_auc is not None
    # seed reproducible (bit-identical macro_auc across two runs)
    assert first.metrics.test_macro_auc == second.metrics.test_macro_auc
    assert first.metrics.test_logloss == pytest.approx(second.metrics.test_logloss)
    assert [item[0] for item in first.feature_importance] == ["x1", "x2"]
    # artifact params (including per_class detail) are strict-JSON serialisable
    json.dumps(first.artifact.params, allow_nan=False)
    assert get_recipe("lgb_multiclass").algorithm == "lgb_multiclass"


def test_compute_multiclass_metrics_is_reasonable_and_json_safe():
    classes = (0, 1, 2)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    proba = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
        [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6],
    ])

    metrics = compute_multiclass_metrics(proba, y_true, classes)

    assert metrics["macro_auc"] == pytest.approx(1.0)
    assert metrics["weighted_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["logloss"] is not None and metrics["logloss"] > 0
    assert metrics["macro_recall"] == pytest.approx(1.0)
    assert set(metrics["per_class"]) == {"0", "1", "2"}
    assert metrics["per_class"]["1"]["support"] == 3
    # strict JSON (no NaN/inf tokens)
    json.dumps(metrics, allow_nan=False)


def test_compute_multiclass_metrics_degenerates_safely_for_single_class():
    classes = (0, 1, 2)
    y_true = np.array([1, 1, 1])
    proba = np.array([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1], [0.3, 0.5, 0.2]])

    metrics = compute_multiclass_metrics(proba, y_true, classes)

    # AUC is undefined with a single observed class -> None (not NaN)
    assert metrics["macro_auc"] is None
    assert metrics["weighted_auc"] is None
    # logloss/accuracy are still defined and finite
    assert metrics["logloss"] is not None
    assert 0.0 <= metrics["accuracy"] <= 1.0
    # classes 0 and 2 have no support -> recall None, support 0
    assert metrics["per_class"]["0"]["support"] == 0
    assert metrics["per_class"]["0"]["recall"] is None
    json.dumps(metrics, allow_nan=False)


def _proposal_runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(repo, backend, settings.datasets_dir)
    return backend, registry


def test_build_modeling_proposal_derives_continuous_target_type_from_regressor(tmp_path):
    """A lgb_regressor recipe ⇒ target_type 'continuous'; the numeric `income` column is
    picked as the target, bad_rate is None, and the primary recipe is lgb_regressor."""
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "income": [3500 + (((i * 37) % 101) * 38) for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "income_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-reg", path, role="sample")

    proposal = build_modeling_proposal(
        registry, backend, "task-reg", tmp_path, recipes=["lgb_regressor"]
    )

    assert proposal.target_type == "continuous"
    assert proposal.recipe == "lgb_regressor"
    assert proposal.recipes == ["lgb_regressor"]
    assert proposal.target_col == "income"
    assert proposal.bad_rate is None
    assert "income" not in proposal.feature_cols
    slots = proposal.template_slots()
    assert slots["target_type"] == "continuous"
    assert slots["selection_policy"] == {"require_pmml": False, "require_handoff": False}


def test_build_modeling_proposal_uses_explicit_target_type_default_recipe(tmp_path):
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "income": [3500 + (((i * 37) % 101) * 38) for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "explicit_income_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-explicit-reg", path, role="sample")

    proposal = build_modeling_proposal(
        registry, backend, "task-explicit-reg", tmp_path, target_type="continuous"
    )

    assert proposal.target_type == "continuous"
    assert proposal.recipe == "lgb_regressor"
    assert proposal.recipes == ["lgb_regressor"]
    assert proposal.target_col == "income"


def test_build_modeling_proposal_stays_binary_for_classification_recipes(tmp_path):
    """A classification recipe leaves target_type 'binary' (default) and resolves the 0/1
    label — the existing binary behaviour is unchanged."""
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "binary_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-bin", path, role="sample")

    proposal = build_modeling_proposal(registry, backend, "task-bin", tmp_path, recipes=["lgb"])

    assert proposal.target_type == "binary"
    assert proposal.target_col == "y"
    assert proposal.bad_rate is not None
    slots = proposal.template_slots()
    assert slots["target_type"] == "binary"
    assert slots["selection_policy"] == {"require_pmml": True, "require_handoff": True}


def test_build_modeling_proposal_surfaces_excluded_categorical_notice(tmp_path):
    """PREP-3/FS-3: candidate inference still excludes string columns from feature_cols
    (default behaviour unchanged), but the setup notes must now say so explicitly instead
    of silently dropping them."""
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "chan": (["online", "offline", "partner"] * (rows // 3 + 1))[:rows],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "categorical_binary_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-cat", path, role="sample")

    proposal = build_modeling_proposal(registry, backend, "task-cat", tmp_path, recipes=["lgb"])

    # Default behaviour is unchanged: the categorical column never enters feature_cols.
    assert "chan" not in proposal.feature_cols
    assert "x1" in proposal.feature_cols
    notice = next((note for note in proposal.notes if "类别列未入模" in note), None)
    assert notice is not None
    assert "chan" in notice
    assert "woe_encode_categorical" in notice
    assert "catboost" in notice


def test_build_modeling_proposal_omits_categorical_notice_when_no_categorical_columns(tmp_path):
    """No string columns in the sample -> no excluded-categorical notice (current
    behaviour for an all-numeric sample stays exactly as before)."""
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "all_numeric_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-nocat", path, role="sample")

    proposal = build_modeling_proposal(registry, backend, "task-nocat", tmp_path, recipes=["lgb"])

    assert not any("类别列未入模" in note for note in proposal.notes)


def test_build_modeling_proposal_detects_sample_weight_candidate_without_feature_leakage(tmp_path):
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "sample_weight": [2.0 if i % 4 == 0 else 1.0 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "weighted_binary_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-weight", path, role="sample")

    proposal = build_modeling_proposal(registry, backend, "task-weight", tmp_path, recipes=["lgb"])

    assert proposal.sample_weight_col == ""
    assert proposal.sample_weight_candidates == ["sample_weight"]
    assert proposal.sample_weight_diagnostics[0]["column"] == "sample_weight"
    assert proposal.sample_weight_diagnostics[0]["valid"] is True
    assert proposal.sample_weight_diagnostics[0]["missing_rate"] == 0.0
    assert proposal.sample_weight_diagnostics[0]["excluded_from_features"] is True
    assert "sample_weight" not in proposal.feature_cols
    slots = proposal.template_slots()
    assert slots["sample_weight_col"] == ""
    assert slots["sample_weight_diagnostics"][0]["column"] == "sample_weight"
    assert slots["passthrough_cols"] == ["sample_weight"]
    assert any("检测到样本权重候选列" in note for note in proposal.notes)


def test_build_modeling_proposal_rejects_zero_weight_candidate(tmp_path):
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "sample_weight": [0.0 if i == 0 else 1.0 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "zero_weight_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-zero-weight", path, role="sample")

    proposal = build_modeling_proposal(registry, backend, "task-zero-weight", tmp_path, recipes=["lgb"])

    assert proposal.sample_weight_candidates == []
    assert proposal.sample_weight_diagnostics[0]["column"] == "sample_weight"
    assert proposal.sample_weight_diagnostics[0]["valid"] is False
    assert proposal.sample_weight_diagnostics[0]["reason"] == "存在非正权重"


def test_build_modeling_proposal_continuous_target_skips_meta_columns(tmp_path):
    backend, registry = _proposal_runtime(tmp_path)
    rows = 120
    frame = pd.DataFrame({
        "limit_flag": [i % 2 for i in range(rows)],
        "monthly_income": [1000 + i * 10 for i in range(rows)],
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "split": ["train"] * 70 + ["test"] * 30 + ["oot"] * 20,
    })
    path = tmp_path / "regression_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-reg", path, role="sample")

    proposal = build_modeling_proposal(
        registry,
        backend,
        "task-reg",
        tmp_path,
        recipes=["lgb_regressor"],
    )

    assert proposal.target_type == "continuous"
    assert proposal.target_col == "monthly_income"


def test_build_modeling_proposal_derives_multiclass_target_type_from_recipe(tmp_path):
    """A lgb_multiclass recipe ⇒ target_type 'multiclass'; the 3-class `grade` column is
    picked as the target, bad_rate is None, and the primary recipe is lgb_multiclass."""
    backend, registry = _proposal_runtime(tmp_path)
    rows = 150
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "x2": [((i * 17) % 89) / 100 for i in range(rows)],
        "risk_grade": ["A" if i % 3 == 0 else ("B" if i % 3 == 1 else "C") for i in range(rows)],
        "split": ["train"] * 90 + ["test"] * 40 + ["oot"] * 20,
    })
    path = tmp_path / "grade_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-mc", path, role="sample")

    proposal = build_modeling_proposal(
        registry, backend, "task-mc", tmp_path, recipes=["lgb_multiclass"]
    )

    assert proposal.target_type == "multiclass"
    assert proposal.recipe == "lgb_multiclass"
    assert proposal.recipes == ["lgb_multiclass"]
    assert proposal.target_col == "risk_grade"
    assert proposal.bad_rate is None
    assert "risk_grade" not in proposal.feature_cols
    slots = proposal.template_slots()
    assert slots["target_type"] == "multiclass"
    assert slots["selection_policy"] == {"require_pmml": False, "require_handoff": False}


def test_build_modeling_proposal_rejects_mixed_recipe_families(tmp_path):
    backend, registry = _proposal_runtime(tmp_path)
    rows = 90
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "y": [1 if i % 5 in {0, 1} else 0 for i in range(rows)],
        "income": [3500 + i * 10 for i in range(rows)],
        "split": ["train"] * 50 + ["test"] * 25 + ["oot"] * 15,
    })
    path = tmp_path / "mixed_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-mixed", path, role="sample")

    with pytest.raises(ValueError, match="不能在同一次训练混用"):
        build_modeling_proposal(
            registry, backend, "task-mixed", tmp_path, recipes=["lgb", "lgb_regressor"]
        )


def test_build_modeling_proposal_rejects_target_type_recipe_mismatch(tmp_path):
    backend, registry = _proposal_runtime(tmp_path)
    rows = 90
    frame = pd.DataFrame({
        "x1": [((i * 37) % 101) / 100 for i in range(rows)],
        "income": [3500 + i * 10 for i in range(rows)],
        "split": ["train"] * 50 + ["test"] * 25 + ["oot"] * 15,
    })
    path = tmp_path / "mismatch_sample.csv"
    frame.to_csv(path, index=False)
    registry.register_from_upload("task-mismatch", path, role="sample")

    with pytest.raises(ValueError, match="目标类型 `binary` 与算法 `lgb_regressor` 不匹配"):
        build_modeling_proposal(
            registry,
            backend,
            "task-mismatch",
            tmp_path,
            target_type="binary",
            recipes=["lgb_regressor"],
        )

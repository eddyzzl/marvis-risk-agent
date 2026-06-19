import pandas as pd
import pytest

from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling import ModelingError
from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.recipes import get_recipe, list_recipes
from marvis.packs.modeling.recipes.common import compute_model_metrics, split_modeling_frame


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


def test_recipe_registry_exposes_four_builtin_recipes():
    recipes = list_recipes()

    assert [recipe.id for recipe in recipes] == ["lgb", "xgb", "lr", "scorecard"]
    assert get_recipe("scorecard").requires_woe is True
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

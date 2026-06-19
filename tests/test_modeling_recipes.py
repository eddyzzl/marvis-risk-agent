import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling import ModelingError
from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.recipes import get_recipe, list_recipes
from marvis.packs.modeling.recipes.common import compute_model_metrics, split_modeling_frame
from marvis.packs.modeling.recipes.lr import train_lr


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
    assert first.artifact.algorithm == "lr"
    assert first.artifact.params["C"] == 0.7
    assert first.artifact.feature_list == ("x1", "x2")
    assert first.metrics.test_auc == pytest.approx(second.metrics.test_auc)
    assert [item[0] for item in first.feature_importance] == [item[0] for item in second.feature_importance]
    assert [item[1] for item in first.feature_importance] == pytest.approx(
        [item[1] for item in second.feature_importance],
    )

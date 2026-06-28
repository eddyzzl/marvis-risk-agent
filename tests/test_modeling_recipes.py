import json

import numpy as np
import pandas as pd
import pytest

from marvis.agent.modeling_setup import build_modeling_proposal
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, init_db
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling import ModelingError
from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.recipes import get_recipe, list_recipes
from marvis.packs.modeling.recipes.common import (
    compute_model_metrics,
    compute_multiclass_metrics,
    split_modeling_frame,
)
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lgb_multiclass import train_lgb_multiclass
from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.mlp import train_mlp
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
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
    assert first_lgb.artifact.algorithm == "lgb"
    assert first_xgb.artifact.algorithm == "xgb"
    assert first_lgb.metrics.test_auc == pytest.approx(second_lgb.metrics.test_auc)
    assert first_xgb.metrics.test_auc == pytest.approx(second_xgb.metrics.test_auc)
    assert [item[0] for item in first_lgb.feature_importance] == ["x1", "x2"]
    assert [item[0] for item in first_xgb.feature_importance] == ["x1", "x2"]


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
    assert first.artifact.algorithm == "scorecard"
    assert set(first.artifact.woe_maps or {}) == {"x1", "x2"}
    assert first.artifact.params["base_score"] == 600
    assert "max_depth" not in first.artifact.params
    assert first.metrics.test_auc == pytest.approx(second.metrics.test_auc)
    assert [item[0] for item in first.feature_importance] == [item[0] for item in second.feature_importance]
    assert [item[1] for item in first.feature_importance] == pytest.approx(
        [item[1] for item in second.feature_importance],
    )


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
    assert proposal.template_slots()["target_type"] == "continuous"


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
    assert proposal.template_slots()["target_type"] == "binary"


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
    assert "sample_weight" not in proposal.feature_cols
    slots = proposal.template_slots()
    assert slots["sample_weight_col"] == ""
    assert slots["passthrough_cols"] == ["sample_weight"]
    assert any("检测到样本权重候选列" in note for note in proposal.notes)


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
    assert proposal.template_slots()["target_type"] == "multiclass"


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

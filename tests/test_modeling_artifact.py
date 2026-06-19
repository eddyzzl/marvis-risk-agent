import lightgbm as lgb
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from marvis.packs.modeling.artifact import export_pmml, load_model, save_model
from marvis.packs.modeling.errors import ModelingError


def test_save_and_load_lr_model_round_trips_predictions(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    model = LogisticRegression().fit(frame[["x1"]], frame["y"])

    artifact = save_model(
        model,
        "lr",
        tmp_path,
        feature_list=("x1",),
        params={"C": 1.0},
    )
    loaded = load_model(artifact, base_dir=tmp_path)

    assert (tmp_path / artifact.model_path).exists()
    assert loaded.predict_proba(frame[["x1"]])[:, 1].tolist() == pytest.approx(
        model.predict_proba(frame[["x1"]])[:, 1].tolist()
    )


def test_save_and_load_lgb_and_xgb_models(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    lgb_model = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 1},
        lgb.Dataset(frame[["x1"]], label=frame["y"]),
        num_boost_round=2,
    )
    xgb_model = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "nthread": 1},
        xgb.DMatrix(frame[["x1"]], label=frame["y"], feature_names=["x1"]),
        num_boost_round=2,
    )

    lgb_artifact = save_model(lgb_model, "lgb", tmp_path, feature_list=("x1",), params={})
    xgb_artifact = save_model(xgb_model, "xgb", tmp_path, feature_list=("x1",), params={})

    assert isinstance(load_model(lgb_artifact, base_dir=tmp_path), lgb.Booster)
    assert isinstance(load_model(xgb_artifact, base_dir=tmp_path), xgb.Booster)


def test_save_scorecard_model_preserves_woe_maps_and_pmml_error_is_clear(tmp_path):
    model = LogisticRegression().fit([[0.1], [0.2], [0.8], [0.9]], [0, 0, 1, 1])
    artifact = save_model(
        model,
        "scorecard",
        tmp_path,
        feature_list=("x1",),
        params={"base_score": 600},
        woe_maps={"x1": {"edges": [-float("inf"), 0.5, float("inf")]}},
    )

    assert artifact.woe_maps == {"x1": {"edges": [-float("inf"), 0.5, float("inf")]}}
    assert isinstance(load_model(artifact, base_dir=tmp_path), LogisticRegression)
    with pytest.raises(ModelingError, match="PMML export is not available"):
        export_pmml(artifact, tmp_path / "sample.parquet", tmp_path / "model.pmml", base_dir=tmp_path)

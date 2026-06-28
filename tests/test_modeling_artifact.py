import lightgbm as lgb
import pandas as pd
import pytest
import xgboost as xgb
from pypmml import Model
from sklearn.linear_model import LogisticRegression

from marvis.feature.contracts import WOEResult
from marvis.packs.modeling.artifact import export_pmml, load_model, save_model


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


def test_export_lr_pmml_can_be_loaded_by_pypmml(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    sample_path = tmp_path / "sample.parquet"
    frame.to_parquet(sample_path, index=False)
    model = LogisticRegression().fit(frame[["x1"]], frame["y"])
    artifact = save_model(
        model,
        "lr",
        tmp_path,
        feature_list=("x1",),
        params={"C": 1.0},
    )

    pmml_path = export_pmml(artifact, sample_path, tmp_path / "model.pmml", base_dir=tmp_path)

    assert pmml_path.exists()
    assert Model.load(pmml_path.as_posix()) is not None


def test_save_scorecard_model_preserves_woe_maps_and_exports_pmml(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    sample_path = tmp_path / "sample.parquet"
    frame.to_parquet(sample_path, index=False)
    model = LogisticRegression().fit([[0.1], [0.2], [0.8], [0.9]], [0, 0, 1, 1])
    woe = WOEResult(
        feature="x1",
        edges=(-float("inf"), 0.5, float("inf")),
        woe_by_bin=(-1.0, 1.0),
        na_woe=0.0,
    )
    artifact = save_model(
        model,
        "scorecard",
        tmp_path,
        feature_list=("x1",),
        params={"base_score": 600},
        woe_maps={"x1": woe},
    )

    assert artifact.woe_maps == {"x1": woe}
    loaded = load_model(artifact, base_dir=tmp_path)
    assert isinstance(loaded["model"], LogisticRegression)
    pmml_path = export_pmml(artifact, sample_path, tmp_path / "model.pmml", base_dir=tmp_path)
    assert pmml_path.exists()
    assert Model.load(pmml_path.as_posix()) is not None

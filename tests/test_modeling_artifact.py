import lightgbm as lgb
import json
import pandas as pd
import pytest
import xgboost as xgb
from pypmml import Model
from sklearn.linear_model import LogisticRegression

from marvis.data.backend import DataBackend
from marvis.feature.contracts import WOEResult
from marvis.packs.modeling import artifact as artifact_module
from marvis.packs.modeling.artifact import export_pmml, load_model, save_model, write_artifact_file
from marvis.packs.modeling.errors import ModelingError, PmmlConverterError


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
    assert not (tmp_path / ".staging").exists()
    meta_path = tmp_path / f"{artifact.id}.model_meta.json"
    latest_meta_path = tmp_path / "model_meta.json"
    assert meta_path.exists()
    assert latest_meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["artifact_id"] == artifact.id
    assert meta["algorithm"] == "lr"
    assert meta["model_path"] == artifact.model_path
    assert meta["feature_list"] == ["x1"]
    assert meta["params"]["C"] == 1.0
    assert meta["seed"] is None
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


def test_pmml_parquet_schema_sample_is_bounded(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "wide_sample.parquet"
    pd.DataFrame({
        "x1": range(250),
        "x2": range(250, 500),
        "unused": ["payload"] * 250,
    }).to_parquet(path, index=False)
    original_read_frame = DataBackend.read_frame
    calls = []

    def tracking_read_frame(self, selected_path, *, columns=None, nrows=None):
        calls.append((tuple(columns or ()), nrows))
        return original_read_frame(self, selected_path, columns=columns, nrows=nrows)

    monkeypatch.setattr(DataBackend, "read_frame", tracking_read_frame)

    sample = artifact_module._read_schema_sample(path, ["x1", "x2"])

    assert calls == [(("x1", "x2"), 100)]
    assert list(sample.columns) == ["x1", "x2"]
    assert len(sample) == 100


@pytest.mark.pmml_runtime
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
    assert not (tmp_path / ".staging").exists()
    pmml_model = Model.load(pmml_path.as_posix())
    assert pmml_model is not None
    pmml_predictions = pmml_model.predict(frame[["x1"]])
    assert pmml_predictions["probability(1)"].tolist() == pytest.approx(
        model.predict_proba(frame[["x1"]])[:, 1].tolist(),
        abs=1e-6,
    )


@pytest.mark.pmml_runtime
@pytest.mark.parametrize(
    ("algorithm", "model"),
    [
        (
            "lgb",
            lgb.LGBMClassifier(
                n_estimators=3,
                min_child_samples=1,
                random_state=7,
                verbosity=-1,
                n_jobs=1,
            ),
        ),
        (
            "xgb",
            xgb.XGBClassifier(
                n_estimators=3,
                max_depth=2,
                learning_rate=0.5,
                eval_metric="logloss",
                random_state=7,
                n_jobs=1,
            ),
        ),
    ],
)
def test_export_tree_sklearn_wrapper_pmml_can_be_loaded_by_pypmml(
    tmp_path,
    algorithm,
    model,
):
    frame = pd.DataFrame({
        "x1": [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95],
        "x2": [0.20, 0.10, 0.25, 0.30, 0.70, 0.80, 0.75, 0.90],
        "y": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    sample_path = tmp_path / "sample.parquet"
    frame.to_parquet(sample_path, index=False)
    model.fit(frame[["x1", "x2"]], frame["y"])
    artifact = save_model(
        model,
        algorithm,
        tmp_path / algorithm,
        feature_list=("x1", "x2"),
        params={},
    )

    pmml_path = export_pmml(
        artifact,
        sample_path,
        tmp_path / algorithm / f"{algorithm}.pmml",
        base_dir=tmp_path / algorithm,
        target_col="y",
    )

    assert pmml_path.exists()
    assert not (tmp_path / algorithm / ".staging").exists()
    pmml_predictions = Model.load(pmml_path.as_posix()).predict(frame[["x1", "x2"]])
    assert pmml_predictions["probability(1)"].tolist() == pytest.approx(
        model.predict_proba(frame[["x1", "x2"]])[:, 1].tolist(),
        abs=1e-6,
    )


@pytest.mark.pmml_runtime
def test_export_pmml_uses_explicit_target_when_sample_weight_precedes_label(tmp_path):
    frame = pd.DataFrame({
        "x1": [0.1, 0.2, 0.8, 0.9],
        "sample_weight": [2.0, 1.0, 1.0, 2.0],
        "y": [0, 0, 1, 1],
        "model_flag": ["train", "train", "test", "test"],
    })
    sample_path = tmp_path / "sample.parquet"
    frame.to_parquet(sample_path, index=False)
    model = LogisticRegression().fit(frame[["x1"]], frame["y"], sample_weight=frame["sample_weight"])
    artifact = save_model(
        model,
        "lr",
        tmp_path,
        feature_list=("x1",),
        params={"sample_weight_col": "sample_weight", "split_col": "model_flag"},
    )

    pmml_path = export_pmml(
        artifact,
        sample_path,
        tmp_path / "weighted.pmml",
        base_dir=tmp_path,
        target_col="y",
    )

    text = pmml_path.read_text(encoding="utf-8")
    assert 'name="y"' in text
    assert 'targetFieldName="sample_weight"' not in text
    assert 'targetFields="sample_weight"' not in text


def test_write_artifact_file_rolls_back_partial_writer_failure(tmp_path):
    def bad_writer(path):
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write_artifact_file(tmp_path, "partial.joblib", bad_writer)

    assert not (tmp_path / "partial.joblib").exists()
    assert not (tmp_path / ".staging").exists()


_JPMML_UNSUPPORTED_SKLEARN_OUTPUT = """Standard output is empty
Standard error:
10月 15, 2026 6:50:03 上午 [com.sklearn2pmml.Main]  main
严重: Failed to convert PKL to PMML

===============================================================================
                                   EXCEPTION
===============================================================================
org.jpmml.sklearn.SkLearnException: Scikit-Learn version '1.9.1' is not supported
Solution:
\tUpgrade the converter to the latest version, or downgrade Scikit-Learn to version '1.9.0'
Context:
\tsklearn.linear_model._logistic.LogisticRegression
Stack trace:
\tat sklearn.Step.checkSkLearnVersion(Step.java:62)
\tat com.sklearn2pmml.Main.main(Main.java:103)
"""

_SKLEARN2PMML_FAILURE = (
    "The SkLearn2PMML application has failed. The Java executable should have printed "
    "more information about the failure into its standard output and/or standard error streams"
)


def _lr_artifact(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    sample_path = tmp_path / "sample.parquet"
    frame.to_parquet(sample_path, index=False)
    model = LogisticRegression().fit(frame[["x1"]], frame["y"])
    return save_model(model, "lr", tmp_path, feature_list=("x1",), params={}), sample_path


def _fake_sklearn2pmml(monkeypatch, *, output: str, error: str):
    import sklearn2pmml

    def fake(estimator, pmml_path, **kwargs):
        # Mirror sklearn2pmml: Java diagnostics go to Python stdout, then a generic error.
        print(output)
        raise RuntimeError(error)

    monkeypatch.setattr(sklearn2pmml, "sklearn2pmml", fake)


def test_export_pmml_reports_converter_rejecting_installed_library_as_environment_error(
    tmp_path, monkeypatch, capsys
):
    artifact, sample_path = _lr_artifact(tmp_path)
    _fake_sklearn2pmml(monkeypatch, output=_JPMML_UNSUPPORTED_SKLEARN_OUTPUT, error=_SKLEARN2PMML_FAILURE)

    with pytest.raises(PmmlConverterError, match="PMML 导出环境不兼容") as raised:
        export_pmml(artifact, sample_path, tmp_path / "model.pmml", base_dir=tmp_path)

    error = raised.value
    assert "Scikit-Learn 1.9.1" in str(error)
    assert "downgrade Scikit-Learn to version '1.9.0'" in str(error)
    assert "uv.lock" in str(error)
    detail = error.to_detail()
    assert detail["kind"] == "pmml_export_environment"
    assert detail["cause"] == "unsupported_library_version"
    assert detail["environment_issue"] is True
    assert detail["library"] == "Scikit-Learn"
    assert detail["installed_version"] == "1.9.1"
    assert detail["java_exception"] == "org.jpmml.sklearn.SkLearnException"
    assert "Step.java:62" in detail["output_tail"]
    json.dumps(detail)
    # The converter output is captured into the error, not leaked to stdout
    # (the tool worker's stdout also carries its result protocol).
    assert "SkLearnException" not in capsys.readouterr().out
    assert not (tmp_path / "model.pmml").exists()
    assert not (tmp_path / ".staging").exists()


@pytest.mark.parametrize(
    ("output", "error", "cause", "kind", "message"),
    [
        (
            "",
            "Java is not installed, or the Java executable is not on system path",
            "java_unavailable",
            "pmml_export_environment",
            "找不到 java",
        ),
        (
            "Standard error:\nError: LinkageError occurred while loading main class com.sklearn2pmml.Main\n"
            "\tjava.lang.UnsupportedClassVersionError: com/sklearn2pmml/Main has been compiled by a "
            "more recent version of the Java Runtime (class file version 55.0)\n",
            _SKLEARN2PMML_FAILURE,
            "java_unsupported",
            "pmml_export_environment",
            "Java 版本过旧",
        ),
        (
            "Standard error:\njava.lang.IllegalArgumentException: Attribute 'repr_' has an unsupported value\n",
            _SKLEARN2PMML_FAILURE,
            "conversion_failed",
            "pmml_conversion_failed",
            "IllegalArgumentException: Attribute 'repr_' has an unsupported value",
        ),
        (
            "Standard output is empty\n",
            _SKLEARN2PMML_FAILURE,
            "conversion_failed",
            "pmml_conversion_failed",
            "SkLearn2PMML application has failed",
        ),
    ],
)
def test_export_pmml_classifies_converter_failures(tmp_path, monkeypatch, output, error, cause, kind, message):
    artifact, sample_path = _lr_artifact(tmp_path)
    _fake_sklearn2pmml(monkeypatch, output=output, error=error)

    with pytest.raises(PmmlConverterError, match=message) as raised:
        export_pmml(artifact, sample_path, tmp_path / "model.pmml", base_dir=tmp_path)

    detail = raised.value.to_detail()
    assert detail["cause"] == cause
    assert detail["kind"] == kind
    assert detail["environment_issue"] is (kind == "pmml_export_environment")


def test_export_pmml_rejects_native_lgb_booster_payload(tmp_path):
    frame = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    sample_path = tmp_path / "sample.parquet"
    frame.to_parquet(sample_path, index=False)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 1},
        lgb.Dataset(frame[["x1"]], label=frame["y"]),
        num_boost_round=2,
    )
    artifact = save_model(booster, "lgb", tmp_path, feature_list=("x1",), params={})

    with pytest.raises(ModelingError, match="sklearn-compatible"):
        export_pmml(artifact, sample_path, tmp_path / "model.pmml", base_dir=tmp_path)


@pytest.mark.pmml_runtime
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
        scorecard_table=[
            {
                "feature": "x1",
                "bin_index": 1,
                "bin_label": "[0, 0.5)",
                "points": 18.0,
            }
        ],
    )

    assert artifact.woe_maps == {"x1": woe}
    assert artifact.scorecard_table[0]["points"] == 18.0
    loaded = load_model(artifact, base_dir=tmp_path)
    assert isinstance(loaded["model"], LogisticRegression)
    assert loaded["scorecard_table"] == list(artifact.scorecard_table)
    meta = json.loads((tmp_path / f"{artifact.id}.model_meta.json").read_text(encoding="utf-8"))
    assert meta["scorecard_table"][0]["bin_label"] == "[0, 0.5)"
    pmml_path = export_pmml(artifact, sample_path, tmp_path / "model.pmml", base_dir=tmp_path)
    assert pmml_path.exists()
    pmml_model = Model.load(pmml_path.as_posix())
    pmml_predictions = pmml_model.predict(frame[["x1"]])
    assert pmml_predictions["probability(1)"].tolist() == pytest.approx(
        loaded["model"].predict_proba([[woe.woe_by_bin[0]], [woe.woe_by_bin[0]], [woe.woe_by_bin[1]], [woe.woe_by_bin[1]]])[:, 1].tolist(),
        abs=1e-6,
    )

"""Native numerical verification never executes archive-directed I/O or callbacks."""
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marvis.orchestrator.eval.runtime_archive_reader import _Unsupported
from marvis.orchestrator.eval.runtime_modeling_calibration_reference import revalidate_calibration
from marvis.orchestrator.eval.runtime_modeling_native_reference import independent_linear_fit
from marvis.orchestrator.eval.runtime_modeling_profile_reference import (
    independent_profile_fit,
    read_numerical_payload,
    read_profile,
)
from marvis.orchestrator.eval.runtime_modeling_tree_reference import independent_tree_fit


@pytest.mark.parametrize("algorithm", ["lr", "mlp"])
def test_pipeline_cache_is_refused_before_refit_or_external_write(tmp_path, monkeypatch, algorithm):
    frame = pd.DataFrame({"x": np.arange(8, dtype=float), "y": [0, 1] * 4})
    learner = LogisticRegression() if algorithm == "lr" else MLPClassifier(hidden_layer_sizes=(2,), max_iter=2, random_state=1)
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), (algorithm, learner)])
    model.fit(frame[["x"]], frame.y)
    external = tmp_path / "outside-archive-cache"
    model.memory = str(external)
    path = tmp_path / "native.joblib"
    joblib.dump(model, path)
    with pytest.raises(_Unsupported, match="cache/metadata"):
        read_profile(path, algorithm, ["x"])
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe model reached clone/refit")
    monkeypatch.setattr("sklearn.base.clone", forbidden)
    with pytest.raises(_Unsupported, match="cache/metadata"):
        if algorithm == "lr":
            independent_linear_fit(model, frame, ["x"], "y")
        else:
            independent_profile_fit(model, frame, {"features": ["x"], "target_col": "y", "params": {}}, metadata={"algorithm": algorithm})
    assert not external.exists()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("library", ["lgb", "xgb", "catboost"])
def test_allowed_estimator_classes_cannot_be_called_as_pickle_reducers(tmp_path, monkeypatch, nested, library):
    from catboost import CatBoostClassifier
    from lightgbm import Booster
    from xgboost import Booster as XBooster
    klass = {"lgb": Booster, "xgb": XBooster, "catboost": CatBoostClassifier}[library]
    class Constructor:
        def __reduce__(self):
            return klass, (None, None, str(tmp_path / "external-model"))
    path = tmp_path / "native.joblib"
    if nested:
        joblib.dump(np.array([Constructor()], dtype=object), path)
    else:
        path.write_bytes(pickle.dumps(Constructor()))
    def forbidden(*args, **kwargs):
        pytest.fail("archive invoked an estimator constructor")
    monkeypatch.setattr(klass, "__init__", forbidden)
    with pytest.raises(_Unsupported, match="data reducer"):
        read_numerical_payload(path)


def test_random_state_helper_cannot_launder_estimator_callback(tmp_path, monkeypatch):
    from lightgbm import Booster
    from numpy.random._pickle import __randomstate_ctor as randomstate_constructor
    class Constructor:
        def __reduce__(self):
            return randomstate_constructor, ("external-model", Booster)
    path = tmp_path / "native.pkl"
    path.write_bytes(pickle.dumps(Constructor()))
    def forbidden(*args, **kwargs):
        pytest.fail("random-state reducer invoked archive callback")
    monkeypatch.setattr(Booster, "__init__", forbidden)
    with pytest.raises(_Unsupported, match="random-state"):
        read_numerical_payload(path)


def test_pickle_extension_cache_cannot_bypass_global_restrictions(tmp_path):
    import copyreg
    code = 245
    assert code not in copyreg._extension_cache
    # An already populated interpreter cache skips find_class in Python's
    # default unpickler. No function is invoked by this benign probe.
    copyreg._extension_cache[code] = PathProbe = type("PathProbe", (), {})
    path = tmp_path / "extension.pkl"
    path.write_bytes(b"\x80\x02\x82" + bytes([code]) + b".")
    try:
        assert pickle.loads(path.read_bytes()) is PathProbe
        with pytest.raises(_Unsupported, match="extension"):
            read_numerical_payload(path)
    finally:
        del copyreg._extension_cache[code]


@pytest.mark.parametrize("payload", [b"(ilightgbm.basic\nBooster\n.", b"(clightgbm.basic\nBooster\no."])
def test_legacy_pickle_constructor_opcodes_are_also_restricted(tmp_path, monkeypatch, payload):
    from lightgbm import Booster
    path = tmp_path / "legacy.pkl"
    path.write_bytes(payload)
    def forbidden(*args, **kwargs):
        pytest.fail("legacy pickle opcode invoked archive constructor")
    monkeypatch.setattr(Booster, "__init__", forbidden)
    with pytest.raises(_Unsupported, match="data reducer"):
        read_numerical_payload(path)


@pytest.mark.parametrize("deferred_state", [False, True])
def test_numpy_reducer_allocation_cannot_bypass_joblib_shape_bound(tmp_path, deferred_state):
    from numpy._core.multiarray import _reconstruct
    class Allocation:
        def __reduce__(self):
            if deferred_state:
                return _reconstruct, (np.ndarray, (0,), "b"), (1, (1_000_000_000,), np.dtype("float64"), False, b"")
            return _reconstruct, (np.ndarray, (1_000_000_000,), "d")
    path = tmp_path / "array.pkl"
    path.write_bytes(pickle.dumps(Allocation()))
    with pytest.raises(_Unsupported, match="allocation"):
        read_numerical_payload(path)


def test_pickle_byte_buffer_cannot_allocate_archive_declared_large_integer(tmp_path):
    class Allocation:
        def __reduce__(self):
            return bytearray, (1_000_000_000,)
    path = tmp_path / "buffer.pkl"
    path.write_bytes(pickle.dumps(Allocation()))
    with pytest.raises(_Unsupported, match="buffer allocation"):
        read_numerical_payload(path)


@pytest.mark.parametrize("algorithm", ["lr", "mlp"])
@pytest.mark.parametrize("step,parameter,value", [
    (0, "missing_values", -99.), (0, "keep_empty_features", True),
    (1, "with_mean", False),
])
def test_undeclared_preprocessing_is_rejected_even_when_fitted_rows_hide_it(tmp_path, algorithm, step, parameter, value):
    frame = pd.DataFrame({"x": np.arange(8, dtype=float), "y": [0, 1] * 4})
    learner = LogisticRegression() if algorithm == "lr" else MLPClassifier(hidden_layer_sizes=(2,), max_iter=2, random_state=1)
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), (algorithm, learner)])
    model.fit(frame[["x"]], frame.y)
    setattr(model.steps[step][1], parameter, value)
    path = tmp_path / "native.joblib"
    joblib.dump(model, path)
    with pytest.raises(_Unsupported, match="preprocessing configuration"):
        read_profile(path, algorithm, ["x"])


@pytest.mark.parametrize("algorithm,key,value", [
    ("xgb", "callbacks", []), ("xgb", "model_file", "/tmp/external-model"),
    ("xgb", "objective", LogisticRegression),
    ("lgb", "callbacks", []), ("lgb", "forcedbins_filename", "/tmp/bins"),
    ("lgb", "input_model", "/tmp/external-model"), ("lgb", "output_model", "/tmp/output"),
    ("lgb", "machines", "127.0.0.1:1"), ("lgb", "num_machines", 2),
    ("catboost", "allow_writing_files", True), ("catboost", "train_dir", "/tmp/train"),
    ("catboost", "save_snapshot", True), ("catboost", "snapshot_file", "/tmp/snapshot"),
    ("catboost", "input_borders", "/tmp/borders"),
])
def test_unsafe_native_params_are_refused_before_any_refit(monkeypatch, algorithm, key, value):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    klass = {"xgb": XGBClassifier, "lgb": LGBMClassifier, "catboost": CatBoostClassifier}[algorithm]
    params = {"allow_writing_files": False} if algorithm == "catboost" else {}
    model = klass(**{**params, key: value})
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe parameters reached clone/refit")
    monkeypatch.setattr("sklearn.base.clone", forbidden)
    frame = pd.DataFrame({"x": [0., 1.], "y": [0, 1]})
    config = {"features": ["x"], "target_col": "y", "params": {}, "recipe_id": algorithm, "early_stopping_rounds": None}
    with pytest.raises(_Unsupported):
        if algorithm == "catboost":
            independent_profile_fit(model, frame, config, metadata={"algorithm": algorithm}, test_frame=frame)
        else:
            independent_tree_fit(model, frame, config)


def test_text_booster_frozen_params_cannot_supply_external_input(monkeypatch):
    import lightgbm
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe frozen parameters reached native library")
    monkeypatch.setattr(lightgbm, "Dataset", forbidden)
    frame = pd.DataFrame({"x": [0., 1.], "y": [0., 1.]})
    config = {"features": ["x"], "target_col": "y", "params": {}, "recipe_id": "lgb_regressor", "early_stopping_rounds": None}
    with pytest.raises(_Unsupported):
        independent_tree_fit(None, frame, config, frozen_params={"num_boost_round": 1, "input_model": "/tmp/external-model"})


@pytest.mark.parametrize("method,attribute,value", [
    ("isotonic", "out_of_bounds", "nan"), ("isotonic", "out_of_bounds", "raise"),
    ("isotonic", "y_min", None), ("isotonic", "increasing", "auto"),
    ("isotonic", "increasing_", False),
    ("sigmoid", "classes_", np.array([1, 0])),
    ("sigmoid", "coef_", np.array([[1., 2.]])), ("sigmoid", "n_features_in_", 2),
    ("sigmoid", "C", 2.),
])
def test_calibrator_configuration_and_shape_tampering_fails_closed(tmp_path, method, attribute, value):
    from sklearn.isotonic import IsotonicRegression
    scores, labels = np.array([.1, .2, .8, .9]), np.array([0, 1, 0, 1])
    model = (IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.).fit(scores, labels)
             if method == "isotonic" else LogisticRegression(solver="lbfgs").fit(scores.reshape(-1, 1), labels))
    setattr(model, attribute, value)
    path = tmp_path / "calibrator.joblib"
    joblib.dump({"method": method, "calibrator": model}, path)
    config = {"dataset_id": "ds", "target_col": "y", "split_col": "split"}
    calibration = {**config, "method": method}
    with pytest.raises((_Unsupported, ValueError), match="calibrator"):
        revalidate_calibration(path, {"params": {"calibration": calibration}}, config, None, None)


@pytest.mark.parametrize("attribute,value", [("X_min_", .5), ("X_max_", .5)])
def test_isotonic_extrapolation_bounds_must_match_independent_fit(tmp_path, attribute, value):
    from sklearn.isotonic import IsotonicRegression
    frame = pd.DataFrame({"score": [.1, .2, .8, .9], "y": [0, 1, 0, 1], "split": ["test"] * 4})
    model = IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.).fit(frame.score, frame.y)
    setattr(model, attribute, value)
    path = tmp_path / "calibrator.joblib"
    joblib.dump({"method": "isotonic", "calibrator": model}, path)
    config = {"dataset_id": "ds", "target_col": "y", "split_col": "split"}
    calibration = {**config, "method": "isotonic", "fit_split": "test", "split": "test", "split_value": "test",
                   "evaluated_on": "fit_sample(in-sample)", "eval_split": "test", "fit_sample_count": 4,
                   "positive_count": 2, "negative_count": 2}
    with pytest.raises(ValueError, match="extrapolation bounds"):
        revalidate_calibration(path, {"params": {"calibration": calibration}}, config, frame, lambda f: f.score.to_numpy())

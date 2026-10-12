"""Restricted native linear-model reader and independent inference/re-fit.

Only installed numerical/sklearn data classes are allowed by the joblib reader.
Archive callbacks, imports and executable globals are refused before invocation.
"""
import importlib
import math

import numpy as np
from joblib.numpy_pickle import NumpyArrayWrapper, NumpyUnpickler
from joblib.numpy_pickle_utils import Unpickler, _ensure_native_byte_order

from .runtime_archive_reader import _Unsupported

_GLOBALS = {
    ("sklearn.pipeline", "Pipeline"), ("sklearn.impute._base", "SimpleImputer"),
    ("sklearn.preprocessing._data", "StandardScaler"),
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("sklearn.linear_model._base", "LinearRegression"),
    ("sklearn.linear_model._ridge", "Ridge"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("builtins", "set"), ("builtins", "slice"), ("builtins", "bytearray"),
    ("builtins", "dict"), ("collections", "defaultdict"), ("collections", "OrderedDict"),
    ("xgboost.sklearn", "XGBClassifier"), ("xgboost.sklearn", "XGBRegressor"),
    ("xgboost.core", "Booster"),
    ("lightgbm.sklearn", "LGBMClassifier"), ("lightgbm.basic", "Booster"),
    ("sklearn.preprocessing._label", "LabelEncoder"),
}


class _BoundedArrayWrapper(NumpyArrayWrapper):
    def read_array(self, unpickler, ensure_native_byte_order):
        if not self.dtype.hasobject:
            return super().read_array(unpickler, ensure_native_byte_order)
        # Older supported joblib versions use plain pickle.load for object
        # arrays. Keep the outer allowlist on the nested stream as well.
        inner = Unpickler(unpickler.file_handle)
        inner.find_class = unpickler.find_class
        array = inner.load()
        return _ensure_native_byte_order(array) if ensure_native_byte_order else array

    def read(self, unpickler, ensure_native_byte_order):
        if (not isinstance(self.shape, tuple) or len(self.shape) > 8
                or any(type(v) is not int or v < 0 for v in self.shape)):
            raise _Unsupported("native array shape requires declared bounded numerical profile")
        count = math.prod(self.shape)
        if count > 5_000_000 or count * self.dtype.itemsize > 128 * 1024**2:
            raise _Unsupported("native array allocation exceeds independent reader bound")
        return super().read(unpickler, ensure_native_byte_order)


class _LinearUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if (module, name) not in _GLOBALS:
            raise _Unsupported("native model serialization requires an unapproved class")
        if (module, name) == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
            return _BoundedArrayWrapper
        return getattr(importlib.import_module(module), name)


def read_linear_reference(path, algorithm, features):
    if algorithm not in {"lr", "lr_regressor", "lr_multiclass"}:
        raise _Unsupported("native model profile requires an independent approved reader")
    with path.open("rb") as stream:
        model = _LinearUnpickler(str(path), stream, ensure_native_byte_order=True).load()
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    if type(model) is not Pipeline or [name for name, _ in model.steps] not in (
        ["impute", "scale", "lr"], ["imputer", "scaler", "lr"],
        ["impute", "scale", "model"], ["imputer", "scaler", "model"],
        ["impute", "scale", "ridge"], ["imputer", "scaler", "ridge"],
        ["impute", "scale", "regressor"],
    ):
        raise _Unsupported("native linear preprocessing chain requires its declared reader")
    imputer, scale, learner = [v for _, v in model.steps]
    if (type(imputer) is not SimpleImputer or type(scale) is not StandardScaler
            or type(learner) not in {LogisticRegression, LinearRegression, Ridge}
            or imputer.strategy != "median" or imputer.add_indicator
            or len(imputer.statistics_) != len(features) or learner.n_features_in_ != len(features)):
        raise _Unsupported("native model preprocessing profile unsupported")
    if algorithm == "lr_regressor" and type(learner) not in {LinearRegression, Ridge}:
        raise ValueError("regression native learner mismatch")
    if algorithm != "lr_regressor" and type(learner) is not LogisticRegression:
        raise ValueError("classification native learner mismatch")

    def predict(frame):
        matrix = frame[list(features)]
        dtype = np.float32 if all(str(value) == "float32" for value in matrix.dtypes) else np.float64
        values = matrix.to_numpy(dtype=dtype, copy=True)
        if np.isinf(values).any():
            raise ValueError("native feature infinity")
        for i, median in enumerate(imputer.statistics_):
            if not math.isfinite(float(median)):
                raise _Unsupported("all-missing feature deletion requires a separate profile")
            values[np.isnan(values[:, i]), i] = median
        if scale.with_mean:
            values -= np.asarray(scale.mean_)
        if scale.with_std:
            values /= np.asarray(scale.scale_)
        logits = values @ np.asarray(learner.coef_).T + np.asarray(learner.intercept_)
        if algorithm == "lr_regressor":
            return np.asarray(logits).reshape(-1)
        if algorithm == "lr":
            flat = np.asarray(logits).reshape(-1)
            # Preserve native float32 arithmetic when the authenticated compact
            # training representation is explicitly selected.
            return 1 / (1 + np.exp(-flat))
        logits -= np.max(logits, axis=1, keepdims=True)
        exponential = np.exp(logits)
        return exponential / exponential.sum(axis=1, keepdims=True)
    return model, predict


def independent_linear_fit(model, frame, features, target, *, weights=None, classes=()):
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    # The installed learner is a numerical primitive; no platform recipe,
    # producer scoring/metric/tuning helper participates in this independent fit.
    rebuilt = clone(model)
    kwargs = {} if weights is None else {f"{model.steps[-1][0]}__sample_weight": weights}
    dtype = int if isinstance(model.steps[-1][1], LogisticRegression) else float
    labels = frame[target].to_numpy(dtype=dtype) if not classes else np.asarray([list(classes).index(v) for v in frame[target]])
    rebuilt.fit(frame[list(features)], labels, **kwargs)
    old_impute, old_scale, old_learner = [v for _, v in model.steps]
    new_impute, new_scale, new_learner = [v for _, v in rebuilt.steps]
    for actual, expected in ((old_impute.statistics_, new_impute.statistics_),
            (old_scale.mean_, new_scale.mean_), (old_scale.scale_, new_scale.scale_),
            (old_learner.coef_, new_learner.coef_), (old_learner.intercept_, new_learner.intercept_)):
        compact = np.asarray(actual).dtype == np.float32
        if not np.allclose(actual, expected, rtol=1e-6 if compact else 1e-8, atol=1e-7 if compact else 1e-10, equal_nan=True):
            raise ValueError("native coefficients or preprocessing differ from independent approved fit population")
    return rebuilt

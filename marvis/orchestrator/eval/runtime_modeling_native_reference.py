"""Restricted native linear-model reader and independent inference/re-fit.

Only installed numerical/sklearn data classes are allowed by the joblib reader.
Archive callbacks, imports and executable globals are refused before invocation.
"""
import importlib
import math
import pickle

import numpy as np
from joblib.numpy_pickle import NumpyArrayWrapper, NumpyUnpickler
from joblib.numpy_pickle_utils import Unpickler, _ensure_native_byte_order
from numpy._core.multiarray import _reconstruct, scalar

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

# Classes are admitted as inert NEWOBJ/BUILD state, not as arbitrary callable
# reducers. In particular Booster constructors accept external model filenames.
_DATA_REDUCERS = {
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("builtins", "set"), ("builtins", "slice"), ("builtins", "bytearray"),
    ("builtins", "dict"), ("collections", "defaultdict"), ("collections", "OrderedDict"),
}


def _bounded_shape(shape, dtype):
    if (not isinstance(shape, tuple) or len(shape) > 8
            or any(type(v) is not int or v < 0 for v in shape)):
        raise _Unsupported("native array shape requires declared bounded numerical profile")
    count = math.prod(shape)
    if count > 5_000_000 or count * np.dtype(dtype).itemsize > 128 * 1024**2:
        raise _Unsupported("native array allocation exceeds independent reader bound")


def _check_reducer(function, args):
    from numpy.random import MT19937
    from numpy.random._pickle import __bit_generator_ctor, __randomstate_ctor
    if function is bytearray:
        if len(args) != 1 or type(args[0]) not in {bytes, bytearray} or len(args[0]) > 128 * 1024**2:
            raise _Unsupported("native byte buffer allocation outside numerical profile")
    elif function is scalar:
        if not args or np.dtype(args[0]).itemsize > 128 * 1024**2:
            raise _Unsupported("native scalar allocation outside numerical profile")
    elif function is __bit_generator_ctor:
        if len(args) > 1 or args and args[0] not in ("MT19937", MT19937):
            raise _Unsupported("native random generator constructor outside numerical profile")
    elif function is __randomstate_ctor:
        if (len(args) > 2 or args and not (type(args[0]) is MT19937 or args[0] == "MT19937")
                or len(args) == 2 and args[1] is not __bit_generator_ctor):
            raise _Unsupported("native random-state callback outside numerical profile")
    elif function is _reconstruct:
        if len(args) != 3 or args[0] is not np.ndarray:
            raise _Unsupported("native array reconstruction outside numerical profile")
        _bounded_shape(args[1], args[2])
    elif not any(function is getattr(importlib.import_module(module), name)
                 for module, name in _DATA_REDUCERS):
        raise _Unsupported("native serialization callable constructor is not a data reducer")


class _DataUnpickler(Unpickler):
    dispatch = Unpickler.dispatch.copy()

    def get_extension(self, code):
        # The process-global pickle extension cache can bypass find_class.
        raise _Unsupported("native serialization extension globals are unsupported")

    def load_reduce(self):
        _check_reducer(self.stack[-2], self.stack[-1])
        Unpickler.load_reduce(self)

    def _validate_build(self):
        if type(self.stack[-2]) is np.ndarray:
            state = self.stack[-1]
            if type(state) is not tuple or len(state) != 5:
                raise _Unsupported("native array state outside bounded numerical profile")
            _bounded_shape(state[1], state[2])

    def load_build(self):
        self._validate_build()
        Unpickler.load_build(self)

    def load_newobj(self):
        if self.stack[-1]:
            raise _Unsupported("native object construction arguments require an approved data reducer")
        Unpickler.load_newobj(self)

    def load_newobj_ex(self):
        if self.stack[-1] or self.stack[-2]:
            raise _Unsupported("native object construction arguments require an approved data reducer")
        Unpickler.load_newobj_ex(self)

    def _instantiate(self, klass, args):
        _check_reducer(klass, args)
        super()._instantiate(klass, args)

    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj_ex


def validate_pipeline_safety(model):
    if getattr(model, "memory", None) is not None or getattr(model, "transform_input", None) is not None:
        raise _Unsupported("native preprocessing cache/metadata routing requires a separate profile")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imputer, scaler = [value for _, value in model.steps[:2]]
    if type(imputer) is not SimpleImputer or type(scaler) is not StandardScaler:
        raise _Unsupported("native preprocessing requires declared median/standard scaling")
    parameters = imputer.get_params(deep=False)
    missing = parameters.pop("missing_values")
    defaults = SimpleImputer(strategy="median").get_params(deep=False)
    defaults.pop("missing_values")
    if (type(missing) is not float or not math.isnan(missing) or parameters != defaults
            or scaler.get_params(deep=False) != StandardScaler().get_params(deep=False)):
        raise _Unsupported("native preprocessing configuration outside declared production profile")


# The reader executes only this closed numerical parameter vocabulary. Unknown
# library kwargs may be filenames, callbacks, devices or distributed endpoints.
_XGB_PARAMS = set("objective base_score booster callbacks colsample_bylevel colsample_bynode colsample_bytree device early_stopping_rounds enable_categorical eval_metric feature_types feature_weights gamma grow_policy importance_type interaction_constraints learning_rate max_bin max_cat_threshold max_cat_to_onehot max_delta_step max_depth max_leaves min_child_weight missing monotone_constraints multi_strategy n_estimators n_jobs num_parallel_tree random_state reg_alpha reg_lambda sampling_method scale_pos_weight subsample tree_method validate_parameters verbosity num_class".split())
_LGB_PARAMS = set("boosting_type boosting class_weight colsample_bytree importance_type learning_rate max_depth min_child_samples min_child_weight min_split_gain n_estimators n_jobs num_leaves objective random_state reg_alpha reg_lambda subsample subsample_for_bin subsample_freq deterministic force_col_wise force_row_wise verbose verbosity seed num_threads num_class metric feature_fraction bagging_fraction bagging_freq max_bin min_data_in_leaf min_sum_hessian_in_leaf lambda_l1 lambda_l2 min_gain_to_split scale_pos_weight is_unbalance monotone_constraints monotone_constraints_method monotone_penalty feature_pre_filter extra_trees path_smooth linear_tree data_random_seed feature_fraction_seed bagging_seed drop_seed extra_seed early_stopping_round early_stopping_rounds num_iterations first_metric_only boost_from_average sigmoid max_delta_step max_depth min_data_in_bin bin_construct_sample_cnt feature_fraction_bynode".split())
_CAT_PARAMS = set("iterations learning_rate depth l2_leaf_reg model_size_reg rsm loss_function border_count feature_border_type per_float_feature_quantization input_borders output_borders random_seed thread_count verbose logging_level metric_period od_type od_wait od_pval eval_metric custom_metric use_best_model best_model_min_trees leaf_estimation_iterations leaf_estimation_method leaf_estimation_backtracking random_strength bootstrap_type bagging_temperature subsample sampling_unit sampling_frequency grow_policy min_data_in_leaf max_leaves score_function class_weights auto_class_weights scale_pos_weight boosting_type boost_from_average langevin diffusion_temperature posterior_sampling monotone_constraints feature_weights first_feature_use_penalties penalties_coefficient allow_const_label nan_mode one_hot_max_size max_ctr_complexity ctr_leaf_count_limit store_all_simple_ctr allow_writing_files cat_features".split()) - {"input_borders", "output_borders"}


def _plain_parameter(value):
    if value is None or type(value) in {str, int, float, bool}:
        return True
    if type(value) in {list, tuple}:
        return all(_plain_parameter(v) for v in value)
    if type(value) is dict:
        return all(type(k) in {str, int, float, bool} and _plain_parameter(v) for k, v in value.items())
    return False


def validate_native_parameters(algorithm, params):
    allowed = _XGB_PARAMS if algorithm.startswith("xgb") else _LGB_PARAMS if algorithm.startswith("lgb") else _CAT_PARAMS
    if set(params) - allowed or not all(_plain_parameter(v) for v in params.values()):
        raise _Unsupported("native learner parameter outside closed numerical profile")
    if params.get("callbacks") is not None:
        raise _Unsupported("native learner callbacks require a separate profile")
    if algorithm.startswith("xgb") and params.get("device") not in {None, "cpu"}:
        raise _Unsupported("native learner device outside local CPU profile")
    if algorithm == "catboost" and params.get("allow_writing_files") is not False:
        raise _Unsupported("native CatBoost must explicitly disable external files")


class _BoundedArrayWrapper(NumpyArrayWrapper):
    def read_array(self, unpickler, ensure_native_byte_order):
        if not self.dtype.hasobject:
            return super().read_array(unpickler, ensure_native_byte_order)
        # Older supported joblib versions use plain pickle.load for object
        # arrays. Keep the outer allowlist on the nested stream as well.
        inner = _DataUnpickler(unpickler.file_handle)
        inner.find_class = unpickler.find_class
        array = inner.load()
        return _ensure_native_byte_order(array) if ensure_native_byte_order else array

    def read(self, unpickler, ensure_native_byte_order):
        _bounded_shape(self.shape, self.dtype)
        return super().read(unpickler, ensure_native_byte_order)


class _LinearUnpickler(NumpyUnpickler, _DataUnpickler):
    dispatch = NumpyUnpickler.dispatch.copy()
    dispatch[pickle.REDUCE[0]] = _DataUnpickler.load_reduce
    dispatch[pickle.NEWOBJ[0]] = _DataUnpickler.load_newobj
    dispatch[pickle.NEWOBJ_EX[0]] = _DataUnpickler.load_newobj_ex

    def load_build(self):
        self._validate_build()
        NumpyUnpickler.load_build(self)

    dispatch[pickle.BUILD[0]] = load_build

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
    validate_pipeline_safety(model)
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
    validate_pipeline_safety(model)
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

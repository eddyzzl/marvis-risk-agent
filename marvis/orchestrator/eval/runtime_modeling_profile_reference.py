"""Independent readers for native MLP, CatBoost, scorecard and seed ensembles.

Only numerical library data classes enter the restricted pickle reader. Model
inference uses retained coefficients/portable trees, never producer scorers or
the learner's predict methods. Exact-population refitting uses installed learners.
"""
import hashlib
import importlib
import json
import math
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .runtime_archive_reader import _Unsupported
from .runtime_modeling_extended_reference import (
    catboost_json_reference,
    ensemble_reference,
)
from .runtime_modeling_native_reference import (
    _LinearUnpickler,
    independent_linear_fit,
    read_linear_reference,
    validate_native_parameters,
    validate_pipeline_safety,
)
from .runtime_modeling_tree_reference import (
    PROFILES,
    early_stop_membership,
    independent_tree_fit,
    read_tree_reference,
)

EXTRA_PROFILES = {"catboost", "mlp", "mlp_regressor", "mlp_multiclass", "scorecard", "ensemble"}
_NUMERICAL_CLASSES = {
    ("catboost.core", "CatBoostClassifier"),
    ("sklearn.neural_network._multilayer_perceptron", "MLPClassifier"),
    ("sklearn.neural_network._multilayer_perceptron", "MLPRegressor"),
    ("sklearn.neural_network._stochastic_optimizers", "AdamOptimizer"),
    ("sklearn.neural_network._stochastic_optimizers", "SGDOptimizer"),
    ("sklearn.preprocessing._label", "LabelBinarizer"),
    ("numpy.random._pickle", "__randomstate_ctor"),
    ("numpy.random._pickle", "__bit_generator_ctor"),
    ("numpy.random._mt19937", "MT19937"),
    ("numpy.random.mtrand", "RandomState"),
    ("sklearn.isotonic", "IsotonicRegression"),
}


@dataclass
class _WoeRecord:
    feature: str
    edges: tuple
    woe_by_bin: tuple
    na_woe: float | None


class _ProfileUnpickler(_LinearUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("marvis.feature.contracts", "WOEResult"):
            # Read the persisted data shape without importing producer code.
            return _WoeRecord
        if (module, name) in _NUMERICAL_CLASSES:
            return getattr(importlib.import_module(module), name)
        return super().find_class(module, name)


def read_numerical_payload(path):
    if Path(path).stat().st_size > 128 * 1024**2:
        raise _Unsupported("native numerical payload exceeds independent reader bound")
    with Path(path).open("rb") as stream:
        return _ProfileUnpickler(str(path), stream, ensure_native_byte_order=True).load()


def _close(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-7, equal_nan=True):
        raise ValueError(message)


def _mlp_prediction(model, features, frame):
    imputer, scaler, learner = [value for _, value in model.steps]
    values = frame[list(features)].to_numpy(dtype=float, copy=True)
    if np.isinf(values).any():
        raise ValueError("native MLP infinite input")
    if not np.isfinite(imputer.statistics_).all():
        raise _Unsupported("native MLP all-missing feature deletion")
    values = np.where(np.isnan(values), imputer.statistics_, values)
    if scaler.with_mean:
        values -= scaler.mean_
    if scaler.with_std:
        values /= scaler.scale_
    for index, (weights, intercept) in enumerate(zip(learner.coefs_, learner.intercepts_, strict=True)):
        if np.asarray(weights).ndim != 2 or len(intercept) != weights.shape[1] or values.shape[1] != weights.shape[0]:
            raise ValueError("native MLP layer dimensions mismatch")
        values = values @ weights + intercept
        activation = learner.out_activation_ if index == len(learner.coefs_) - 1 else learner.activation
        if activation == "relu":
            values = np.maximum(values, 0)
        elif activation == "tanh":
            values = np.tanh(values)
        elif activation == "logistic":
            # Stable logistic, independently evaluated from fitted arrays.
            from scipy.special import expit
            values = expit(values)
        elif activation == "softmax":
            values = np.exp(values - values.max(axis=1, keepdims=True))
            values /= values.sum(axis=1, keepdims=True)
        elif activation != "identity":
            raise _Unsupported("native MLP unknown activation")
    return values[:, 0] if values.shape[1] == 1 else values


def _cat_envelope(model):
    # Export is a numerical serialization operation, not model prediction.
    with tempfile.TemporaryDirectory(prefix="marvis-independent-cat-") as root:
        path = Path(root) / "model.json"
        model.save_model(str(path), format="json")
        if path.stat().st_size > 128 * 1024**2:
            raise _Unsupported("CatBoost portable tree envelope exceeds bound")
        payload = json.loads(path.read_text())
    # Training timestamps and library run GUIDs are not model semantics.
    return {key: payload[key] for key in ("features_info", "oblivious_trees", "scale_and_bias") if key in payload}


def _woe_matrix(frame, features, maps):
    import pandas as pd
    columns = {}
    for feature in features:
        mapping = maps[feature]
        edges, scores = list(mapping.edges), list(mapping.woe_by_bin)
        if len(edges) != len(scores) + 1 or any(a >= b for a, b in pairwise(edges)):
            raise ValueError("native scorecard WOE edge contract mismatch")
        columns[feature] = [float(scores[bisect_right(edges[1:-1], float(v))]) if math.isfinite(float(v))
                            else float(mapping.na_woe or 0.0) for v in frame[feature]]
    return pd.DataFrame(columns, index=frame.index)


def read_profile(path, algorithm, features, *, metadata=None, dependency_paths=None):
    if algorithm in PROFILES:
        return read_tree_reference(path, algorithm, features)
    if algorithm in {"lr", "lr_regressor", "lr_multiclass"}:
        return read_linear_reference(path, algorithm, features)
    if algorithm not in EXTRA_PROFILES:
        raise _Unsupported("native model algorithm has no independent declared profile")
    model = read_numerical_payload(path)
    if algorithm.startswith("mlp"):
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        if type(model) is not Pipeline or [n for n, _ in model.steps] != ["impute", "scale", "mlp"]:
            raise _Unsupported("native MLP preprocessing chain unsupported")
        validate_pipeline_safety(model)
        imputer, scale, learner = [v for _, v in model.steps]
        expected = MLPRegressor if algorithm == "mlp_regressor" else MLPClassifier
        if (type(imputer) is not SimpleImputer or imputer.strategy != "median" or imputer.add_indicator
                or type(scale) is not StandardScaler or type(learner) is not expected
                or learner.n_features_in_ != len(features) or len(learner.coefs_) > 64):
            raise _Unsupported("native MLP installed profile mismatch")
        return model, lambda frame: _mlp_prediction(model, features, frame)
    if algorithm == "catboost":
        from catboost import CatBoostClassifier
        if type(model) is not CatBoostClassifier or model.feature_names_ != list(features):
            raise ValueError("native CatBoost class/feature contract mismatch")
        validate_native_parameters(algorithm, model.get_params())
        envelope = _cat_envelope(model)
        return model, lambda frame: np.asarray(catboost_json_reference(envelope, frame[list(features)].to_numpy().tolist()))
    if algorithm == "scorecard":
        from sklearn.linear_model import LogisticRegression
        if (not isinstance(model, dict) or type(model.get("model")) is not LogisticRegression
                or list(model.get("features", ())) != list(features)
                or set(model.get("woe_maps", {})) != set(features)
                or any(type(v) is not _WoeRecord for v in model["woe_maps"].values())):
            raise ValueError("native scorecard payload contract mismatch")
        if metadata is not None:
            actual = {k: vars(v) for k, v in model["woe_maps"].items()}
            if json.dumps(actual, sort_keys=True) != json.dumps(metadata["woe_maps"], sort_keys=True):
                raise ValueError("scorecard serialized and registered WOE maps differ")
            if model.get("scorecard_table") != metadata["scorecard_table"]:
                raise ValueError("scorecard serialized and registered point tables differ")
        def predict(frame):
            from scipy.special import expit
            encoded = _woe_matrix(frame, features, model["woe_maps"])
            return expit(encoded.to_numpy() @ model["model"].coef_[0] + model["model"].intercept_[0])
        return model, predict
    members = model.get("members") if isinstance(model, dict) else None
    params = (metadata or {}).get("params", {})
    if (not isinstance(model, dict) or model.get("kind") != "seed_bagging" or not members or len(members) > 100
            or len(members) != params.get("ensemble_member_count")
            or [m["artifact_id"] for m in members] != params.get("ensemble_member_artifact_ids")
            or model.get("base_recipe") != params.get("ensemble_base_recipe")
            or model.get("weights") != [1 / len(members)] * len(members)):
        raise ValueError("native ensemble membership/weight contract mismatch")
    children = []
    for member in members:
        if member["algorithm"] != model["base_recipe"] or member["algorithm"] not in {"lr", "mlp", "xgb", "lgb", "catboost"}:
            raise ValueError("native ensemble member recipe mismatch")
        if dependency_paths is None or member["model_path"] not in dependency_paths:
            raise ValueError("native ensemble member lacks authenticated bytes")
        children.append(read_profile(dependency_paths[member["model_path"]], member["algorithm"], features))
    model = {**model, "independent_children": children}
    return model, lambda frame: np.asarray(ensemble_reference([p(frame).tolist() for _, p in children], model["weights"]))


def _scorecard_bins(values, labels, params):
    """Recompute supervised chi-square merges and monotonicity from fit rows."""
    from scipy.stats import chi2_contingency

    from .runtime_modeling_reference import edges_reference
    finite = [(float(v), int(y)) for v, y in zip(values, labels, strict=True) if math.isfinite(float(v))]
    maximum = int(params.get("scorecard_max_bins", 6))
    if maximum < 1 or maximum > 1000:
        raise _Unsupported("scorecard bin budget outside independent bounds")
    if not finite or len({v for v, _ in finite}) <= 1:
        edges = [-math.inf, math.inf]
    else:
        edges = edges_reference([v for v, _ in finite], min(100, maximum * 20, len(finite)))
    cells = [[0, 0] for _ in range(len(edges) - 1)]
    for value, label in finite:
        cells[bisect_right(edges[1:-1], value)][label] += 1
    if any(sum(c) for c in cells):
        edges = [edges[0], *(edges[i + 1] for i, cell in enumerate(cells) if sum(cell))]
        cells = [c for c in cells if sum(c)]
    def statistic(i):
        return chi2_contingency(np.asarray([cells[i], cells[i + 1]], dtype=float) + .5, correction=False)[:2]
    def merge(i):
        cells[i] = [a + b for a, b in zip(cells[i], cells.pop(i + 1), strict=True)]
        edges.pop(i + 1)
    while len(cells) > maximum:
        merge(min(range(len(cells) - 1), key=lambda i: statistic(i)[0]))
    while len(cells) > 2:
        chosen = max(range(len(cells) - 1), key=lambda i: statistic(i)[1])
        if statistic(chosen)[1] <= .05:
            break
        merge(chosen)
    while len(cells) > 1 and min(map(sum, cells)) < .05 * len(finite):
        smallest = min(range(len(cells)), key=lambda i: sum(cells[i]))
        choices = [i for i in (smallest - 1, smallest) if 0 <= i < len(cells) - 1]
        merge(min(choices, key=lambda i: statistic(i)[0]))
    direction = None
    if params.get("enforce_monotonic", True):
        direction = str(params.get("monotonic_direction") or "auto")
        direction = {"asc": "increasing", "up": "increasing", "desc": "decreasing", "down": "decreasing"}.get(direction, direction)
        if direction == "auto":
            rates = [c[1] / sum(c) if sum(c) else 0 for c in cells]
            inc = sum(max(a - b, 0) for a, b in pairwise(rates))
            dec = sum(max(b - a, 0) for a, b in pairwise(rates))
            corr = np.corrcoef(np.asarray(finite).T)[0, 1] if len(finite) > 1 and np.std([v for v, _ in finite]) else 0
            direction = "decreasing" if dec < inc or dec == inc and corr < 0 else "increasing"
        if direction not in {"increasing", "decreasing"}:
            raise ValueError("native scorecard monotonic direction invalid")
        while len(cells) > 1:
            rates = [c[1] / sum(c) if sum(c) else 0 for c in cells]
            violating = [i for i in range(len(cells) - 1)
                         if (rates[i] > rates[i + 1] if direction == "increasing" else rates[i] < rates[i + 1])]
            if not violating:
                break
            merge(min(violating, key=lambda i: statistic(i)[0]))
    edges[0], edges[-1] = -math.inf, math.inf
    return edges, direction


def _scorecard_fit(model, frame, config, weights):
    from sklearn.base import clone
    labels = frame[config["target_col"]].astype(int).tolist()
    features, params = config["features"], config["params"]
    total_bad, total_good = sum(labels), len(labels) - sum(labels)
    expected_maps = {}
    table = {(row["feature"], row["bin_index"]): row for row in model["scorecard_table"]}
    if len(table) != len(model["scorecard_table"]) or ("__base__", -999) not in table:
        raise ValueError("scorecard point table has duplicate bins or lacks base points")
    expected_keys = {("__base__", -999)}
    for feature in features:
        values = frame[feature].tolist()
        edges, direction = _scorecard_bins(values, labels, params)
        actual = model["woe_maps"][feature]
        _close(actual.edges, edges, "scorecard edges differ from independent fit population")
        groups = {i: [0, 0] for i in range(len(edges) - 1)}
        for value, label in zip(values, labels, strict=True):
            index = bisect_right(edges[1:-1], float(value)) if math.isfinite(float(value)) else -1
            groups.setdefault(index, [0, 0])[label] += 1
        woes = {}
        for index, (good, bad) in groups.items():
            expected_keys.add((feature, index))
            gd, bd = (good + .5) / (total_good + .5 * len(groups)), (bad + .5) / (total_bad + .5 * len(groups))
            woe = math.log(gd / bd)
            woes[index] = woe
            row = table.get((feature, index))
            if row is None or row["count"] != good + bad or row["bad_count"] != bad or row["good_count"] != good:
                raise ValueError("scorecard counts differ from independent fit population")
            _close([row["woe"], row["iv_contribution"]], [woe, (gd - bd) * woe], "scorecard WOE/IV mismatch")
            _close(row["bad_rate"], bad / (good + bad) if good + bad else 0.0, "scorecard bad rate mismatch")
            lower = edges[index] if index >= 0 else math.nan
            upper = edges[index + 1] if index >= 0 else math.nan
            if (row["lower"] != (lower if math.isfinite(lower) else None)
                    or row["upper"] != (upper if math.isfinite(upper) else None)):
                raise ValueError("scorecard point table edges differ from independent fit")
            if row.get("monotonic_direction") != direction:
                raise ValueError("scorecard monotonic direction mismatch")
        expected_maps[feature] = _WoeRecord(feature, tuple(edges), tuple(woes[i] for i in range(len(edges) - 1)), woes.get(-1))
        _close(actual.woe_by_bin, expected_maps[feature].woe_by_bin, "scorecard WOE encoding mismatch")
        if actual.na_woe != expected_maps[feature].na_woe and not (
                actual.na_woe is not None and expected_maps[feature].na_woe is not None
                and math.isclose(actual.na_woe, expected_maps[feature].na_woe, rel_tol=1e-10)):
            raise ValueError("scorecard missing-value WOE mismatch")
    if set(table) != expected_keys:
        raise ValueError("scorecard point table includes unbound bins")
    learner = model["model"]
    rebuilt = clone(learner).fit(_woe_matrix(frame, features, expected_maps), labels, sample_weight=weights)
    _close(learner.coef_, rebuilt.coef_, "scorecard coefficients differ from independent fit population")
    _close(learner.intercept_, rebuilt.intercept_, "scorecard intercept differs from independent fit population")
    factor = float(model["params"].get("pdo", 50)) / math.log(2)
    offset = float(model["params"].get("base_score", 600)) - factor * math.log(float(model["params"].get("base_odds", 50)))
    for row in model["scorecard_table"]:
        coefficient = None if row["feature"] == "__base__" else float(learner.coef_[0][features.index(row["feature"])])
        if coefficient is not None:
            _close(row["coefficient"], coefficient, "scorecard point table coefficient mismatch")
        expected = offset - factor * float(learner.intercept_[0]) if coefficient is None else -factor * coefficient * row["woe"]
        _close(row["points"], expected, "scorecard point scale mismatch")
    base = table[("__base__", -999)]
    _close([base["factor"], base["offset"], base["pdo"], base["base_score"], base["base_odds"]],
           [factor, offset, model["params"].get("pdo", 50), model["params"].get("base_score", 600), model["params"].get("base_odds", 50)],
           "scorecard point table scale parameters mismatch")
    return rebuilt


def independent_profile_fit(model, frame, config, *, metadata, test_frame=None, classes=(), frozen_params=None):
    from sklearn.base import clone
    algorithm, features, target = metadata["algorithm"], config["features"], config["target_col"]
    weight_col = config["params"].get("sample_weight_col")
    weights = frame[weight_col].to_numpy(dtype=float) if weight_col else None
    if algorithm in PROFILES:
        return independent_tree_fit(model, frame, config, classes=classes, frozen_params=frozen_params)
    if algorithm in {"lr", "lr_regressor", "lr_multiclass"}:
        return independent_linear_fit(model, frame, features, target, weights=weights, classes=classes)
    if algorithm == "scorecard":
        return _scorecard_fit(model, frame, config, weights)
    if algorithm.startswith("mlp"):
        validate_pipeline_safety(model)
        labels = frame[target].to_numpy() if not classes else np.asarray([list(classes).index(v) for v in frame[target]])
        rebuilt = clone(model).fit(frame[features], labels, mlp__sample_weight=weights)
        for old, new in zip(model.steps[:2], rebuilt.steps[:2], strict=True):
            for attribute in ("statistics_", "mean_", "scale_"):
                if hasattr(old[1], attribute):
                    _close(getattr(old[1], attribute), getattr(new[1], attribute), "MLP preprocessing differs from independent fit population")
        old, new = model.steps[-1][1], rebuilt.steps[-1][1]
        for a, b in zip([*old.coefs_, *old.intercepts_], [*new.coefs_, *new.intercepts_], strict=True):
            _close(a, b, "MLP weights differ from independent fit population or early-stop replay")
        if old.n_iter_ != new.n_iter_:
            raise ValueError("MLP early-stop iteration mismatch")
        return rebuilt
    if algorithm == "catboost":
        validate_native_parameters(algorithm, model.get_params())
        train, valid = early_stop_membership(frame, config) if config.get("early_stopping_rounds") else (frame, test_frame)
        if valid is None or valid.empty:
            raise _Unsupported("CatBoost requires its retained evaluation population")
        rebuilt = clone(model)
        rebuilt.fit(train[features], train[target].to_numpy(dtype=int),
                    sample_weight=train[weight_col].to_numpy(dtype=float) if weight_col else None,
                    eval_set=(valid[features], valid[target].to_numpy(dtype=int)), verbose=False)
        if _cat_envelope(model) != _cat_envelope(rebuilt):
            raise ValueError("CatBoost trees differ from independent fit population or early-stop replay")
        return rebuilt
    if algorithm == "ensemble":
        if len(model["independent_children"]) != int(config["params"].get("n_members") or 5):
            raise ValueError("native ensemble frozen member count mismatch")
        if model["base_recipe"] != (config["params"].get("base_recipe") or "lgb"):
            raise ValueError("native ensemble frozen base recipe mismatch")
        for index, (child, _) in enumerate(model["independent_children"]):
            params = {k: v for k, v in config["params"].items() if k not in {"base_recipe", "n_members"}}
            seed = int(hashlib.sha256(f"{int(config['seed'])}:ensemble_member:{index}".encode()).hexdigest()[:8], 16) % 2_147_483_647
            recipe = model["base_recipe"]
            child_config = {**config, "recipe_id": recipe, "params": params, "seed": seed}
            learner = child.steps[-1][1] if hasattr(child, "steps") else child
            child_seed = getattr(learner, "random_state", None)
            if recipe == "catboost":
                child_seed = learner.get_params().get("random_seed")
            if child_seed != seed:
                raise ValueError("native ensemble member seed mismatch")
            if hasattr(learner, "get_params"):
                for key, value in params.items():
                    if key in learner.get_params() and json.dumps(learner.get_params()[key]) != json.dumps(value):
                        raise ValueError("native ensemble member frozen hyperparameters mismatch")
            independent_profile_fit(child, frame, child_config, metadata={"algorithm": recipe},
                                    test_frame=test_frame, classes=classes)
        return model
    raise _Unsupported("native fit requires an independent declared profile")

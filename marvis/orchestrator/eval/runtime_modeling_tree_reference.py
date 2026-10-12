"""Independent numerical-tree envelopes and exact-population native refitting.

The restricted loader admits only installed data/numerical classes. Prediction
walks authenticated portable trees, without invoking producer scoring code.
Refitting uses the installed library's numerical learner, never a platform recipe.
"""
import hashlib
import json
import math

import numpy as np

from .runtime_archive_reader import _Unsupported
from .runtime_modeling_native_reference import _LinearUnpickler

PROFILES = {"xgb", "xgb_regressor", "xgb_multiclass", "lgb", "lgb_regressor", "lgb_multiclass"}


def _sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def _response(margins, kind):
    if kind == "binary":
        return np.asarray([_sigmoid(float(v[0])) for v in margins])
    if kind == "regression":
        return np.asarray([v[0] for v in margins])
    values = np.asarray(margins, dtype=float)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def _xgb_leaf(tree, row):
    node = 0
    limit = len(tree["left_children"])
    if limit > 1_000_000 or any(tree["split_type"]):
        raise _Unsupported("categorical/oversize XGB trees require a declared reference")
    for _ in range(limit):
        left = tree["left_children"][node]
        if left == -1:
            return float(tree["split_conditions"][node])
        value = row[tree["split_indices"][node]]
        yes = bool(tree["default_left"][node]) if math.isnan(value) else value < tree["split_conditions"][node]
        node = left if yes else tree["right_children"][node]
        if not 0 <= node < limit:
            raise ValueError("XGB tree child outside authenticated tree")
    raise ValueError("XGB tree cycle")


def _lgb_leaf(tree, row):
    node = tree
    for _ in range(512):
        if "leaf_value" in node:
            return float(node["leaf_value"])
        if node["decision_type"] != "<=":
            raise _Unsupported("categorical LGB tree requires a declared reference")
        value = row[node["split_feature"]]
        missing = math.isnan(value) or node["missing_type"] == "Zero" and abs(value) <= 1e-35
        if missing and node["missing_type"] != "None":
            yes = bool(node["default_left"])
        else:
            # Native LightGBM treats NaN as zero when missing_type is None.
            yes = (0 if math.isnan(value) else value) <= float(node["threshold"])
        node = node["left_child"] if yes else node["right_child"]
    raise ValueError("LGB tree depth exceeds declared parser bound")


def tree_envelope(model, algorithm, features):
    kind = "regression" if algorithm.endswith("regressor") else "multiclass" if algorithm.endswith("multiclass") else "binary"
    if algorithm.startswith("xgb"):
        raw = json.loads(model.get_booster().save_raw(raw_format="json"))
        learner = raw["learner"]
        gradient = learner["gradient_booster"]
        if gradient["name"] != "gbtree" or learner["feature_names"] != list(features):
            raise _Unsupported("non-gbtree or unnamed XGB profile unsupported")
        objective = {"binary": "binary:logistic", "regression": "reg:squarederror", "multiclass": "multi:softprob"}[kind]
        if learner["objective"]["name"] != objective:
            raise _Unsupported("XGB objective requires its independent link reference")
        tree_model = gradient["model"]
        if int(tree_model["gbtree_model_param"]["num_parallel_tree"]) != 1:
            raise _Unsupported("parallel XGB forest profile unsupported")
        base = json.loads(learner["learner_model_param"]["base_score"])
        base = base if isinstance(base, list) else [base]
        if kind == "binary":
            if not 0 < base[0] < 1:
                raise ValueError("XGB invalid base probability")
            base = [math.log(base[0] / (1 - base[0]))]
        count = int(learner["learner_model_param"]["num_class"]) if kind == "multiclass" else 1
        if len(base) == 1:
            base *= count
        if len(base) != count:
            raise ValueError("XGB base/class envelope mismatch")
        best = learner["attributes"].get("best_iteration")
        stop = tree_model["iteration_indptr"][int(best) + 1] if best is not None else len(tree_model["trees"])
        envelope = {"library": "xgb", "kind": kind, "base": base, "classes": count,
                    "trees": tree_model["trees"][:stop], "tree_info": tree_model["tree_info"][:stop],
                    "full_tree_model": tree_model, "objective": learner["objective"], "features": list(features)}
    else:
        booster = model.booster_ if hasattr(model, "booster_") else model
        dump = booster.dump_model(num_iteration=booster.best_iteration or -1)
        if dump["feature_names"] != list(features) or dump["average_output"] or dump.get("pandas_categorical"):
            raise _Unsupported("LGB categorical/average-output profile unsupported")
        objective = dump["objective"]
        expected = {"binary": "binary sigmoid:1", "regression": "regression", "multiclass": "multiclass num_class:"}[kind]
        if not objective.startswith(expected):
            raise _Unsupported("LGB objective requires its independent link reference")
        envelope = {"library": "lgb", "kind": kind, "classes": dump["num_tree_per_iteration"],
                    "trees": dump["tree_info"], "features": list(features), "objective": objective}
    if len(envelope["trees"]) > 100_000:
        raise _Unsupported("native tree inventory exceeds declared bound")
    return envelope


def portable_prediction(envelope, frame):
    values = frame[envelope["features"]].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("native tree feature infinity")
    margins = []
    for row in values:
        if envelope["library"] == "xgb":
            row = row.astype(np.float32)
            output = np.asarray(envelope["base"], dtype=np.float32)
            for tree, group in zip(envelope["trees"], envelope["tree_info"], strict=True):
                output[group] = np.float32(output[group] + np.float32(_xgb_leaf(tree, row)))
        else:
            output = np.zeros(envelope["classes"])
            for index, tree in enumerate(envelope["trees"]):
                output[index % len(output)] += _lgb_leaf(tree["tree_structure"], row)
        margins.append(output.tolist())
    return _response(margins, envelope["kind"])


def read_tree_reference(path, algorithm, features):
    if algorithm not in PROFILES:
        raise _Unsupported("native tree algorithm requires a declared profile")
    if algorithm in {"lgb_regressor", "lgb_multiclass"}:
        import lightgbm as lgb
        model = lgb.Booster(model_str=path.read_text())
    else:
        with path.open("rb") as stream:
            model = _LinearUnpickler(str(path), stream, ensure_native_byte_order=True).load()
        import xgboost as xgb
        import lightgbm as lgb
        expected = xgb.XGBRegressor if algorithm == "xgb_regressor" else xgb.XGBClassifier if algorithm.startswith("xgb") else lgb.LGBMClassifier
        if type(model) is not expected or model.n_features_in_ != len(features):
            raise ValueError("native tree installed class mismatch")
    envelope = tree_envelope(model, algorithm, features)
    return model, lambda frame: portable_prediction(envelope, frame)


def early_stop_membership(train, config):
    seed = int(hashlib.sha256(f"{int(config['seed'])}:valid_fold".encode()).hexdigest()[:8], 16) % 2_147_483_647
    # Native contract samples unique row groups; test/OOT never participate.
    columns = config["params"].get("valid_group_cols") or []
    columns = [columns] if isinstance(columns, str) else columns
    columns = [c for c in columns if c in train]
    if columns:
        # Tuple identities and first-seen integer groups reproduce the declared
        # null-aware group contract without reading any outer holdout label.
        mapping = {}
        groups = []
        for row in train[columns].itertuples(index=False, name=None):
            key = tuple(None if value is None or isinstance(value, float) and math.isnan(value) else value for value in row)
            mapping.setdefault(key, len(mapping))
            groups.append(mapping[key])
        groups = np.asarray(groups)
    else:
        groups = np.arange(len(train))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    count = int(round(len(train) * .15))
    valid, covered = set(), 0
    for group in unique:
        if covered >= count or len(valid) >= len(unique) - 1:
            break
        valid.add(int(group))
        covered += sum(groups == group)
    fit = [i for i, group in enumerate(groups) if group not in valid]
    held = [i for i, group in enumerate(groups) if group in valid]
    if not fit or not held:
        raise _Unsupported("insufficient train population for early-stopping")
    return train.iloc[fit], train.iloc[held]


def independent_tree_fit(model, frame, config, *, classes=(), frozen_params=None):
    from sklearn.base import clone
    import lightgbm as lgb
    features, target, algorithm = config["features"], config["target_col"], config["recipe_id"]
    train = frame
    valid = None
    if config["early_stopping_rounds"]:
        train, valid = early_stop_membership(frame, config)
    labels = train[target].to_numpy() if not classes else np.asarray([list(classes).index(v) for v in train[target]])
    if algorithm in {"lgb_regressor", "lgb_multiclass"}:
        params = dict(frozen_params or {})
        # Text boosters do not retain constructor params. The caller must supply
        # authenticated frozen installed params, independently validated upstream.
        if not params:
            raise _Unsupported("native text-booster fit requires frozen parameter envelope")
        rounds = int(params.pop("num_boost_round"))
        dtrain = lgb.Dataset(train[features], label=labels)
        kwargs = {}
        if valid is not None:
            yvalid = valid[target].to_numpy() if not classes else np.asarray([list(classes).index(v) for v in valid[target]])
            kwargs = {"valid_sets": [lgb.Dataset(valid[features], label=yvalid, reference=dtrain)],
                      "callbacks": [lgb.early_stopping(config["early_stopping_rounds"], verbose=False)]}
        rebuilt = lgb.train(params, dtrain, num_boost_round=rounds, **kwargs)
    else:
        rebuilt = clone(model)
        kwargs = {}
        if valid is not None:
            yvalid = valid[target].to_numpy() if not classes else np.asarray([list(classes).index(v) for v in valid[target]])
            kwargs["eval_set"] = [(valid[features], yvalid)]
        if algorithm.startswith("xgb"):
            kwargs["verbose"] = False
        elif valid is not None:
            kwargs["callbacks"] = [lgb.early_stopping(config["early_stopping_rounds"], verbose=False)]
        rebuilt.fit(train[features], labels, **kwargs)
    original, replayed = tree_envelope(model, algorithm, features), tree_envelope(rebuilt, algorithm, features)
    if json.dumps(original, sort_keys=True, allow_nan=False) != json.dumps(replayed, sort_keys=True, allow_nan=False):
        raise ValueError("native portable tree envelope differs from independent exact-population refit")
    return rebuilt

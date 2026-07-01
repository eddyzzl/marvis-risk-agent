from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from marvis.feature.binning import equal_frequency_edges
from marvis.feature.metrics import (
    feature_auc,
    feature_ks,
    feature_psi,
    weighted_feature_auc,
    weighted_feature_ks,
    weighted_feature_psi,
)
from marvis.packs.modeling.contracts import ModelMetrics, TrainConfig
from marvis.packs.modeling.errors import ModelingError
from marvis.validation.overfitting import overfitting_check

_SAMPLE_WEIGHT_PARAM_KEYS = frozenset({
    "sample_weight_col",
    "sample_weight_column",
    "weight_col",
})
_MONOTONE_CONSTRAINT_KEYS = ("monotone_constraints", "monotonic_constraints")


def split_modeling_frame(
    frame: pd.DataFrame,
    config: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if config.split_col not in frame.columns:
        raise ModelingError(f"missing split column: {config.split_col}")
    train_value = config.split_values.get("train")
    test_value = config.split_values.get("test")
    if train_value is None or test_value is None:
        raise ModelingError("split_values must include train and test")
    train = frame[frame[config.split_col] == train_value]
    test = frame[frame[config.split_col] == test_value]
    if train.empty:
        raise ModelingError("missing train split")
    if test.empty:
        raise ModelingError("missing test split")
    oot_value = config.split_values.get("oot")
    oot = frame[frame[config.split_col] == oot_value] if oot_value is not None else None
    if oot is not None and oot.empty:
        oot = None
    return train, test, oot


def model_params(params: dict | None) -> dict:
    """Drop platform-only controls before passing params into estimator constructors."""
    return {
        str(key): value
        for key, value in dict(params or {}).items()
        if str(key) not in _SAMPLE_WEIGHT_PARAM_KEYS
    }


def resolve_auto_scale_pos_weight(params: dict, train: pd.DataFrame, config: TrainConfig) -> dict:
    """Resolve ``scale_pos_weight='auto'`` using the effective training split.

    Scenario templates can request automatic class-imbalance handling, but
    LightGBM/XGBoost require a numeric ratio. Use the same sample-weight column
    as fitting when present so reject-inference / business weights are reflected.
    """
    out = dict(params)
    raw = out.get("scale_pos_weight")
    if not isinstance(raw, str) or raw.strip().lower() != "auto":
        return out
    target = pd.to_numeric(train[config.target_col], errors="coerce").to_numpy(dtype=float)
    weights = sample_weight_values(train, config)
    pos = _weighted_label_count(target, weights, 1.0)
    neg = _weighted_label_count(target, weights, 0.0)
    out["scale_pos_weight"] = float(neg / pos) if pos > 0 and neg > 0 else 1.0
    return out


def normalized_monotone_constraints(config: TrainConfig) -> tuple[int, ...] | None:
    raw = None
    for key in _MONOTONE_CONSTRAINT_KEYS:
        value = config.params.get(key)
        if value not in (None, ""):
            raw = value
            break
    if raw in (None, ""):
        return None
    features = tuple(str(feature) for feature in config.features)
    if isinstance(raw, dict):
        values = tuple(_constraint_value(raw.get(feature, 0), feature=feature) for feature in features)
    elif isinstance(raw, str):
        text = raw.strip().strip("()[]")
        values = tuple(
            _constraint_value(item.strip(), feature=f"index {index}")
            for index, item in enumerate(text.split(","))
            if item.strip()
        )
    else:
        values = tuple(_constraint_value(item, feature=f"index {index}") for index, item in enumerate(raw))
    if len(values) != len(features):
        raise ModelingError(
            f"monotone_constraints length {len(values)} does not match feature count {len(features)}"
        )
    return values


def _constraint_value(value, *, feature: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"monotone constraint for {feature} must be -1, 0, or 1") from exc
    if number not in {-1, 0, 1}:
        raise ModelingError(f"monotone constraint for {feature} must be -1, 0, or 1")
    return number


def sample_weight_col(config: TrainConfig) -> str:
    for key in _SAMPLE_WEIGHT_PARAM_KEYS:
        value = config.params.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def artifact_params(params: dict, config: TrainConfig) -> dict:
    out = dict(params)
    column = sample_weight_col(config)
    if column:
        out["sample_weight_col"] = column
    return out


def sample_weight_values(frame: pd.DataFrame, config: TrainConfig) -> np.ndarray | None:
    column = sample_weight_col(config)
    if not column:
        return None
    if column not in frame.columns:
        raise ModelingError(f"sample weight column not found in modeling frame: {column}")
    weights = pd.to_numeric(frame[column], errors="coerce")
    if weights.isna().any():
        raise ModelingError(f"sample weight column `{column}` contains null or non-numeric values")
    if (weights <= 0).any():
        raise ModelingError(f"sample weight column `{column}` contains non-positive values")
    if float(weights.sum()) <= 0:
        raise ModelingError(f"sample weight column `{column}` must have a positive total weight")
    return weights.to_numpy(dtype=float)


def _weighted_label_count(labels: np.ndarray, weights: np.ndarray | None, value: float) -> float:
    mask = labels == value
    if weights is None:
        return float(mask.sum())
    return float(weights[mask].sum())


def compute_model_metrics(
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    config: TrainConfig,
    *,
    oot_has_labels: bool = True,
) -> ModelMetrics:
    train_scores = _scores(score_fn, train)
    test_scores = _scores(score_fn, test)
    oot_scores = None if oot is None else _scores(score_fn, oot)
    # train/test are label-resolved upstream; OOT may be scoring-only (no labels) -> never
    # coerce its NaN target into a class, just skip OOT's label-dependent metrics.
    train_target = train[config.target_col].to_numpy(dtype=float)
    test_target = test[config.target_col].to_numpy(dtype=float)
    oot_target = (
        None
        if oot is None or not oot_has_labels
        else oot[config.target_col].to_numpy(dtype=float)
    )

    train_ks = feature_ks(train_scores, train_target)
    test_ks = feature_ks(test_scores, test_target)
    oot_ks = None if oot_scores is None or oot_target is None else feature_ks(oot_scores, oot_target)
    train_auc = feature_auc(train_scores, train_target)
    test_auc = feature_auc(test_scores, test_target)
    oot_auc = None if oot_scores is None or oot_target is None else feature_auc(oot_scores, oot_target)
    edges = equal_frequency_edges(train_scores, 10)
    weighted = _weighted_binary_metrics(
        train_scores,
        test_scores,
        oot_scores,
        train_target,
        test_target,
        oot_target,
        train,
        test,
        oot,
        config,
        edges,
    )
    gap_tt, gap_to, overfit_flag = overfitting_check(train_ks, test_ks, oot_ks)
    return ModelMetrics(
        train_ks=train_ks,
        test_ks=test_ks,
        oot_ks=oot_ks,
        train_auc=train_auc,
        test_auc=test_auc,
        oot_auc=oot_auc,
        psi_test_vs_train=feature_psi(train_scores, test_scores, edges),
        psi_oot_vs_train=None if oot_scores is None else feature_psi(train_scores, oot_scores, edges),
        overfit_train_test_gap=gap_tt,
        overfit_train_oot_gap=gap_to,
        overfit_flag=overfit_flag,
        **weighted,
    )


def _weighted_binary_metrics(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    oot_scores: np.ndarray | None,
    train_target: np.ndarray,
    test_target: np.ndarray,
    oot_target: np.ndarray | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    config: TrainConfig,
    edges: np.ndarray,
) -> dict[str, float | None]:
    if not sample_weight_col(config):
        return {}
    train_weight = sample_weight_values(train, config)
    test_weight = sample_weight_values(test, config)
    oot_weight = sample_weight_values(oot, config) if oot is not None else None
    return {
        "weighted_train_ks": weighted_feature_ks(train_scores, train_target, train_weight),
        "weighted_test_ks": weighted_feature_ks(test_scores, test_target, test_weight),
        "weighted_oot_ks": (
            None
            if oot_scores is None or oot_target is None or oot_weight is None
            else weighted_feature_ks(oot_scores, oot_target, oot_weight)
        ),
        "weighted_train_auc": weighted_feature_auc(train_scores, train_target, train_weight),
        "weighted_test_auc": weighted_feature_auc(test_scores, test_target, test_weight),
        "weighted_oot_auc": (
            None
            if oot_scores is None or oot_target is None or oot_weight is None
            else weighted_feature_auc(oot_scores, oot_target, oot_weight)
        ),
        "weighted_psi_test_vs_train": weighted_feature_psi(
            train_scores,
            test_scores,
            edges,
            base_weights=train_weight,
            compare_weights=test_weight,
        ),
        "weighted_psi_oot_vs_train": (
            None
            if oot_scores is None or oot_weight is None
            else weighted_feature_psi(
                train_scores,
                oot_scores,
                edges,
                base_weights=train_weight,
                compare_weights=oot_weight,
            )
        ),
    }


def compute_regression_metrics(
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    config: TrainConfig,
    *,
    oot_has_labels: bool = True,
) -> ModelMetrics:
    train_pred = _scores(score_fn, train)
    test_pred = _scores(score_fn, test)
    oot_pred = None if oot is None else _scores(score_fn, oot)
    train_target = train[config.target_col].to_numpy(dtype=float)
    test_target = test[config.target_col].to_numpy(dtype=float)
    oot_target = (
        None
        if oot is None or not oot_has_labels
        else oot[config.target_col].to_numpy(dtype=float)
    )

    train_rmse, train_mae, train_r2 = _regression_values(train_target, train_pred)
    test_rmse, test_mae, test_r2 = _regression_values(test_target, test_pred)
    oot_values = (
        None
        if oot_pred is None or oot_target is None
        else _regression_values(oot_target, oot_pred)
    )
    oot_rmse = None if oot_values is None else oot_values[0]
    oot_mae = None if oot_values is None else oot_values[1]
    oot_r2 = None if oot_values is None else oot_values[2]

    return ModelMetrics(
        train_ks=None,
        test_ks=None,
        oot_ks=None,
        train_auc=None,
        test_auc=None,
        oot_auc=None,
        psi_test_vs_train=None,
        psi_oot_vs_train=None,
        overfit_train_test_gap=test_rmse - train_rmse,
        overfit_train_oot_gap=None if oot_rmse is None else oot_rmse - train_rmse,
        overfit_flag=False,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        oot_rmse=oot_rmse,
        train_mae=train_mae,
        test_mae=test_mae,
        oot_mae=oot_mae,
        train_r2=train_r2,
        test_r2=test_r2,
        oot_r2=oot_r2,
    )


def compute_multiclass_model_metrics(
    proba_fn: Callable[[pd.DataFrame], np.ndarray],
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    config: TrainConfig,
    classes: tuple,
    *,
    oot_has_labels: bool = True,
) -> tuple[ModelMetrics, dict]:
    """Multiclass metrics for a K-class probability model.

    Mirrors ``compute_model_metrics``/``compute_regression_metrics`` but fills the
    multiclass scalar fields (macro_auc/logloss/accuracy) and returns a per-split
    ``per_class`` detail map alongside the ModelMetrics (the binary KS/AUC and the
    regression RMSE/MAE fields stay None). OOT label-dependent metrics are skipped
    when OOT carries no labels — its NaN target is never coerced into a class."""
    train_proba = _proba_2d(proba_fn, train, classes)
    test_proba = _proba_2d(proba_fn, test, classes)
    oot_proba = None if oot is None else _proba_2d(proba_fn, oot, classes)
    train_target = train[config.target_col].to_numpy()
    test_target = test[config.target_col].to_numpy()
    oot_target = (
        None
        if oot is None or not oot_has_labels
        else oot[config.target_col].to_numpy()
    )

    train_m = compute_multiclass_metrics(train_proba, train_target, classes)
    test_m = compute_multiclass_metrics(test_proba, test_target, classes)
    oot_m = (
        None
        if oot_proba is None or oot_target is None
        else compute_multiclass_metrics(oot_proba, oot_target, classes)
    )

    metrics = ModelMetrics(
        train_ks=None,
        test_ks=None,
        oot_ks=None,
        train_auc=None,
        test_auc=None,
        oot_auc=None,
        psi_test_vs_train=None,
        psi_oot_vs_train=None,
        overfit_train_test_gap=_gap(train_m["macro_auc"], test_m["macro_auc"]),
        overfit_train_oot_gap=None if oot_m is None else _gap(train_m["macro_auc"], oot_m["macro_auc"]),
        overfit_flag=False,
        train_macro_auc=train_m["macro_auc"],
        test_macro_auc=test_m["macro_auc"],
        oot_macro_auc=None if oot_m is None else oot_m["macro_auc"],
        train_logloss=train_m["logloss"],
        test_logloss=test_m["logloss"],
        oot_logloss=None if oot_m is None else oot_m["logloss"],
        train_accuracy=train_m["accuracy"],
        test_accuracy=test_m["accuracy"],
        oot_accuracy=None if oot_m is None else oot_m["accuracy"],
    )
    per_class = {
        "train": train_m["per_class"],
        "test": test_m["per_class"],
        "oot": None if oot_m is None else oot_m["per_class"],
    }
    return metrics, per_class


def compute_multiclass_metrics(proba_2d, y_true, classes) -> dict:
    """Deterministic, JSON-safe multiclass metrics for an N×K probability matrix.

    ``classes`` is the ordered class list matching the probability columns. Returns a
    dict with macro_auc (one-vs-rest macro), weighted_auc, logloss, accuracy,
    macro_recall, and a per-class {recall, precision, support} map. Any value that is
    undefined or non-finite (single observed class, degenerate input) becomes None so
    the payload is strict JSON (no NaN/inf)."""
    from sklearn.metrics import accuracy_score, log_loss

    classes = tuple(classes)
    proba = np.asarray(proba_2d, dtype=float)
    labels = np.asarray(list(y_true))
    n_rows = labels.shape[0]
    if proba.ndim != 2 or proba.shape[0] != n_rows or proba.shape[1] != len(classes):
        raise ModelingError(
            f"multiclass proba shape {proba.shape} does not match rows {n_rows} / classes {len(classes)}"
        )

    class_array = np.asarray(classes)
    pred_idx = np.argmax(proba, axis=1) if n_rows else np.empty(0, dtype=int)
    # Index the class array (not a Python list comprehension) so predictions keep the
    # class dtype and sklearn sees a consistent target type with the labels.
    predictions = class_array[pred_idx] if n_rows else class_array[:0]
    accuracy = _safe_float(accuracy_score(labels, predictions)) if n_rows else None

    macro_auc = _macro_auc(proba, labels, classes, average="macro")
    weighted_auc = _macro_auc(proba, labels, classes, average="weighted")

    try:
        logloss = _safe_float(log_loss(labels, proba, labels=list(classes))) if n_rows else None
    except ValueError:
        logloss = None

    per_class = _per_class_metrics(labels, predictions, classes)
    recalls = [row["recall"] for row in per_class.values() if row["recall"] is not None]
    macro_recall = _safe_float(float(np.mean(recalls))) if recalls else None

    return {
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "logloss": logloss,
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "per_class": per_class,
    }


def _macro_auc(proba: np.ndarray, labels: np.ndarray, classes: tuple, *, average: str) -> float | None:
    """One-vs-rest AUC, averaged. Returns None when fewer than two classes are present
    in the labels (AUC is undefined) or sklearn rejects the input."""
    from sklearn.metrics import roc_auc_score

    if labels.shape[0] == 0:
        return None
    present = set(labels.tolist())
    if len(present) < 2:
        return None
    try:
        value = roc_auc_score(
            labels,
            proba,
            multi_class="ovr",
            average=average,
            labels=list(classes),
        )
    except ValueError:
        return None
    return _safe_float(value)


def _per_class_metrics(labels: np.ndarray, predictions: np.ndarray, classes: tuple) -> dict:
    from sklearn.metrics import precision_score, recall_score

    per_class: dict = {}
    has_rows = labels.shape[0] > 0
    for cls in classes:
        support = int(np.sum(labels == cls)) if has_rows else 0
        if has_rows:
            recall = _safe_float(
                recall_score(labels, predictions, labels=[cls], average="micro", zero_division=0)
            ) if support else None
            precision = _safe_float(
                precision_score(labels, predictions, labels=[cls], average="micro", zero_division=0)
            ) if int(np.sum(predictions == cls)) else None
        else:
            recall = None
            precision = None
        per_class[_class_key(cls)] = {
            "recall": recall,
            "precision": precision,
            "support": support,
        }
    return per_class


def _class_key(value) -> str:
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        return str(float(value))
    return str(value)


def _gap(train_value: float | None, other_value: float | None) -> float:
    if train_value is None or other_value is None:
        return 0.0
    return float(train_value - other_value)


def _safe_float(value) -> float | None:
    """Map a metric to a finite float or None (NaN/inf → None) for strict JSON."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _proba_2d(
    proba_fn: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    classes: tuple,
) -> np.ndarray:
    proba = np.asarray(proba_fn(frame), dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(frame) or proba.shape[1] != len(classes):
        raise ModelingError(
            f"proba shape {proba.shape} does not match rows {len(frame)} / classes {len(classes)}"
        )
    return proba


def _scores(score_fn: Callable[[pd.DataFrame], np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(score_fn(frame), dtype=float)
    if scores.shape[0] != len(frame):
        raise ModelingError(f"score length {scores.shape[0]} does not match rows {len(frame)}")
    return scores


def _regression_values(target: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    residual = pred - target
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    total = float(np.sum(np.square(target - np.mean(target))))
    if total <= np.finfo(float).eps:
        r2 = 0.0
    else:
        r2 = float(1 - (np.sum(np.square(residual)) / total))
    return rmse, mae, r2


__all__ = [
    "artifact_params",
    "compute_model_metrics",
    "compute_multiclass_metrics",
    "compute_multiclass_model_metrics",
    "compute_regression_metrics",
    "model_params",
    "normalized_monotone_constraints",
    "sample_weight_col",
    "sample_weight_values",
    "split_modeling_frame",
]

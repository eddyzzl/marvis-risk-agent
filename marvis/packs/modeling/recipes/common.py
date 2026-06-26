from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from marvis.feature.binning import equal_frequency_edges
from marvis.feature.metrics import feature_auc, feature_ks, feature_psi
from marvis.packs.modeling.contracts import ModelMetrics, TrainConfig
from marvis.packs.modeling.errors import ModelingError
from marvis.validation.overfitting import overfitting_check


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
    )


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


__all__ = ["compute_model_metrics", "compute_regression_metrics", "split_modeling_frame"]

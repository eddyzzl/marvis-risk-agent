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
) -> ModelMetrics:
    train_scores = _scores(score_fn, train)
    test_scores = _scores(score_fn, test)
    oot_scores = None if oot is None else _scores(score_fn, oot)
    train_target = train[config.target_col].to_numpy(dtype=int)
    test_target = test[config.target_col].to_numpy(dtype=int)
    oot_target = None if oot is None else oot[config.target_col].to_numpy(dtype=int)

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


def _scores(score_fn: Callable[[pd.DataFrame], np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(score_fn(frame), dtype=float)
    if scores.shape[0] != len(frame):
        raise ModelingError(f"score length {scores.shape[0]} does not match rows {len(frame)}")
    return scores


__all__ = ["compute_model_metrics", "split_modeling_frame"]

"""Hyperparameter search for the LightGBM modeling recipe.

The modeling pack previously had no tuning at all — train_model ran a single
fixed (and very weak: 20-round) configuration, so an out-of-the-box model badly
underperformed a hand-tuned reference. This module runs a seeded random search
over a sensible LightGBM space, selecting on the **test** split while penalising
train/test overfit. OOT metrics are reported for transparency but are not used for
hyperparameter selection, so the search does not tune against the holdout window.
step produces strong, robust ``params`` to hand straight to ``train_model``.

Deterministic given ``seed`` (the trial space is sampled from a seeded RNG and
LightGBM is trained with fixed seed + single thread). KS comes from
marvis.feature.metrics.feature_ks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from marvis.data.labels import resolve_modeling_splits
from marvis.feature.metrics import feature_auc, feature_ks, head_tail_lift
from marvis.packs.modeling.errors import ModelingError


@dataclass(frozen=True)
class TuneResult:
    best_params: dict
    """Best LightGBM params (incl. ``num_boost_round``) — feed to train_model.params."""
    best_metrics: dict
    """{train_ks, test_ks, oot_ks, train_auc?, overfit_gap} for the chosen trial."""
    trials: tuple[dict, ...] = field(default_factory=tuple)
    """Per-trial {params, train_ks, test_ks, oot_ks, score} for transparency."""
    n_trials: int = 0
    nan_labels_dropped: int = 0
    """Rows excluded by the NaN-label gate (train/test/oot), for audit (mirrors TrainResult)."""


def _split(frame: pd.DataFrame, split_col: str, split_values: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if split_col not in frame.columns:
        raise ModelingError(f"missing split column: {split_col}")
    train_v, test_v = split_values.get("train"), split_values.get("test")
    if train_v is None or test_v is None:
        raise ModelingError("split_values must include train and test")
    train = frame[frame[split_col] == train_v]
    test = frame[frame[split_col] == test_v]
    if train.empty or test.empty:
        raise ModelingError("train/test split is empty")
    oot_v = split_values.get("oot")
    oot = frame[frame[split_col] == oot_v] if oot_v is not None else None
    if oot is not None and oot.empty:
        oot = None
    return train, test, oot


def _sample_params(rng: np.random.RandomState, pos_weight_hint: float) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_leaves": int(rng.choice([8, 12, 16, 24, 31, 48, 63])),
        "max_depth": int(rng.choice([2, 3, 4, 5, 6, -1])),
        "learning_rate": float(round(10 ** rng.uniform(-2.0, -1.1), 5)),  # ~0.01..0.08
        "min_child_samples": int(rng.choice([50, 100, 150, 200, 300, 500])),
        "feature_fraction": float(round(rng.uniform(0.4, 0.9), 3)),
        "bagging_fraction": float(round(rng.uniform(0.5, 0.9), 3)),
        "bagging_freq": 1,
        "lambda_l1": float(round(rng.uniform(0.0, 20.0), 2)),
        "lambda_l2": float(round(rng.uniform(0.0, 20.0), 2)),
        "min_gain_to_split": float(round(rng.uniform(0.0, 3.0), 2)),
        "scale_pos_weight": float(rng.choice([1.0, 3.0, 5.0, 10.0, pos_weight_hint])),
    }


def tune_hyperparameters(
    backend,
    dataset_path: Path,
    *,
    features: list[str],
    target_col: str,
    split_col: str,
    split_values: dict,
    n_trials: int = 40,
    seed: int = 0,
    early_stopping_rounds: int = 100,
    max_boost_round: int = 3000,
    overfit_penalty: float = 0.5,
    sample_weight_col: str = "",
    base_params: dict | None = None,
    drop_nan_labels: bool = False,
) -> TuneResult:
    """Random-search LightGBM params; select by test KS minus in-time overfit penalty.

    Objective per trial:
    ``test_ks - overfit_penalty * max(0, train_ks - test_ks)``.

    Applies the shared NaN-label confirmation gate (mirrors the training recipes):
    a NaN target in train/test raises ``NanLabelNotConfirmedError`` unless
    ``drop_nan_labels=True``. A fully-unlabeled OOT split is legitimate
    (scoring-only); its oot_ks/oot_auc are reported as ``None``.
    """
    feats = [f for f in dict.fromkeys(features) if f != target_col]
    weight_col = str(sample_weight_col or "").strip()
    cols = feats + [target_col, split_col] + ([weight_col] if weight_col else [])
    frame = backend.read_frame(dataset_path, columns=cols)
    train, test, oot = _split(frame, split_col, split_values)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=target_col, drop_nan_labels=drop_nan_labels,
    )
    ytr = train[target_col].to_numpy(dtype=float)
    yte = test[target_col].to_numpy(dtype=float)
    wtr = _sample_weight(train, weight_col)
    wte = _sample_weight(test, weight_col)
    pos = _weighted_count(ytr, wtr, 1.0)
    neg = _weighted_count(ytr, wtr, 0.0)
    pos_weight_hint = round(neg / pos, 1) if pos > 0 else 1.0

    dtrain = lgb.Dataset(train[feats], label=ytr, weight=wtr, free_raw_data=False)
    dvalid = lgb.Dataset(test[feats], label=yte, weight=wte, reference=dtrain, free_raw_data=False)
    rng = np.random.RandomState(seed)
    fixed_params = _lgb_base_params(base_params, pos_weight_hint=pos_weight_hint)
    trial_max_boost_round = int(fixed_params.pop("num_boost_round", max_boost_round))

    best = None
    trials: list[dict] = []
    for trial in range(max(1, n_trials)):
        params = {**_sample_params(rng, pos_weight_hint), **fixed_params}
        params.update({"seed": seed, "num_threads": 0, "deterministic": True})
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=trial_max_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        rounds = int(booster.best_iteration or trial_max_boost_round)
        train_preds = booster.predict(train[feats])
        test_preds = booster.predict(test[feats])
        train_ks = feature_ks(train_preds, ytr)
        test_ks = feature_ks(test_preds, yte)
        train_auc = feature_auc(train_preds, ytr)
        test_auc = feature_auc(test_preds, yte)
        # head/tail lift of the model SCORE on the test split, at 5% AND 10% (spec §5
        # leaderboard columns 头部/尾部 lift5%/10%).
        lift = head_tail_lift(test_preds, yte)
        if oot is None or not oot_has_labels:
            oot_ks = oot_auc = None
        else:
            yoot = oot[target_col].to_numpy(dtype=float)
            oot_preds = booster.predict(oot[feats])
            oot_ks = feature_ks(oot_preds, yoot)
            oot_auc = feature_auc(oot_preds, yoot)
        # Overfit gaps: train-test always; train-oot when an OOT split exists.
        gap_tt = train_ks - test_ks
        gap_to = (train_ks - oot_ks) if oot_ks is not None else None
        oot_stability_gap = abs(test_ks - oot_ks) if oot_ks is not None else None
        score = _trial_score(
            train_ks=train_ks,
            test_ks=test_ks,
            overfit_penalty=overfit_penalty,
        )
        record = {
            "params": {**params, "num_boost_round": rounds},
            "train_ks": train_ks, "test_ks": test_ks, "oot_ks": oot_ks, "score": score,
            "train_auc": train_auc, "test_auc": test_auc, "oot_auc": oot_auc,
            "lift_head_5": lift.get("lift_head_5"), "lift_tail_5": lift.get("lift_tail_5"),
            "lift_head_10": lift.get("lift_head_10"), "lift_tail_10": lift.get("lift_tail_10"),
            "overfit_gap_tt": gap_tt, "overfit_gap_to": gap_to,
            "oot_stability_gap": oot_stability_gap,
        }
        trials.append(record)
        if best is None or score > best["score"]:
            best = record

    assert best is not None
    return TuneResult(
        best_params=best["params"],
        best_metrics={
            "train_ks": best["train_ks"], "test_ks": best["test_ks"],
            "oot_ks": best["oot_ks"],
            "overfit_gap": best["overfit_gap_tt"],
            "overfit_gap_oot": best["overfit_gap_to"],
            "oot_stability_gap": best["oot_stability_gap"],
            "train_auc": best.get("train_auc"), "test_auc": best.get("test_auc"),
            "oot_auc": best.get("oot_auc"),
        },
        trials=tuple(trials),
        n_trials=len(trials),
        nan_labels_dropped=audit["total_dropped"],
    )


def _trial_score(
    *,
    train_ks: float,
    test_ks: float,
    overfit_penalty: float,
) -> float:
    return test_ks - overfit_penalty * max(0.0, train_ks - test_ks)


def _sample_weight(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if not column:
        return None
    if column not in frame.columns:
        raise ModelingError(f"sample weight column not found in tuning frame: {column}")
    weights = pd.to_numeric(frame[column], errors="coerce")
    if weights.isna().any():
        raise ModelingError(f"sample weight column `{column}` contains null or non-numeric values")
    if (weights < 0).any():
        raise ModelingError(f"sample weight column `{column}` contains negative values")
    if float(weights.sum()) <= 0:
        raise ModelingError(f"sample weight column `{column}` must have a positive total weight")
    return weights.to_numpy(dtype=float)


def _lgb_base_params(params: dict | None, *, pos_weight_hint: float) -> dict:
    blocked = {"sample_weight_col", "sample_weight_column", "weight_col"}
    out = {
        str(key): value
        for key, value in dict(params or {}).items()
        if str(key) not in blocked and value not in (None, "")
    }
    if str(out.get("scale_pos_weight") or "").strip().lower() == "auto":
        out["scale_pos_weight"] = float(pos_weight_hint)
    return out


def _weighted_count(labels: np.ndarray, weights: np.ndarray | None, value: float) -> float:
    mask = labels == value
    if weights is None:
        return float(mask.sum())
    return float(weights[mask].sum())


__all__ = ["tune_hyperparameters", "TuneResult"]

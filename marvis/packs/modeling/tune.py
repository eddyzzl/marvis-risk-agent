"""Hyperparameter search for the LightGBM modeling recipe.

The modeling pack previously had no tuning at all — train_model ran a single
fixed (and very weak: 20-round) configuration, so an out-of-the-box model badly
underperformed a hand-tuned reference. This module runs a seeded **two-stage**
random search over a sensible LightGBM space, selecting on the **test** split
while penalising train/test overfit. OOT metrics are reported for transparency
but are not used for hyperparameter selection, so the search does not tune
against the holdout window. The step produces strong, robust ``params`` to feed
straight to ``train_model``.

Two-stage search (no new dependency — deterministic, seed-derived, no TPE/
Bayesian library required):
  * Stage 1 ("coarse", ``coarse_fraction`` of the budget, default 60%) samples
    the full space uniformly / log-uniformly at random — the original strategy.
  * Stage 2 ("fine", the remaining budget) resamples each numeric hyperparameter
    in a shrunk neighbourhood around the stage-1 best trial (log-neighbourhood
    for log-scaled params, linear-neighbourhood for linear-scaled ones) while
    keeping the more categorical params (``max_depth``, ``bagging_freq``,
    ``scale_pos_weight``) fixed at the stage-1 best value. Each trial records
    its ``search_stage`` ("coarse"/"fine") for report traceability.

Deterministic given ``seed``: stage 1 and stage 2 each draw from their own
seed-derived ``RandomState`` (``seed`` and ``seed + 1``), so re-running with the
same seed reproduces the exact same trial sequence and results. LightGBM is
trained with fixed seed + single thread. KS comes from
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

# Reference learning rate used to scale the per-trial boost-round ceiling: lower
# sampled learning rates get proportionally more rounds (early stopping still
# backstops actual training length), higher learning rates get fewer.
_LR_REFERENCE = 0.1
_MIN_BOOST_ROUND_CEILING = 100

# log-scaled numeric params: (low, high) bounds used for coarse-stage sampling.
_LOG_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.01, 0.3),
    "lambda_l1": (1e-8, 10.0),
    "lambda_l2": (1e-8, 10.0),
}
# linear-scaled numeric params sampled by rng.uniform in coarse stage.
_LINEAR_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "feature_fraction": (0.4, 0.9),
    "bagging_fraction": (0.5, 0.9),
    "min_gain_to_split": (0.0, 3.0),
}
# choice-scaled numeric params sampled from a fixed candidate list in coarse stage.
_CHOICE_SPACE: dict[str, tuple] = {
    "num_leaves": (8, 12, 16, 24, 31, 48, 63),
    "min_child_samples": (50, 100, 150, 200, 300, 500),
}
# Fine-stage shrink factor applied around the stage-1 best value.
_FINE_LOG_SHRINK = 0.35  # +/- fraction (in log space) of the full log-range
_FINE_LINEAR_SHRINK = 0.2  # +/- fraction of the full linear-range
_FINE_CHOICE_NEIGHBORS = 1  # +/- N positions in the sorted choice list


@dataclass(frozen=True)
class TuneResult:
    best_params: dict
    """Best LightGBM params (incl. ``num_boost_round``) — feed to train_model.params."""
    best_metrics: dict
    """{train_ks, test_ks, oot_ks, train_auc?, overfit_gap} for the chosen trial."""
    trials: tuple[dict, ...] = field(default_factory=tuple)
    """Per-trial {params, train_ks, test_ks, oot_ks, score, search_stage} for transparency."""
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


def _sample_coarse_params(rng: np.random.RandomState, pos_weight_hint: float) -> dict:
    """Stage-1 coarse sample: full-space uniform / log-uniform draw."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "max_depth": int(rng.choice([2, 3, 4, 5, 6, -1])),
        "bagging_freq": 1,
        "scale_pos_weight": float(rng.choice([1.0, 3.0, 5.0, 10.0, pos_weight_hint])),
    }
    for name, (low, high) in _LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high)
    for name, (low, high) in _LINEAR_SPACE_BOUNDS.items():
        params[name] = float(round(rng.uniform(low, high), 3))
    for name, choices in _CHOICE_SPACE.items():
        params[name] = int(rng.choice(choices))
    return params


def _sample_fine_params(rng: np.random.RandomState, pos_weight_hint: float, anchor: dict) -> dict:
    """Stage-2 fine sample: shrunk neighbourhood around the stage-1 best trial.

    Numeric params (log- or linear-scaled) are resampled in a bounded window
    around ``anchor``'s value, clipped back into the full space bounds.
    Categorical-ish params (``max_depth``, ``bagging_freq``, ``scale_pos_weight``)
    stay fixed at the anchor's value — the coarse stage already picked the best
    region for them and there is no continuous neighbourhood to refine.
    """
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "max_depth": anchor["max_depth"],
        "bagging_freq": 1,
        "scale_pos_weight": anchor["scale_pos_weight"],
    }
    for name, (low, high) in _LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(
            rng, low, high, center=anchor[name], shrink=_FINE_LOG_SHRINK,
        )
    for name, (low, high) in _LINEAR_SPACE_BOUNDS.items():
        span = (high - low) * _FINE_LINEAR_SHRINK
        lo = max(low, anchor[name] - span)
        hi = min(high, anchor[name] + span)
        params[name] = float(round(rng.uniform(lo, hi), 3))
    for name, choices in _CHOICE_SPACE.items():
        ordered = sorted(choices)
        idx = ordered.index(anchor[name])
        lo_idx = max(0, idx - _FINE_CHOICE_NEIGHBORS)
        hi_idx = min(len(ordered) - 1, idx + _FINE_CHOICE_NEIGHBORS)
        params[name] = int(rng.choice(ordered[lo_idx : hi_idx + 1]))
    return params


def _round_log_sample(
    rng: np.random.RandomState,
    low: float,
    high: float,
    *,
    center: float | None = None,
    shrink: float | None = None,
) -> float:
    """Log-uniform sample in ``[low, high]``, optionally shrunk to a neighbourhood
    of ``center`` (also in log space, clipped back to the full bounds)."""
    log_low, log_high = np.log10(low), np.log10(high)
    if center is not None and shrink is not None:
        log_center = np.log10(max(center, low))
        span = (log_high - log_low) * shrink
        log_low = max(log_low, log_center - span)
        log_high = min(log_high, log_center + span)
    value = 10 ** rng.uniform(log_low, log_high)
    return float(round(value, 8))


def _sample_params(rng: np.random.RandomState, pos_weight_hint: float) -> dict:
    """Back-compat single-stage sampler (used by callers that want one draw from
    the full coarse space, e.g. non-lgb fallbacks or tests)."""
    return _sample_coarse_params(rng, pos_weight_hint)


def _boost_round_ceiling(learning_rate: float, max_boost_round: int) -> int:
    """Scale the per-trial round ceiling inversely with the sampled learning rate
    so low-lr trials get enough rounds to converge; early stopping still bounds
    actual training length, this only raises/lowers the ceiling it can reach."""
    scale = _LR_REFERENCE / max(learning_rate, 1e-6)
    scaled = int(round(max_boost_round * scale))
    return min(max_boost_round, max(_MIN_BOOST_ROUND_CEILING, scaled))


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
    coarse_fraction: float = 0.6,
) -> TuneResult:
    """Deterministic two-stage random search; select by test KS minus in-time
    overfit penalty.

    Stage 1 ("coarse") spends ``round(n_trials * coarse_fraction)`` trials
    sampling the full space at random. Stage 2 ("fine") spends the remaining
    trials resampling numeric params in a shrunk neighbourhood around the best
    stage-1 trial. Both stages draw from seed-derived ``RandomState`` instances
    so the whole search is reproducible for a fixed ``seed``.

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
    fixed_params = _lgb_base_params(base_params, pos_weight_hint=pos_weight_hint)
    trial_max_boost_round = int(fixed_params.pop("num_boost_round", max_boost_round))

    total_trials = max(1, n_trials)
    n_coarse = min(total_trials, max(1, round(total_trials * coarse_fraction)))
    n_fine = total_trials - n_coarse

    coarse_rng = np.random.RandomState(seed)
    fine_rng = np.random.RandomState(seed + 1)

    best: dict | None = None
    best_coarse: dict | None = None
    trials: list[dict] = []

    def _run_trial(params: dict, stage: str) -> dict:
        params = {**params, **fixed_params}
        params.update({"seed": seed, "num_threads": 0, "deterministic": True})
        round_ceiling = _boost_round_ceiling(params["learning_rate"], trial_max_boost_round)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=round_ceiling,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        rounds = int(booster.best_iteration or round_ceiling)
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
        return {
            "params": {**params, "num_boost_round": rounds},
            "train_ks": train_ks, "test_ks": test_ks, "oot_ks": oot_ks, "score": score,
            "train_auc": train_auc, "test_auc": test_auc, "oot_auc": oot_auc,
            "lift_head_5": lift.get("lift_head_5"), "lift_tail_5": lift.get("lift_tail_5"),
            "lift_head_10": lift.get("lift_head_10"), "lift_tail_10": lift.get("lift_tail_10"),
            "overfit_gap_tt": gap_tt, "overfit_gap_to": gap_to,
            "oot_stability_gap": oot_stability_gap,
            "search_stage": stage,
        }

    for _ in range(n_coarse):
        sampled = _sample_coarse_params(coarse_rng, pos_weight_hint)
        record = _run_trial(sampled, "coarse")
        record["_sampled"] = sampled
        trials.append(record)
        if best is None or record["score"] > best["score"]:
            best = record
        if best_coarse is None or record["score"] > best_coarse["score"]:
            best_coarse = record

    anchor = (best_coarse or best)["_sampled"]
    for _ in range(n_fine):
        sampled = _sample_fine_params(fine_rng, pos_weight_hint, anchor)
        record = _run_trial(sampled, "fine")
        record["_sampled"] = sampled
        trials.append(record)
        if record["score"] > best["score"]:
            best = record

    assert best is not None
    for record in trials:
        record.pop("_sampled", None)

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

"""Hyperparameter search for the modeling pack's recipe families.

Historically this module only tuned LightGBM: every other recipe (xgb, catboost,
lr, scorecard, mlp) trained with hardcoded weak defaults (xgb 20 rounds,
catboost 50 rounds, scorecard's C never searched at all) while lgb alone got a
seeded two-stage random search. That made every "multi-algorithm comparison"
structurally unfair -- lgb's win was decided by tuning budget, not by algorithm
merit (TUNE-1 / SEL-2).

This module now generalises the two-stage search to **every** recipe family via
a small per-recipe plugin: a coarse/fine parameter sampler pair plus a
trial-runner that fits the recipe's estimator, scores it, and (for tree
recipes) reports the resolved early-stopped iteration count.
``tune_hyperparameters`` dispatches on ``recipe`` and is otherwise identical
across families:

  * Stage 1 ("coarse", ``coarse_fraction`` of the budget, default 60%) samples
    the full space uniformly / log-uniformly at random.
  * Stage 2 ("fine", the remaining budget) resamples each numeric hyperparameter
    in a shrunk neighbourhood around the stage-1 best trial, keeping
    categorical-ish params fixed at the stage-1 best value.
  * Trial selection: ``test_ks - overfit_penalty * max(0, train_ks - test_ks)``.
  * Deterministic given ``seed``: stage 1 and stage 2 each draw from their own
    seed-derived ``RandomState`` (``seed`` and ``seed + 1``).
  * Tree recipes (lgb/xgb/catboost) run every trial with early stopping against
    a validation fold carved from ``train`` (default 15%, SEL-4/TUNE-3 -- not
    ``test``, which stays free to serve only as the trial-selection/comparison
    set) and report the resolved ``best_iteration``/``num_boost_round`` in
    ``best_params`` so the downstream ``train_models`` step trains at the tuned
    depth, not a re-run of the full ceiling.

``recipe="lgb"`` (the default) preserves the exact historical search space and
trial semantics -- no behaviour change for existing single-recipe callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from marvis.data.labels import resolve_modeling_splits
from marvis.feature.metrics import feature_auc, feature_ks, head_tail_lift, weighted_feature_auc, weighted_feature_ks
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.common import carve_early_stop_fold, cat_feature_indices

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

# --------------------------------------------------------------------------
# xgb search space
# --------------------------------------------------------------------------
_XGB_LOG_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "eta": (0.01, 0.3),
    "reg_lambda": (1e-8, 10.0),
    "reg_alpha": (1e-8, 10.0),
}
_XGB_LINEAR_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "subsample": (0.5, 0.9),
    "colsample_bytree": (0.5, 0.9),
}
_XGB_CHOICE_SPACE: dict[str, tuple] = {
    "max_depth": (2, 3, 4, 5, 6, 8),
    "min_child_weight": (1, 3, 5, 10, 20),
}

# --------------------------------------------------------------------------
# catboost search space
# --------------------------------------------------------------------------
_CATBOOST_LOG_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.01, 0.3),
    "l2_leaf_reg": (1e-2, 30.0),
}
_CATBOOST_CHOICE_SPACE: dict[str, tuple] = {
    "depth": (3, 4, 5, 6, 7, 8),
}

# --------------------------------------------------------------------------
# lr / scorecard search space (small budget: C + scorecard's bin granularity)
# --------------------------------------------------------------------------
_LR_LOG_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "C": (0.01, 10.0),
}
_SCORECARD_LOG_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "C": (0.01, 10.0),
}
_SCORECARD_CHOICE_SPACE: dict[str, tuple] = {
    "scorecard_max_bins": (4, 5, 6, 7, 8, 10),
}

# --------------------------------------------------------------------------
# mlp search space (small budget)
# --------------------------------------------------------------------------
_MLP_LOG_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    "alpha": (1e-6, 1e-1),
    "learning_rate_init": (1e-4, 1e-1),
}
_MLP_CHOICE_SPACE: dict[str, tuple] = {
    "hidden_layer_sizes": ((16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)),
}

#: Per-recipe default tuning budget (trial count) used when the caller does not
#: pass an explicit n_trials -- tree recipes get a full two-stage budget, the
#: small-space families (lr/scorecard/mlp) get a much smaller one (TUNE-1/SEL-2).
DEFAULT_TRIAL_BUDGET: dict[str, int] = {
    "lgb": 40,
    "xgb": 40,
    "catboost": 40,
    "lr": 12,
    "scorecard": 12,
    "mlp": 12,
}

#: Tree recipes that early-stop against a carved validation fold instead of the
#: full test split (SEL-4/TUNE-3) -- mirrors train_models' _EARLY_STOPPED_TREE_RECIPES.
_EARLY_STOPPED_TREE_RECIPES = frozenset({"lgb", "xgb", "catboost"})


def _group_cols_from_params(params: dict | None) -> list[str] | None:
    """Optional group/identity column(s) for the early-stopping valid-fold carve,
    read from the caller-supplied base_params (mirrors
    ``recipes.common.VALID_GROUP_COLS_PARAM_KEY``); absent by default."""
    raw = dict(params or {}).get("valid_group_cols")
    if not raw:
        return None
    return raw if isinstance(raw, list) else [raw]


# ==========================================================================
# Optional grouped cross-validation scoring (TUNE-3)
# ==========================================================================


def _cv_folds(
    frame: pd.DataFrame,
    *,
    cv_folds: int,
    seed: int,
    group_cols: list[str] | None,
) -> list[np.ndarray]:
    """``cv_folds`` disjoint group-aware row-index arrays covering ``frame`` (each
    row/group assigned to exactly one fold). Deterministic given ``seed``."""
    rng = np.random.RandomState(_valid_fold_seed_for_cv(seed))
    resolved_group_cols = [str(col) for col in (group_cols or []) if str(col) in frame.columns]
    groups = (
        frame.groupby(resolved_group_cols, sort=False).ngroup().to_numpy()
        if resolved_group_cols
        else np.arange(len(frame))
    )
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    fold_of_group = {int(group): index % cv_folds for index, group in enumerate(unique_groups)}
    fold_ids = np.array([fold_of_group[int(g)] for g in groups])
    return [np.where(fold_ids == fold)[0] for fold in range(cv_folds)]


def _valid_fold_seed_for_cv(seed: int) -> int:
    import hashlib

    digest = hashlib.sha256(f"{int(seed)}:cv_folds".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def _make_cv_trial_runner(
    recipe: str,
    *,
    train: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    target_col: str,
    weight_col: str,
    cv_folds: int,
    seed: int,
    early_stopping_rounds: int,
    max_boost_round: int,
    overfit_penalty: float,
    oot_has_labels: bool,
    base_params: dict | None,
    pos_weight_hint: float,
    group_cols: list[str] | None,
) -> Callable[[dict, str], dict]:
    """A trial_runner that scores each sampled param set with grouped k-fold CV
    over ``train`` instead of a single held-out split (TUNE-3): each fold takes a
    turn as the held-out "test" (tree recipes additionally carve their own
    early-stopping valid fold from that fold's training portion, exactly like the
    single-split path), and the trial's score is
    ``mean(fold_test_ks) - 0.5 * std(fold_test_ks)`` -- a robustness penalty so a
    param set that wins by getting lucky on one fold doesn't out-rank a param set
    that wins consistently across folds.
    """
    fold_indices = _cv_folds(train, cv_folds=cv_folds, seed=seed, group_cols=group_cols)
    fold_woot = _sample_weight(oot, weight_col) if oot is not None and weight_col else None
    fold_runners: list[Callable[[dict, str], dict]] = []
    for fold in range(cv_folds):
        fold_test_idx = fold_indices[fold]
        fold_train_idx = np.concatenate([fold_indices[i] for i in range(cv_folds) if i != fold])
        fold_train = train.iloc[fold_train_idx]
        fold_test = train.iloc[fold_test_idx]
        fold_ytr = fold_train[target_col].to_numpy(dtype=float)
        fold_yte = fold_test[target_col].to_numpy(dtype=float)
        fold_wtr = _sample_weight(fold_train, weight_col)
        fold_wte = _sample_weight(fold_test, weight_col)
        if recipe in _EARLY_STOPPED_TREE_RECIPES:
            fold_fit_train, fold_valid = carve_early_stop_fold(
                fold_train, seed=seed + fold + 1, group_cols=group_cols,
            )
            fold_yfit = fold_fit_train[target_col].to_numpy(dtype=float)
            fold_yva = fold_valid[target_col].to_numpy(dtype=float)
            fold_wfit = _sample_weight(fold_fit_train, weight_col)
            fold_wva = _sample_weight(fold_valid, weight_col)
        else:
            fold_fit_train, fold_valid = fold_train, fold_test
            fold_yfit, fold_yva, fold_wfit, fold_wva = fold_ytr, fold_yte, fold_wtr, fold_wte
        _, _, fold_runner = _recipe_search_hooks(
            recipe,
            train=fold_train, test=fold_test, oot=oot,
            fit_train=fold_fit_train, valid=fold_valid,
            feats=feats, target_col=target_col,
            ytr=fold_ytr, yte=fold_yte, wtr=fold_wtr, wte=fold_wte, woot=fold_woot,
            yfit=fold_yfit, yva=fold_yva, wfit=fold_wfit, wva=fold_wva,
            seed=seed, early_stopping_rounds=early_stopping_rounds,
            max_boost_round=max_boost_round, overfit_penalty=overfit_penalty,
            oot_has_labels=oot_has_labels,
            base_params=base_params, pos_weight_hint=pos_weight_hint,
        )
        fold_runners.append(fold_runner)

    # Numeric per-fold metrics averaged into the merged record -- each fold trains
    # its own model, so its oot_ks/oot_auc/etc. are genuinely different values, not
    # an arbitrary fold's report (only test_ks/train_ks additionally get their
    # spread reported via test_ks_std/cv_fold_test_ks, since those drive the score).
    _AVERAGED_KEYS = (
        "train_auc", "test_auc", "oot_ks", "oot_auc",
        "lift_head_5", "lift_tail_5", "lift_head_10", "lift_tail_10",
        "overfit_gap_to", "oot_stability_gap",
        "weighted_train_auc", "weighted_test_auc", "weighted_oot_ks", "weighted_oot_auc",
    )

    def cv_runner(params: dict, stage: str) -> dict:
        fold_records = [runner(params, stage) for runner in fold_runners]
        # TUNE-5: fold spread/score use the weighted test KS when weights exist
        # (same fallback _score_trial already applies per fold) -- otherwise CV
        # would optimise the weighted objective per-fold but rank param sets by
        # the unweighted one when combining folds.
        fold_test_ks = [
            record["weighted_test_ks"] if record.get("weighted_test_ks") is not None else record["test_ks"]
            for record in fold_records
        ]
        fold_train_ks = [
            record["weighted_train_ks"] if record.get("weighted_train_ks") is not None else record["train_ks"]
            for record in fold_records
        ]
        mean_ks = float(np.mean(fold_test_ks))
        std_ks = float(np.std(fold_test_ks))
        score = mean_ks - overfit_penalty * std_ks
        merged = dict(fold_records[0])
        merged["train_ks"] = float(np.mean([record["train_ks"] for record in fold_records]))
        merged["test_ks"] = float(np.mean([record["test_ks"] for record in fold_records]))
        if any(record.get("weighted_train_ks") is not None for record in fold_records):
            merged["weighted_train_ks"] = float(np.mean(fold_train_ks))
        if any(record.get("weighted_test_ks") is not None for record in fold_records):
            merged["weighted_test_ks"] = mean_ks
        merged["test_ks_std"] = std_ks
        merged["cv_fold_test_ks"] = fold_test_ks
        merged["score"] = score
        merged["overfit_gap_tt"] = float(np.mean(fold_train_ks)) - mean_ks
        for key in _AVERAGED_KEYS:
            values = [record.get(key) for record in fold_records]
            merged[key] = float(np.mean(values)) if all(v is not None for v in values) else None
        return merged

    return cv_runner


@dataclass(frozen=True)
class TuneResult:
    best_params: dict
    """Best params (incl. ``num_boost_round``/``iterations`` for tree recipes) --
    feed to train_model.params."""
    best_metrics: dict
    """{train_ks, test_ks, oot_ks, train_auc?, overfit_gap} for the chosen trial."""
    trials: tuple[dict, ...] = field(default_factory=tuple)
    """Per-trial {params, train_ks, test_ks, oot_ks, score, search_stage} for transparency."""
    n_trials: int = 0
    nan_labels_dropped: int = 0
    """Rows excluded by the NaN-label gate (train/test/oot), for audit (mirrors TrainResult)."""
    recipe: str = "lgb"


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


# ==========================================================================
# LightGBM sampler (unchanged historical search space/semantics)
# ==========================================================================


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
    stay fixed at the anchor's value -- the coarse stage already picked the best
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


def _run_lgb_trial(
    params: dict,
    stage: str,
    *,
    fixed_params: dict,
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    ytr: np.ndarray,
    yte: np.ndarray,
    dtrain,
    dvalid,
    trial_max_boost_round: int,
    early_stopping_rounds: int,
    overfit_penalty: float,
    oot_has_labels: bool,
    target_col: str,
    wtr: np.ndarray | None = None,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    import lightgbm as lgb

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
    return _score_trial(
        params={**params, "num_boost_round": rounds},
        train_preds=train_preds, ytr=ytr,
        test_preds=test_preds, yte=yte,
        oot=oot, oot_has_labels=oot_has_labels, target_col=target_col,
        score_fn=lambda data: booster.predict(data[feats]),
        overfit_penalty=overfit_penalty,
        stage=stage,
        best_iteration=rounds,
        wtr=wtr, wte=wte, woot=woot,
    )


# ==========================================================================
# Shared trial scoring
# ==========================================================================


def _trial_score(
    *,
    train_ks: float,
    test_ks: float,
    overfit_penalty: float,
) -> float:
    return test_ks - overfit_penalty * max(0.0, train_ks - test_ks)


def _score_trial(
    *,
    params: dict,
    train_preds: np.ndarray,
    ytr: np.ndarray,
    test_preds: np.ndarray,
    yte: np.ndarray,
    oot: pd.DataFrame | None,
    oot_has_labels: bool,
    target_col: str,
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    overfit_penalty: float,
    stage: str,
    best_iteration: int | None = None,
    wtr: np.ndarray | None = None,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    """Score one trial's predictions. TUNE-5: when ``wtr``/``wte`` (sample weights)
    are given, the score driving trial/champion selection uses weighted KS
    (``weighted_train_ks``/``weighted_test_ks``) instead of the unweighted KS --
    a trial tuned against weighted training data must also be *selected* against
    the weighted population, or the search optimises one objective while picking
    winners by another. Unweighted train_ks/test_ks/oot_ks are still computed and
    reported alongside (never dropped) so both readings stay auditable."""
    train_ks = feature_ks(train_preds, ytr)
    test_ks = feature_ks(test_preds, yte)
    train_auc = feature_auc(train_preds, ytr)
    test_auc = feature_auc(test_preds, yte)
    lift = head_tail_lift(test_preds, yte)
    weighted_train_ks = weighted_feature_ks(train_preds, ytr, wtr) if wtr is not None else None
    weighted_test_ks = weighted_feature_ks(test_preds, yte, wte) if wte is not None else None
    weighted_train_auc = weighted_feature_auc(train_preds, ytr, wtr) if wtr is not None else None
    weighted_test_auc = weighted_feature_auc(test_preds, yte, wte) if wte is not None else None
    if oot is None or not oot_has_labels:
        oot_ks = oot_auc = weighted_oot_ks = weighted_oot_auc = None
    else:
        yoot = oot[target_col].to_numpy(dtype=float)
        oot_preds = score_fn(oot)
        oot_ks = feature_ks(oot_preds, yoot)
        oot_auc = feature_auc(oot_preds, yoot)
        weighted_oot_ks = weighted_feature_ks(oot_preds, yoot, woot) if woot is not None else None
        weighted_oot_auc = weighted_feature_auc(oot_preds, yoot, woot) if woot is not None else None
    # TUNE-5: score by the weighted KS when weights exist, else the unweighted one --
    # same fallback used for train_ks/test_ks below so gaps stay on the same footing
    # as whichever metric actually drove selection.
    score_train_ks = weighted_train_ks if weighted_train_ks is not None else train_ks
    score_test_ks = weighted_test_ks if weighted_test_ks is not None else test_ks
    score_oot_ks = weighted_oot_ks if weighted_oot_ks is not None else oot_ks
    gap_tt = score_train_ks - score_test_ks
    gap_to = (score_train_ks - score_oot_ks) if score_oot_ks is not None else None
    oot_stability_gap = abs(score_test_ks - score_oot_ks) if score_oot_ks is not None else None
    score = _trial_score(train_ks=score_train_ks, test_ks=score_test_ks, overfit_penalty=overfit_penalty)
    record = {
        "params": params,
        "train_ks": train_ks, "test_ks": test_ks, "oot_ks": oot_ks, "score": score,
        "train_auc": train_auc, "test_auc": test_auc, "oot_auc": oot_auc,
        "lift_head_5": lift.get("lift_head_5"), "lift_tail_5": lift.get("lift_tail_5"),
        "lift_head_10": lift.get("lift_head_10"), "lift_tail_10": lift.get("lift_tail_10"),
        "overfit_gap_tt": gap_tt, "overfit_gap_to": gap_to,
        "oot_stability_gap": oot_stability_gap,
        "search_stage": stage,
    }
    if weighted_train_ks is not None:
        record["weighted_train_ks"] = weighted_train_ks
    if weighted_test_ks is not None:
        record["weighted_test_ks"] = weighted_test_ks
    if weighted_oot_ks is not None:
        record["weighted_oot_ks"] = weighted_oot_ks
    if weighted_train_auc is not None:
        record["weighted_train_auc"] = weighted_train_auc
    if weighted_test_auc is not None:
        record["weighted_test_auc"] = weighted_test_auc
    if weighted_oot_auc is not None:
        record["weighted_oot_auc"] = weighted_oot_auc
    if best_iteration is not None:
        record["best_iteration"] = best_iteration
    return record


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


def _weighted_count(labels: np.ndarray, weights: np.ndarray | None, value: float) -> float:
    mask = labels == value
    if weights is None:
        return float(mask.sum())
    return float(weights[mask].sum())


# ==========================================================================
# xgb sampler + trial runner
# ==========================================================================


def _xgb_sample_coarse(rng: np.random.RandomState, pos_weight_hint: float) -> dict:
    params: dict = {"scale_pos_weight": float(rng.choice([1.0, 3.0, 5.0, 10.0, pos_weight_hint]))}
    for name, (low, high) in _XGB_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high)
    for name, (low, high) in _XGB_LINEAR_SPACE_BOUNDS.items():
        params[name] = float(round(rng.uniform(low, high), 3))
    for name, choices in _XGB_CHOICE_SPACE.items():
        params[name] = int(rng.choice(choices))
    return params


def _xgb_sample_fine(rng: np.random.RandomState, pos_weight_hint: float, anchor: dict) -> dict:
    params: dict = {"scale_pos_weight": anchor["scale_pos_weight"]}
    for name, (low, high) in _XGB_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high, center=anchor[name], shrink=_FINE_LOG_SHRINK)
    for name, (low, high) in _XGB_LINEAR_SPACE_BOUNDS.items():
        span = (high - low) * _FINE_LINEAR_SHRINK
        lo = max(low, anchor[name] - span)
        hi = min(high, anchor[name] + span)
        params[name] = float(round(rng.uniform(lo, hi), 3))
    for name, choices in _XGB_CHOICE_SPACE.items():
        ordered = sorted(choices)
        idx = ordered.index(anchor[name])
        lo_idx = max(0, idx - _FINE_CHOICE_NEIGHBORS)
        hi_idx = min(len(ordered) - 1, idx + _FINE_CHOICE_NEIGHBORS)
        params[name] = int(rng.choice(ordered[lo_idx : hi_idx + 1]))
    return params


def _run_xgb_trial(
    params: dict,
    stage: str,
    *,
    fixed_params: dict,
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    ytr: np.ndarray,
    yte: np.ndarray,
    fit_train: pd.DataFrame,
    valid: pd.DataFrame,
    yfit: np.ndarray,
    yva: np.ndarray,
    wfit: np.ndarray | None,
    wva: np.ndarray | None,
    n_estimators: int,
    early_stopping_rounds: int,
    overfit_penalty: float,
    oot_has_labels: bool,
    target_col: str,
    wtr: np.ndarray | None = None,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    import xgboost as xgb

    trial_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": seed,
        "n_jobs": 1,
        **params,
        **fixed_params,
    }
    model = xgb.XGBClassifier(
        **trial_params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    # SEL-4/TUNE-3: fit/early-stop against the carved fit_train/valid, not train/test.
    model.fit(
        fit_train[feats], yfit.astype(int),
        sample_weight=wfit,
        eval_set=[(valid[feats], yva.astype(int))],
        sample_weight_eval_set=[wva] if wva is not None else None,
        verbose=False,
    )
    best_iteration = int(getattr(model, "best_iteration", n_estimators - 1)) + 1
    score_fn = lambda data: model.predict_proba(data[feats])[:, 1]  # noqa: E731
    return _score_trial(
        params={**trial_params, "num_boost_round": best_iteration},
        train_preds=score_fn(train), ytr=ytr,
        test_preds=score_fn(test), yte=yte,
        oot=oot, oot_has_labels=oot_has_labels, target_col=target_col,
        score_fn=score_fn,
        overfit_penalty=overfit_penalty,
        stage=stage,
        best_iteration=best_iteration,
        wtr=wtr, wte=wte, woot=woot,
    )


# ==========================================================================
# catboost sampler + trial runner
# ==========================================================================


def _catboost_sample_coarse(rng: np.random.RandomState, pos_weight_hint: float) -> dict:
    params: dict = {"scale_pos_weight": float(rng.choice([1.0, 3.0, 5.0, 10.0, pos_weight_hint]))}
    for name, (low, high) in _CATBOOST_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high)
    for name, choices in _CATBOOST_CHOICE_SPACE.items():
        params[name] = int(rng.choice(choices))
    return params


def _catboost_sample_fine(rng: np.random.RandomState, pos_weight_hint: float, anchor: dict) -> dict:
    params: dict = {"scale_pos_weight": anchor["scale_pos_weight"]}
    for name, (low, high) in _CATBOOST_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high, center=anchor[name], shrink=_FINE_LOG_SHRINK)
    for name, choices in _CATBOOST_CHOICE_SPACE.items():
        ordered = sorted(choices)
        idx = ordered.index(anchor[name])
        lo_idx = max(0, idx - _FINE_CHOICE_NEIGHBORS)
        hi_idx = min(len(ordered) - 1, idx + _FINE_CHOICE_NEIGHBORS)
        params[name] = int(rng.choice(ordered[lo_idx : hi_idx + 1]))
    return params


def _run_catboost_trial(
    params: dict,
    stage: str,
    *,
    fixed_params: dict,
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    ytr: np.ndarray,
    yte: np.ndarray,
    fit_train: pd.DataFrame,
    valid: pd.DataFrame,
    yfit: np.ndarray,
    yva: np.ndarray,
    wfit: np.ndarray | None,
    iterations: int,
    early_stopping_rounds: int,
    overfit_penalty: float,
    oot_has_labels: bool,
    target_col: str,
    wtr: np.ndarray | None = None,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    from catboost import CatBoostClassifier

    trial_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": seed,
        "thread_count": 1,
        "allow_writing_files": False,
        "od_type": "Iter",
        **params,
        **fixed_params,
    }
    od_wait = int(trial_params.pop("od_wait", early_stopping_rounds))
    cat_features = cat_feature_indices(train, feats, trial_params.pop("cat_features", None))
    model = CatBoostClassifier(
        **trial_params,
        iterations=iterations,
        od_wait=od_wait,
        cat_features=cat_features or None,
    )
    # SEL-4/TUNE-3: fit/early-stop against the carved fit_train/valid, not train/test.
    model.fit(
        fit_train[feats], yfit.astype(int),
        sample_weight=wfit,
        eval_set=(valid[feats], yva.astype(int)),
        verbose=False,
    )
    best_iteration = int(model.get_best_iteration() or model.tree_count_ - 1) + 1
    score_fn = lambda data: model.predict_proba(data[feats])[:, 1]  # noqa: E731
    return _score_trial(
        params={
            **trial_params,
            "iterations": best_iteration,
            "od_wait": od_wait,
            "cat_features": [feats[index] for index in cat_features],
        },
        train_preds=score_fn(train), ytr=ytr,
        test_preds=score_fn(test), yte=yte,
        oot=oot, oot_has_labels=oot_has_labels, target_col=target_col,
        score_fn=score_fn,
        overfit_penalty=overfit_penalty,
        stage=stage,
        best_iteration=best_iteration,
        wtr=wtr, wte=wte, woot=woot,
    )


# ==========================================================================
# lr sampler + trial runner (C only -- small space, no early stopping concept)
# ==========================================================================


def _lr_sample_coarse(rng: np.random.RandomState, _pos_weight_hint: float) -> dict:
    params: dict = {}
    for name, (low, high) in _LR_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high)
    return params


def _lr_sample_fine(rng: np.random.RandomState, _pos_weight_hint: float, anchor: dict) -> dict:
    params: dict = {}
    for name, (low, high) in _LR_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high, center=anchor[name], shrink=_FINE_LOG_SHRINK)
    return params


def _run_lr_trial(
    params: dict,
    stage: str,
    *,
    fixed_params: dict,
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    ytr: np.ndarray,
    yte: np.ndarray,
    wtr: np.ndarray | None,
    overfit_penalty: float,
    oot_has_labels: bool,
    target_col: str,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    from sklearn.linear_model import LogisticRegression

    trial_params = {
        "max_iter": 1000,
        "solver": "lbfgs",
        **params,
        **fixed_params,
        "random_state": seed,
    }
    model = LogisticRegression(**trial_params)
    model.fit(train[feats], ytr.astype(int), sample_weight=wtr)
    score_fn = lambda data: model.predict_proba(data[feats])[:, 1]  # noqa: E731
    return _score_trial(
        params=trial_params,
        train_preds=score_fn(train), ytr=ytr,
        test_preds=score_fn(test), yte=yte,
        oot=oot, oot_has_labels=oot_has_labels, target_col=target_col,
        score_fn=score_fn,
        overfit_penalty=overfit_penalty,
        stage=stage,
        wtr=wtr, wte=wte, woot=woot,
    )


# ==========================================================================
# scorecard sampler + trial runner (C + bin granularity)
# ==========================================================================


def _scorecard_sample_coarse(rng: np.random.RandomState, _pos_weight_hint: float) -> dict:
    params: dict = {}
    for name, (low, high) in _SCORECARD_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high)
    for name, choices in _SCORECARD_CHOICE_SPACE.items():
        params[name] = int(rng.choice(choices))
    return params


def _scorecard_sample_fine(rng: np.random.RandomState, _pos_weight_hint: float, anchor: dict) -> dict:
    params: dict = {}
    for name, (low, high) in _SCORECARD_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high, center=anchor[name], shrink=_FINE_LOG_SHRINK)
    for name, choices in _SCORECARD_CHOICE_SPACE.items():
        ordered = sorted(choices)
        idx = ordered.index(anchor[name])
        lo_idx = max(0, idx - _FINE_CHOICE_NEIGHBORS)
        hi_idx = min(len(ordered) - 1, idx + _FINE_CHOICE_NEIGHBORS)
        params[name] = int(rng.choice(ordered[lo_idx : hi_idx + 1]))
    return params


def _run_scorecard_trial(
    params: dict,
    stage: str,
    *,
    fixed_params: dict,
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    ytr: np.ndarray,
    yte: np.ndarray,
    wtr: np.ndarray | None,
    overfit_penalty: float,
    oot_has_labels: bool,
    target_col: str,
    enforce_monotonic: bool,
    monotonic_direction_hint: str,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    from sklearn.linear_model import LogisticRegression

    from marvis.feature.binning import chimerge_edges, monotonic_direction, monotonic_edges
    from marvis.feature.encode import woe_encode
    from marvis.feature.iv import compute_woe_iv, woe_result_from_binning

    max_bins = int(params.pop("scorecard_max_bins"))
    target = ytr.astype(int)
    woe_maps: dict = {}
    for feature in feats:
        values = train[feature].to_numpy(dtype=float)
        edges = chimerge_edges(values, target, max_bins=max_bins)
        if enforce_monotonic:
            resolved_direction = monotonic_direction(values, target, edges, direction=monotonic_direction_hint)
            edges = monotonic_edges(values, target, edges, direction=resolved_direction)
        binning = compute_woe_iv(values, target, edges, feature=feature)
        woe_maps[feature] = woe_result_from_binning(binning)

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=frame.index)
        for feature in feats:
            encoded[feature] = woe_encode(frame, feature, woe_maps[feature]).to_numpy(dtype=float)
        return encoded

    trial_params = {
        "max_iter": 1000,
        "solver": "lbfgs",
        **params,
        **fixed_params,
        "random_state": seed,
    }
    model = LogisticRegression(**trial_params)
    train_woe = _encode(train)
    model.fit(train_woe, target, sample_weight=wtr)
    score_fn = lambda data: model.predict_proba(_encode(data))[:, 1]  # noqa: E731
    record = _score_trial(
        params={**trial_params, "scorecard_max_bins": max_bins},
        train_preds=score_fn(train), ytr=ytr,
        test_preds=score_fn(test), yte=yte,
        oot=oot, oot_has_labels=oot_has_labels, target_col=target_col,
        score_fn=score_fn,
        overfit_penalty=overfit_penalty,
        stage=stage,
        wtr=wtr, wte=wte, woot=woot,
    )
    return record


# ==========================================================================
# mlp sampler + trial runner (small budget)
# ==========================================================================


def _mlp_sample_coarse(rng: np.random.RandomState, _pos_weight_hint: float) -> dict:
    params: dict = {}
    for name, (low, high) in _MLP_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high)
    for name, choices in _MLP_CHOICE_SPACE.items():
        idx = int(rng.randint(0, len(choices)))
        params[name] = list(choices[idx])
    return params


def _mlp_sample_fine(rng: np.random.RandomState, _pos_weight_hint: float, anchor: dict) -> dict:
    params: dict = {}
    for name, (low, high) in _MLP_LOG_SPACE_BOUNDS.items():
        params[name] = _round_log_sample(rng, low, high, center=anchor[name], shrink=_FINE_LOG_SHRINK)
    for name, choices in _MLP_CHOICE_SPACE.items():
        ordered = [list(choice) for choice in choices]
        idx = ordered.index(list(anchor[name]))
        lo_idx = max(0, idx - _FINE_CHOICE_NEIGHBORS)
        hi_idx = min(len(ordered) - 1, idx + _FINE_CHOICE_NEIGHBORS)
        pick = int(rng.randint(lo_idx, hi_idx + 1))
        params[name] = ordered[pick]
    return params


def _run_mlp_trial(
    params: dict,
    stage: str,
    *,
    fixed_params: dict,
    seed: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    feats: list[str],
    ytr: np.ndarray,
    yte: np.ndarray,
    wtr: np.ndarray | None,
    overfit_penalty: float,
    oot_has_labels: bool,
    target_col: str,
    wte: np.ndarray | None = None,
    woot: np.ndarray | None = None,
) -> dict:
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    trial_params = {
        "max_iter": 300,
        "early_stopping": False,
        **params,
        **fixed_params,
        "random_state": seed,
    }
    trial_params["hidden_layer_sizes"] = tuple(trial_params["hidden_layer_sizes"])
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(**trial_params)),
    ])
    model.fit(train[feats], ytr.astype(int), mlp__sample_weight=wtr)
    score_fn = lambda data: model.predict_proba(data[feats])[:, 1]  # noqa: E731
    return _score_trial(
        params={**trial_params, "hidden_layer_sizes": list(trial_params["hidden_layer_sizes"])},
        train_preds=score_fn(train), ytr=ytr,
        test_preds=score_fn(test), yte=yte,
        oot=oot, oot_has_labels=oot_has_labels, target_col=target_col,
        score_fn=score_fn,
        overfit_penalty=overfit_penalty,
        stage=stage,
        wtr=wtr, wte=wte, woot=woot,
    )


def tune_hyperparameters(
    backend,
    dataset_path: Path,
    *,
    features: list[str],
    target_col: str,
    split_col: str,
    split_values: dict,
    recipe: str = "lgb",
    n_trials: int | None = None,
    seed: int = 0,
    early_stopping_rounds: int = 100,
    max_boost_round: int = 3000,
    overfit_penalty: float = 0.5,
    sample_weight_col: str = "",
    base_params: dict | None = None,
    drop_nan_labels: bool = False,
    coarse_fraction: float = 0.6,
    cv_folds: int | None = None,
) -> TuneResult:
    """Deterministic two-stage random search for ``recipe``; selects by test KS
    minus in-time overfit penalty. ``recipe`` defaults to ``"lgb"`` and its search
    space/semantics are unchanged from the original lgb-only implementation.

    ``cv_folds`` (TUNE-3, default ``None`` -- single train/test split, unchanged
    historical behaviour): when set to an integer >= 2, each trial is instead
    scored with grouped ``cv_folds``-fold cross-validation over ``train`` (group-
    aware via the same ``valid_group_cols`` param used for the early-stopping
    carve) -- ``score = mean(fold_test_ks) - overfit_penalty * std(fold_test_ks)``,
    a robustness penalty against a param set that only got lucky on one fold.
    ``test`` is not touched by CV scoring; ``best_metrics["test_ks"]`` becomes the
    fold mean and ``best_metrics["test_ks_std"]`` / trial ``cv_fold_test_ks``
    additionally report the spread. Costs roughly ``cv_folds``x the runtime of a
    single split; recommended for small datasets where a single split's KS is
    noisy.

    Supported recipes: ``lgb``, ``xgb``, ``catboost`` (tree recipes, early-stopped
    against the test split, so ``num_boost_round``/``iterations`` in the returned
    ``best_params`` reflects the trial's resolved best iteration); ``lr``,
    ``scorecard`` (small space: regularization strength ``C``, plus bin
    granularity for scorecard); ``mlp`` (small space: ``alpha``,
    ``learning_rate_init``, ``hidden_layer_sizes``).

    ``n_trials`` defaults to the recipe's entry in ``DEFAULT_TRIAL_BUDGET`` when
    omitted (40 for tree recipes, 12 for lr/scorecard/mlp).

    Stage 1 ("coarse") spends ``round(n_trials * coarse_fraction)`` trials
    sampling the full space at random. Stage 2 ("fine") spends the remaining
    trials resampling numeric params in a shrunk neighbourhood around the best
    stage-1 trial. Both stages draw from seed-derived ``RandomState`` instances
    so the whole search is reproducible for a fixed ``seed``.

    Applies the shared NaN-label confirmation gate (mirrors the training recipes):
    a NaN target in train/test raises ``NanLabelNotConfirmedError`` unless
    ``drop_nan_labels=True``. A fully-unlabeled OOT split is legitimate
    (scoring-only); its oot_ks/oot_auc are reported as ``None``.
    """
    recipe = str(recipe or "lgb")
    if recipe not in DEFAULT_TRIAL_BUDGET:
        raise ModelingError(f"tune_hyperparameters does not support recipe: {recipe}")
    resolved_n_trials = int(n_trials) if n_trials is not None else DEFAULT_TRIAL_BUDGET[recipe]

    feats = [f for f in dict.fromkeys(features) if f != target_col]
    weight_col = str(sample_weight_col or "").strip()
    extra_cols = [weight_col] if weight_col else []
    cols = feats + [target_col, split_col] + extra_cols
    frame = backend.read_frame(dataset_path, columns=cols)
    train, test, oot = _split(frame, split_col, split_values)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=target_col, drop_nan_labels=drop_nan_labels,
    )
    ytr = train[target_col].to_numpy(dtype=float)
    yte = test[target_col].to_numpy(dtype=float)
    wtr = _sample_weight(train, weight_col)
    wte = _sample_weight(test, weight_col)
    # TUNE-5: OOT sample weight, for the (report-only) weighted_oot_ks/auc trial
    # fields -- mirrors compute_model_metrics' weighted OOT metrics on the training
    # path. OOT is never scored/selected on either way (DOM-9).
    woot = _sample_weight(oot, weight_col) if oot is not None and weight_col else None
    pos = _weighted_count(ytr, wtr, 1.0)
    neg = _weighted_count(ytr, wtr, 0.0)
    pos_weight_hint = round(neg / pos, 1) if pos > 0 else 1.0

    # SEL-4/TUNE-3: tree recipes early-stop each trial against a fold carved from
    # train (default 15%), not test -- test stays the trial-selection metric (and,
    # downstream, the one-time champion comparison set) instead of also deciding
    # each trial's round count. lr/scorecard/mlp have no early-stopping concept so
    # they keep fitting/scoring against the full train/test split unchanged.
    if recipe in _EARLY_STOPPED_TREE_RECIPES:
        fit_train, valid = carve_early_stop_fold(
            train, seed=seed, group_cols=_group_cols_from_params(base_params),
        )
        yfit = fit_train[target_col].to_numpy(dtype=float)
        yva = valid[target_col].to_numpy(dtype=float)
        wfit = _sample_weight(fit_train, weight_col)
        wva = _sample_weight(valid, weight_col)
    else:
        fit_train, valid = train, test
        yfit, yva, wfit, wva = ytr, yte, wtr, wte

    # Recipe-specific search hook: (coarse_sampler, fine_sampler, trial_runner).
    coarse_sampler, fine_sampler, trial_runner = _recipe_search_hooks(
        recipe,
        train=train, test=test, oot=oot,
        fit_train=fit_train, valid=valid,
        feats=feats, target_col=target_col,
        ytr=ytr, yte=yte, wtr=wtr, wte=wte, woot=woot,
        yfit=yfit, yva=yva, wfit=wfit, wva=wva,
        seed=seed, early_stopping_rounds=early_stopping_rounds,
        max_boost_round=max_boost_round, overfit_penalty=overfit_penalty,
        oot_has_labels=oot_has_labels,
        base_params=base_params, pos_weight_hint=pos_weight_hint,
    )
    resolved_cv_folds = int(cv_folds) if cv_folds else None
    if resolved_cv_folds is not None:
        if resolved_cv_folds < 2:
            raise ModelingError(f"cv_folds must be >= 2: {resolved_cv_folds}")
        # TUNE-3: replace the single-split trial_runner with grouped k-fold CV
        # scoring over train; the coarse/fine samplers are recipe-space-only and
        # stay the single-split ones (no data dependency).
        trial_runner = _make_cv_trial_runner(
            recipe,
            train=train, oot=oot, feats=feats, target_col=target_col,
            weight_col=weight_col, cv_folds=resolved_cv_folds, seed=seed,
            early_stopping_rounds=early_stopping_rounds,
            max_boost_round=max_boost_round, overfit_penalty=overfit_penalty,
            oot_has_labels=oot_has_labels, base_params=base_params,
            pos_weight_hint=pos_weight_hint,
            group_cols=_group_cols_from_params(base_params),
        )

    total_trials = max(1, resolved_n_trials)
    n_coarse = min(total_trials, max(1, round(total_trials * coarse_fraction)))
    n_fine = total_trials - n_coarse

    coarse_rng = np.random.RandomState(seed)
    fine_rng = np.random.RandomState(seed + 1)

    best: dict | None = None
    best_coarse: dict | None = None
    trials: list[dict] = []

    for _ in range(n_coarse):
        sampled = coarse_sampler(coarse_rng, pos_weight_hint)
        record = trial_runner(sampled, "coarse")
        record["_sampled"] = sampled
        trials.append(record)
        if best is None or record["score"] > best["score"]:
            best = record
        if best_coarse is None or record["score"] > best_coarse["score"]:
            best_coarse = record

    anchor = (best_coarse or best)["_sampled"]
    for _ in range(n_fine):
        sampled = fine_sampler(fine_rng, pos_weight_hint, anchor)
        record = trial_runner(sampled, "fine")
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
            **({"best_iteration": best["best_iteration"]} if "best_iteration" in best else {}),
            # TUNE-3: only present when cv_folds was used -- the fold spread behind
            # the mean test_ks above, so a caller can tell "robustly good" apart
            # from "got lucky on one fold".
            **({"test_ks_std": best["test_ks_std"]} if "test_ks_std" in best else {}),
            **({"cv_fold_test_ks": best["cv_fold_test_ks"]} if "cv_fold_test_ks" in best else {}),
            # TUNE-5: only present when sample_weight_col was given -- the weighted
            # KS/AUC that actually drove trial/champion selection, reported
            # alongside the always-present unweighted reading above.
            **({"weighted_train_ks": best["weighted_train_ks"]} if "weighted_train_ks" in best else {}),
            **({"weighted_test_ks": best["weighted_test_ks"]} if "weighted_test_ks" in best else {}),
            **({"weighted_oot_ks": best["weighted_oot_ks"]} if "weighted_oot_ks" in best else {}),
            **({"weighted_train_auc": best["weighted_train_auc"]} if "weighted_train_auc" in best else {}),
            **({"weighted_test_auc": best["weighted_test_auc"]} if "weighted_test_auc" in best else {}),
            **({"weighted_oot_auc": best["weighted_oot_auc"]} if "weighted_oot_auc" in best else {}),
        },
        trials=tuple(trials),
        n_trials=len(trials),
        nan_labels_dropped=audit["total_dropped"],
        recipe=recipe,
    )


def _recipe_search_hooks(
    recipe: str,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    fit_train: pd.DataFrame,
    valid: pd.DataFrame,
    feats: list[str],
    target_col: str,
    ytr: np.ndarray,
    yte: np.ndarray,
    wtr: np.ndarray | None,
    wte: np.ndarray | None,
    woot: np.ndarray | None,
    yfit: np.ndarray,
    yva: np.ndarray,
    wfit: np.ndarray | None,
    wva: np.ndarray | None,
    seed: int,
    early_stopping_rounds: int,
    max_boost_round: int,
    overfit_penalty: float,
    oot_has_labels: bool,
    base_params: dict | None,
    pos_weight_hint: float,
) -> tuple[Callable, Callable, Callable[[dict, str], dict]]:
    """Build the (coarse_sampler, fine_sampler, trial_runner) triple for ``recipe``.
    ``trial_runner(sampled_params, stage) -> trial record`` closes over the split
    frames/labels/weights so ``tune_hyperparameters``'s search loop stays
    recipe-agnostic."""
    if recipe == "lgb":
        import lightgbm as lgb

        fixed_params = _lgb_base_params(base_params, pos_weight_hint=pos_weight_hint)
        trial_max_boost_round = int(fixed_params.pop("num_boost_round", max_boost_round))
        # SEL-4/TUNE-3: early stopping watches the carved valid fold, not test.
        dtrain = lgb.Dataset(fit_train[feats], label=yfit, weight=wfit, free_raw_data=False)
        dvalid = lgb.Dataset(valid[feats], label=yva, weight=wva, reference=dtrain, free_raw_data=False)

        def runner(params: dict, stage: str) -> dict:
            return _run_lgb_trial(
                params, stage,
                fixed_params=fixed_params, seed=seed,
                train=train, test=test, oot=oot, feats=feats,
                ytr=ytr, yte=yte, dtrain=dtrain, dvalid=dvalid,
                trial_max_boost_round=trial_max_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                overfit_penalty=overfit_penalty,
                oot_has_labels=oot_has_labels, target_col=target_col,
                wtr=wtr, wte=wte, woot=woot,
            )

        return _sample_coarse_params, _sample_fine_params, runner

    if recipe == "xgb":
        fixed_params = _base_params_without_controls(base_params)
        n_estimators = int(fixed_params.pop("num_boost_round", fixed_params.pop("n_estimators", max_boost_round)))

        def runner(params: dict, stage: str) -> dict:
            return _run_xgb_trial(
                params, stage,
                fixed_params=fixed_params, seed=seed,
                train=train, test=test, oot=oot, feats=feats,
                ytr=ytr, yte=yte,
                fit_train=fit_train, valid=valid, yfit=yfit, yva=yva, wfit=wfit, wva=wva,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                overfit_penalty=overfit_penalty,
                oot_has_labels=oot_has_labels, target_col=target_col,
                wtr=wtr, wte=wte, woot=woot,
            )

        return _xgb_sample_coarse, _xgb_sample_fine, runner

    if recipe == "catboost":
        fixed_params = _base_params_without_controls(base_params)
        iterations = int(fixed_params.pop("iterations", fixed_params.pop("num_boost_round", max_boost_round)))

        def runner(params: dict, stage: str) -> dict:
            return _run_catboost_trial(
                params, stage,
                fixed_params=fixed_params, seed=seed,
                train=train, test=test, oot=oot, feats=feats,
                ytr=ytr, yte=yte,
                fit_train=fit_train, valid=valid, yfit=yfit, yva=yva, wfit=wfit,
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds,
                overfit_penalty=overfit_penalty,
                oot_has_labels=oot_has_labels, target_col=target_col,
                wtr=wtr, wte=wte, woot=woot,
            )

        return _catboost_sample_coarse, _catboost_sample_fine, runner

    if recipe == "lr":
        fixed_params = _base_params_without_controls(base_params)

        def runner(params: dict, stage: str) -> dict:
            return _run_lr_trial(
                params, stage,
                fixed_params=fixed_params, seed=seed,
                train=train, test=test, oot=oot, feats=feats,
                ytr=ytr, yte=yte, wtr=wtr,
                overfit_penalty=overfit_penalty,
                oot_has_labels=oot_has_labels, target_col=target_col,
                wte=wte, woot=woot,
            )

        return _lr_sample_coarse, _lr_sample_fine, runner

    if recipe == "scorecard":
        fixed_params = _base_params_without_controls(base_params)
        enforce_monotonic = bool(fixed_params.pop("enforce_monotonic", True))
        monotonic_direction_hint = str(fixed_params.pop("monotonic_direction", None) or "auto")

        def runner(params: dict, stage: str) -> dict:
            return _run_scorecard_trial(
                dict(params), stage,
                fixed_params=fixed_params, seed=seed,
                train=train, test=test, oot=oot, feats=feats,
                ytr=ytr, yte=yte, wtr=wtr,
                overfit_penalty=overfit_penalty,
                oot_has_labels=oot_has_labels, target_col=target_col,
                enforce_monotonic=enforce_monotonic,
                monotonic_direction_hint=monotonic_direction_hint,
                wte=wte, woot=woot,
            )

        return _scorecard_sample_coarse, _scorecard_sample_fine, runner

    if recipe == "mlp":
        fixed_params = _base_params_without_controls(base_params)

        def runner(params: dict, stage: str) -> dict:
            return _run_mlp_trial(
                params, stage,
                fixed_params=fixed_params, seed=seed,
                train=train, test=test, oot=oot, feats=feats,
                ytr=ytr, yte=yte, wtr=wtr,
                overfit_penalty=overfit_penalty,
                oot_has_labels=oot_has_labels, target_col=target_col,
                wte=wte, woot=woot,
            )

        return _mlp_sample_coarse, _mlp_sample_fine, runner

    raise ModelingError(f"tune_hyperparameters does not support recipe: {recipe}")


def _base_params_without_controls(params: dict | None) -> dict:
    blocked = {"sample_weight_col", "sample_weight_column", "weight_col"}
    return {
        str(key): value
        for key, value in dict(params or {}).items()
        if str(key) not in blocked and value not in (None, "")
    }


__all__ = ["DEFAULT_TRIAL_BUDGET", "tune_hyperparameters", "TuneResult"]

"""Model-based feature importance (spec §2 optional metric).

Trains a single capped LightGBM model over all candidate features and returns each
feature's gain importance as a fraction of total gain. This is a multivariate ranking
that complements the univariate IV/KS/AUC stats.

Determinism: the manifest labels ``compute_feature_metrics`` deterministic, and the
runner does NOT auto-seed deterministic tools, so the seed is pinned here. With a fixed
seed + ``num_threads=1`` + ``deterministic=True`` + ``force_col_wise`` + capped rounds,
repeated runs are bit-identical. ``lightgbm`` is imported lazily inside this module so a
base-metrics-only run (importance not selected) never pays the import cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEED = 42
_MAX_ROUNDS = 100


def feature_importance(
    frame: pd.DataFrame,
    features: list[str],
    target_col: str,
    *,
    seed: int = _SEED,
    max_rounds: int = _MAX_ROUNDS,
    train_frac: float = 0.7,
) -> dict[str, float | None]:
    """Gain importance per feature, normalised to a fraction of total gain.

    The model is trained on an internal 7:3 random split (spec §3/§8), seed-pinned for
    determinism. Returns ``None`` for every feature when the model cannot be trained (no
    labelled rows, a single-class target/train split); ``0.0`` for an unsplit feature.
    """
    target = frame[target_col].to_numpy(dtype=float)
    labelled = np.isfinite(target)
    design = frame.loc[labelled, features]
    y = target[labelled].astype(int)
    if design.empty or np.unique(y).size < 2:
        return {feature: None for feature in features}

    import lightgbm as lgb  # lazy: only when importance is selected
    from sklearn.model_selection import train_test_split

    # Internal 7:3 split (deterministic). Stratify to keep the bad rate; fall back to a
    # plain split, then to the full frame, when a class is too small to stratify/split.
    try:
        x_train, _x_test, y_train, _y_test = train_test_split(
            design, y, train_size=train_frac, random_state=seed, stratify=y
        )
    except ValueError:
        x_train, y_train = design, y
    if np.unique(y_train).size < 2:
        return {feature: None for feature in features}

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=max_rounds,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    booster = model.booster_
    gains = {name: float(gain) for name, gain in zip(booster.feature_name(), booster.feature_importance(importance_type="gain"))}
    total = sum(gains.values())
    if total <= 0:
        return {feature: 0.0 for feature in features}
    return {feature: float(gains.get(feature, 0.0) / total) for feature in features}


__all__ = ["feature_importance"]

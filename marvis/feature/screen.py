"""Leakage-aware, scale-friendly univariate feature screening.

Credit modeling samples routinely carry thousands of columns that mix genuine
predictive features with *leakage*: the future-outcome label encodings (e.g.
``max_overdue_his`` with near-perfect KS) and earlier models' own score/output
columns (``predprob``/``pred_pmml``). Blindly feeding every numeric column to a
tree model produces a catastrophic, useless model. This module screens a wide
candidate set in memory-bounded column batches, ranks features by univariate KS,
and *flags* suspected leakage so the agent can hand the candidate set back to the
user for confirmation (the "确认特征集" step) instead of silently selecting it.

Deterministic: KS/IV come from marvis.feature.metrics; no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import pandas as pd

from marvis.feature.metrics import feature_ks, feature_metrics

# Columns whose names strongly suggest a model output / score that would leak the
# target (an earlier model's prediction). These are *soft* flags surfaced for
# user confirmation, never auto-dropped, because legitimate third-party scores
# (e.g. ``pred_tongyong``) share the prefix.
_SUSPECTED_OUTPUT = re.compile(
    r"(^|_)(pred|predprob|prob|proba|probability|score|prediction)(_|$)|pmml|\.pkl$|_pkl$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScreenResult:
    ranked: tuple[tuple[str, float], ...]
    """Clean, usable features ordered by descending univariate KS: (feature, ks)."""
    selected: tuple[str, ...]
    """Proposed candidate set (top_k of ``ranked``) — for the user to confirm."""
    leakage: tuple[tuple[str, float, str], ...]
    """Hard leakage flags (KS >= leakage_ks): (feature, ks, reason)."""
    suspected: tuple[tuple[str, float, str], ...]
    """Soft flags (name looks like a model output/score): (feature, ks, reason)."""
    unusable: tuple[tuple[str, str], ...]
    """Dropped as non-numeric / near-constant / mostly-missing: (feature, reason)."""
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-feature {ks, missing_rate, unique_count}; iv added for selected ones."""
    n_screened: int = 0


def _dev_mask(
    frame: pd.DataFrame,
    split_col: str | None,
    holdout_values: tuple[str, ...],
) -> np.ndarray:
    """Rows used for screening — exclude the holdout (e.g. OOT) so selection never
    peeks at out-of-time data."""
    n = len(frame)
    if not split_col or split_col not in frame.columns or not holdout_values:
        return np.ones(n, dtype=bool)
    split = frame[split_col].astype("string")
    held = split.isin(list(holdout_values)).to_numpy(na_value=False)
    return ~held


def screen_features(
    backend,
    dataset_path: Path,
    *,
    features: list[str],
    target_col: str,
    split_col: str | None = None,
    holdout_values: tuple[str, ...] = ("oot",),
    leakage_ks: float = 0.40,
    max_missing_rate: float = 0.95,
    min_unique: int = 2,
    top_k: int | None = None,
    batch_size: int = 500,
) -> ScreenResult:
    """Screen ``features`` against ``target_col`` and propose a clean candidate set.

    - ``leakage_ks``: any single feature with univariate KS >= this on the dev rows
      is flagged as suspected leakage (a well-built multivariate credit model rarely
      exceeds ~0.35 KS, so a *single* column above ~0.40 is almost always the label
      or a near-duplicate of it).
    - ``holdout_values``: split labels (in ``split_col``) excluded from screening.
    - ``top_k``: size of the proposed candidate set (default: all clean features).
    """
    feats = [f for f in dict.fromkeys(features) if f != target_col]
    base_cols = [target_col] + ([split_col] if split_col else [])
    base = backend.read_frame(dataset_path, columns=base_cols)
    target = base[target_col].to_numpy(dtype=float)
    dev = _dev_mask(base, split_col, holdout_values)
    target_dev = target[dev]

    scores: dict[str, dict[str, float]] = {}
    leakage: list[tuple[str, float, str]] = []
    suspected: list[tuple[str, float, str]] = []
    unusable: list[tuple[str, str]] = []
    clean: list[tuple[str, float]] = []

    for start in range(0, len(feats), max(1, batch_size)):
        batch = feats[start : start + batch_size]
        frame = backend.read_frame(dataset_path, columns=batch)
        for col in batch:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            v_dev = values[dev]
            finite = np.isfinite(v_dev)
            missing_rate = float(1.0 - finite.mean()) if v_dev.size else 1.0
            unique = int(np.unique(v_dev[finite]).size)
            if missing_rate >= max_missing_rate:
                unusable.append((col, f"missing rate {missing_rate:.2f} >= {max_missing_rate}"))
                continue
            if unique < min_unique:
                unusable.append((col, f"only {unique} distinct non-null value(s)"))
                continue
            ks = feature_ks(v_dev, target_dev)
            scores[col] = {"ks": ks, "missing_rate": missing_rate, "unique_count": unique}
            if ks >= leakage_ks:
                leakage.append((col, ks, f"univariate KS {ks:.3f} >= {leakage_ks} — suspected target leakage"))
                continue
            if _SUSPECTED_OUTPUT.search(col):
                suspected.append((col, ks, "name looks like a model output/score — confirm it is an allowed input"))
            clean.append((col, ks))

    clean.sort(key=lambda z: z[1], reverse=True)
    ranked = tuple(clean)
    selected = tuple(c for c, _ in (clean[:top_k] if top_k else clean))

    # Enrich the proposed set with IV (heavier WoE binning) only for what we keep.
    if selected:
        sel_set = set(selected)
        for start in range(0, len(selected), max(1, batch_size)):
            batch = list(selected[start : start + batch_size])
            frame = backend.read_frame(dataset_path, columns=batch)
            for col in batch:
                if col not in sel_set:
                    continue
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)[dev]
                try:
                    iv = float(feature_metrics(values, target_dev, feature=col).iv)
                except Exception:
                    iv = 0.0
                scores.setdefault(col, {})["iv"] = iv

    return ScreenResult(
        ranked=ranked,
        selected=selected,
        leakage=tuple(sorted(leakage, key=lambda z: z[1], reverse=True)),
        suspected=tuple(suspected),
        unusable=tuple(unusable),
        scores=scores,
        n_screened=len(scores) + len(unusable),
    )


__all__ = ["screen_features", "ScreenResult"]

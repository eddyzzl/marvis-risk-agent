"""Deterministic label-quality helpers for the V2 packs.

A NaN target carries no supervision signal and must NEVER be silently coerced to a
class (INV-1 / INV-2). These helpers either raise ``NanLabelNotConfirmedError`` with
structured diagnostics (default) or drop the offending rows (opt-in). Non-numeric
targets remain a hard error (mirrors ``marvis/validation/checks.py``), kept separate
from the NaN case. Feature NaN handling is intentionally untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.data.errors import NanLabelNotConfirmedError


def nan_label_mask(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    """Boolean mask of rows whose label is non-finite (NaN/inf).

    Numeric strings ("0"/"1") are parsed; genuinely non-numeric targets raise
    (a hard error, distinct from the NaN case).
    """
    values = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    return ~np.isfinite(values)


def require_labels_confirmed(
    frame: pd.DataFrame,
    target_col: str,
    *,
    drop_nan_labels: bool,
    scope: str = "dataset",
) -> int:
    """Confirmation gate for paths where the deterministic core drops NaN targets itself.

    Use when the target is passed as float to a core that already isfinite-guards it
    (IV/KS/AUC/WOE/chimerge/tree). This only decides stop-vs-proceed; it does NOT filter
    rows, so binning edges keep their full-feature-distribution semantics. Returns the
    count of NaN-label rows the core will exclude (for audit). Raises
    :class:`NanLabelNotConfirmedError` when NaN labels exist and ``drop_nan_labels`` is False.
    """
    mask = nan_label_mask(frame, target_col)
    n_nan = int(mask.sum())
    if n_nan and not drop_nan_labels:
        raise NanLabelNotConfirmedError(
            target_col=target_col,
            n_total=int(len(frame)),
            n_nan=n_nan,
            scope=scope,
        )
    return n_nan


def resolve_labeled_frame(
    frame: pd.DataFrame,
    target_col: str,
    *,
    drop_nan_labels: bool,
    scope: str = "dataset",
) -> tuple[pd.DataFrame, int]:
    """Return ``(frame_without_nan_labels, n_dropped)``.

    Raises :class:`NanLabelNotConfirmedError` when NaN labels exist and
    ``drop_nan_labels`` is False. NaN labels are never coerced to a class.
    """
    mask = nan_label_mask(frame, target_col)
    n_nan = int(mask.sum())
    if n_nan == 0:
        return frame, 0
    if not drop_nan_labels:
        raise NanLabelNotConfirmedError(
            target_col=target_col,
            n_total=int(len(frame)),
            n_nan=n_nan,
            scope=scope,
        )
    return frame.loc[~mask], n_nan


def resolve_modeling_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame | None,
    *,
    target_col: str,
    drop_nan_labels: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, bool, dict]:
    """Resolve NaN labels per modeling split.

    Policy:
    - ``train`` / ``test`` MUST have labels; any NaN triggers the confirmation gate.
    - ``oot`` fully unlabeled is legitimate (scoring-only): rows are kept, but
      ``oot_has_labels`` is False so label metrics (KS/AUC) are reported as unavailable.
    - ``oot`` partially labeled triggers the gate (avoids treating a half-labelled OOT
      as a scoring set by accident).

    Returns ``(train, test, oot, oot_has_labels, audit)`` where ``audit`` is
    ``{"by_split": {role: {"n_total", "n_nan"}}, "total_dropped": int}``.
    """
    splits: dict[str, pd.DataFrame] = {"train": train, "test": test}
    if oot is not None:
        splits["oot"] = oot

    masks: dict[str, np.ndarray] = {}
    by_split: dict[str, dict] = {}
    for role, rows in splits.items():
        mask = nan_label_mask(rows, target_col)
        masks[role] = mask
        by_split[role] = {"n_total": int(len(rows)), "n_nan": int(mask.sum())}

    required_nan = by_split["train"]["n_nan"] + by_split["test"]["n_nan"]
    if required_nan and not drop_nan_labels:
        raise NanLabelNotConfirmedError(
            target_col=target_col,
            n_total=int(len(train) + len(test)),
            n_nan=int(required_nan),
            scope="train/test",
            by_split=by_split,
        )

    train_clean = train.loc[~masks["train"]]
    test_clean = test.loc[~masks["test"]]
    total_dropped = int(required_nan)

    oot_clean = oot
    oot_has_labels = False
    if oot is not None:
        n_nan = by_split["oot"]["n_nan"]
        total = by_split["oot"]["n_total"]
        if total == 0 or n_nan == total:
            # Empty OOT, or fully unlabeled OOT used for scoring only.
            oot_clean = oot
            oot_has_labels = False
        elif n_nan == 0:
            oot_clean = oot
            oot_has_labels = True
        else:  # partially labeled -> confirmation gate
            if not drop_nan_labels:
                raise NanLabelNotConfirmedError(
                    target_col=target_col,
                    n_total=int(total),
                    n_nan=int(n_nan),
                    scope="oot",
                    by_split=by_split,
                )
            oot_clean = oot.loc[~masks["oot"]]
            oot_has_labels = True
            total_dropped += int(n_nan)

    audit = {"by_split": by_split, "total_dropped": int(total_dropped)}
    return train_clean, test_clean, oot_clean, oot_has_labels, audit


__all__ = [
    "nan_label_mask",
    "require_labels_confirmed",
    "resolve_labeled_frame",
    "resolve_modeling_splits",
]

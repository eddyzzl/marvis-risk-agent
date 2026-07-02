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

from marvis.feature.metrics import DEFAULT_IV_BINS, feature_ks, feature_metrics
from marvis.feature.transform import detect_sentinel_values

# Columns whose names strongly suggest a model output / score that would leak the
# target (an earlier model's prediction). These are *soft* flags surfaced for
# user confirmation, never auto-dropped, because legitimate third-party scores
# (e.g. ``pred_tongyong``) share the prefix.
_SUSPECTED_OUTPUT = re.compile(
    r"(^|_)(pred|predprob|prob|proba|probability|score|prediction)(_|$)|pmml|\.pkl$|_pkl$",
    re.IGNORECASE,
)

# FS-4 watch-band: pooled-dev KS below the hard ``leakage_ks`` gate but at or above this
# floor is strong enough to be conditional/partial leakage (a field that only leaks in
# part of the population or is a non-linear near-copy of the label), so it is surfaced for
# confirmation instead of passing silently as a high-ranked feature.
LEAKAGE_WATCH_LOW = 0.30
# FS-4 split-shift: a per-feature |ks_train - ks_test| above this flags a migration-type
# leak (weak in-sample, anomalously strong in a later split) that the pooled KS averages away.
SPLIT_SHIFT_THRESHOLD = 0.15
# FS-4/FS-6 default non-holdout split labels used to derive the train vs test masks for
# per-split KS. A dataset whose split_col carries none of these yields no per-split flags.
_TRAIN_VALUES = ("train",)
_TEST_VALUES = ("test",)
# FS-7 "missing is informative" note: a column below this coverage but still notably
# discriminative (KS >= _NOTABLE_KS) is annotated because its KS — computed on non-missing
# rows only — understates it. _NOTABLE_KS reuses the credit-scoring "notably discriminative"
# bar (0.15) already used for the FS-4 split-shift band. Neither constant touches ranking.
_LOW_COVERAGE = 0.50
_NOTABLE_KS = 0.15


@dataclass(frozen=True)
class ScreenResult:
    ranked: tuple[tuple[str, float | None], ...]
    """Clean, usable features ordered by descending univariate KS: (feature, ks)."""
    selected: tuple[str, ...]
    """Proposed candidate set (top_k of ``ranked``) — for the user to confirm."""
    leakage: tuple[tuple[str, float, str], ...]
    """Hard leakage flags (KS >= leakage_ks): (feature, ks, reason)."""
    suspected: tuple[tuple[str, float, str], ...]
    """Soft flags (name looks like a model output/score): (feature, ks, reason)."""
    unusable: tuple[tuple[str, str], ...]
    """Dropped as non-numeric / near-constant / mostly-missing: (feature, reason)."""
    scores: dict[str, dict[str, float | None]] = field(default_factory=dict)
    """Per-feature {ks, missing_rate, unique_count}; iv added for selected ones."""
    n_screened: int = 0
    sentinel_columns: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    """Columns with a suspected sentinel/special value (PREP-4): {feature: [(value, share), ...]}.
    Purely informational (screen_features never drops or auto-treats these) so the caller
    can prompt for a sentinel_values confirmation before fitting impute/cap/normalize/bin/woe."""
    split_shift: tuple[tuple[str, float, str], ...] = ()
    """FS-4 split-shift suspicions: features whose |ks_train - ks_test| exceeds
    ``SPLIT_SHIFT_THRESHOLD`` — a migration-type / conditional-leakage signal (a field
    backfilled or redefined over time) that the pooled-dev KS gate cannot see. Purely
    informational: (feature, |ks_train - ks_test|, reason)."""
    leakage_watch: tuple[tuple[str, float, str], ...] = ()
    """FS-4 watch-band: features whose pooled-dev KS lands in [LEAKAGE_WATCH_LOW, leakage_ks)
    — strong-but-below the hard leakage gate, surfaced for confirmation rather than blocked:
    (feature, ks, reason)."""
    ks_decay_watch: tuple[tuple[str, float, str], ...] = ()
    """FS-6 KS-decay flags (only when ``max_ks_decay`` is set): features whose test/train KS
    retention ratio falls below the threshold — worked in-sample, decayed out-of-sample.
    Informational, never dropped: (feature, ks_decay, reason)."""


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


def _split_value_masks(
    frame: pd.DataFrame,
    split_col: str | None,
    train_values: tuple[str, ...] = _TRAIN_VALUES,
    test_values: tuple[str, ...] = _TEST_VALUES,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """FS-4/FS-6 per-split masks: rows labelled train vs test in ``split_col`` (holdout
    labels like OOT are neither). Returns ``(None, None)`` when ``split_col`` is absent or
    either split is empty, so callers simply produce no per-split KS (never an error)."""
    if not split_col or split_col not in frame.columns:
        return None, None
    split = frame[split_col].astype("string")
    train_mask = split.isin(list(train_values)).to_numpy(na_value=False)
    test_mask = split.isin(list(test_values)).to_numpy(na_value=False)
    if not train_mask.any() or not test_mask.any():
        return None, None
    return train_mask, test_mask


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
    max_ks_decay: float | None = None,
) -> ScreenResult:
    """Screen ``features`` against ``target_col`` and propose a clean candidate set.

    - ``leakage_ks``: any single feature with univariate KS >= this on the dev rows
      is flagged as suspected leakage (a well-built multivariate credit model rarely
      exceeds ~0.35 KS, so a *single* column above ~0.40 is almost always the label
      or a near-duplicate of it).
    - ``holdout_values``: split labels (in ``split_col``) excluded from screening.
    - ``top_k``: size of the proposed candidate set (default: all clean features).
    - ``max_ks_decay`` (FS-6): when a train/test split exists, ``scores`` always carries
      per-feature ``ks_train``/``ks_test``/``ks_decay`` (test KS ÷ train KS). Default
      ``None`` is display-only — no filtering. When set, features whose retention ratio
      falls *below* it are flagged in ``ks_decay_watch`` (still informational, not dropped).
    """
    feats = [f for f in dict.fromkeys(features) if f != target_col]
    base_cols = [target_col] + ([split_col] if split_col else [])
    base = backend.read_frame(dataset_path, columns=base_cols)
    target = base[target_col].to_numpy(dtype=float)
    dev = _dev_mask(base, split_col, holdout_values)
    target_dev = target[dev]
    # FS-4/FS-6: per-split (train vs test) target vectors for split-shift + KS-decay
    # detection. None when the dataset has no usable train/test split (no per-split flags).
    train_mask, test_mask = _split_value_masks(base, split_col)
    target_train = target[train_mask] if train_mask is not None else None
    target_test = target[test_mask] if test_mask is not None else None

    scores: dict[str, dict[str, float | None]] = {}
    leakage: list[tuple[str, float, str]] = []
    suspected: list[tuple[str, float, str]] = []
    unusable: list[tuple[str, str]] = []
    clean: list[tuple[str, float]] = []
    sentinel_columns: dict[str, list[tuple[float, float]]] = {}
    split_shift: list[tuple[str, float, str]] = []
    leakage_watch: list[tuple[str, float, str]] = []
    ks_decay_watch: list[tuple[str, float, str]] = []

    for start in range(0, len(feats), max(1, batch_size)):
        batch = feats[start : start + batch_size]
        frame = backend.read_frame(dataset_path, columns=batch)
        for col in batch:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            v_dev = values[dev]
            finite = np.isfinite(v_dev)
            missing_rate = float(1.0 - finite.mean()) if v_dev.size else 1.0
            unique = int(np.unique(v_dev[finite]).size)
            sentinel_hits = detect_sentinel_values(v_dev)
            if sentinel_hits:
                sentinel_columns[col] = sentinel_hits
            if missing_rate >= max_missing_rate:
                unusable.append((col, f"missing rate {missing_rate:.2f} >= {max_missing_rate}"))
                continue
            if unique < min_unique:
                unusable.append((col, f"only {unique} distinct non-null value(s)"))
                continue
            ks = feature_ks(v_dev, target_dev)
            coverage = 1.0 - missing_rate  # FS-7: explicit so callers need not derive it.
            scores[col] = {
                "ks": ks,
                "missing_rate": missing_rate,
                "coverage": coverage,
                "unique_count": unique,
            }
            # FS-7: a low-coverage yet discriminative column ("missing is itself informative"
            # — e.g. a bureau-query field where "no record" correlates with risk) is
            # systematically underrated because KS is computed on non-missing rows only. Note
            # it (informational; does NOT change ranking, which stays KS-based). "High" KS uses
            # 0.15, the same notably-discriminative bar the FS-4 split-shift band uses.
            if coverage < _LOW_COVERAGE and ks >= _NOTABLE_KS:
                scores[col]["note"] = "缺失即信息候选：覆盖率低但区分力强（KS仅按非缺失行计算，可能被低估）"
            # FS-4/FS-6: per-split train vs test KS (only when both splits exist). Recorded
            # for every screened column so the split-shift flag and KS-decay report share it.
            ks_train = ks_test = ks_decay = None
            if target_train is not None and target_test is not None:
                ks_train = feature_ks(values[train_mask], target_train)
                ks_test = feature_ks(values[test_mask], target_test)
                # FS-6 KS decay/retention ratio: test KS as a fraction of train KS. None when
                # train KS is 0 (no train signal to retain). A low ratio means "worked in
                # sample, failed out-of-sample" — surfaced (and optionally gated by max_ks_decay).
                ks_decay = (ks_test / ks_train) if ks_train > 0 else None
                scores[col]["ks_train"] = ks_train
                scores[col]["ks_test"] = ks_test
                scores[col]["ks_decay"] = ks_decay
            if ks >= leakage_ks:
                leakage.append((col, ks, f"univariate KS {ks:.3f} >= {leakage_ks} — suspected target leakage"))
                continue
            # FS-4 split-shift: a big train/test KS gap on a below-gate feature is a
            # migration-type / conditional-leakage signal the pooled KS averages away.
            if ks_train is not None and ks_test is not None:
                delta = abs(ks_train - ks_test)
                if delta > SPLIT_SHIFT_THRESHOLD:
                    split_shift.append((
                        col,
                        delta,
                        f"split_shift: |ks_train {ks_train:.3f} - ks_test {ks_test:.3f}| = "
                        f"{delta:.3f} > {SPLIT_SHIFT_THRESHOLD} — confirm the field is stable over time",
                    ))
            # FS-4 watch-band: strong-but-below the hard gate — informational, not blocked.
            if ks >= LEAKAGE_WATCH_LOW:
                leakage_watch.append((
                    col,
                    ks,
                    f"leakage_watch: univariate KS {ks:.3f} in [{LEAKAGE_WATCH_LOW}, {leakage_ks}) "
                    "— strong single-variable signal, confirm it is not partial/conditional leakage",
                ))
            # FS-6 KS decay: only flags when the caller opts in via max_ks_decay (default off).
            if max_ks_decay is not None and ks_decay is not None and ks_decay < max_ks_decay:
                ks_decay_watch.append((
                    col,
                    ks_decay,
                    f"ks_decay {ks_decay:.3f} < {max_ks_decay} (ks_train {ks_train:.3f} -> "
                    f"ks_test {ks_test:.3f}) — discrimination decays out-of-sample, confirm stability",
                ))
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
                    iv = float(feature_metrics(values, target_dev, feature=col, bins=DEFAULT_IV_BINS).iv)
                except Exception:
                    iv = 0.0
                col_scores = scores.setdefault(col, {})
                col_scores["iv"] = iv
                # FS-9: record the binning convention so IV is comparable across tools —
                # this path always uses equal-frequency DEFAULT_IV_BINS bins.
                col_scores["iv_binning"] = f"equal_frequency_{DEFAULT_IV_BINS}"

    return ScreenResult(
        ranked=ranked,
        selected=selected,
        leakage=tuple(sorted(leakage, key=lambda z: z[1], reverse=True)),
        suspected=tuple(suspected),
        unusable=tuple(unusable),
        scores=scores,
        n_screened=len(scores) + len(unusable),
        sentinel_columns=sentinel_columns,
        split_shift=tuple(sorted(split_shift, key=lambda z: z[1], reverse=True)),
        leakage_watch=tuple(sorted(leakage_watch, key=lambda z: z[1], reverse=True)),
        ks_decay_watch=tuple(sorted(ks_decay_watch, key=lambda z: z[1])),
    )


def sentinel_screen_notice(sentinel_columns: dict[str, list[tuple[float, float]]]) -> str:
    """Screen-gate copy (PREP-4): tell the caller which columns look like they carry
    sentinel/special values (e.g. -999/9999 "no hit" codes) so they can pass
    sentinel_values to impute/cap/normalize/bin_feature/woe_encode before fitting,
    instead of those values silently skewing fit statistics as real observations."""
    columns = ", ".join(sorted(sentinel_columns))
    return (
        f"检测到 {len(sentinel_columns)} 列疑似哨兵/特殊值（如 -999/9999 表示查无/超限）：{columns}；"
        "建议在 impute/cap/normalize/bin_feature/woe_encode 中传入 sentinel_values 按缺失处理，"
        "否则这些值会污染填充/标准化/截断/分箱的拟合统计量。"
    )


def screen_features_non_binary(
    backend,
    dataset_path: Path,
    *,
    features: list[str],
    target_col: str,
    split_col: str | None = None,
    holdout_values: tuple[str, ...] = ("oot",),
    max_missing_rate: float = 0.95,
    min_unique: int = 2,
    top_k: int | None = None,
) -> ScreenResult:
    """Screen regression/multiclass candidates without binary KS leakage math.

    Eligibility still uses the same dev-row contract as binary screening: OOT or
    other holdout rows must not decide whether a feature is usable. ``ranked``
    contains every clean feature, while ``selected`` applies ``top_k``.
    """
    feats = [f for f in dict.fromkeys(features) if f != target_col]
    base = (
        backend.read_frame(dataset_path, columns=[split_col])
        if split_col
        else None
    )
    dev = _dev_mask(base, split_col, holdout_values) if base is not None else None

    scores: dict[str, dict[str, float | None]] = {}
    unusable: list[tuple[str, str]] = []
    clean: list[tuple[str, None]] = []
    if feats:
        frame = backend.read_frame(dataset_path, columns=feats)
        for col in feats:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            v_dev = values if dev is None else values[dev]
            finite = np.isfinite(v_dev)
            missing_rate = float(1.0 - finite.mean()) if v_dev.size else 1.0
            unique = int(np.unique(v_dev[finite]).size)
            scores[col] = {"ks": None, "missing_rate": missing_rate, "unique_count": unique}
            if missing_rate >= max_missing_rate:
                unusable.append((col, "high_missing"))
                continue
            if unique < min_unique:
                unusable.append((col, "constant"))
                continue
            clean.append((col, None))

    ranked = tuple(clean)
    selected = tuple(c for c, _ in (clean[:top_k] if top_k else clean))
    return ScreenResult(
        ranked=ranked,
        selected=selected,
        leakage=(),
        suspected=(),
        unusable=tuple(unusable),
        scores=scores,
        n_screened=len(feats),
    )


__all__ = ["ScreenResult", "screen_features", "screen_features_non_binary", "sentinel_screen_notice"]

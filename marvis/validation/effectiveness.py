from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from marvis.formatting import ratio as _ratio
from marvis.formatting import score_interval as _score_interval
from marvis.feature.metrics import feature_auc as _feature_auc
from marvis.feature.metrics import feature_ks as _feature_ks
from marvis.feature.metrics import head_tail_lift as _feature_head_tail_lift
from marvis.validation.binning import (
    accumulate_bin_metrics,
    assign_bins,
    bin_distribution,
    bin_table,
    compute_ks,
    compute_psi,
    equal_frequency_bin_edges,
    reverse_score_bins_for_good_to_bad,
)
from marvis.validation.config import ValidationConfig
from marvis.validation.checks import validate_required_splits
from marvis.validation.results import (
    BinRow,
    EffectivenessResult,
    MonthlyKsRow,
    MonthlyPsiRow,
    OverallRow,
    PsiStabilityRow,
    RocKsCurve,
)
from marvis.validation.time_periods import month_key_series


@dataclass(frozen=True)
class EffectivenessContext:
    edges: Any
    train_distribution: Any


def run_effectiveness(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> EffectivenessResult:
    _check_cancelled(cancellation_check)
    validate_required_splits(
        sample,
        split_col=config.split_col,
        split_values=config.split_values,
    )
    context = prepare_effectiveness_context(
        sample=sample,
        config=config,
        cancellation_check=cancellation_check,
    )
    overall = compute_overall_ks(
        sample=sample,
        config=config,
        cancellation_check=cancellation_check,
    )
    overall = compute_overall_psi(
        sample=sample,
        config=config,
        context=context,
        overall=overall,
        cancellation_check=cancellation_check,
    )
    return build_effectiveness_result(
        overall=overall,
        bin_tables=compute_bin_tables(
            sample=sample,
            config=config,
            context=context,
            cancellation_check=cancellation_check,
        ),
        monthly_ks=compute_monthly_ks(
            sample=sample,
            config=config,
            cancellation_check=cancellation_check,
        ),
        monthly_psi=compute_monthly_psi(
            sample=sample,
            config=config,
            context=context,
            cancellation_check=cancellation_check,
        ),
        psi_stability_table=compute_psi_stability_table(
            sample=sample,
            config=config,
            cancellation_check=cancellation_check,
        ),
        roc_ks_curves=compute_roc_ks_curves(
            sample=sample,
            config=config,
            cancellation_check=cancellation_check,
        ),
    )


def prepare_effectiveness_context(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> EffectivenessContext:
    _check_cancelled(cancellation_check)
    score_col = config.score_col
    split_col = config.split_col
    splits = config.split_values

    train_rows = sample[sample[split_col] == splits["train"]]
    edges = equal_frequency_bin_edges(train_rows[score_col].to_numpy(dtype=float), config.bin_count)
    train_distribution = bin_distribution(train_rows[score_col].to_numpy(dtype=float), edges)
    return EffectivenessContext(edges=edges, train_distribution=train_distribution)


def compute_overall_ks(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> list[OverallRow]:
    score_col = config.score_col
    target_col = config.target_col
    split_col = config.split_col
    splits = config.split_values

    overall: list[OverallRow] = []
    for split_key in ("train", "test", "oot"):
        _check_cancelled(cancellation_check)
        rows_split = sample[sample[split_col] == splits[split_key]]
        scores = rows_split[score_col].to_numpy(dtype=float)
        labels = rows_split[target_col].to_numpy(dtype=int)
        sample_count = int(len(rows_split))
        bad_count = int(labels.sum()) if sample_count else 0
        bad_rate = float(labels.mean()) if sample_count else 0.0
        ks = compute_ks(scores, labels)
        auc = compute_auc(scores, labels)
        head_lift, tail_lift = compute_head_tail_lift(scores, labels)
        overall.append(OverallRow(
            split=split_key,
            ks=float(ks),
            psi_vs_train=0.0,
            sample_count=sample_count,
            bad_rate=bad_rate,
            bad_count=bad_count,
            auc=auc,
            head_lift_5pct=head_lift,
            tail_lift_5pct=tail_lift,
        ))
    return overall


def compute_overall_psi(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    context: EffectivenessContext,
    overall: list[OverallRow],
    cancellation_check: Callable[[], None] | None = None,
) -> list[OverallRow]:
    score_col = config.score_col
    split_col = config.split_col
    splits = config.split_values
    psi_by_split: dict[str, float] = {}
    for split_key in ("train", "test", "oot"):
        _check_cancelled(cancellation_check)
        rows_split = sample[sample[split_col] == splits[split_key]]
        scores = rows_split[score_col].to_numpy(dtype=float)
        distribution = bin_distribution(scores, context.edges)
        psi_by_split[split_key] = (
            0.0
            if split_key == "train"
            else float(compute_psi(context.train_distribution, distribution))
        )
    return [
        replace(row, psi_vs_train=psi_by_split.get(row.split, 0.0))
        for row in overall
    ]


def compute_bin_tables(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    context: EffectivenessContext,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, list]:
    score_col = config.score_col
    target_col = config.target_col
    split_col = config.split_col
    splits = config.split_values
    bin_tables: dict[str, list] = {}
    for split_key in ("train", "test", "oot"):
        _check_cancelled(cancellation_check)
        rows_split = sample[sample[split_col] == splits[split_key]]
        if rows_split.empty:
            bin_tables[split_key] = []
            continue
        bin_tables[split_key] = _model_analysis_eval_table(
            rows_split,
            edges=context.edges,
            score_col=score_col,
            target_col=target_col,
        )
    return bin_tables


def compute_psi_stability_table(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> list[PsiStabilityRow]:
    _check_cancelled(cancellation_check)
    score_col = config.score_col
    split_col = config.split_col
    splits = config.split_values
    expected_rows = sample[sample[split_col].isin([splits["train"], splits["test"]])]
    actual_rows = sample[sample[split_col] == splits["oot"]]
    if expected_rows.empty or actual_rows.empty:
        return []

    expected_scores = expected_rows[score_col].to_numpy(dtype=float)
    actual_scores = actual_rows[score_col].to_numpy(dtype=float)
    edges = equal_frequency_bin_edges(expected_scores, config.bin_count)
    expected_bins = _bin_counts(expected_scores, edges)
    actual_bins = _bin_counts(actual_scores, edges)
    expected_total = int(expected_bins.sum())
    actual_total = int(actual_bins.sum())
    expected_dist = _normalized_distribution(expected_bins, expected_total)
    actual_dist = _normalized_distribution(actual_bins, actual_total)
    psi_by_bin = _psi_contributions(expected_dist, actual_dist)
    rows: list[PsiStabilityRow] = []
    for index, (expected_count, actual_count) in enumerate(zip(expected_bins, actual_bins), start=1):
        _check_cancelled(cancellation_check)
        expected_pct = _ratio(expected_count, expected_total)
        actual_pct = _ratio(actual_count, actual_total)
        rows.append(
            PsiStabilityRow(
                bin_label=_score_interval(edges[index - 1], edges[index]),
                expected_count=int(expected_count),
                expected_pct=float(expected_pct),
                actual_count=int(actual_count),
                actual_pct=float(actual_pct),
                psi=float(psi_by_bin[index - 1]),
            )
        )
    return rows


def _normalized_distribution(counts: np.ndarray, total: int) -> np.ndarray:
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _psi_contributions(
    expected_dist: np.ndarray,
    actual_dist: np.ndarray,
    *,
    smoothing: float = 1e-6,
) -> np.ndarray:
    expected = np.where(expected_dist <= 0, smoothing, expected_dist.astype(float))
    actual = np.where(actual_dist <= 0, smoothing, actual_dist.astype(float))
    expected_total = float(expected.sum())
    actual_total = float(actual.sum())
    if expected_total <= 0 or actual_total <= 0:
        return np.zeros_like(expected, dtype=float)
    expected = expected / expected_total
    actual = actual / actual_total
    return np.maximum(0.0, (actual - expected) * np.log(actual / expected))


def compute_roc_ks_curves(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> dict[str, RocKsCurve]:
    score_col = config.score_col
    target_col = config.target_col
    split_col = config.split_col
    splits = config.split_values
    curves: dict[str, RocKsCurve] = {}
    for split_key in ("train", "test", "oot"):
        _check_cancelled(cancellation_check)
        rows_split = sample[sample[split_col] == splits[split_key]]
        curves[split_key] = _roc_ks_curve(
            split=split_key,
            scores=rows_split[score_col].to_numpy(dtype=float),
            labels=rows_split[target_col].to_numpy(dtype=int),
        )
    return curves


def compute_monthly_ks(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> list[MonthlyKsRow]:
    score_col = config.score_col
    target_col = config.target_col
    monthly_ks: list[MonthlyKsRow] = []
    months = month_key_series(sample[config.time_col], column_name=config.time_col)
    for month, group in sample.groupby(months, sort=True):
        _check_cancelled(cancellation_check)
        scores = group[score_col].to_numpy(dtype=float)
        labels = group[target_col].to_numpy(dtype=int)
        sample_count = int(len(group))
        bad_count = int(labels.sum()) if sample_count else 0
        bad_rate = float(labels.mean()) if sample_count else 0.0
        head_lift, tail_lift = compute_head_tail_lift(scores, labels)
        monthly_ks.append(MonthlyKsRow(
            month=str(month),
            ks=float(compute_ks(scores, labels)),
            sample_count=sample_count,
            bad_count=bad_count,
            bad_rate=bad_rate,
            auc=compute_auc(scores, labels),
            head_lift_5pct=head_lift,
            tail_lift_5pct=tail_lift,
        ))
    return monthly_ks


def _model_analysis_eval_table(
    dataframe: pd.DataFrame,
    *,
    edges,
    score_col: str,
    target_col: str,
) -> list[BinRow]:
    rows = bin_table(dataframe, edges, score_col=score_col, target_col=target_col)
    if _should_reverse_eval_bins(dataframe, score_col=score_col, target_col=target_col):
        rows = list(reversed(rows))
    return _recompute_cumulative_bin_metrics(rows)


def _should_reverse_eval_bins(
    dataframe: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
) -> bool:
    valid = dataframe[[score_col, target_col]].dropna()
    if len(valid) < 2:
        return False
    scores = valid[score_col].to_numpy(dtype=float)
    labels = valid[target_col].to_numpy(dtype=float)
    return reverse_score_bins_for_good_to_bad(scores, labels)


def _recompute_cumulative_bin_metrics(rows: list[BinRow]) -> list[BinRow]:
    """T2-2: re-accumulate the cumulative fields for the given (already-ordered)
    marginal BinRows, renumbering bin_index in walk order. Now delegates to the shared
    ``accumulate_bin_metrics`` kernel (reverse=False, since the caller already reversed
    the list when the score is inversely correlated) so this cannot drift from
    ``bin_table``'s cumulation. Behaviour-preserving vs. the previous hand-rolled loop.
    """
    marginals = [
        (row.score_lower, row.score_upper, row.sample_count, row.bad_count)
        for row in rows
    ]
    return accumulate_bin_metrics(marginals, reverse=False)


def compute_monthly_psi(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    context: EffectivenessContext,
    cancellation_check: Callable[[], None] | None = None,
) -> list[MonthlyPsiRow]:
    score_col = config.score_col
    monthly_psi: list[MonthlyPsiRow] = []
    months = month_key_series(sample[config.time_col], column_name=config.time_col)
    grouped: list[tuple[str, pd.DataFrame]] = []
    for month, group in sample.groupby(months, sort=True):
        _check_cancelled(cancellation_check)
        grouped.append((str(month), group))
    if not grouped:
        return monthly_psi

    first_month, first_group = grouped[0]
    last_month, last_group = grouped[-1]
    first_distribution = bin_distribution(first_group[score_col].to_numpy(dtype=float), context.edges)
    last_distribution = bin_distribution(last_group[score_col].to_numpy(dtype=float), context.edges)
    previous_distribution = None
    previous_month = ""

    for month, group in grouped:
        _check_cancelled(cancellation_check)
        scores = group[score_col].to_numpy(dtype=float)
        distribution = bin_distribution(scores, context.edges)
        psi_mom = None if previous_distribution is None else float(compute_psi(previous_distribution, distribution))
        monthly_psi.append(
            MonthlyPsiRow(
                month=month,
                psi_vs_train=float(compute_psi(context.train_distribution, distribution)),
                psi_first_month=0.0 if month == first_month else float(compute_psi(first_distribution, distribution)),
                psi_last_month=0.0 if month == last_month else float(compute_psi(last_distribution, distribution)),
                psi_mom=psi_mom,
                psi_mom_reference_month=previous_month,
                psi_mom_has_calendar_gap=_has_calendar_month_gap(previous_month, month),
            )
        )
        previous_distribution = distribution
        previous_month = month
    return monthly_psi


def _check_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _has_calendar_month_gap(previous_month: str, current_month: str) -> bool:
    previous_ordinal = _month_ordinal(previous_month)
    current_ordinal = _month_ordinal(current_month)
    if previous_ordinal is None or current_ordinal is None:
        return False
    return current_ordinal - previous_ordinal > 1


def _month_ordinal(month: str) -> int | None:
    text = str(month)
    if len(text) != 6 or not text.isdigit():
        return None
    year = int(text[:4])
    month_number = int(text[4:])
    if not 1 <= month_number <= 12:
        return None
    return year * 12 + month_number


def compute_auc(scores, labels) -> float:
    """T2-4: delegates to feature/metrics.py::feature_auc, the platform-wide
    reference AUC (scipy rankdata, mask on finite score+label pairs). This module
    previously reimplemented the Mann-Whitney AUC with pandas rank and a
    ``labels.astype(int)`` that silently mis-mapped NaN labels; routing through the
    canonical impl removes the duplicate and adopts its NaN-target masking (an
    improvement, not a regression) so the two call surfaces can never drift again.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return _feature_auc(scores, labels)


def compute_head_tail_lift(scores, labels, fraction: float = 0.05) -> tuple[float | None, float | None]:
    """Return validation-report lift ordered from good head to bad tail.

    The shared feature metric is direction-aware but names the high-risk end ``head``
    and the low-risk end ``tail``. Model-validation reports use the opposite population
    convention: the table starts with good/low-risk customers (head) and ends with
    bad/high-risk customers (tail). Keep the shared calculation unchanged for feature
    analysis and response-model selection, and adapt only its names at this validation
    boundary.
    """
    result = _feature_head_tail_lift(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
        fractions=(fraction,),
        min_rows=1,
    )
    pct = int(round(fraction * 100))
    risk_head_lift = result.get(f"lift_head_{pct}")
    risk_tail_lift = result.get(f"lift_tail_{pct}")
    return risk_tail_lift, risk_head_lift


def build_effectiveness_result(
    *,
    overall: list[OverallRow],
    bin_tables: dict[str, list],
    monthly_ks: list[MonthlyKsRow],
    monthly_psi: list[MonthlyPsiRow],
    psi_stability_table: list[PsiStabilityRow] | None = None,
    roc_ks_curves: dict[str, RocKsCurve] | None = None,
) -> EffectivenessResult:
    return EffectivenessResult(
        overall=overall,
        bin_tables=bin_tables,
        monthly_ks=monthly_ks,
        monthly_psi=monthly_psi,
        psi_stability_table=psi_stability_table or [],
        roc_ks_curves=roc_ks_curves or {},
    )


def _bin_counts(scores, edges) -> np.ndarray:
    bins = assign_bins(scores, edges)
    valid = bins > 0
    return np.bincount(bins[valid], minlength=len(edges))[1:len(edges)]


def _roc_ks_curve(*, split: str, scores, labels) -> RocKsCurve:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if len(scores) == 0 or positive_count == 0 or negative_count == 0:
        return RocKsCurve(
            split=split,
            fpr=[0.0, 1.0],
            tpr=[0.0, 1.0],
            ks_curve=[0.0, 0.0],
            ks=0.0,
            population_at_ks=0.0,
        )

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cum_bad = np.cumsum(sorted_labels)
    cum_good = np.cumsum(1 - sorted_labels)
    threshold_indexes = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]
    tpr = np.r_[0.0, cum_bad[threshold_indexes] / positive_count]
    fpr = np.r_[0.0, cum_good[threshold_indexes] / negative_count]
    population = np.r_[0.0, (threshold_indexes + 1) / len(sorted_scores)]
    ks_curve = tpr - fpr
    ks_index = int(np.argmax(np.abs(ks_curve)))
    # T2-4: the reported KS scalar is taken from feature/metrics.py::feature_ks (the
    # platform reference), not from this descending-order curve's own argmax, so the
    # report's KS cannot drift from the canonical KS. The curve arrays (fpr/tpr/ks_curve)
    # and population_at_ks are still derived here because the ROC/KS plot needs them; the
    # local argmax and feature_ks agree on the max |cum_bad - cum_good| magnitude.
    return RocKsCurve(
        split=split,
        fpr=[float(value) for value in fpr],
        tpr=[float(value) for value in tpr],
        ks_curve=[float(value) for value in ks_curve],
        ks=float(_feature_ks(scores, labels.astype(float))),
        population_at_ks=float(population[ks_index]),
    )

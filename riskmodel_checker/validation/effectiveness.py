from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from riskmodel_checker.validation.binning import (
    bin_distribution,
    bin_table,
    compute_ks,
    compute_psi,
    equal_frequency_bin_edges,
)
from riskmodel_checker.validation.config import ValidationConfig
from riskmodel_checker.validation.checks import validate_required_splits
from riskmodel_checker.validation.results import (
    BinRow,
    EffectivenessResult,
    MonthlyKsRow,
    MonthlyPsiRow,
    OverallRow,
    PsiStabilityRow,
    RocKsCurve,
)
from riskmodel_checker.validation.time_periods import month_key_series


@dataclass(frozen=True)
class EffectivenessContext:
    edges: Any
    train_distribution: Any


def run_effectiveness(*, sample: pd.DataFrame, config: ValidationConfig) -> EffectivenessResult:
    validate_required_splits(
        sample,
        split_col=config.split_col,
        split_values=config.split_values,
    )
    context = prepare_effectiveness_context(sample=sample, config=config)
    overall = compute_overall_ks(sample=sample, config=config)
    overall = compute_overall_psi(
        sample=sample,
        config=config,
        context=context,
        overall=overall,
    )
    return build_effectiveness_result(
        overall=overall,
        bin_tables=compute_bin_tables(sample=sample, config=config, context=context),
        monthly_ks=compute_monthly_ks(sample=sample, config=config),
        monthly_psi=compute_monthly_psi(sample=sample, config=config, context=context),
        psi_stability_table=compute_psi_stability_table(sample=sample, config=config),
        roc_ks_curves=compute_roc_ks_curves(sample=sample, config=config),
    )


def prepare_effectiveness_context(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
) -> EffectivenessContext:
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
) -> list[OverallRow]:
    score_col = config.score_col
    target_col = config.target_col
    split_col = config.split_col
    splits = config.split_values

    overall: list[OverallRow] = []
    for split_key in ("train", "test", "oot"):
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
) -> list[OverallRow]:
    score_col = config.score_col
    split_col = config.split_col
    splits = config.split_values
    psi_by_split: dict[str, float] = {}
    for split_key in ("train", "test", "oot"):
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
) -> dict[str, list]:
    score_col = config.score_col
    target_col = config.target_col
    split_col = config.split_col
    splits = config.split_values
    bin_tables: dict[str, list] = {}
    for split_key in ("train", "test", "oot"):
        rows_split = sample[sample[split_col] == splits[split_key]]
        if rows_split.empty:
            bin_tables[split_key] = []
            continue
        edges = equal_frequency_bin_edges(rows_split[score_col].to_numpy(dtype=float), config.bin_count)
        bin_tables[split_key] = _model_analysis_eval_table(
            rows_split,
            edges=edges,
            score_col=score_col,
            target_col=target_col,
        )
    return bin_tables


def compute_psi_stability_table(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
) -> list[PsiStabilityRow]:
    score_col = config.score_col
    split_col = config.split_col
    splits = config.split_values
    expected_rows = sample[sample[split_col].isin([splits["train"], splits["test"]])]
    actual_rows = sample[sample[split_col] == splits["oot"]]
    if expected_rows.empty or actual_rows.empty:
        return []

    expected_scores = expected_rows[score_col].to_numpy(dtype=float)
    actual_scores = actual_rows[score_col].to_numpy(dtype=float)
    edges = _psi_reference_edges(expected_scores, config.bin_count)
    expected_bins = _bin_counts(expected_scores, edges)
    actual_bins = _bin_counts(actual_scores, edges)
    expected_total = int(expected_bins.sum())
    actual_total = int(actual_bins.sum())
    rows: list[PsiStabilityRow] = []
    for index, (expected_count, actual_count) in enumerate(zip(expected_bins, actual_bins), start=1):
        expected_pct = _ratio(expected_count, expected_total)
        actual_pct = _ratio(actual_count, actual_total)
        rows.append(
            PsiStabilityRow(
                bin_label=_score_interval(edges[index - 1], edges[index]),
                expected_count=int(expected_count),
                expected_pct=float(expected_pct),
                actual_count=int(actual_count),
                actual_pct=float(actual_pct),
                psi=_psi_component(expected_pct, actual_pct),
            )
        )
    return rows


def compute_roc_ks_curves(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
) -> dict[str, RocKsCurve]:
    score_col = config.score_col
    target_col = config.target_col
    split_col = config.split_col
    splits = config.split_values
    curves: dict[str, RocKsCurve] = {}
    for split_key in ("train", "test", "oot"):
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
) -> list[MonthlyKsRow]:
    score_col = config.score_col
    target_col = config.target_col
    monthly_ks: list[MonthlyKsRow] = []
    months = month_key_series(sample[config.time_col], column_name=config.time_col)
    for month, group in sample.groupby(months, sort=True):
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
    correlation = np.corrcoef(
        valid[score_col].to_numpy(dtype=float),
        valid[target_col].to_numpy(dtype=float),
    )[0, 1]
    return not bool(correlation > 0)


def _recompute_cumulative_bin_metrics(rows: list[BinRow]) -> list[BinRow]:
    total = sum(row.sample_count for row in rows)
    total_bad = sum(row.bad_count for row in rows)
    total_good = total - total_bad
    overall_bad_rate = _ratio(total_bad, total)
    cumulative_count = 0
    cumulative_bad = 0
    recomputed: list[BinRow] = []
    for index, row in enumerate(rows, start=1):
        cumulative_count += row.sample_count
        cumulative_bad += row.bad_count
        cumulative_good = cumulative_count - cumulative_bad
        bad_rate = _ratio(row.bad_count, row.sample_count)
        cumulative_bad_pct = _ratio(cumulative_bad, total_bad)
        cumulative_good_pct = _ratio(cumulative_good, total_good)
        recomputed.append(
            BinRow(
                bin_index=index,
                score_lower=row.score_lower,
                score_upper=row.score_upper,
                sample_count=row.sample_count,
                bad_count=row.bad_count,
                bad_rate=bad_rate,
                cum_sample_pct=_ratio(cumulative_count, total),
                cum_bad_pct=cumulative_bad_pct,
                lift=_ratio(bad_rate, overall_bad_rate),
                ks=float(abs(cumulative_bad_pct - cumulative_good_pct)),
            )
        )
    return recomputed


def compute_monthly_psi(
    *,
    sample: pd.DataFrame,
    config: ValidationConfig,
    context: EffectivenessContext,
) -> list[MonthlyPsiRow]:
    score_col = config.score_col
    monthly_psi: list[MonthlyPsiRow] = []
    months = month_key_series(sample[config.time_col], column_name=config.time_col)
    grouped = [(str(month), group) for month, group in sample.groupby(months, sort=True)]
    if not grouped:
        return monthly_psi

    first_month, first_group = grouped[0]
    last_month, last_group = grouped[-1]
    first_distribution = bin_distribution(first_group[score_col].to_numpy(dtype=float), context.edges)
    last_distribution = bin_distribution(last_group[score_col].to_numpy(dtype=float), context.edges)
    previous_distribution = None

    for month, group in grouped:
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
            )
        )
        previous_distribution = distribution
    return monthly_psi


def compute_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if len(scores) == 0 or positive_count == 0 or negative_count == 0:
        return 0.0

    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    positive_rank_sum = float(ranks[labels == 1].sum())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    return float(auc)


def compute_head_tail_lift(scores, labels, fraction: float = 0.05) -> tuple[float | None, float | None]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    if len(scores) == 0:
        return None, None
    bad_rate = float(labels.mean())
    bucket_size = int(len(scores) * fraction)
    if bucket_size <= 0 or bad_rate == 0.0:
        return None, None

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    head_lift = float(sorted_labels[:bucket_size].mean() / bad_rate)
    tail_lift = float(sorted_labels[-bucket_size:].mean() / bad_rate)
    return head_lift, tail_lift


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


def _psi_reference_edges(scores, bin_count: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.asarray([-np.inf, np.inf], dtype=float)
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(scores, quantiles))
    if len(edges) < 2:
        return np.asarray([-np.inf, np.inf], dtype=float)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_counts(scores, edges) -> np.ndarray:
    bins = np.searchsorted(np.asarray(edges, dtype=float)[1:-1], np.asarray(scores, dtype=float), side="left")
    bins = np.clip(bins, 0, len(edges) - 2)
    return np.bincount(bins, minlength=len(edges) - 1)


def _psi_component(expected_pct: float, actual_pct: float, smoothing: float = 1e-7) -> float:
    expected = max(float(expected_pct), smoothing)
    actual = max(float(actual_pct), smoothing)
    return float((expected - actual) * np.log(expected / actual))


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
    ks_curve = tpr - fpr
    ks_index = int(np.argmax(np.abs(ks_curve)))
    return RocKsCurve(
        split=split,
        fpr=[float(value) for value in fpr],
        tpr=[float(value) for value in tpr],
        ks_curve=[float(value) for value in ks_curve],
        ks=float(abs(ks_curve[ks_index])),
        population_at_ks=float(fpr[ks_index]),
    )


def _score_interval(lower: float, upper: float) -> str:
    return f"[{_compact_number(lower)},{_compact_number(upper)}]"


def _compact_number(value: float) -> str:
    if value == -np.inf:
        return "-inf"
    if value == np.inf:
        return "inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0

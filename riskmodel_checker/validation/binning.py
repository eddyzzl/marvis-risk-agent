import numpy as np
import pandas as pd

from riskmodel_checker.validation.results import BinRow


def equal_frequency_bin_edges(scores, bin_count: int):
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([-np.inf, np.inf], dtype=float)
    edges = np.quantile(values, quantiles)
    return edges


def assign_bins(scores, edges):
    scores = np.asarray(scores, dtype=float)
    inner_edges = np.asarray(edges, dtype=float)[1:-1]
    # side="left" gives right-inclusive bins (lower, upper]: a score equal to an
    # inner edge falls into the LOWER bin (matches pd.cut(right=True)). Scores
    # below/above the outer edges clamp to the first/last bin.
    raw = np.searchsorted(inner_edges, scores, side="left") + 1
    return np.clip(raw, 1, len(edges) - 1)


def bin_distribution(scores, edges) -> np.ndarray:
    """Proportion of scores falling in each bin (zeros vector when empty)."""
    if len(scores) == 0:
        return np.zeros(len(edges) - 1, dtype=float)
    bins = assign_bins(scores, edges)
    counts = np.bincount(bins, minlength=len(edges))[1:len(edges)]
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def compute_ks(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    if len(scores) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    total_bad = int(labels.sum())
    total_good = len(labels) - total_bad
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cum_bad = np.cumsum(sorted_labels)
    cum_total = np.arange(1, len(sorted_labels) + 1)
    threshold_indexes = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]
    bad_cdf = cum_bad[threshold_indexes] / total_bad
    good_cdf = (cum_total[threshold_indexes] - cum_bad[threshold_indexes]) / total_good
    return float(np.max(np.abs(bad_cdf - good_cdf)))


def compute_psi(expected_distribution, actual_distribution, smoothing: float = 1e-6) -> float:
    expected = np.asarray(expected_distribution, dtype=float)
    actual = np.asarray(actual_distribution, dtype=float)
    expected = np.where(expected == 0, smoothing, expected)
    actual = np.where(actual == 0, smoothing, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def bin_table(
    dataframe: pd.DataFrame,
    edges,
    *,
    score_col: str,
    target_col: str,
) -> list[BinRow]:
    scores = dataframe[score_col].to_numpy(dtype=float)
    labels = dataframe[target_col].to_numpy(dtype=int)
    bins = assign_bins(scores, edges)
    total = len(scores)
    total_bad = int(labels.sum())
    overall_bad_rate = total_bad / total if total else 0.0

    rows: list[BinRow] = []
    cum_count = 0
    cum_bad = 0
    for bin_index in range(1, len(edges)):
        mask = bins == bin_index
        count = int(mask.sum())
        bad = int(labels[mask].sum())
        bad_rate = (bad / count) if count else 0.0
        cum_count += count
        cum_bad += bad
        cum_sample_pct = cum_count / total if total else 0.0
        cum_bad_pct = cum_bad / total_bad if total_bad else 0.0
        cum_good_pct = (cum_count - cum_bad) / (total - total_bad) if (total - total_bad) else 0.0
        lift = (bad_rate / overall_bad_rate) if overall_bad_rate else 0.0
        rows.append(
            BinRow(
                bin_index=bin_index,
                score_lower=float(edges[bin_index - 1]),
                score_upper=float(edges[bin_index]),
                sample_count=count,
                bad_count=bad,
                bad_rate=bad_rate,
                cum_sample_pct=cum_sample_pct,
                cum_bad_pct=cum_bad_pct,
                lift=lift,
                ks=float(abs(cum_bad_pct - cum_good_pct)),
            )
        )
    return rows

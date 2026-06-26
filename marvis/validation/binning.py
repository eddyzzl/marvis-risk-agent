import numpy as np
import pandas as pd

from marvis.feature.binning import assign_bins as _feature_assign_bins
from marvis.feature.binning import equal_frequency_edges
from marvis.feature.metrics import compute_psi as _feature_compute_psi
from marvis.feature.metrics import feature_ks
from marvis.validation.results import BinRow


def equal_frequency_bin_edges(scores, bin_count: int):
    return equal_frequency_edges(np.asarray(scores, dtype=float), bin_count)


def assign_bins(scores, edges):
    assigned = _feature_assign_bins(np.asarray(scores, dtype=float), np.asarray(edges, dtype=float))
    return np.where(assigned >= 0, assigned + 1, 0)


def bin_distribution(scores, edges) -> np.ndarray:
    """Proportion of scores falling in each bin (zeros vector when empty)."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.zeros(len(edges) - 1, dtype=float)
    bins = assign_bins(scores, edges)
    valid = bins > 0
    counts = np.bincount(bins[valid], minlength=len(edges))[1:len(edges)]
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def compute_ks(scores, labels) -> float:
    return feature_ks(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int))


def compute_psi(expected_distribution, actual_distribution, smoothing: float = 1e-6) -> float:
    return _feature_compute_psi(expected_distribution, actual_distribution, smoothing=smoothing)


def bin_table(
    dataframe: pd.DataFrame,
    edges,
    *,
    score_col: str,
    target_col: str,
) -> list[BinRow]:
    scores = dataframe[score_col].to_numpy(dtype=float)
    labels = pd.to_numeric(dataframe[target_col], errors="coerce").to_numpy(dtype=float)
    bins = assign_bins(scores, edges)
    valid = (bins > 0) & np.isfinite(labels)
    labels = labels[valid].astype(int)
    bins = bins[valid]
    total = len(labels)
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

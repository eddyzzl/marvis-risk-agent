import numpy as np
import pandas as pd

from marvis.feature.binning import assign_bins as _feature_assign_bins
from marvis.feature.binning import equal_frequency_edges
from marvis.feature.correlation import safe_correlation
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


def accumulate_bin_metrics(
    marginals: list[tuple[float, float, int, int]],
    *,
    reverse: bool = False,
) -> list[BinRow]:
    """T2-2: the single cumulative-accumulation kernel for score-band bin tables.

    Takes per-bin marginals ``(score_lower, score_upper, sample_count, bad_count)`` in
    ascending edge order and produces the running-cumulative :class:`BinRow` fields
    (``cum_sample_pct``, ``cum_bad_pct``, ``lift``, ``ks``) with the platform convention
    ``ks = |cum_bad_pct - cum_good_pct|`` and ``lift = bad_rate / overall_bad_rate``.

    ``reverse=True`` walks the bins highest-first (used by the effectiveness eval table
    when the score is negatively correlated with the label) and renumbers ``bin_index``
    ``1..N`` in walk order -- byte-identical to the previous hand-rolled loops in
    ``bin_table`` (reverse=False) and ``_recompute_cumulative_bin_metrics``. Denominators
    are the labeled/count totals of the supplied marginals; amount-weighted and
    count-vs-labeled-split tables (report_compute / report_tools) deliberately do NOT use
    this kernel because their denominators differ.
    """
    ordered = list(reversed(marginals)) if reverse else list(marginals)
    total = sum(count for _, _, count, _ in ordered)
    total_bad = sum(bad for _, _, _, bad in ordered)
    total_good = total - total_bad
    overall_bad_rate = _ratio_or_zero(total_bad, total)

    rows: list[BinRow] = []
    cum_count = 0
    cum_bad = 0
    for walk_index, (score_lower, score_upper, count, bad) in enumerate(ordered, start=1):
        bad_rate = _ratio_or_zero(bad, count)
        cum_count += count
        cum_bad += bad
        cum_good = cum_count - cum_bad
        cum_sample_pct = _ratio_or_zero(cum_count, total)
        cum_bad_pct = _ratio_or_zero(cum_bad, total_bad)
        cum_good_pct = _ratio_or_zero(cum_good, total_good)
        lift = _ratio_or_zero(bad_rate, overall_bad_rate)
        rows.append(
            BinRow(
                bin_index=walk_index,
                score_lower=float(score_lower),
                score_upper=float(score_upper),
                sample_count=int(count),
                bad_count=int(bad),
                bad_rate=bad_rate,
                cum_sample_pct=cum_sample_pct,
                cum_bad_pct=cum_bad_pct,
                lift=lift,
                ks=float(abs(cum_bad_pct - cum_good_pct)),
            )
        )
    return rows


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def bin_table(
    dataframe: pd.DataFrame,
    edges,
    *,
    score_col: str,
    target_col: str,
    reverse: bool = False,
) -> list[BinRow]:
    scores = dataframe[score_col].to_numpy(dtype=float)
    labels = pd.to_numeric(dataframe[target_col], errors="coerce").to_numpy(dtype=float)
    bins = assign_bins(scores, edges)
    valid = (bins > 0) & np.isfinite(labels)
    labels = labels[valid].astype(int)
    bins = bins[valid]

    marginals: list[tuple[float, float, int, int]] = []
    for bin_index in range(1, len(edges)):
        mask = bins == bin_index
        marginals.append((
            float(edges[bin_index - 1]),
            float(edges[bin_index]),
            int(mask.sum()),
            int(labels[mask].sum()),
        ))
    return accumulate_bin_metrics(marginals, reverse=reverse)


def reverse_score_bins_for_good_to_bad(scores, labels) -> bool:
    """Whether ascending score bands must be reversed to read good-to-bad.

    The decision is made once from the baseline score and bad-label relationship.
    Callers that compare stressed scenarios should reuse this same value for every
    scenario so the population direction cannot flip between tables.
    """
    score_values = np.asarray(scores, dtype=float)
    label_values = np.asarray(labels, dtype=float)
    return bool(safe_correlation(score_values, label_values) < 0)

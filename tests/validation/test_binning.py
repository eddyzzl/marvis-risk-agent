import math
import numpy as np
import pandas as pd
import pytest

from marvis.validation.binning import (
    accumulate_bin_metrics,
    equal_frequency_bin_edges,
    assign_bins,
    bin_table,
    compute_ks,
    compute_psi,
)


def test_equal_frequency_bin_edges_uses_quantiles():
    scores = np.arange(0, 100)
    edges = equal_frequency_bin_edges(scores, bin_count=10)
    assert len(edges) == 11
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert edges[5] == pytest.approx(49.5, abs=1.0)


def test_equal_frequency_bin_edges_dedupes_when_many_ties():
    scores = np.array([0.0] * 50 + [1.0] * 50)
    edges = equal_frequency_bin_edges(scores, bin_count=10)
    assert len(edges) < 11
    assert len(edges) == len(set(edges))


def test_equal_frequency_bin_edges_filters_non_finite_scores():
    scores = np.array([0.0, 1.0, np.nan, np.inf, -np.inf])
    edges = equal_frequency_bin_edges(scores, bin_count=2)

    assert edges.tolist() == [-np.inf, 0.5, np.inf]


def test_equal_frequency_bin_edges_returns_catchall_when_no_finite_scores():
    edges = equal_frequency_bin_edges([np.nan, np.inf], bin_count=10)

    assert edges.tolist() == [-np.inf, np.inf]


def test_assign_bins_maps_scores_to_indices():
    edges = np.array([0.0, 0.2, 0.5, 1.0])
    scores = np.array([0.1, 0.3, 0.7, 0.9, 0.0, 1.0])
    indices = assign_bins(scores, edges)
    assert list(indices) == [1, 2, 3, 3, 1, 3]


def test_compute_ks_known_values():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ks = compute_ks(scores, labels)
    assert ks == pytest.approx(1.0)


def test_compute_ks_zero_when_no_signal():
    scores = np.array([0.5] * 10)
    labels = np.array([0, 1] * 5)
    assert compute_ks(scores, labels) == pytest.approx(0.0)


def test_compute_psi_zero_when_distributions_match():
    expected = np.array([0.1] * 10)
    actual = np.array([0.1] * 10)
    assert compute_psi(expected, actual) == pytest.approx(0.0)


def test_compute_psi_handles_zero_buckets():
    expected = np.array([0.5, 0.5, 0.0])
    actual = np.array([0.5, 0.3, 0.2])
    psi = compute_psi(expected, actual)
    assert psi > 0
    assert math.isfinite(psi)


def test_bin_table_basic_shape():
    df = pd.DataFrame({
        "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "y":     [0,   0,   0,   0,   1,   1,   1,   1,   1,   0],
    })
    edges = np.array([0.0, 0.5, 1.0])
    rows = bin_table(df, edges, score_col="score", target_col="y")
    assert len(rows) == 2
    assert rows[0].bin_index == 1
    assert rows[0].sample_count == 4
    assert rows[0].bad_count == 0
    assert rows[0].bad_rate == pytest.approx(0.0)
    assert rows[1].sample_count == 6
    assert rows[1].bad_count == 5


# --- T2-2: score-band cumulation kernel (accumulate_bin_metrics) ---

def _legacy_accumulate(marginals, *, reverse=False):
    """The cumulative-accumulation loop exactly as it was hand-rolled in bin_table /
    _recompute_cumulative_bin_metrics before T2-2, kept as the ground truth the shared
    kernel must reproduce bit-for-bit."""
    ordered = list(reversed(marginals)) if reverse else list(marginals)
    total = sum(count for _, _, count, _ in ordered)
    total_bad = sum(bad for _, _, _, bad in ordered)
    total_good = total - total_bad
    overall_bad_rate = (total_bad / total) if total else 0.0
    out = []
    cum_count = 0
    cum_bad = 0
    for index, (lower, upper, count, bad) in enumerate(ordered, start=1):
        bad_rate = (bad / count) if count else 0.0
        cum_count += count
        cum_bad += bad
        cum_good = cum_count - cum_bad
        cum_sample_pct = (cum_count / total) if total else 0.0
        cum_bad_pct = (cum_bad / total_bad) if total_bad else 0.0
        cum_good_pct = (cum_good / total_good) if total_good else 0.0
        lift = (bad_rate / overall_bad_rate) if overall_bad_rate else 0.0
        out.append((
            index, float(lower), float(upper), int(count), int(bad), bad_rate,
            cum_sample_pct, cum_bad_pct, lift, abs(cum_bad_pct - cum_good_pct),
        ))
    return out


def _as_tuple(row):
    return (
        row.bin_index, row.score_lower, row.score_upper, row.sample_count,
        row.bad_count, row.bad_rate, row.cum_sample_pct, row.cum_bad_pct,
        row.lift, row.ks,
    )


def test_accumulate_bin_metrics_matches_legacy_loop_both_directions():
    """T2-2 two-sided consistency: the shared accumulate_bin_metrics kernel must
    reproduce the previously hand-rolled cumulation loop bit-for-bit, for both the
    forward (bin_table) and reversed (_recompute on inverse-correlated scores) walks,
    across random marginal shapes including empty bins and all-good/all-bad tables."""
    rng = np.random.default_rng(17)
    for _ in range(300):
        n_bins = int(rng.integers(1, 12))
        edges = np.sort(rng.uniform(-5, 5, size=n_bins + 1))
        marginals = []
        for i in range(n_bins):
            count = int(rng.integers(0, 50))
            bad = int(rng.integers(0, count + 1)) if count else 0
            marginals.append((float(edges[i]), float(edges[i + 1]), count, bad))
        for reverse in (False, True):
            got = [_as_tuple(r) for r in accumulate_bin_metrics(marginals, reverse=reverse)]
            assert got == _legacy_accumulate(marginals, reverse=reverse)


def test_bin_table_golden_values_preserved():
    """T2-2 golden lock: bin_table output pinned to concrete numbers so the kernel
    extraction cannot silently shift any value (INV-1)."""
    rng = np.random.default_rng(2024)
    n = 80
    score = np.round(rng.uniform(0, 1, size=n), 4)
    y = (rng.uniform(0, 1, size=n) < score * 0.6).astype(float)
    df = pd.DataFrame({"score": score, "y": y})
    df.loc[[2, 5, 9, 14, 33, 50], "y"] = np.nan
    edges = np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf])
    rows = bin_table(df, edges, score_col="score", target_col="y")
    got = [
        (r.bin_index, r.sample_count, r.bad_count, r.bad_rate, r.cum_sample_pct,
         r.cum_bad_pct, r.lift, r.ks)
        for r in rows
    ]
    assert got == [
        (1, 13, 1, 0.07692307692307693, 0.17567567567567569, 0.045454545454545456, 0.25874125874125875, 0.1853146853146853),
        (2, 23, 4, 0.17391304347826086, 0.4864864864864865, 0.22727272727272727, 0.5849802371541502, 0.36888111888111885),
        (3, 20, 10, 0.5, 0.7567567567567568, 0.6818181818181818, 1.6818181818181817, 0.10664335664335667),
        (4, 7, 2, 0.2857142857142857, 0.8513513513513513, 0.7727272727272727, 0.9610389610389609, 0.11188811188811187),
        (5, 11, 5, 0.45454545454545453, 1.0, 1.0, 1.5289256198347105, 0.0),
    ]


def test_accumulate_bin_metrics_empty_bins_yield_zero_not_none():
    """T2-2 empty-guard lock: the labeled/count-denominator kernel returns 0.0 on empty
    bins (never None). Pins the convention so a later refactor cannot silently unify it
    with report_tools._score_band_rows, which deliberately returns None on empty bins."""
    rows = accumulate_bin_metrics([(0.0, 1.0, 0, 0), (1.0, 2.0, 4, 2)], reverse=False)
    assert rows[0].bad_rate == 0.0
    assert rows[0].lift == 0.0
    assert rows[0].cum_sample_pct == 0.0
    assert all(r.bad_rate is not None for r in rows)

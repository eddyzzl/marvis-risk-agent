import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from marvis.feature.binning import equal_frequency_edges
from marvis.feature.errors import FeatureError
from marvis.feature.metrics import (
    BOOTSTRAP_KS_LARGE_SAMPLE_N_BOOT,
    BOOTSTRAP_KS_LARGE_SAMPLE_ROWS,
    bootstrap_ks_ci,
    compute_psi,
    feature_auc,
    feature_ks,
    feature_lift,
    feature_metrics,
    feature_psi,
)


def _naive_ks(scores: np.ndarray, target: np.ndarray) -> float:
    best = 0.0
    for threshold in sorted(set(scores)):
        bad = target == 1
        good = target == 0
        bad_rate = np.mean(scores[bad] <= threshold)
        good_rate = np.mean(scores[good] <= threshold)
        best = max(best, abs(float(bad_rate - good_rate)))
    return best


def test_feature_ks_matches_naive_threshold_scan():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
    target = np.array([0, 0, 1, 1, 0, 1])

    assert feature_ks(scores, target) == pytest.approx(_naive_ks(scores, target))
    assert feature_ks(scores, np.zeros_like(target)) == 0.0


def test_bootstrap_ks_ci_is_deterministic_and_brackets_the_point_estimate():
    """SEL-5: same seed + same scores/target always reproduces the exact same
    interval (the platform's deterministic-metrics invariant), and the interval
    always contains the point estimate it is bracketing."""
    rng = np.random.RandomState(3)
    n = 400
    target = (rng.rand(n) < 0.35).astype(float)
    scores = target * 0.5 + rng.rand(n) * 0.7

    first = bootstrap_ks_ci(scores, target, seed=42, n_boot=64)
    second = bootstrap_ks_ci(scores, target, seed=42, n_boot=64)

    assert first == second
    assert first["ks"] == pytest.approx(feature_ks(scores, target))
    assert first["ci_low"] <= first["ks"] <= first["ci_high"]
    assert first["n_boot"] == 64
    assert first["std"] >= 0.0

    # A different seed generally produces a different interval (not required to
    # differ in principle, but with 64 replicates over 400 rows the resampled
    # quantiles essentially never collide bit-for-bit across independent draws).
    third = bootstrap_ks_ci(scores, target, seed=7, n_boot=64)
    assert third["ci_low"] != first["ci_low"] or third["ci_high"] != first["ci_high"]


def test_bootstrap_ks_ci_downshifts_replicates_for_large_samples():
    """SEL-5: samples above BOOTSTRAP_KS_LARGE_SAMPLE_ROWS auto-downshift n_boot
    to bound wall-clock cost, unless the caller passed an explicit n_boot."""
    rng = np.random.RandomState(5)
    n = BOOTSTRAP_KS_LARGE_SAMPLE_ROWS + 1000
    target = (rng.rand(n) < 0.2).astype(float)
    scores = rng.rand(n)

    auto = bootstrap_ks_ci(scores, target, seed=1)
    assert auto["n_boot"] == BOOTSTRAP_KS_LARGE_SAMPLE_N_BOOT

    explicit = bootstrap_ks_ci(scores, target, seed=1, n_boot=10)
    assert explicit["n_boot"] == 10


def test_bootstrap_ks_ci_degenerate_single_class_returns_zero_width_interval():
    """SEL-5: fewer than 2 rows in either class can't support resampling -- this
    is a reporting enhancement, not a hard blocker, so it degrades to a
    zero-width interval around the (0.0) point estimate rather than raising."""
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    target = np.zeros_like(scores)

    result = bootstrap_ks_ci(scores, target, seed=1)

    assert result == {"ks": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0, "n_boot": 0}


def test_feature_auc_matches_sklearn_rank_auc():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
    target = np.array([0, 0, 1, 1, 0, 1])

    assert feature_auc(scores, target) == pytest.approx(roc_auc_score(target, scores))
    assert feature_auc(scores, np.ones_like(target)) == 0.5


def test_feature_metrics_reports_direction_agnostic_auc_for_single_features():
    values = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    target = np.array([1, 1, 0, 0])

    assert feature_auc(values, target) == pytest.approx(0.0)
    assert feature_metrics(values, target, feature="protective", bins=2).auc == pytest.approx(1.0)


def test_compute_psi_and_feature_psi_use_shared_edges_and_smoothing():
    psi = compute_psi(np.array([0.5, 0.5]), np.array([0.75, 0.25]))
    zero_safe = compute_psi(np.array([1.0, 0.0]), np.array([0.5, 0.5]))
    edges = equal_frequency_edges(np.array([1, 2, 3, 4], dtype=float), 2)
    feature_value = feature_psi(
        np.array([1, 2, 3, 4], dtype=float),
        np.array([1, 1, 1, 4], dtype=float),
        edges,
    )

    assert psi == pytest.approx((0.75 - 0.5) * np.log(0.75 / 0.5) + (0.25 - 0.5) * np.log(0.25 / 0.5))
    assert zero_safe >= 0
    assert feature_value >= 0
    with pytest.raises(FeatureError):
        compute_psi(np.array([0.5]), np.array([0.5, 0.5]))


def test_compute_psi_renormalizes_after_zero_bucket_smoothing():
    expected = np.array([1.0, 0.0])
    actual = np.array([0.5, 0.5])

    normalized_expected = np.array([1.0, 1e-6])
    normalized_expected = normalized_expected / normalized_expected.sum()
    expected_value = float(np.sum((actual - normalized_expected) * np.log(actual / normalized_expected)))

    assert compute_psi(expected, actual) == pytest.approx(expected_value)
    assert compute_psi(np.zeros(2), np.zeros(2), smoothing=0.0) == 0.0


def test_feature_lift_and_feature_metrics_ranges():
    values = np.array([1, 2, 3, 4, 5, np.nan], dtype=float)
    target = np.array([0, 0, 1, 1, 1, 0])

    lifts = feature_lift(values, target, bins=3)
    metrics = feature_metrics(
        values,
        target,
        feature="score",
        bins=3,
        compare_values=np.array([1, 1, 2, 4, 5, 6], dtype=float),
    )

    assert lifts[0] >= 1
    assert metrics.feature == "score"
    assert 0 <= metrics.ks <= 1
    assert 0 <= metrics.auc <= 1
    assert metrics.psi is not None and metrics.psi >= 0
    assert metrics.missing_rate == pytest.approx(1 / 6)
    assert metrics.unique_count == 5

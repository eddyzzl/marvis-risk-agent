"""Regression tests for the S1a score-direction primitives (marvis.data.direction).

Covers the four DirectionCheckResult status branches (skipped/inconclusive/conflict/
consistent) and normalize_score_direction's validation contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from marvis.data.direction import (
    CORR_CONFLICT_THRESHOLD,
    MIN_CORR_SAMPLE_SIZE,
    SCORE_DIRECTIONS,
    check_score_direction,
    normalize_score_direction,
)


def test_normalize_score_direction_returns_none_for_falsy_input() -> None:
    assert normalize_score_direction(None) is None
    assert normalize_score_direction("") is None


def test_normalize_score_direction_accepts_known_values() -> None:
    for value in SCORE_DIRECTIONS:
        assert normalize_score_direction(value) == value


def test_normalize_score_direction_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="invalid score_direction"):
        normalize_score_direction("sideways")


def test_check_score_direction_skips_below_min_sample_size() -> None:
    n = MIN_CORR_SAMPLE_SIZE - 1
    scores = np.linspace(0, 1, n)
    target = (scores > 0.5).astype(float)

    result = check_score_direction(scores, target, declared_direction="higher_is_riskier")

    assert result.status == "skipped"
    assert result.reason == "insufficient_labeled_sample"
    assert result.n == n
    assert result.corr is None


def test_check_score_direction_inconclusive_when_corr_below_threshold() -> None:
    n = MIN_CORR_SAMPLE_SIZE + 10
    scores = np.arange(n, dtype=float)
    # Alternating 1/0 target vs. a monotone score is deterministically ~0 corr
    # (|corr| ~= 0.043 for n=40), well under the 0.05 threshold.
    target = np.array([1.0, 0.0] * (n // 2))

    result = check_score_direction(scores, target, declared_direction="higher_is_riskier")

    assert result.status == "inconclusive"
    assert result.reason == "corr_below_threshold"
    assert abs(result.corr) < CORR_CONFLICT_THRESHOLD


def test_check_score_direction_conflict_when_declared_direction_disagrees() -> None:
    n = MIN_CORR_SAMPLE_SIZE * 2
    scores = np.linspace(0, 1, n)
    # Higher score -> higher target (positive corr) implies higher_is_riskier.
    target = (scores > 0.5).astype(float)

    result = check_score_direction(scores, target, declared_direction="higher_is_better")

    assert result.status == "conflict"
    assert result.implied_direction == "higher_is_riskier"
    assert result.corr > 0
    assert result.n == n


def test_check_score_direction_consistent_when_declared_direction_matches() -> None:
    n = MIN_CORR_SAMPLE_SIZE * 2
    scores = np.linspace(0, 1, n)
    target = (scores > 0.5).astype(float)

    result = check_score_direction(scores, target, declared_direction="higher_is_riskier")

    assert result.status == "consistent"
    assert result.implied_direction == "higher_is_riskier"
    assert result.n == n


def test_check_score_direction_drops_non_finite_pairs_before_counting() -> None:
    n = MIN_CORR_SAMPLE_SIZE
    scores = np.concatenate([np.linspace(0, 1, n), [np.nan, np.inf]])
    target = np.concatenate([(np.linspace(0, 1, n) > 0.5).astype(float), [1.0, 0.0]])

    result = check_score_direction(scores, target, declared_direction="higher_is_riskier")

    assert result.n == n

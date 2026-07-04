"""T3-1: dual-path reconciliation framework tests.

Covers the pure framework (agreement passes, injected divergence produces a
BLOCKING typed red flag carrying both path values), tolerance grading, NaN
handling, and the naive reference implementations agreeing with the production
KS / bad_rate kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from marvis.reconcile import (
    EXACT_ABS_TOL,
    FLOAT_REL_TOL,
    RECONCILE_MISMATCH_FLAG,
    ReconcileReport,
    naive_bad_rate,
    naive_ks,
    reconcile,
)


def test_identical_values_reconcile_and_carry_no_red_flag():
    result = reconcile(1234, 1234, label="join match count")
    assert result.consistent is True
    assert result.abs_diff == 0.0
    assert result.red_flag() is None


def test_injected_divergence_produces_blocking_typed_red_flag_with_both_values():
    # Two paths disagree by 3 matched rows on a count -> must block, not warn.
    result = reconcile(
        1234,
        1231,
        label="join match count",
        primary_path="duckdb_sql",
        secondary_path="pandas",
    )
    assert result.consistent is False
    flag = result.red_flag()
    assert flag is not None
    # Typed + blocking, not a soft warning.
    assert flag["code"] == RECONCILE_MISMATCH_FLAG
    assert flag["blocking"] is True
    # The payload MUST carry BOTH path values + the difference (plan risk row).
    assert flag["primary"] == 1234.0
    assert flag["secondary"] == 1231.0
    assert flag["primary_path"] == "duckdb_sql"
    assert flag["secondary_path"] == "pandas"
    assert flag["abs_diff"] == 3.0
    assert "1234" in flag["message"] and "1231" in flag["message"]


def test_reconcile_flag_token_is_recognized_by_auto_safety_layer():
    # The mismatch code must contain an AUTO_HIGH_RISK_FLAG_TOKENS token so a
    # reconciliation-mismatch gate cannot be silently auto-confirmed.
    from marvis.agent.auto_drive import AUTO_HIGH_RISK_FLAG_TOKENS  # noqa: PLC0415

    assert any(token in RECONCILE_MISMATCH_FLAG for token in AUTO_HIGH_RISK_FLAG_TOKENS)


def test_exact_tolerance_rejects_float_noise_but_blocks_real_gap():
    # Floating noise below 1e-9 absolute is tolerated on the exact path.
    near = reconcile(0.5, 0.5 + 4e-10, label="rate")
    assert near.consistent is True
    # A genuine gap above tolerance blocks.
    far = reconcile(0.5, 0.5001, label="rate")
    assert far.consistent is False


def test_float_path_tolerance_is_looser_than_exact():
    # A 1e-7 relative wobble passes on the float path (1e-6) but fails on exact (1e-9).
    primary, secondary = 100.0, 100.0 * (1 + 5e-7)
    assert reconcile(primary, secondary, label="EL", rel_tol=FLOAT_REL_TOL).consistent is True
    assert reconcile(primary, secondary, label="EL").consistent is False


def test_both_tolerances_required_to_flag_guards_tiny_denominator():
    # Two absolutely-identical tiny values must not manufacture a relative blow-up.
    result = reconcile(1e-12, 1e-12 + EXACT_ABS_TOL / 2, label="tiny")
    assert result.consistent is True


def test_nan_handling_agrees_on_undefined_but_blocks_on_disagreement():
    both_nan = reconcile(float("nan"), float("nan"), label="ks")
    assert both_nan.consistent is True
    one_nan = reconcile(float("nan"), 0.42, label="ks")
    assert one_nan.consistent is False
    assert one_nan.red_flag()["code"] == RECONCILE_MISMATCH_FLAG


def test_report_aggregates_blocking_state_and_red_flags():
    ok = reconcile(1.0, 1.0, label="a")
    bad = reconcile(1.0, 2.0, label="b")
    report = ReconcileReport(results=(ok, bad))
    assert report.blocking is True
    flags = report.red_flags()
    assert len(flags) == 1
    assert flags[0]["label"] == "b"
    payload = report.to_dict()
    assert payload["blocking"] is True
    assert len(payload["results"]) == 2


def test_all_consistent_report_is_not_blocking():
    report = ReconcileReport(results=(reconcile(3.0, 3.0, label="a"),))
    assert report.blocking is False
    assert report.red_flags() == []


def test_naive_bad_rate_matches_production_kernel():
    from marvis.feature.binning import _bad_rate  # noqa: PLC0415

    target = [1, 0, 1, 1, 0, 0, 1, 0]
    naive = naive_bad_rate(target)
    prod = _bad_rate({"bad": sum(target), "count": len(target)})
    assert reconcile(prod, naive, label="bad_rate").consistent is True
    assert naive == pytest.approx(4 / 8)


def test_naive_ks_matches_production_feature_ks():
    from marvis.feature.metrics import feature_ks  # noqa: PLC0415

    rng = np.random.default_rng(7)
    scores = rng.normal(size=500)
    # Give bads systematically higher scores so KS is well above 0.
    target = (scores + rng.normal(scale=0.5, size=500) > 0).astype(int)
    prod = feature_ks(scores, target)
    naive = naive_ks(scores, target)
    result = reconcile(prod, naive, label="KS", rel_tol=FLOAT_REL_TOL)
    assert result.consistent is True, result.to_dict()


def test_naive_ks_handles_single_class_and_empty():
    assert naive_ks([1.0, 2.0, 3.0], [1, 1, 1]) == 0.0
    assert naive_ks([], []) == 0.0

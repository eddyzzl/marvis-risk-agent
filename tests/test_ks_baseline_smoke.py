"""T4-2 smoke anchor: the KS-baseline harness runs end-to-end on synthetic data.

This is NOT a KS-standard gate — it does not certify the agent meets any public
baseline. It is the harness's own end-to-end self-test: on a small, deterministic,
signal-bearing synthetic dataset the runner must ingest → split → train → return a
real, non-trivial KS, deterministically. It proves the plumbing works even in a
sandbox with no external data, so a broken harness fails here (fast tier) rather
than only when a user later drops a real public dataset in.

The real KS-standard gate runs the SAME ``run_ks`` runner against a user-provided
public dataset (GiveMeSomeCredit / Home Credit) and compares to the ground-truth
baselines — see ``scripts/ks_baseline.py`` and ``docs/ks_baseline/README.md``.
"""

from __future__ import annotations

from pathlib import Path

from support.ks_harness import (
    DEFAULT_KS_TOLERANCE,
    SplitSpec,
    compare_to_baseline,
    load_baselines,
    run_ks,
)

from marvis.sample_data import generate_sample_frame

_SMOKE_FEATURES = [
    "credit_score",
    "debt_income_ratio",
    "monthly_income",
    "loan_amount",
    "history_overdue_count",
    "account_age_months",
]


def _run(tmp_path, *, seed_rows: int = 2000, seed: int = 20260701, recipe: str = "lr"):
    frame = generate_sample_frame(n_rows=seed_rows, seed=seed)
    return run_ks(
        frame,
        features=_SMOKE_FEATURES,
        target_col="y",
        dataset_name="synthetic_smoke",
        recipe=recipe,
        split=SplitSpec(seed=seed),
        workdir=tmp_path / "work",
    )


def test_harness_runs_end_to_end_and_reports_a_real_ks(tmp_path):
    result = _run(tmp_path)
    assert result.n_rows == 2000
    assert result.n_features == len(_SMOKE_FEATURES)
    # a real signal, not noise: the synthetic label is a logistic function of the
    # features, so KS must be comfortably above chance (KS=0) on every split.
    assert result.test_ks is not None and result.test_ks > 0.2
    assert result.train_ks is not None and result.train_ks > 0.2
    assert result.oot_ks is not None and result.oot_ks > 0.2
    assert result.test_auc is not None and result.test_auc > 0.6


def test_harness_is_deterministic(tmp_path):
    first = _run(tmp_path / "a")
    second = _run(tmp_path / "b")
    assert first.test_ks == second.test_ks
    assert first.train_ks == second.train_ks
    assert first.oot_ks == second.oot_ks


def test_smoke_anchor_meets_its_stored_baseline(tmp_path):
    """The synthetic anchor carries its OWN baseline (a conservative floor, not a
    KS-standard) so the baseline-compare mechanism is itself exercised in CI."""
    baselines = load_baselines(
        Path(__file__).parents[1] / "docs" / "ks_baseline" / "baselines.json"
    )
    assert "synthetic_smoke" in baselines, "smoke anchor baseline must be present"
    baseline_ks = float(baselines["synthetic_smoke"]["baseline_ks"])

    result = _run(tmp_path)
    verdict = compare_to_baseline(result, baseline_ks, tolerance=DEFAULT_KS_TOLERANCE)
    assert verdict.passed, verdict.render()
    # the smoke baseline is deliberately conservative so the anchor is stable.
    assert baseline_ks <= result.test_ks + 1e-9


def test_baseline_compare_flags_a_regression(tmp_path):
    """The compare mechanism must FAIL when the agent KS falls below the floor —
    guards the gate itself (a gate that always passes is worthless)."""
    result = _run(tmp_path)
    # pretend the human-tuned baseline is far above what the agent achieves.
    inflated_baseline = (result.test_ks or 0.0) + 0.10
    verdict = compare_to_baseline(result, inflated_baseline, tolerance=DEFAULT_KS_TOLERANCE)
    assert not verdict.passed
    assert verdict.margin is not None and verdict.margin < 0

"""T3-1: cross-check headline numbers against independent naive reference
implementations via the reconcile framework.

Covered here (reference implementation feasible + high value):
  - vintage cum_bad_rate (validation/vintage.py) vs a naive numpy accumulation
  - bad_rate + KS are covered in test_reconcile.py against the production kernels

Explicitly NOT covered by this MVP (documented deferral):
  - EL total (packs/analysis/loss.py): its value comes from a Markov absorbing-
    chain migration matrix + matrix power; a faithful independent reference has to
    reconstruct the exact migration-matrix normalization (force loss row = identity,
    drop the exit column, row-normalize) which is complex enough that the reference
    would likely share the production bug rather than catch it. Listed as a follow-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.reconcile import FLOAT_REL_TOL, reconcile
from marvis.validation.vintage import compute_vintage_curve


def _naive_incremental_cum_bad_rate(
    frame: pd.DataFrame, *, cohort_col: str, mob_col: str, target_col: str
) -> dict[tuple[str, int], float]:
    """The plainest possible incremental vintage cumulative bad rate: for each
    cohort, accumulate the bad count over ascending MOB and divide by the fixed
    cohort base (max sample count across the cohort's MOBs), clipped to 1.0.
    Independent of compute_vintage_curve's pandas groupby machinery."""
    out: dict[tuple[str, int], float] = {}
    for cohort in sorted(frame[cohort_col].astype(str).unique()):
        sub = frame[frame[cohort_col].astype(str) == cohort]
        mobs = sorted(int(m) for m in sub[mob_col].unique())
        per_mob_counts = {
            int(m): int((sub[mob_col].astype(int) == int(m)).sum()) for m in mobs
        }
        base = float(max(per_mob_counts.values())) if per_mob_counts else 0.0
        cumulative_bad = 0.0
        for mob in mobs:
            rows = sub[sub[mob_col].astype(int) == mob]
            cumulative_bad += float((rows[target_col].astype(int) == 1).sum())
            ratio = 0.0 if base == 0 else min(cumulative_bad / base, 1.0)
            out[(cohort, mob)] = ratio
    return out


def _vintage_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    records = []
    for cohort in ("2023-01", "2023-02"):
        for mob in range(1, 5):
            n = 30
            # More bad as MOB grows (incremental-style marginal contributions).
            bad = rng.integers(0, 1 + mob, size=n)
            for value in bad:
                records.append({"cohort": cohort, "mob": mob, "target": int(value > 0)})
    return pd.DataFrame(records)


def test_vintage_cum_bad_rate_matches_naive_reference():
    frame = _vintage_frame()
    points = compute_vintage_curve(
        frame, cohort_col="cohort", mob_col="mob", target_col="target",
        label_semantics="incremental",
    )
    reference = _naive_incremental_cum_bad_rate(
        frame, cohort_col="cohort", mob_col="mob", target_col="target"
    )
    assert points  # sanity: the curve produced points
    for point in points:
        expected = reference[(point.cohort, point.mob)]
        result = reconcile(
            point.cum_bad_rate,
            expected,
            rel_tol=FLOAT_REL_TOL,
            label=f"vintage cum_bad_rate {point.cohort}@MOB{point.mob}",
            primary_path="vintage_kernel",
            secondary_path="naive_numpy",
        )
        assert result.consistent is True, result.to_dict()


def test_vintage_reference_would_catch_a_broken_accumulation():
    """Guard the guard: if the reference and kernel accumulated differently, the
    reconcile layer must flag it. Here we deliberately reconcile the kernel's
    cumulative value against the per-MOB marginal bad_rate (a WRONG "cumulative"),
    and assert the divergence blocks -- proving the cross-check has teeth."""
    frame = _vintage_frame()
    points = compute_vintage_curve(
        frame, cohort_col="cohort", mob_col="mob", target_col="target",
        label_semantics="incremental",
    )
    # Find a point where cumulative != marginal so the check is meaningful.
    divergent = [p for p in points if abs(p.cum_bad_rate - p.bad_rate) > 1e-6]
    assert divergent, "test data should produce a genuine cumulative-vs-marginal gap"
    p = divergent[0]
    result = reconcile(p.cum_bad_rate, p.bad_rate, label="broken cum check")
    assert result.consistent is False

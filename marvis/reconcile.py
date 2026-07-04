"""T3-1: dual-path reconciliation of headline decision numbers.

A headline gate number (join match count, vintage cum_bad_rate, EL total, KS,
bad_rate) is only trustworthy if it survives being computed a *second*,
independent way. :func:`reconcile` compares a ``primary`` (the authoritative,
usually optimized production kernel) against a ``secondary`` (a genuinely
independent second path -- e.g. DuckDB SQL vs a pandas/numpy reference) and,
when they disagree beyond tolerance, produces a **blocking red flag** (typed,
carried in the gate payload) rather than a soft warning.

The point is to turn a human's forced confirmation from rubber-stamping a single
number into *seeing a disagreement between two paths* when one exists.

Tolerances are graded (T3 spec / plan risk row): an exact integer/count path
reconciles at ``1e-9`` (only floating-representation noise is tolerated); a
floating-point kernel (matrix power, cumulative division) reconciles at ``1e-6``.
A disagreement must exceed BOTH the relative and absolute tolerance to flag, so a
tiny denominator can't manufacture a false relative blow-up on two values that
are absolutely identical to the last bit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

#: default tolerance for an exact path (integer counts, exact SQL vs exact python)
EXACT_REL_TOL = 1e-9
EXACT_ABS_TOL = 1e-9

#: default tolerance for a floating-point kernel (division / matrix power / cumsum)
FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-9

#: typed red-flag code stamped on a blocking reconciliation mismatch. Contains an
#: ``approval`` / ``irreversible`` token so the AUTO safety layer
#: (gates/contracts._gate_risk_reason) treats a mismatch gate as non-auto-confirmable.
RECONCILE_MISMATCH_FLAG = "reconcile_mismatch_blocking_approval"


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of comparing two independently-computed values for one number.

    ``consistent`` is False only when the two paths disagree beyond BOTH
    tolerances -- that is the blocking condition. ``red_flag`` renders the typed
    payload entry the gate carries so a mismatch cannot be auto-confirmed.
    """

    label: str
    primary: float
    secondary: float
    abs_diff: float
    rel_diff: float
    rel_tol: float
    abs_tol: float
    consistent: bool
    primary_path: str = "primary"
    secondary_path: str = "secondary"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "primary": self.primary,
            "secondary": self.secondary,
            "primary_path": self.primary_path,
            "secondary_path": self.secondary_path,
            "abs_diff": self.abs_diff,
            "rel_diff": self.rel_diff,
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "consistent": self.consistent,
        }

    def red_flag(self) -> dict[str, Any] | None:
        """The typed, blocking red-flag payload for a mismatch, or None when the
        two paths agree. The message shows BOTH path values + their difference so
        a human sees the divergence, not just a bare alarm (plan risk row: a
        red flag payload MUST carry both values)."""
        if self.consistent:
            return None
        return {
            "code": RECONCILE_MISMATCH_FLAG,
            "label": self.label,
            "blocking": True,
            "message": (
                f"对账不一致：{self.label} 两路计算结果分歧。"
                f"权威路({self.primary_path})={self.primary}，"
                f"独立路({self.secondary_path})={self.secondary}，"
                f"绝对差={self.abs_diff:.3g}（阈值 {self.abs_tol:.1g}），"
                f"相对差={self.rel_diff:.3g}（阈值 {self.rel_tol:.1g}）。"
                "请核对两条计算路径，确认哪一个正确后再继续。"
            ),
            "primary": self.primary,
            "secondary": self.secondary,
            "primary_path": self.primary_path,
            "secondary_path": self.secondary_path,
            "abs_diff": self.abs_diff,
            "rel_diff": self.rel_diff,
        }


def reconcile(
    primary: float,
    secondary: float,
    *,
    rel_tol: float = EXACT_REL_TOL,
    abs_tol: float = EXACT_ABS_TOL,
    label: str,
    primary_path: str = "primary",
    secondary_path: str = "secondary",
) -> ReconcileResult:
    """Compare two independently-computed values for a single headline number.

    A mismatch is blocking (``consistent=False``) only when the absolute
    difference exceeds ``abs_tol`` AND the relative difference exceeds
    ``rel_tol``. Requiring both prevents a near-zero denominator from turning a
    bit-identical pair into a spurious relative blow-up, and prevents two large
    numbers that differ by an absolutely tiny amount from flagging.

    NaN handling: two NaNs are treated as consistent (both paths agree the number
    is undefined); exactly one NaN is always a blocking mismatch (the paths
    disagree on whether the number exists).
    """
    primary_f = float(primary)
    secondary_f = float(secondary)

    primary_nan = math.isnan(primary_f)
    secondary_nan = math.isnan(secondary_f)
    if primary_nan and secondary_nan:
        return ReconcileResult(
            label=label,
            primary=primary_f,
            secondary=secondary_f,
            abs_diff=0.0,
            rel_diff=0.0,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            consistent=True,
            primary_path=primary_path,
            secondary_path=secondary_path,
        )
    if primary_nan or secondary_nan:
        return ReconcileResult(
            label=label,
            primary=primary_f,
            secondary=secondary_f,
            abs_diff=math.inf,
            rel_diff=math.inf,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            consistent=False,
            primary_path=primary_path,
            secondary_path=secondary_path,
        )

    abs_diff = abs(primary_f - secondary_f)
    scale = max(abs(primary_f), abs(secondary_f))
    rel_diff = abs_diff / scale if scale > 0 else 0.0
    consistent = abs_diff <= abs_tol or rel_diff <= rel_tol
    return ReconcileResult(
        label=label,
        primary=primary_f,
        secondary=secondary_f,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        consistent=consistent,
        primary_path=primary_path,
        secondary_path=secondary_path,
    )


@dataclass(frozen=True)
class ReconcileReport:
    """A bundle of per-number reconciliations attached to one gate payload.

    ``blocking`` is True if any member reconciliation disagreed; ``red_flags``
    is the list of typed blocking flags the gate must surface.
    """

    results: tuple[ReconcileResult, ...] = ()

    @property
    def blocking(self) -> bool:
        return any(not result.consistent for result in self.results)

    def red_flags(self) -> list[dict[str, Any]]:
        return [flag for result in self.results if (flag := result.red_flag()) is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocking": self.blocking,
            "results": [result.to_dict() for result in self.results],
            "red_flags": self.red_flags(),
        }


# --- Naive reference implementations (independent second paths) --------------
# These are deliberately the simplest, most direct numpy/python versions of a
# headline number, kept INDEPENDENT of the production kernels so a bug in the
# optimized path shows up as a reconciliation mismatch. They are NOT used for
# production display -- only as the ``secondary`` in reconcile().


def naive_bad_rate(target: Any) -> float:
    """count(target==1) / count(non-null target) -- the plainest bad rate."""
    import numpy as np

    array = np.asarray(list(target), dtype="float64")
    finite = array[~np.isnan(array)]
    if finite.size == 0:
        return float("nan")
    bad = float(np.sum(finite == 1.0))
    return bad / float(finite.size)


def naive_ks(scores: Any, target: Any) -> float:
    """Direct KS: max gap between the cumulative bad and good distributions over
    all rank positions. Independent of feature.metrics.feature_ks -- this version
    evaluates the gap at EVERY sorted row (no change-point compression), which is
    numerically equal to the change-point max for the same data."""
    import numpy as np

    scores_arr = np.asarray(list(scores), dtype="float64")
    target_arr = np.asarray(list(target), dtype="float64")
    mask = np.isfinite(scores_arr) & np.isfinite(target_arr)
    mask &= (target_arr == 0.0) | (target_arr == 1.0)
    scores_arr = scores_arr[mask]
    target_arr = target_arr[mask]
    if scores_arr.size == 0:
        return 0.0
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_target = target_arr[order]
    total_bad = float(np.sum(sorted_target == 1.0))
    total_good = float(np.sum(sorted_target == 0.0))
    if total_bad == 0.0 or total_good == 0.0:
        return 0.0
    cum_bad = np.cumsum(sorted_target == 1.0) / total_bad
    cum_good = np.cumsum(sorted_target == 0.0) / total_good
    return float(np.max(np.abs(cum_bad - cum_good)))


__all__ = [
    "EXACT_ABS_TOL",
    "EXACT_REL_TOL",
    "FLOAT_ABS_TOL",
    "FLOAT_REL_TOL",
    "RECONCILE_MISMATCH_FLAG",
    "ReconcileReport",
    "ReconcileResult",
    "naive_bad_rate",
    "naive_ks",
    "reconcile",
]

"""Score-direction primitives shared across packs (S1a).

A "score" (PD, points, or any monotone risk ranking) is only meaningful once its
direction is known: does a higher value mean higher risk, or lower risk? Several
independent tool consumers (tradeoff_view, reject_inference, build_strategy) need
to agree on this, and where a labeled sample exists we can check the declared
direction deterministically against the empirical corr(score, target) sign. This
module holds the shared enum, the deterministic self-check, and nothing else --
callers own their own default-direction semantics (see the spec at
docs/plans/specs/v2-s1a-score-direction-spec.md §2 for consumer-specific defaults).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from marvis.feature.correlation import safe_correlation


ScoreDirection = Literal["higher_is_riskier", "higher_is_better"]

SCORE_DIRECTIONS: tuple[ScoreDirection, ...] = ("higher_is_riskier", "higher_is_better")

# Statistical thresholds for the deterministic direction self-check (§3.2/§3.3 of the
# spec). Both are explicitly flagged there as suggested defaults pending user sign-off
# (see spec §6 open question 1), not pre-existing platform constants.
MIN_CORR_SAMPLE_SIZE = 30
CORR_CONFLICT_THRESHOLD = 0.05


def normalize_score_direction(value: str | None) -> ScoreDirection | None:
    """Validate + normalize a caller-supplied direction string.

    Returns None when value is falsy (caller must supply their own default --
    this module never guesses; see consumer-specific default semantics in the spec).
    Raises ValueError for any non-empty value outside SCORE_DIRECTIONS (typed as
    plain ValueError, not a DataLayerError subclass -- this is a schema-shape
    problem, not a data-quality gate; manifest enum validation should already
    catch it before this runs, this is defense in depth).
    """
    if not value:
        return None
    normalized = str(value)
    if normalized not in SCORE_DIRECTIONS:
        raise ValueError(f"invalid score_direction: {normalized!r}; expected one of {SCORE_DIRECTIONS}")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class DirectionCheckResult:
    status: Literal["skipped", "inconclusive", "conflict", "consistent"]
    n: int
    corr: float | None
    reason: str | None = None
    implied_direction: ScoreDirection | None = None


def check_score_direction(
    scores: np.ndarray,
    target: np.ndarray,
    *,
    declared_direction: ScoreDirection,
    min_sample_size: int = MIN_CORR_SAMPLE_SIZE,
    corr_threshold: float = CORR_CONFLICT_THRESHOLD,
) -> DirectionCheckResult:
    """Deterministic direction self-check, shared by tradeoff_view/reject_inference.

    Does not modify scores/target; pure function, safe to call repeatedly with the
    same result (INV-1 determinism requirement).
    """
    scores_arr = np.asarray(scores, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    finite_mask = np.isfinite(scores_arr) & np.isfinite(target_arr)
    scores_f, target_f = scores_arr[finite_mask], target_arr[finite_mask]
    n = int(scores_f.size)
    if n < min_sample_size:
        return DirectionCheckResult(status="skipped", reason="insufficient_labeled_sample", n=n, corr=None)
    corr = safe_correlation(scores_f, target_f)
    if abs(corr) < corr_threshold:
        return DirectionCheckResult(status="inconclusive", reason="corr_below_threshold", n=n, corr=corr)
    implied: ScoreDirection = "higher_is_riskier" if corr > 0 else "higher_is_better"
    if implied != declared_direction:
        return DirectionCheckResult(status="conflict", n=n, corr=corr, implied_direction=implied)
    return DirectionCheckResult(status="consistent", n=n, corr=corr, implied_direction=implied)


__all__ = [
    "CORR_CONFLICT_THRESHOLD",
    "MIN_CORR_SAMPLE_SIZE",
    "SCORE_DIRECTIONS",
    "DirectionCheckResult",
    "ScoreDirection",
    "check_score_direction",
    "normalize_score_direction",
]

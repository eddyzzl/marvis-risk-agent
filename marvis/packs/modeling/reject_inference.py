from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from marvis.data.direction import ScoreDirection, check_score_direction
from marvis.data.errors import ScoreDirectionConflictError
from marvis.packs.modeling.errors import ModelingError
from marvis.validation.binning import assign_bins, equal_frequency_bin_edges


INFERRED_TARGET_COL = "__reject_inference_target__"
SAMPLE_WEIGHT_COL = "__reject_inference_weight__"
SOURCE_COL = "__reject_inference_source__"
METHOD_COL = "__reject_inference_method__"

# S1a: default matches the pre-existing hard-coded behavior (_risk_order's
# argsort(-safe_scores) treats high score as high risk -> priority to be labeled bad).
_DEFAULT_REJECT_INFERENCE_DIRECTION: ScoreDirection = "higher_is_riskier"

_APPROVED_VALUES = frozenset({"1", "true", "yes", "y", "approved", "approve", "accepted", "pass", "通过", "同意"})
_REJECTED_VALUES = frozenset({"0", "false", "no", "n", "rejected", "reject", "declined", "deny", "拒绝", "未通过"})


@dataclass(frozen=True)
class RejectInferenceResult:
    frame: pd.DataFrame
    diagnostics: dict
    target_col: str
    sample_weight_col: str


def reject_inference(
    frame: pd.DataFrame,
    *,
    target_col: str,
    decision_col: str,
    method: str = "parceling",
    score_col: str | None = None,
    reject_bad_rate: float | None = None,
    reject_weight: float = 1.0,
    output_target_col: str = INFERRED_TARGET_COL,
    output_weight_col: str = SAMPLE_WEIGHT_COL,
    score_direction: ScoreDirection | None = None,
    confirm_direction_conflict: bool = False,
) -> RejectInferenceResult:
    """Controlled reject-inference MVP.

    Supported methods:
    - ``parceling``: assigns deterministic 0/1 inferred labels to rejected rows at the
      requested reject bad rate, ordered by ``score_col`` when supplied.
    - ``fuzzy_augmentation``: duplicates each rejected row into a bad and good row with
      fractional sample weights that sum to ``reject_weight``.

    This intentionally records assumptions in diagnostics and output columns; it is not
    a silent correction for selection bias.
    """
    if target_col not in frame.columns:
        raise ModelingError(f"target column not found: {target_col}")
    if decision_col not in frame.columns:
        raise ModelingError(f"decision column not found: {decision_col}")
    if score_col and score_col not in frame.columns:
        raise ModelingError(f"score column not found: {score_col}")
    method = str(method or "parceling").strip().lower()
    if method not in {"parceling", "fuzzy_augmentation"}:
        raise ModelingError(f"unsupported reject inference method: {method}")
    if reject_weight <= 0:
        raise ModelingError("reject_weight must be positive")

    accepted_mask, rejected_mask = _decision_masks(frame[decision_col], decision_col=decision_col)
    accepted = frame.loc[accepted_mask].copy()
    rejected = frame.loc[rejected_mask].copy()
    if accepted.empty:
        raise ModelingError("reject inference requires accepted rows with observed labels")
    if rejected.empty:
        raise ModelingError("reject inference requires rejected rows")
    labels = pd.to_numeric(accepted[target_col], errors="coerce")
    accepted = accepted.loc[labels.notna()].copy()
    labels = pd.to_numeric(accepted[target_col], errors="coerce")
    if accepted.empty:
        raise ModelingError("accepted rows have no observed labels")
    if set(labels.dropna().unique().tolist()) - {0, 1, 0.0, 1.0}:
        raise ModelingError("target must be binary 0/1 for reject inference")
    accepted_bad_rate = float(labels.mean())
    inferred_bad_rate = _resolve_reject_bad_rate(accepted_bad_rate, reject_bad_rate)

    accepted[output_target_col] = labels.astype(int).to_numpy()
    accepted[output_weight_col] = 1.0
    accepted[SOURCE_COL] = "accepted_observed"
    accepted[METHOD_COL] = method

    effective_direction = score_direction or _DEFAULT_REJECT_INFERENCE_DIRECTION
    direction_diagnostics = None
    if score_col:
        direction_diagnostics = _direction_self_check(
            accepted,
            score_col=score_col,
            output_target_col=output_target_col,
            declared_direction=effective_direction,
        )
        if direction_diagnostics.status == "conflict" and not confirm_direction_conflict:
            raise ScoreDirectionConflictError(
                tool="reject_inference",
                score_col=score_col,
                target_col=target_col,
                declared_direction=effective_direction,
                implied_direction=direction_diagnostics.implied_direction,
                corr=direction_diagnostics.corr,
                n_labeled=direction_diagnostics.n,
            )

    if method == "parceling":
        inferred = _parcel_rejected(
            rejected,
            score_col=score_col,
            bad_rate=inferred_bad_rate,
            weight=reject_weight,
            output_target_col=output_target_col,
            output_weight_col=output_weight_col,
            score_direction=effective_direction,
        )
    else:
        inferred = _fuzzy_augment_rejected(
            rejected,
            bad_rate=inferred_bad_rate,
            weight=reject_weight,
            output_target_col=output_target_col,
            output_weight_col=output_weight_col,
            score_col=score_col,
            accepted=accepted,
            accepted_target_col=output_target_col,
            accepted_bad_rate=accepted_bad_rate,
        )
    inferred[METHOD_COL] = method

    result = pd.concat([accepted, inferred], ignore_index=True, sort=False)
    diagnostics = {
        "method": method,
        "accepted_rows": int(accepted.shape[0]),
        "rejected_rows": int(rejected.shape[0]),
        "output_rows": int(result.shape[0]),
        "accepted_bad_rate": accepted_bad_rate,
        "reject_bad_rate_assumption": inferred_bad_rate,
        "reject_weight": float(reject_weight),
        "score_col": score_col,
        "target_col": output_target_col,
        "sample_weight_col": output_weight_col,
        "score_direction": effective_direction,
        "assumption": (
            "Rejected labels are inferred from a configured bad-rate assumption; "
            "use sensitivity analysis before business sign-off."
        ),
    }
    if direction_diagnostics is not None:
        diagnostics["direction_diagnostics"] = {
            "status": direction_diagnostics.status,
            "n": direction_diagnostics.n,
            "corr": direction_diagnostics.corr,
            "reason": direction_diagnostics.reason,
            "implied_direction": direction_diagnostics.implied_direction,
        }
    return RejectInferenceResult(
        frame=result,
        diagnostics=diagnostics,
        target_col=output_target_col,
        sample_weight_col=output_weight_col,
    )


def _decision_masks(series: pd.Series, *, decision_col: str) -> tuple[pd.Series, pd.Series]:
    text = series.astype("string").str.strip().str.lower()
    is_reject_flag = "reject" in decision_col.lower() or "declin" in decision_col.lower() or "拒" in decision_col
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        truthy = numeric == 1
        falsy = numeric == 0
        return (falsy, truthy) if is_reject_flag else (truthy, falsy)
    approved = text.isin(_APPROVED_VALUES)
    rejected = text.isin(_REJECTED_VALUES)
    return (rejected, approved) if is_reject_flag else (approved, rejected)


def _resolve_reject_bad_rate(accepted_bad_rate: float, reject_bad_rate: float | None) -> float:
    if reject_bad_rate is None:
        return min(0.95, max(accepted_bad_rate, accepted_bad_rate * 1.5))
    value = float(reject_bad_rate)
    if not 0.0 <= value <= 1.0:
        raise ModelingError("reject_bad_rate must be between 0 and 1")
    return value


def _parcel_rejected(
    frame: pd.DataFrame,
    *,
    score_col: str | None,
    bad_rate: float,
    weight: float,
    output_target_col: str,
    output_weight_col: str,
    score_direction: ScoreDirection | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    order = _risk_order(out, score_col, score_direction)
    bad_count = int(round(out.shape[0] * bad_rate))
    labels = np.zeros(out.shape[0], dtype=int)
    if bad_count > 0:
        labels[order[:bad_count]] = 1
    out[output_target_col] = labels
    out[output_weight_col] = float(weight)
    out[SOURCE_COL] = "rejected_inferred"
    return out


#: DOM-12: number of equal-frequency bins used to map accepted-population empirical
#: bad rates onto rejected rows by score when a score_col is available.
_FUZZY_SCORE_BIN_COUNT = 10


def _fuzzy_augment_rejected(
    frame: pd.DataFrame,
    *,
    bad_rate: float,
    weight: float,
    output_target_col: str,
    output_weight_col: str,
    score_col: str | None = None,
    accepted: pd.DataFrame | None = None,
    accepted_target_col: str | None = None,
    accepted_bad_rate: float | None = None,
) -> pd.DataFrame:
    per_row_bad_rate = _per_row_fuzzy_bad_rate(
        frame,
        fallback_bad_rate=bad_rate,
        score_col=score_col,
        accepted=accepted,
        accepted_target_col=accepted_target_col,
        accepted_bad_rate=accepted_bad_rate,
    )
    bad = frame.copy()
    good = frame.copy()
    bad[output_target_col] = 1
    good[output_target_col] = 0
    bad_weights = float(weight) * per_row_bad_rate
    good_weights = float(weight) * (1.0 - per_row_bad_rate)
    bad[output_weight_col] = bad_weights
    good[output_weight_col] = good_weights
    bad[SOURCE_COL] = "rejected_inferred_bad"
    good[SOURCE_COL] = "rejected_inferred_good"
    bad_mask = bad_weights > 0
    good_mask = good_weights > 0
    parts = []
    if bool(bad_mask.any()):
        parts.append(bad.loc[bad_mask])
    if bool(good_mask.any()):
        parts.append(good.loc[good_mask])
    if not parts:
        raise ModelingError("fuzzy reject inference produced no positive-weight rows")
    return pd.concat(parts, ignore_index=True, sort=False)


def _per_row_fuzzy_bad_rate(
    frame: pd.DataFrame,
    *,
    fallback_bad_rate: float,
    score_col: str | None,
    accepted: pd.DataFrame | None,
    accepted_target_col: str | None,
    accepted_bad_rate: float | None,
) -> np.ndarray:
    """Per-record PD for fuzzy augmentation (DOM-12).

    With a score column and enough accepted rows to bin, each rejected row gets the
    empirical bad rate of its equal-frequency score bin (computed on accepted rows),
    scaled so the population-weighted average matches ``fallback_bad_rate`` (the
    already-resolved bad-rate assumption -- either the observed accepted rate or an
    explicit ``reject_bad_rate`` override). This replaces the old single global rate
    applied uniformly to every rejected row. Falls back to the flat global rate when
    no score column is supplied (industry MVP default, unchanged).
    """
    n = frame.shape[0]
    if (
        not score_col
        or accepted is None
        or accepted_target_col is None
        or accepted_bad_rate is None
        or score_col not in frame.columns
        or score_col not in accepted.columns
        or accepted.shape[0] < _FUZZY_SCORE_BIN_COUNT
    ):
        return np.full(n, float(fallback_bad_rate), dtype=float)

    accepted_scores = pd.to_numeric(accepted[score_col], errors="coerce").to_numpy(dtype=float)
    accepted_labels = accepted[accepted_target_col].to_numpy(dtype=float)
    valid = np.isfinite(accepted_scores)
    if int(valid.sum()) < _FUZZY_SCORE_BIN_COUNT:
        return np.full(n, float(fallback_bad_rate), dtype=float)

    edges = equal_frequency_bin_edges(accepted_scores[valid], _FUZZY_SCORE_BIN_COUNT)
    accepted_bins = assign_bins(accepted_scores, edges)
    bin_count = len(edges) - 1
    bin_bad_rate = np.full(bin_count + 1, float(fallback_bad_rate), dtype=float)
    for bin_index in range(1, bin_count + 1):
        mask = accepted_bins == bin_index
        if mask.any():
            bin_bad_rate[bin_index] = float(accepted_labels[mask].mean())

    # Anchor the binned rates to the resolved bad-rate assumption: when an explicit
    # reject_bad_rate overrides the raw accepted rate, scale every bin proportionally
    # so the (accepted-weighted) average still lands on fallback_bad_rate, then clip
    # back into [0, 1] (deterministic, no randomness -- INV-1).
    if accepted_bad_rate and accepted_bad_rate > 0:
        scale = float(fallback_bad_rate) / float(accepted_bad_rate)
        bin_bad_rate = np.clip(bin_bad_rate * scale, 0.0, 1.0)
    else:
        bin_bad_rate = np.clip(bin_bad_rate, 0.0, 1.0)

    rejected_scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
    rejected_bins = assign_bins(rejected_scores, edges)
    per_row = np.where(
        rejected_bins > 0,
        bin_bad_rate[np.clip(rejected_bins, 0, bin_count)],
        float(fallback_bad_rate),
    )
    return per_row


def _risk_order(
    frame: pd.DataFrame,
    score_col: str | None,
    score_direction: ScoreDirection | None = None,
) -> np.ndarray:
    if not score_col:
        return np.arange(frame.shape[0])
    effective_direction = score_direction or _DEFAULT_REJECT_INFERENCE_DIRECTION
    scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
    if effective_direction == "higher_is_riskier":
        safe_scores = np.where(np.isfinite(scores), scores, -math.inf)
        return np.argsort(-safe_scores, kind="mergesort")  # current behavior, unchanged
    # higher_is_better -> low score is high risk, prioritize low scores for bad labels
    safe_scores = np.where(np.isfinite(scores), scores, math.inf)
    return np.argsort(safe_scores, kind="mergesort")


def _direction_self_check(
    accepted: pd.DataFrame,
    *,
    score_col: str,
    output_target_col: str,
    declared_direction: ScoreDirection,
):
    scores = pd.to_numeric(accepted[score_col], errors="coerce").to_numpy(dtype=float)
    target = accepted[output_target_col].to_numpy(dtype=float)
    return check_score_direction(scores, target, declared_direction=declared_direction)


__all__ = [
    "INFERRED_TARGET_COL",
    "METHOD_COL",
    "RejectInferenceResult",
    "SAMPLE_WEIGHT_COL",
    "SOURCE_COL",
    "reject_inference",
]

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from marvis.packs.modeling.errors import ModelingError


INFERRED_TARGET_COL = "__reject_inference_target__"
SAMPLE_WEIGHT_COL = "__reject_inference_weight__"
SOURCE_COL = "__reject_inference_source__"
METHOD_COL = "__reject_inference_method__"

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

    if method == "parceling":
        inferred = _parcel_rejected(
            rejected,
            score_col=score_col,
            bad_rate=inferred_bad_rate,
            weight=reject_weight,
            output_target_col=output_target_col,
            output_weight_col=output_weight_col,
        )
    else:
        inferred = _fuzzy_augment_rejected(
            rejected,
            bad_rate=inferred_bad_rate,
            weight=reject_weight,
            output_target_col=output_target_col,
            output_weight_col=output_weight_col,
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
        "assumption": (
            "Rejected labels are inferred from a configured bad-rate assumption; "
            "use sensitivity analysis before business sign-off."
        ),
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
) -> pd.DataFrame:
    out = frame.copy()
    order = _risk_order(out, score_col)
    bad_count = int(round(out.shape[0] * bad_rate))
    labels = np.zeros(out.shape[0], dtype=int)
    if bad_count > 0:
        labels[order[:bad_count]] = 1
    out[output_target_col] = labels
    out[output_weight_col] = float(weight)
    out[SOURCE_COL] = "rejected_inferred"
    return out


def _fuzzy_augment_rejected(
    frame: pd.DataFrame,
    *,
    bad_rate: float,
    weight: float,
    output_target_col: str,
    output_weight_col: str,
) -> pd.DataFrame:
    bad = frame.copy()
    good = frame.copy()
    bad[output_target_col] = 1
    good[output_target_col] = 0
    bad[output_weight_col] = float(weight) * bad_rate
    good[output_weight_col] = float(weight) * (1.0 - bad_rate)
    bad[SOURCE_COL] = "rejected_inferred_bad"
    good[SOURCE_COL] = "rejected_inferred_good"
    parts = []
    if float(weight) * bad_rate > 0:
        parts.append(bad)
    if float(weight) * (1.0 - bad_rate) > 0:
        parts.append(good)
    if not parts:
        raise ModelingError("fuzzy reject inference produced no positive-weight rows")
    return pd.concat(parts, ignore_index=True, sort=False)


def _risk_order(frame: pd.DataFrame, score_col: str | None) -> np.ndarray:
    if not score_col:
        return np.arange(frame.shape[0])
    scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
    safe_scores = np.where(np.isfinite(scores), scores, -math.inf)
    return np.argsort(-safe_scores, kind="mergesort")


__all__ = [
    "INFERRED_TARGET_COL",
    "METHOD_COL",
    "RejectInferenceResult",
    "SAMPLE_WEIGHT_COL",
    "SOURCE_COL",
    "reject_inference",
]

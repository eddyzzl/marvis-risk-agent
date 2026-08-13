"""Segment-level observational evidence built on an applied Strategy Pool result.

This module is deliberately *evidence-only*.  Given the apply output produced by
``marvis.packs.strategy.pool_apply`` (action / value / rule_id / entry_id /
reason_code columns) plus a segment column and a target column, it reports the
observed counts and rates (件数、通过率、坏率、拒绝原因分布) for each segment and
for the overall population.

Hard constraint: this module never computes or emits a fairness verdict.  No code
path labels a segment as "fair" or "unfair", and no such wording is stored on the
returned objects -- only observed numbers and neutral column / segment labels.
Any human-facing comparative phrasing must come from the draft templates in
:data:`NEUTRAL_PHRASING_TEMPLATES`, which are explicitly marked as drafts pending
human compliance review and are never attached to the evidence as a compliance
claim.

Definitions (mirroring the approval backtest conventions):

* 通过率 (approval rate) = rows whose action equals ``"approval"`` / row count.
* 坏率 (bad rate) = rows whose target equals ``1`` / labelled (non-missing) rows;
  ``None`` when a group has no labelled rows (never a fabricated ``0.0``).
* 拒绝原因分布 = reason-code counts over rows whose action equals ``"reject"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from marvis.packs.strategy.errors import StrategyError

_APPROVE_ACTION = "approval"
_REJECT_ACTION = "reject"

#: Reserved, documented label for the overall (full-population) evidence row.
OVERALL_SEGMENT = "__overall__"
#: Deterministic label used for missing segment values and missing reason codes.
MISSING_LABEL = "<missing>"

NEUTRAL_PHRASING_DRAFT_NOTICE = (
    "草稿：以下措辞模板仅供人工起草参考，尚未经过合规审阅，"
    "不得作为对外合规结论使用。"
)

NEUTRAL_PHRASING_TEMPLATES: tuple[str, ...] = (
    "分群「{segment}」的通过率为 {approval_rate:.1%}，"
    "总体通过率为 {overall_approval_rate:.1%}，"
    "二者相差 {approval_delta_pp:+.1f} 个百分点。",
    "分群「{segment}」的坏率为 {bad_rate:.1%}，"
    "总体坏率为 {overall_bad_rate:.1%}，"
    "二者相差 {bad_delta_pp:+.1f} 个百分点。",
    "分群「{segment}」的件数为 {count}，占总体件数的 {count_share:.1%}。",
    "分群「{segment}」的拒绝原因分布为 {reason_code_distribution}。",
)


class FairnessEvidenceError(StrategyError):
    """Typed fail-closed error for fairness-evidence input validation."""

    code = "fairness_evidence_invalid_input"


@dataclass(frozen=True)
class SegmentEvidence:
    """Observed statistics for one segment or the overall population."""

    segment: str
    count: int
    approved_count: int
    approval_rate: float
    bad_count: int
    labelled_count: int
    bad_rate: float | None
    reason_code_distribution: Mapping[str, int] | None = None


@dataclass(frozen=True)
class SegmentDisparityEvidence:
    """Deterministic per-segment and overall observational evidence."""

    segment_column: str
    action_column: str
    reason_code_column: str | None
    target_column: str
    overall: SegmentEvidence
    segments: tuple[SegmentEvidence, ...]


def segment_disparity_evidence(
    rows_with_segment_column: pd.DataFrame,
    segment_column: str,
    outcome_columns: Mapping[str, str],
    target_col: str,
) -> SegmentDisparityEvidence:
    """Compute segment-level observational evidence over an applied Pool result.

    ``rows_with_segment_column`` is a DataFrame that already carries both the
    apply output columns (action, and optionally reason_code) and a segment
    column plus the target column.  ``outcome_columns`` maps semantic names to
    the actual column names; it must provide ``"action"`` and may provide
    ``"reason_code"`` (skipped when absent).  A
    :class:`~marvis.packs.strategy.pool_apply.StrategyPoolApplyResult`'s
    ``output_columns`` mapping can be passed directly.

    Missing segment / target / outcome columns fail closed with
    :class:`FairnessEvidenceError`.  The result is deterministic and side-effect
    free; segment rows are sorted by their neutral string label.
    """

    if not isinstance(rows_with_segment_column, pd.DataFrame):
        raise FairnessEvidenceError(
            "fairness evidence requires a DataFrame input"
        )
    if not isinstance(segment_column, str) or not segment_column:
        raise FairnessEvidenceError("segment_column must be a non-empty string")
    if not isinstance(target_col, str) or not target_col:
        raise FairnessEvidenceError("target_col must be a non-empty string")
    if not isinstance(outcome_columns, Mapping):
        raise FairnessEvidenceError("outcome_columns must be a mapping")

    frame = rows_with_segment_column
    _require_column(frame, segment_column, "segment")
    _require_column(frame, target_col, "target")

    action_column = _outcome_column(outcome_columns, "action", required=True)
    _require_column(frame, action_column, "outcome action")
    reason_code_column = _outcome_column(
        outcome_columns, "reason_code", required=False
    )
    if reason_code_column is not None:
        _require_column(frame, reason_code_column, "outcome reason_code")

    action = frame[action_column]
    target = _target_series(frame, target_col)
    reason_codes = (
        None if reason_code_column is None else frame[reason_code_column]
    )

    overall = _build_evidence(
        segment=OVERALL_SEGMENT,
        action=action,
        target=target,
        reason_codes=reason_codes,
    )

    segment_rows: list[SegmentEvidence] = []
    for segment_value, group in frame.groupby(
        segment_column, sort=False, dropna=False
    ):
        index = group.index
        segment_rows.append(
            _build_evidence(
                segment=_label(segment_value),
                action=action.loc[index],
                target=target.loc[index],
                reason_codes=(
                    None if reason_codes is None else reason_codes.loc[index]
                ),
            )
        )
    segment_rows.sort(key=lambda row: row.segment)

    return SegmentDisparityEvidence(
        segment_column=segment_column,
        action_column=action_column,
        reason_code_column=reason_code_column,
        target_column=target_col,
        overall=overall,
        segments=tuple(segment_rows),
    )


def _build_evidence(
    *,
    segment: str,
    action: pd.Series,
    target: pd.Series,
    reason_codes: pd.Series | None,
) -> SegmentEvidence:
    count = int(len(action))
    approved_count = int(action.eq(_APPROVE_ACTION).sum())
    labelled_count = int(target.notna().sum())
    bad_count = int(target.eq(1).sum())
    distribution = None
    if reason_codes is not None:
        rejected = action.eq(_REJECT_ACTION)
        distribution = MappingProxyType(
            _reason_code_distribution(reason_codes.loc[rejected])
        )
    return SegmentEvidence(
        segment=segment,
        count=count,
        approved_count=approved_count,
        approval_rate=_ratio(approved_count, count),
        bad_count=bad_count,
        labelled_count=labelled_count,
        bad_rate=(
            None if labelled_count == 0 else float(bad_count) / float(labelled_count)
        ),
        reason_code_distribution=distribution,
    )


def _reason_code_distribution(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).to_dict()
    labelled = {_label(key): int(count) for key, count in counts.items()}
    return dict(sorted(labelled.items()))


def _target_series(frame: pd.DataFrame, target_col: str) -> pd.Series:
    try:
        return pd.to_numeric(frame[target_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise FairnessEvidenceError(
            f"target column {target_col!r} must be numeric"
        ) from exc


def _require_column(frame: pd.DataFrame, column: str, role: str) -> None:
    if column not in frame.columns:
        raise FairnessEvidenceError(f"missing {role} column: {column!r}")


def _outcome_column(
    outcome_columns: Mapping[str, str],
    key: str,
    *,
    required: bool,
) -> str | None:
    value = outcome_columns.get(key)
    if value is None:
        if required:
            raise FairnessEvidenceError(
                f"outcome_columns must provide a {key!r} column"
            )
        return None
    if not isinstance(value, str) or not value:
        raise FairnessEvidenceError(
            f"outcome_columns[{key!r}] must be a non-empty string"
        )
    return value


def _label(value: Any) -> str:
    return MISSING_LABEL if _is_missing(value) else str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _ratio(numerator: int | float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


__all__ = [
    "FairnessEvidenceError",
    "MISSING_LABEL",
    "NEUTRAL_PHRASING_DRAFT_NOTICE",
    "NEUTRAL_PHRASING_TEMPLATES",
    "OVERALL_SEGMENT",
    "SegmentDisparityEvidence",
    "SegmentEvidence",
    "segment_disparity_evidence",
]

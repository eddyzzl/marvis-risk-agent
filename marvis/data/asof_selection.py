"""Bounded deterministic as-of selection over already authenticated frames."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral

import pandas as pd

from marvis.data.time_contracts import (
    AsOfJoinSpec, Assurance, DatasetTimeContract, TimeColumn, temporal_assurance,
)
from marvis.decision_twin._canonical import content_hash
from marvis.decision_twin.replay import validate_point_in_time_visibility


@dataclass(frozen=True)
class AsOfSelection:
    frame: pd.DataFrame
    memberships: tuple[tuple[int, int | None], ...]
    assurance: Assurance
    reasons: tuple[str, ...]

    @property
    def membership_sha256(self) -> str:
        return content_hash(self.memberships)


def _times(frame: pd.DataFrame, column: TimeColumn) -> list[pd.Timestamp]:
    result = []
    for position, value in enumerate(frame[column.column]):
        if not isinstance(value, (str, datetime, pd.Timestamp)) or pd.isna(value):
            raise ValueError(f"invalid timestamp in {column.column} at row {position}")
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError("missing timestamp")
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(column.timezone, ambiguous="raise", nonexistent="raise")
            result.append(timestamp.tz_convert("UTC"))
        except Exception as exc:
            raise ValueError(f"invalid/ambiguous timestamp in {column.column} at row {position}") from exc
    return result


def _keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple]:
    if not columns:
        return [()] * len(frame)
    result = []
    for row in frame[list(columns)].itertuples(index=False, name=None):
        key = []
        for value in row:
            # Exact identifiers only: bool, float and coercion can silently join different subjects.
            if isinstance(value, bool) or not isinstance(value, (str, Integral)) or pd.isna(value):
                raise ValueError("entity/partition/row identifiers must be non-null strings or integers")
            if isinstance(value, str) and not value.strip():
                raise ValueError("entity/partition/row identifiers must not be blank")
            key.append(("int", int(value)) if isinstance(value, Integral) else ("str", value))
        result.append(tuple(key))
    return result


def _check_frame(frame, contract, max_rows):
    if len(frame) > max_rows:
        raise ValueError("source row budget exceeded")
    if not frame.columns.is_unique:
        raise ValueError("duplicate column names")
    required = {contract.row_id, *contract.entity_columns}
    required.update(item.column for item in contract.time_columns())
    if contract.version_column:
        required.add(contract.version_column)
    if not required.issubset(frame.columns):
        raise ValueError("time contract columns are missing")
    ids = _keys(frame, (contract.row_id,))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row identity")


def select_asof_rows(
    decisions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    decision_contract: DatasetTimeContract,
    feature_contract: DatasetTimeContract,
    spec: AsOfJoinSpec,
) -> AsOfSelection:
    """Latest visible event, then highest visible version; intervals are [from,to).

    Future/late versions are excluded, never date-rounded or silently deduplicated.
    Missing availability can be explored only with an explicit unknown assurance.
    Repeated subjects at different decisions are retained; identical subject/time
    anchors and ambiguous revision identities are rejected. No fitting occurs here.
    """
    dc, fc = decision_contract, feature_contract
    if decisions_df.empty:
        raise ValueError("decision cohort must not be empty")
    if dc.role != "decision" or fc.role != "feature_snapshot":
        raise ValueError("as-of selection requires decision and feature_snapshot roles")
    if len(dc.entity_columns) != len(fc.entity_columns):
        raise ValueError("entity key arity differs")
    for frame, contract in ((decisions_df, dc), (features_df, fc)):
        _check_frame(frame, contract, spec.max_source_rows)
    status = temporal_assurance(dc, fc)
    if spec.mode == "verified" and status.assurance != "verified":
        raise ValueError("verified mode requires recorded historical temporal evidence")
    if not set(spec.feature_columns).issubset(features_df.columns):
        raise ValueError("feature columns are missing")
    names = [spec.feature_prefix + name for name in spec.feature_columns]
    if set(names) & set(decisions_df.columns):
        raise ValueError("output feature column collision")
    for left, right in spec.partition_pairs:
        if left not in decisions_df or right not in features_df:
            raise ValueError("partition column is missing")
    dkeys = _keys(decisions_df, dc.entity_columns)
    fkeys = _keys(features_df, fc.entity_columns)
    dparts = _keys(decisions_df, tuple(pair[0] for pair in spec.partition_pairs))
    fparts = _keys(features_df, tuple(pair[1] for pair in spec.partition_pairs))
    dt = _times(decisions_df, dc.decision_at)
    et = _times(features_df, fc.event_at)
    at = _times(features_df, fc.available_at) if fc.available_at else None
    start = _times(features_df, fc.effective_from) if fc.effective_from else None
    end = _times(features_df, fc.effective_to) if fc.effective_to else None
    ws = _times(features_df, fc.window_start) if fc.window_start else None
    we = _times(features_df, fc.window_end) if fc.window_end else None
    versions = list(features_df[fc.version_column])
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in versions):
        raise ValueError("versions must be positive integers")
    if len(set(zip(dkeys, dt, strict=True))) != len(dt):
        raise ValueError("duplicate entity/decision_at anchor")
    groups = {}
    revision_ids = set()
    for i, key in enumerate(fkeys):
        identity = (key, et[i], int(versions[i]))
        if identity in revision_ids:
            raise ValueError("duplicate entity/event/version identity")
        revision_ids.add(identity)
        if at is not None and at[i] < et[i]:
            raise ValueError("available_at precedes event_at")
        if start is not None and start[i] >= end[i]:
            raise ValueError("effective interval must have from < to")
        if ws is not None and (ws[i] > we[i] or we[i] > et[i]):
            raise ValueError("aggregate window crosses event_at")
        groups.setdefault(key, []).append(i)
    for rows in groups.values():
        rows.sort(key=lambda i: (et[i], int(versions[i])))
        if at is not None:
            for prev, nxt in zip(rows, rows[1:]):
                if et[prev] == et[nxt] and at[prev] > at[nxt]:
                    raise ValueError("revision availability contradicts version order")
    event_index = {key: [et[i] for i in rows] for key, rows in groups.items()}
    memberships = []
    checks = 0
    for i, decision_at in enumerate(dt):
        validate_point_in_time_visibility(decision_at=decision_at, as_of=spec.as_of, visible_fields=())
        key = dkeys[i]
        rows = groups.get(key, [])
        upper = bisect_right(event_index.get(key, []), decision_at)
        selected = None
        for position in range(upper - 1, -1, -1):
            j = rows[position]
            checks += 1
            if checks > spec.max_candidate_checks:
                raise ValueError("candidate check budget exceeded")
            if spec.lookback_seconds is not None and et[j] < decision_at - pd.Timedelta(seconds=spec.lookback_seconds):
                break
            if at is not None and at[j] > decision_at:
                continue
            # The latest visible snapshot supersedes older revisions. A partition
            # or interval mismatch must not resurrect a stale older version.
            if fparts[j] != dparts[i]:
                break
            if start is not None and not start[j] <= decision_at < end[j]:
                break
            if ws is not None and spec.lookback_seconds is not None:
                if ws[j] < decision_at - pd.Timedelta(seconds=spec.lookback_seconds):
                    break
            validate_point_in_time_visibility(
                decision_at=decision_at, as_of=spec.as_of,
                visible_fields=((fc.event_at.column, et[j]),) + (((fc.available_at.column, at[j]),) if at is not None else ()),
            )
            selected = j
            break
        if selected is None and spec.require_match:
            raise ValueError(f"no eligible as-of feature for decision row {i}")
        memberships.append((i, selected))
    output = decisions_df.reset_index(drop=True).copy(deep=True)
    # Reindex preserves source feature dtypes where possible; -1 is never a row position.
    selected_features = features_df[list(spec.feature_columns)].reset_index(drop=True).reindex(
        [j if j is not None else -1 for _, j in memberships],
    ).reset_index(drop=True)
    selected_features.columns = names
    output = pd.concat([output, selected_features], axis=1)
    return AsOfSelection(output, tuple(memberships), status.assurance, status.reasons)

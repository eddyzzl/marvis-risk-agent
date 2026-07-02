"""Two-level deduplication (join spec §6).

The sample must stay 1:1 after a join, so a feature table whose key is not unique has
to be deduplicated — but NOT blindly. The spec mandates two levels:

* **Level 1 — safe dedup:** drop rows that are identical across the WHOLE row (key plus
  every value). This is lossless and applied automatically.
* **Level 2 — conflict detection:** after level 1, if a key still maps to more than one
  row, those rows disagree on some value (同人同天、特征值不一致). This is a data-quality
  red flag: it is REPORTED (which keys, which columns), never silently dropped. The
  caller decides whether to resolve it (deterministically) or fix the upstream data.

Everything here is deterministic — ``drop_duplicates(keep="first")`` and ``groupby``
preserve row order — so repeated runs produce identical output and reports.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from marvis.data.contracts import ConflictReport

_SAMPLE_CAP = 50


def two_level_dedup(frame: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, ConflictReport]:
    """Apply level-1 safe dedup and detect (do not drop) level-2 same-key conflicts.

    Returns the level-1-deduplicated frame (conflicts still present) and a
    :class:`ConflictReport`. Resolving a reported conflict is an explicit, separate step.
    """
    before = len(frame)
    # Level 1: identical whole rows are pure duplicates → drop, keeping the first.
    deduped = frame.drop_duplicates(keep="first", ignore_index=True)
    safe_dropped = before - len(deduped)

    keys = [str(col) for col in key_cols]
    if not keys:
        return deduped, _empty_report((), safe_dropped)

    # Level 2: only among FULLY-keyed rows — a row with a missing key can't match the
    # anchor in a 1:1 join, so unkeyable rows are not "the same entity disagreeing" and
    # must not be folded into a conflict. Among keyed rows, any key with >1 surviving
    # row has DISAGREEING values → conflict.
    keyed = deduped.dropna(subset=keys)
    duplicated = keyed.duplicated(subset=keys, keep=False)
    conflicts = keyed[duplicated]
    if conflicts.empty:
        return deduped, _empty_report(tuple(keys), safe_dropped)

    grouped = conflicts.groupby(keys, sort=False, dropna=False)
    value_cols = [col for col in deduped.columns if col not in keys]
    conflict_columns = tuple(
        col for col in value_cols if bool(grouped[col].nunique(dropna=False).gt(1).any())
    )
    distinct_keys = conflicts[keys].drop_duplicates().head(_SAMPLE_CAP)
    sample_keys = tuple(
        tuple(_scalar(value) for value in row)
        for row in distinct_keys.itertuples(index=False, name=None)
    )
    # UX-6: for each sampled key (same order/cap as sample_keys), the actual row values
    # the conflict columns took -- lets a dedup gate show "k=123 时 amount 两行分别为
    # 500/800" instead of a bare conflict count. `grouped` was built from `conflicts`
    # (already dropna(subset=keys)-filtered above), so every sampled key is a real,
    # non-NaN group -- get_group cannot KeyError here. groupby(keys, ...) with `keys` a
    # list always keys get_group by tuple, even for a single-column key.
    sample_conflicts = tuple(
        _conflict_sample(grouped.get_group(row), conflict_columns)
        for row in distinct_keys.itertuples(index=False, name=None)
    )
    report = ConflictReport(
        key_columns=tuple(keys),
        conflict_columns=conflict_columns,
        n_conflict_keys=int(grouped.ngroups),
        n_conflict_rows=int(len(conflicts)),
        safe_dropped=int(safe_dropped),
        sample_keys=sample_keys,
        sample_conflicts=sample_conflicts,
    )
    return deduped, report


def _empty_report(keys: tuple[str, ...], safe_dropped: int) -> ConflictReport:
    return ConflictReport(
        key_columns=keys,
        conflict_columns=(),
        n_conflict_keys=0,
        n_conflict_rows=0,
        safe_dropped=int(safe_dropped),
        sample_keys=(),
    )


def _conflict_sample(rows: pd.DataFrame, conflict_columns: tuple[str, ...]) -> dict:
    """The conflict_columns values a single conflicting key's rows actually took, capped
    to avoid a pathological same-key conflict (many disagreeing rows) bloating the
    payload."""
    return {
        str(col): [_scalar(value) for value in rows[col].head(_SAMPLE_CAP).tolist()]
        for col in conflict_columns
    }


def _scalar(value):
    """Coerce a key value to a JSON-safe Python scalar.

    ``sample_keys`` is serialized to the driver/DB and out through Starlette's
    ``JSONResponse(allow_nan=False)``, so this must never return a numpy type, a
    ``Timestamp`` (the spec's 同人同天 key is a DATE column!), a non-finite float, or any
    other non-JSON value. Unknown types fall back to ``str`` rather than leaking through.
    """
    if isinstance(value, (np.ndarray, list, tuple, dict, set)):
        return str(value)  # array-like cell → never a scalar key; stringify
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)  # final fallback: guarantee a JSON-safe scalar


__all__ = ["two_level_dedup"]

"""Shared authenticated source windows for installed periodic analysis."""

from datetime import timedelta

import pandas as pd

from marvis.operations.monitoring import MonitoringExecutionError


def source_window(executor, binding, period, *, lookback_seconds=0):
    try:
        authenticated = executor.registry.authenticate_dataset_binding(
            binding.dataset_id,
            expected_task_id=binding.task_id,
            expected_content_hash=binding.dataset_content_hash,
        )
        if authenticated.row_count > binding.max_source_rows:
            raise ValueError("source row budget")
        frame = executor.registry.read_authenticated_binding_snapshot(authenticated)
        if binding.time_col not in frame:
            raise ValueError("declared time field absent")
        times = pd.to_datetime(frame[binding.time_col], errors="raise", format="mixed")
        if times.isna().any():
            raise ValueError("unknown source event time")
        if times.dt.tz is None:
            times = times.dt.tz_localize(
                binding.timezone, ambiguous="raise", nonexistent="raise"
            )
        times = times.dt.tz_convert("UTC")
    except Exception as exc:
        raise MonitoringExecutionError("monitoring_source_error") from exc
    start = period.starts_at - timedelta(seconds=lookback_seconds)
    window = frame.loc[(times >= start) & (times < period.ends_at)].copy()
    current = (times >= period.starts_at) & (times < period.ends_at)
    reason = (
        "no_new_data"
        if not current.any()
        else "window_incomplete"
        if binding.complete_through < period.ends_at
        else "future_source_watermark"
        if binding.complete_through > executor.clock()
        else None
    )
    return (
        authenticated,
        window,
        reason,
        {
            "source_dataset_id": binding.dataset_id,
            "source_content_hash": binding.dataset_content_hash,
            "window_starts_at": start.isoformat(),
            "window_ends_at": period.ends_at.isoformat(),
            "window_row_count": len(window),
            "current_window_row_count": int(current.sum()),
            "coverage_assurance": "publisher_declared",
            "declared_complete_through": binding.complete_through.isoformat(),
            "business_time_and_maturity_assurance": "not_independently_established",
        },
    )

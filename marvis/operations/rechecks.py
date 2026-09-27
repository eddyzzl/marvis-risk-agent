"""Explicit one-window reassessment using existing schedule/lease coordination."""

from __future__ import annotations

from dataclasses import replace
import hashlib

from marvis.operations.bindings import RecheckSource
from marvis.operations.contracts import FixedIntervalCalendar
from marvis.operations.repository import ScheduleRevisionConflict


def request_recheck(
    runtime, schedule_id, *, period_key, expected_revision, idempotency_key, binding
):
    store = runtime.store
    latest = store.get_schedule(schedule_id)
    if latest is None:
        raise KeyError(schedule_id)
    if latest.contract.revision != expected_revision:
        raise ScheduleRevisionConflict("source schedule revision changed")
    period = store.get_period(schedule_id, period_key)
    if period is None:
        raise KeyError(period_key)
    if period.state not in {"succeeded", "terminal_failed"}:
        raise ScheduleRevisionConflict("original period is still running or retrying")
    original = store.get_schedule(
        schedule_id, revision=period.schedule_revision
    ).contract
    source = original.monitoring_binding
    if (
        source is None
        or binding.task_id != source.task_id
        or binding.target_id != source.target_id
    ):
        raise ValueError("recheck must retain original task and monitored target")
    identity = hashlib.sha256(
        f"{schedule_id}\0{period_key}\0{idempotency_key}".encode()
    ).hexdigest()
    contract = replace(
        original,
        schedule_id=f"recheck_{identity}",
        revision=1,
        active_from=period.period.starts_at,
        active_until=period.period.ends_at,
        calendar=FixedIntervalCalendar(
            anchor_at=period.period.starts_at,
            interval_seconds=int(
                (period.period.ends_at - period.period.starts_at).total_seconds()
            ),
        ),
        catch_up_budget=1,
        enabled=True,
        monitoring_binding=binding,
        recheck_of=RecheckSource(
            schedule_id=schedule_id,
            period_key=period_key,
            schedule_revision=expected_revision,
        ),
    )
    existing = store.get_schedule(contract.schedule_id)
    if existing is not None:
        if existing.contract != contract:
            raise ScheduleRevisionConflict(
                "idempotency key was used with different recheck inputs"
            )
        return existing
    try:
        return runtime.publish_schedule(contract, expected_revision=0)
    except ScheduleRevisionConflict:
        existing = store.get_schedule(contract.schedule_id)
        if existing is not None and existing.contract == contract:
            return existing
        raise

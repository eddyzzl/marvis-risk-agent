from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from marvis.operations.contracts import (
    FixedIntervalCalendar,
    MonitoringOutcome,
    RetryPolicy,
    ScheduleContract,
)
from marvis.operations.integration import (
    OperationsRuntime,
    UnknownMonitoringReference,
)
from marvis.settings import build_settings


FIXED_NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _schedule(*, monitoring_ref: str, schedule_id: str = "schedule-one") -> ScheduleContract:
    anchor = FIXED_NOW - timedelta(minutes=2)
    return ScheduleContract(
        schedule_id=schedule_id,
        revision=1,
        monitoring_ref=monitoring_ref,
        active_from=anchor,
        calendar=FixedIntervalCalendar(anchor_at=anchor, interval_seconds=60),
        catch_up_budget=1,
        lease_seconds=30,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            multiplier=1,
            max_backoff_seconds=0,
        ),
    )


def test_runtime_installs_store_on_settings_db_and_freezes_exact_allowlist(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    calls = []

    def execute(request):
        calls.append(request)
        return MonitoringOutcome(
            level="green",
            evidence_ref=f"local-monitoring://{request.run_id}",
            evidence_hash="a" * 64,
        )

    allowlist = {"bound-monitor-one": execute}
    runtime = OperationsRuntime(
        settings,
        executor_allowlist=allowlist,
        owner_id="test-runtime",
        clock=lambda: FIXED_NOW,
    )
    allowlist["added-after-construction"] = execute

    assert runtime.store.db_path == settings.db_path
    assert runtime.allowed_monitoring_refs == ("bound-monitor-one",)

    runtime.publish_schedule(
        _schedule(monitoring_ref="bound-monitor-one"),
        expected_revision=0,
    )
    report = runtime.tick(catch_up_budget=1, notification_budget=0)

    assert report.succeeded == 1
    assert len(calls) == 1
    assert calls[0].monitoring_ref == "bound-monitor-one"

    with pytest.raises(UnknownMonitoringReference):
        runtime.publish_schedule(
            _schedule(
                monitoring_ref="added-after-construction",
                schedule_id="late-binding",
            ),
            expected_revision=0,
        )
    assert runtime.store.get_schedule("late-binding") is None


@pytest.mark.parametrize(
    "malicious_ref",
    [
        "os.system",
        "attacker_module:run",
        "../../plugins/evil.py",
    ],
)
def test_runtime_never_resolves_import_or_callable_syntax(tmp_path, malicious_ref):
    runtime = OperationsRuntime(
        build_settings(tmp_path / "workspace"),
        executor_allowlist={},
        owner_id="test-runtime",
        clock=lambda: FIXED_NOW,
    )

    with pytest.raises(UnknownMonitoringReference):
        runtime.publish_schedule(
            _schedule(monitoring_ref=malicious_ref),
            expected_revision=0,
        )


def test_manual_recovery_reports_expired_period_and_notification_leases(tmp_path):
    current = [FIXED_NOW]
    runtime = OperationsRuntime(
        build_settings(tmp_path / "workspace"),
        executor_allowlist={"bound-monitor": lambda _request: None},
        owner_id="test-runtime",
        clock=lambda: current[0],
    )
    schedule = _schedule(monitoring_ref="bound-monitor")
    runtime.publish_schedule(schedule, expected_revision=0)
    claim = runtime.store.claim_period(
        schedule,
        schedule.calendar.period(0),
        owner_id="crashed-worker",
    )
    assert claim is not None
    current[0] = FIXED_NOW + timedelta(seconds=31)

    recovery = runtime.recover()

    assert recovery.period_leases_recovered == 1
    assert recovery.notification_leases_recovered == 0
    assert recovery.period_keys == (schedule.calendar.period(0).key,)
    assert recovery.notification_ids == ()

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from marvis.db import init_db, sqlite_health
from marvis.operations import (
    FixedIntervalCalendar,
    OperationsStore,
    MonitoringOutcome,
    RetryPolicy,
    ScheduleContract,
    ScheduleRevisionConflict,
    StaleRunLeaseError,
)


NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)


class _Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _schedule(*, revision: int = 1) -> ScheduleContract:
    return ScheduleContract(
        schedule_id="monitor-score-v1",
        revision=revision,
        monitoring_ref="monitoring-plan-private-42",
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        calendar=FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=3600,
        ),
        catch_up_budget=3,
        lease_seconds=30,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=5,
            multiplier=2,
            max_backoff_seconds=20,
        ),
    )


def test_store_installs_idempotently_and_appends_schedule_revisions_with_cas(tmp_path):
    db_path = tmp_path / "marvis.sqlite"
    store = OperationsStore(db_path, clock=lambda: NOW)

    first = store.publish_schedule(_schedule(), expected_revision=0)
    second_store = OperationsStore(db_path, clock=lambda: NOW)
    second = second_store.publish_schedule(
        replace(_schedule(revision=2), catch_up_budget=2),
        expected_revision=1,
    )

    assert first.contract.revision == 1
    assert second.contract.revision == 2
    assert [record.contract.revision for record in store.list_schedule_revisions(
        "monitor-score-v1"
    )] == [1, 2]
    with pytest.raises(ScheduleRevisionConflict, match="expected 1, found 2"):
        store.publish_schedule(_schedule(revision=2), expected_revision=1)


def test_operations_schema_coexists_with_main_marvis_schema_version(tmp_path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    before = sqlite_health(db_path)

    OperationsStore(db_path, clock=lambda: NOW)
    OperationsStore(db_path, clock=lambda: NOW)

    after = sqlite_health(db_path)
    assert after["schema_version"] == before["schema_version"]
    assert after["schema_version_expected"] == before["schema_version_expected"]


def test_period_claim_is_idempotent_and_preserves_append_only_run_event_ledgers(
    tmp_path,
):
    db_path = tmp_path / "marvis.sqlite"
    store_a = OperationsStore(db_path, clock=lambda: NOW)
    store_b = OperationsStore(db_path, clock=lambda: NOW)
    schedule = _schedule()
    store_a.publish_schedule(schedule, expected_revision=0)
    period = schedule.calendar.period(0)

    claim = store_a.claim_period(schedule, period, owner_id="scheduler-a")
    duplicate = store_b.claim_period(schedule, period, owner_id="scheduler-b")

    assert claim is not None
    assert duplicate is None
    outcome = store_a.complete_run(
        claim,
        MonitoringOutcome(
            level="green",
            evidence_ref="artifact:deterministic-monitoring-1",
            evidence_hash="b" * 64,
        ),
    )
    assert store_b.claim_period(schedule, period, owner_id="scheduler-b") is None
    assert store_a.get_period(schedule.schedule_id, period.key).state == "succeeded"
    assert store_a.get_outcome(schedule.schedule_id, period.key) == outcome
    assert [run.attempt_number for run in store_a.list_runs(
        schedule.schedule_id, period.key
    )] == [1]
    assert [event.event_type for event in store_a.list_events(
        schedule.schedule_id, period.key
    )] == ["run_claimed", "run_succeeded"]


def test_expired_lease_recovers_with_backoff_and_retry_attempts_are_hard_bounded(
    tmp_path,
):
    clock = _Clock(NOW)
    store = OperationsStore(tmp_path / "marvis.sqlite", clock=clock)
    schedule = replace(
        _schedule(),
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=5,
            multiplier=2,
            max_backoff_seconds=20,
        ),
    )
    store.publish_schedule(schedule, expected_revision=0)
    period = schedule.calendar.period(0)
    first = store.claim_period(schedule, period, owner_id="crashed-scheduler")
    assert first is not None

    clock.now = NOW + timedelta(seconds=31)
    with pytest.raises(StaleRunLeaseError):
        store.complete_run(
            first,
            MonitoringOutcome(
                level="green",
                evidence_ref="artifact:too-late",
                evidence_hash="c" * 64,
            ),
        )
    recovered = store.recover_expired_leases()

    assert recovered[0].state == "retry_wait"
    assert recovered[0].next_attempt_at == NOW + timedelta(seconds=36)
    clock.now = NOW + timedelta(seconds=35)
    assert store.claim_period(schedule, period, owner_id="scheduler-b") is None
    clock.now = NOW + timedelta(seconds=36)
    second = store.claim_period(schedule, period, owner_id="scheduler-b")
    assert second is not None
    terminal = store.fail_run(second, error_code="monitoring_execution_failed")

    assert terminal.state == "terminal_failed"
    clock.now = NOW + timedelta(days=1)
    assert store.claim_period(schedule, period, owner_id="scheduler-c") is None
    assert [run.attempt_number for run in store.list_runs(
        schedule.schedule_id, period.key
    )] == [1, 2]
    assert [event.event_type for event in store.list_events(
        schedule.schedule_id, period.key
    )] == ["run_claimed", "run_lease_expired", "run_claimed", "run_failed"]

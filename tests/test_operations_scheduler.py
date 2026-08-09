from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import threading

from marvis.operations import (
    FixedIntervalCalendar,
    LocalScheduler,
    MonitoringOutcome,
    OperationsStore,
    RetryPolicy,
    ScheduleContract,
)


NOW = datetime(2026, 8, 1, 5, 30, tzinfo=UTC)


class _Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _schedule() -> ScheduleContract:
    return ScheduleContract(
        schedule_id="hourly-monitor",
        revision=1,
        monitoring_ref="private-monitoring-plan-99",
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        calendar=FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=3600,
        ),
        catch_up_budget=4,
        lease_seconds=30,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=5,
            multiplier=2,
            max_backoff_seconds=20,
        ),
    )


def _successful_outcome(request) -> MonitoringOutcome:
    evidence_hash = hashlib.sha256(request.period.key.encode("utf-8")).hexdigest()
    return MonitoringOutcome(
        level="green",
        evidence_ref=f"artifact:monitoring:{request.period.index}",
        evidence_hash=evidence_hash,
    )


def test_tick_catches_up_oldest_periods_without_exceeding_hard_budget(tmp_path):
    store = OperationsStore(tmp_path / "marvis.sqlite", clock=lambda: NOW)
    schedule = _schedule()
    store.publish_schedule(schedule, expected_revision=0)
    executed = []

    def execute(request):
        executed.append(request)
        return _successful_outcome(request)

    scheduler = LocalScheduler(
        store,
        executor=execute,
        owner_id="scheduler-a",
        clock=lambda: NOW,
    )

    first = scheduler.tick(catch_up_budget=2)
    second = scheduler.tick(catch_up_budget=2)

    assert first.attempted == first.succeeded == 2
    assert second.attempted == second.succeeded == 2
    assert [request.period.index for request in executed] == [0, 1, 2, 3]


def test_concurrent_schedulers_create_one_effective_run_for_schedule_period(tmp_path):
    one_period_due = datetime(2026, 8, 1, 1, 30, tzinfo=UTC)
    db_path = tmp_path / "marvis.sqlite"
    setup_store = OperationsStore(db_path, clock=lambda: one_period_due)
    schedule = _schedule()
    setup_store.publish_schedule(schedule, expected_revision=0)
    entered_executor = threading.Event()
    release_executor = threading.Event()
    executed_run_ids = []
    executed_lock = threading.Lock()

    def execute(request):
        with executed_lock:
            executed_run_ids.append(request.run_id)
        entered_executor.set()
        assert release_executor.wait(timeout=2)
        return _successful_outcome(request)

    schedulers = [
        LocalScheduler(
            OperationsStore(db_path, clock=lambda: one_period_due),
            executor=execute,
            owner_id=f"scheduler-{index}",
            clock=lambda: one_period_due,
        )
        for index in range(2)
    ]
    rendezvous = threading.Barrier(2)

    def tick(scheduler):
        rendezvous.wait(timeout=2)
        return scheduler.tick(catch_up_budget=1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(tick, scheduler) for scheduler in schedulers]
        assert entered_executor.wait(timeout=2)
        release_executor.set()
        reports = [future.result(timeout=2) for future in futures]

    period = schedule.calendar.period(0)
    assert sum(report.attempted for report in reports) == 1
    assert len(executed_run_ids) == 1
    assert len(setup_store.list_runs(schedule.schedule_id, period.key)) == 1
    assert setup_store.get_period(schedule.schedule_id, period.key).state == "succeeded"


def test_scheduler_retries_only_after_persisted_backoff_then_recovers(tmp_path):
    clock = _Clock(datetime(2026, 8, 1, 1, 30, tzinfo=UTC))
    store = OperationsStore(tmp_path / "marvis.sqlite", clock=clock)
    schedule = replace(_schedule(), catch_up_budget=1)
    store.publish_schedule(schedule, expected_revision=0)
    attempts = []

    def execute(request):
        attempts.append(request.attempt_number)
        if request.attempt_number == 1:
            raise RuntimeError("transient private failure details")
        return _successful_outcome(request)

    scheduler = LocalScheduler(
        store,
        executor=execute,
        owner_id="scheduler-restarted",
        clock=clock,
    )

    first = scheduler.tick(catch_up_budget=1)
    clock.now += timedelta(seconds=4)
    too_early = scheduler.tick(catch_up_budget=1)
    clock.now += timedelta(seconds=1)
    recovered = scheduler.tick(catch_up_budget=1)

    period = schedule.calendar.period(0)
    assert (first.failed, too_early.attempted, recovered.succeeded) == (1, 0, 1)
    assert attempts == [1, 2]
    assert store.get_period(schedule.schedule_id, period.key).state == "succeeded"


def test_new_scheduler_recovers_crashed_run_after_process_restart(tmp_path):
    clock = _Clock(datetime(2026, 8, 1, 1, 30, tzinfo=UTC))
    db_path = tmp_path / "marvis.sqlite"
    before_restart = OperationsStore(db_path, clock=clock)
    schedule = replace(
        _schedule(),
        catch_up_budget=1,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            multiplier=2,
            max_backoff_seconds=0,
        ),
    )
    before_restart.publish_schedule(schedule, expected_revision=0)
    period = schedule.calendar.period(0)
    abandoned = before_restart.claim_period(
        schedule,
        period,
        owner_id="scheduler-before-crash",
    )
    assert abandoned is not None
    clock.now += timedelta(seconds=31)
    executed_attempts = []

    def execute(request):
        executed_attempts.append(request.attempt_number)
        return _successful_outcome(request)

    after_restart = OperationsStore(db_path, clock=clock)
    report = LocalScheduler(
        after_restart,
        executor=execute,
        owner_id="scheduler-after-restart",
        clock=clock,
    ).tick(catch_up_budget=1)

    assert (report.recovered, report.succeeded) == (1, 1)
    assert executed_attempts == [2]
    assert [event.event_type for event in after_restart.list_events(
        schedule.schedule_id, period.key
    )] == [
        "run_claimed",
        "run_lease_expired",
        "run_claimed",
        "run_succeeded",
    ]

from datetime import UTC, datetime, timedelta
import json

import pytest

from marvis.operations import (
    FixedIntervalCalendar,
    HumanEscalationSuggestion,
    LocalScheduler,
    MonitoringOutcome,
    NotificationRedactor,
    OperationsStore,
    OutcomeRecord,
    RetryPolicy,
    ScheduleContract,
)


NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)


class _Clock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_notification_redactor_exposes_only_allowlisted_tokens_and_human_review_flag():
    outcome = OutcomeRecord(
        outcome_id="outcome-private-7",
        run_id="run-private-8",
        schedule_id="schedule-customer-acme",
        period_key="period-private-2026-08",
        outcome=MonitoringOutcome(
            level="red",
            evidence_ref="s3://customer-acme/raw-monitoring-evidence.json",
            evidence_hash="e" * 64,
            escalation=HumanEscalationSuggestion(reason_code="threshold_breached"),
        ),
        outcome_hash="f" * 64,
        recorded_at=NOW,
    )

    notification = NotificationRedactor(
        b"local-redaction-secret-at-least-32-bytes"
    ).redact(outcome)
    payload = notification.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["schema_version"] == "operations.notification.redacted.v1"
    assert payload["event_type"] == "monitoring_completed"
    assert payload["level"] == "red"
    assert payload["human_review_required"] is True
    assert payload["automatic_action_permitted"] is False
    for raw_secret in (
        outcome.outcome_id,
        outcome.run_id,
        outcome.schedule_id,
        outcome.period_key,
        outcome.outcome.evidence_ref,
        outcome.outcome.evidence_hash,
    ):
        assert raw_secret not in serialized


def test_notification_failure_retries_independently_without_changing_monitoring_outcome(
    tmp_path,
):
    clock = _Clock(datetime(2026, 8, 1, 1, 30, tzinfo=UTC))
    store = OperationsStore(tmp_path / "marvis.sqlite", clock=clock)
    schedule = ScheduleContract(
        schedule_id="schedule-customer-secret",
        revision=1,
        monitoring_ref="monitoring-plan-customer-secret",
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        calendar=FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=3600,
        ),
        catch_up_budget=1,
        lease_seconds=30,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            multiplier=2,
            max_backoff_seconds=0,
        ),
    )
    store.publish_schedule(schedule, expected_revision=0)

    def execute(_request):
        return MonitoringOutcome(
            level="red",
            evidence_ref="artifact:raw-customer-secret-evidence",
            evidence_hash="a" * 64,
            escalation=HumanEscalationSuggestion(reason_code="threshold_breached"),
        )

    class FailingAdapter:
        def __init__(self):
            self.received = []

        def send(self, notification):
            self.received.append(notification)
            raise RuntimeError("smtp credential secret must never be audited")

    adapter = FailingAdapter()
    scheduler = LocalScheduler(
        store,
        executor=execute,
        owner_id="scheduler-a",
        clock=clock,
        notification_adapter=adapter,
        notification_redactor=NotificationRedactor(
            b"notification-redaction-secret-32bytes"
        ),
        notification_retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=5,
            multiplier=2,
            max_backoff_seconds=5,
        ),
    )

    first = scheduler.tick(catch_up_budget=1, notification_budget=1)
    period = schedule.calendar.period(0)
    persisted_outcome = store.get_outcome(schedule.schedule_id, period.key)
    pending = store.list_notifications(schedule.schedule_id, period.key)[0]

    assert (first.succeeded, first.notifications_failed) == (1, 1)
    assert persisted_outcome.outcome.level == "red"
    assert pending.status == "pending"
    assert pending.attempt_count == 1
    assert pending.next_attempt_at == clock.now + timedelta(seconds=5)
    redacted_payload = json.dumps(adapter.received[0].to_dict(), sort_keys=True)
    for raw_secret in (
        schedule.schedule_id,
        schedule.monitoring_ref,
        persisted_outcome.outcome.evidence_ref,
        persisted_outcome.outcome.evidence_hash,
    ):
        assert raw_secret not in redacted_payload

    clock.now += timedelta(seconds=5)
    second = scheduler.tick(catch_up_budget=1, notification_budget=1)
    exhausted = store.list_notifications(schedule.schedule_id, period.key)[0]
    audited_payload = json.dumps(
        [event.payload for event in store.list_events(schedule.schedule_id, period.key)],
        sort_keys=True,
    )

    assert (second.attempted, second.notifications_failed) == (0, 1)
    assert exhausted.status == "exhausted"
    assert exhausted.attempt_count == 2
    assert store.get_period(schedule.schedule_id, period.key).state == "succeeded"
    assert store.get_outcome(schedule.schedule_id, period.key) == persisted_outcome
    assert "smtp credential secret" not in audited_payload


def test_restart_reconciles_committed_outcome_missing_its_outbox_record(tmp_path):
    clock = _Clock(datetime(2026, 8, 1, 1, 30, tzinfo=UTC))
    db_path = tmp_path / "marvis.sqlite"
    before_restart = OperationsStore(db_path, clock=clock)
    schedule = ScheduleContract(
        schedule_id="restart-notification-monitor",
        revision=1,
        monitoring_ref="monitoring-plan-restart",
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        calendar=FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=3600,
        ),
        catch_up_budget=1,
        lease_seconds=30,
        retry_policy=RetryPolicy(
            max_attempts=1,
            initial_backoff_seconds=0,
            multiplier=1,
            max_backoff_seconds=0,
        ),
    )
    before_restart.publish_schedule(schedule, expected_revision=0)
    period = schedule.calendar.period(0)
    claim = before_restart.claim_period(
        schedule,
        period,
        owner_id="scheduler-before-outbox-crash",
    )
    assert claim is not None
    outcome = before_restart.complete_run(
        claim,
        MonitoringOutcome(
            level="amber",
            evidence_ref="artifact:committed-before-crash",
            evidence_hash="b" * 64,
        ),
    )
    assert before_restart.list_notifications(schedule.schedule_id, period.key) == ()

    class CapturingAdapter:
        def __init__(self):
            self.received = []

        def send(self, notification):
            self.received.append(notification)

    adapter = CapturingAdapter()
    after_restart = OperationsStore(db_path, clock=clock)
    scheduler = LocalScheduler(
        after_restart,
        executor=lambda _request: pytest.fail("completed period must not rerun"),
        owner_id="scheduler-after-outbox-crash",
        clock=clock,
        notification_adapter=adapter,
        notification_redactor=NotificationRedactor(
            b"notification-recovery-secret-32bytes"
        ),
        notification_retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=1,
            multiplier=2,
            max_backoff_seconds=2,
        ),
    )

    report = scheduler.tick(catch_up_budget=1, notification_budget=1)

    sent = after_restart.list_notifications(schedule.schedule_id, period.key)[0]
    assert (report.attempted, report.notifications_sent) == (0, 1)
    assert sent.status == "sent"
    assert sent.outcome_id == outcome.outcome_id
    assert len(adapter.received) == 1

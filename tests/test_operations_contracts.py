from datetime import UTC, datetime
from dataclasses import replace

import pytest

from marvis.operations import (
    FixedIntervalCalendar,
    HumanEscalationSuggestion,
    MonitoringOutcome,
    RetryPolicy,
    ScheduleContract,
)


def test_fixed_interval_calendar_emits_stable_versioned_period_keys_with_limit():
    calendar = FixedIntervalCalendar(
        anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
        interval_seconds=3600,
    )

    periods = calendar.due_periods(
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        now=datetime(2026, 8, 1, 3, 30, tzinfo=UTC),
        limit=2,
    )

    assert [period.key for period in periods] == [
        "operations.calendar.fixed_interval.v1:2026-08-01T00:00:00Z/2026-08-01T01:00:00Z",
        "operations.calendar.fixed_interval.v1:2026-08-01T01:00:00Z/2026-08-01T02:00:00Z",
    ]
    assert [period.index for period in periods] == [0, 1]


def test_schedule_contract_round_trips_versioned_calendar_and_retry_policy():
    schedule = ScheduleContract(
        schedule_id="monthly-score-monitor",
        revision=3,
        monitoring_ref="monitoring-plan-42",
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        calendar=FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=86400,
        ),
        catch_up_budget=4,
        lease_seconds=90,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=10,
            multiplier=2,
            max_backoff_seconds=30,
        ),
    )

    payload = schedule.to_dict()

    assert payload["schema_version"] == "operations.schedule.v1"
    assert payload["calendar"]["schema_version"] == (
        "operations.calendar.fixed_interval.v1"
    )
    assert payload["retry_policy"]["schema_version"] == "operations.retry.v1"
    assert ScheduleContract.from_dict(payload) == schedule


def test_escalation_contract_is_a_human_only_suggestion_without_effect_authority():
    outcome = MonitoringOutcome(
        level="red",
        evidence_ref="artifact:monitoring-result-7",
        evidence_hash="a" * 64,
        escalation=HumanEscalationSuggestion(reason_code="threshold_breached"),
    )

    escalation = outcome.to_dict()["escalation"]

    assert escalation == {
        "schema_version": "operations.human_escalation.v1",
        "action": "review_monitoring_result",
        "reason_code": "threshold_breached",
        "requires_human_confirmation": True,
        "automatic_action_permitted": False,
    }


def test_integer_contract_fields_reject_fractional_values():
    with pytest.raises(ValueError, match="interval_seconds"):
        FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=1.5,
        )

    schedule = ScheduleContract(
        schedule_id="typed-schedule",
        revision=1,
        monitoring_ref="monitoring-plan",
        active_from=datetime(2026, 8, 1, tzinfo=UTC),
        calendar=FixedIntervalCalendar(
            anchor_at=datetime(2026, 8, 1, tzinfo=UTC),
            interval_seconds=3600,
        ),
        catch_up_budget=1,
        lease_seconds=10,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=1,
            multiplier=2,
            max_backoff_seconds=4,
        ),
    )
    for field in ("revision", "catch_up_budget", "lease_seconds"):
        with pytest.raises(ValueError, match=field):
            replace(schedule, **{field: 1.5})
    with pytest.raises(ValueError, match="attempt_number"):
        schedule.retry_policy.delay_after_failure(1.5)

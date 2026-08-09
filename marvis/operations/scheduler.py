"""Synchronous local scheduler built on fenced SQLite leases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Protocol

from marvis.operations.contracts import (
    MAX_CATCH_UP_BUDGET,
    CalendarPeriod,
    MonitoringOutcome,
    RetryPolicy,
)
from marvis.operations.notifications import NotificationAdapter, NotificationRedactor
from marvis.operations.repository import (
    OperationsStore,
    OutcomeRecord,
    RunClaim,
    StaleNotificationLeaseError,
    StaleRunLeaseError,
)


@dataclass(frozen=True)
class ScheduledRunRequest:
    """Pinned request passed to the deterministic monitoring integration."""

    run_id: str
    schedule_id: str
    schedule_revision: int
    monitoring_ref: str
    period: CalendarPeriod
    attempt_number: int


class MonitoringExecutor(Protocol):
    def __call__(self, request: ScheduledRunRequest) -> MonitoringOutcome: ...


@dataclass(frozen=True)
class TickReport:
    attempted: int
    succeeded: int
    failed: int
    recovered: int
    claimed_period_keys: tuple[str, ...]
    notifications_attempted: int
    notifications_sent: int
    notifications_failed: int
    notifications_recovered: int


class LocalScheduler:
    """Run due local monitoring periods under a strict per-tick budget."""

    def __init__(
        self,
        store: OperationsStore,
        *,
        executor: MonitoringExecutor,
        owner_id: str,
        clock: Callable[[], datetime] | None = None,
        notification_adapter: NotificationAdapter | None = None,
        notification_redactor: NotificationRedactor | None = None,
        notification_retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not isinstance(store, OperationsStore):
            raise ValueError("store must be an OperationsStore")
        if not callable(executor):
            raise ValueError("executor must be callable")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be non-empty text")
        if (notification_adapter is None) != (notification_redactor is None):
            raise ValueError(
                "notification_adapter and notification_redactor must be provided together"
            )
        if notification_adapter is not None and not callable(
            getattr(notification_adapter, "send", None)
        ):
            raise ValueError("notification_adapter must provide send(notification)")
        if notification_redactor is not None and not isinstance(
            notification_redactor, NotificationRedactor
        ):
            raise ValueError("notification_redactor must be a NotificationRedactor")
        if notification_retry_policy is not None and not isinstance(
            notification_retry_policy, RetryPolicy
        ):
            raise ValueError("notification_retry_policy must be a RetryPolicy")
        self._store = store
        self._executor = executor
        self._owner_id = owner_id.strip()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._notification_adapter = notification_adapter
        self._notification_redactor = notification_redactor
        self._notification_retry_policy = notification_retry_policy or RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=5,
            multiplier=2,
            max_backoff_seconds=60,
        )

    def tick(
        self,
        *,
        catch_up_budget: int,
        notification_budget: int = 10,
    ) -> TickReport:
        if (
            isinstance(catch_up_budget, bool)
            or not isinstance(catch_up_budget, int)
            or not 1 <= catch_up_budget <= MAX_CATCH_UP_BUDGET
        ):
            raise ValueError(
                f"catch_up_budget must be between 1 and {MAX_CATCH_UP_BUDGET}"
            )
        if (
            isinstance(notification_budget, bool)
            or not isinstance(notification_budget, int)
            or not 0 <= notification_budget <= 100
        ):
            raise ValueError("notification_budget must be between 0 and 100")
        now = _utc_datetime(self._clock())
        recovered = self._store.recover_expired_leases()
        schedules = self._store.list_active_schedules()
        active_by_id = {record.contract.schedule_id: record for record in schedules}
        per_schedule_attempts = {
            record.contract.schedule_id: 0 for record in schedules
        }
        attempted = 0
        succeeded = 0
        failed = 0
        claimed_period_keys: list[str] = []
        retryable = self._store.list_retryable_periods(
            now=now,
            limit=catch_up_budget,
        )
        for period_record in retryable:
            if attempted >= catch_up_budget:
                break
            latest = active_by_id.get(period_record.schedule_id)
            if latest is None:
                continue
            if (
                per_schedule_attempts[period_record.schedule_id]
                >= latest.contract.catch_up_budget
            ):
                continue
            pinned = self._store.get_schedule(
                period_record.schedule_id,
                revision=period_record.schedule_revision,
            )
            if pinned is None:
                continue
            claim = self._store.claim_period(
                pinned.contract,
                period_record.period,
                owner_id=self._owner_id,
            )
            if claim is None:
                continue
            attempted += 1
            per_schedule_attempts[period_record.schedule_id] += 1
            claimed_period_keys.append(period_record.period.key)
            if self._execute_claim(pinned.contract.monitoring_ref, claim) is not None:
                succeeded += 1
            else:
                failed += 1
        while attempted < catch_up_budget:
            made_progress = False
            for record in schedules:
                schedule = record.contract
                if attempted >= catch_up_budget:
                    break
                if (
                    per_schedule_attempts[schedule.schedule_id]
                    >= schedule.catch_up_budget
                ):
                    continue
                period = self._store.next_unseen_due_period(schedule, now=now)
                if period is None:
                    continue
                claim = self._store.claim_period(
                    schedule,
                    period,
                    owner_id=self._owner_id,
                )
                if claim is None:
                    continue
                made_progress = True
                attempted += 1
                per_schedule_attempts[schedule.schedule_id] += 1
                claimed_period_keys.append(period.key)
                if self._execute_claim(schedule.monitoring_ref, claim) is not None:
                    succeeded += 1
                else:
                    failed += 1
            if not made_progress:
                break
        (
            notifications_attempted,
            notifications_sent,
            notifications_failed,
            notifications_recovered,
        ) = self._flush_notifications(notification_budget)
        return TickReport(
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            recovered=len(recovered),
            claimed_period_keys=tuple(claimed_period_keys),
            notifications_attempted=notifications_attempted,
            notifications_sent=notifications_sent,
            notifications_failed=notifications_failed,
            notifications_recovered=notifications_recovered,
        )

    def _execute_claim(
        self,
        monitoring_ref: str,
        claim: RunClaim,
    ) -> OutcomeRecord | None:
        request = ScheduledRunRequest(
            run_id=claim.run_id,
            schedule_id=claim.schedule_id,
            schedule_revision=claim.schedule_revision,
            monitoring_ref=monitoring_ref,
            period=claim.period,
            attempt_number=claim.attempt_number,
        )
        try:
            outcome = self._executor(request)
            outcome_record = self._store.complete_run(claim, outcome)
        except Exception:
            try:
                self._store.fail_run(
                    claim,
                    error_code="monitoring_execution_failed",
                )
            except StaleRunLeaseError:
                pass
            return None
        if self._notification_redactor is not None:
            try:
                notification = self._notification_redactor.redact(outcome_record)
                self._store.enqueue_notification(
                    outcome_record,
                    notification,
                    retry_policy=self._notification_retry_policy,
                )
            except Exception:
                # Outcome is already durably committed. Notification plumbing is
                # deliberately unable to rewrite that deterministic result.
                pass
        return outcome_record

    def _flush_notifications(self, budget: int) -> tuple[int, int, int, int]:
        if self._notification_adapter is None or budget == 0:
            return (0, 0, 0, 0)
        reconciled = self._reconcile_missing_outbox_records(budget)
        recovered = self._store.recover_expired_notification_leases()
        attempted = 0
        sent = 0
        failed = 0
        while attempted < budget:
            claim = self._store.claim_notification(owner_id=self._owner_id)
            if claim is None:
                break
            attempted += 1
            try:
                self._notification_adapter.send(claim.notification.notification)
            except Exception:
                try:
                    self._store.fail_notification(
                        claim,
                        error_code="notification_delivery_failed",
                    )
                except StaleNotificationLeaseError:
                    pass
                failed += 1
            else:
                try:
                    self._store.complete_notification(claim)
                except StaleNotificationLeaseError:
                    # Another scheduler already expired/fenced this delivery.
                    # The stable notification_id lets an adapter deduplicate a
                    # later retry if the external side effect already landed.
                    failed += 1
                else:
                    sent += 1
        return (attempted, sent, failed, len(recovered) + reconciled)

    def _reconcile_missing_outbox_records(self, budget: int) -> int:
        if self._notification_redactor is None:
            return 0
        reconciled = 0
        for outcome in self._store.list_unnotified_outcomes(limit=budget):
            try:
                notification = self._notification_redactor.redact(outcome)
                self._store.enqueue_notification(
                    outcome,
                    notification,
                    retry_policy=self._notification_retry_policy,
                )
            except Exception:
                continue
            reconciled += 1
        return reconciled


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)

"""SQLite persistence seam for local operations scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Mapping
import uuid

from marvis.db_schema import connect
from marvis.operations.contracts import (
    CalendarPeriod,
    MonitoringOutcome,
    RetryPolicy,
    ScheduleContract,
)
from marvis.operations.notifications import RedactedNotification
from marvis.operations.schema import install_operations_schema


_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")


class ScheduleRevisionConflict(RuntimeError):
    """A schedule append did not match the current immutable revision."""


class StaleRunLeaseError(RuntimeError):
    """A worker attempted to finish a run after losing its lease."""


class StaleNotificationLeaseError(RuntimeError):
    """An adapter attempted to finish delivery after losing its lease."""


@dataclass(frozen=True)
class ScheduleRecord:
    contract: ScheduleContract
    contract_hash: str
    created_at: datetime


@dataclass(frozen=True)
class RunClaim:
    run_id: str
    schedule_id: str
    schedule_revision: int
    period: CalendarPeriod
    attempt_number: int
    owner_id: str
    lease_token: str
    claimed_at: datetime
    lease_expires_at: datetime


@dataclass(frozen=True)
class PeriodRecord:
    schedule_id: str
    period: CalendarPeriod
    schedule_revision: int
    state: str
    attempt_count: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime | None
    outcome_id: str | None
    updated_at: datetime


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    schedule_id: str
    period_key: str
    schedule_revision: int
    attempt_number: int
    owner_id: str
    claimed_at: datetime
    lease_expires_at: datetime


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    run_id: str
    schedule_id: str
    period_key: str
    outcome: MonitoringOutcome
    outcome_hash: str
    recorded_at: datetime


@dataclass(frozen=True)
class EventRecord:
    sequence: int
    event_id: str
    schedule_id: str
    period_key: str
    run_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class NotificationRecord:
    notification_id: str
    outcome_id: str
    schedule_id: str
    period_key: str
    run_id: str
    notification: RedactedNotification
    payload_hash: str
    status: str
    attempt_count: int
    max_attempts: int
    next_attempt_at: datetime | None
    lease_owner: str | None
    lease_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class NotificationClaim:
    notification: NotificationRecord
    owner_id: str
    lease_token: str
    lease_expires_at: datetime


class OperationsStore:
    """Public persistence API for versioned schedules and operations ledgers."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._clock = clock or (lambda: datetime.now(UTC))
        install_operations_schema(self.db_path)

    def publish_schedule(
        self,
        contract: ScheduleContract,
        *,
        expected_revision: int,
    ) -> ScheduleRecord:
        if not isinstance(contract, ScheduleContract):
            raise ValueError("contract must be a ScheduleContract")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        created_at = _utc_datetime(self._clock(), "clock")
        payload_json = _canonical_json(contract.to_dict())
        contract_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT MAX(revision) AS revision
                  FROM operations_schedules
                 WHERE schedule_id = ?
                """,
                (contract.schedule_id,),
            ).fetchone()
            actual_revision = int(row["revision"] or 0)
            if actual_revision != expected_revision:
                raise ScheduleRevisionConflict(
                    "stale schedule revision: "
                    f"expected {expected_revision}, found {actual_revision}"
                )
            if contract.revision != actual_revision + 1:
                raise ScheduleRevisionConflict(
                    "new schedule revision must be exactly "
                    f"{actual_revision + 1}, found {contract.revision}"
                )
            try:
                conn.execute(
                    """
                    INSERT INTO operations_schedules(
                        schedule_id, revision, schema_version, contract_json,
                        contract_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        contract.schedule_id,
                        contract.revision,
                        contract.schema_version,
                        payload_json,
                        contract_hash,
                        _iso_z(created_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ScheduleRevisionConflict(
                    f"duplicate schedule revision: {contract.schedule_id} "
                    f"revision {contract.revision}"
                ) from exc
        return ScheduleRecord(
            contract=contract,
            contract_hash=contract_hash,
            created_at=created_at,
        )

    def list_schedule_revisions(self, schedule_id: str) -> tuple[ScheduleRecord, ...]:
        normalized_id = _required_text(schedule_id, "schedule_id")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT contract_json, contract_hash, created_at
                  FROM operations_schedules
                 WHERE schedule_id = ?
                 ORDER BY revision
                """,
                (normalized_id,),
            ).fetchall()
        return tuple(_schedule_record(row) for row in rows)

    def get_schedule(
        self,
        schedule_id: str,
        *,
        revision: int | None = None,
    ) -> ScheduleRecord | None:
        normalized_id = _required_text(schedule_id, "schedule_id")
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
        ):
            raise ValueError("revision must be a positive integer")
        with connect(self.db_path) as conn:
            if revision is None:
                row = conn.execute(
                    """
                    SELECT contract_json, contract_hash, created_at
                      FROM operations_schedules
                     WHERE schedule_id = ?
                     ORDER BY revision DESC
                     LIMIT 1
                    """,
                    (normalized_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT contract_json, contract_hash, created_at
                      FROM operations_schedules
                     WHERE schedule_id = ? AND revision = ?
                    """,
                    (normalized_id, revision),
                ).fetchone()
        return None if row is None else _schedule_record(row)

    def list_active_schedules(self) -> tuple[ScheduleRecord, ...]:
        """Return the enabled latest immutable revision of every schedule."""

        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT schedule.contract_json, schedule.contract_hash,
                       schedule.created_at
                  FROM operations_schedules AS schedule
                  JOIN (
                        SELECT schedule_id, MAX(revision) AS revision
                          FROM operations_schedules
                         GROUP BY schedule_id
                  ) AS latest
                    ON latest.schedule_id = schedule.schedule_id
                   AND latest.revision = schedule.revision
                 ORDER BY schedule.schedule_id
                """
            ).fetchall()
        records = tuple(_schedule_record(row) for row in rows)
        return tuple(record for record in records if record.contract.enabled)

    def next_unseen_due_period(
        self,
        schedule: ScheduleContract,
        *,
        now: datetime,
    ) -> CalendarPeriod | None:
        """Return the oldest complete period not yet admitted to coordination state."""

        if not isinstance(schedule, ScheduleContract):
            raise ValueError("schedule must be a ScheduleContract")
        reference = _utc_datetime(now, "now")
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT MAX(period_ends_at) AS last_end
                  FROM operations_periods
                 WHERE schedule_id = ?
                """,
                (schedule.schedule_id,),
            ).fetchone()
        active_from = schedule.active_from
        if row is not None and row["last_end"] is not None:
            active_from = max(
                active_from,
                _parse_datetime(str(row["last_end"]), "period_ends_at"),
            )
        periods = schedule.calendar.due_periods(
            active_from=active_from,
            now=reference,
            limit=1,
        )
        return None if not periods else periods[0]

    def list_retryable_periods(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> tuple[PeriodRecord, ...]:
        reference = _utc_datetime(now, "now")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM operations_periods
                 WHERE state = 'retry_wait' AND next_attempt_at <= ?
                 ORDER BY next_attempt_at, period_ends_at, schedule_id, period_key
                 LIMIT ?
                """,
                (_iso_z(reference), limit),
            ).fetchall()
        return tuple(_period_record(row) for row in rows)

    def claim_period(
        self,
        schedule: ScheduleContract,
        period: CalendarPeriod,
        *,
        owner_id: str,
    ) -> RunClaim | None:
        """Atomically claim one schedule period, or return ``None`` if unavailable."""

        if not isinstance(schedule, ScheduleContract):
            raise ValueError("schedule must be a ScheduleContract")
        if not isinstance(period, CalendarPeriod):
            raise ValueError("period must be a CalendarPeriod")
        expected_period = schedule.calendar.period(period.index)
        if expected_period != period:
            raise ValueError("period does not belong to the schedule calendar")
        if period.starts_at < schedule.active_from:
            raise ValueError("period begins before schedule activation")
        normalized_owner = _required_text(owner_id, "owner_id")
        now = _utc_datetime(self._clock(), "clock")
        if period.ends_at > now:
            raise ValueError("period is not due yet")
        lease_expires_at = now + timedelta(seconds=schedule.lease_seconds)
        contract_hash = _payload_hash(schedule.to_dict())
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            schedule_row = conn.execute(
                """
                SELECT contract_hash FROM operations_schedules
                 WHERE schedule_id = ? AND revision = ?
                """,
                (schedule.schedule_id, schedule.revision),
            ).fetchone()
            if schedule_row is None:
                raise KeyError(
                    f"schedule revision not found: {schedule.schedule_id} "
                    f"revision {schedule.revision}"
                )
            if str(schedule_row["contract_hash"]) != contract_hash:
                raise ScheduleRevisionConflict(
                    "schedule contract does not match the persisted revision"
                )
            period_row = conn.execute(
                """
                SELECT * FROM operations_periods
                 WHERE schedule_id = ? AND period_key = ?
                """,
                (schedule.schedule_id, period.key),
            ).fetchone()
            if period_row is not None:
                state = str(period_row["state"])
                if state != "retry_wait":
                    return None
                next_attempt_at = _optional_datetime(period_row["next_attempt_at"])
                if next_attempt_at is not None and next_attempt_at > now:
                    return None
                attempt_count = int(period_row["attempt_count"])
                if attempt_count >= schedule.retry_policy.max_attempts:
                    return None
            else:
                attempt_count = 0

            run_id = uuid.uuid4().hex
            lease_token = uuid.uuid4().hex
            attempt_number = attempt_count + 1
            if period_row is None:
                conn.execute(
                    """
                    INSERT INTO operations_periods(
                        schedule_id, period_key, schedule_revision, period_index,
                        period_starts_at, period_ends_at, state, attempt_count,
                        lease_owner, lease_token, lease_expires_at, next_attempt_at,
                        outcome_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, NULL, NULL, ?)
                    """,
                    (
                        schedule.schedule_id,
                        period.key,
                        schedule.revision,
                        period.index,
                        _iso_z(period.starts_at),
                        _iso_z(period.ends_at),
                        attempt_number,
                        normalized_owner,
                        lease_token,
                        _iso_z(lease_expires_at),
                        _iso_z(now),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE operations_periods
                       SET state = 'running', attempt_count = ?, lease_owner = ?,
                           lease_token = ?, lease_expires_at = ?,
                           next_attempt_at = NULL, updated_at = ?
                     WHERE schedule_id = ? AND period_key = ?
                    """,
                    (
                        attempt_number,
                        normalized_owner,
                        lease_token,
                        _iso_z(lease_expires_at),
                        _iso_z(now),
                        schedule.schedule_id,
                        period.key,
                    ),
                )
            conn.execute(
                """
                INSERT INTO operations_runs(
                    run_id, schedule_id, period_key, schedule_revision,
                    attempt_number, owner_id, lease_token, claimed_at,
                    lease_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    schedule.schedule_id,
                    period.key,
                    schedule.revision,
                    attempt_number,
                    normalized_owner,
                    lease_token,
                    _iso_z(now),
                    _iso_z(lease_expires_at),
                ),
            )
            _write_event(
                conn,
                schedule_id=schedule.schedule_id,
                period_key=period.key,
                run_id=run_id,
                event_type="run_claimed",
                payload={
                    "attempt_number": attempt_number,
                    "lease_expires_at": _iso_z(lease_expires_at),
                },
                created_at=now,
            )
        return RunClaim(
            run_id=run_id,
            schedule_id=schedule.schedule_id,
            schedule_revision=schedule.revision,
            period=period,
            attempt_number=attempt_number,
            owner_id=normalized_owner,
            lease_token=lease_token,
            claimed_at=now,
            lease_expires_at=lease_expires_at,
        )

    def complete_run(
        self,
        claim: RunClaim,
        outcome: MonitoringOutcome,
    ) -> OutcomeRecord:
        if not isinstance(claim, RunClaim):
            raise ValueError("claim must be a RunClaim")
        if not isinstance(outcome, MonitoringOutcome):
            raise ValueError("outcome must be a MonitoringOutcome")
        now = _utc_datetime(self._clock(), "clock")
        outcome_json = _canonical_json(outcome.to_dict())
        outcome_hash = hashlib.sha256(outcome_json.encode("utf-8")).hexdigest()
        outcome_id = uuid.uuid4().hex
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT state, lease_token, lease_expires_at
                  FROM operations_periods
                 WHERE schedule_id = ? AND period_key = ?
                """,
                (claim.schedule_id, claim.period.key),
            ).fetchone()
            if (
                row is None
                or str(row["state"]) != "running"
                or str(row["lease_token"]) != claim.lease_token
                or _optional_datetime(row["lease_expires_at"]) is None
                or _optional_datetime(row["lease_expires_at"]) <= now
            ):
                raise StaleRunLeaseError("run lease is no longer current")
            conn.execute(
                """
                INSERT INTO operations_outcomes(
                    outcome_id, run_id, schedule_id, period_key, schema_version,
                    outcome_json, outcome_hash, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome_id,
                    claim.run_id,
                    claim.schedule_id,
                    claim.period.key,
                    outcome.schema_version,
                    outcome_json,
                    outcome_hash,
                    _iso_z(now),
                ),
            )
            conn.execute(
                """
                UPDATE operations_periods
                   SET state = 'succeeded', lease_owner = NULL, lease_token = NULL,
                       lease_expires_at = NULL, next_attempt_at = NULL,
                       outcome_id = ?, updated_at = ?
                 WHERE schedule_id = ? AND period_key = ?
                """,
                (
                    outcome_id,
                    _iso_z(now),
                    claim.schedule_id,
                    claim.period.key,
                ),
            )
            _write_event(
                conn,
                schedule_id=claim.schedule_id,
                period_key=claim.period.key,
                run_id=claim.run_id,
                event_type="run_succeeded",
                payload={
                    "level": outcome.level,
                    "outcome_hash": outcome_hash,
                    "outcome_id": outcome_id,
                },
                created_at=now,
            )
            row = conn.execute(
                "SELECT * FROM operations_outcomes WHERE outcome_id = ?",
                (outcome_id,),
            ).fetchone()
            assert row is not None
            return _outcome_record(row)

    def fail_run(self, claim: RunClaim, *, error_code: str) -> PeriodRecord:
        """Record a sanitized failure and schedule only a bounded retry."""

        if not isinstance(claim, RunClaim):
            raise ValueError("claim must be a RunClaim")
        if not isinstance(error_code, str) or not _ERROR_CODE_RE.fullmatch(error_code):
            raise ValueError("error_code must be a lower_snake_case identifier")
        now = _utc_datetime(self._clock(), "clock")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM operations_periods
                 WHERE schedule_id = ? AND period_key = ?
                """,
                (claim.schedule_id, claim.period.key),
            ).fetchone()
            if (
                row is None
                or str(row["state"]) != "running"
                or str(row["lease_token"]) != claim.lease_token
                or _optional_datetime(row["lease_expires_at"]) is None
                or _optional_datetime(row["lease_expires_at"]) <= now
            ):
                raise StaleRunLeaseError("run lease is no longer current")
            schedule = _load_schedule_on_connection(
                conn, claim.schedule_id, int(row["schedule_revision"])
            )
            return _record_attempt_failure(
                conn,
                row=row,
                schedule=schedule,
                run_id=claim.run_id,
                event_type="run_failed",
                event_payload={"error_code": error_code},
                now=now,
            )

    def recover_expired_leases(self) -> tuple[PeriodRecord, ...]:
        """Expire crashed workers and move their attempts into bounded retry state."""

        now = _utc_datetime(self._clock(), "clock")
        recovered: list[PeriodRecord] = []
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM operations_periods
                 WHERE state = 'running' AND lease_expires_at <= ?
                 ORDER BY lease_expires_at, schedule_id, period_key
                """,
                (_iso_z(now),),
            ).fetchall()
            for row in rows:
                schedule_id = str(row["schedule_id"])
                schedule = _load_schedule_on_connection(
                    conn, schedule_id, int(row["schedule_revision"])
                )
                run_row = conn.execute(
                    "SELECT run_id FROM operations_runs WHERE lease_token = ?",
                    (str(row["lease_token"]),),
                ).fetchone()
                if run_row is None:
                    raise ValueError("running period has no matching immutable run")
                recovered.append(
                    _record_attempt_failure(
                        conn,
                        row=row,
                        schedule=schedule,
                        run_id=str(run_row["run_id"]),
                        event_type="run_lease_expired",
                        event_payload={"reason_code": "lease_expired"},
                        now=now,
                    )
                )
        return tuple(recovered)

    def get_period(self, schedule_id: str, period_key: str) -> PeriodRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM operations_periods
                 WHERE schedule_id = ? AND period_key = ?
                """,
                (
                    _required_text(schedule_id, "schedule_id"),
                    _required_text(period_key, "period_key"),
                ),
            ).fetchone()
        return None if row is None else _period_record(row)

    def get_outcome(self, schedule_id: str, period_key: str) -> OutcomeRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM operations_outcomes
                 WHERE schedule_id = ? AND period_key = ?
                """,
                (
                    _required_text(schedule_id, "schedule_id"),
                    _required_text(period_key, "period_key"),
                ),
            ).fetchone()
        return None if row is None else _outcome_record(row)

    def list_unnotified_outcomes(self, *, limit: int) -> tuple[OutcomeRecord, ...]:
        """Find committed outcomes left between result commit and outbox enqueue."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT outcome.*
                  FROM operations_outcomes AS outcome
             LEFT JOIN operations_notification_outbox AS notification
                    ON notification.outcome_id = outcome.outcome_id
                 WHERE notification.outcome_id IS NULL
                 ORDER BY outcome.recorded_at, outcome.outcome_id
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(_outcome_record(row) for row in rows)

    def list_runs(self, schedule_id: str, period_key: str) -> tuple[RunRecord, ...]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM operations_runs
                 WHERE schedule_id = ? AND period_key = ?
                 ORDER BY attempt_number
                """,
                (
                    _required_text(schedule_id, "schedule_id"),
                    _required_text(period_key, "period_key"),
                ),
            ).fetchall()
        return tuple(_run_record(row) for row in rows)

    def list_events(self, schedule_id: str, period_key: str) -> tuple[EventRecord, ...]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM operations_events
                 WHERE schedule_id = ? AND period_key = ?
                 ORDER BY sequence
                """,
                (
                    _required_text(schedule_id, "schedule_id"),
                    _required_text(period_key, "period_key"),
                ),
            ).fetchall()
        return tuple(_event_record(row) for row in rows)

    def enqueue_notification(
        self,
        outcome: OutcomeRecord,
        notification: RedactedNotification,
        *,
        retry_policy: RetryPolicy,
    ) -> NotificationRecord:
        """Persist an already-redacted allowlist payload after outcome commit."""

        if not isinstance(outcome, OutcomeRecord):
            raise ValueError("outcome must be an OutcomeRecord")
        if not isinstance(notification, RedactedNotification):
            raise ValueError("notification must be a RedactedNotification")
        if not isinstance(retry_policy, RetryPolicy):
            raise ValueError("retry_policy must be a RetryPolicy")
        if notification.level != outcome.outcome.level:
            raise ValueError("notification level must match the monitoring outcome")
        if notification.human_review_required != (
            outcome.outcome.escalation is not None
        ):
            raise ValueError("notification human-review flag must match the outcome")
        now = _utc_datetime(self._clock(), "clock")
        payload_json = _canonical_json(notification.to_dict())
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            outcome_row = conn.execute(
                """
                SELECT run_id, schedule_id, period_key
                  FROM operations_outcomes
                 WHERE outcome_id = ?
                """,
                (outcome.outcome_id,),
            ).fetchone()
            if outcome_row is None:
                raise KeyError(f"monitoring outcome not found: {outcome.outcome_id}")
            if (
                str(outcome_row["run_id"]) != outcome.run_id
                or str(outcome_row["schedule_id"]) != outcome.schedule_id
                or str(outcome_row["period_key"]) != outcome.period_key
            ):
                raise ValueError("monitoring outcome binding mismatch")
            existing = conn.execute(
                """
                SELECT * FROM operations_notification_outbox WHERE outcome_id = ?
                """,
                (outcome.outcome_id,),
            ).fetchone()
            if existing is not None:
                record = _notification_record(existing)
                if (
                    record.notification.level != notification.level
                    or record.notification.human_review_required
                    != notification.human_review_required
                ):
                    raise ValueError("conflicting notification for monitoring outcome")
                return record
            conn.execute(
                """
                INSERT INTO operations_notification_outbox(
                    notification_id, outcome_id, schedule_id, period_key, run_id,
                    schema_version, payload_json, payload_hash, status,
                    attempt_count, max_attempts, initial_backoff_seconds,
                    backoff_multiplier, max_backoff_seconds, next_attempt_at,
                    lease_owner, lease_token, lease_expires_at, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?,
                    NULL, NULL, NULL, ?, ?
                )
                """,
                (
                    notification.notification_id,
                    outcome.outcome_id,
                    outcome.schedule_id,
                    outcome.period_key,
                    outcome.run_id,
                    notification.schema_version,
                    payload_json,
                    payload_hash,
                    retry_policy.max_attempts,
                    retry_policy.initial_backoff_seconds,
                    retry_policy.multiplier,
                    retry_policy.max_backoff_seconds,
                    _iso_z(now),
                    _iso_z(now),
                    _iso_z(now),
                ),
            )
            _write_event(
                conn,
                schedule_id=outcome.schedule_id,
                period_key=outcome.period_key,
                run_id=outcome.run_id,
                event_type="notification_queued",
                payload={
                    "notification_id": notification.notification_id,
                    "payload_hash": payload_hash,
                },
                created_at=now,
            )
            row = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE notification_id = ?
                """,
                (notification.notification_id,),
            ).fetchone()
            assert row is not None
            return _notification_record(row)

    def claim_notification(
        self,
        *,
        owner_id: str,
        lease_seconds: int = 30,
    ) -> NotificationClaim | None:
        normalized_owner = _required_text(owner_id, "owner_id")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds < 1
        ):
            raise ValueError("lease_seconds must be a positive integer")
        now = _utc_datetime(self._clock(), "clock")
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        lease_token = uuid.uuid4().hex
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE status = 'pending' AND next_attempt_at <= ?
                 ORDER BY next_attempt_at, created_at, notification_id
                 LIMIT 1
                """,
                (_iso_z(now),),
            ).fetchone()
            if row is None:
                return None
            attempt_count = int(row["attempt_count"]) + 1
            conn.execute(
                """
                UPDATE operations_notification_outbox
                   SET status = 'sending', attempt_count = ?, lease_owner = ?,
                       lease_token = ?, lease_expires_at = ?, updated_at = ?
                 WHERE notification_id = ?
                """,
                (
                    attempt_count,
                    normalized_owner,
                    lease_token,
                    _iso_z(lease_expires_at),
                    _iso_z(now),
                    str(row["notification_id"]),
                ),
            )
            _write_event(
                conn,
                schedule_id=str(row["schedule_id"]),
                period_key=str(row["period_key"]),
                run_id=str(row["run_id"]),
                event_type="notification_send_claimed",
                payload={
                    "notification_id": str(row["notification_id"]),
                    "attempt_number": attempt_count,
                    "lease_expires_at": _iso_z(lease_expires_at),
                },
                created_at=now,
            )
            updated = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE notification_id = ?
                """,
                (str(row["notification_id"]),),
            ).fetchone()
            assert updated is not None
            return NotificationClaim(
                notification=_notification_record(updated),
                owner_id=normalized_owner,
                lease_token=lease_token,
                lease_expires_at=lease_expires_at,
            )

    def complete_notification(self, claim: NotificationClaim) -> NotificationRecord:
        return self._finish_notification(claim, error_code=None)

    def fail_notification(
        self,
        claim: NotificationClaim,
        *,
        error_code: str,
    ) -> NotificationRecord:
        if not isinstance(error_code, str) or not _ERROR_CODE_RE.fullmatch(error_code):
            raise ValueError("error_code must be a lower_snake_case identifier")
        return self._finish_notification(claim, error_code=error_code)

    def _finish_notification(
        self,
        claim: NotificationClaim,
        *,
        error_code: str | None,
    ) -> NotificationRecord:
        if not isinstance(claim, NotificationClaim):
            raise ValueError("claim must be a NotificationClaim")
        now = _utc_datetime(self._clock(), "clock")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE notification_id = ?
                """,
                (claim.notification.notification_id,),
            ).fetchone()
            if (
                row is None
                or str(row["status"]) != "sending"
                or str(row["lease_token"]) != claim.lease_token
                or _optional_datetime(row["lease_expires_at"]) is None
                or _optional_datetime(row["lease_expires_at"]) <= now
            ):
                raise StaleNotificationLeaseError(
                    "notification lease is no longer current"
                )
            if error_code is None:
                status = "sent"
                next_attempt_at = None
                event_type = "notification_sent"
                payload: dict[str, Any] = {
                    "notification_id": str(row["notification_id"]),
                    "attempt_number": int(row["attempt_count"]),
                }
            else:
                policy = _notification_retry_policy(row)
                terminal = int(row["attempt_count"]) >= policy.max_attempts
                status = "exhausted" if terminal else "pending"
                next_attempt_at = None
                if not terminal:
                    next_attempt_at = now + timedelta(
                        seconds=policy.delay_after_failure(int(row["attempt_count"]))
                    )
                event_type = "notification_send_failed"
                payload = {
                    "notification_id": str(row["notification_id"]),
                    "attempt_number": int(row["attempt_count"]),
                    "error_code": error_code,
                    "terminal": terminal,
                    "next_attempt_at": (
                        None
                        if next_attempt_at is None
                        else _iso_z(next_attempt_at)
                    ),
                }
            conn.execute(
                """
                UPDATE operations_notification_outbox
                   SET status = ?, next_attempt_at = ?, lease_owner = NULL,
                       lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                 WHERE notification_id = ?
                """,
                (
                    status,
                    None if next_attempt_at is None else _iso_z(next_attempt_at),
                    _iso_z(now),
                    str(row["notification_id"]),
                ),
            )
            _write_event(
                conn,
                schedule_id=str(row["schedule_id"]),
                period_key=str(row["period_key"]),
                run_id=str(row["run_id"]),
                event_type=event_type,
                payload=payload,
                created_at=now,
            )
            updated = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE notification_id = ?
                """,
                (str(row["notification_id"]),),
            ).fetchone()
            assert updated is not None
            return _notification_record(updated)

    def recover_expired_notification_leases(self) -> tuple[NotificationRecord, ...]:
        now = _utc_datetime(self._clock(), "clock")
        recovered: list[NotificationRecord] = []
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE status = 'sending' AND lease_expires_at <= ?
                 ORDER BY lease_expires_at, notification_id
                """,
                (_iso_z(now),),
            ).fetchall()
            for row in rows:
                policy = _notification_retry_policy(row)
                terminal = int(row["attempt_count"]) >= policy.max_attempts
                status = "exhausted" if terminal else "pending"
                next_attempt_at = None
                if not terminal:
                    next_attempt_at = now + timedelta(
                        seconds=policy.delay_after_failure(int(row["attempt_count"]))
                    )
                conn.execute(
                    """
                    UPDATE operations_notification_outbox
                       SET status = ?, next_attempt_at = ?, lease_owner = NULL,
                           lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                     WHERE notification_id = ?
                    """,
                    (
                        status,
                        (
                            None
                            if next_attempt_at is None
                            else _iso_z(next_attempt_at)
                        ),
                        _iso_z(now),
                        str(row["notification_id"]),
                    ),
                )
                _write_event(
                    conn,
                    schedule_id=str(row["schedule_id"]),
                    period_key=str(row["period_key"]),
                    run_id=str(row["run_id"]),
                    event_type="notification_lease_expired",
                    payload={
                        "notification_id": str(row["notification_id"]),
                        "attempt_number": int(row["attempt_count"]),
                        "terminal": terminal,
                        "next_attempt_at": (
                            None
                            if next_attempt_at is None
                            else _iso_z(next_attempt_at)
                        ),
                    },
                    created_at=now,
                )
                updated = conn.execute(
                    """
                    SELECT * FROM operations_notification_outbox
                     WHERE notification_id = ?
                    """,
                    (str(row["notification_id"]),),
                ).fetchone()
                assert updated is not None
                recovered.append(_notification_record(updated))
        return tuple(recovered)

    def list_notifications(
        self,
        schedule_id: str,
        period_key: str,
    ) -> tuple[NotificationRecord, ...]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM operations_notification_outbox
                 WHERE schedule_id = ? AND period_key = ?
                 ORDER BY created_at, notification_id
                """,
                (
                    _required_text(schedule_id, "schedule_id"),
                    _required_text(period_key, "period_key"),
                ),
            ).fetchall()
        return tuple(_notification_record(row) for row in rows)


def _schedule_record(row) -> ScheduleRecord:
    payload = json.loads(str(row["contract_json"]))
    if not isinstance(payload, dict):
        raise ValueError("persisted schedule contract must be an object")
    return ScheduleRecord(
        contract=ScheduleContract.from_dict(payload),
        contract_hash=str(row["contract_hash"]),
        created_at=_parse_datetime(str(row["created_at"]), "created_at"),
    )


def _period_record(row) -> PeriodRecord:
    starts_at = _parse_datetime(str(row["period_starts_at"]), "period_starts_at")
    ends_at = _parse_datetime(str(row["period_ends_at"]), "period_ends_at")
    return PeriodRecord(
        schedule_id=str(row["schedule_id"]),
        period=CalendarPeriod(
            key=str(row["period_key"]),
            index=int(row["period_index"]),
            starts_at=starts_at,
            ends_at=ends_at,
        ),
        schedule_revision=int(row["schedule_revision"]),
        state=str(row["state"]),
        attempt_count=int(row["attempt_count"]),
        lease_owner=_optional_text(row["lease_owner"]),
        lease_expires_at=_optional_datetime(row["lease_expires_at"]),
        next_attempt_at=_optional_datetime(row["next_attempt_at"]),
        outcome_id=_optional_text(row["outcome_id"]),
        updated_at=_parse_datetime(str(row["updated_at"]), "updated_at"),
    )


def _run_record(row) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        schedule_id=str(row["schedule_id"]),
        period_key=str(row["period_key"]),
        schedule_revision=int(row["schedule_revision"]),
        attempt_number=int(row["attempt_number"]),
        owner_id=str(row["owner_id"]),
        claimed_at=_parse_datetime(str(row["claimed_at"]), "claimed_at"),
        lease_expires_at=_parse_datetime(
            str(row["lease_expires_at"]), "lease_expires_at"
        ),
    )


def _outcome_record(row) -> OutcomeRecord:
    payload = _json_object(row["outcome_json"], "outcome_json")
    return OutcomeRecord(
        outcome_id=str(row["outcome_id"]),
        run_id=str(row["run_id"]),
        schedule_id=str(row["schedule_id"]),
        period_key=str(row["period_key"]),
        outcome=MonitoringOutcome.from_dict(payload),
        outcome_hash=str(row["outcome_hash"]),
        recorded_at=_parse_datetime(str(row["recorded_at"]), "recorded_at"),
    )


def _event_record(row) -> EventRecord:
    return EventRecord(
        sequence=int(row["sequence"]),
        event_id=str(row["event_id"]),
        schedule_id=str(row["schedule_id"]),
        period_key=str(row["period_key"]),
        run_id=_optional_text(row["run_id"]),
        event_type=str(row["event_type"]),
        payload=_json_object(row["payload_json"], "payload_json"),
        created_at=_parse_datetime(str(row["created_at"]), "created_at"),
    )


def _notification_record(row) -> NotificationRecord:
    payload = _json_object(row["payload_json"], "payload_json")
    return NotificationRecord(
        notification_id=str(row["notification_id"]),
        outcome_id=str(row["outcome_id"]),
        schedule_id=str(row["schedule_id"]),
        period_key=str(row["period_key"]),
        run_id=str(row["run_id"]),
        notification=RedactedNotification.from_dict(payload),
        payload_hash=str(row["payload_hash"]),
        status=str(row["status"]),
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        next_attempt_at=_optional_datetime(row["next_attempt_at"]),
        lease_owner=_optional_text(row["lease_owner"]),
        lease_expires_at=_optional_datetime(row["lease_expires_at"]),
        created_at=_parse_datetime(str(row["created_at"]), "created_at"),
        updated_at=_parse_datetime(str(row["updated_at"]), "updated_at"),
    )


def _notification_retry_policy(row) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=int(row["max_attempts"]),
        initial_backoff_seconds=float(row["initial_backoff_seconds"]),
        multiplier=float(row["backoff_multiplier"]),
        max_backoff_seconds=float(row["max_backoff_seconds"]),
    )


def _write_event(
    conn,
    *,
    schedule_id: str,
    period_key: str,
    run_id: str | None,
    event_type: str,
    payload: Mapping[str, Any],
    created_at: datetime,
) -> None:
    conn.execute(
        """
        INSERT INTO operations_events(
            event_id, schedule_id, period_key, run_id, event_type,
            payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            schedule_id,
            period_key,
            run_id,
            event_type,
            _canonical_json(dict(payload)),
            _iso_z(created_at),
        ),
    )


def _record_attempt_failure(
    conn,
    *,
    row,
    schedule: ScheduleContract,
    run_id: str,
    event_type: str,
    event_payload: Mapping[str, Any],
    now: datetime,
) -> PeriodRecord:
    attempt_count = int(row["attempt_count"])
    terminal = attempt_count >= schedule.retry_policy.max_attempts
    next_attempt_at = None
    if not terminal:
        next_attempt_at = now + timedelta(
            seconds=schedule.retry_policy.delay_after_failure(attempt_count)
        )
    state = "terminal_failed" if terminal else "retry_wait"
    schedule_id = str(row["schedule_id"])
    period_key = str(row["period_key"])
    conn.execute(
        """
        UPDATE operations_periods
           SET state = ?, lease_owner = NULL, lease_token = NULL,
               lease_expires_at = NULL, next_attempt_at = ?, updated_at = ?
         WHERE schedule_id = ? AND period_key = ?
        """,
        (
            state,
            None if next_attempt_at is None else _iso_z(next_attempt_at),
            _iso_z(now),
            schedule_id,
            period_key,
        ),
    )
    payload = {
        **dict(event_payload),
        "attempt_number": attempt_count,
        "terminal": terminal,
        "next_attempt_at": (
            None if next_attempt_at is None else _iso_z(next_attempt_at)
        ),
    }
    _write_event(
        conn,
        schedule_id=schedule_id,
        period_key=period_key,
        run_id=run_id,
        event_type=event_type,
        payload=payload,
        created_at=now,
    )
    updated = conn.execute(
        """
        SELECT * FROM operations_periods
         WHERE schedule_id = ? AND period_key = ?
        """,
        (schedule_id, period_key),
    ).fetchone()
    assert updated is not None
    return _period_record(updated)


def _load_schedule_on_connection(
    conn,
    schedule_id: str,
    revision: int,
) -> ScheduleContract:
    row = conn.execute(
        """
        SELECT contract_json FROM operations_schedules
         WHERE schedule_id = ? AND revision = ?
        """,
        (schedule_id, revision),
    ).fetchone()
    if row is None:
        raise KeyError(
            f"schedule revision not found: {schedule_id} revision {revision}"
        )
    return ScheduleContract.from_dict(
        _json_object(row["contract_json"], "contract_json")
    )


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_object(value: object, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else _parse_datetime(str(value), "datetime")


def _utc_datetime(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _parse_datetime(value: str, field: str) -> datetime:
    try:
        return _utc_datetime(
            datetime.fromisoformat(value.replace("Z", "+00:00")), field
        )
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO datetime") from exc


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

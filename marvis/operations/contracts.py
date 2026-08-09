"""Typed contracts for the local operations scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
import re
from typing import Any, Mapping


CALENDAR_SCHEMA_VERSION = "operations.calendar.fixed_interval.v1"
RETRY_SCHEMA_VERSION = "operations.retry.v1"
SCHEDULE_SCHEMA_VERSION = "operations.schedule.v1"
MAX_CATCH_UP_BUDGET = 100
MAX_RETRY_ATTEMPTS = 10
HUMAN_ESCALATION_SCHEMA_VERSION = "operations.human_escalation.v1"
MONITORING_OUTCOME_SCHEMA_VERSION = "operations.monitoring_outcome.v1"
_MONITORING_LEVELS = frozenset({"green", "amber", "red", "not_available"})
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CalendarPeriod:
    """One immutable, canonical calendar window."""

    key: str
    index: int
    starts_at: datetime
    ends_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _required_text(self.key, "period key"))
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("period index must be a non-negative integer")
        starts_at = _utc_datetime(self.starts_at, "period starts_at")
        ends_at = _utc_datetime(self.ends_at, "period ends_at")
        if ends_at <= starts_at:
            raise ValueError("period ends_at must be after starts_at")
        object.__setattr__(self, "starts_at", starts_at)
        object.__setattr__(self, "ends_at", ends_at)


@dataclass(frozen=True)
class FixedIntervalCalendar:
    """UTC fixed-interval calendar with stable period identities."""

    anchor_at: datetime
    interval_seconds: int
    schema_version: str = CALENDAR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_at", _utc_datetime(self.anchor_at, "anchor_at"))
        if self.schema_version != CALENDAR_SCHEMA_VERSION:
            raise ValueError(f"unsupported calendar schema: {self.schema_version}")
        if (
            isinstance(self.interval_seconds, bool)
            or not isinstance(self.interval_seconds, int)
            or self.interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a positive integer")

    def due_periods(
        self,
        *,
        active_from: datetime,
        now: datetime,
        limit: int,
    ) -> tuple[CalendarPeriod, ...]:
        """Return at most ``limit`` complete periods from activation onward."""

        start = _utc_datetime(active_from, "active_from")
        reference = _utc_datetime(now, "now")
        if start < self.anchor_at:
            raise ValueError("active_from must not precede calendar anchor_at")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        first_index = math.ceil(
            (start - self.anchor_at).total_seconds() / self.interval_seconds
        )
        periods: list[CalendarPeriod] = []
        for index in range(first_index, first_index + limit):
            period = self.period(index)
            if period.ends_at > reference:
                break
            periods.append(period)
        return tuple(periods)

    def period(self, index: int) -> CalendarPeriod:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("period index must be a non-negative integer")
        starts_at = self.anchor_at + timedelta(
            seconds=index * self.interval_seconds
        )
        ends_at = starts_at + timedelta(seconds=self.interval_seconds)
        return CalendarPeriod(
            key=f"{self.schema_version}:{_iso_z(starts_at)}/{_iso_z(ends_at)}",
            index=index,
            starts_at=starts_at,
            ends_at=ends_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "anchor_at": _iso_z(self.anchor_at),
            "interval_seconds": self.interval_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FixedIntervalCalendar:
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            anchor_at=_parse_datetime(payload.get("anchor_at"), "anchor_at"),
            interval_seconds=_required_int(
                payload.get("interval_seconds"), "interval_seconds"
            ),
        )


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry contract."""

    max_attempts: int
    initial_backoff_seconds: float
    multiplier: float
    max_backoff_seconds: float
    schema_version: str = RETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RETRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported retry schema: {self.schema_version}")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= MAX_RETRY_ATTEMPTS
        ):
            raise ValueError(
                f"max_attempts must be between 1 and {MAX_RETRY_ATTEMPTS}"
            )
        initial = _non_negative_number(
            self.initial_backoff_seconds, "initial_backoff_seconds"
        )
        multiplier = _positive_number(self.multiplier, "multiplier")
        maximum = _non_negative_number(
            self.max_backoff_seconds, "max_backoff_seconds"
        )
        if maximum < initial:
            raise ValueError(
                "max_backoff_seconds must be at least initial_backoff_seconds"
            )
        object.__setattr__(self, "initial_backoff_seconds", initial)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "max_backoff_seconds", maximum)

    def delay_after_failure(self, attempt_number: int) -> float:
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise ValueError("attempt_number must be a positive integer")
        return min(
            self.initial_backoff_seconds
            * (self.multiplier ** (attempt_number - 1)),
            self.max_backoff_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_attempts": self.max_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "multiplier": self.multiplier,
            "max_backoff_seconds": self.max_backoff_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetryPolicy:
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            max_attempts=_required_int(payload.get("max_attempts"), "max_attempts"),
            initial_backoff_seconds=_required_number(
                payload.get("initial_backoff_seconds"), "initial_backoff_seconds"
            ),
            multiplier=_required_number(payload.get("multiplier"), "multiplier"),
            max_backoff_seconds=_required_number(
                payload.get("max_backoff_seconds"), "max_backoff_seconds"
            ),
        )


@dataclass(frozen=True)
class ScheduleContract:
    """Immutable revision of one local monitoring schedule."""

    schedule_id: str
    revision: int
    monitoring_ref: str
    active_from: datetime
    calendar: FixedIntervalCalendar
    catch_up_budget: int
    lease_seconds: int
    retry_policy: RetryPolicy
    enabled: bool = True
    schema_version: str = SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schedule schema: {self.schema_version}")
        object.__setattr__(
            self, "schedule_id", _required_text(self.schedule_id, "schedule_id")
        )
        object.__setattr__(
            self, "monitoring_ref", _required_text(self.monitoring_ref, "monitoring_ref")
        )
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValueError("revision must be a positive integer")
        if not isinstance(self.calendar, FixedIntervalCalendar):
            raise ValueError("calendar must be a FixedIntervalCalendar")
        active_from = _utc_datetime(self.active_from, "active_from")
        if active_from < self.calendar.anchor_at:
            raise ValueError("active_from must not precede calendar anchor_at")
        object.__setattr__(self, "active_from", active_from)
        if (
            isinstance(self.catch_up_budget, bool)
            or not isinstance(self.catch_up_budget, int)
            or not 1 <= self.catch_up_budget <= MAX_CATCH_UP_BUDGET
        ):
            raise ValueError(
                f"catch_up_budget must be between 1 and {MAX_CATCH_UP_BUDGET}"
            )
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, int)
            or self.lease_seconds < 1
        ):
            raise ValueError("lease_seconds must be a positive integer")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise ValueError("retry_policy must be a RetryPolicy")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "revision": self.revision,
            "monitoring_ref": self.monitoring_ref,
            "active_from": _iso_z(self.active_from),
            "calendar": self.calendar.to_dict(),
            "catch_up_budget": self.catch_up_budget,
            "lease_seconds": self.lease_seconds,
            "retry_policy": self.retry_policy.to_dict(),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScheduleContract:
        calendar = payload.get("calendar")
        retry_policy = payload.get("retry_policy")
        if not isinstance(calendar, Mapping):
            raise ValueError("calendar must be an object")
        if not isinstance(retry_policy, Mapping):
            raise ValueError("retry_policy must be an object")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            schedule_id=str(payload.get("schedule_id", "")),
            revision=_required_int(payload.get("revision"), "revision"),
            monitoring_ref=str(payload.get("monitoring_ref", "")),
            active_from=_parse_datetime(payload.get("active_from"), "active_from"),
            calendar=FixedIntervalCalendar.from_dict(calendar),
            catch_up_budget=_required_int(
                payload.get("catch_up_budget"), "catch_up_budget"
            ),
            lease_seconds=_required_int(
                payload.get("lease_seconds"), "lease_seconds"
            ),
            retry_policy=RetryPolicy.from_dict(retry_policy),
            enabled=_required_bool(payload.get("enabled"), "enabled"),
        )


@dataclass(frozen=True)
class HumanEscalationSuggestion:
    """A review hint that deliberately carries no side-effect authority."""

    reason_code: str
    schema_version: str = HUMAN_ESCALATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HUMAN_ESCALATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported human escalation schema: {self.schema_version}"
            )
        if not isinstance(self.reason_code, str) or not _REASON_CODE_RE.fullmatch(
            self.reason_code
        ):
            raise ValueError("reason_code must be a lower_snake_case identifier")

    @property
    def action(self) -> str:
        return "review_monitoring_result"

    @property
    def requires_human_confirmation(self) -> bool:
        return True

    @property
    def automatic_action_permitted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action": self.action,
            "reason_code": self.reason_code,
            "requires_human_confirmation": self.requires_human_confirmation,
            "automatic_action_permitted": self.automatic_action_permitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HumanEscalationSuggestion:
        if payload.get("action") != "review_monitoring_result":
            raise ValueError("unsupported escalation action")
        if payload.get("requires_human_confirmation") is not True:
            raise ValueError("escalation must require human confirmation")
        if payload.get("automatic_action_permitted") is not False:
            raise ValueError("escalation must forbid automatic action")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            reason_code=str(payload.get("reason_code", "")),
        )


@dataclass(frozen=True)
class MonitoringOutcome:
    """Deterministic result reference persisted before notification delivery."""

    level: str
    evidence_ref: str
    evidence_hash: str
    escalation: HumanEscalationSuggestion | None = None
    schema_version: str = MONITORING_OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MONITORING_OUTCOME_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported monitoring outcome schema: {self.schema_version}"
            )
        if self.level not in _MONITORING_LEVELS:
            raise ValueError(f"unsupported monitoring level: {self.level}")
        object.__setattr__(
            self, "evidence_ref", _required_text(self.evidence_ref, "evidence_ref")
        )
        if not isinstance(self.evidence_hash, str) or not _SHA256_RE.fullmatch(
            self.evidence_hash
        ):
            raise ValueError("evidence_hash must be a lowercase SHA-256 hex digest")
        if self.escalation is not None and not isinstance(
            self.escalation, HumanEscalationSuggestion
        ):
            raise ValueError("escalation must be a HumanEscalationSuggestion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "level": self.level,
            "evidence_ref": self.evidence_ref,
            "evidence_hash": self.evidence_hash,
            "escalation": (
                None if self.escalation is None else self.escalation.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MonitoringOutcome:
        raw_escalation = payload.get("escalation")
        if raw_escalation is not None and not isinstance(raw_escalation, Mapping):
            raise ValueError("escalation must be an object or null")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            level=str(payload.get("level", "")),
            evidence_ref=str(payload.get("evidence_ref", "")),
            evidence_hash=str(payload.get("evidence_hash", "")),
            escalation=(
                None
                if raw_escalation is None
                else HumanEscalationSuggestion.from_dict(raw_escalation)
            ),
        )


def _utc_datetime(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO datetime") from exc
    return _utc_datetime(parsed, field)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise ValueError(f"{field} must be non-empty text up to 200 characters")
    return value.strip()


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _non_negative_number(value: object, field: str) -> float:
    number = _required_number(value, field)
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _positive_number(value: object, field: str) -> float:
    number = _required_number(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number

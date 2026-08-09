"""Redacted notification contracts and outbound adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import Any, Mapping, Protocol, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from marvis.operations.repository import OutcomeRecord


REDACTED_NOTIFICATION_SCHEMA_VERSION = "operations.notification.redacted.v1"
_TOKEN_RE = re.compile(r"tok_[0-9a-f]{32}")


@dataclass(frozen=True)
class RedactedNotification:
    """The complete allowlist payload visible to an external adapter."""

    notification_id: str
    schedule_token: str
    period_token: str
    run_token: str
    outcome_token: str
    level: str
    human_review_required: bool
    schema_version: str = REDACTED_NOTIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REDACTED_NOTIFICATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported redacted notification schema: {self.schema_version}"
            )
        if not isinstance(self.notification_id, str) or not self.notification_id:
            raise ValueError("notification_id must be non-empty text")
        for field, value in (
            ("schedule_token", self.schedule_token),
            ("period_token", self.period_token),
            ("run_token", self.run_token),
            ("outcome_token", self.outcome_token),
        ):
            if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
                raise ValueError(f"{field} must be a redacted token")
        if self.level not in {"green", "amber", "red", "not_available"}:
            raise ValueError(f"unsupported monitoring level: {self.level}")
        if not isinstance(self.human_review_required, bool):
            raise ValueError("human_review_required must be a boolean")

    @property
    def event_type(self) -> str:
        return "monitoring_completed"

    @property
    def automatic_action_permitted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "notification_id": self.notification_id,
            "event_type": self.event_type,
            "schedule_token": self.schedule_token,
            "period_token": self.period_token,
            "run_token": self.run_token,
            "outcome_token": self.outcome_token,
            "level": self.level,
            "human_review_required": self.human_review_required,
            "automatic_action_permitted": self.automatic_action_permitted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RedactedNotification:
        if payload.get("event_type") != "monitoring_completed":
            raise ValueError("unsupported notification event_type")
        if payload.get("automatic_action_permitted") is not False:
            raise ValueError("notification must forbid automatic action")
        human_review_required = payload.get("human_review_required")
        if not isinstance(human_review_required, bool):
            raise ValueError("human_review_required must be a boolean")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            notification_id=str(payload.get("notification_id", "")),
            schedule_token=str(payload.get("schedule_token", "")),
            period_token=str(payload.get("period_token", "")),
            run_token=str(payload.get("run_token", "")),
            outcome_token=str(payload.get("outcome_token", "")),
            level=str(payload.get("level", "")),
            human_review_required=human_review_required,
        )


class NotificationAdapter(Protocol):
    """External delivery seam; adapters never receive raw monitoring records."""

    def send(self, notification: RedactedNotification) -> None: ...


class NotificationRedactor:
    """Tokenize internal identifiers and drop all non-allowlisted evidence data."""

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("redaction secret must contain at least 16 bytes")
        self._secret = secret

    def redact(self, outcome: OutcomeRecord) -> RedactedNotification:
        return RedactedNotification(
            notification_id=uuid.uuid4().hex,
            schedule_token=self._token("schedule", outcome.schedule_id),
            period_token=self._token("period", outcome.period_key),
            run_token=self._token("run", outcome.run_id),
            outcome_token=self._token("outcome", outcome.outcome_id),
            level=outcome.outcome.level,
            human_review_required=outcome.outcome.escalation is not None,
        )

    def _token(self, domain: str, value: str) -> str:
        digest = hmac.new(
            self._secret,
            f"{domain}\0{value}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"tok_{digest[:32]}"

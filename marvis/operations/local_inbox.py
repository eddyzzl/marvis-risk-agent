"""Idempotent local application delivery, separate from the scheduler outbox."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json

from marvis.db_schema import connect
from marvis.operations.notifications import RedactedNotification


class LocalInboxAdapter:
    """Commit a recipient-visible inbox item before returning delivery success.

    This receipt means delivered to the local app inbox, not read by a person,
    shown by the OS, or delivered to an institution's external channel.
    """

    def __init__(self, db_path, *, clock=None):
        self.db_path = db_path
        self.clock = clock or (lambda: datetime.now(UTC))

    def send(self, notification: RedactedNotification) -> None:
        payload = json.dumps(
            notification.to_dict(), sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT payload_hash FROM operations_local_inbox WHERE notification_id = ?",
                (notification.notification_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != digest:
                    raise ValueError(
                        "notification identity reused with changed content"
                    )
                return
            conn.execute(
                "INSERT INTO operations_local_inbox(notification_id, payload_json, payload_hash, delivered_at) VALUES (?, ?, ?, ?)",
                (
                    notification.notification_id,
                    payload,
                    digest,
                    self.clock().isoformat(),
                ),
            )

    def list(self, *, limit=100):
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT i.*, o.schedule_id, o.period_key, o.outcome_id
                FROM operations_local_inbox i JOIN operations_notification_outbox o
                ON o.notification_id = i.notification_id ORDER BY i.delivered_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "notification_id": row["notification_id"],
                "payload": json.loads(row["payload_json"]),
                "payload_hash": row["payload_hash"],
                "delivered_at": row["delivered_at"],
                "delivery_target": "local_application_inbox",
                "read_at": row["read_at"],
                "read_by": row["read_by"],
                "schedule_id": row["schedule_id"],
                "period_key": row["period_key"],
                "outcome_id": row["outcome_id"],
            }
            for row in rows
        ]

    def acknowledge(self, notification_id: str, principal_id: str):
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """UPDATE operations_local_inbox SET read_at = COALESCE(read_at, ?),
                read_by = COALESCE(read_by, ?) WHERE notification_id = ?""",
                (self.clock().isoformat(), principal_id, notification_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(notification_id)

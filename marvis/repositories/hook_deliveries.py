"""Durable completion snapshots and one-attempt Hook delivery claims.

A started delivery is deliberately never reclaimed automatically. The process
may have completed its external side effect before losing its receipt. This is
deduplication, not an exactly-once promise or a second execution runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from marvis.db_schema import connect
from marvis.state_machine import ConflictError


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class HookDeliveryRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def load_checkpoint(self, identity: str, binding_hash: str) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT binding_hash, snapshot_json FROM hook_completion_checkpoints WHERE identity = ?",
                (identity,),
            ).fetchone()
        if row is None:
            return None
        if row["binding_hash"] != binding_hash:
            raise ConflictError("completion checkpoint binding changed")
        return json.loads(row["snapshot_json"])

    def store_checkpoint(
        self, identity: str, binding_hash: str, snapshot: dict
    ) -> dict:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO hook_completion_checkpoints VALUES (?, ?, ?, ?)",
                (identity, binding_hash, _json(snapshot), _now()),
            )
        # First writer wins; concurrent reviewers must dispatch its same payload.
        return self.load_checkpoint(identity, binding_hash)

    def prepare_event(
        self,
        *,
        event_id: str,
        event: str,
        task_id: str,
        payload_hash: str | None = None,
        targets: list[dict],
    ) -> list[dict]:
        return self.prepare_events(
            [
                {
                    "event_id": event_id,
                    "event": event,
                    "task_id": task_id,
                    "payload_hash": payload_hash,
                    "targets": targets,
                }
            ]
        )[0]

    def prepare_events(self, events: list[dict]) -> list[list[dict]]:
        """Freeze all obligations of one completion in a single transaction."""
        results = []
        with connect(self.db_path) as conn:
            for item in events:
                event_id, event, task_id = (
                    item["event_id"],
                    item["event"],
                    item["task_id"],
                )
                bound_hash = item.get("payload_hash")
                conn.execute(
                    "INSERT OR IGNORE INTO hook_events VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        event,
                        task_id,
                        bound_hash or "",
                        _json(item["targets"]),
                        _now(),
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM hook_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                if (row["event"], row["task_id"]) != (event, task_id) or (
                    bound_hash
                    and row["payload_hash"]
                    and row["payload_hash"] != bound_hash
                ):
                    raise ConflictError(
                        "hook event identity reused with different binding"
                    )
                if bound_hash and not row["payload_hash"]:
                    conn.execute(
                        "UPDATE hook_events SET payload_hash = ? WHERE event_id = ? AND payload_hash = ''",
                        (bound_hash, event_id),
                    )
                results.append(json.loads(row["targets_json"]))
        return results

    def claim(
        self, event_id: str, target_ref: str, *, required: bool
    ) -> tuple[bool, dict]:
        now = _now()
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO hook_deliveries
                   VALUES (?, ?, ?, 'started', NULL, ?, ?)""",
                (event_id, target_ref, int(required), now, now),
            )
            claimed = cursor.rowcount == 1
            row = conn.execute(
                "SELECT * FROM hook_deliveries WHERE event_id = ? AND target_ref = ?",
                (event_id, target_ref),
            ).fetchone()
        return claimed, dict(row)

    def finish(
        self, event_id: str, target_ref: str, *, status: str, result: dict
    ) -> None:
        if status not in {"succeeded", "failed", "unknown"}:
            raise ValueError("invalid hook delivery terminal status")
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """UPDATE hook_deliveries SET status = ?, result_json = ?, updated_at = ?
                   WHERE event_id = ? AND target_ref = ? AND status = 'started'""",
                (status, _json(result), _now(), event_id, target_ref),
            )
            if cursor.rowcount != 1:
                raise ConflictError("hook delivery is no longer owned by this attempt")

    def list_deliveries(self, event_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM hook_deliveries WHERE event_id = ? ORDER BY target_ref",
                (event_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def event_targets(self, event_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT targets_json FROM hook_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return json.loads(row["targets_json"]) if row is not None else []

"""Native-only aggregate checkpoints; publication is a one-way execution phase."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import time

from marvis.db_schema import connect
from marvis.strategy_workbench.contracts import (
    SEARCH_TOOLS,
    WorkbenchError,
    canonical,
    digest,
)
from marvis.strategy_workbench.service import public_search, require_owner

CHECKPOINT_VERSION = "strategy.native-search-checkpoint.v1"


class SearchInterrupted(Exception):
    def __init__(self, state):
        self.state = state
        super().__init__(
            "策略搜索已暂停，已保留候选检查点"
            if state == "paused"
            else "策略搜索已取消，已保留已完成进度"
        )

    def to_detail(self):
        return {"kind": "search_" + self.state, "message": str(self)}


def _key(settings):
    root = settings.workspace / ".strategy_workbench"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / "checkpoint-key"
    import uuid

    temporary = root / ("checkpoint-key-" + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(os.urandom(32))
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    if path.is_symlink():
        raise WorkbenchError("workbench_checkpoint_key_unavailable")
    key = path.read_bytes()
    if len(key) != 32:
        raise WorkbenchError("workbench_checkpoint_key_unavailable")
    return key


def _sign(key, value):
    return hmac.new(key, canonical(value).encode(), hashlib.sha256).hexdigest()


def producer_hash():
    root = Path(__file__).parents[1]
    paths = set((root / "packs/strategy").rglob("*.py"))
    paths.update((root / "data").rglob("*.py"))
    paths.update(
        (root / name)
        for name in (
            "strategy_workbench/control.py",
            "strategy_workbench/contracts.py",
            "plugins/contracts.py",
            "plugins/sdk.py",
        )
    )
    names = sorted(str(path.relative_to(root)) for path in paths)
    return digest(
        {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    )


class NativeSearchControl:
    def __init__(self, runtime, ctx, inputs):
        self.settings = runtime.settings
        self.ctx = ctx
        self.db_path = runtime.settings.db_path
        self.run_id = getattr(ctx, "invocation_id", None)
        self.id = None
        self.saved = []
        self.total = 0
        if not self.run_id:
            return
        self.key = _key(runtime.settings)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT s.* FROM strategy_workbench_searches s JOIN plan_step_runs r ON r.step_id=s.step_id WHERE r.id=? AND r.tool_ref=s.tool_ref AND r.status='running' AND s.task_id=?",
                (self.run_id, ctx.task_id),
            ).fetchone()
            if row is None:
                return
            require_owner(conn, row["actor_id"])
            run = conn.execute(
                "SELECT input_json,invocation_contract_json FROM plan_step_runs WHERE id=?",
                (self.run_id,),
            ).fetchone()
            if (
                not run["invocation_contract_json"]
                or json.loads(run["input_json"]) != inputs
            ):
                raise WorkbenchError("workbench_native_input_binding_changed")
            binding = digest(
                {
                    "inputs": inputs,
                    "invocation": json.loads(run["invocation_contract_json"]),
                    "producer": producer_hash(),
                }
            )
            if row["input_hash"] and row["input_hash"] != binding:
                raise WorkbenchError("workbench_search_inputs_or_producer_changed")
            if row["state"] not in {"queued", "resuming"}:
                raise WorkbenchError("workbench_search_not_dispatchable")
            self.id = row["id"]
            self.task_id, self.actor_id, self.tool_ref = (
                row["task_id"],
                row["actor_id"],
                row["tool_ref"],
            )
            self.data_hash = row["data_hash"]
            if row["checkpoint_json"]:
                checkpoint = json.loads(row["checkpoint_json"])
                if (
                    not hmac.compare_digest(
                        _sign(self.key, checkpoint), row["checkpoint_mac"] or ""
                    )
                    or checkpoint["input_hash"] != binding
                    or checkpoint["search_id"] != self.id
                    or checkpoint["task_id"] != self.task_id
                    or checkpoint["actor_id"] != self.actor_id
                    or checkpoint["tool_ref"] != self.tool_ref
                    or checkpoint["data_hash"] != self.data_hash
                ):
                    raise WorkbenchError("workbench_checkpoint_authentication_failed")
                self.saved = checkpoint["aggregates"]
                self.total = checkpoint["total"]
                if len(self.saved) != row["completed"]:
                    raise WorkbenchError("workbench_checkpoint_cursor_invalid")
            conn.execute(
                "UPDATE strategy_workbench_searches SET state='running',input_hash=?,invocation_id=?,revision=revision+1,updated_at=? WHERE id=?",
                (binding, self.run_id, time.time(), self.id),
            )
            conn.execute(
                "INSERT INTO strategy_workbench_search_attempts VALUES(?,?,?,NULL,NULL,NULL,NULL)",
                (self.run_id, self.id, binding),
            )
            self.input_hash = binding

    def bind_search(self, identity):
        if self.id is None:
            return
        value = digest(identity)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute(
                "SELECT data_hash FROM strategy_workbench_searches WHERE id=?",
                (self.id,),
            ).fetchone()[0]
            if old and old != value:
                raise WorkbenchError("workbench_search_data_changed")
            conn.execute(
                "UPDATE strategy_workbench_searches SET data_hash=? WHERE id=?",
                (value, self.id),
            )
        self.data_hash = value

    def boundary(self, aggregates, total):
        if self.id is None:
            return
        if (
            not isinstance(total, int)
            or total < len(aggregates)
            or len(aggregates) < len(self.saved)
        ):
            raise WorkbenchError("workbench_checkpoint_cursor_invalid")
        if self.total and self.total != total:
            raise WorkbenchError("workbench_checkpoint_budget_changed")
        # Inspect control requests at every candidate boundary, but persist only
        # bounded batches and the final/interrupt boundary (avoid quadratic writes).
        with connect(self.db_path) as conn:
            pending = conn.execute(
                "SELECT state FROM strategy_workbench_searches WHERE id=?", (self.id,)
            ).fetchone()[0]
        if pending == "running" and len(aggregates) % 20 and len(aggregates) != total:
            return
        checkpoint = {
            "schema_version": CHECKPOINT_VERSION,
            "search_id": self.id,
            "task_id": self.task_id,
            "actor_id": self.actor_id,
            "tool_ref": self.tool_ref,
            "run_id": self.run_id,
            "data_hash": self.data_hash,
            "input_hash": self.input_hash,
            "total": total,
            "aggregates": aggregates,
        }
        raw = canonical(checkpoint)
        if len(raw.encode()) > 64 * 1024 * 1024:
            raise WorkbenchError("workbench_checkpoint_budget_exceeded")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT state,invocation_id FROM strategy_workbench_searches WHERE id=?",
                (self.id,),
            ).fetchone()
            if (
                row is None
                or row["invocation_id"] != self.run_id
                or row["state"]
                not in {"running", "pause_requested", "cancel_requested"}
            ):
                raise WorkbenchError("workbench_search_claim_changed")
            state = {"pause_requested": "paused", "cancel_requested": "cancelled"}.get(
                row["state"], "running"
            )
            mac = _sign(self.key, checkpoint)
            conn.execute(
                "UPDATE strategy_workbench_searches SET state=?,checkpoint_json=?,checkpoint_mac=?,completed=?,total=?,revision=revision+1,updated_at=? WHERE id=?",
                (state, raw, mac, len(aggregates), total, time.time(), self.id),
            )
            if state != "running":
                conn.execute(
                    "UPDATE strategy_workbench_search_attempts SET halted_state=?,checkpoint_mac=?,halted_at=?,checkpoint_json=? WHERE run_id=?",
                    (state, mac, time.time(), raw, self.run_id),
                )
        self.total = total
        reporter = getattr(self.ctx, "report_progress", None)
        if callable(reporter):
            try:
                reporter(
                    {
                        "kind": "strategy_candidate_search",
                        "state": state,
                        "completed_candidates": len(aggregates),
                        "total_candidates": total,
                        "percent": round(len(aggregates) * 100 / max(1, total), 2),
                    }
                )
            except Exception:
                pass  # Telemetry cannot turn a saved checkpoint into a failed result.
        if state != "running":
            raise SearchInterrupted(state)

    def publishing(self):
        if self.id is None:
            return
        # Final checkpoint handles a concurrent pause before admission to publication.
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT checkpoint_json FROM strategy_workbench_searches WHERE id=?",
                (self.id,),
            ).fetchone()
        checkpoint = json.loads(row["checkpoint_json"])
        self.boundary(checkpoint["aggregates"], checkpoint["total"])
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE strategy_workbench_searches SET state='publishing',revision=revision+1,updated_at=? WHERE id=? AND invocation_id=? AND state='running'",
                (time.time(), self.id, self.run_id),
            )
            if cursor.rowcount != 1:
                raise WorkbenchError("workbench_publication_raced_control")

    def completed(self, output):
        if self.id is None:
            return output
        refs = [
            {
                "artifact_id": item.get("artifact_id", item.get("id")),
                "kind": item.get("kind"),
                "content_hash": item.get("content_hash"),
            }
            for item in output.get("artifacts", [])
            if isinstance(item, dict)
        ]
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE strategy_workbench_searches SET state='published',result_json=?,revision=revision+1,updated_at=? WHERE id=? AND invocation_id=? AND state='publishing'",
                (canonical(refs), time.time(), self.id, self.run_id),
            )
            if cursor.rowcount != 1:
                raise WorkbenchError("workbench_publication_receipt_changed")
        return output


def certified_unpublished_run(conn, run_id):
    """Only native checkpoint exits prove no tool publication; JSON callers cannot mint one."""
    if not run_id:
        return False
    row = conn.execute(
        "SELECT a.input_hash,a.halted_state,a.checkpoint_mac,a.checkpoint_json,s.id,s.task_id,s.actor_id,s.data_hash,s.input_hash AS current_hash,s.tool_ref,r.tool_ref AS actual_tool,r.input_json,r.invocation_contract_json FROM strategy_workbench_search_attempts a JOIN strategy_workbench_searches s ON s.id=a.search_id JOIN plan_step_runs r ON r.id=a.run_id WHERE a.run_id=?",
        (run_id,),
    ).fetchone()
    if (
        not row
        or row["halted_state"] not in {"paused", "cancelled"}
        or not row["checkpoint_mac"]
        or row["tool_ref"] not in SEARCH_TOOLS
        or row["tool_ref"] != row["actual_tool"]
    ):
        return False
    try:
        actual = digest(
            {
                "inputs": json.loads(row["input_json"]),
                "invocation": json.loads(row["invocation_contract_json"]),
                "producer": producer_hash(),
            }
        )
    except (TypeError, ValueError):
        return False
    return actual == row["input_hash"] == row[
        "current_hash"
    ] and authenticated_checkpoint(
        conn, row["checkpoint_json"], row["checkpoint_mac"], row, run_id=run_id
    )


def authenticated_checkpoint(conn, raw, mac, row, *, run_id=None):
    try:
        database = next(
            item["file"]
            for item in conn.execute("PRAGMA database_list")
            if item["name"] == "main"
        )
        key_path = Path(database).parent / ".strategy_workbench/checkpoint-key"
        if key_path.is_symlink():
            return False
        key = key_path.read_bytes()
        payload = json.loads(raw)
        return (
            len(key) == 32
            and hmac.compare_digest(_sign(key, payload), mac or "")
            and payload["schema_version"] == CHECKPOINT_VERSION
            and payload["search_id"] == row["id"]
            and payload["task_id"] == row["task_id"]
            and payload["actor_id"] == row["actor_id"]
            and payload["tool_ref"] == row["tool_ref"]
            and payload["data_hash"] == row["data_hash"]
            and payload["input_hash"] == row["input_hash"]
            and (run_id is None or payload["run_id"] == run_id)
            and 0 <= len(payload["aggregates"]) <= payload["total"]
        )
    except (OSError, TypeError, ValueError, KeyError, StopIteration):
        return False


def control_search(
    settings, task_id, actor, search_id, action, request, *, conn=None, verified=None
):
    from marvis.strategy_workbench.service import authenticated_evidence, evidence_stamp
    from contextlib import nullcontext

    owns = conn is None
    if action == "resume":
        if owns:
            verified = authenticated_evidence(settings, task_id)
        if verified is None:
            raise WorkbenchError("workbench_verified_evidence_required")
        current, stamp = verified
    with connect(settings.db_path) if owns else nullcontext(conn) as conn:
        if owns:
            conn.execute("BEGIN IMMEDIATE")
        require_owner(conn, actor)
        row = conn.execute(
            "SELECT * FROM strategy_workbench_searches WHERE id=? AND task_id=? AND actor_id=?",
            (search_id, task_id, actor),
        ).fetchone()
        if row is None:
            raise WorkbenchError("workbench_search_not_found", 404)
        if request.expected_revision != row["revision"]:
            raise WorkbenchError("workbench_search_conflict")
        if request.evidence_hash != row["evidence_hash"]:
            raise WorkbenchError("workbench_evidence_changed")
        if action == "resume" and (
            evidence_stamp(conn, task_id) != stamp or row["evidence_hash"] != current
        ):
            raise WorkbenchError("workbench_evidence_changed")
        state = {
            "pause": "pause_requested",
            "cancel": "cancel_requested",
            "resume": "resuming",
        }[action]
        allowed = {
            "pause": {"running"},
            "cancel": {"running", "pause_requested"},
            "resume": {"paused"},
        }[action]
        if row["state"] not in allowed:
            raise WorkbenchError("workbench_search_control_unavailable")
        if action == "resume" and not certified_unpublished_run(
            conn, row["invocation_id"]
        ):
            raise WorkbenchError("workbench_resume_requires_native_checkpoint")
        if action == "resume" and not authenticated_checkpoint(
            conn,
            row["checkpoint_json"],
            row["checkpoint_mac"],
            row,
            run_id=row["invocation_id"],
        ):
            raise WorkbenchError("workbench_checkpoint_authentication_failed")
        conn.execute(
            "UPDATE strategy_workbench_searches SET state=?,revision=revision+1,updated_at=? WHERE id=?",
            (state, time.time(), search_id),
        )
        return public_search(
            conn.execute(
                "SELECT * FROM strategy_workbench_searches WHERE id=?", (search_id,)
            ).fetchone()
        )


def frame_fingerprint(frame):
    import pandas as pd

    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def finish_native_invocation(db_path, invocation_id, result):
    if not invocation_id:
        return
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id,state FROM strategy_workbench_searches WHERE invocation_id=?",
            (invocation_id,),
        ).fetchone()
        if not row or row["state"] in {
            "paused",
            "cancelled",
            "completed",
            "failed",
            "uncertain",
        }:
            return
        state = (
            "completed"
            if result.ok and row["state"] == "published"
            else "uncertain"
            if row["state"] in {"publishing", "published"}
            else "failed"
        )
        conn.execute(
            "UPDATE strategy_workbench_searches SET state=?,revision=revision+1,updated_at=? WHERE id=?",
            (state, time.time(), row["id"]),
        )

"""Opt-in, content-free measurements for an isolated runtime evaluation.

Inactive in ordinary application processes. These spans measure enclosing wall
time, including nested work; their durations must not be added across scopes.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import json
import os
import sqlite3
import threading
import time
import uuid


SCOPES = frozenset({
    "plugin", "adhoc", "validation_scan", "validation_notebook",
    "validation_pmml", "validation_metrics", "validation_report", "queue",
})
_observer = None
QUEUE_TERMINALS = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
REVISION_SCOPES = frozenset({"replan_attempt", "explore_attempt"})
REVISION_KINDS = frozenset({"created", "structural_replan", "explore_append", "upstream_revision"})
WAIT_SCOPES = frozenset({"report_confirmation_wait", "plan_confirmation_wait", "workflow_confirmation_wait"})
WAIT_ENDS = frozenset({"resumed", "superseded", "withdrawn", "censored"})


class _Journal:
    def __init__(self, path, origin_ns):
        self.origin_ns = origin_ns
        self.lock = threading.RLock()
        self.count = 0
        self.failed = False
        self.closed = False
        self.pending = {}
        self.pending_revisions = {}
        self.pending_states = {}
        self.waits = {}
        self.file = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8")
        self.emit("confirmation_plan_states_enabled")

    def emit(self, event, scope=None, identity=None, *, revision=None, revision_kind=None,
             task_sha256=None, binding_sha256=None):
        with self.lock:
            if self.closed:
                return
            if self.count >= 20000 and event != "closed":
                self.failed = True
                return
            row = {"event": event, "elapsed_ns": time.monotonic_ns() - self.origin_ns}
            if event == "closed":
                row["valid"] = not self.failed
                self.closed = True
            elif event == "plan_revision":
                row.update(identity=identity, revision=revision, kind=revision_kind)
            elif scope in WAIT_SCOPES:
                row.update(scope=scope, identity=identity, task_sha256=task_sha256, binding_sha256=binding_sha256)
            elif event not in {"queue_terminals_enabled", "plan_revisions_enabled", "confirmation_waits_enabled", "confirmation_plan_states_enabled"}:
                row.update(scope=scope, identity=identity)
            try:
                self.file.write(json.dumps(row, separators=(",", ":")) + "\n")
                self.file.flush()
                self.count += 1
            except (OSError, ValueError):
                # Measurement failure must not change a business outcome. A
                # missing or invalid closure makes the reader fail closed.
                self.failed = True

    def seal(self):
        with self.lock:
            if self.pending or self.pending_revisions or self.pending_states:
                self.failed = True
            for task_id in list(self.waits):
                self.close_wait(task_id, "censored")
        self.emit("closed")

    def close_wait(self, task_id, outcome):
        with self.lock:
            waiting = self.waits.pop(task_id, None)
            if waiting is not None:
                self.emit(outcome, **waiting)


@contextmanager
def observe_runtime(path, origin_ns):
    global _observer
    if _observer is not None:
        raise RuntimeError("runtime_observer_already_active")
    observer = _Journal(path, origin_ns)
    _observer = observer
    try:
        yield observer
    finally:
        _observer = None
        observer.seal()
        observer.file.close()


def measured(scope):
    if scope not in (SCOPES | REVISION_SCOPES) - {"queue"}:
        raise ValueError("invalid_measurement_scope")

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            observer = _observer
            if observer is None:
                return function(*args, **kwargs)
            identity = uuid.uuid4().hex
            observer.emit("started", scope, identity)
            try:
                result = function(*args, **kwargs)
            except BaseException:
                observer.emit("raised", scope, identity)
                raise
            observer.emit("returned", scope, identity)
            return result
        return wrapped
    return decorate


def observe_queue(job_id, *, running=False):
    observer = _observer
    if observer is not None:
        observer.emit("running" if running else "queued", "queue", sha256(job_id.encode()).hexdigest())


def observing_runtime():
    return _observer is not None


def queue_terminal_on_commit(conn, job_id, status):
    """Stage a known queued -> terminal transition, never emit before commit.

    The caller holds the write transaction and has established the prior queued
    state. The shared connection owner confirms commit and verifies final state;
    plain external connections cannot accidentally certify uncommitted events.
    """
    observer = _observer
    if observer is not None and status in QUEUE_TERMINALS:
        with observer.lock:
            if not observer.closed:
                observer.pending.setdefault(conn, {})[job_id] = status


def observe_transaction_commit(conn):
    observer = _observer
    if observer is None:
        return
    with observer.lock:
        pending = observer.pending.pop(conn, {})
        revisions = observer.pending_revisions.pop(conn, {})
        states = observer.pending_states.pop(conn, set())
    for job_id, expected in pending.items():
        try:
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            # A savepoint may have rolled the staged mutation back, or a later
            # statement may have replaced it. Neither certifies this transition.
            if row is not None and row[0] == expected:
                observer.emit(expected, "queue", sha256(job_id.encode()).hexdigest())
        except sqlite3.Error:
            with observer.lock:
                observer.failed = True
    for plan_id, entries in revisions.items():
        try:
            row = conn.execute("SELECT replan_count FROM plans WHERE id=?", (plan_id,)).fetchone()
            if row is not None and row[0] == entries[-1][1]:
                identity = sha256(plan_id.encode()).hexdigest()
                for kind, revision in entries:
                    observer.emit("plan_revision", identity=identity, revision=revision, revision_kind=kind)
        except sqlite3.Error:
            with observer.lock:
                observer.failed = True

    if states:
        try:
            from marvis.repositories.tasks import TaskRepository
            from marvis.runtime_waits import refresh_confirmation_wait

            for db_path, task_id in states:
                refresh_confirmation_wait(TaskRepository(db_path), task_id)
        except Exception:
            invalidate_runtime_observation()


def discard_transaction_observations(conn):
    observer = _observer
    if observer is not None:
        with observer.lock:
            observer.pending.pop(conn, None)
            observer.pending_revisions.pop(conn, None)
            observer.pending_states.pop(conn, None)


def plan_revision_on_commit(conn, plan_id, kind, revision):
    observer = _observer
    if observer is None:
        return
    with observer.lock:
        if kind not in REVISION_KINDS or type(revision) is not int or revision < 0:
            observer.failed = True
        elif not observer.closed:
            observer.pending_revisions.setdefault(conn, {}).setdefault(plan_id, []).append((kind, revision))


def confirmation_wait(task_id, scope, binding):
    observer = _observer
    if observer is None:
        return
    if scope not in WAIT_SCOPES:
        raise ValueError("invalid_confirmation_wait_scope")
    binding_hash = sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with observer.lock:
        if observer.closed:
            return
        previous = observer.waits.get(task_id)
        if previous and previous["scope"] == scope and previous["binding_sha256"] == binding_hash:
            return
        observer.close_wait(task_id, "superseded")
        waiting = {"scope": scope, "identity": uuid.uuid4().hex,
                   "task_sha256": sha256(task_id.encode()).hexdigest(), "binding_sha256": binding_hash}
        observer.waits[task_id] = waiting
        observer.emit("waiting", **waiting)


def end_confirmation_wait(task_id, outcome="resumed"):
    observer = _observer
    if observer is not None:
        if outcome not in WAIT_ENDS:
            raise ValueError("invalid_confirmation_wait_outcome")
        observer.close_wait(task_id, outcome)


def invalidate_runtime_observation():
    observer = _observer
    if observer is not None:
        with observer.lock:
            observer.failed = True


def plan_state_on_commit(conn, db_path, *, plan_id=None, step_id=None):
    """Refresh task readiness only after the owning transaction has committed.

    Staging also covers early returns in mutation methods. Rollbacks discard it;
    a savepoint rollback merely refreshes the unchanged committed state. Plans
    without an owning task (repository-only fixtures) cannot open a task wait.
    """
    observer = _observer
    if observer is None:
        return
    try:
        if step_id is not None:
            row = conn.execute(
                "SELECT p.task_id FROM plan_steps s JOIN plans p ON p.id=s.plan_id "
                "JOIN tasks t ON t.id=p.task_id WHERE s.id=?", (step_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT p.task_id FROM plans p JOIN tasks t ON t.id=p.task_id WHERE p.id=?", (plan_id,),
            ).fetchone()
        if row is not None:
            with observer.lock:
                if not observer.closed:
                    observer.pending_states.setdefault(conn, set()).add((db_path, row[0]))
    except Exception:
        invalidate_runtime_observation()

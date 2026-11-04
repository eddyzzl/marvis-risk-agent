"""Opt-in side-car timing spans for human waits, subprocesses and tools.

Inactive in ordinary application processes. Kept separate from
``runtime_observations`` so archived backend journals and their digests stay
unchanged. Rows carry only a phase, a fixed label, an identity and a clock
offset; never arguments, paths or content. Spans may nest and overlap; the
formal collector resolves overlaps, callers must not add durations.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import os
import threading
import time
import uuid

SCHEMA = "marvis.process-timing-journal.v1"
PHASES = frozenset({"human_wait", "subprocess", "tool"})
LABELS = frozenset({
    "plan_step", "worker_process", "notebook_process", "arrow_sample_process",
    "version_check", "operator_wait", "stub_tool", "stub_process",
})
MAX_ROWS = 20000
_observer = None


class _Journal:
    def __init__(self, path, origin_ns):
        self.origin_ns = origin_ns
        self.lock = threading.Lock()
        self.count = 0
        self.failed = False
        self.closed = False
        self.open_spans = set()
        self.file = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8")
        self._write({"event": "opened", "schema": SCHEMA})

    def _write(self, row):
        row["elapsed_ns"] = time.monotonic_ns() - self.origin_ns
        if self.count >= MAX_ROWS and row["event"] != "closed":
            self.failed = True
            return
        try:
            self.file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.file.flush()
            self.count += 1
        except (OSError, ValueError):
            # A measurement failure never changes the business outcome; the
            # invalid seal makes the collector fail closed instead.
            self.failed = True

    def emit(self, event, phase, label, identity):
        with self.lock:
            if self.closed:
                return
            if event == "start":
                self.open_spans.add(identity)
            else:
                self.open_spans.discard(identity)
            self._write({"event": event, "phase": phase, "label": label, "identity": identity})

    def seal(self):
        with self.lock:
            if self.closed:
                return
            valid = not self.failed and not self.open_spans
            self._write({"event": "closed", "valid": valid})
            self.closed = True


@contextmanager
def observe_process_timing(path, origin_ns):
    global _observer
    if _observer is not None:
        raise RuntimeError("process_timing_observer_already_active")
    observer = _Journal(path, origin_ns)
    _observer = observer
    try:
        yield observer
    finally:
        _observer = None
        observer.seal()
        observer.file.close()


def _check(phase, label):
    if phase not in PHASES or label not in LABELS:
        raise ValueError("invalid_process_timing_span")


@contextmanager
def timed_span(phase, label):
    _check(phase, label)
    observer = _observer
    if observer is None:
        yield
        return
    identity = uuid.uuid4().hex
    observer.emit("start", phase, label, identity)
    try:
        yield
    except BaseException:
        observer.emit("raised", phase, label, identity)
        raise
    observer.emit("end", phase, label, identity)


def timed(phase, label):
    _check(phase, label)

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _observer is None:
                return function(*args, **kwargs)
            with timed_span(phase, label):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def read_process_timing(path):
    """Return validated rows, or ``[{"event": "unavailable"}]`` on any defect."""
    try:
        if path.is_symlink() or path.stat().st_size > 8_000_000:
            raise ValueError("invalid_process_timing_journal")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        validate_process_timing(rows)
        return rows
    except (OSError, ValueError, TypeError, KeyError):
        return [{"event": "unavailable"}]


def validate_process_timing(rows):
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_ROWS + 1:
        raise ValueError("invalid_process_timing_journal")
    if rows[0].get("event") != "opened" or rows[0].get("schema") != SCHEMA:
        raise ValueError("invalid_process_timing_header")
    previous = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("invalid_process_timing_row")
        kind = row.get("event")
        if kind == "opened":
            valid = index == 0 and set(row) == {"event", "schema", "elapsed_ns"}
        elif kind == "closed":
            valid = index == len(rows) - 1 and set(row) == {"event", "valid", "elapsed_ns"} and type(row["valid"]) is bool
        else:
            identity = row.get("identity")
            valid = (kind in {"start", "end", "raised"}
                     and set(row) == {"event", "phase", "label", "identity", "elapsed_ns"}
                     and row["phase"] in PHASES and row["label"] in LABELS
                     and isinstance(identity, str) and len(identity) == 32
                     and all(c in "0123456789abcdef" for c in identity))
        stamp = row.get("elapsed_ns")
        if not valid or type(stamp) is not int or stamp < previous:
            raise ValueError("invalid_process_timing_row")
        previous = stamp

"""Application lifecycle owner for the existing bounded local scheduler."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import threading


class OperationsLoop:
    def __init__(self, runtime, *, interval_seconds=1.0):
        self.runtime = runtime
        self.interval_seconds = interval_seconds
        self._stop = runtime.stop_event
        self._lock = threading.Lock()
        self._thread = None
        self._last_report = None
        self._last_finished_at = None
        self._last_error = None

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="marvis-operations", daemon=True
            )
            self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    def status(self):
        with self._lock:
            return {
                "alive": self._thread is not None and self._thread.is_alive(),
                "stopping": self._stop.is_set(),
                "last_finished_at": self._last_finished_at,
                "last_error_code": self._last_error,
                "last_report": self._last_report,
            }

    def _run(self):
        while not self._stop.is_set():
            try:
                report = self.runtime.tick(catch_up_budget=5, notification_budget=10)
            except Exception:
                # No exception bodies: they may contain source paths or identifiers.
                with self._lock:
                    self._last_error = "operations_tick_failed"
            else:
                with self._lock:
                    self._last_report = asdict(report)
                    self._last_error = None
            finally:
                with self._lock:
                    self._last_finished_at = datetime.now(UTC).isoformat()
            self._stop.wait(self.interval_seconds)

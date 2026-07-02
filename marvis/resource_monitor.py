from __future__ import annotations

import threading
from typing import Any, Callable

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in stripped runtimes.
    psutil = None


class ProcessTreeResourceMonitor:
    """Samples the RSS of a process tree on a background thread and calls
    ``on_limit`` (then terminates the tree) once the configured memory
    ceiling is exceeded.

    This is the psutil-based soft-monitoring technique already proven for
    notebook kernels (see ``marvis.notebooks._NotebookResourceMonitor``),
    generalized so any process tree (a plugin/pack tool worker, a notebook
    kernel, ...) can reuse it by supplying a ``pid_getter`` callable.
    """

    def __init__(
        self,
        *,
        pid_getter: Callable[[], int | None],
        memory_limit_mb: int | None,
        interval_seconds: float = 0.5,
        on_limit: Callable[[], None] | None = None,
    ) -> None:
        self._pid_getter = pid_getter
        self._memory_limit_mb = memory_limit_mb
        self._memory_limit_bytes = (
            int(memory_limit_mb) * 1024 * 1024 if memory_limit_mb is not None else None
        )
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._on_limit = on_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pid: int | None = None
        self._peak_rss_bytes: int | None = None
        self._memory_limit_exceeded = False
        self._monitor_started = False
        self._monitor_degraded = False
        self._monitor_error: str | None = None

    @property
    def memory_limit_exceeded(self) -> bool:
        with self._lock:
            return self._memory_limit_exceeded

    def __enter__(self) -> "ProcessTreeResourceMonitor":
        if self._memory_limit_bytes is None:
            return self
        if psutil is None:
            self._set_degraded("psutil unavailable")
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="marvis-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "memory_limit_mb": self._memory_limit_mb,
                "peak_rss_mb": _bytes_to_mb(self._peak_rss_bytes),
                "pid": self._pid,
                "memory_limit_exceeded": self._memory_limit_exceeded,
                "monitor_started": self._monitor_started,
                "monitor_degraded": self._monitor_degraded,
                "monitor_error": self._monitor_error,
            }

    def _run(self) -> None:
        process = None
        while not self._stop.is_set():
            if process is None:
                pid = self._pid_getter()
                if pid is None:
                    self._stop.wait(self._interval_seconds)
                    continue
                try:
                    process = psutil.Process(pid)
                except Exception as exc:
                    self._set_degraded(f"process unavailable: {exc}")
                    return
                with self._lock:
                    self._pid = int(pid)
                    self._monitor_started = True
            try:
                rss = process_tree_rss(process)
            except Exception as exc:
                self._set_degraded(f"rss unavailable: {exc}")
                return
            exceeded = False
            with self._lock:
                self._peak_rss_bytes = max(self._peak_rss_bytes or 0, rss)
                exceeded = (
                    self._memory_limit_bytes is not None
                    and rss > self._memory_limit_bytes
                )
                if exceeded:
                    self._memory_limit_exceeded = True
            if exceeded:
                if self._on_limit is not None:
                    self._on_limit()
                terminate_process_tree(process)
                return
            self._stop.wait(self._interval_seconds)

    def _set_degraded(self, message: str) -> None:
        with self._lock:
            self._monitor_degraded = True
            self._monitor_error = message


def process_tree_rss(process: Any) -> int:
    rss = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            rss += int(child.memory_info().rss)
        except Exception:
            continue
    return rss


def terminate_process_tree(process: Any) -> None:
    try:
        children = list(process.children(recursive=True))
    except Exception:
        children = []
    targets = children + [process]
    for target in targets:
        try:
            target.terminate()
        except Exception:
            pass
    alive: list[Any] = []
    wait_procs = getattr(psutil, "wait_procs", None) if psutil is not None else None
    if callable(wait_procs):
        try:
            _, alive = wait_procs(targets, timeout=2.0)
        except Exception:
            alive = targets
    else:
        for target in targets:
            try:
                target.wait(timeout=2.0)
            except Exception:
                alive.append(target)
    for target in alive:
        try:
            target.kill()
        except Exception:
            pass


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)

import asyncio
from dataclasses import asdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Any

from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import nbformat

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in stripped runtimes.
    psutil = None

from marvis.notebook_cancellation import (
    NotebookCancellationToken,
    NotebookCancelled,
)
from marvis.notebook_contract import (
    build_contract_head_cell_source,
    build_contract_tail_cell_source,
    precheck_notebook_contract,
)
from marvis.notebook_steps import NotebookStepPlan, notebook_step_plan


@dataclass(frozen=True)
class NotebookRunResult:
    succeeded: bool
    failed_cell_index: int | None
    error_name: str | None
    error_value: str | None
    step_events: dict[str, Any] | None = None
    cancelled: bool = False
    resource_usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class AppendedCellExecutionPolicy:
    scope: str
    reason: str
    allowed_marvis_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scope = self.scope.strip()
        reason = self.reason.strip()
        allowed_marvis_kinds = tuple(
            kind.strip() for kind in self.allowed_marvis_kinds if kind.strip()
        )
        if not scope:
            raise ValueError("appended-cell execution policy scope is required")
        if not reason:
            raise ValueError("appended-cell execution policy reason is required")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "allowed_marvis_kinds", allowed_marvis_kinds)

    def validate_cell_metadata(self, metadata: dict[str, Any] | None) -> None:
        if not self.allowed_marvis_kinds:
            return
        marvis_kind = None
        if metadata is not None:
            raw_kind = metadata.get("marvis")
            if isinstance(raw_kind, str):
                marvis_kind = raw_kind.strip()
        if marvis_kind not in self.allowed_marvis_kinds:
            allowed = ", ".join(self.allowed_marvis_kinds)
            raise RuntimeError(
                "live notebook appended-cell execution policy "
                f"{self.scope!r} only allows marvis cell kinds: {allowed}"
            )


class NotebookResourceLimitExceeded(RuntimeError):
    """Raised when the notebook kernel exceeds the configured RSS budget."""


class NotebookSubprocessTimeout(TimeoutError):
    """Raised when the isolated notebook worker exceeds the wall-clock timeout."""


_LIVE_SESSIONS_LOCK = threading.Lock()
_LIVE_SESSIONS: dict[str, "NotebookExecutionSession"] = {}


class NotebookExecutionSession:
    def __init__(
        self,
        *,
        notebook_path: Path,
        executed_path: Path,
        log_path: Path,
        timeout: int = 3600,
        kernel_name: str = "python3",
        progress_path: Path | None = None,
        execution_cwd: Path | None = None,
        cancellation_token: NotebookCancellationToken | None = None,
        memory_limit_mb: int | None = None,
        resource_poll_interval_seconds: float = 0.5,
        allow_appended_execution: bool = False,
        appended_execution_policy: AppendedCellExecutionPolicy | None = None,
    ) -> None:
        if allow_appended_execution and appended_execution_policy is None:
            raise ValueError(
                "live notebook appended-cell execution requires an explicit policy"
            )
        self.notebook_path = notebook_path
        self.executed_path = executed_path
        self.log_path = log_path
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.progress_path = progress_path
        self.execution_cwd = execution_cwd
        self.cancellation_token = cancellation_token
        self.memory_limit_mb = _normalize_memory_limit_mb(memory_limit_mb)
        self.resource_poll_interval_seconds = max(0.05, float(resource_poll_interval_seconds))
        self.notebook = nbformat.read(notebook_path, as_version=4)
        self.original_cell_count = len(self.notebook.cells)
        self.allow_appended_execution = bool(
            allow_appended_execution or appended_execution_policy is not None
        )
        self.appended_execution_policy = appended_execution_policy
        self.plan = notebook_step_plan(self.notebook)
        self.cell_events: dict[int, dict[str, Any]] = {}
        self.closed = False
        self._progress_suspended = False
        self._defer_all_cell_completions = False
        self._deferred_completion_cell_indexes: set[int] = set()
        self._pending_completion_cell_indexes: set[int] = set()
        self.client = self._build_client()
        if cancellation_token is not None:
            cancellation_token.bind_client(self.client)

    def execute_notebook(self, *, keep_alive: bool = True) -> NotebookRunResult:
        self._ensure_open()
        self.executed_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_progress()

        def execute() -> None:
            self._raise_if_cancelled()
            kwargs: dict[str, Any] = {
                "cwd": str(self.execution_cwd or self.notebook_path.parent),
            }
            if keep_alive:
                kwargs["cleanup_kc"] = False
            self.client.execute(**kwargs)

        previous_defer_all = self._defer_all_cell_completions
        self._defer_all_cell_completions = True
        try:
            return self._run_with_result(execute, log_path=self.log_path)
        finally:
            self._defer_all_cell_completions = previous_defer_all

    def execute_code_cell(
        self,
        source: str,
        *,
        log_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
        record_progress: bool = False,
    ) -> NotebookRunResult:
        self._ensure_appended_execution_allowed(metadata=metadata)
        cell_index = self.append_code_cell(
            source,
            metadata=metadata,
            record_progress=record_progress,
        )
        return self.execute_existing_code_cell(
            cell_index,
            log_path=log_path,
            record_progress=record_progress,
        )

    def append_code_cell(
        self,
        source: str,
        *,
        metadata: dict[str, Any] | None = None,
        record_progress: bool = False,
    ) -> int:
        self._ensure_open()
        self._ensure_appended_execution_allowed(metadata=metadata)
        cell = nbformat.v4.new_code_cell(source)
        if metadata:
            cell.metadata.update(metadata)
        cell_index = len(self.notebook.cells)
        self.notebook.cells.append(cell)
        self.plan = notebook_step_plan(self.notebook)
        if record_progress:
            self._write_progress()
        return cell_index

    def execute_existing_code_cell(
        self,
        cell_index: int,
        *,
        log_path: Path | None = None,
        record_progress: bool = False,
    ) -> NotebookRunResult:
        self._ensure_open()
        try:
            cell = self.notebook.cells[cell_index]
        except IndexError as exc:
            raise ValueError(f"notebook cell index out of range: {cell_index}") from exc
        if cell.cell_type != "code":
            raise ValueError(f"notebook cell is not code: {cell_index}")
        if cell_index >= self.original_cell_count:
            self._ensure_appended_execution_allowed(metadata=dict(cell.metadata or {}))
        target_log_path = log_path or self.log_path
        target_log_path.parent.mkdir(parents=True, exist_ok=True)

        def execute() -> None:
            self._raise_if_cancelled()
            self.client.execute_cell(
                cell,
                cell_index,
                execution_count=getattr(self.client, "code_cells_executed", 0) + 1,
            )

        previous_progress_suspended = self._progress_suspended
        self._progress_suspended = self._progress_suspended or not record_progress
        self._deferred_completion_cell_indexes.add(cell_index)
        try:
            result = self._run_with_result(execute, log_path=target_log_path)
            if not result.succeeded:
                _finalize_cell_event_after_execute(
                    self.cell_events,
                    cell=cell,
                    cell_index=cell_index,
                    succeeded=False,
                )
                self._pending_completion_cell_indexes.discard(cell_index)
                self._write_progress()
            return replace(result, step_events=self._step_events())
        finally:
            self._deferred_completion_cell_indexes.discard(cell_index)
            self._progress_suspended = previous_progress_suspended

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        cleanup = getattr(self.client, "_cleanup_kernel", None)
        if callable(cleanup):
            try:
                _consume_awaitable(cleanup())
                return
            except Exception:
                pass
        kernel_manager = getattr(self.client, "km", None)
        if kernel_manager is not None:
            _call_kernel_method(kernel_manager, "shutdown_kernel", now=True)

    def _build_client(self):
        callbacks = {
            "on_cell_start": self._record_cell_start,
            "on_cell_error": self._record_cell_error,
            "on_cell_executed": self._record_cell_executed,
            "on_cell_complete": self._record_cell_complete,
        }
        try:
            return NotebookClient(
                self.notebook,
                timeout=self.timeout,
                kernel_name=self.kernel_name,
                **callbacks,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            client = NotebookClient(
                self.notebook,
                timeout=self.timeout,
                kernel_name=self.kernel_name,
            )
            for name, callback in callbacks.items():
                setattr(client, name, callback)
            return client

    def _run_with_result(self, execute, *, log_path: Path) -> NotebookRunResult:
        monitor = _NotebookResourceMonitor(
            client_getter=lambda: self.client,
            memory_limit_mb=self.memory_limit_mb,
            interval_seconds=self.resource_poll_interval_seconds,
            on_limit=self._stop_kernel_for_resource_limit,
        )
        try:
            with monitor:
                execute()
        except CellExecutionError:
            nbformat.write(self.notebook, self.executed_path)
            failed_cell_index = _failed_cell_index(self.notebook)
            error_name, error_value = _error_output(self.notebook)
            if monitor.memory_limit_exceeded:
                return self._resource_limit_result(
                    log_path=log_path,
                    failed_cell_index=failed_cell_index,
                    resource_usage=monitor.snapshot(),
                )
            if self.cancellation_token is not None and self.cancellation_token.is_cancelled():
                _write_cancel_log(log_path)
                return NotebookRunResult(
                    succeeded=False,
                    failed_cell_index=failed_cell_index,
                    error_name="NotebookCancelled",
                    error_value="notebook execution cancelled",
                    step_events=self._step_events(),
                    cancelled=True,
                    resource_usage=monitor.snapshot(),
                )
            _write_failure_log(log_path, failed_cell_index, error_name, error_value)
            return NotebookRunResult(
                succeeded=False,
                failed_cell_index=failed_cell_index,
                error_name=error_name,
                error_value=error_value,
                step_events=self._step_events(),
                resource_usage=monitor.snapshot(),
            )
        except NotebookCancelled as error:
            nbformat.write(self.notebook, self.executed_path)
            _write_cancel_log(log_path)
            return NotebookRunResult(
                succeeded=False,
                failed_cell_index=_failed_cell_index(self.notebook),
                error_name=error.__class__.__name__,
                error_value=str(error),
                step_events=self._step_events(),
                cancelled=True,
                resource_usage=monitor.snapshot(),
            )
        except Exception as error:
            nbformat.write(self.notebook, self.executed_path)
            failed_cell_index = _failed_cell_index(self.notebook)
            if monitor.memory_limit_exceeded:
                return self._resource_limit_result(
                    log_path=log_path,
                    failed_cell_index=failed_cell_index,
                    resource_usage=monitor.snapshot(),
                )
            error_name = error.__class__.__name__
            error_value = str(error)
            if self.cancellation_token is not None and self.cancellation_token.is_cancelled():
                _write_cancel_log(log_path)
                return NotebookRunResult(
                    succeeded=False,
                    failed_cell_index=failed_cell_index,
                    error_name="NotebookCancelled",
                    error_value="notebook execution cancelled",
                    step_events=self._step_events(),
                    cancelled=True,
                    resource_usage=monitor.snapshot(),
                )
            _write_failure_log(log_path, failed_cell_index, error_name, error_value)
            return NotebookRunResult(
                succeeded=False,
                failed_cell_index=failed_cell_index,
                error_name=error_name,
                error_value=error_value,
                step_events=self._step_events(),
                resource_usage=monitor.snapshot(),
            )

        if monitor.memory_limit_exceeded:
            nbformat.write(self.notebook, self.executed_path)
            return self._resource_limit_result(
                log_path=log_path,
                failed_cell_index=_failed_cell_index(self.notebook),
                resource_usage=monitor.snapshot(),
            )
        self._finalize_pending_cell_completions()
        nbformat.write(self.notebook, self.executed_path)
        log_path.write_text("succeeded\n", encoding="utf-8")
        self._write_progress()
        return NotebookRunResult(
            succeeded=True,
            failed_cell_index=None,
            error_name=None,
            error_value=None,
            step_events=self._step_events(),
            resource_usage=monitor.snapshot(),
        )

    def _resource_limit_result(
        self,
        *,
        log_path: Path,
        failed_cell_index: int | None,
        resource_usage: dict[str, Any],
    ) -> NotebookRunResult:
        error_value = _resource_limit_message(resource_usage)
        _write_resource_limit_log(log_path, failed_cell_index, error_value, resource_usage)
        return NotebookRunResult(
            succeeded=False,
            failed_cell_index=failed_cell_index,
            error_name=NotebookResourceLimitExceeded.__name__,
            error_value=error_value,
            step_events=self._step_events(),
            resource_usage=resource_usage,
        )

    def _stop_kernel_for_resource_limit(self) -> None:
        kernel_manager = getattr(self.client, "km", None)
        if kernel_manager is None:
            return
        _call_kernel_method(kernel_manager, "interrupt_kernel")
        _call_kernel_method(kernel_manager, "shutdown_kernel", now=True)

    def _record_cell_start(self, **kwargs) -> None:
        self._raise_if_cancelled()
        cell_index = kwargs.get("cell_index")
        if isinstance(cell_index, int):
            self._finalize_pending_cell_completions(exclude_cell_index=cell_index)
        _record_cell_start(self.cell_events, **kwargs)
        self._write_progress()

    def _record_cell_error(self, **kwargs) -> None:
        _record_cell_error(self.cell_events, **kwargs)
        self._write_progress()

    def _record_cell_executed(self, **kwargs) -> None:
        cell_index = kwargs.get("cell_index")
        defer_completion = self._should_defer_cell_completion(cell_index)
        _record_cell_executed(
            self.cell_events,
            **kwargs,
            defer_completion=defer_completion,
        )
        if defer_completion and isinstance(cell_index, int):
            self._pending_completion_cell_indexes.add(cell_index)
        self._write_progress()

    def _record_cell_complete(self, **kwargs) -> None:
        cell_index = kwargs.get("cell_index")
        defer_completion = self._should_defer_cell_completion(cell_index)
        _record_cell_complete(
            self.cell_events,
            **kwargs,
            defer_completion=defer_completion,
        )
        if defer_completion and isinstance(cell_index, int):
            self._pending_completion_cell_indexes.add(cell_index)
        self._write_progress()

    def _should_defer_cell_completion(self, cell_index: Any) -> bool:
        return (
            self._defer_all_cell_completions
            or (
                isinstance(cell_index, int)
                and cell_index in self._deferred_completion_cell_indexes
            )
        )

    def _finalize_pending_cell_completions(
        self,
        *,
        exclude_cell_index: int | None = None,
    ) -> None:
        for cell_index in sorted(self._pending_completion_cell_indexes):
            if cell_index == exclude_cell_index:
                continue
            event = self.cell_events.get(cell_index)
            if event is None or event.get("status") == "failed":
                self._pending_completion_cell_indexes.discard(cell_index)
                continue
            _finalize_cell_event_after_execute(
                self.cell_events,
                cell=self.notebook.cells[cell_index],
                cell_index=cell_index,
                succeeded=True,
            )
            self._pending_completion_cell_indexes.discard(cell_index)

    def _write_progress(self) -> None:
        if self._progress_suspended:
            return
        if self.progress_path is not None:
            _write_step_events(self.progress_path, self.plan, self.cell_events)

    def _step_events(self) -> dict[str, Any]:
        return _build_step_events(self.plan, self.cell_events)

    def _raise_if_cancelled(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("notebook execution session is closed")

    def _ensure_appended_execution_allowed(
        self,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.allow_appended_execution or self.appended_execution_policy is None:
            raise RuntimeError(
                "live notebook appended-cell execution is disabled for this session; "
                "use isolated notebook execution or create the session with explicit appended-cell permission"
            )
        self.appended_execution_policy.validate_cell_metadata(metadata)


class _NotebookResourceMonitor:
    def __init__(
        self,
        *,
        client_getter,
        memory_limit_mb: int | None,
        interval_seconds: float,
        on_limit,
    ) -> None:
        self._client_getter = client_getter
        self._memory_limit_mb = memory_limit_mb
        self._memory_limit_bytes = (
            int(memory_limit_mb) * 1024 * 1024 if memory_limit_mb is not None else None
        )
        self._interval_seconds = interval_seconds
        self._on_limit = on_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._kernel_pid: int | None = None
        self._peak_rss_bytes: int | None = None
        self._memory_limit_exceeded = False
        self._monitor_started = False
        self._monitor_degraded = False
        self._monitor_error: str | None = None

    @property
    def memory_limit_exceeded(self) -> bool:
        with self._lock:
            return self._memory_limit_exceeded

    def __enter__(self):
        if self._memory_limit_bytes is None:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="marvis-notebook-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            if (
                self._memory_limit_bytes is not None
                and not self._monitor_started
                and not self._memory_limit_exceeded
                and self._monitor_error is None
            ):
                self._monitor_degraded = True
                self._monitor_error = "kernel pid unavailable"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "memory_limit_mb": self._memory_limit_mb,
                "peak_rss_mb": _bytes_to_mb(self._peak_rss_bytes),
                "kernel_pid": self._kernel_pid,
                "memory_limit_exceeded": self._memory_limit_exceeded,
                "monitor_started": self._monitor_started,
                "monitor_degraded": self._monitor_degraded,
                "monitor_error": self._monitor_error,
            }

    def _run(self) -> None:
        if psutil is None:
            self._set_degraded("psutil unavailable")
            return
        process = None
        while not self._stop.is_set():
            if process is None:
                pid = _kernel_pid(self._client_getter())
                if pid is None:
                    self._stop.wait(self._interval_seconds)
                    continue
                try:
                    process = psutil.Process(pid)
                except Exception as exc:
                    self._set_degraded(f"kernel process unavailable: {exc}")
                    return
                with self._lock:
                    self._kernel_pid = int(pid)
                    self._monitor_started = True
            try:
                rss = _process_tree_rss(process)
            except Exception as exc:
                self._set_degraded(f"kernel rss unavailable: {exc}")
                return
            with self._lock:
                self._peak_rss_bytes = max(self._peak_rss_bytes or 0, rss)
                exceeded = (
                    self._memory_limit_bytes is not None
                    and rss > self._memory_limit_bytes
                )
                if exceeded:
                    self._memory_limit_exceeded = True
            if exceeded:
                self._on_limit()
                _terminate_process_tree(process)
                return
            self._stop.wait(self._interval_seconds)

    def _set_degraded(self, message: str) -> None:
        with self._lock:
            self._monitor_degraded = True
            self._monitor_error = message


def register_live_notebook_session(
    task_id: str,
    session: NotebookExecutionSession,
) -> None:
    with _LIVE_SESSIONS_LOCK:
        previous = _LIVE_SESSIONS.get(task_id)
        _LIVE_SESSIONS[task_id] = session
    if previous is not None and previous is not session:
        previous.close()


def get_live_notebook_session(task_id: str) -> NotebookExecutionSession | None:
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.get(task_id)
    if session is not None and session.closed:
        close_live_notebook_session(task_id)
        return None
    return session


def close_live_notebook_session(task_id: str) -> None:
    with _LIVE_SESSIONS_LOCK:
        session = _LIVE_SESSIONS.pop(task_id, None)
    if session is not None:
        session.close()


def prepare_execution_notebook_v3(
    *,
    source_notebook: Path,
    output_notebook: Path,
    sample_path: Path,
    contract_meta_path: Path,
    code_scores_path: Path,
    feature_importance_path: Path,
    model_params_path: Path,
    extra_code_cells: list[tuple[str, str]] | None = None,
) -> Path:
    if source_notebook.resolve() == output_notebook.resolve():
        raise ValueError("source and output notebook paths must differ")

    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    contract_meta_path.parent.mkdir(parents=True, exist_ok=True)
    code_scores_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    model_params_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbformat.read(source_notebook, as_version=4)
    precheck_notebook_contract(notebook)

    head = nbformat.v4.new_code_cell(
        build_contract_head_cell_source(
            sample_path=sample_path,
            contract_meta_path=contract_meta_path,
            code_scores_path=code_scores_path,
            feature_importance_path=feature_importance_path,
            model_params_path=model_params_path,
            package_root=Path(__file__).resolve().parents[1],
        )
    )
    head.metadata["marvis"] = "head"
    tail = nbformat.v4.new_code_cell(build_contract_tail_cell_source())
    tail.metadata["marvis"] = "tail"
    notebook.cells.insert(0, head)
    notebook.cells.append(tail)
    for kind, source in extra_code_cells or []:
        cell = nbformat.v4.new_code_cell(source)
        cell.metadata["marvis"] = kind
        notebook.cells.append(cell)
    nbformat.write(notebook, output_notebook)
    return output_notebook


def _normalize_memory_limit_mb(memory_limit_mb: int | None) -> int | None:
    if memory_limit_mb is None:
        return None
    value = int(memory_limit_mb)
    if value <= 0:
        return None
    return value


def _kernel_pid(client: Any) -> int | None:
    kernel_manager = getattr(client, "km", None)
    if kernel_manager is None:
        return None
    provisioner = getattr(kernel_manager, "provisioner", None)
    for owner in (provisioner, kernel_manager):
        if owner is None:
            continue
        pid = _as_pid(getattr(owner, "pid", None))
        if pid is not None:
            return pid
        process = getattr(owner, "process", None)
        pid = _as_pid(getattr(process, "pid", None))
        if pid is not None:
            return pid
    return None


def _as_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _process_tree_rss(process) -> int:
    rss = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            rss += int(child.memory_info().rss)
        except Exception:
            continue
    return rss


def _terminate_process_tree(process) -> None:
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
    wait_procs = getattr(psutil, "wait_procs", None) if psutil is not None else None
    alive = []
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


def _call_kernel_method(kernel_manager: Any, method_name: str, **kwargs) -> None:
    method = getattr(kernel_manager, method_name, None)
    if not callable(method):
        return
    try:
        result = method(**kwargs)
    except TypeError:
        if not kwargs:
            return
        try:
            result = method()
        except Exception:
            return
    except Exception:
        return
    try:
        _consume_awaitable(result)
    except Exception:
        pass


def _consume_awaitable(value: Any) -> None:
    if not inspect.isawaitable(value):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(value)
        return
    errors: list[BaseException] = []

    def run_in_thread() -> None:
        try:
            asyncio.run(value)
        except BaseException as exc:  # pragma: no cover - re-raised below.
            errors.append(exc)

    thread = threading.Thread(
        target=run_in_thread,
        name="marvis-notebook-async-cleanup",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=5.0)
    if errors:
        raise errors[0]


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)


def _resource_limit_message(resource_usage: dict[str, Any]) -> str:
    limit = resource_usage.get("memory_limit_mb")
    peak = resource_usage.get("peak_rss_mb")
    if peak is None:
        return f"notebook kernel RSS exceeded memory limit {limit} MB"
    return f"notebook kernel RSS {peak} MB exceeded memory limit {limit} MB"


def _failed_cell_index(notebook: nbformat.NotebookNode) -> int | None:
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return cell_index
    return None


def _error_output(notebook: nbformat.NotebookNode) -> tuple[str | None, str | None]:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return output.get("ename"), output.get("evalue")
    return None, None


def _write_failure_log(
    log_path: Path,
    failed_cell_index: int | None,
    error_name: str | None,
    error_value: str | None,
) -> None:
    log_path.write_text(
        "\n".join(
            [
                "failed",
                f"failed_cell_index={failed_cell_index}",
                f"error_name={error_name}",
                f"error_value={error_value}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_resource_limit_log(
    log_path: Path,
    failed_cell_index: int | None,
    error_value: str,
    resource_usage: dict[str, Any],
) -> None:
    log_path.write_text(
        "\n".join(
            [
                "failed",
                f"failed_cell_index={failed_cell_index}",
                f"error_name={NotebookResourceLimitExceeded.__name__}",
                f"error_value={error_value}",
                f"memory_limit_mb={resource_usage.get('memory_limit_mb')}",
                f"peak_rss_mb={resource_usage.get('peak_rss_mb')}",
                f"kernel_pid={resource_usage.get('kernel_pid')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_cancel_log(log_path: Path) -> None:
    log_path.write_text("cancelled\n", encoding="utf-8")


def run_notebook(
    notebook_path: Path,
    executed_path: Path,
    log_path: Path,
    timeout: int = 3600,
    kernel_name: str = "python3",
    progress_path: Path | None = None,
    execution_cwd: Path | None = None,
    cancellation_token: NotebookCancellationToken | None = None,
    memory_limit_mb: int | None = None,
    resource_poll_interval_seconds: float = 0.5,
    isolated: bool = False,
) -> NotebookRunResult:
    if isolated and cancellation_token is None:
        return _run_notebook_in_subprocess(
            notebook_path=notebook_path,
            executed_path=executed_path,
            log_path=log_path,
            timeout=timeout,
            kernel_name=kernel_name,
            progress_path=progress_path,
            execution_cwd=execution_cwd,
            memory_limit_mb=memory_limit_mb,
            resource_poll_interval_seconds=resource_poll_interval_seconds,
        )
    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=timeout,
        kernel_name=kernel_name,
        progress_path=progress_path,
        execution_cwd=execution_cwd,
        cancellation_token=cancellation_token,
        memory_limit_mb=memory_limit_mb,
        resource_poll_interval_seconds=resource_poll_interval_seconds,
    )
    try:
        return session.execute_notebook(keep_alive=False)
    finally:
        session.close()


def _run_notebook_in_subprocess(
    *,
    notebook_path: Path,
    executed_path: Path,
    log_path: Path,
    timeout: int,
    kernel_name: str,
    progress_path: Path | None,
    execution_cwd: Path | None,
    memory_limit_mb: int | None,
    resource_poll_interval_seconds: float,
) -> NotebookRunResult:
    job = {
        "notebook_path": str(Path(notebook_path)),
        "executed_path": str(Path(executed_path)),
        "log_path": str(Path(log_path)),
        "timeout": int(timeout),
        "kernel_name": kernel_name,
        "progress_path": None if progress_path is None else str(Path(progress_path)),
        "execution_cwd": None if execution_cwd is None else str(Path(execution_cwd)),
        "memory_limit_mb": memory_limit_mb,
        "resource_poll_interval_seconds": float(resource_poll_interval_seconds),
    }
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.Popen(
        [sys.executable, "-m", "marvis.notebook_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_notebook_worker_env(),
        cwd=str(execution_cwd or Path(notebook_path).parent),
        start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = process.communicate(json.dumps(job, ensure_ascii=False), timeout=int(timeout) + 5)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        stdout, stderr = process.communicate()
        _write_subprocess_timeout_artifacts(
            notebook_path=Path(notebook_path),
            executed_path=Path(executed_path),
            log_path=Path(log_path),
            timeout=int(timeout),
        )
        return NotebookRunResult(
            succeeded=False,
            failed_cell_index=None,
            error_name=NotebookSubprocessTimeout.__name__,
            error_value=f"isolated notebook worker timed out after {timeout}s",
            resource_usage={
                "subprocess_isolated": True,
                "worker_pid": process.pid,
                "worker_started_at": started,
                "worker_returncode": process.returncode,
                "stdout_tail": _tail_text(stdout),
                "stderr_tail": _tail_text(stderr),
            },
        )
    protocol = _parse_notebook_worker_result(stdout)
    if protocol is None:
        _write_failure_log(
            Path(log_path),
            None,
            "NotebookWorkerProtocolError",
            f"worker returned invalid protocol with exit code {process.returncode}",
        )
        return NotebookRunResult(
            succeeded=False,
            failed_cell_index=None,
            error_name="NotebookWorkerProtocolError",
            error_value=f"worker returned invalid protocol with exit code {process.returncode}",
            resource_usage={
                "subprocess_isolated": True,
                "worker_pid": process.pid,
                "worker_started_at": started,
                "worker_returncode": process.returncode,
                "stdout_tail": _tail_text(stdout),
                "stderr_tail": _tail_text(stderr),
            },
        )
    if not protocol.get("ok"):
        error_value = str(protocol.get("error") or "isolated notebook worker failed")
        _write_failure_log(Path(log_path), None, "NotebookWorkerError", error_value)
        return NotebookRunResult(
            succeeded=False,
            failed_cell_index=None,
            error_name="NotebookWorkerError",
            error_value=error_value,
            resource_usage={
                "subprocess_isolated": True,
                "worker_pid": process.pid,
                "worker_started_at": started,
                "worker_returncode": process.returncode,
                "stdout_tail": _tail_text(stdout),
                "stderr_tail": _tail_text(stderr),
            },
        )
    result_payload = protocol.get("result")
    if not isinstance(result_payload, dict):
        _write_failure_log(Path(log_path), None, "NotebookWorkerProtocolError", "worker result missing")
        return NotebookRunResult(
            succeeded=False,
            failed_cell_index=None,
            error_name="NotebookWorkerProtocolError",
            error_value="worker result missing",
            resource_usage={
                "subprocess_isolated": True,
                "worker_pid": process.pid,
                "worker_started_at": started,
                "worker_returncode": process.returncode,
            },
        )
    resource_usage = dict(result_payload.get("resource_usage") or {})
    resource_usage.update({
        "subprocess_isolated": True,
        "worker_pid": process.pid,
        "worker_started_at": started,
        "worker_returncode": process.returncode,
    })
    return NotebookRunResult(
        succeeded=bool(result_payload.get("succeeded")),
        failed_cell_index=result_payload.get("failed_cell_index"),
        error_name=result_payload.get("error_name"),
        error_value=result_payload.get("error_value"),
        step_events=result_payload.get("step_events"),
        cancelled=bool(result_payload.get("cancelled")),
        resource_usage=resource_usage,
    )


def notebook_run_result_to_dict(result: NotebookRunResult) -> dict:
    return asdict(result)


def notebook_run_result_from_dict(payload: dict) -> NotebookRunResult:
    return NotebookRunResult(
        succeeded=bool(payload.get("succeeded")),
        failed_cell_index=payload.get("failed_cell_index"),
        error_name=payload.get("error_name"),
        error_value=payload.get("error_value"),
        step_events=payload.get("step_events"),
        cancelled=bool(payload.get("cancelled")),
        resource_usage=payload.get("resource_usage") if isinstance(payload.get("resource_usage"), dict) else None,
    )


def _parse_notebook_worker_result(stdout: str) -> dict[str, Any] | None:
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_subprocess_timeout_artifacts(
    *,
    notebook_path: Path,
    executed_path: Path,
    log_path: Path,
    timeout: int,
) -> None:
    executed_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not executed_path.exists():
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.write(notebook, executed_path)
    _write_failure_log(
        log_path,
        None,
        NotebookSubprocessTimeout.__name__,
        f"isolated notebook worker timed out after {timeout}s",
    )


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _notebook_worker_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {
            "CONDA_PREFIX",
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PATH",
            "PYTHONHASHSEED",
            "PYTHONIOENCODING",
            "PYTHONPATH",
            "PYTHONUTF8",
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_FILE",
            "TEMP",
            "TMP",
            "TMPDIR",
        }
        and value
    }
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _tail_text(value: str | None, *, limit: int = 4000) -> str:
    return "" if value is None else value[-limit:]


def _write_step_events(
    progress_path: Path,
    plan: NotebookStepPlan,
    cell_events: dict[int, dict[str, Any]],
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = progress_path.with_name(f"{progress_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            _build_step_events(plan, cell_events),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(progress_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_cell_start(
    cell_events: dict[int, dict[str, Any]],
    *,
    cell,
    cell_index: int,
    **_: Any,
) -> None:
    cell_events[cell_index] = {
        "cell_index": cell_index,
        "cell_type": cell.cell_type,
        "status": "running",
        "started_at": _utc_now(),
        "ended_at": None,
        "stdout_preview": "",
        "stderr_preview": "",
        "exception_name": None,
        "exception_value": None,
        "traceback_preview": "",
    }


def _record_cell_error(
    cell_events: dict[int, dict[str, Any]],
    *,
    cell,
    cell_index: int,
    execute_reply: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    event = cell_events.setdefault(cell_index, {"cell_index": cell_index})
    content = (execute_reply or {}).get("content", {})
    event.update(
        {
            "cell_type": cell.cell_type,
            "status": "failed",
            "exception_name": content.get("ename"),
            "exception_value": content.get("evalue"),
            "traceback_preview": _preview("\n".join(content.get("traceback", []))),
        }
    )


def _record_cell_executed(
    cell_events: dict[int, dict[str, Any]],
    *,
    cell,
    cell_index: int,
    defer_completion: bool = False,
    **_: Any,
) -> None:
    event = cell_events.setdefault(cell_index, {"cell_index": cell_index})
    if event.get("status") != "failed" and not defer_completion:
        event["status"] = "succeeded"
    event["cell_type"] = cell.cell_type


def _record_cell_complete(
    cell_events: dict[int, dict[str, Any]],
    *,
    cell,
    cell_index: int,
    defer_completion: bool = False,
    **_: Any,
) -> None:
    event = cell_events.setdefault(cell_index, {"cell_index": cell_index})
    if event.get("status") != "failed" and not defer_completion:
        event["status"] = "succeeded"
    if not defer_completion:
        event["ended_at"] = _utc_now()
    stdout, stderr = _output_previews(cell)
    event["stdout_preview"] = stdout
    event["stderr_preview"] = stderr


def _finalize_cell_event_after_execute(
    cell_events: dict[int, dict[str, Any]],
    *,
    cell,
    cell_index: int,
    succeeded: bool,
) -> None:
    event = cell_events.get(cell_index)
    if event is None:
        return
    if succeeded:
        event["status"] = "succeeded"
    elif event.get("status") != "failed":
        event["status"] = "failed"
    event["cell_type"] = cell.cell_type
    event["ended_at"] = _utc_now()
    stdout, stderr = _output_previews(cell)
    event["stdout_preview"] = stdout
    event["stderr_preview"] = stderr


def _output_previews(cell) -> tuple[str, str]:
    stdout: list[str] = []
    stderr: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            if output.get("name") == "stderr":
                stderr.append(str(output.get("text", "")))
            else:
                stdout.append(str(output.get("text", "")))
    return _preview("".join(stdout)), _preview("".join(stderr))


def _preview(value: str, limit: int = 800) -> str:
    return value[:limit]


def _build_step_events(
    plan: NotebookStepPlan,
    cell_events: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    cells = []
    for cell_index, event in sorted(cell_events.items()):
        cells.append(
            {
                **event,
                "step_id": plan.cell_to_step.get(cell_index),
            }
        )

    steps = []
    for order, step in enumerate(plan.steps, start=1):
        step_cell_events = [
            cell_events[cell_index]
            for cell_index in step.cell_indexes
            if cell_index in cell_events
        ]
        statuses = [
            cell_events.get(cell_index, {}).get("status", "pending")
            for cell_index in step.cell_indexes
        ]
        if any(status == "failed" for status in statuses):
            status = "failed"
        elif any(status == "running" for status in statuses):
            status = "running"
        elif statuses and all(status == "succeeded" for status in statuses):
            status = "succeeded"
        else:
            status = "pending"
        started_at, ended_at, elapsed_seconds = _step_timing(step_cell_events, status)
        steps.append(
            {
                "id": step.id,
                "step_order": order,
                "title": step.title,
                "status": status,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_seconds": elapsed_seconds,
                "cell_count": len(step.cell_indexes),
                "cell_indexes": step.cell_indexes,
                "source_previews": step.source_previews,
                "system": step.system,
            }
        )

    return {"steps": steps, "cells": cells}


def _step_timing(
    events: list[dict[str, Any]],
    status: str,
) -> tuple[str | None, str | None, float | None]:
    started_values = [
        str(event.get("started_at"))
        for event in events
        if event.get("started_at")
    ]
    if not started_values:
        return None, None, None

    started_at = min(started_values)
    start_dt = _parse_timestamp(started_at)
    if start_dt is None:
        return started_at, None, None

    ended_values = [
        str(event.get("ended_at"))
        for event in events
        if event.get("ended_at")
    ]
    if status == "running" or not ended_values:
        end_dt = datetime.now(timezone.utc)
        ended_at = None
    else:
        ended_at = max(ended_values)
        end_dt = _parse_timestamp(ended_at) or datetime.now(timezone.utc)

    elapsed = max(0.0, (end_dt - start_dt).total_seconds())
    return started_at, ended_at, float(round(elapsed, 3))


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed

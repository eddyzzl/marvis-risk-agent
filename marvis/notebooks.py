from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import nbformat

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
    ) -> None:
        self.notebook_path = notebook_path
        self.executed_path = executed_path
        self.log_path = log_path
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.progress_path = progress_path
        self.execution_cwd = execution_cwd
        self.cancellation_token = cancellation_token
        self.notebook = nbformat.read(notebook_path, as_version=4)
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
                cleanup()
                return
            except Exception:
                pass
        kernel_manager = getattr(self.client, "km", None)
        if kernel_manager is not None:
            try:
                kernel_manager.shutdown_kernel(now=True)
            except Exception:
                pass

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
        try:
            execute()
        except CellExecutionError:
            nbformat.write(self.notebook, self.executed_path)
            failed_cell_index = _failed_cell_index(self.notebook)
            error_name, error_value = _error_output(self.notebook)
            if self.cancellation_token is not None and self.cancellation_token.is_cancelled():
                _write_cancel_log(log_path)
                return NotebookRunResult(
                    succeeded=False,
                    failed_cell_index=failed_cell_index,
                    error_name="NotebookCancelled",
                    error_value="notebook execution cancelled",
                    step_events=self._step_events(),
                    cancelled=True,
                )
            _write_failure_log(log_path, failed_cell_index, error_name, error_value)
            return NotebookRunResult(
                succeeded=False,
                failed_cell_index=failed_cell_index,
                error_name=error_name,
                error_value=error_value,
                step_events=self._step_events(),
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
            )
        except Exception as error:
            nbformat.write(self.notebook, self.executed_path)
            failed_cell_index = _failed_cell_index(self.notebook)
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
                )
            _write_failure_log(log_path, failed_cell_index, error_name, error_value)
            return NotebookRunResult(
                succeeded=False,
                failed_cell_index=failed_cell_index,
                error_name=error_name,
                error_value=error_value,
                step_events=self._step_events(),
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
        )

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
    nbformat.write(notebook, output_notebook)
    return output_notebook


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
) -> NotebookRunResult:
    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=timeout,
        kernel_name=kernel_name,
        progress_path=progress_path,
        execution_cwd=execution_cwd,
        cancellation_token=cancellation_token,
    )
    try:
        return session.execute_notebook(keep_alive=False)
    finally:
        session.close()


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

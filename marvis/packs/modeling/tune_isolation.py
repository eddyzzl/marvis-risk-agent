"""Short-lived worker boundary for one modeling tuning recipe.

The public ``tune_hyperparameters`` tool is itself executed in a governed
plugin worker.  Running LightGBM, XGBoost, and CatBoost sequentially inside
that same long-lived process retains native allocator high-water marks between
recipes even after Python objects are collected.  This module keeps the outer
worker as a lightweight checkpoint/progress aggregator and runs exactly one
recipe in each nested ``subprocess_worker`` invocation.

The nested worker deliberately remains in the outer worker's process group.
That gives both required lifecycle guarantees:

* normal recipe completion exits the interpreter and releases all native RSS;
* host timeout/cancellation of the outer worker also kills the active recipe
  worker, so no orphaned learner can survive the governed tool invocation.

No shell is involved.  The child uses the same versioned JSON worker protocol,
network guard, deterministic seeding, and progress-file channel as every other
built-in tool worker.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Callable

from marvis.packs.modeling.errors import ModelingError
from marvis.plugins.contracts import MAX_PROGRESS_BYTES, PROTOCOL_VERSION
from marvis.plugins.runner import (
    DEFAULT_RSS_MEMORY_LIMIT_MB,
    WorkerResourceLimitExceeded,
    _aggregate_cpu_limit_seconds,
    _check_worker_protocol_version,
    _cleanup_progress_path,
    _parse_worker_result,
    _run_worker,
    _try_new_progress_path,
)
from marvis.redaction import redact_text, redact_value
from marvis.safe_paths import assert_within


_RECIPE_WORKER_TIMEOUT_SECONDS = 12 * 60 * 60
_RECIPE_WORKER_MEMORY_LIMIT_MB = DEFAULT_RSS_MEMORY_LIMIT_MB
MAX_RECIPE_PROGRESS_JOURNAL_BYTES = 8 * 1024 * 1024
_MAX_ERROR_TEXT_CHARS = 4_000
logger = logging.getLogger(__name__)


class IsolatedRecipeTuningError(ModelingError):
    """Preserve child diagnostics across the nested worker boundary."""

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        self.detail = dict(detail) if isinstance(detail, dict) else None
        super().__init__(message)

    def to_detail(self) -> dict:
        return self.detail or {"kind": "isolated_recipe_tuning_failed"}


def run_tuning_recipe_isolated(
    recipe_inputs: dict,
    *,
    ctx,
    progress_callback: Callable[[dict], None] | None,
) -> dict:
    """Run one recipe in a fresh interpreter and return its JSON result.

    ``recipe_inputs`` is an internal, parent-constructed payload rather than a
    user-facing tool input.  It contains the exact arguments previously passed
    to :func:`tune_hyperparameters`; therefore RNG seeds, trial ordering, and
    result selection semantics are unchanged.
    """

    journal_path = (
        _try_new_progress_path(Path(ctx.workspace))
        if callable(progress_callback)
        else None
    )
    watcher = (
        _RecipeProgressJournalWatcher(journal_path, progress_callback)
        if journal_path is not None and callable(progress_callback)
        else None
    )
    child_inputs = dict(recipe_inputs)
    if journal_path is not None:
        child_inputs["_progress_journal_path"] = str(journal_path)
    job = {
        "protocol_version": PROTOCOL_VERSION,
        "module": "marvis.packs.modeling.tune_isolation",
        "entrypoint": "tool_tune_one_recipe_isolated",
        "inputs": child_inputs,
        "task_id": str(ctx.task_id),
        "seed": int(recipe_inputs["seed"]),
        "datasets_root": str(ctx.datasets_root),
        "workspace": str(ctx.workspace),
        "memory_limit_mb": _RECIPE_WORKER_MEMORY_LIMIT_MB,
        "cpu_limit_seconds": _aggregate_cpu_limit_seconds(
            _RECIPE_WORKER_TIMEOUT_SECONDS
        ),
        "file_size_limit_mb": 2048,
        "plugin_paths": [],
        "side_effects": ["read:dataset"],
        "builtin": True,
    }
    if watcher is not None:
        watcher.start()
    try:
        try:
            completed = _run_worker(
                sys.executable,
                job,
                timeout=_RECIPE_WORKER_TIMEOUT_SECONDS,
                rss_limit_mb=_RECIPE_WORKER_MEMORY_LIMIT_MB,
                # Trial events use the append-only internal journal below.  The
                # public latest-snapshot progress file can coalesce fast trials;
                # the journal preserves every callback in order before the recipe
                # process exits.
                progress_path=None,
                progress_callback=None,
                # Keep the nested worker in the aggregate worker's process group.
                # See module docstring and runner._run_worker for cancellation.
                start_new_session=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise IsolatedRecipeTuningError(
                f"isolated tuning worker timed out after {int(exc.timeout)} seconds",
                detail={
                    "kind": "timeout",
                    "subkind": "isolated_recipe_timeout",
                    "timeout_seconds": int(exc.timeout),
                    "stdout": _safe_diagnostic_text(exc.output),
                    "stderr": _safe_diagnostic_text(exc.stderr),
                },
            ) from exc
        except WorkerResourceLimitExceeded as exc:
            limits = _sanitize_error_value(exc.resource_usage)
            child_detail = _sanitize_error_value(
                getattr(exc, "error_detail", None)
            )
            child_detail = child_detail if isinstance(child_detail, dict) else {}
            child_kind = str(child_detail.get("kind") or "worker_rss_limit")
            raise IsolatedRecipeTuningError(
                _safe_diagnostic_text(exc),
                detail={
                    "kind": "resource_limit",
                    "subkind": (
                        "isolated_recipe_cpu_limit"
                        if child_kind == "worker_cpu_limit"
                        else "isolated_recipe_rss_limit"
                    ),
                    "resource_limits": limits if isinstance(limits, dict) else {},
                    "child_error_detail": child_detail,
                },
            ) from exc
    finally:
        if watcher is not None:
            watcher.close()
        _cleanup_progress_path(journal_path)
    protocol = _parse_worker_result(completed.stdout)
    if protocol is None:
        sigxcpu = getattr(signal, "SIGXCPU", None)
        if sigxcpu is not None and completed.returncode == -int(sigxcpu):
            limits = {
                "cpu_limit_seconds": int(job["cpu_limit_seconds"]),
                "cpu_limit_exceeded": True,
                "termination_signal": "SIGXCPU",
                "worker_returncode": int(completed.returncode),
            }
            raise IsolatedRecipeTuningError(
                "isolated tuning worker exceeded its aggregate CPU limit",
                detail={
                    "kind": "resource_limit",
                    "subkind": "isolated_recipe_cpu_limit",
                    "resource_limits": limits,
                    "stderr": _safe_diagnostic_text(completed.stderr),
                },
            )
        raise IsolatedRecipeTuningError(
            "isolated tuning worker returned no valid protocol result",
            detail={
                "kind": "isolated_recipe_tuning_protocol",
                "worker_returncode": int(completed.returncode),
                "stderr": _safe_diagnostic_text(completed.stderr),
            },
        )
    version_error = _check_worker_protocol_version(protocol)
    if version_error is not None:
        raise IsolatedRecipeTuningError(
            version_error,
            detail={"kind": "protocol_version_mismatch"},
        )
    if not protocol.get("ok"):
        detail = protocol.get("error_detail")
        sanitized_detail = _sanitize_error_value(detail)
        if not isinstance(sanitized_detail, dict):
            sanitized_detail = {
                "kind": str(protocol.get("error_kind") or "execution"),
                "traceback": _safe_diagnostic_text(protocol.get("traceback")),
            }
        resource_limits = _sanitize_error_value(protocol.get("resource_limits"))
        if isinstance(resource_limits, dict):
            sanitized_detail["resource_limits"] = resource_limits
        if str(protocol.get("error_kind") or "") in {"resource", "resource_limit"}:
            sanitized_detail["kind"] = "resource_limit"
            sanitized_detail.setdefault("subkind", "isolated_recipe_resource_limit")
        raise IsolatedRecipeTuningError(
            _safe_diagnostic_text(
                protocol.get("error") or "isolated recipe tuning failed"
            ),
            detail=sanitized_detail,
        )
    output = protocol.get("output")
    if not isinstance(output, dict):
        raise IsolatedRecipeTuningError(
            f"isolated tuning worker returned {type(output).__name__}, expected object",
            detail={"kind": "isolated_recipe_tuning_protocol"},
        )
    return output


def tool_tune_one_recipe_isolated(inputs: dict, ctx) -> dict:
    """Nested-worker entrypoint: load/tune exactly one recipe, then exit."""

    # Imports stay inside the child entrypoint.  The aggregate worker never
    # materialises a training frame and never calls the in-process tuner.
    from marvis.packs.modeling._common import _jsonable
    from marvis.packs.modeling._runtime import _runtime
    from marvis.packs.modeling.tune import tune_hyperparameters

    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    dataset_path = runtime.registry.resolve_path(dataset.id)
    result = tune_hyperparameters(
        runtime.backend,
        dataset_path,
        features=[str(item) for item in inputs["features"]],
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]),
        recipe=str(inputs["recipe"]),
        n_trials=int(inputs["n_trials"]),
        seed=int(inputs["seed"]),
        early_stopping_rounds=int(inputs["early_stopping_rounds"]),
        max_boost_round=int(inputs["max_boost_round"]),
        overfit_penalty=float(inputs["overfit_penalty"]),
        sample_weight_col=str(inputs.get("sample_weight_col") or ""),
        base_params=dict(inputs.get("base_params") or {}),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        cv_folds=(
            int(inputs["cv_folds"])
            if inputs.get("cv_folds") is not None
            else None
        ),
        progress_callback=lambda event: _append_progress_journal(
            ctx,
            inputs.get("_progress_journal_path"),
            event,
        ),
    )
    return {
        # Internal lifecycle evidence only; the aggregate tool never copies it
        # into its public output.  Keeping the PID alongside the protocol result
        # lets regression tests prove that each interpreter has exited before
        # the next recipe starts.
        "_worker_pid": os.getpid(),
        "best_params": _jsonable(result.best_params),
        "best_metrics": _jsonable(result.best_metrics),
        "n_trials": int(result.n_trials),
        "trials": _jsonable(result.trials),
        "nan_labels_dropped": int(result.nan_labels_dropped),
    }


def _append_progress_journal(ctx, raw_path, event: dict) -> None:
    """Append one bounded event; telemetry failure never changes tuning."""

    if not raw_path or not isinstance(event, dict):
        return
    try:
        path = assert_within(Path(ctx.workspace), Path(str(raw_path)))
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        if len(encoded) > MAX_PROGRESS_BYTES:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            current_size = int(os.fstat(descriptor).st_size)
            if current_size + len(encoded) > MAX_RECIPE_PROGRESS_JOURNAL_BYTES:
                return
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write to isolated tuning progress journal")
                view = view[written:]
        finally:
            os.close(descriptor)
    except Exception:
        logger.warning("failed to append isolated tuning progress", exc_info=True)


class _RecipeProgressJournalWatcher:
    """Forward every JSONL trial event exactly once and in write order."""

    def __init__(self, path: Path, callback: Callable[[dict], None]) -> None:
        self._path = Path(path)
        self._callback = callback
        self._offset = 0
        self._buffer = b""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"marvis-recipe-progress-{self._path.stem[:12]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._deliver_available()
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._deliver_available()

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self._deliver_available()

    def _deliver_available(self) -> None:
        with self._lock:
            try:
                with self._path.open("rb") as handle:
                    handle.seek(self._offset)
                    chunk = handle.read()
                    self._offset = handle.tell()
            except (FileNotFoundError, OSError):
                return
            if not chunk:
                return
            self._buffer += chunk
            lines = self._buffer.split(b"\n")
            self._buffer = lines.pop()
        for line in lines:
            if not line or len(line) > MAX_PROGRESS_BYTES:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
                if isinstance(payload, dict):
                    self._callback(payload)
            except Exception:
                # The observation channel must not affect result ordering or
                # make an otherwise valid recipe fail.
                logger.warning("failed to forward isolated tuning progress", exc_info=True)


def _safe_diagnostic_text(value, *, limit: int = _MAX_ERROR_TEXT_CHARS) -> str:
    """Redact first, then truncate anything that can cross into persisted audit."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return redact_text(text)[: max(0, int(limit))]


def _sanitize_error_value(value):
    """Bound and redact nested worker diagnostics while preserving typed fields."""

    return _bound_error_value(redact_value(value).value)


def _bound_error_value(value, *, depth: int = 0):
    if depth >= 5:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str | bytes):
        return _safe_diagnostic_text(value)
    if isinstance(value, dict):
        return {
            _safe_diagnostic_text(key, limit=120): _bound_error_value(item, depth=depth + 1)
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple)):
        return [_bound_error_value(item, depth=depth + 1) for item in value[:30]]
    return _safe_diagnostic_text(value)


__all__ = [
    "IsolatedRecipeTuningError",
    "MAX_RECIPE_PROGRESS_JOURNAL_BYTES",
    "run_tuning_recipe_isolated",
    "tool_tune_one_recipe_isolated",
]

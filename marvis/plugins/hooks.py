from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import logging

from marvis.db import PluginRepository
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry
from marvis.plugins.runner import ToolResult


logger = logging.getLogger(__name__)
HookListener = Callable[[str, dict], None]


class HookDispatcher:
    def __init__(
        self,
        plugin_registry: PluginRegistry,
        tool_runner,
        repo: PluginRepository | None = None,
    ):
        self._plugins = plugin_registry
        self._runner = tool_runner
        self._repo = repo
        self._index: dict[str, list[ToolRef]] = {}
        self._listeners: dict[str, list[HookListener]] = defaultdict(list)

    def rebuild_index(self) -> None:
        index: dict[str, list[ToolRef]] = defaultdict(list)
        for manifest in self._plugins.list():
            for hook in manifest.hooks:
                index[hook.event].append(
                    ToolRef(manifest.name, hook.tool, manifest.version)
                )
        self._index = dict(index)

    def register_listener(self, event: str, listener: HookListener) -> None:
        self._listeners[str(event)].append(listener)

    def listener_count(self, event: str) -> int:
        return len(self._listeners.get(str(event), []))

    def dispatch(self, event: str, payload: dict, *, task_id: str) -> list[ToolResult]:
        for listener in self._listeners.get(event, []):
            listener_ref = _listener_ref(listener)
            if not self._write_listener_started(event, listener_ref, task_id):
                continue
            error: Exception | None = None
            try:
                listener(event, payload)
            except Exception as exc:
                error = exc
                logger.warning(
                    "builtin hook listener failed for %s/%s: %s",
                    event,
                    task_id,
                    exc,
                )
            self._write_listener_audit(event, listener_ref, task_id, error)
        results: list[ToolResult] = []
        for ref in self._index.get(event, []):
            start_error = self._write_dispatch_started(event, ref, task_id)
            if start_error is not None:
                results.append(start_error)
                continue
            try:
                result = self._runner.invoke(ref, payload, task_id=task_id)
            except Exception as exc:
                result = ToolResult(
                    ok=False,
                    output=None,
                    error=str(exc),
                    error_kind="hook",
                    duration_ms=0,
                )
            audit_error = self._write_audit(event, ref, task_id, result)
            results.append(audit_error or result)
        return results

    def _write_dispatch_started(
        self,
        event: str,
        ref: ToolRef,
        task_id: str,
    ) -> ToolResult | None:
        if self._repo is None:
            return None
        try:
            self._repo.write_audit(
                kind="hook.dispatch.started",
                target_ref=_target_ref(ref),
                outcome="started",
                detail={
                    "event": event,
                    "task_id": task_id,
                },
            )
        except Exception as exc:
            return _audit_failure_result("start", exc)
        return None

    def _write_audit(
        self,
        event: str,
        ref: ToolRef,
        task_id: str,
        result: ToolResult,
    ) -> ToolResult | None:
        if self._repo is None:
            return None
        try:
            self._repo.write_audit(
                kind="hook.dispatch",
                target_ref=_target_ref(ref),
                outcome="succeeded" if result.ok else "failed",
                detail={
                    "event": event,
                    "task_id": task_id,
                    "error_kind": result.error_kind,
                    "duration_ms": result.duration_ms,
                },
            )
        except Exception as exc:
            return _audit_failure_result("finish", exc, result=result)
        return None

    def _write_listener_started(
        self,
        event: str,
        listener_ref: str,
        task_id: str,
    ) -> bool:
        if self._repo is None:
            return True
        try:
            self._repo.write_audit(
                kind="hook.listener.started",
                target_ref=listener_ref,
                outcome="started",
                detail={
                    "event": event,
                    "task_id": task_id,
                },
            )
        except Exception as exc:
            logger.warning(
                "builtin hook listener checkpoint failed for %s/%s: %s",
                event,
                task_id,
                exc,
            )
            return False
        return True

    def _write_listener_audit(
        self,
        event: str,
        listener_ref: str,
        task_id: str,
        error: Exception | None,
    ) -> None:
        if self._repo is None:
            return
        try:
            self._repo.write_audit(
                kind="hook.listener",
                target_ref=listener_ref,
                outcome="failed" if error else "succeeded",
                detail={
                    "event": event,
                    "task_id": task_id,
                    "error_kind": error.__class__.__name__ if error else None,
                },
            )
        except Exception as exc:
            logger.warning(
                "builtin hook listener audit failed for %s/%s: %s",
                event,
                task_id,
                exc,
            )


def _target_ref(ref: ToolRef) -> str:
    suffix = f"@{ref.version}" if ref.version else ""
    return f"{ref.label()}{suffix}"


def _listener_ref(listener: HookListener) -> str:
    module = getattr(listener, "__module__", "")
    name = getattr(listener, "__qualname__", getattr(listener, "__name__", repr(listener)))
    return f"builtin:{module}.{name}".strip(".")


def _audit_failure_result(
    phase: str,
    exc: Exception,
    *,
    result: ToolResult | None = None,
) -> ToolResult:
    detail = {
        "audit_phase": phase,
        "audit_error": str(exc),
    }
    if result is not None:
        detail["result_ok"] = result.ok
        detail["result_error_kind"] = result.error_kind
    return ToolResult(
        ok=False,
        output=None,
        error=f"audit {phase} failed: {exc}",
        error_kind="audit",
        duration_ms=result.duration_ms if result is not None else 0,
        error_detail=detail,
    )

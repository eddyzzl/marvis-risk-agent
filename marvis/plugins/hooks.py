from __future__ import annotations

from collections import defaultdict

from marvis.db import PluginRepository
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry
from marvis.plugins.runner import ToolResult


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

    def rebuild_index(self) -> None:
        index: dict[str, list[ToolRef]] = defaultdict(list)
        for manifest in self._plugins.list():
            for hook in manifest.hooks:
                index[hook.event].append(
                    ToolRef(manifest.name, hook.tool, manifest.version)
                )
        self._index = dict(index)

    def dispatch(self, event: str, payload: dict, *, task_id: str) -> list[ToolResult]:
        results: list[ToolResult] = []
        for ref in self._index.get(event, []):
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
            self._write_audit(event, ref, task_id, result)
            results.append(result)
        return results

    def _write_audit(
        self,
        event: str,
        ref: ToolRef,
        task_id: str,
        result: ToolResult,
    ) -> None:
        if self._repo is None:
            return
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


def _target_ref(ref: ToolRef) -> str:
    suffix = f"@{ref.version}" if ref.version else ""
    return f"{ref.label()}{suffix}"

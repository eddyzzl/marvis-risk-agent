from __future__ import annotations

from collections import defaultdict

from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry
from marvis.plugins.runner import ToolResult


class HookDispatcher:
    def __init__(self, plugin_registry: PluginRegistry, tool_runner):
        self._plugins = plugin_registry
        self._runner = tool_runner
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
                results.append(self._runner.invoke(ref, payload, task_id=task_id))
            except Exception as exc:
                results.append(
                    ToolResult(
                        ok=False,
                        output=None,
                        error=str(exc),
                        error_kind="hook",
                        duration_ms=0,
                    )
                )
        return results

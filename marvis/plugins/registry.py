from __future__ import annotations

import json

from marvis.db import PluginRepository
from marvis.plugins.errors import (
    DuplicatePluginError,
    PluginNotFoundError,
    ToolNotFoundError,
)
from marvis.plugins.manifest import (
    PluginManifest,
    ToolRef,
    ToolSpec,
    parse_manifest,
)


class PluginRegistry:
    def __init__(self, repo: PluginRepository):
        self._repo = repo
        self._plugins: dict[str, tuple[PluginManifest, bool]] = {}

    def load_from_db(self) -> None:
        self._plugins.clear()
        for row in self._repo.list_plugins(include_disabled=True):
            data = json.loads(row["manifest_json"])
            manifest = parse_manifest(data, builtin=bool(row["builtin"]))
            self._plugins[manifest.name] = (manifest, bool(row["enabled"]))

    def register(self, manifest: PluginManifest, *, enabled: bool = True) -> None:
        existing = self._plugins.get(manifest.name)
        if existing is not None and existing[0].version == manifest.version:
            raise DuplicatePluginError(f"{manifest.name}@{manifest.version} already registered")
        self._repo.upsert_plugin(manifest, enabled=enabled)
        self._repo.write_audit(
            kind="plugin.register",
            target_ref=manifest.name,
            outcome="succeeded",
            detail={
                "version": manifest.version,
                "builtin": manifest.builtin,
                "enabled": bool(enabled),
            },
        )
        self._plugins[manifest.name] = (manifest, bool(enabled))

    def remove(self, name: str) -> None:
        manifest, _enabled = self._require(name)
        if manifest.builtin:
            raise ValueError("cannot remove builtin plugin")
        self._repo.delete_plugin(name)
        self._repo.write_audit(
            kind="plugin.remove",
            target_ref=name,
            outcome="succeeded",
            detail={"version": manifest.version},
        )
        del self._plugins[name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        manifest, _current = self._require(name)
        self._repo.set_enabled(name, enabled)
        self._repo.write_audit(
            kind="plugin.enable" if enabled else "plugin.disable",
            target_ref=name,
            outcome="succeeded",
            detail={"version": manifest.version, "enabled": bool(enabled)},
        )
        self._plugins[name] = (manifest, bool(enabled))

    def get(self, name: str) -> PluginManifest:
        return self._require(name)[0]

    def is_enabled(self, name: str) -> bool:
        return self._require(name)[1]

    def list(self, *, include_disabled: bool = False) -> list[PluginManifest]:
        items = sorted(self._plugins.items(), key=lambda item: item[0])
        return [
            manifest
            for _name, (manifest, enabled) in items
            if include_disabled or enabled
        ]

    def _require(self, name: str) -> tuple[PluginManifest, bool]:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise PluginNotFoundError(name) from exc


class ToolRegistry:
    def __init__(self, plugin_registry: PluginRegistry):
        self._plugins = plugin_registry

    def resolve(self, ref: ToolRef) -> ToolSpec:
        _manifest, tool = self.resolve_with_manifest(ref)
        return tool

    def resolve_with_manifest(self, ref: ToolRef) -> tuple[PluginManifest, ToolSpec]:
        try:
            manifest = self._plugins.get(ref.plugin)
        except PluginNotFoundError:
            raise
        if not self._plugins.is_enabled(ref.plugin):
            raise PluginNotFoundError(ref.plugin)
        if ref.version and ref.version != manifest.version:
            raise ToolNotFoundError(
                f"{ref.label()} version {ref.version} does not match {manifest.version}"
            )
        for tool in manifest.tools:
            if tool.name == ref.tool:
                return manifest, tool
        raise ToolNotFoundError(ref.label())

    def catalog_for_planner(self) -> list[dict]:
        catalog: list[dict] = []
        for manifest in self._plugins.list():
            for tool in manifest.tools:
                catalog.append({
                    "plugin": manifest.name,
                    "tool": tool.name,
                    "version": manifest.version,
                    "summary": tool.summary,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "determinism": tool.determinism,
                })
        return catalog

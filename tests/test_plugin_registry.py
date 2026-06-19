import pytest

from marvis.db import PluginRepository, init_db
from marvis.plugins.errors import (
    DuplicatePluginError,
    PluginNotFoundError,
    ToolNotFoundError,
)
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _manifest(version: str = "0.1.0", *, tool_name: str = "echo"):
    return parse_manifest(
        {
            "name": "_sample",
            "version": version,
            "display_name": "Sample Echo Pack",
            "description": "Runtime smoke-test pack",
            "module": "marvis.packs._sample.tools",
            "tools": [
                {
                    "name": tool_name,
                    "summary": "Echo a message",
                    "input_schema": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"echoed": {"type": "string"}},
                        "required": ["echoed"],
                    },
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_echo",
                    "side_effects": ["read:input"],
                }
            ],
            "hooks": [],
            "permissions": [],
        },
        builtin=True,
    )


def test_plugin_registry_registers_and_loads_from_repository(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)

    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)

    loaded = PluginRegistry(repo)
    loaded.load_from_db()

    assert loaded.get("_sample").version == "0.1.0"
    assert loaded.is_enabled("_sample") is True
    assert [plugin.name for plugin in loaded.list()] == ["_sample"]


def test_plugin_registry_rejects_same_name_and_version_duplicate(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)

    with pytest.raises(DuplicatePluginError):
        registry.register(_manifest(), enabled=True)


def test_plugin_registry_upgrade_replaces_existing_manifest(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)

    registry.register(_manifest("0.2.0", tool_name="reverse"), enabled=False)

    assert registry.get("_sample").version == "0.2.0"
    assert registry.is_enabled("_sample") is False
    assert registry.get("_sample").tools[0].name == "reverse"


def test_plugin_registry_remove_builtin_is_rejected(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)

    with pytest.raises(ValueError, match="builtin"):
        registry.remove("_sample")


def test_tool_registry_resolves_enabled_tools_and_rejects_disabled_or_wrong_version(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)
    tools = ToolRegistry(registry)

    assert tools.resolve(ToolRef("_sample", "echo")).name == "echo"
    assert tools.resolve(ToolRef("_sample", "echo", "0.1.0")).name == "echo"

    with pytest.raises(ToolNotFoundError, match="version"):
        tools.resolve(ToolRef("_sample", "echo", "9.9.9"))

    registry.set_enabled("_sample", False)
    with pytest.raises(PluginNotFoundError):
        tools.resolve(ToolRef("_sample", "echo"))


def test_tool_registry_catalog_for_planner_is_compact_and_safe(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)

    catalog = ToolRegistry(registry).catalog_for_planner()

    assert catalog == [
        {
            "plugin": "_sample",
            "tool": "echo",
            "version": "0.1.0",
            "summary": "Echo a message",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"echoed": {"type": "string"}},
                "required": ["echoed"],
            },
            "determinism": "deterministic",
            "side_effects": ["read:input"],
        }
    ]
    assert "entrypoint" not in catalog[0]

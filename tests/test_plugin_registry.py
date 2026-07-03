import pytest

from marvis.db import PluginRepository, init_db
from marvis.plugins.errors import (
    DuplicatePluginError,
    ManifestError,
    PluginNotFoundError,
    ToolNotFoundError,
)
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _manifest(version: str = "0.1.0", *, tool_name: str = "echo", python_requires: str = ""):
    return parse_manifest(
        {
            "name": "_sample",
            "version": version,
            "display_name": "Sample Echo Pack",
            "description": "Runtime smoke-test pack",
            "module": "marvis.packs._sample.tools",
            "python_requires": python_requires,
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
            "permissions": ["read:input"],
        },
        builtin=True,
    )


def _uploaded_manifest():
    return parse_manifest(
        {
            "name": "uploaded_pack",
            "version": "0.1.0",
            "display_name": "Uploaded Pack",
            "description": "Uploaded runtime pack",
            "module": "uploaded_pack.tools",
            "tools": [
                {
                    "name": "echo",
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
                }
            ],
            "hooks": [],
            "permissions": [],
        },
        builtin=False,
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
    assert repo.list_audit(kind="plugin.register")[0]["target_ref"] == "_sample"


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
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_manifest(), enabled=True)

    registry.register(_manifest("0.2.0", tool_name="reverse"), enabled=False)

    assert registry.get("_sample").version == "0.2.0"
    assert registry.is_enabled("_sample") is False
    assert registry.get("_sample").tools[0].name == "reverse"
    assert [audit["target_ref"] for audit in repo.list_audit(kind="plugin.register")] == [
        "_sample",
        "_sample",
    ]


def test_plugin_registry_remove_builtin_is_rejected(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(), enabled=True)

    with pytest.raises(ValueError, match="builtin"):
        registry.remove("_sample")


def test_plugin_registry_enable_disable_and_remove_write_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    registry.register(_uploaded_manifest(), enabled=True)

    registry.set_enabled("uploaded_pack", False)
    registry.set_enabled("uploaded_pack", True)
    registry.remove("uploaded_pack")

    audits = repo.list_audit()
    assert [audit["kind"] for audit in audits] == [
        "plugin.register",
        "plugin.disable",
        "plugin.enable",
        "plugin.remove",
    ]
    assert [audit["target_ref"] for audit in audits] == [
        "uploaded_pack",
        "uploaded_pack",
        "uploaded_pack",
        "uploaded_pack",
    ]


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


def test_tool_registry_rejects_plugins_with_incompatible_python_requires(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    registry.register(_manifest(python_requires=">=999.0"), enabled=True)
    tools = ToolRegistry(registry)

    with pytest.raises(ToolNotFoundError, match="requires Python"):
        tools.resolve(ToolRef("_sample", "echo"))
    assert tools.catalog_for_planner() == []


def test_parse_manifest_requires_side_effects_to_be_declared_in_permissions():
    with pytest.raises(ManifestError, match="side_effects not declared"):
        parse_manifest(
            {
                "name": "unsafe_pack",
                "version": "0.1.0",
                "display_name": "Unsafe Pack",
                "description": "Missing permissions",
                "module": "unsafe_pack.tools",
                "tools": [
                    {
                        "name": "write_data",
                        "summary": "Writes a dataset",
                        "input_schema": {"type": "object", "additionalProperties": False},
                        "output_schema": {"type": "object", "additionalProperties": False},
                        "determinism": "deterministic",
                        "timeout_seconds": 10,
                        "failure_policy": "fail",
                        "entrypoint": "run",
                        "side_effects": ["write:dataset"],
                    }
                ],
                "hooks": [],
                "permissions": ["read:dataset"],
            },
            builtin=False,
        )


def test_parse_manifest_rejects_unknown_permissions_and_side_effects():
    base = {
        "name": "bad_vocab_pack",
        "version": "0.1.0",
        "display_name": "Bad Vocabulary Pack",
        "description": "Unknown permission values",
        "module": "bad_vocab.tools",
        "tools": [
            {
                "name": "run",
                "summary": "Run",
                "input_schema": {"type": "object", "additionalProperties": False},
                "output_schema": {"type": "object", "additionalProperties": False},
                "determinism": "deterministic",
                "timeout_seconds": 10,
                "failure_policy": "fail",
                "entrypoint": "run",
                "side_effects": [],
            }
        ],
        "hooks": [],
        "permissions": ["write:anything"],
    }

    with pytest.raises(ManifestError, match="unknown permission"):
        parse_manifest(base, builtin=False)

    base["permissions"] = ["read:dataset"]
    base["tools"][0]["side_effects"] = ["write:anything"]
    with pytest.raises(ManifestError, match="unknown permission"):
        parse_manifest(base, builtin=False)


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
        }
    ]
    assert "entrypoint" not in catalog[0]
    assert "side_effects" not in catalog[0]

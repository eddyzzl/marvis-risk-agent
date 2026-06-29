import json

import pytest

from marvis.db import PluginRepository, connect, init_db
import marvis.db as db_module
from marvis.plugins.errors import PluginNotFoundError
from marvis.plugins.manifest import parse_manifest


def _manifest(version: str = "0.1.0", *, tools: list[dict] | None = None):
    return parse_manifest(
        {
            "name": "_sample",
            "version": version,
            "display_name": "Sample Echo Pack",
            "description": "Runtime smoke-test pack",
            "module": "marvis.packs._sample.tools",
            "tools": tools
            or [
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
        builtin=True,
    )


def test_plugin_repository_upserts_and_lists_plugins_and_tools(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)

    repo.upsert_plugin(_manifest(), enabled=True)

    plugin = repo.get_plugin("_sample")
    assert plugin is not None
    assert plugin["name"] == "_sample"
    assert plugin["version"] == "0.1.0"
    assert plugin["enabled"] is True
    assert plugin["builtin"] is True
    assert plugin["tool_count"] == 1
    assert repo.list_plugins() == [plugin]
    assert repo.list_tools()[0]["name"] == "echo"
    assert json.loads(repo.list_tools()[0]["input_schema_json"])["required"] == ["message"]


def test_plugin_repository_upgrade_replaces_tools_in_one_transaction(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    repo.upsert_plugin(_manifest(), enabled=True)

    repo.upsert_plugin(
        _manifest(
            "0.2.0",
            tools=[
                {
                    "name": "reverse",
                    "summary": "Reverse a message",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                    "output_schema": {"type": "object", "properties": {}, "required": []},
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_reverse",
                }
            ],
        ),
        enabled=False,
    )

    assert repo.get_plugin("_sample")["version"] == "0.2.0"
    assert repo.get_plugin("_sample")["enabled"] is False
    assert [tool["name"] for tool in repo.list_tools()] == ["reverse"]


def test_plugin_repository_delete_cascades_tools(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    repo.upsert_plugin(_manifest(), enabled=True)

    repo.delete_plugin("_sample")

    assert repo.get_plugin("_sample") is None
    assert repo.list_tools() == []


def test_plugin_repository_set_enabled_missing_raises_plugin_error(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)

    with pytest.raises(PluginNotFoundError):
        repo.set_enabled("missing", True)


def test_plugin_repository_rolls_back_register_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.upsert_plugin_with_audit(
            _manifest(),
            enabled=True,
            audit={
                "kind": "plugin.register",
                "target_ref": "_sample",
                "outcome": "succeeded",
                "detail": {"version": "0.1.0"},
            },
        )

    assert repo.get_plugin("_sample") is None
    assert repo.list_tools() == []


def test_plugin_repository_rolls_back_enabled_change_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    repo.upsert_plugin(_manifest(), enabled=True)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.set_enabled_with_audit(
            "_sample",
            False,
            audit={
                "kind": "plugin.disable",
                "target_ref": "_sample",
                "outcome": "succeeded",
                "detail": {"version": "0.1.0", "enabled": False},
            },
        )

    assert repo.get_plugin("_sample")["enabled"] is True


def test_plugin_repository_rolls_back_delete_when_audit_write_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    repo.upsert_plugin(_manifest(), enabled=True)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.delete_plugin_with_audit(
            "_sample",
            audit={
                "kind": "plugin.remove",
                "target_ref": "_sample",
                "outcome": "succeeded",
                "detail": {"version": "0.1.0"},
            },
        )

    assert repo.get_plugin("_sample") is not None
    assert repo.list_tools()[0]["name"] == "echo"


def test_plugin_repository_writes_audit_records(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)

    repo.write_audit(
        kind="tool.invoke",
        target_ref="_sample.echo",
        actor="system",
        inputs_hash="abc123",
        outcome="succeeded",
        detail={"duration_ms": 4},
    )

    audits = repo.list_audit(kind="tool.invoke")
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "_sample.echo"
    assert audits[0]["inputs_hash"] == "abc123"
    assert audits[0]["detail"] == {"duration_ms": 4}


def test_init_db_creates_plugin_tables_with_foreign_keys(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"plugins", "tools", "audit"} <= table_names

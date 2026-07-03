import json
import zipfile
from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.plugins.errors import DuplicatePluginError, ManifestError, PluginError
from marvis.plugins.loader import (
    compute_checksum,
    install_plugin,
    load_builtin_packs,
    load_manifest,
)
from marvis.plugins.registry import PluginRegistry


def _manifest(
    name: str = "sample_pack",
    *,
    version: str = "0.1.0",
    schema: dict | None = None,
    permissions: tuple[str, ...] = (),
    side_effects: tuple[str, ...] = (),
):
    object_schema = schema or {"type": "object", "properties": {}, "required": []}
    return {
        "name": name,
        "version": version,
        "display_name": "Sample Pack",
        "description": "Loader test pack",
        "module": f"{name}.tools",
        "tools": [
            {
                "name": "echo",
                "summary": "Echo a message",
                "input_schema": object_schema,
                "output_schema": object_schema,
                "determinism": "deterministic",
                "timeout_seconds": 10,
                "failure_policy": "fail",
                "side_effects": list(side_effects),
                "entrypoint": "tool_echo",
            }
        ],
        "hooks": [],
        "permissions": list(permissions),
        "checksum": "uploaded-value-is-ignored",
    }


def _write_pack(
    root: Path,
    name: str = "sample_pack",
    *,
    version: str = "0.1.0",
    schema: dict | None = None,
    permissions: tuple[str, ...] = (),
    side_effects: tuple[str, ...] = (),
    tool_body: str = "def tool_echo(inputs, ctx):\n    return {}\n",
) -> Path:
    pack_dir = root / name
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.json").write_text(
        json.dumps(
            _manifest(
                name,
                version=version,
                schema=schema,
                permissions=permissions,
                side_effects=side_effects,
            )
        ),
        encoding="utf-8",
    )
    (pack_dir / "tools.py").write_text(tool_body, encoding="utf-8")
    return pack_dir


def _zip_pack(pack_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(pack_dir.rglob("*")):
            archive.write(path, path.relative_to(pack_dir.parent).as_posix())


def test_load_manifest_reads_and_validates_manifest_file(tmp_path):
    pack_dir = _write_pack(tmp_path)

    manifest = load_manifest(pack_dir, builtin=True)

    assert manifest.name == "sample_pack"
    assert manifest.builtin is True
    assert manifest.checksum == ""


def test_load_manifest_accepts_explicit_process_spawn_permission(tmp_path):
    pack_dir = _write_pack(
        tmp_path,
        permissions=("process:spawn",),
        side_effects=("process:spawn",),
    )

    manifest = load_manifest(pack_dir, builtin=False)

    assert manifest.permissions == ("process:spawn",)
    assert manifest.tools[0].side_effects == ("process:spawn",)


def test_load_builtin_packs_registers_packs_idempotently(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    packs_root = tmp_path / "packs"
    _write_pack(packs_root)

    load_builtin_packs(registry, packs_root)
    load_builtin_packs(registry, packs_root)

    assert [plugin.name for plugin in registry.list()] == ["sample_pack"]


def test_install_plugin_zip_registers_with_platform_checksum(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    source_pack = _write_pack(tmp_path / "source")
    zip_path = tmp_path / "sample_pack.zip"
    _zip_pack(source_pack, zip_path)

    manifest = install_plugin(zip_path, tmp_path / "installed", registry)

    assert manifest.name == "sample_pack"
    assert manifest.builtin is False
    installed_dir = tmp_path / "installed" / "sample_pack"
    assert manifest.checksum == compute_checksum(installed_dir)
    assert registry.get("sample_pack").checksum == compute_checksum(installed_dir)
    assert not (tmp_path / "installed" / ".staging").exists()


def test_install_plugin_rejects_duplicate_name_and_version(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    source_pack = _write_pack(tmp_path / "source")
    zip_path = tmp_path / "sample_pack.zip"
    _zip_pack(source_pack, zip_path)

    install_plugin(zip_path, tmp_path / "installed", registry)

    with pytest.raises(DuplicatePluginError):
        install_plugin(zip_path, tmp_path / "installed", registry)


def test_loader_rejects_invalid_json_schema(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    source_pack = _write_pack(
        tmp_path / "source",
        schema={"type": "not-a-json-schema-type"},
    )
    zip_path = tmp_path / "bad_schema.zip"
    _zip_pack(source_pack, zip_path)

    with pytest.raises(PluginError, match="invalid json schema"):
        install_plugin(zip_path, tmp_path / "installed", registry)


def test_install_plugin_rejects_zip_path_traversal(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../manifest.json", json.dumps(_manifest()))

    with pytest.raises(PluginError, match="unsafe zip path"):
        install_plugin(zip_path, tmp_path / "installed", registry)


def test_install_plugin_rejects_invalid_zip_file(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    zip_path = tmp_path / "not-a-zip.zip"
    zip_path.write_bytes(b"not a zip")

    with pytest.raises(PluginError, match="invalid plugin zip"):
        install_plugin(zip_path, tmp_path / "installed", registry)


def test_install_plugin_rolls_back_existing_files_when_register_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = PluginRegistry(PluginRepository(db_path))
    installed_root = tmp_path / "installed"
    source_v1 = _write_pack(
        tmp_path / "source-v1",
        tool_body="def tool_echo(inputs, ctx):\n    return {'version': 1}\n",
    )
    zip_v1 = tmp_path / "sample_pack-v1.zip"
    _zip_pack(source_v1, zip_v1)
    install_plugin(zip_v1, installed_root, registry)
    installed_tool = installed_root / "sample_pack" / "tools.py"
    original_content = installed_tool.read_text(encoding="utf-8")

    source_v2 = _write_pack(
        tmp_path / "source-v2",
        version="0.2.0",
        tool_body="def tool_echo(inputs, ctx):\n    return {'version': 2}\n",
    )
    zip_v2 = tmp_path / "sample_pack-v2.zip"
    _zip_pack(source_v2, zip_v2)

    def fail_register(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(registry, "register", fail_register)

    with pytest.raises(RuntimeError, match="db unavailable"):
        install_plugin(zip_v2, installed_root, registry)

    assert installed_tool.read_text(encoding="utf-8") == original_content
    assert not (installed_root / ".staging").exists()
    assert not any(path.name.endswith(".bak") for path in installed_root.iterdir())


def test_load_manifest_missing_file_raises_manifest_error(tmp_path):
    with pytest.raises(ManifestError, match="manifest.json"):
        load_manifest(tmp_path, builtin=True)

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import uuid
import zipfile

from jsonschema import Draft202012Validator

from marvis.plugins.errors import DuplicatePluginError, ManifestError, PluginError
from marvis.plugins.manifest import PluginManifest, parse_manifest
from marvis.plugins.registry import PluginRegistry


MAX_PLUGIN_ZIP_BYTES = 50 * 1024 * 1024


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            _update_file_digest(digest, child)
        return digest.hexdigest()
    _update_file_digest(digest, path)
    return digest.hexdigest()


def _update_file_digest(digest, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def load_manifest(plugin_dir: Path, *, builtin: bool) -> PluginManifest:
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ManifestError(f"manifest.json not found in {plugin_dir}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest.json: {exc}") from exc
    manifest = parse_manifest(data, builtin=builtin)
    _assert_manifest_schemas(manifest)
    return manifest


def install_plugin(
    zip_path: Path,
    plugins_dir: Path,
    registry: PluginRegistry,
) -> PluginManifest:
    if not zip_path.is_file():
        raise PluginError(f"plugin zip not found: {zip_path}")
    if zip_path.stat().st_size > MAX_PLUGIN_ZIP_BYTES:
        raise PluginError("plugin zip is too large")

    with tempfile.TemporaryDirectory(prefix="marvis-plugin-") as temp_name:
        temp_dir = Path(temp_name)
        _safe_extract_zip(zip_path, temp_dir)
        unpacked_dir = _plugin_root_from_extract(temp_dir)
        checksum = compute_checksum(unpacked_dir)
        manifest = replace(load_manifest(unpacked_dir, builtin=False), checksum=checksum)
        _raise_if_duplicate_same_version(registry, manifest)

        plugins_dir.mkdir(parents=True, exist_ok=True)
        destination = plugins_dir / manifest.name
        backup = None
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.name}-staging-",
            dir=plugins_dir,
        ) as staging_parent_name:
            staging_parent = Path(staging_parent_name)
            staged = staging_parent / manifest.name
            shutil.copytree(unpacked_dir, staged)
            try:
                if destination.exists():
                    backup = plugins_dir / f".{manifest.name}-backup-{uuid.uuid4().hex}"
                    destination.rename(backup)
                staged.rename(destination)
                registry.register(manifest, enabled=True)
            except Exception:
                _remove_path(destination)
                if backup is not None and backup.exists():
                    backup.rename(destination)
                raise
            else:
                if backup is not None and backup.exists():
                    _remove_path(backup)
        return manifest


def load_builtin_packs(registry: PluginRegistry, packs_root: Path) -> None:
    if not packs_root.exists():
        return
    for plugin_dir in sorted(path for path in packs_root.iterdir() if path.is_dir()):
        if not (plugin_dir / "manifest.json").is_file():
            continue
        manifest = load_manifest(plugin_dir, builtin=True)
        try:
            registry.register(manifest, enabled=True)
        except DuplicatePluginError:
            # Builtin discovery runs at app startup and should be idempotent.
            continue


def _assert_manifest_schemas(manifest: PluginManifest) -> None:
    for tool in manifest.tools:
        _assert_valid_jsonschema(tool.input_schema)
        _assert_valid_jsonschema(tool.output_schema)


def _assert_valid_jsonschema(schema: dict) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise PluginError(f"invalid json schema: {exc}") from exc


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                _assert_safe_zip_member(info.filename)
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise PluginError("invalid plugin zip") from exc


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _assert_safe_zip_member(filename: str) -> None:
    path = PurePosixPath(filename)
    if path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise PluginError(f"unsafe zip path: {filename}")


def _plugin_root_from_extract(extract_dir: Path) -> Path:
    if (extract_dir / "manifest.json").is_file():
        return extract_dir
    candidates = [
        path for path in extract_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ]
    if len(candidates) != 1:
        raise ManifestError("plugin archive must contain exactly one manifest.json")
    return candidates[0]


def _raise_if_duplicate_same_version(
    registry: PluginRegistry,
    manifest: PluginManifest,
) -> None:
    try:
        existing = registry.get(manifest.name)
    except PluginError:
        return
    if existing.version == manifest.version:
        raise DuplicatePluginError(f"{manifest.name}@{manifest.version} already registered")

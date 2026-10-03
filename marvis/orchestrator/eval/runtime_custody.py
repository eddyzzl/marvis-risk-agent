"""Opt-in private originals for later review, never an acceptance authority.

The application and its workers must have stopped before capture. Native signed
records retain their original bytes; their workspace key is deliberately omitted.
Manifest integrity requires a digest retained independently of this directory.
No restore, receipt re-signing or interpretation of saved ``passed`` flags occurs.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import tempfile

from .runtime_contracts import digest


class CustodyError(ValueError):
    """A bounded error code, safe to include in public execution metadata."""


MAX_FILES = 50_000
MAX_BYTES = 20 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
_WORKSPACE_DIRS = (
    "tasks", "datasets", "plugins", "report_templates", "branding",
    "material_uploads", "source", "reference_decision", "historical_event_private",
    "operations", "cache",
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


def _safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise CustodyError("custody_path_alias_rejected")
    return path


def _private_dir(path: Path, *, create: bool = False) -> Path:
    path = _safe_path(path)
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise CustodyError("custody_directory_must_be_private")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise CustodyError("custody_directory_owner_mismatch")
    return path


def prepare_custody_run(root: Path, run_id: str, *, forbidden: tuple[Path, ...]) -> Path:
    """Use an explicit private location disjoint from public outputs and inputs."""
    root = _safe_path(root)
    for path in forbidden:
        other = path.resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise CustodyError("custody_location_must_be_separate")
    if not _ID.fullmatch(run_id):
        raise CustodyError("custody_invalid_run_id")
    _private_dir(root, create=True)
    run = root / run_id
    run.mkdir(mode=0o700, exist_ok=False)
    return run


def _files(root: Path):
    if root.is_symlink():
        raise CustodyError("custody_source_link_rejected")
    for ordinal, entry in enumerate(root.rglob("*"), 1):
        if ordinal > MAX_FILES * 4:
            raise CustodyError("custody_inventory_limit")
        info = entry.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CustodyError("custody_source_link_or_special_file_rejected")
        if entry.resolve() != entry:
            raise CustodyError("custody_source_escape_rejected")
        yield entry


def _read_secret(path: Path) -> bytes:
    if not path.exists():
        return b""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65_536:
        raise CustodyError("custody_secret_source_invalid")
    return path.read_bytes().strip()


def _known_secrets(workspace: Path, supplied: tuple[str, ...]) -> tuple[bytes, ...]:
    secrets = [s.encode() for s in supplied if s]
    secrets.append(_read_secret(workspace / "plugin_admin_token"))
    settings = workspace / "settings"
    if settings.is_symlink():
        raise CustodyError("custody_secret_source_invalid")
    llm = _read_secret(settings / "llm.json")
    if llm:
        try:
            models = json.loads(llm)["models"]
            secrets.extend(m["api_key"].encode() for m in models if m.get("api_key"))
        except (ValueError, TypeError, KeyError, AttributeError):
            raise CustodyError("custody_model_settings_invalid") from None
    private = workspace / "source_private"
    if private.exists():
        secrets.extend(_read_secret(p) for p in _files(private))
    return tuple(set(s for s in secrets if s))


def _file_identity(path: Path, secrets: tuple[bytes, ...] = ()) -> dict:
    hasher = hashlib.sha256()
    size = 0
    overlap = max((len(s) for s in secrets), default=1) - 1
    tail = b""
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            if size > MAX_BYTES:
                raise CustodyError("custody_size_limit")
            if any(secret in tail + block for secret in secrets):
                raise CustodyError("custody_secret_found_in_originals")
            hasher.update(block)
            tail = (tail + block)[-overlap:] if overlap else b""
    return {"sha256": hasher.hexdigest(), "size_bytes": size}


def _json_new(path: Path, value, secrets: tuple[bytes, ...] = ()) -> None:
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    if any(secret in raw for secret in secrets):
        raise CustodyError("custody_secret_found_in_originals")
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_bounded(source: Path, destination: Path, expected_size: int) -> None:
    if expected_size > MAX_BYTES:
        raise CustodyError("custody_size_limit")
    with source.open("rb") as src, destination.open("xb") as dst:
        destination.chmod(0o600)
        copied_bytes = 0
        while block := src.read(1024 * 1024):
            copied_bytes += len(block)
            if copied_bytes > expected_size:
                raise CustodyError("custody_source_changed")
            dst.write(block)
        if copied_bytes != expected_size:
            raise CustodyError("custody_source_changed")
        dst.flush()
        os.fsync(dst.fileno())


def _sqlite_snapshot(source: Path, destination: Path, secrets: tuple[bytes, ...]) -> None:
    if source.is_symlink() or source.stat().st_nlink != 1:
        raise CustodyError("custody_database_alias_rejected")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.is_symlink() or (sidecar.exists() and sidecar.stat().st_nlink != 1):
            raise CustodyError("custody_database_alias_rejected")
        if sidecar.exists():
            _file_identity(sidecar, secrets)
    _file_identity(source, secrets)
    # Even a read-only SQLite connection can checkpoint/remove WAL sidecars on
    # close when the directory is writable. Work on a private physical snapshot
    # so opening SQLite cannot mutate the original post-stop evidence window.
    with tempfile.TemporaryDirectory(prefix=".sqlite-snapshot-", dir=destination.parent) as tmp:
        copied = Path(tmp) / source.name
        total = 0
        for suffix in ("", "-wal", "-journal"):
            original = source.with_name(source.name + suffix)
            if original.exists():
                target = copied.with_name(copied.name + suffix)
                size = original.stat().st_size
                total += size
                if total > MAX_BYTES:
                    raise CustodyError("custody_size_limit")
                _copy_bounded(original, target, size)
        with closing(sqlite3.connect(copied)) as src:
            _reject_database_credentials(src)
            with closing(sqlite3.connect(destination)) as dst:
                destination.chmod(0o600)
                src.backup(dst)
                dst.execute("PRAGMA journal_mode=DELETE")
                if dst.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise CustodyError("custody_database_integrity_failed")


def _reject_database_credentials(conn) -> None:
    # Current runtime sessions store one-way hashes, not bearer credentials.
    # Reject future populated credential columns rather than silently redacting
    # signed/native records or pretending their original bytes were retained.
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, condition in (
        ("approval_records", "status IN ('issued', 'reserved')"),
        ("validation_batch_material_uploads", "1=1"),
        ("operations_periods", "lease_token IS NOT NULL"),
        ("operations_notification_outbox", "lease_token IS NOT NULL"),
    ):
        if table in tables and conn.execute(f"SELECT 1 FROM {table} WHERE {condition} LIMIT 1").fetchone():
            # Do not change native audit records or copy an unconsumed capability.
            # Expired-but-not-retired records conservatively fail as well.
            raise CustodyError("custody_unretired_capability_present")
    for table in tables:
        quoted_table = '"' + table.replace('"', '""') + '"'
        for column in conn.execute(f"PRAGMA table_info({quoted_table})"):
            name = column[1]
            if re.search(r"(?:api_?key|password|secret|credential|(?:private|signing|encryption|hmac)_?key|access_token|refresh_token|session_token)$", name, re.I):
                quoted_column = '"' + name.replace('"', '""') + '"'
                if conn.execute(
                    f"SELECT 1 FROM {quoted_table} WHERE {quoted_column} IS NOT NULL AND {quoted_column} != '' LIMIT 1"
                ).fetchone():
                    raise CustodyError("custody_database_credential_column_populated")


def _workspace_window(workspace: Path) -> dict:
    files = []
    for name in _WORKSPACE_DIRS:
        root = workspace / name
        if root.exists() or root.is_symlink():
            files.extend(_files(root))
    for name in ("marvis.sqlite", "marvis.sqlite-wal", "marvis.sqlite-journal"):
        file = workspace / name
        if file.exists():
            files.append(file)
    return {str(p): (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in files}


def capture_case(
    *, run_dir: Path, workspace: Path, dataset_root: Path, case,
    private: dict, bindings: dict, owned_processes_stopped: bool,
    forbidden_secrets: tuple[str, ...] = (),
) -> dict:
    """Atomically publish originals. A failed capture publishes no case bundle."""
    if not owned_processes_stopped:
        raise CustodyError("custody_processes_not_stopped")
    run_dir = _private_dir(run_dir)
    workspace = workspace.resolve()
    if not _ID.fullmatch(case.id):
        raise CustodyError("custody_invalid_case_id")
    target = run_dir / case.id
    if target.exists() or target.is_symlink():
        raise CustodyError("custody_case_collision")
    secrets = _known_secrets(workspace, forbidden_secrets)
    original_window = _workspace_window(workspace)
    staged = Path(tempfile.mkdtemp(prefix=".pending-", dir=run_dir))
    try:
        staged_workspace = staged / "workspace"
        staged_workspace.mkdir(mode=0o700)
        count = total = 0

        def copy(source: Path, destination: Path):
            nonlocal count, total
            info = source.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise CustodyError("custody_source_link_or_special_file_rejected")
            count += 1
            total += info.st_size
            if count > MAX_FILES or total > MAX_BYTES:
                raise CustodyError("custody_inventory_limit")
            identity = _file_identity(source, secrets)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _copy_bounded(source, destination, info.st_size)
            if _file_identity(destination, secrets) != identity:
                raise CustodyError("custody_source_changed")

        for name in _WORKSPACE_DIRS:
            origin = workspace / name
            if origin.exists() or origin.is_symlink():
                for source in _files(origin):
                    copy(source, staged_workspace / source.relative_to(workspace))
        database = workspace / "marvis.sqlite"
        if not database.is_file():
            raise CustodyError("custody_database_missing")
        _sqlite_snapshot(database, staged_workspace / database.name, secrets)
        material_mapping = []
        for ordinal, material in enumerate(case.materials):
            source = _safe_path(dataset_root / material.path)
            if not source.is_relative_to(dataset_root.resolve()):
                raise CustodyError("custody_material_escape_rejected")
            destination = staged / "inputs" / f"{ordinal:03d}{source.suffix}"
            copy(source, destination)
            if _file_identity(destination)["sha256"] != material.sha256:
                raise CustodyError("custody_material_changed")
            material_mapping.append({"original_relative_path": material.path,
                                     "archive_path": destination.relative_to(staged).as_posix(),
                                     "sha256": material.sha256, "role": material.role})
        _json_new(staged / "case.json", case.model_dump(), secrets)
        _json_new(staged / "private.json", private, secrets)
        _json_new(staged / "bindings.json", bindings, secrets)
        if original_window != _workspace_window(workspace):
            raise CustodyError("custody_workspace_changed_during_capture")
        inventory = {}
        for path in _files(staged):
            path.chmod(0o600)
            inventory[path.relative_to(staged).as_posix()] = _file_identity(path, secrets)
        if len(inventory) > MAX_FILES or sum(i["size_bytes"] for i in inventory.values()) > MAX_BYTES:
            raise CustodyError("custody_inventory_limit")
        # mkdir(parents=True) uses the umask for intermediate directories.
        for path in staged.rglob("*"):
            if path.is_dir():
                path.chmod(0o700)
        manifest = {
            "schema": "marvis.runtime-private-originals.v1", "case_id": case.id,
            "bindings_sha256": inventory["bindings.json"]["sha256"],
            "original_workspace": str(workspace), "files": inventory,
            "materials": material_mapping,
            "database_capture": "sqlite_committed_snapshot_after_owned_processes_stopped",
            "database_byte_identity": "logical_committed_snapshot_not_physical_database_copy",
            "operational_bindings": "retired_native_identifiers_preserved_unretired_capabilities_rejected",
            "excluded": ["settings", "plugin_admin_token", "source_private", "runtime_config", "server_log"],
            "workspace_directories": list(_WORKSPACE_DIRS),
            "omitted_workspace_entries": sorted(p.name for p in workspace.iterdir()
                if p.name not in (*_WORKSPACE_DIRS, "marvis.sqlite", "marvis.sqlite-wal", "marvis.sqlite-shm")),
            "native_key_dependent_reverification": "unavailable_original_key_not_exported",
            "absolute_native_paths": "preserved_no_rebinding_or_resigning",
            "acceptance_claim": "not_established",
        }
        _json_new(staged / "manifest.json", manifest, secrets)
        manifest_identity = _file_identity(staged / "manifest.json", secrets)
        if manifest_identity["size_bytes"] > MAX_MANIFEST_BYTES:
            raise CustodyError("custody_manifest_limit")
        # The run directory is private and newly owned by this run. Never replace
        # an existing case, even an empty directory, during publication.
        if target.exists() or target.is_symlink():
            raise CustodyError("custody_case_collision")
        staged.rename(target)
        directory_fd = os.open(run_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {"status": "retained", "manifest_sha256": manifest_identity["sha256"],
                "native_key_dependent_reverification": "unavailable_original_key_not_exported",
                "acceptance_claim": "not_established"}
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def verify_case_archive(path: Path, *, expected_manifest_sha256: str) -> dict:
    """Check original bytes against a caller-held digest; does not score success."""
    path = _private_dir(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise CustodyError("custody_expected_digest_required")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise CustodyError("custody_manifest_invalid")
    raw = manifest_path.read_bytes()
    if digest(raw) != expected_manifest_sha256:
        raise CustodyError("custody_manifest_changed")
    manifest = json.loads(raw)
    inventory = manifest.get("files")
    if manifest.get("schema") != "marvis.runtime-private-originals.v1" or not isinstance(inventory, dict):
        raise CustodyError("custody_manifest_invalid")
    if len(inventory) > MAX_FILES:
        raise CustodyError("custody_inventory_limit")
    declared_total = 0
    for relative, entry in inventory.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or "\\" in relative or relative == "manifest.json"
                or not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}
                or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0):
            raise CustodyError("custody_manifest_invalid")
        declared_total += entry["size_bytes"]
        if declared_total > MAX_BYTES:
            raise CustodyError("custody_inventory_limit")
    actual = {}
    for ordinal, directory in enumerate(path.rglob("*"), 1):
        if ordinal > MAX_FILES * 4:
            raise CustodyError("custody_inventory_limit")
        if directory.is_dir():
            _private_dir(directory)
    actual_total = 0
    for file in _files(path):
        if stat.S_IMODE(file.stat().st_mode) != 0o600:
            raise CustodyError("custody_file_permissions_changed")
        if file != manifest_path:
            relative = file.relative_to(path).as_posix()
            size = file.stat().st_size
            actual_total += size
            if len(actual) >= MAX_FILES or actual_total > MAX_BYTES:
                raise CustodyError("custody_inventory_limit")
            if relative not in inventory or size != inventory[relative]["size_bytes"]:
                raise CustodyError("custody_originals_changed")
            actual[relative] = _file_identity(file)
    if actual != inventory:
        raise CustodyError("custody_originals_changed")
    if len(actual) > MAX_FILES or sum(i["size_bytes"] for i in actual.values()) > MAX_BYTES:
        raise CustodyError("custody_inventory_limit")
    database = path / "workspace" / "marvis.sqlite"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise CustodyError("custody_database_integrity_failed")
    return {"archived_bytes_match_manifest": True, "case_id": manifest["case_id"],
            "acceptance_claim": "not_established",
            "native_key_dependent_reverification": "unavailable_original_key_not_exported"}

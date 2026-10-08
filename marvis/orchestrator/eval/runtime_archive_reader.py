"""Bounded, read-only access to privately retained runtime originals.

Readers copy authenticated bytes before parsing, never reopen old native paths,
and never import code or restore credentials from an archive.
"""
from pathlib import Path
import json
import re

from .runtime_contracts import digest
from .runtime_custody import verify_case_archive

_MAX_BYTES = 128 * 1024**2
_MAX_SNAPSHOT_BYTES = 1024**3

class _Unsupported(ValueError):
    pass


def _file(archive, relative, *, maximum=_MAX_BYTES):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("archive path boundary")
    path = archive / relative
    if path.resolve() != path or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("archive file boundary")
    return path


def _json(archive, relative):
    return json.loads(_file(archive, relative, maximum=16 * 1024**2).read_bytes())


def _authenticated_snapshot(archive, destination, expected_digest):
    from .runtime_custody import _copy_bounded

    raw = _file(archive, "manifest.json", maximum=16 * 1024**2).read_bytes()
    if digest(raw) != expected_digest:
        raise ValueError("manifest changed before snapshot")
    inventory = json.loads(raw)["files"]
    if sum(item["size_bytes"] for item in inventory.values()) > _MAX_SNAPSHOT_BYTES:
        raise _Unsupported("private recomputation snapshot exceeds one GiB")
    # The independently held manifest authenticates this copy before any domain
    # parser sees it. Replacing the original afterward cannot change its bytes.
    for relative, item in inventory.items():
        target = destination / relative
        for parent in reversed(target.parents):
            if parent.is_relative_to(destination):
                parent.mkdir(mode=0o700, exist_ok=True)
        _copy_bounded(_file(archive, relative, maximum=_MAX_SNAPSHOT_BYTES), target, item["size_bytes"])
    manifest = destination / "manifest.json"
    manifest.touch(mode=0o600, exist_ok=False)
    manifest.write_bytes(raw)
    verify_case_archive(destination, expected_manifest_sha256=expected_digest)


def _public_source_identity(source):
    # A separately supplied binding may contain accidental credentials or
    # institution metadata. Compare its complete contents internally, but never
    # echo arbitrary extensions in a review result (including mismatch results).
    patterns = {"commit": r"[0-9a-f]{40}", "source_sha256": r"[0-9a-f]{64}",
                "dirty_diff_sha256": r"[0-9a-f]{64}"}
    return {key: source[key] for key, pattern in patterns.items()
            if isinstance(source.get(key), str) and re.fullmatch(pattern, source[key])}


def _original_path(archive, original_root, absolute):
    root, path = Path(original_root), Path(absolute)
    if not root.is_absolute() or not path.is_absolute() or ".." in path.parts:
        raise ValueError("original path identity invalid")
    relative = path.relative_to(root)
    # This maps immutable identities to retained bytes; it does not rewrite DB,
    # signatures or the old directory, and never opens the former absolute path.
    return _file(archive, Path("workspace") / relative)

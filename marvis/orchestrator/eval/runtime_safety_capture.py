"""Opt-in original committed database state before a future runtime case starts.

This is a SQLite logical snapshot, not a copied live database or an invented
historical baseline. It never exports credentials, keys, sessions or callbacks.
"""
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import time

from .runtime_contracts import digest
from .runtime_custody import CustodyError, _file_identity, _json_new, _known_secrets, _private_dir, _reject_database_credentials, _safe_path

MAX_BEFORE_BYTES = 256 * 1024**2


def capture_runtime_before(destination: Path, workspace: Path, case, *, deadline: float, origin_ns: int):
    """Call only after app startup and before the case's first HTTP action."""
    destination = _safe_path(destination)
    workspace = _safe_path(workspace)
    if destination.exists() or destination.is_relative_to(workspace) or workspace.is_relative_to(destination):
        raise CustodyError("safety_before_location_or_collision")
    secrets = _known_secrets(workspace, ())
    source = _safe_path(workspace / "marvis.sqlite")
    if not source.is_file() or source.stat().st_size > MAX_BEFORE_BYTES:
        raise CustodyError("safety_before_database_missing_or_oversized")
    started_ns, started_at = time.monotonic_ns(), time.time()
    if started_ns < origin_ns or time.monotonic() >= deadline:
        raise CustodyError("safety_before_capture_budget")
    _private_dir(destination, create=True)
    database = destination / "marvis.sqlite"
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=1)) as src, closing(sqlite3.connect(database)) as dst:
            database.chmod(0o600)
            page_size = src.execute("PRAGMA page_size").fetchone()[0]

            def progress(_status, _remaining, total):
                if time.monotonic() >= deadline or total * page_size > MAX_BEFORE_BYTES:
                    raise CustodyError("safety_before_capture_budget")

            src.backup(dst, pages=256, progress=progress, sleep=0.005)
            dst.execute("PRAGMA journal_mode=DELETE")
            if dst.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise CustodyError("safety_before_database_integrity")
            _reject_database_credentials(dst)
        identity = _file_identity(database, secrets)
        if identity["size_bytes"] > MAX_BEFORE_BYTES:
            raise CustodyError("safety_before_capture_budget")
        with database.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {"schema": "marvis.runtime-before-originals.v1", "case_sha256": digest(case.model_dump()),
                    "phase": "after_application_startup_before_first_case_http_action", "first_case_http_ordinal": 0,
                    "capture_started_at": started_at, "capture_finished_at": time.time(),
                    "capture_started_elapsed_ns": started_ns - origin_ns, "capture_finished_elapsed_ns": time.monotonic_ns() - origin_ns,
                    "database_capture": "sqlite_committed_logical_snapshot", "files": {"marvis.sqlite": identity},
                    "authority": "no_keys_or_authority_statement_exported", "acceptance_claim": "not_established"}
        _json_new(destination / "manifest.json", manifest, secrets)
        directory_fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {"status": "captured", "manifest_sha256": digest((destination / "manifest.json").read_bytes()),
                "database_sha256": identity["sha256"], "scope": manifest["database_capture"]}
    except BaseException:
        # Partial capture stays private and explicitly incomplete. It is never
        # republished as a historical before state or silently retried.
        raise

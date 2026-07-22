from __future__ import annotations

from collections.abc import Iterator
import hashlib
import hmac
import json
import mimetypes
import os
from pathlib import Path
import stat
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from urllib.parse import quote, unquote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.background import BackgroundTask

from marvis.api_task_helpers import get_task_or_404
from marvis.db import StrategyRepository, TaskRepository, connect
from marvis.errors import conflict, not_found
from marvis.files import sha256_file

from marvis.output.word_preview import docx_to_html_preview
from marvis.repositories.task_artifacts import TaskArtifactRepository

router = APIRouter(prefix="/api", tags=["artifacts"])

_SNAPSHOT_CHUNK_BYTES = 64 * 1024
_SNAPSHOT_MEMORY_LIMIT_BYTES = 1024 * 1024

_STRATEGY_ARTIFACT_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".py": "text/x-python",
    ".sql": "application/sql",
    ".svg": "image/svg+xml",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@router.get("/tasks/{task_id}/strategy-artifacts")
def list_strategy_artifacts(task_id: str, request: Request) -> dict:
    """List safe strategy artifacts without exposing their stored paths.

    Strategy tools historically persisted absolute paths.  This endpoint is the
    adapter boundary that turns those rows into task-owned artifact ids; callers
    never receive a filesystem path and can download only through the guarded
    id route below.
    """

    settings = request.app.state.settings
    get_task_or_404(TaskRepository(settings.db_path), task_id)
    rows = StrategyRepository(settings.db_path).list_strategy_artifacts_for_task(
        task_id
    )
    registry_records = _registered_artifact_records(settings=settings)
    integrity_failures: dict[Path, str | None] = {}
    artifacts = []
    for row in rows:
        path = _available_task_artifact_path(
            settings=settings,
            task_id=task_id,
            stored_path=row.get("path"),
        )
        if path is not None:
            if path not in integrity_failures:
                integrity_failures[path] = _artifact_path_integrity_failure(
                    settings=settings,
                    task_id=task_id,
                    candidate=path,
                    records=registry_records,
                )
            if integrity_failures[path] is not None:
                path = None
        artifact_id = str(row.get("id") or "")
        filename = _artifact_filename(row.get("path"))
        status = str(row.get("asset_status") or row.get("status") or "draft")
        if status == "adopted":
            status = "adopted_local"
        available = path is not None and bool(artifact_id)
        artifacts.append(
            {
                "id": artifact_id,
                "kind": str(row.get("kind") or ""),
                "filename": filename,
                "strategy_id": str(row.get("strategy_id") or ""),
                "strategy_type": str(row.get("strategy_type") or ""),
                "version": row.get("version"),
                "asset_status": status,
                "created_at": str(row.get("created_at") or ""),
                "available": available,
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}/strategy-artifacts/"
                    f"{quote(artifact_id, safe='')}/download"
                    if available
                    else None
                ),
            }
        )
    return {"task_id": task_id, "artifacts": artifacts}


@router.get("/tasks/{task_id}/strategy-artifacts/{artifact_id}/download")
def download_strategy_artifact(
    task_id: str,
    artifact_id: str,
    request: Request,
) -> StreamingResponse:
    settings = request.app.state.settings
    get_task_or_404(TaskRepository(settings.db_path), task_id)
    row = StrategyRepository(settings.db_path).get_strategy_artifact_for_task(
        task_id,
        artifact_id,
    )
    if row is None:
        raise not_found("strategy artifact not found")
    path = _available_task_artifact_path(
        settings=settings,
        task_id=task_id,
        stored_path=row.get("path"),
    )
    if path is None:
        raise not_found("strategy artifact not found")
    try:
        snapshot, content_length = _verified_artifact_snapshot(
            settings=settings,
            task_id=task_id,
            candidate=path,
            required_content_hash=row.get("content_hash"),
            required_content_size=row.get("content_size"),
        )
    except _ArtifactSnapshotFailure as exc:
        raise conflict("strategy artifact integrity check failed") from exc
    return _attachment_response(
        snapshot,
        filename=path.name,
        media_type=_STRATEGY_ARTIFACT_MEDIA_TYPES[path.suffix.lower()],
        content_length=content_length,
    )


@router.get("/tasks/{task_id}/task-artifacts")
def list_task_artifacts(task_id: str, request: Request) -> dict:
    settings = request.app.state.settings
    get_task_or_404(TaskRepository(settings.db_path), task_id)
    rows = TaskArtifactRepository(settings.db_path).list_for_task(task_id)
    registry_records = _registered_artifact_records(settings=settings)
    integrity_failures: dict[Path, str | None] = {}
    artifacts = []
    for row in rows:
        path = _available_task_artifact_path(
            settings=settings,
            task_id=task_id,
            stored_path=row.get("path"),
        )
        if path is not None:
            if path not in integrity_failures:
                integrity_failures[path] = _artifact_path_integrity_failure(
                    settings=settings,
                    task_id=task_id,
                    candidate=path,
                    records=registry_records,
                )
            if integrity_failures[path] is not None:
                path = None
        artifact_id = str(row.get("id") or "")
        available = path is not None and bool(artifact_id)
        artifacts.append(
            {
                "id": artifact_id,
                "kind": str(row.get("kind") or ""),
                "filename": _artifact_filename(row.get("path")),
                "origin_tool": str(row.get("origin_tool") or ""),
                "content_hash": str(row.get("content_hash") or ""),
                "created_at": str(row.get("created_at") or ""),
                "available": available,
                "download_url": (
                    f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
                    f"{quote(artifact_id, safe='')}/download"
                    if available
                    else None
                ),
            }
        )
    return {"task_id": task_id, "artifacts": artifacts}


@router.get("/tasks/{task_id}/task-artifacts/{artifact_id}/download")
def download_task_artifact(
    task_id: str,
    artifact_id: str,
    request: Request,
    expected_content_hash: str | None = None,
) -> StreamingResponse:
    settings = request.app.state.settings
    get_task_or_404(TaskRepository(settings.db_path), task_id)
    row = TaskArtifactRepository(settings.db_path).get_for_task(task_id, artifact_id)
    if row is None:
        raise not_found("task artifact not found")
    if expected_content_hash is not None and (
        len(expected_content_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_content_hash)
        or not isinstance(row.get("content_hash"), str)
        or not hmac.compare_digest(expected_content_hash, row["content_hash"])
    ):
        raise conflict("task artifact expected content hash changed")
    path = _available_task_artifact_path(
        settings=settings,
        task_id=task_id,
        stored_path=row.get("path"),
    )
    if path is None:
        raise not_found("task artifact not found")
    try:
        snapshot, content_length = _verified_artifact_snapshot(
            settings=settings,
            task_id=task_id,
            candidate=path,
            required_content_hash=row.get("content_hash"),
        )
    except _ArtifactSnapshotFailure as exc:
        raise conflict("task artifact integrity check failed") from exc
    return _attachment_response(
        snapshot,
        filename=path.name,
        media_type=_STRATEGY_ARTIFACT_MEDIA_TYPES[path.suffix.lower()],
        content_length=content_length,
    )


@router.get("/artifacts/{artifact_path:path}/preview")
def preview_artifact(artifact_path: str, request: Request):
    path = _resolve_task_artifact_path(request, artifact_path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return HTMLResponse(docx_to_html_preview(path))
    if suffix in {".html", ".htm"}:
        return FileResponse(path, media_type="text/html", filename=path.name)
    if suffix == ".pdf":
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    raise not_found("artifact preview not available")


@router.get("/artifacts/{artifact_path:path}")
def download_artifact(artifact_path: str, request: Request) -> StreamingResponse:
    path = _resolve_task_artifact_path(
        request,
        artifact_path,
        enforce_integrity=False,
    )
    task_id = path.relative_to(request.app.state.settings.tasks_dir).parts[0]
    try:
        snapshot, content_length = _verified_artifact_snapshot(
            settings=request.app.state.settings,
            task_id=task_id,
            candidate=path,
        )
    except _ArtifactSnapshotFailure as exc:
        raise conflict(exc.failure) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "text/plain"
    return _attachment_response(
        snapshot,
        filename=path.name,
        media_type=media_type,
        content_length=content_length,
    )


def _attachment_response(
    snapshot: BinaryIO,
    *,
    filename: str,
    media_type: str,
    content_length: int,
) -> StreamingResponse:
    encoded_filename = quote(filename, safe="")
    disposition = (
        f'attachment; filename="{filename}"'
        if encoded_filename == filename
        else f"attachment; filename*=utf-8''{encoded_filename}"
    )
    headers = {
        "Content-Disposition": disposition,
        "Content-Length": str(content_length),
    }
    if Path(filename).suffix.lower() == ".svg":
        headers["X-Content-Type-Options"] = "nosniff"
    try:
        return StreamingResponse(
            _snapshot_chunks(snapshot),
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(snapshot.close),
        )
    except Exception:
        snapshot.close()
        raise


def _snapshot_chunks(snapshot: BinaryIO) -> Iterator[bytes]:
    try:
        while chunk := snapshot.read(_SNAPSHOT_CHUNK_BYTES):
            yield chunk
    finally:
        snapshot.close()


class _ArtifactSnapshotFailure(RuntimeError):
    def __init__(self, failure: str) -> None:
        self.failure = failure
        super().__init__(failure)


def _verified_artifact_snapshot(
    *,
    settings,
    task_id: str,
    candidate: Path,
    required_content_hash: object = None,
    required_content_size: object = None,
) -> tuple[BinaryIO, int]:
    registry_records = _registered_artifact_records(settings=settings)
    snapshot: BinaryIO = SpooledTemporaryFile(
        max_size=_SNAPSHOT_MEMORY_LIMIT_BYTES,
        mode="w+b",
    )
    digest = hashlib.sha256()
    content_length = 0
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("artifact snapshot source is not a regular file")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while chunk := source.read(_SNAPSHOT_CHUNK_BYTES):
                snapshot.write(chunk)
                digest.update(chunk)
                content_length += len(chunk)
        content_hash = digest.hexdigest()
        if (
            required_content_size is not None
            and required_content_size != content_length
        ):
            raise _ArtifactSnapshotFailure("artifact integrity check failed")
        if required_content_hash is not None and (
            not isinstance(required_content_hash, str)
            or not hmac.compare_digest(content_hash, required_content_hash)
        ):
            raise _ArtifactSnapshotFailure("artifact integrity check failed")
        failure = _artifact_path_integrity_failure(
            settings=settings,
            task_id=task_id,
            candidate=candidate,
            records=registry_records,
            actual_size=content_length,
            actual_hash=content_hash,
        )
        if failure is not None:
            raise _ArtifactSnapshotFailure(failure)
        snapshot.seek(0)
        return snapshot, content_length
    except _ArtifactSnapshotFailure:
        snapshot.close()
        raise
    except OSError as exc:
        snapshot.close()
        raise _ArtifactSnapshotFailure("artifact integrity check failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _resolve_task_artifact_path(
    request: Request,
    artifact_path: str,
    *,
    enforce_integrity: bool = True,
) -> Path:
    raw = unquote(str(artifact_path or ""))
    if not raw or raw.startswith(("/", "\\")):
        raise not_found("artifact not found")
    settings = request.app.state.settings
    try:
        declared_root = settings.tasks_dir.absolute()
        if declared_root.is_symlink():
            raise not_found("artifact not found")
        declared = (settings.workspace / raw).absolute()
        relative = declared.relative_to(declared_root)
        if not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise not_found("artifact not found")
        task_id = relative.parts[0]
        get_task_or_404(TaskRepository(settings.db_path), task_id)

        cursor = declared_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise not_found("artifact not found")

        root = settings.tasks_dir.resolve(strict=True)
        candidate = declared.resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise not_found("artifact not found") from exc
    if not candidate.is_file():
        raise not_found("artifact not found")
    if enforce_integrity:
        _enforce_registered_artifact_integrity(
            settings=settings,
            task_id=task_id,
            candidate=candidate,
        )
    return candidate


def _enforce_registered_artifact_integrity(
    *,
    settings,
    task_id: str,
    candidate: Path,
) -> None:
    """Fail closed when a generic path has an immutable registry identity."""

    failure = _artifact_path_integrity_failure(
        settings=settings,
        task_id=task_id,
        candidate=candidate,
    )
    if failure is not None:
        raise conflict(failure)


def _artifact_path_integrity_failure(
    *,
    settings,
    task_id: str,
    candidate: Path,
    records: list[dict[str, object]] | None = None,
    actual_size: int | None = None,
    actual_hash: str | None = None,
) -> str | None:
    """Return a path-level registry failure, or ``None`` when safe to serve.

    Integrity belongs to the resolved file identity, not to the artifact id used
    to reach it.  A legacy row may therefore coexist with a verified alias, but
    it cannot downgrade that alias's hash/size contract.  A path with legacy
    rows only remains readable for compatibility.
    """

    matching = _registered_artifact_records_for_path(
        settings=settings,
        candidate=candidate,
        records=records,
    )
    if not matching:
        return None
    if any(str(record["task_id"]) != task_id for record in matching):
        return "artifact registry ownership mismatch"

    expected: list[tuple[str, int | None]] = []
    for record in matching:
        content_hash = record["content_hash"]
        content_size = record["content_size"]
        if record["registry"] == "strategy":
            provenance_json = record["provenance_json"]
            integrity_values = (content_hash, content_size, provenance_json)
            if all(value is None for value in integrity_values):
                continue
            if any(value is None for value in integrity_values):
                return "artifact registry contains invalid integrity metadata"
            if not _is_canonical_json_object(provenance_json):
                return "artifact registry contains invalid integrity metadata"
        if not isinstance(content_hash, str) or len(content_hash) != 64:
            return "artifact registry contains invalid integrity metadata"
        normalized_hash = content_hash.lower()
        if any(character not in "0123456789abcdef" for character in normalized_hash):
            return "artifact registry contains invalid integrity metadata"
        if record["registry"] == "strategy":
            if (
                isinstance(content_size, bool)
                or not isinstance(content_size, int)
                or content_size < 0
            ):
                return "artifact registry contains invalid integrity metadata"
        else:
            content_size = None
        expected.append((normalized_hash, content_size))

    if not expected:
        return None
    if (actual_size is None) != (actual_hash is None):
        return "artifact integrity check failed"
    if actual_size is None:
        try:
            actual_size = candidate.stat().st_size
        except OSError:
            return "artifact integrity check failed"
    if any(size is not None and size != actual_size for _, size in expected):
        return "artifact integrity check failed"
    if actual_hash is None:
        try:
            actual_hash = sha256_file(candidate)
        except OSError:
            return "artifact integrity check failed"
    if any(
        not hmac.compare_digest(actual_hash, content_hash)
        for content_hash, _ in expected
    ):
        return "artifact integrity check failed"
    return None


def _is_canonical_json_object(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            return False
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return hmac.compare_digest(canonical, value)


def _registered_artifact_records_for_path(
    *,
    settings,
    candidate: Path,
    records: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return registry rows whose stored path resolves to ``candidate``.

    Paths in older rows can be absolute or workspace-relative, and may not be
    lexically normalized.  Resolving each stored value avoids making an exact
    SQL string match into an integrity bypass.
    """

    available_records = (
        _registered_artifact_records(settings=settings) if records is None else records
    )
    return [
        record for record in available_records if record["resolved_path"] == candidate
    ]


def _registered_artifact_records(*, settings) -> list[dict[str, object]]:
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            """
            SELECT 'strategy' AS registry, s.task_id, a.path,
                   a.content_hash, a.content_size, a.provenance_json
              FROM strategy_artifacts a
              JOIN strategies s ON s.id = a.strategy_id
            UNION ALL
            SELECT 'task' AS registry, task_id, path, content_hash,
                   NULL AS content_size, provenance_json
              FROM task_artifacts
            """
        ).fetchall()
    records = []
    for row in rows:
        raw_path = str(row["path"] or "")
        if not raw_path:
            continue
        stored = Path(raw_path)
        declared = stored if stored.is_absolute() else settings.workspace / stored
        try:
            registered_path = declared.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        records.append(
            {
                "registry": str(row["registry"]),
                "task_id": str(row["task_id"]),
                "content_hash": row["content_hash"],
                "content_size": row["content_size"],
                "provenance_json": row["provenance_json"],
                "resolved_path": registered_path,
            }
        )
    return records


def _artifact_filename(stored_path: object) -> str:
    value = str(stored_path or "")
    if not value:
        return "artifact"
    return Path(value).name or "artifact"


def _available_task_artifact_path(
    *,
    settings,
    task_id: str,
    stored_path: object,
) -> Path | None:
    """Resolve a recorded artifact without following any task-local symlink."""

    raw = str(stored_path or "")
    if not raw:
        return None
    try:
        declared_task_root = (settings.tasks_dir / task_id).absolute()
        if declared_task_root.is_symlink():
            return None
        task_root = declared_task_root.resolve(strict=True)
        task_root.relative_to(settings.tasks_dir.resolve(strict=True))

        stored = Path(raw)
        candidate = (
            stored if stored.is_absolute() else settings.workspace / stored
        ).absolute()
        relative = candidate.relative_to(declared_task_root)

        cursor = declared_task_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None

        resolved = candidate.resolve(strict=True)
        resolved.relative_to(task_root)
        if (
            not resolved.is_file()
            or resolved.suffix.lower() not in _STRATEGY_ARTIFACT_MEDIA_TYPES
        ):
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None

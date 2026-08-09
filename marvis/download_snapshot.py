"""Verified, immutable-in-response snapshots for local file downloads."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import hmac
import os
from pathlib import Path
import stat
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from urllib.parse import quote

from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask


SNAPSHOT_CHUNK_BYTES = 64 * 1024
SNAPSHOT_MEMORY_LIMIT_BYTES = 1024 * 1024


class SnapshotIntegrityError(RuntimeError):
    """The source could not be frozen with its required byte identity."""


def verified_file_snapshot(
    candidate: Path,
    *,
    required_content_hash: str | None = None,
    required_content_size: int | None = None,
) -> tuple[BinaryIO, int, str]:
    """Copy one opened regular file into a private snapshot while hashing it."""

    snapshot: BinaryIO = SpooledTemporaryFile(
        max_size=SNAPSHOT_MEMORY_LIMIT_BYTES,
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
            raise OSError("snapshot source is not a regular file")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            while chunk := source.read(SNAPSHOT_CHUNK_BYTES):
                snapshot.write(chunk)
                digest.update(chunk)
                content_length += len(chunk)
        content_hash = digest.hexdigest()
        if (
            required_content_size is not None
            and required_content_size != content_length
        ):
            raise SnapshotIntegrityError("file snapshot integrity check failed")
        if required_content_hash is not None and not hmac.compare_digest(
            content_hash,
            required_content_hash,
        ):
            raise SnapshotIntegrityError("file snapshot integrity check failed")
        snapshot.seek(0)
        return snapshot, content_length, content_hash
    except SnapshotIntegrityError:
        snapshot.close()
        raise
    except OSError as exc:
        snapshot.close()
        raise SnapshotIntegrityError(
            "file snapshot integrity check failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def attachment_response(
    snapshot: BinaryIO,
    *,
    filename: str,
    media_type: str,
    content_length: int,
    range_header: str | None = None,
) -> Response:
    """Stream only the retained snapshot, never reopen the mutable source path."""

    encoded_filename = quote(filename, safe="")
    disposition = (
        f'attachment; filename="{filename}"'
        if encoded_filename == filename
        else f"attachment; filename*=utf-8''{encoded_filename}"
    )
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": disposition,
        "Content-Length": str(content_length),
    }
    if Path(filename).suffix.lower() == ".svg":
        headers["X-Content-Type-Options"] = "nosniff"
    try:
        byte_range = _single_byte_range(range_header, content_length)
    except ValueError:
        snapshot.close()
        return Response(
            status_code=416,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes */{content_length}",
                "Content-Length": "0",
            },
        )
    if byte_range is not None:
        start, end = byte_range
        response_length = end - start + 1
        snapshot.seek(start)
        headers["Content-Length"] = str(response_length)
        headers["Content-Range"] = f"bytes {start}-{end}/{content_length}"
    else:
        response_length = None
    try:
        return StreamingResponse(
            _snapshot_chunks(snapshot, remaining=response_length),
            status_code=206 if byte_range is not None else 200,
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(snapshot.close),
        )
    except Exception:
        snapshot.close()
        raise


def _snapshot_chunks(
    snapshot: BinaryIO,
    *,
    remaining: int | None = None,
) -> Iterator[bytes]:
    try:
        while remaining is None or remaining > 0:
            read_size = (
                SNAPSHOT_CHUNK_BYTES
                if remaining is None
                else min(SNAPSHOT_CHUNK_BYTES, remaining)
            )
            chunk = snapshot.read(read_size)
            if not chunk:
                break
            yield chunk
            if remaining is not None:
                remaining -= len(chunk)
    finally:
        snapshot.close()


def _single_byte_range(
    range_header: str | None,
    content_length: int,
) -> tuple[int, int] | None:
    """Parse one RFC 9110 byte range without ever touching the source path."""

    if not range_header:
        return None
    unit, separator, raw_spec = range_header.strip().partition("=")
    if not separator or unit.lower() != "bytes":
        return None
    if content_length <= 0 or "," in raw_spec:
        raise ValueError("range is not satisfiable")
    start_text, dash, end_text = raw_spec.strip().partition("-")
    if not dash or (not start_text and not end_text):
        raise ValueError("range is invalid")
    if not start_text:
        if not end_text.isdigit() or int(end_text) <= 0:
            raise ValueError("suffix range is invalid")
        suffix_length = min(int(end_text), content_length)
        return content_length - suffix_length, content_length - 1
    if not start_text.isdigit():
        raise ValueError("range start is invalid")
    start = int(start_text)
    if start >= content_length:
        raise ValueError("range start is beyond the snapshot")
    if not end_text:
        return start, content_length - 1
    if not end_text.isdigit():
        raise ValueError("range end is invalid")
    end = min(int(end_text), content_length - 1)
    if end < start:
        raise ValueError("range end precedes start")
    return start, end

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from marvis.errors import server_error, unprocessable

from marvis.safe_paths import assert_within


router = APIRouter(prefix="/api", tags=["materials"])
MATERIAL_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_MATERIAL_UPLOAD_FILES = 2000


def _new_material_upload_dir(settings) -> Path:
    uploads_root = Path(settings.workspace).resolve() / "material_uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        candidate = uploads_root / uuid4().hex
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise server_error("failed to allocate material upload directory")


def _validate_upload_relative_path(raw_path: str | None) -> PurePosixPath:
    value = str(raw_path or "").replace("\\", "/").strip()
    if not value:
        raise unprocessable("invalid upload path: empty filename")
    path = PurePosixPath(value)
    invalid = (
        path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    )
    if invalid:
        raise unprocessable(f"invalid upload path: {value}")
    return path


async def _save_upload_file(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size_bytes = 0
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(MATERIAL_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            size_bytes += len(chunk)
            output.write(chunk)
    await upload.close()
    return size_bytes


@router.post("/material-uploads", status_code=201)
async def upload_materials(
    request: Request,
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(default=None),
) -> dict:
    if not files:
        raise unprocessable("at least one material file is required")
    if len(files) > MAX_MATERIAL_UPLOAD_FILES:
        raise unprocessable(f"too many material files: max_files={MAX_MATERIAL_UPLOAD_FILES}")
    if relative_paths and len(relative_paths) != len(files):
        raise unprocessable("relative_paths count must match uploaded files count")

    upload_dir = _new_material_upload_dir(request.app.state.settings)
    saved_files: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    try:
        for index, upload in enumerate(files):
            raw_relative_path = (
                relative_paths[index]
                if relative_paths and index < len(relative_paths)
                else upload.filename
            )
            relative_path = _validate_upload_relative_path(raw_relative_path)
            relative_path_text = relative_path.as_posix()
            if relative_path_text in seen_paths:
                raise unprocessable(f"duplicate upload path: {relative_path_text}")
            seen_paths.add(relative_path_text)

            destination = (upload_dir / Path(*relative_path.parts)).resolve()
            try:
                destination = assert_within(upload_dir, destination)
            except PermissionError as exc:
                raise unprocessable(f"invalid upload path: {relative_path_text}") from exc
            size_bytes = await _save_upload_file(upload, destination)
            saved_files.append(
                {
                    "relative_path": relative_path_text,
                    "size_bytes": size_bytes,
                }
            )
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    return {"source_dir": str(upload_dir), "files": saved_files}

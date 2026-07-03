from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse
from marvis.errors import not_found

from marvis.output.word_preview import docx_to_html_preview

router = APIRouter(prefix="/api", tags=["artifacts"])


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
def download_artifact(artifact_path: str, request: Request) -> FileResponse:
    path = _resolve_task_artifact_path(request, artifact_path)
    return FileResponse(path, filename=path.name)


def _resolve_task_artifact_path(request: Request, artifact_path: str) -> Path:
    raw = unquote(str(artifact_path or ""))
    if not raw or raw.startswith(("/", "\\")):
        raise not_found("artifact not found")
    root = request.app.state.settings.tasks_dir.resolve()
    candidate = (request.app.state.settings.workspace / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise not_found("artifact not found") from exc
    if not candidate.is_file():
        raise not_found("artifact not found")
    return candidate

from __future__ import annotations

from types import SimpleNamespace

from fastapi import APIRouter, HTTPException, Request

from marvis.drafts.errors import DraftNotFound, DraftStateError, FetchError, PromotionError
from marvis.drafts.promotion import promote_draft, reject_draft, validate_for_promotion
from marvis.drafts.tools import (
    tool_distill_learning,
    tool_draft_script,
    tool_fetch_url,
    tool_web_search,
)
from marvis.plugins.errors import DuplicatePluginError
from marvis.routers.plugins import _require_plugin_admin


router = APIRouter(prefix="/api/drafts", tags=["drafts"])


@router.get("")
def list_drafts(
    request: Request,
    task_id: str | None = None,
    status: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    repo = request.app.state.draft_repo
    bounded_limit = None if limit is None else max(1, min(int(limit), 500))
    bounded_offset = max(0, int(offset))
    query_limit = bounded_limit + 1 if bounded_limit is not None else None
    if task_id:
        drafts = request.app.state.draft_registry.list_for_task(
            task_id,
            status=status,
            limit=query_limit,
            offset=bounded_offset,
        )
    else:
        drafts = repo.list_all_drafts(
            status=status,
            limit=query_limit,
            offset=bounded_offset,
        )
    has_more = False
    if bounded_limit is not None and len(drafts) > bounded_limit:
        has_more = True
        drafts = drafts[:bounded_limit]
    return {
        "drafts": [_public_draft(draft, include_code=False) for draft in drafts],
        "has_more": has_more,
        "limit": bounded_limit,
        "offset": bounded_offset,
    }


@router.get("/{draft_id}")
def get_draft(request: Request, draft_id: str) -> dict:
    draft = _draft_or_404(request, draft_id)
    repo = request.app.state.draft_repo
    learning_note = None
    if draft.learning_note_id:
        learning_note = repo.get_learning_note(draft.learning_note_id)
    return {
        "draft": _public_draft(draft, include_code=True),
        "runs": [_public_run(run) for run in repo.list_runs(draft.id)],
        "learning_note": _public_learning_note(learning_note),
    }


@router.post("/{draft_id}/run")
def run_draft(request: Request, draft_id: str, payload: dict) -> dict:
    draft = _draft_or_404(request, draft_id)
    run = request.app.state.draft_sandbox.run_draft(
        draft.id,
        dict(payload.get("inputs") or {}),
        task_id=draft.task_id,
    )
    return _public_run(run)


@router.post("/web-search")
def web_search(payload: dict) -> dict:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    max_results = _bounded_int(payload.get("max_results"), "max_results", default=5, minimum=1, maximum=10)
    try:
        return tool_web_search(
            {"query": query, "max_results": max_results},
            SimpleNamespace(),
        )
    except FetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/fetch-url")
def fetch_url(payload: dict) -> dict:
    url = str(payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="url is required")
    tool_payload = {"url": url}
    max_bytes = _bounded_int(payload.get("max_bytes"), "max_bytes", default=500_000, minimum=1, maximum=500_000)
    tool_payload["max_bytes"] = max_bytes
    try:
        return tool_fetch_url(tool_payload, SimpleNamespace())
    except FetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/learning-notes")
def create_learning_note(request: Request, payload: dict) -> dict:
    query = str(payload.get("query") or "").strip()
    contents = _required_text_list(payload.get("contents"), "contents")
    sources = _required_text_list(payload.get("sources"), "sources")
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    tool_payload = {
        "query": query,
        "contents": contents,
        "sources": sources,
    }
    model_id = str(payload.get("model_id") or "").strip()
    if model_id:
        tool_payload["model_id"] = model_id
    result = tool_distill_learning(
        tool_payload,
        SimpleNamespace(
            workspace=request.app.state.settings.workspace,
            datasets_root=request.app.state.settings.datasets_dir,
        ),
    )
    note = request.app.state.draft_repo.get_learning_note(str(result["learning_note_id"]))
    if note is None:
        raise HTTPException(status_code=500, detail="learning note was not saved")
    return {"learning_note": _public_learning_note(note)}


@router.post("/author")
def author_draft(request: Request, payload: dict) -> dict:
    task_id = str(payload.get("task_id") or "").strip()
    goal = str(payload.get("goal") or "").strip()
    if not task_id:
        raise HTTPException(status_code=422, detail="task_id is required")
    if not goal:
        raise HTTPException(status_code=422, detail="goal is required")
    tool_payload = {"goal": goal}
    learning_note_id = str(payload.get("learning_note_id") or "").strip()
    if learning_note_id:
        tool_payload["learning_note_id"] = learning_note_id
    model_id = str(payload.get("model_id") or "").strip()
    if model_id:
        tool_payload["model_id"] = model_id
    result = tool_draft_script(
        tool_payload,
        SimpleNamespace(
            task_id=task_id,
            workspace=request.app.state.settings.workspace,
            datasets_root=request.app.state.settings.datasets_dir,
        ),
    )
    try:
        draft = request.app.state.draft_registry.get(str(result["draft_id"]))
    except DraftNotFound as exc:
        raise HTTPException(status_code=500, detail="draft was not saved") from exc
    return {"draft": _public_draft(draft, include_code=False)}


@router.post("/{draft_id}/promote")
def promote(request: Request, draft_id: str, payload: dict) -> dict:
    _require_plugin_admin(request)
    draft = _draft_or_404(request, draft_id)
    check = validate_for_promotion(
        draft,
        sandbox=request.app.state.draft_sandbox,
        test_cases=list(payload.get("test_cases") or []),
    )
    if not check.passed:
        raise HTTPException(status_code=422, detail={"check": _public_check(check)})
    try:
        manifest = promote_draft(
            draft,
            registry=request.app.state.plugin_registry,
            drafts=request.app.state.draft_registry,
            plugins_dir=request.app.state.settings.plugins_dir,
            check=check,
        )
    except DuplicatePluginError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (DraftStateError, PromotionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request.app.state.hook_dispatcher.rebuild_index()
    return {
        "check": _public_check(check),
        "plugin": {
            "name": manifest.name,
            "version": manifest.version,
            "display_name": manifest.display_name,
            "tool_count": len(manifest.tools),
        },
    }


@router.post("/{draft_id}/reject")
def reject(request: Request, draft_id: str, payload: dict) -> dict:
    _require_plugin_admin(request)
    draft = _draft_or_404(request, draft_id)
    try:
        reject_draft(
            draft,
            drafts=request.app.state.draft_registry,
            reason=str(payload.get("reason") or ""),
            audit_repo=request.app.state.plugin_repo,
        )
    except DraftStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


def _draft_or_404(request: Request, draft_id: str):
    try:
        return request.app.state.draft_registry.get(draft_id)
    except DraftNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _public_draft(draft, *, include_code: bool) -> dict:
    return {
        "id": draft.id,
        "task_id": draft.task_id,
        "name": draft.name,
        "summary": draft.summary,
        "code": draft.code if include_code else None,
        "input_schema": draft.input_schema,
        "output_schema": draft.output_schema,
        "determinism": draft.determinism,
        "source": draft.source,
        "learning_note_id": draft.learning_note_id,
        "status": draft.status,
        "created_at": draft.created_at,
    }


def _public_run(run) -> dict:
    return {
        "id": run.id,
        "draft_id": run.draft_id,
        "task_id": run.task_id,
        "inputs_hash": run.inputs_hash,
        "ok": run.ok,
        "output": run.output,
        "error": run.error,
        "at": run.at,
    }


def _public_learning_note(note) -> dict | None:
    if note is None:
        return None
    return {
        "id": note.id,
        "query": note.query,
        "sources": list(note.sources),
        "distilled": note.distilled,
        "created_at": note.created_at,
    }


def _public_check(check) -> dict:
    return {
        "passed": check.passed,
        "problems": list(check.problems),
        "test_result": check.test_result,
    }


def _required_text_list(value, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise HTTPException(status_code=422, detail=f"{field} must be a non-empty list")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise HTTPException(status_code=422, detail=f"{field} must be a non-empty list")
    return items


def _bounded_int(value, field: str, *, default: int, minimum: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise HTTPException(status_code=422, detail=f"{field} must be between {minimum} and {maximum}")
    return parsed

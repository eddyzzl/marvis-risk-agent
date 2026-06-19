from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from marvis.drafts.errors import DraftNotFound, DraftStateError, PromotionError
from marvis.drafts.promotion import promote_draft, reject_draft, validate_for_promotion
from marvis.plugins.errors import DuplicatePluginError
from marvis.routers.plugins import _require_plugin_admin


router = APIRouter(prefix="/api/drafts", tags=["drafts"])


@router.get("")
def list_drafts(request: Request, task_id: str | None = None, status: str | None = None) -> dict:
    repo = request.app.state.draft_repo
    if task_id:
        drafts = request.app.state.draft_registry.list_for_task(task_id, status=status)
    else:
        drafts = repo.list_all_drafts(status=status)
    return {"drafts": [_public_draft(draft, include_code=False) for draft in drafts]}


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

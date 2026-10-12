from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from marvis.production_governance.router import _local_principal_id
from marvis.strategy_workbench.contracts import (
    ControlRequest,
    SaveDraft,
    WorkbenchError,
)
from marvis.strategy_workbench.control import control_search
from marvis.strategy_workbench.service import WorkbenchService

router = APIRouter(
    prefix="/api/tasks/{task_id}/strategy-workbench", tags=["strategy-workbench"]
)


def _service(request):
    return WorkbenchService(request.app.state.settings)


def _error(exc):
    raise HTTPException(exc.status, detail={"code": exc.code}) from exc


async def _body(request, contract):
    raw = await request.body()
    if len(raw) > 96 * 1024:
        raise WorkbenchError("workbench_request_too_large", 413)
    try:
        return contract.model_validate_json(raw)
    except ValueError as exc:
        raise WorkbenchError("workbench_invalid_request", 422) from exc


@router.get("/draft")
def get_draft(task_id: str, request: Request, revision: int | None = None):
    try:
        return _service(request).get(task_id, _local_principal_id(request), revision)
    except WorkbenchError as exc:
        _error(exc)


@router.put("/draft")
async def save_draft(task_id: str, request: Request):
    try:
        return _service(request).save(
            task_id, _local_principal_id(request), await _body(request, SaveDraft)
        )
    except WorkbenchError as exc:
        _error(exc)


@router.get("/executions")
def executions(task_id: str, request: Request):
    try:
        return _service(request).executions(task_id, _local_principal_id(request))
    except WorkbenchError as exc:
        _error(exc)


@router.post("/executions/{search_id}/{action}", status_code=202)
async def control(
    task_id: str,
    search_id: str,
    action: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if action not in {"pause", "cancel", "resume"}:
        raise HTTPException(404)
    from marvis.routers.plans import _fail_plan_job, _run_plan_job, _start_plan_job
    from marvis.state_machine import ConflictError

    settings = request.app.state.settings
    try:
        body = await _body(request, ControlRequest)
        actor = _local_principal_id(request)
        if action != "resume":
            return control_search(settings, task_id, actor, search_id, action, body)
        # Reuse the original plan's lease, DAG, confirmations and effect gates.
        # CAS admission and the original step reset commit in one SQLite transaction.
        service = _service(request)
        listing = service.executions(task_id, actor)
        row = next((item for item in listing["items"] if item["id"] == search_id), None)
        if row is None:
            raise WorkbenchError("workbench_search_not_found", 404)
        from marvis.strategy_workbench.service import authenticated_evidence

        verified = authenticated_evidence(settings, task_id)
        job_id = _start_plan_job(request, task_id)
        try:
            request.app.state.plan_repo.retry_failed_step(
                row["plan_id"],
                row["step_id"],
                before_reset=lambda conn: control_search(
                    settings,
                    task_id,
                    actor,
                    search_id,
                    action,
                    body,
                    conn=conn,
                    verified=verified,
                ),
                preserve_target_confirmation=True,
            )
        except (WorkbenchError, ConflictError, KeyError) as exc:
            _fail_plan_job(settings.db_path, job_id, exc)
            raise
        background_tasks.add_task(
            _run_plan_job,
            job_id,
            settings.db_path,
            request.app.state.plan_executor,
            row["plan_id"],
        )
        return {
            "id": search_id,
            "state": "resuming",
            "plan_id": row["plan_id"],
            "job_id": job_id,
        }
    except WorkbenchError as exc:
        _error(exc)
    except ConflictError as exc:
        raise HTTPException(409, detail={"code": "workbench_plan_not_ready"}) from exc

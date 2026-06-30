from fastapi import APIRouter, Request, Response

from marvis.api_task_payloads import task_payload
from marvis.db import TaskRepository


router = APIRouter(prefix="/api", tags=["tasks"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks")
def list_tasks(
    request: Request,
    response: Response,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    repo = _repo(request)
    bounded_limit = None if limit is None else max(1, min(int(limit), 500))
    bounded_offset = max(0, int(offset))
    query_limit = bounded_limit + 1 if bounded_limit is not None else None
    tasks = repo.list_tasks(limit=query_limit, offset=bounded_offset)
    has_more = False
    if bounded_limit is not None and len(tasks) > bounded_limit:
        has_more = True
        tasks = tasks[:bounded_limit]
    if bounded_limit is not None or bounded_offset:
        response.headers["X-Result-Limit"] = "" if bounded_limit is None else str(bounded_limit)
        response.headers["X-Result-Offset"] = str(bounded_offset)
        response.headers["X-Result-Has-More"] = "true" if has_more else "false"
    return [
        task_payload(repo, task, request.app.state.settings.tasks_dir)
        for task in tasks
    ]

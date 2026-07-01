from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from marvis.agent_memory.api_support import (
    memory_api_filter_match,
    memory_distillation_detail,
    memory_distillation_payload,
    memory_entry_payload,
)
from marvis.agent_memory.consolidation import ConsolidationScheduler
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.evolution import EvolutionManager
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import TaskRepository


router = APIRouter(prefix="/api", tags=["agent-memory"])
DEFAULT_AGENT_MEMORY_LIMIT = 500
MAX_AGENT_MEMORY_LIMIT = 500
AGENT_MEMORY_SCAN_PAGE_SIZE = 500


@router.get("/agent-memory")
def list_agent_memory(
    request: Request,
    memory_type: str | None = None,
    status: str | None = None,
    source_task_id: str | None = None,
    model_name: str | None = None,
    channel: str | None = None,
    month: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    bounded_limit = _bounded_limit(limit)
    bounded_offset = _bounded_offset(offset)
    try:
        if any(value not in (None, "") for value in (model_name, channel, month)):
            entries, has_more = _list_payload_filtered_entries(
                store,
                status=status,
                memory_type=memory_type,
                source_task_id=source_task_id,
                model_name=model_name,
                channel=channel,
                month=month,
                limit=bounded_limit,
                offset=bounded_offset,
            )
        else:
            entries = store.list_entries(
                status=status,
                memory_type=memory_type,
                source_task_id=source_task_id,
                limit=bounded_limit + 1,
                offset=bounded_offset,
            )
            has_more = len(entries) > bounded_limit
            entries = entries[:bounded_limit]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid memory filter: {exc}") from exc
    items = [memory_entry_payload(entry) for entry in entries]
    return {
        "items": items,
        "limit": bounded_limit,
        "offset": bounded_offset,
        "has_more": has_more,
    }


@router.get("/agent-memory/distillations")
def list_agent_memory_distillations(
    request: Request,
    category: str | None = None,
    include_superseded: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    bounded_limit = _bounded_limit(limit)
    bounded_offset = _bounded_offset(offset)
    try:
        distillations = store.list_distillations(
            category=category,
            include_superseded=include_superseded,
            limit=bounded_limit + 1,
            offset=bounded_offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid distillation filter: {exc}") from exc
    has_more = len(distillations) > bounded_limit
    return {
        "items": [memory_distillation_payload(item) for item in distillations[:bounded_limit]],
        "limit": bounded_limit,
        "offset": bounded_offset,
        "has_more": has_more,
    }


@router.post("/agent-memory/consolidate")
def consolidate_agent_memory(
    request: Request,
    category: str | None = None,
) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    scheduler = ConsolidationScheduler(
        DistillationEngine(store),
        EvolutionManager(store),
        store,
        async_mode=False,
    )
    try:
        result = scheduler.consolidate_all([category] if category else None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"consolidated": result}


@router.get("/agent-memory/distillations/{distillation_id}")
def get_agent_memory_distillation(distillation_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        distillation = store.get_distillation(distillation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory distillation not found") from exc
    return memory_distillation_detail(store, distillation)


@router.post("/agent-memory/distillations/{distillation_id}/rollback")
def rollback_agent_memory_distillation(distillation_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    predecessor = store.find_superseded_by(distillation_id)
    try:
        EvolutionManager(store).rollback(distillation_id)
        distillation = store.get_distillation(distillation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory distillation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    restored = (
        memory_distillation_payload(store.get_distillation(predecessor.id))
        if predecessor is not None
        else None
    )
    return {
        "distillation": memory_distillation_payload(distillation),
        "restored": restored,
    }


@router.get("/agent-memory/{memory_id}")
def get_agent_memory(memory_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entry = store.get_entry(memory_id, include_deleted=True, audit=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    return {
        "memory": memory_entry_payload(entry),
        "events": store.list_events(memory_id),
    }


@router.post("/agent-memory/{memory_id}/disable")
def disable_agent_memory(memory_id: str, request: Request) -> dict:
    return _set_agent_memory_status(request, memory_id, "disabled")


@router.post("/agent-memory/{memory_id}/enable")
def enable_agent_memory(memory_id: str, request: Request) -> dict:
    return _set_agent_memory_status(request, memory_id, "active")


@router.delete("/agent-memory/{memory_id}")
def delete_agent_memory(memory_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entry = store.delete(memory_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    return {"memory": memory_entry_payload(entry), "events": store.list_events(memory_id)}


@router.get("/tasks/{task_id}/agent/messages/{message_id}/memory-references")
def get_agent_message_memory_references(
    task_id: str,
    message_id: str,
    request: Request,
) -> dict:
    repo = TaskRepository(request.app.state.settings.db_path)
    if repo.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")
    for message in repo.list_agent_messages(task_id):
        if message.get("id") == message_id:
            references = (message.get("metadata") or {}).get("memory_references")
            return {
                "task_id": task_id,
                "message_id": message_id,
                "memory_references": references if isinstance(references, list) else [],
            }
    raise HTTPException(status_code=404, detail="Agent message not found")


def _set_agent_memory_status(request: Request, memory_id: str, status: str) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entry = store.set_status(memory_id, status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"memory": memory_entry_payload(entry), "events": store.list_events(memory_id)}


def _bounded_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_AGENT_MEMORY_LIMIT
    return max(1, min(int(limit), MAX_AGENT_MEMORY_LIMIT))


def _bounded_offset(offset: int) -> int:
    return max(0, int(offset))


def _list_payload_filtered_entries(
    store: AgentMemoryStore,
    *,
    status: str | None,
    memory_type: str | None,
    source_task_id: str | None,
    model_name: str | None,
    channel: str | None,
    month: str | None,
    limit: int,
    offset: int,
):
    matched = []
    scan_offset = 0
    required_count = offset + limit + 1
    while len(matched) < required_count:
        page = store.list_entries(
            status=status,
            memory_type=memory_type,
            source_task_id=source_task_id,
            limit=AGENT_MEMORY_SCAN_PAGE_SIZE,
            offset=scan_offset,
        )
        if not page:
            break
        for entry in page:
            item = memory_entry_payload(entry)
            if memory_api_filter_match(
                item,
                source_task_id=source_task_id,
                model_name=model_name,
                channel=channel,
                month=month,
            ):
                matched.append(entry)
        if len(page) < AGENT_MEMORY_SCAN_PAGE_SIZE:
            break
        scan_offset += AGENT_MEMORY_SCAN_PAGE_SIZE
    window = matched[offset : offset + limit + 1]
    return window[:limit], len(window) > limit

"""Governed write boundary for Agent memory candidates.

Hooks receive identifiers and categorical metadata only.  Candidate summaries,
payloads, reasons, and message ids stay inside the memory policy/redaction
boundary owned by :class:`AgentMemoryStore`.
"""

from __future__ import annotations

import logging

from marvis.agent_memory.models import MemoryCandidate


logger = logging.getLogger(__name__)


def save_memory_candidate(
    store,
    candidate: MemoryCandidate,
    *,
    task_id: str | None = None,
    hook_dispatcher=None,
):
    """Persist one candidate and emit non-veto, privacy-bounded hooks.

    ``memory.before_save`` is an observation point, not an authorization or
    mutation seam: hook failures never change whether the deterministic memory
    policy accepts the candidate.  ``memory.after_save`` fires only after an
    active entry has committed.
    """

    resolved_task_id = task_id or candidate.source_task_id
    _dispatch_memory_hook(
        hook_dispatcher,
        "memory.before_save",
        {
            "task_id": resolved_task_id,
            "memory_type": candidate.memory_type,
            "confidence": candidate.confidence,
        },
        task_id=resolved_task_id,
    )
    entry = store.create(candidate, task_id=task_id)
    if entry.status == "active":
        _dispatch_memory_hook(
            hook_dispatcher,
            "memory.after_save",
            {
                "task_id": resolved_task_id,
                "memory_id": entry.id,
                "memory_type": entry.memory_type,
                "status": entry.status,
            },
            task_id=resolved_task_id,
        )
    return entry


def _dispatch_memory_hook(
    hook_dispatcher,
    event: str,
    payload: dict,
    *,
    task_id: str | None,
) -> None:
    if hook_dispatcher is None or not task_id:
        return
    try:
        hook_dispatcher.dispatch(event, payload, task_id=task_id)
    except Exception as exc:
        logger.warning(
            "%s hook dispatch failed for task %s: %s",
            event,
            task_id,
            exc,
        )


__all__ = ["save_memory_candidate"]

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

from marvis.agent_memory.extractors import (
    USER_PREFERENCE_RESERVED_TOPIC,
    classify_user_preference_capture,
    extract_user_preference,
)
from marvis.agent_memory.retrieval import MemoryQuery, retrieve_with_distillations
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import TaskRepository
from marvis.domain import TaskRecord
from marvis.memory_policy import load_memory_policy


# MEM-9: shown to the user (via add_agent_message, same channel memory
# reference badges already use) when an explicit "please remember" marker was
# present but the preference was declined because it targets the reserved
# skill/runtime capability -- a receipt instead of a silent drop.
RESERVED_TOPIC_RECEIPT_TEXT = "该偏好涉及保留能力（技能/运行时配置），本次未记录为记忆。"


logger = logging.getLogger(__name__)


def capture_user_preference_memory(
    settings,
    task_id: str,
    message: dict,
    *,
    extractor=extract_user_preference,
    hook_dispatcher=None,
) -> None:
    # Memory policy gate (auto_distill): this function performs AUTOMATIC per-turn
    # capture of user messages as memory candidates. When the flag is OFF the user
    # has disabled automatic distillation, so capture must be a no-op. (The flag
    # governs only automatic capture; the explicit user-triggered
    # POST /agent-memory/consolidate endpoint stays functional regardless.)
    if not load_memory_policy(settings.workspace).auto_distill:
        return
    extractor_message = {"content": message.get("content"), "id": message.get("id")}
    candidate = extractor(extractor_message)
    if candidate is None:
        # MEM-9: an explicit "please remember" marker that got declined for
        # touching the reserved skill/runtime topic deserves a receipt, not a
        # silent drop -- this is the user's one direct, self-initiated
        # contract with the memory system. Every other reason for returning
        # None (no marker present at all, or the generic redaction/size
        # policy rejecting it -- already audited as a 'reject' memory event
        # by store.create) stays silent as before.
        if (
            extractor is extract_user_preference
            and classify_user_preference_capture(extractor_message) == USER_PREFERENCE_RESERVED_TOPIC
        ):
            _add_reserved_topic_receipt(settings, task_id)
        return
    store = AgentMemoryStore(settings.db_path)
    try:
        entry = store.create(
            replace(
                candidate,
                source_task_id=task_id,
                source_message_id=str(message.get("id") or ""),
            )
        )
    except Exception as exc:
        logger.warning(
            "failed to save user preference memory for task %s: %s",
            task_id,
            exc,
        )
        return
    if entry.status == "active":
        dispatch_memory_after_save(hook_dispatcher, task_id=task_id, memory_type=entry.memory_type)


def _add_reserved_topic_receipt(settings, task_id: str) -> None:
    try:
        TaskRepository(settings.db_path).add_agent_message(
            task_id,
            role="assistant",
            stage="chat",
            content=RESERVED_TOPIC_RECEIPT_TEXT,
            metadata={"intent": "memory_capture_declined", "reason": "reserved_topic"},
        )
    except Exception as exc:
        logger.warning(
            "failed to record reserved-topic memory receipt for task %s: %s",
            task_id,
            exc,
        )


def dispatch_memory_after_save(
    hook_dispatcher,
    *,
    task_id: str | None,
    memory_type: str,
) -> None:
    # The 'memory.after_save' consolidation trigger (CONSOLIDATION_TRIGGERS)
    # previously had zero emit call sites: it was declared but never fired, so
    # V2-only workflows -- which never touch the V1.1 validation.completed /
    # report.after_generate hooks -- could accumulate raw memories forever
    # without ever being distilled. This fires it from every active-memory
    # capture point instead.
    if hook_dispatcher is None or not task_id:
        return
    try:
        hook_dispatcher.dispatch(
            "memory.after_save",
            {"task_id": task_id, "memory_type": memory_type},
            task_id=task_id,
        )
    except Exception as exc:
        logger.warning(
            "memory.after_save hook dispatch failed for task %s: %s",
            task_id,
            exc,
        )


def memory_entry_payload(entry) -> dict:
    return {
        "id": entry.id,
        "memory_type": entry.memory_type,
        "status": entry.status,
        "summary": entry.summary,
        "payload": entry.payload,
        "source_task_id": entry.source_task_id,
        "source_message_id": entry.source_message_id,
        "confidence": entry.confidence,
        "reason": entry.reason,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "deleted_at": entry.deleted_at,
    }


def memory_distillation_payload(distillation) -> dict:
    return {
        "id": distillation.id,
        "kind": "distillation",
        "category": distillation.category,
        "memory_type": distillation.category,
        "scope_key": distillation.scope_key,
        "status": distillation.status,
        "summary": distillation.distilled_summary,
        "payload": distillation.structured,
        "source_memory_ids": list(distillation.source_memory_ids),
        "support_count": distillation.support_count,
        "confidence": distillation.confidence,
        "superseded_by": distillation.superseded_by,
        "created_at": distillation.created_at,
        "updated_at": distillation.updated_at,
    }


def memory_distillation_detail(
    store: AgentMemoryStore,
    distillation,
) -> dict:
    source_memories = []
    for memory_id in distillation.source_memory_ids:
        try:
            source = store.get_entry(memory_id, include_deleted=True, audit=False)
        except KeyError:
            continue
        source_memories.append(memory_entry_payload(source))
    predecessor = store.find_superseded_by(distillation.id)
    return {
        "distillation": memory_distillation_payload(distillation),
        "source_memories": source_memories,
        "predecessor": (
            memory_distillation_payload(predecessor)
            if predecessor is not None
            else None
        ),
        "events": store.list_distillation_events(distillation.id),
    }


def memory_api_filter_match(
    item: dict,
    *,
    source_task_id: str | None,
    model_name: str | None,
    channel: str | None,
    month: str | None,
) -> bool:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    checks = (
        (source_task_id, item.get("source_task_id")),
        (model_name, payload.get("model_name")),
        (channel, payload.get("channel")),
        (month, payload.get("month")),
    )
    return all(
        expected in (None, "") or str(actual or "") == str(expected)
        for expected, actual in checks
    )


def agent_memory_context_from_store(
    store: AgentMemoryStore,
    task: TaskRecord,
    *,
    stage: str,
    user_message: str = "",
    evidence: dict | None = None,
) -> dict | None:
    # Memory policy gate (reference_cross_task): when this flag is OFF the user
    # has disabled injecting prior-task agent memory into the prompt context, so
    # we short-circuit here and inject nothing. Gating in this single function
    # covers every caller (all turn handlers retrieve memory through this path).
    # The workspace is derived from the store's db_path (workspace/marvis.sqlite).
    workspace = Path(store.db_path).parent
    if not load_memory_policy(workspace).reference_cross_task:
        return None
    query = agent_memory_query(task, user_message=user_message, evidence=evidence)
    # Separate quotas so high-scoring distillations cannot squeeze precise
    # single-task raw experience entirely out of the limit=6 prompt budget.
    packets = retrieve_with_distillations(store, query, limit=6, raw_quota=3)
    if not packets:
        return None
    raw_packets: list[tuple[str, dict]] = []
    for packet in packets:
        if packet.get("kind") == "distillation":
            continue
        memory_id = packet.get("id")
        if memory_id:
            raw_packets.append((str(memory_id), packet))
    found_ids = store.record_retrievals(
        [memory_id for memory_id, _ in raw_packets],
        task_id=task.id,
    )
    memories = [
        packet
        for packet in packets
        if packet.get("kind") == "distillation"
        or str(packet.get("id") or "") in found_ids
    ]
    if not memories:
        return None
    return {
        "scope": "cross_task_agent_memory",
        "stage": stage,
        "memories": memories,
    }


def agent_memory_query(
    task: TaskRecord,
    *,
    user_message: str = "",
    evidence: dict | None = None,
) -> MemoryQuery:
    validation_results = (evidence or {}).get("validation_results")
    dimensions = agent_memory_dimensions_from_validation_results(validation_results)
    return MemoryQuery(
        model_name=task.model_name or dimensions.get("model_name"),
        scope=dimensions.get("scope"),
        channel=dimensions.get("channel"),
        month=dimensions.get("month"),
        keywords=agent_memory_keywords(task, user_message, dimensions),
    )


def agent_memory_dimensions_from_validation_results(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    dimensions: dict[str, str] = {}
    for key in ("model_name", "model_version", "scope", "channel", "month"):
        item = value.get(key)
        if item not in (None, ""):
            dimensions[key] = str(item)
    basic_info = value.get("basic_info")
    if isinstance(basic_info, dict):
        for source_key, target_key in (
            ("model_name", "model_name"),
            ("model_version", "model_version"),
            ("model_scope", "scope"),
            ("scope", "scope"),
            ("channel", "channel"),
            ("month", "month"),
        ):
            item = basic_info.get(source_key)
            if target_key not in dimensions and item not in (None, ""):
                dimensions[target_key] = str(item)
    return dimensions


def agent_memory_keywords(
    task: TaskRecord,
    user_message: str,
    dimensions: dict[str, str],
) -> tuple[str, ...]:
    values = [
        task.model_name,
        task.model_version,
        task.algorithm,
        dimensions.get("scope"),
        dimensions.get("channel"),
        dimensions.get("month"),
    ]
    compact_message = "".join(str(user_message or "").split())
    for marker in (
        "A卡",
        "B卡",
        "C卡",
        "额度",
        "利率",
        "前筛",
        "KS",
        "AUC",
        "PSI",
        "bad_flag",
        "RMC_SAMPLE_DF",
    ):
        if marker.lower() in compact_message.lower():
            values.append(marker)
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value or "").strip())
    )


def audit_agent_memory_use_from_store(
    store: AgentMemoryStore,
    message: dict,
    *,
    task_id: str,
) -> None:
    metadata = message.get("metadata") or {}
    references = metadata.get("memory_references")
    if not isinstance(references, list):
        return
    for reference in references:
        if not isinstance(reference, dict):
            continue
        memory_id = reference.get("id")
        if not memory_id:
            continue
        if reference.get("kind") == "distillation":
            try:
                store.record_distillation_use(
                    str(memory_id),
                    task_id=task_id,
                    message_id=message.get("id"),
                    use_reason=str(reference.get("use_reason") or "agent"),
                )
            except (KeyError, ValueError):
                continue
            continue
        try:
            store.record_use(
                str(memory_id),
                task_id=task_id,
                message_id=message.get("id"),
                use_reason=str(reference.get("use_reason") or "agent"),
            )
        except KeyError:
            continue

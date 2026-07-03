from __future__ import annotations

import json
from typing import Any

from marvis.agent_memory.policy import PAYLOAD_FIELD_ALLOWLISTS


MEMORY_USAGE_RULES = (
    "跨任务记忆只能辅助解释、参数建议、风险提醒、历史对比和报告措辞；"
    "不能改变 KS/AUC/PSI/分数一致性等平台确定性验证结果。"
    "若提到历史模型效果对比，只能依据 cross_task_memory.memories 中的结构化 payload "
    "和当前 available_evidence/evidence 中的平台结构化结果。"
)
MEMORY_PROMPT_SUMMARY_MAX_CHARS = 400
# Distillation structured payloads can carry an unbounded source_task_ids/
# source_memory_ids list once support_count grows into the dozens; a weak
# model gets no value from a wall of uuids and DISTILL_SYS explicitly forbids
# echoing task ids back. Bound both to a count plus a short, deterministic
# sample instead of dropping the field outright (keeps some traceability).
DISTILLATION_ID_SAMPLE_SIZE = 3
CROSS_TASK_MEMORY_CHAR_BUDGET = 3000


def normalize_memory_context(memory_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(memory_context, dict):
        return None
    raw_memories = memory_context.get("memories")
    if not isinstance(raw_memories, list):
        return None
    memories = [
        _memory_packet(memory)
        for memory in raw_memories
        if isinstance(memory, dict) and memory.get("id")
    ]
    if not memories:
        return None
    fitted, truncated = _fit_memories_to_budget(memories)
    if not fitted:
        return None
    return {
        "scope": str(memory_context.get("scope") or "cross_task_agent_memory"),
        "usage_rules": MEMORY_USAGE_RULES,
        "memories": fitted,
        # LLM-5: memory injection is one of the three named highest-volume
        # prompt touch points — surfaced so callers can set the audit-visible
        # ``truncated`` flag on their complete() call (see marvis.agent.service
        # for the two integration points; other callers of this module get the
        # same detection but may not yet plumb it through to a complete() call).
        "truncated": truncated,
    }


def memory_context_was_truncated(normalized_memory_context: dict[str, Any] | None) -> bool:
    """Whether ``normalize_memory_context`` had to drop memories to fit budget."""
    if not isinstance(normalized_memory_context, dict):
        return False
    return bool(normalized_memory_context.get("truncated"))


_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _fit_memories_to_budget(
    memories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    # Final safeguard against an unbounded cross_task_memory section: even
    # after per-packet trimming (_bounded_distillation_payload, source id
    # sampling), enough packets can still add up past a sane prompt budget.
    # Keep the highest-confidence packets first, in their original relative
    # order, dropping lowest-confidence packets once the budget is exceeded.
    total_chars = sum(_packet_char_len(packet) for packet in memories)
    if total_chars <= CROSS_TASK_MEMORY_CHAR_BUDGET:
        return memories, False
    ordered = sorted(
        enumerate(memories),
        key=lambda item: (-_CONFIDENCE_RANK.get(str(item[1].get("confidence") or ""), 0), item[0]),
    )
    kept: list[tuple[int, dict[str, Any]]] = []
    running_total = 0
    for index, packet in ordered:
        packet_len = _packet_char_len(packet)
        if kept and running_total + packet_len > CROSS_TASK_MEMORY_CHAR_BUDGET:
            continue
        kept.append((index, packet))
        running_total += packet_len
    kept.sort(key=lambda item: item[0])
    return [packet for _, packet in kept], len(kept) < len(memories)


def _packet_char_len(packet: dict[str, Any]) -> int:
    return len(json.dumps(packet, ensure_ascii=False, separators=(",", ":")))


def memory_references(
    memory_context: dict[str, Any] | None,
    *,
    use_reason: str,
) -> list[dict[str, Any]]:
    # Built directly from the raw memories (not through _memory_packet /
    # normalize_memory_context) so the audit trail keeps the full
    # source_memory_ids list for traceability even though the prompt-facing
    # packet bounds it to a count + sample (MEM-11). The prompt budget only
    # governs what the model sees, not what gets audited.
    if not isinstance(memory_context, dict):
        return []
    raw_memories = memory_context.get("memories")
    if not isinstance(raw_memories, list):
        return []
    references: list[dict[str, Any]] = []
    for memory in raw_memories:
        if not isinstance(memory, dict) or not memory.get("id"):
            continue
        reference = {
            "kind": memory.get("kind") or "raw",
            "id": str(memory["id"]),
            "memory_type": memory.get("memory_type"),
            "source_task_id": memory.get("source_task_id"),
            "confidence": memory.get("confidence") or "medium",
            "use_reason": use_reason,
        }
        if memory.get("support_count") is not None:
            reference["support_count"] = int(memory.get("support_count") or 0)
        if isinstance(memory.get("source_memory_ids"), list):
            reference["source_memory_ids"] = [
                str(item) for item in memory["source_memory_ids"]
            ]
        references.append(reference)
    return references


def attach_memory_metadata(
    metadata: dict[str, Any],
    memory_context: dict[str, Any] | None,
    *,
    use_reason: str,
) -> dict[str, Any]:
    references = memory_references(memory_context, use_reason=use_reason)
    if references:
        metadata["memory_references"] = references
    return metadata


def add_memory_to_prompt_payload(
    payload: dict[str, Any],
    memory_context: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_memory_context(memory_context)
    if normalized is not None:
        # `truncated` is bookkeeping for the caller (LLM-5 audit trail via
        # memory_context_was_truncated), not something the model should see in
        # its own prompt payload.
        prompt_facing = {key: value for key, value in normalized.items() if key != "truncated"}
        payload["cross_task_memory"] = prompt_facing
        payload["instructions"] = (
            str(payload.get("instructions") or "")
            + "跨任务记忆只可作为解释和风险提醒依据，不能改变平台确定性指标；"
            "引用历史对比时必须说明来自历史记忆并保持置信度克制。"
        )
    return payload


def _memory_packet(memory: dict[str, Any]) -> dict[str, Any]:
    summary_text = _truncate_text(memory.get("summary"), MEMORY_PROMPT_SUMMARY_MAX_CHARS)
    age_days = memory.get("age_days")
    if isinstance(age_days, int) and age_days >= 0:
        summary_text = f"{summary_text}（{age_days} 天前）"
    packet = {
        "kind": memory.get("kind") or "raw",
        "id": str(memory.get("id")),
        "memory_type": memory.get("memory_type"),
        "summary": summary_text,
        "source_task_id": memory.get("source_task_id"),
        "confidence": memory.get("confidence") or "medium",
        "match_reason": memory.get("match_reason") or "",
    }
    if age_days is not None:
        packet["age_days"] = age_days
    if memory.get("observed_at"):
        packet["observed_at"] = memory["observed_at"]
    if memory.get("support_count") is not None:
        packet["support_count"] = int(memory.get("support_count") or 0)
    if isinstance(memory.get("source_memory_ids"), list):
        packet["source_memory_ids_count"] = len(memory["source_memory_ids"])
        packet["source_memory_ids_sample"] = _id_sample(memory["source_memory_ids"])
    payload = memory.get("payload")
    if isinstance(payload, dict):
        packet["payload"] = (
            _bounded_distillation_payload(payload)
            if packet["kind"] == "distillation"
            else _bounded_payload(payload, memory.get("memory_type"))
        )
    return packet


def _bounded_distillation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # DISTILL_SYS explicitly forbids echoing task ids back; source_task_ids
    # (and any other *_ids list) grows unbounded with support_count, so it is
    # replaced with a count plus a short deterministic sample instead of the
    # full list. Every other structured field (metric_ranges, scopes,
    # channels, support, fields, months_covered, ...) passes through as-is.
    bounded: dict[str, Any] = {}
    for key, value in payload.items():
        if key.endswith("_ids") and isinstance(value, list):
            bounded[f"{key}_count"] = len(value)
            bounded[f"{key}_sample"] = _id_sample(value)
            continue
        bounded[key] = value
    return bounded


def _id_sample(ids: list[Any]) -> list[str]:
    return [str(item) for item in ids[-DISTILLATION_ID_SAMPLE_SIZE:]]


def _bounded_payload(payload: dict[str, Any], memory_type: Any) -> dict[str, Any]:
    # Filter by the memory's own type allowlist (the same set policy enforces at
    # ingestion). The previous hard-coded model_experience set silently dropped the
    # structured payload of every non-model memory type into `{}`.
    allowed_fields = PAYLOAD_FIELD_ALLOWLISTS.get(str(memory_type or ""), frozenset())
    return {key: value for key, value in payload.items() if key in allowed_fields}


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

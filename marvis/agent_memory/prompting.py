from __future__ import annotations

from typing import Any

from marvis.agent_memory.policy import PAYLOAD_FIELD_ALLOWLISTS


MEMORY_USAGE_RULES = (
    "跨任务记忆只能辅助解释、参数建议、风险提醒、历史对比和报告措辞；"
    "不能改变 KS/AUC/PSI/分数一致性等平台确定性验证结果。"
    "若提到历史模型效果对比，只能依据 cross_task_memory.memories 中的结构化 payload "
    "和当前 available_evidence/evidence 中的平台结构化结果。"
)
MEMORY_PROMPT_SUMMARY_MAX_CHARS = 400


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
    return {
        "scope": str(memory_context.get("scope") or "cross_task_agent_memory"),
        "usage_rules": MEMORY_USAGE_RULES,
        "memories": memories,
    }


def memory_references(
    memory_context: dict[str, Any] | None,
    *,
    use_reason: str,
) -> list[dict[str, Any]]:
    normalized = normalize_memory_context(memory_context)
    if normalized is None:
        return []
    references: list[dict[str, Any]] = []
    for memory in normalized["memories"]:
        references.append(
            {
                "id": memory["id"],
                "memory_type": memory.get("memory_type"),
                "source_task_id": memory.get("source_task_id"),
                "confidence": memory.get("confidence") or "medium",
                "use_reason": use_reason,
            }
        )
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
        payload["cross_task_memory"] = normalized
        payload["instructions"] = (
            str(payload.get("instructions") or "")
            + "跨任务记忆只可作为解释和风险提醒依据，不能改变平台确定性指标；"
            "引用历史对比时必须说明来自历史记忆并保持置信度克制。"
        )
    return payload


def _memory_packet(memory: dict[str, Any]) -> dict[str, Any]:
    packet = {
        "id": str(memory.get("id")),
        "memory_type": memory.get("memory_type"),
        "summary": _truncate_text(memory.get("summary"), MEMORY_PROMPT_SUMMARY_MAX_CHARS),
        "source_task_id": memory.get("source_task_id"),
        "confidence": memory.get("confidence") or "medium",
        "match_reason": memory.get("match_reason") or "",
    }
    payload = memory.get("payload")
    if isinstance(payload, dict):
        packet["payload"] = _bounded_payload(payload, memory.get("memory_type"))
    return packet


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

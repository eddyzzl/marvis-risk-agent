from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import uuid
from typing import Any

from marvis.agent_memory.models import normalize_memory_type
from marvis.agent_memory.policy import classify_distillation_payload


CONFIDENCE_THRESHOLDS = {"high": 4, "medium": 2}
MAX_DISTILLED_SUMMARY_CHARS = 400
DISTILLATION_STATUSES = ("active", "rolled_back")
DISTILL_SYS = (
    "你在压缩 MARVIS 的历史记忆。只能基于给定的结构化字段和原始记忆措辞，输出一句话经验。"
    "禁止引入任何未在输入中出现的事实、数字或结论。不要输出任务 ID。"
)


@dataclass(frozen=True)
class MemoryDistillation:
    id: str
    category: str
    scope_key: str
    distilled_summary: str
    structured: dict[str, Any] = field(default_factory=dict)
    source_memory_ids: tuple[str, ...] = ()
    support_count: int = 0
    confidence: str = "low"
    superseded_by: str | None = None
    status: str = "active"
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_memory_type(self.category))
        object.__setattr__(self, "scope_key", str(self.scope_key))
        object.__setattr__(
            self,
            "distilled_summary",
            str(self.distilled_summary)[:MAX_DISTILLED_SUMMARY_CHARS],
        )
        object.__setattr__(
            self,
            "source_memory_ids",
            tuple(str(memory_id) for memory_id in self.source_memory_ids),
        )
        object.__setattr__(self, "support_count", int(self.support_count))
        object.__setattr__(self, "confidence", normalize_distillation_confidence(self.confidence))
        object.__setattr__(self, "status", normalize_distillation_status(self.status))


def new_distillation(
    *,
    category: str,
    scope_key: str,
    distilled_summary: str,
    structured: dict[str, Any] | None = None,
    source_memory_ids: tuple[str, ...] = (),
    support_count: int = 0,
    confidence: str | None = None,
) -> MemoryDistillation:
    now = _now_iso()
    return MemoryDistillation(
        id=f"dist_{uuid.uuid4().hex}",
        category=category,
        scope_key=scope_key,
        distilled_summary=distilled_summary,
        structured=structured or {},
        source_memory_ids=source_memory_ids,
        support_count=support_count,
        confidence=confidence or confidence_from_support(support_count),
        superseded_by=None,
        created_at=now,
        updated_at=now,
    )


def confidence_from_support(support_count: int) -> str:
    support = int(support_count)
    if support >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if support >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def normalize_distillation_confidence(value: str) -> str:
    normalized = str(value or "low").strip().lower()
    if normalized not in {"high", "medium", "low"}:
        raise ValueError(f"unsupported distillation confidence: {value}")
    return normalized


def normalize_distillation_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in DISTILLATION_STATUSES:
        raise ValueError(f"unsupported distillation status: {value}")
    return normalized


class DistillationEngine:
    def __init__(self, store, llm_factory=None, policy=None):
        self._store = store
        self._llm_factory = llm_factory
        self._policy = policy

    def distill_category(self, category: str) -> list[MemoryDistillation]:
        normalized = normalize_memory_type(category)
        entries = self._store.list_entries(memory_type=normalized, status="active", limit=2000)
        groups = self._group_by_scope(entries)
        results = []
        for scope_key, members in groups.items():
            try:
                distilled = self._distill_group(normalized, scope_key, members)
            except Exception:
                continue
            if distilled is not None:
                results.append(distilled)
        return results

    def _group_by_scope(self, entries: list[Any]) -> dict[str, list[Any]]:
        groups: dict[str, list[Any]] = {}
        for entry in entries:
            groups.setdefault(self._scope_key_for(entry), []).append(entry)
        return groups

    def _scope_key_for(self, entry: Any) -> str:
        category = _entry_category(entry)
        payload = _entry_payload(entry)
        if category == "field_convention":
            fields = sorted(key for key, value in payload.items() if value not in (None, ""))
            return f"field_convention:{','.join(fields) or 'general'}"
        if category == "validation_pitfall":
            return f"validation_pitfall:{payload.get('failure_kind') or 'general'}"
        if category == "model_experience":
            return ":".join(
                [
                    "model_experience",
                    str(payload.get("model_name") or ""),
                    str(payload.get("scope") or ""),
                    str(payload.get("channel") or ""),
                ]
            )
        if category == "task_experience":
            return f"task_experience:{payload.get('status') or payload.get('failure_type') or 'general'}"
        if category == "user_preference":
            return "user_preference:general"
        return f"{category}:general"

    def _distill_group(
        self,
        category: str,
        scope_key: str,
        members: list[Any],
    ) -> MemoryDistillation | None:
        structured = self._merge_structured(category, members)
        support = len(members)
        summary = self._summarize(category, scope_key, members, structured)
        summary = summary[:MAX_DISTILLED_SUMMARY_CHARS]
        verdict = (
            self._policy(summary, structured)
            if callable(self._policy)
            else classify_distillation_payload(summary, structured)
        )
        if not verdict.allowed:
            return None
        return new_distillation(
            category=category,
            scope_key=scope_key,
            distilled_summary=summary,
            structured=structured,
            source_memory_ids=tuple(_entry_id(member) for member in members),
            support_count=support,
        )

    def _merge_structured(self, category: str, members: list[Any]) -> dict:
        payloads = [_entry_payload(member) for member in members]
        if category == "field_convention":
            fields: dict[str, list[str]] = {}
            for payload in payloads:
                for key, value in payload.items():
                    if value not in (None, ""):
                        fields.setdefault(str(key), []).append(str(value))
            return {
                "fields": {key: sorted(set(values)) for key, values in sorted(fields.items())},
                "support": len(members),
            }
        if category == "validation_pitfall":
            return {
                "pitfall_type": str(payloads[0].get("failure_kind") or "general") if payloads else "general",
                "messages": sorted({str(payload.get("message")) for payload in payloads if payload.get("message")}),
                "support": len(members),
            }
        if category == "model_experience":
            return _merge_model_experience(payloads, len(members))
        if category == "user_preference":
            return {
                "statements": sorted({str(payload.get("preference")) for payload in payloads if payload.get("preference")}),
                "support": len(members),
            }
        if category == "task_experience":
            counts: dict[str, int] = {}
            for payload in payloads:
                tag = str(payload.get("status") or payload.get("failure_type") or payload.get("package") or "general")
                counts[tag] = counts.get(tag, 0) + 1
            return {"outcome_tags": counts, "support": len(members)}
        return {"support": len(members)}

    def _summarize(
        self,
        category: str,
        scope_key: str,
        members: list[Any],
        structured: dict,
    ) -> str:
        if self._llm_factory is not None:
            try:
                raw = self._llm_factory().complete(
                    system_prompt=DISTILL_SYS,
                    user_prompt=build_distill_prompt(category, scope_key, members, structured),
                    stream=False,
                )
                text = " ".join(str(raw or "").split())
                if text:
                    return text
            except Exception:
                pass
        return _template_summary(category, structured)


def build_distill_prompt(category: str, scope_key: str, members: list[Any], structured: dict) -> str:
    return json.dumps(
        {
            "category": category,
            "scope_key": scope_key,
            "structured": structured,
            "source_summaries": [_entry_summary(member) for member in members],
            "instruction": "Only rewrite facts present above into one concise sentence. Do not add metrics or task ids.",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _merge_model_experience(payloads: list[dict[str, Any]], support: int) -> dict:
    metrics = {}
    for metric in ("ks", "auc", "psi"):
        values = [float(payload[metric]) for payload in payloads if payload.get(metric) is not None]
        if values:
            metrics[metric] = {"min": min(values), "max": max(values)}
    return {
        "model_name": _first_present(payloads, "model_name"),
        "scopes": sorted({str(payload.get("scope")) for payload in payloads if payload.get("scope")}),
        "channels": sorted({str(payload.get("channel")) for payload in payloads if payload.get("channel")}),
        "metric_ranges": metrics,
        "source_task_ids": sorted({str(payload.get("source_task_id")) for payload in payloads if payload.get("source_task_id")}),
        "support": support,
    }


def _template_summary(category: str, structured: dict) -> str:
    if category == "field_convention":
        parts = [
            f"{field} 常见取值: {', '.join(values)}"
            for field, values in structured.get("fields", {}).items()
        ]
        return "字段口径经验：" + "；".join(parts)
    if category == "validation_pitfall":
        return f"验证坑点经验：{structured.get('pitfall_type', 'general')} 类问题重复出现。"
    if category == "model_experience":
        model = structured.get("model_name") or "历史模型"
        return f"模型经验：{model} 在相近范围内已有 {structured.get('support', 0)} 条历史记录。"
    if category == "user_preference":
        statements = structured.get("statements") or []
        return "用户偏好经验：" + ("；".join(statements[:3]) if statements else "有重复偏好记录。")
    if category == "task_experience":
        return f"任务经验：{structured.get('outcome_tags', {})}"
    return f"{category} 经验已归并。"


def _first_present(payloads: list[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _entry_id(entry: Any) -> str:
    return str(entry.get("id") if isinstance(entry, dict) else getattr(entry, "id"))


def _entry_category(entry: Any) -> str:
    return str(entry.get("memory_type") if isinstance(entry, dict) else getattr(entry, "memory_type"))


def _entry_payload(entry: Any) -> dict[str, Any]:
    payload = entry.get("payload") if isinstance(entry, dict) else getattr(entry, "payload")
    return payload if isinstance(payload, dict) else {}


def _entry_summary(entry: Any) -> str:
    return str(entry.get("summary") if isinstance(entry, dict) else getattr(entry, "summary"))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CONFIDENCE_THRESHOLDS",
    "DISTILLATION_STATUSES",
    "MAX_DISTILLED_SUMMARY_CHARS",
    "MemoryDistillation",
    "DISTILL_SYS",
    "DistillationEngine",
    "build_distill_prompt",
    "confidence_from_support",
    "new_distillation",
    "normalize_distillation_confidence",
    "normalize_distillation_status",
]

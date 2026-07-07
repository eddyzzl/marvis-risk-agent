from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from typing import Any, Iterable

from marvis.agent_memory.models import MEMORY_TYPES


METRIC_FIELDS = ("ks", "auc", "psi")
MODEL_PAYLOAD_FIELDS = (
    "ks",
    "auc",
    "psi",
    "month",
    "channel",
    "model_name",
    "model_version",
    "scope",
    "important_feature_sources",
)
GENERAL_PAYLOAD_FIELDS = (
    "target_col",
    "score_col",
    "split_col",
    "time_col",
    "channel_col",
    "status",
    "failure_kind",
)
LOW_CONFIDENCE_VALUES = {"low", "very_low", "rejected"}
RECENCY_HALF_LIFE_DAYS = 90
RECENCY_STALE_DAYS = 365
RECENCY_RECENT_BONUS = 10
RECENCY_STALE_PENALTY = -10
# Raw-memory recall ceiling, per memory_type: model_experience is the highest
# volume, highest-value category (comparison anchors), so it gets its own
# generous window instead of being crowded out by preference/convention
# entries once the store grows past a couple hundred rows.
RAW_RECALL_LIMIT_PER_KIND = 200
RAW_RECALL_LIMIT_OTHER_KINDS = 200
MODEL_FAMILY_PATTERNS = (
    ("a_card", (r"\ba\s*card\b", r"a卡")),
    ("b_card", (r"\bb\s*card\b", r"b卡")),
    ("c_card", (r"\bc\s*card\b", r"c卡")),
    ("amount", (r"\bamount\b", r"额度")),
    ("rate", (r"\brate\b", r"利率")),
    ("pre_screening", (r"\bpre[-_\s]?screening\b", r"前筛")),
)


@dataclass(frozen=True)
class MemoryQuery:
    model_name: str | None = None
    scope: str | None = None
    channel: str | None = None
    month: str | None = None
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemorySearchResult:
    entry: Any
    confidence: str
    score: int
    match_reason: str
    context_packet: dict[str, Any]


def normalize_model_family(value: str | None) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    for family, patterns in MODEL_FAMILY_PATTERNS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return family
    return None


def retrieve_relevant_memories(
    entries: Iterable[Any],
    query: MemoryQuery,
    limit: int = 5,
) -> list[MemorySearchResult]:
    results: list[MemorySearchResult] = []
    for entry in entries:
        record = _MemoryRecord(entry)
        if not _is_usable_memory(record):
            continue
        score, reasons = (
            _score_record(record, query)
            if record.memory_type == "model_experience"
            else _score_general_record(record, query)
        )
        if score <= 0:
            continue

        confidence = _score_confidence(score)
        if confidence == "low":
            # Symmetric with the distillation-side policy (retrieve_with_distillations
            # already skips confidence == 'low' distillations): a single generic
            # keyword hit is not enough signal to spend a weak model's limited
            # prompt budget on an otherwise-unrelated raw memory.
            continue
        packet = _context_packet(record, confidence, ", ".join(reasons))
        results.append(
            MemorySearchResult(
                entry=entry,
                confidence=confidence,
                score=score,
                match_reason=packet["match_reason"],
                context_packet=packet,
            )
        )

    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def retrieve_with_distillations(
    store,
    query_context: dict[str, Any] | MemoryQuery,
    *,
    limit: int = 6,
    raw_quota: int | None = None,
) -> list[dict[str, Any]]:
    query = _query_from_context(query_context)
    context = _query_context_dict(query)
    packets: list[dict[str, Any]] = []
    covered_source_ids: set[str] = set()
    distillation_limit = limit if raw_quota is None else max(0, limit - raw_quota)
    distillations = store.search_distillations(context, active_only=True, limit=limit)
    if distillation_limit > 0:
        for distillation in distillations:
            if distillation.confidence == "low":
                continue
            packet = _distillation_packet(distillation)
            packets.append(packet)
            covered_source_ids.update(_source_memory_ids(packet))
            if len(packets) >= distillation_limit:
                break

    remaining = limit - len(packets)
    if raw_quota is not None:
        remaining = min(remaining, raw_quota)
    if remaining <= 0:
        return packets[:limit]
    raw_results = retrieve_relevant_memories(
        _recall_raw_entries(store),
        query,
        limit=max(remaining * 4, remaining),
    )
    raw_results = _dedupe_raw_results(
        raw_results,
        covered_source_ids=covered_source_ids,
        limit=remaining,
    )
    packets.extend(_raw_packet(result.context_packet) for result in raw_results)
    return packets[:limit]


def _recall_raw_entries(store) -> list[Any]:
    """Pull candidate raw entries for scoring, targeted by memory_type instead
    of a single "most recent 200 across all types" scan. Without this,
    model_experience anchors (and any other type) silently fall out of the
    recall window as unrelated preference/convention entries accumulate."""
    entries: list[Any] = []
    seen_ids: set[str] = set()
    for memory_type in MEMORY_TYPES:
        per_kind_limit = (
            RAW_RECALL_LIMIT_PER_KIND
            if memory_type == "model_experience"
            else RAW_RECALL_LIMIT_OTHER_KINDS
        )
        for entry in store.list_entries(memory_type=memory_type, limit=per_kind_limit):
            entry_id = str(entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None))
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append(entry)
    return entries


def compare_model_experience(
    current_payload: dict[str, Any],
    history_entries: Iterable[Any],
    limit: int = 8,
) -> dict[str, Any]:
    query = MemoryQuery(
        model_name=_optional_text(current_payload.get("model_name")),
        scope=_optional_text(current_payload.get("scope")),
        channel=_optional_text(current_payload.get("channel")),
        month=_optional_text(current_payload.get("month")),
    )
    packets: list[dict[str, Any]] = []
    dimensions = {
        "models": set(),
        "months": set(),
        "channels": set(),
        "metrics": set(),
    }

    for result in retrieve_relevant_memories(history_entries, query, limit=limit):
        record = _MemoryRecord(result.entry)
        if not _is_comparable_model_experience(current_payload, record.payload):
            continue

        payload = record.payload
        for field_name, dimension_name in (
            ("model_name", "models"),
            ("month", "months"),
            ("channel", "channels"),
        ):
            value = payload.get(field_name)
            if value not in (None, ""):
                dimensions[dimension_name].add(str(value))
        for metric in METRIC_FIELDS:
            if payload.get(metric) is not None:
                dimensions["metrics"].add(metric)

        reason = _comparison_reason(current_payload, payload)
        confidence = "high" if result.confidence == "high" else _record_confidence(record)
        packets.append(_context_packet(record, confidence, reason))

    packets = packets[:limit]
    return {
        "current": _current_summary(current_payload),
        "dimensions": {
            "models": sorted(dimensions["models"]),
            "months": sorted(dimensions["months"]),
            "channels": sorted(dimensions["channels"]),
            "metrics": [
                metric for metric in METRIC_FIELDS if metric in dimensions["metrics"]
            ],
        },
        "context_packets": packets,
        "usage": (
            "memory context is bounded and may be used only for explanation, "
            "risk reminders, and historical comparison; deterministic metrics "
            "must come from platform validation results"
        ),
    }


class _MemoryRecord:
    def __init__(self, entry: Any) -> None:
        self.entry = entry

    @property
    def memory_id(self) -> Any:
        return self._get("id")

    @property
    def memory_type(self) -> str:
        return str(self._get("memory_type") or "")

    @property
    def summary(self) -> str:
        return str(self._get("summary") or "")

    @property
    def payload(self) -> dict[str, Any]:
        payload = self._get("payload")
        return payload if isinstance(payload, dict) else {}

    @property
    def source_task_id(self) -> str | None:
        value = self._get("source_task_id") or self.payload.get("source_task_id")
        return str(value) if value not in (None, "") else None

    @property
    def confidence(self) -> str:
        return str(self._get("confidence") or "medium").strip().lower()

    @property
    def status(self) -> str:
        return str(self._get("status") or "active").strip().lower()

    @property
    def created_at(self) -> str | None:
        value = self._get("created_at")
        return str(value) if value not in (None, "") else None

    @property
    def age_days(self) -> int | None:
        return _age_in_days(self.created_at)

    def _get(self, field_name: str) -> Any:
        if isinstance(self.entry, dict):
            return self.entry.get(field_name)
        return getattr(self.entry, field_name, None)


def _is_usable_model_experience(record: _MemoryRecord) -> bool:
    if not _is_usable_memory(record):
        return False
    if record.memory_type != "model_experience":
        return False
    return True


def _is_usable_memory(record: _MemoryRecord) -> bool:
    if record.status != "active":
        return False
    return record.confidence not in LOW_CONFIDENCE_VALUES


def _score_record(record: _MemoryRecord, query: MemoryQuery) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    payload = record.payload

    if _same_text(query.model_name, payload.get("model_name")):
        score += 45
        reasons.append("exact model")
    elif _same_model_family(query.model_name, payload.get("model_name")):
        score += 25
        reasons.append("model family")
    elif _text_contains(payload.get("model_name"), query.model_name):
        score += 20
        reasons.append("model keyword")

    if _same_text(query.scope, payload.get("scope")):
        score += 25
        reasons.append("exact scope")
    elif _shared_scope_keywords(query.scope, payload.get("scope")):
        score += 15
        reasons.append("scope keyword")

    if _same_text(query.channel, payload.get("channel")):
        score += 15
        reasons.append("exact channel")
    if _same_text(query.month, payload.get("month")):
        score += 15
        reasons.append("exact month")

    for keyword in query.keywords:
        if _text_contains(record.summary, keyword) or _text_contains(
            payload.get("scope"), keyword
        ):
            score += 5
            reasons.append(f"keyword:{keyword}")

    recency_delta, recency_reason = _recency_bonus(record.age_days)
    score += recency_delta
    if recency_reason:
        reasons.append(recency_reason)

    return score, reasons


def _score_general_record(
    record: _MemoryRecord,
    query: MemoryQuery,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    searchable = f"{record.summary} {_payload_text(record.payload)}"
    if _same_model_family(query.model_name, searchable):
        score += 20
        reasons.append("model family")
    for keyword in query.keywords:
        if _text_contains(searchable, keyword):
            score += 15
            reasons.append(f"keyword:{keyword}")
    if query.scope and _text_contains(searchable, query.scope):
        score += 10
        reasons.append("scope keyword")
    if query.channel and _text_contains(searchable, query.channel):
        score += 10
        reasons.append("channel keyword")

    recency_delta, recency_reason = _recency_bonus(record.age_days)
    score += recency_delta
    if recency_reason:
        reasons.append(recency_reason)

    return score, reasons


def _age_in_days(created_at: str | None) -> int | None:
    if not created_at:
        return None
    try:
        parsed = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - parsed
    return max(0, delta.days)


def _recency_bonus(age_days: int | None) -> tuple[int, str | None]:
    # Deterministic recency term: entries observed within the half-life window
    # get a small boost, entries older than the stale threshold get a small
    # penalty, everything in between (and entries with unknown age) is neutral.
    if age_days is None:
        return 0, None
    if age_days <= RECENCY_HALF_LIFE_DAYS:
        return RECENCY_RECENT_BONUS, "recent"
    if age_days > RECENCY_STALE_DAYS:
        return RECENCY_STALE_PENALTY, "stale"
    return 0, None


def _score_confidence(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def _record_confidence(record: _MemoryRecord) -> str:
    return "high" if record.confidence == "high" else "medium"


def _context_packet(
    record: _MemoryRecord,
    confidence: str,
    match_reason: str,
) -> dict[str, Any]:
    return {
        "id": record.memory_id,
        "memory_type": record.memory_type,
        "summary": record.summary,
        "payload": {
            field_name: record.payload[field_name]
            for field_name in (*MODEL_PAYLOAD_FIELDS, *GENERAL_PAYLOAD_FIELDS)
            if field_name in record.payload
        },
        "source_task_id": record.source_task_id,
        "confidence": confidence,
        "match_reason": match_reason,
        "observed_at": record.created_at,
        "age_days": record.age_days,
    }


def _distillation_packet(distillation) -> dict[str, Any]:
    observed_at = str(getattr(distillation, "updated_at", "") or "") or None
    return {
        "kind": "distillation",
        "id": distillation.id,
        "memory_type": distillation.category,
        "category": distillation.category,
        "summary": distillation.distilled_summary,
        "payload": distillation.structured,
        "confidence": distillation.confidence,
        "support_count": distillation.support_count,
        "source_memory_ids": list(distillation.source_memory_ids),
        "source_task_id": None,
        "match_reason": "distilled memory",
        "observed_at": observed_at,
        "age_days": _age_in_days(observed_at),
    }


def _raw_packet(packet: dict[str, Any]) -> dict[str, Any]:
    out = dict(packet)
    out["kind"] = "raw"
    return out


def _source_memory_ids(packet: dict[str, Any]) -> set[str]:
    ids = packet.get("source_memory_ids")
    if not isinstance(ids, (list, tuple, set)):
        return set()
    return {str(item) for item in ids if str(item).strip()}


def _dedupe_raw_results(
    raw_results: Iterable[MemorySearchResult],
    *,
    covered_source_ids: set[str],
    limit: int,
) -> list[MemorySearchResult]:
    selected: list[MemorySearchResult] = []
    seen_fingerprints: set[tuple[str, str, str]] = set()
    for result in raw_results:
        packet = result.context_packet
        raw_id = str(packet.get("id") or "")
        if raw_id and raw_id in covered_source_ids:
            continue
        fingerprint = _raw_memory_fingerprint(packet)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected


def _raw_memory_fingerprint(packet: dict[str, Any]) -> tuple[str, str, str]:
    payload = packet.get("payload")
    return (
        str(packet.get("memory_type") or ""),
        _normalize_text(packet.get("summary")),
        json.dumps(
            payload if isinstance(payload, dict) else {},
            default=str,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _query_from_context(query_context: dict[str, Any] | MemoryQuery) -> MemoryQuery:
    if isinstance(query_context, MemoryQuery):
        return query_context
    return MemoryQuery(
        model_name=_optional_text(query_context.get("model_name")),
        scope=_optional_text(query_context.get("scope")),
        channel=_optional_text(query_context.get("channel")),
        month=_optional_text(query_context.get("month")),
        keywords=tuple(str(item) for item in query_context.get("keywords", ()) if str(item).strip()),
    )


def _query_context_dict(query: MemoryQuery) -> dict[str, Any]:
    return {
        "model_name": query.model_name,
        "scope": query.scope,
        "channel": query.channel,
        "month": query.month,
        "keywords": list(query.keywords),
    }


def _current_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": payload.get("model_name"),
        "model_family": normalize_model_family(payload.get("model_name")),
        "model_version": payload.get("model_version"),
        "scope": payload.get("scope"),
        "month": payload.get("month"),
        "channel": payload.get("channel"),
        "metrics": {
            metric: payload[metric] for metric in METRIC_FIELDS if metric in payload
        },
    }


def _comparison_reason(
    current_payload: dict[str, Any], history_payload: dict[str, Any]
) -> str:
    reasons: list[str] = []
    for current_field, history_field, label in (
        ("model_name", "model_name", "same model"),
        ("scope", "scope", "same scope"),
        ("channel", "channel", "same channel"),
        ("month", "month", "same month"),
    ):
        if _same_text(
            current_payload.get(current_field), history_payload.get(history_field)
        ):
            reasons.append(label)
    if _same_model_family(
        current_payload.get("model_name"), history_payload.get("model_name")
    ):
        reasons.append("same model family")
    return ", ".join(dict.fromkeys(reasons)) or "comparison context"


def _is_comparable_model_experience(
    current_payload: dict[str, Any],
    history_payload: dict[str, Any],
) -> bool:
    return (
        _same_text(current_payload.get("model_name"), history_payload.get("model_name"))
        or _same_model_family(
            current_payload.get("model_name"), history_payload.get("model_name")
        )
        or _same_text(current_payload.get("scope"), history_payload.get("scope"))
        or _shared_scope_keywords(
            current_payload.get("scope"), history_payload.get("scope")
        )
    )


def _same_model_family(left: Any, right: Any) -> bool:
    left_family = normalize_model_family(left)
    right_family = normalize_model_family(right)
    return left_family is not None and left_family == right_family


def _same_text(left: Any, right: Any) -> bool:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    return bool(normalized_left and normalized_left == normalized_right)


def _text_contains(container: Any, needle: Any) -> bool:
    normalized_container = _normalize_text(container)
    normalized_needle = _normalize_text(needle)
    return bool(
        normalized_container
        and normalized_needle
        and (
            normalized_needle in normalized_container
            or normalized_container in normalized_needle
        )
    )


def _shared_scope_keywords(left: Any, right: Any) -> bool:
    left_keywords = _scope_keywords(left)
    right_keywords = _scope_keywords(right)
    return bool(left_keywords and right_keywords and left_keywords & right_keywords)


def _scope_keywords(value: Any) -> set[str]:
    text = _normalize_text(value)
    keywords = {
        token for token in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", text) if token
    }
    for marker in ("mob3", "mob6", "mob12", "贷前", "贷中", "申请", "客群"):
        if marker.lower() in text:
            keywords.add(marker.lower())
    family = normalize_model_family(text)
    if family:
        keywords.add(family)
    return keywords


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _payload_text(payload: dict[str, Any]) -> str:
    return " ".join(str(value) for value in payload.values() if value not in (None, ""))

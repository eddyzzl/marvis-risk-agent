from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from riskmodel_checker.agent_memory.models import (
    MODEL_EXPERIENCE_REQUIRED_FIELDS,
    MemoryCandidate,
)
from riskmodel_checker.agent_memory.policy import classify_memory_candidate


PITFALL_KINDS = {"notebook", "pmml", "field", "execution", "report"}


def extract_model_experience(result: dict[str, Any]) -> MemoryCandidate | None:
    payload = _model_experience_payload(result)
    if payload is None:
        return None

    candidate = MemoryCandidate(
        memory_type="model_experience",
        summary=(
            f"{payload['model_name']}{payload['model_version']}在{payload['month']}"
            f"{payload['channel']}渠道KS为{payload['ks']}，AUC为{payload['auc']}，"
            f"PSI为{payload['psi']}。"
        ),
        payload=payload,
        source_task_id=str(payload["source_task_id"]),
        confidence="high",
        reason="structured validation result",
    )
    return _allow(candidate)


def extract_validation_pitfall(result: dict[str, Any]) -> list[MemoryCandidate]:
    task_id = _first_text(result, "task_id", "source_task_id")
    candidates: list[MemoryCandidate] = []
    for failure in _iter_failures(result.get("failures")):
        kind = _pitfall_kind(failure)
        if kind not in PITFALL_KINDS:
            continue
        message = _failure_message(failure)
        if not message:
            continue
        candidate = MemoryCandidate(
            memory_type="validation_pitfall",
            summary=f"{kind} validation pitfall: {message}",
            payload={"failure_kind": kind, "message": message},
            source_task_id=task_id,
            confidence="medium",
            reason="structured validation failure",
        )
        allowed = _allow(candidate)
        if allowed is not None:
            candidates.append(allowed)
    return candidates


def extract_task_experience(task: dict[str, Any]) -> MemoryCandidate | None:
    status = str(task.get("status") or "").strip().lower()
    if status not in {"completed", "failed"}:
        return None
    summary = _clean_text(task.get("summary"))
    if not summary:
        return None

    candidate = MemoryCandidate(
        memory_type="task_experience",
        summary=summary,
        payload={"status": status},
        source_task_id=_first_text(task, "task_id", "source_task_id"),
        confidence="medium",
        reason="task lifecycle summary",
    )
    return _allow(candidate)


def extract_field_convention(task: dict[str, Any]) -> MemoryCandidate | None:
    payload = {
        field_name: _clean_text(task.get(field_name))
        for field_name in (
            "target_col",
            "score_col",
            "split_col",
            "time_col",
            "channel_col",
        )
        if _clean_text(task.get(field_name))
    }
    if not payload:
        return None
    summary_parts = [
        f"{label}={payload[field_name]}"
        for field_name, label in (
            ("target_col", "目标字段"),
            ("score_col", "分数字段"),
            ("split_col", "样本分组字段"),
            ("time_col", "时间字段"),
            ("channel_col", "渠道字段"),
        )
        if field_name in payload
    ]
    candidate = MemoryCandidate(
        memory_type="field_convention",
        summary="字段口径：" + "，".join(summary_parts),
        payload=payload,
        source_task_id=_first_text(task, "task_id", "source_task_id"),
        confidence="medium",
        reason="task field settings",
    )
    return _allow(candidate)


def extract_user_preference(message: dict[str, Any]) -> MemoryCandidate | None:
    text = _clean_text(message.get("text") or message.get("content"))
    if not text or _mentions_reserved_skill_runtime(text):
        return None

    preference = _explicit_preference(text)
    if not preference:
        return None

    candidate = MemoryCandidate(
        memory_type="user_preference",
        summary=preference,
        payload={"preference": preference},
        source_message_id=_first_text(message, "message_id", "id"),
        confidence="high",
        reason="explicit user memory instruction",
    )
    return _allow(candidate)


def extract_memory_candidates(
    *,
    task_result: dict[str, Any] | None = None,
    messages: Iterable[dict[str, Any]] | None = None,
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []

    if task_result:
        model_experience = extract_model_experience(task_result)
        if model_experience is not None:
            candidates.append(model_experience)
        candidates.extend(extract_validation_pitfall(task_result))
        task_experience = extract_task_experience(task_result)
        if task_experience is not None:
            candidates.append(task_experience)
        field_convention = extract_field_convention(task_result)
        if field_convention is not None:
            candidates.append(field_convention)

    for message in messages or ():
        preference = extract_user_preference(message)
        if preference is not None:
            candidates.append(preference)

    return candidates


def _model_experience_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    payload = {
        "ks": _first_value(result, metrics, "ks"),
        "auc": _first_value(result, metrics, "auc"),
        "psi": _first_value(result, metrics, "psi"),
        "month": _first_value(result, metrics, "month"),
        "channel": _first_value(result, metrics, "channel"),
        "model_name": _first_value(result, metrics, "model_name"),
        "model_version": _first_value(result, metrics, "model_version"),
        "scope": _first_value(result, metrics, "scope"),
        "source_task_id": _first_value(result, metrics, "source_task_id", "task_id"),
        "important_feature_sources": _first_value(
            result, metrics, "important_feature_sources", "feature_sources"
        ),
    }
    if any(_is_missing(payload[field]) for field in MODEL_EXPERIENCE_REQUIRED_FIELDS):
        return None
    return payload


def _iter_failures(failures: Any) -> Iterable[Any]:
    if isinstance(failures, list | tuple):
        return failures
    if failures:
        return (failures,)
    return ()


def _pitfall_kind(failure: Any) -> str:
    if isinstance(failure, dict):
        kind = str(failure.get("kind") or failure.get("type") or "").strip().lower()
        if kind:
            return kind
        text = _failure_message(failure).lower()
    else:
        text = str(failure or "").lower()

    if "notebook" in text or "rmc_" in text:
        return "notebook"
    if "pmml" in text:
        return "pmml"
    if "field" in text or "column" in text or "字段" in text:
        return "field"
    if "execution" in text or "timeout" in text or "执行" in text:
        return "execution"
    if "report" in text or "报告" in text:
        return "report"
    return ""


def _failure_message(failure: Any) -> str:
    if isinstance(failure, dict):
        return _clean_text(
            failure.get("message")
            or failure.get("summary")
            or failure.get("error")
            or failure.get("reason")
        )
    return _clean_text(failure)


def _explicit_preference(text: str) -> str:
    markers = ("请记住：", "请记住:", "记住：", "记住:", "纠正一下：", "纠正一下:")
    for marker in markers:
        if text.startswith(marker):
            return text[len(marker) :].strip()
    return ""


def _mentions_reserved_skill_runtime(text: str) -> bool:
    lowered = text.lower()
    return "skill" in lowered or "runtime" in lowered


def _first_value(
    primary: dict[str, Any], secondary: dict[str, Any], *keys: str
) -> Any:
    for key in keys:
        for source in (primary, secondary):
            if key in source and not _is_missing(source[key]):
                return source[key]
    return None


def _first_text(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if not _is_missing(value):
            return str(value)
    return None


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _allow(candidate: MemoryCandidate) -> MemoryCandidate | None:
    decision = classify_memory_candidate(candidate)
    if decision.allowed:
        return candidate
    return None

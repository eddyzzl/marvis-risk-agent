from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from marvis.agent_memory.models import (
    FEATURE_EXPERIENCE_REQUIRED_FIELDS,
    JOIN_EXPERIENCE_REQUIRED_FIELDS,
    MODEL_EXPERIENCE_REQUIRED_FIELDS,
    RISK_ANALYSIS_EXPERIENCE_REQUIRED_FIELDS,
    STRATEGY_EXPERIENCE_REQUIRED_FIELDS,
    MemoryCandidate,
)
from marvis.agent_memory.policy import classify_memory_candidate


PITFALL_KINDS = {"notebook", "pmml", "field", "execution", "report"}
USER_PREFERENCE_MAX_CHARS = 200
_RISK_ANALYSIS_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "raw_row",
        "raw_rows",
        "raw_data",
        "raw_records",
        "customer_row",
        "customer_rows",
        "report_text",
        "report_content",
        "full_report",
        "upload_path",
        "source_path",
    }
)
_RISK_ANALYSIS_PII_KEY_PATTERN = re.compile(
    r"(?:customer|cust|client|user)[_-]?(?:id|no|number)$|"
    r"(?:mobile|phone|id[_-]?card)$|(?:客户号|客户编号|用户编号|手机号|身份证号?)$",
    re.IGNORECASE,
)


def extract_model_experience(result: dict[str, Any]) -> MemoryCandidate | None:
    payload = _model_experience_payload(result)
    if payload is None:
        return None

    candidate = MemoryCandidate(
        memory_type="model_experience",
        summary=(
            f"{payload['model_name']}{payload['model_version']}在{payload['month']}"
            f"{payload['channel']}渠道KS为{_metric_display(payload['ks'])}，"
            f"AUC为{_metric_display(payload['auc'])}，"
            f"PSI为{_metric_display(payload['psi'])}。"
        ),
        payload=payload,
        source_task_id=str(payload["source_task_id"]),
        confidence="high",
        reason="structured validation result",
    )
    return _allow(candidate)


def extract_join_experience(result: dict[str, Any]) -> MemoryCandidate | None:
    payload = _join_experience_payload(result)
    if payload is None:
        return None

    candidate = MemoryCandidate(
        memory_type="join_experience",
        summary=(
            f"{payload['scope']}拼接{payload['feature_table_count']}张特征表，"
            f"命中率{payload['match_rate']}，样本行数{payload['anchor_rows']}"
            f"→{payload['joined_rows']}。"
        ),
        payload=payload,
        source_task_id=str(payload["source_task_id"]),
        confidence="high",
        reason="structured join execution result",
    )
    return _allow(candidate)


def extract_feature_experience(result: dict[str, Any]) -> MemoryCandidate | None:
    raw_evidence = result.get("recommendation_evidence")
    evidence = dict(raw_evidence) if isinstance(raw_evidence, dict) else {}
    payload = {
        "feature_count": _first_value(result, {}, "feature_count"),
        "recommended_features": list(result.get("recommended_features") or []),
        "avoid_features": list(result.get("avoid_features") or []),
        "recommendation_confidence": str(
            result.get("recommendation_confidence") or ""
        ).strip().lower(),
        "recommendation_evidence": evidence,
        "target_col": _first_value(result, {}, "target_col"),
        "scope": _first_value(result, {}, "scope"),
        "source_task_id": _first_value(result, {}, "source_task_id", "task_id"),
    }
    if any(
        _is_missing(payload[field])
        for field in FEATURE_EXPERIENCE_REQUIRED_FIELDS
        if field not in {"recommended_features", "avoid_features"}
    ):
        return None
    target_col = str(payload["target_col"] or "").strip()
    if str(payload["scope"] or "").strip() != f"feature:target={target_col}":
        return None
    recommended_set = {
        str(feature or "").strip()
        for feature in payload["recommended_features"]
        if str(feature or "").strip()
    }
    avoid_set = {
        str(feature or "").strip()
        for feature in payload["avoid_features"]
        if str(feature or "").strip()
    }
    conflicts = recommended_set & avoid_set
    payload["recommended_features"] = sorted(recommended_set - conflicts)
    payload["avoid_features"] = sorted(avoid_set - conflicts)
    actionable_features = set(payload["recommended_features"]) | set(
        payload["avoid_features"]
    )
    if not actionable_features:
        return None
    payload["recommendation_evidence"] = {
        feature: evidence[feature]
        for feature in sorted(actionable_features)
        if feature in evidence and evidence[feature]
    }
    recommended = "、".join(str(item) for item in payload["recommended_features"][:8]) or "无明确推荐"
    avoid = "、".join(str(item) for item in payload["avoid_features"][:8]) or "未识别明显坑点"
    has_actionable_advice = bool(actionable_features)
    has_governed_evidence = (
        set(payload["recommendation_evidence"]) == actionable_features
    )
    requested_confidence = payload["recommendation_confidence"]
    if has_governed_evidence and requested_confidence == "high":
        confidence = "high"
    elif has_actionable_advice:
        confidence = "medium"
    else:
        confidence = "low"
    candidate = MemoryCandidate(
        memory_type="feature_experience",
        summary=(
            f"{payload['scope']}分析 {payload['feature_count']} 个特征；"
            f"推荐 {recommended}；谨慎使用 {avoid}。"
        ),
        payload=payload,
        source_task_id=str(payload["source_task_id"]),
        confidence=confidence,
        reason=(
            "structured feature analysis result with governed recommendation evidence"
            if has_governed_evidence
            else "structured feature analysis result without governed recommendation evidence"
        ),
    )
    return _allow(candidate)


def extract_strategy_experience(result: dict[str, Any]) -> MemoryCandidate | None:
    payload = _strategy_experience_payload(result)
    if payload is None:
        return None
    profit_summary = (
        f"预期利润{payload['expected_profit']}"
        if payload["expected_profit"] is not None
        else "预期利润未计算"
    )

    candidate = MemoryCandidate(
        memory_type="strategy_experience",
        summary=(
            f"{payload['scope']}采纳{payload['strategy_type']}策略，"
            f"{payload['cutoff_summary']}，审批率{payload['approval_rate']}，"
            f"通过坏率{payload['approved_bad_rate']}，{profit_summary}。"
        ),
        payload=payload,
        source_task_id=str(payload["source_task_id"]),
        confidence="high",
        reason="structured strategy adoption result",
    )
    return _allow(candidate)


def extract_risk_analysis_experience(result: dict[str, Any]) -> MemoryCandidate | None:
    payload = _risk_analysis_experience_payload(result)
    if payload is None:
        return None
    scope = payload["product_scope"]
    scope_text = "、".join(scope) if isinstance(scope, list) else scope
    metric_text = "，".join(
        f"{name}={_metric_display(value)}"
        for name, value in list(payload["headline_metrics"].items())[:4]
    )
    try:
        candidate = MemoryCandidate(
            memory_type="risk_analysis_experience",
            summary=(
                f"{payload['as_of_period']} {scope_text}完成"
                f"{payload['analysis_kind']}分析，{metric_text}。"
            ),
            payload=payload,
            source_task_id=str(payload["source_task_id"]),
            confidence="high",
            reason="structured risk analysis report result",
        )
    except (TypeError, ValueError):
        return None
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

    preference = _truncate_text(_explicit_preference(text), USER_PREFERENCE_MAX_CHARS)
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


# MEM-9: capture_user_preference_memory() (api_support.py) previously called
# extract_user_preference() and, on None, silently dropped the turn with no
# feedback to the user -- indistinguishable from "the user never asked to
# remember anything" in the first place. classify_user_preference_capture()
# exposes *why* a marked "please remember" instruction did not get stored, so
# the caller can send a receipt only when the user actually invoked the
# explicit-memory contract (a marker was present) and it was declined.
USER_PREFERENCE_CAPTURED = "captured"
USER_PREFERENCE_NO_MARKER = "no_marker"
USER_PREFERENCE_RESERVED_TOPIC = "reserved_topic"
USER_PREFERENCE_POLICY_REJECTED = "policy_rejected"


def classify_user_preference_capture(message: dict[str, Any]) -> str:
    text = _clean_text(message.get("text") or message.get("content"))
    if not text:
        return USER_PREFERENCE_NO_MARKER
    if _mentions_reserved_skill_runtime(text):
        return (
            USER_PREFERENCE_RESERVED_TOPIC
            if _EXPLICIT_PREFERENCE_MARKER_PATTERN.search(text)
            else USER_PREFERENCE_NO_MARKER
        )
    preference = _truncate_text(_explicit_preference(text), USER_PREFERENCE_MAX_CHARS)
    if not preference:
        return USER_PREFERENCE_NO_MARKER
    candidate = MemoryCandidate(
        memory_type="user_preference",
        summary=preference,
        payload={"preference": preference},
        source_message_id=_first_text(message, "message_id", "id"),
        confidence="high",
        reason="explicit user memory instruction",
    )
    return (
        USER_PREFERENCE_CAPTURED
        if _allow(candidate) is not None
        else USER_PREFERENCE_POLICY_REJECTED
    )


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
        risk_analysis_experience = extract_risk_analysis_experience(task_result)
        if risk_analysis_experience is not None:
            candidates.append(risk_analysis_experience)
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


def _join_experience_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    payload = {
        "match_rate": _first_value(result, {}, "match_rate"),
        "anchor_rows": _first_value(result, {}, "anchor_rows"),
        "joined_rows": _first_value(result, {}, "joined_rows"),
        "feature_table_count": _first_value(result, {}, "feature_table_count"),
        "scope": _first_value(result, {}, "scope"),
        "source_task_id": _first_value(result, {}, "source_task_id", "task_id"),
    }
    if any(_is_missing(payload[field]) for field in JOIN_EXPERIENCE_REQUIRED_FIELDS):
        return None
    return payload


def _strategy_experience_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    payload = {
        "strategy_type": _first_value(result, {}, "strategy_type"),
        "cutoff_summary": _first_value(result, {}, "cutoff_summary"),
        "approval_rate": _first_value(result, {}, "approval_rate"),
        "approved_bad_rate": _first_value(result, {}, "approved_bad_rate"),
        "expected_profit": _first_value(result, {}, "expected_profit"),
        "scope": _first_value(result, {}, "scope"),
        "source_task_id": _first_value(result, {}, "source_task_id", "task_id"),
    }
    if any(_is_missing(payload[field]) for field in STRATEGY_EXPERIENCE_REQUIRED_FIELDS):
        return None
    return payload


def _risk_analysis_experience_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(result, dict) or _risk_analysis_source_has_forbidden_fields(result):
        return None
    report_file = _risk_analysis_report_file(result)
    if report_file is None:
        return None
    product_scope = _risk_product_scope(result.get("product_scope"))
    headline_metrics = _risk_headline_metrics(result.get("headline_metrics"))
    assumptions = _risk_text_list(
        result.get("assumptions"),
        limit=12,
        max_chars=200,
    )
    key_points = _risk_text_list(
        result.get("key_points"),
        limit=12,
        max_chars=240,
    )
    red_flags = _risk_red_flags(result.get("red_flags"))
    column_map = _risk_column_map(result.get("column_map"))
    if any(
        value is None
        for value in (
            product_scope,
            headline_metrics,
            assumptions,
            key_points,
            red_flags,
            column_map,
        )
    ):
        return None
    payload = {
        "analysis_kind": _clean_text(result.get("analysis_kind")),
        "source_task_id": _first_text(result, "source_task_id", "task_id"),
        "product_scope": product_scope,
        "as_of_period": _clean_text(result.get("as_of_period")),
        "headline_metrics": headline_metrics,
        "assumptions": assumptions,
        "key_points": key_points,
        "red_flags": red_flags,
        "column_map": column_map,
        "report_file": report_file,
    }
    if any(_is_missing(payload[field]) for field in RISK_ANALYSIS_EXPERIENCE_REQUIRED_FIELDS):
        return None
    return payload


def _risk_analysis_source_has_forbidden_fields(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            raw_key = str(key).strip()
            normalized_key = re.sub(r"[^a-z0-9_]+", "_", raw_key.lower()).strip("_")
            if normalized_key in _RISK_ANALYSIS_FORBIDDEN_SOURCE_KEYS:
                return True
            if _RISK_ANALYSIS_PII_KEY_PATTERN.search(raw_key) or _RISK_ANALYSIS_PII_KEY_PATTERN.search(
                normalized_key
            ):
                return True
            if _risk_analysis_source_has_forbidden_fields(nested):
                return True
        return False
    if isinstance(value, list | tuple):
        return any(_risk_analysis_source_has_forbidden_fields(item) for item in value)
    return False


def _risk_analysis_report_file(result: dict[str, Any]) -> str | None:
    explicit = _clean_text(result.get("report_file"))
    if explicit:
        posix = PurePosixPath(explicit)
        windows = PureWindowsPath(explicit)
        if (
            explicit in {".", ".."}
            or posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or posix.name != explicit
            or windows.name != explicit
        ):
            return None
        return explicit
    report_path = _clean_text(result.get("report_path"))
    if not report_path:
        return None
    basename = PurePosixPath(report_path.replace("\\", "/")).name
    return basename if basename not in {"", ".", ".."} else None


def _risk_product_scope(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        return _bounded_risk_text(_clean_text(value), 120) or None
    if not isinstance(value, list):
        return None
    products: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _clean_text(item):
            return None
        products.append(_bounded_risk_text(_clean_text(item), 80))
    return products[:8] or None


def _risk_headline_metrics(value: Any) -> dict[str, Any] | None:
    metrics: dict[str, Any] = {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        normalized_items: list[tuple[Any, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                return None
            name = _clean_text(item.get("name"))
            unit = _clean_text(item.get("unit"))
            if not name or "value" not in item:
                return None
            normalized_items.append((f"{name} [{unit}]" if unit else name, item["value"]))
        items = normalized_items
    else:
        return None
    for raw_name, raw_value in list(items)[:16]:
        if not isinstance(raw_name, str):
            return None
        name = _bounded_risk_text(_clean_text(raw_name), 64)
        if not name or isinstance(raw_value, dict | list | tuple | set):
            return None
        if not isinstance(raw_value, str | int | float | bool):
            return None
        value = (
            _bounded_risk_text(_clean_text(raw_value), 120)
            if isinstance(raw_value, str)
            else raw_value
        )
        if value == "":
            return None
        metrics[name] = value
    return metrics or None


def _risk_text_list(
    value: Any,
    *,
    limit: int,
    max_chars: int,
) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _clean_text(item):
            return None
        result.append(_bounded_risk_text(_clean_text(item), max_chars))
    return result[:limit]


def _risk_red_flags(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            flag = _clean_text(item)
        elif isinstance(item, dict):
            flag = _clean_text(
                item.get("code") or item.get("kind") or item.get("id") or item.get("message")
            )
        else:
            return None
        if not flag:
            return None
        result.append(_bounded_risk_text(flag, 80))
    return result[:12]


def _risk_column_map(value: Any) -> dict[str, str] | None:
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for canonical, source in value.items():
        if not isinstance(canonical, str) or not isinstance(source, str):
            return None
        canonical_text = _clean_text(canonical)
        source_text = _clean_text(source)
        if not canonical_text or not source_text:
            return None
        result[_bounded_risk_text(canonical_text, 80)] = _bounded_risk_text(
            source_text,
            120,
        )
    return result


def _bounded_risk_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


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
        "model_version": _first_value(result, metrics, "model_version") or "未标注",
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


# MEM-9: explicit user "remember this" triggers. Kept intentionally
# conservative -- these are markers the user must type themselves, no
# whole-message LLM judgment -- but widened beyond a hard text.startswith so
# a marker mid-sentence ("好的，请记住：...") is still captured, and beyond
# the original six literal strings to cover the other common phrasings users
# actually type ("记一下", "以后都/以后请/以后统一").
_EXPLICIT_PREFERENCE_MARKER_PATTERN = re.compile(
    r"(?:请记住|记住|记一下|纠正一下|以后都|以后请|以后统一)[：:，,]?\s*"
)


def _explicit_preference(text: str) -> str:
    match = _EXPLICIT_PREFERENCE_MARKER_PATTERN.search(text)
    if not match:
        return ""
    return text[match.end() :].strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


# MEM-9: the reserved skill/runtime veto used to fire on a bare substring
# match ('runtime' inside an unrelated lightgbm hyperparameter sentence was
# enough to silently drop the whole preference). Narrowed to word-boundary
# matching, and to require the message's *topic* to actually be about a
# skill/tool runtime -- a skill/runtime marker together with an execute/run/
# invoke marker -- rather than any message that merely mentions the word.
_SKILL_RUNTIME_TOPIC_PATTERN = re.compile(
    r"(?:\bskill\b|\bruntime\b|技能|运行时)", re.IGNORECASE
)
_SKILL_RUNTIME_ACTION_PATTERN = re.compile(
    r"(?:\brun\b|\bexecute\b|执行|运行|调用|触发)", re.IGNORECASE
)


def _mentions_reserved_skill_runtime(text: str) -> bool:
    return bool(_SKILL_RUNTIME_TOPIC_PATTERN.search(text)) and bool(
        _SKILL_RUNTIME_ACTION_PATTERN.search(text)
    )


def _first_value(
    primary: dict[str, Any], secondary: dict[str, Any], *keys: str
) -> Any:
    for key in keys:
        for source in (primary, secondary):
            if key in source and not _is_missing(source[key]):
                return source[key]
    return None


def _metric_display(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


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

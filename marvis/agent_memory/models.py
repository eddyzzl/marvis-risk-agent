from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


MEMORY_TYPES = (
    "user_preference",
    "field_convention",
    "validation_pitfall",
    "task_experience",
    "model_experience",
    "feature_experience",
    "join_experience",
    "strategy_experience",
    "risk_analysis_experience",
    "skill_experience_reserved",
)
MEMORY_STATUSES = ("active", "disabled", "deleted", "rejected")
MODEL_EXPERIENCE_REQUIRED_FIELDS = (
    "ks",
    "auc",
    "psi",
    "month",
    "channel",
    "model_name",
    "model_version",
    "scope",
    "source_task_id",
    "important_feature_sources",
)
FEATURE_EXPERIENCE_REQUIRED_FIELDS = (
    "feature_count",
    "recommended_features",
    "avoid_features",
    "scope",
    "source_task_id",
)
JOIN_EXPERIENCE_REQUIRED_FIELDS = (
    "match_rate",
    "anchor_rows",
    "joined_rows",
    "feature_table_count",
    "scope",
    "source_task_id",
)
STRATEGY_EXPERIENCE_REQUIRED_FIELDS = (
    "strategy_type",
    "cutoff_summary",
    "approval_rate",
    "approved_bad_rate",
    "scope",
    "source_task_id",
)
RISK_ANALYSIS_EXPERIENCE_FIELDS = (
    "analysis_kind",
    "source_task_id",
    "product_scope",
    "as_of_period",
    "headline_metrics",
    "assumptions",
    "key_points",
    "red_flags",
    "column_map",
    "report_file",
)
RISK_ANALYSIS_EXPERIENCE_REQUIRED_FIELDS = (
    "analysis_kind",
    "source_task_id",
    "product_scope",
    "as_of_period",
    "headline_metrics",
    "report_file",
)

_RISK_ANALYSIS_MAX_PRODUCTS = 8
_RISK_ANALYSIS_MAX_METRICS = 16
_RISK_ANALYSIS_MAX_ASSUMPTIONS = 12
_RISK_ANALYSIS_MAX_KEY_POINTS = 12
_RISK_ANALYSIS_MAX_RED_FLAGS = 12
_RISK_ANALYSIS_MAX_COLUMN_MAPPINGS = 32


def normalize_memory_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MEMORY_TYPES:
        raise ValueError(f"unsupported memory type: {value}")
    return normalized


def normalize_memory_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MEMORY_STATUSES:
        raise ValueError(f"unsupported memory status: {value}")
    return normalized


def validate_model_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in MODEL_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name))
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required model_experience fields: {joined}")
    return payload


def validate_join_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in JOIN_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name))
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required join_experience fields: {joined}")
    return payload


def validate_feature_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in FEATURE_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name)) and field_name not in {"recommended_features", "avoid_features"}
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required feature_experience fields: {joined}")
    for field_name in ("recommended_features", "avoid_features"):
        if not isinstance(payload.get(field_name), list):
            raise ValueError(f"feature_experience {field_name} must be a list")
    return payload


def validate_strategy_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in STRATEGY_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name))
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required strategy_experience fields: {joined}")
    return payload


def validate_risk_analysis_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in RISK_ANALYSIS_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name))
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required risk_analysis_experience fields: {joined}")

    _require_short_text(payload["analysis_kind"], "analysis_kind", max_chars=64)
    _require_short_text(payload["source_task_id"], "source_task_id", max_chars=128)
    _validate_product_scope(payload["product_scope"])
    _require_short_text(payload["as_of_period"], "as_of_period", max_chars=40)
    _validate_headline_metrics(payload["headline_metrics"])
    _validate_short_text_list(
        payload.get("assumptions"),
        "assumptions",
        max_items=_RISK_ANALYSIS_MAX_ASSUMPTIONS,
        max_chars=200,
    )
    _validate_short_text_list(
        payload.get("key_points"),
        "key_points",
        max_items=_RISK_ANALYSIS_MAX_KEY_POINTS,
        max_chars=240,
    )
    _validate_short_text_list(
        payload.get("red_flags"),
        "red_flags",
        max_items=_RISK_ANALYSIS_MAX_RED_FLAGS,
        max_chars=80,
    )
    _validate_column_map(payload.get("column_map"))
    _validate_report_file(payload["report_file"])
    return payload


def _validate_product_scope(value: Any) -> None:
    if isinstance(value, str):
        _require_short_text(value, "product_scope", max_chars=120)
        return
    if not isinstance(value, list) or not value or len(value) > _RISK_ANALYSIS_MAX_PRODUCTS:
        raise ValueError("product_scope must be a string or a non-empty short list")
    for item in value:
        _require_short_text(item, "product_scope item", max_chars=80)


def _validate_headline_metrics(value: Any) -> None:
    if not isinstance(value, dict) or not value or len(value) > _RISK_ANALYSIS_MAX_METRICS:
        raise ValueError("headline_metrics must be a non-empty bounded flat map")
    for name, metric in value.items():
        _require_short_text(name, "headline metric name", max_chars=64)
        if isinstance(metric, bool):
            continue
        if isinstance(metric, int | float):
            if isinstance(metric, float) and not math.isfinite(metric):
                raise ValueError("headline metric values must be finite")
            continue
        _require_short_text(metric, "headline metric value", max_chars=120)


def _validate_short_text_list(
    value: Any,
    field_name: str,
    *,
    max_items: int,
    max_chars: int,
) -> None:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{field_name} must be a bounded list")
    for item in value:
        _require_short_text(item, f"{field_name} item", max_chars=max_chars)


def _validate_column_map(value: Any) -> None:
    if not isinstance(value, dict) or len(value) > _RISK_ANALYSIS_MAX_COLUMN_MAPPINGS:
        raise ValueError("column_map must be a bounded flat map")
    for canonical, source in value.items():
        _require_short_text(canonical, "canonical column name", max_chars=80)
        _require_short_text(source, "source column name", max_chars=120)


def _validate_report_file(value: Any) -> None:
    name = _require_short_text(value, "report_file", max_chars=180)
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        name in {".", ".."}
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or posix.name != name
        or windows.name != name
    ):
        raise ValueError("report_file must be a basename, not a path")


def _require_short_text(value: Any, field_name: str, *, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = " ".join(value.split()).strip()
    if not normalized or len(normalized) > max_chars:
        raise ValueError(f"{field_name} must be non-empty and at most {max_chars} chars")
    return normalized


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    source_task_id: str | None = None
    source_message_id: str | None = None
    confidence: str = "medium"
    reason: str = ""

    def __post_init__(self) -> None:
        normalized_type = normalize_memory_type(self.memory_type)
        object.__setattr__(self, "memory_type", normalized_type)
        if normalized_type == "model_experience":
            validate_model_experience_payload(self.payload)
        elif normalized_type == "feature_experience":
            validate_feature_experience_payload(self.payload)
        elif normalized_type == "join_experience":
            validate_join_experience_payload(self.payload)
        elif normalized_type == "strategy_experience":
            validate_strategy_experience_payload(self.payload)
        elif normalized_type == "risk_analysis_experience":
            validate_risk_analysis_experience_payload(self.payload)

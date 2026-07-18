"""Deterministic natural-language routing for report-ready dataset analysis.

The parser only selects analysis sections and already-known columns.  It never
computes a metric and never invents a column: all numbers remain owned by the
``data_ops.profile_dataset`` tool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re


_SECTION_ORDER = (
    "overview",
    "target",
    "missing",
    "distribution",
    "correlation",
)
_GENERAL_HINTS = (
    "分析这份样本",
    "分析当前样本",
    "样本分析",
    "数据概况",
    "数据概览",
    "数据画像",
    "profile dataset",
    "dataset profile",
)
_TARGET_HINTS = (
    "目标分布",
    "target分布",
    "target 分布",
    "target distribution",
    "标签分布",
)
_MISSING_HINTS = ("缺失", "空值", "missing", "null rate", "null率")
_CORRELATION_HINTS = ("相关性", "相关矩阵", "correlation", "correlation matrix")
_DISTRIBUTION_HINTS = (
    "字段分布",
    "变量分布",
    "数据分布",
    "频数",
    "直方图",
    "histogram",
    "frequency",
)
_OVERVIEW_HINTS = ("数据概况", "数据概览", "字段概况", "字段详情", "overview")
_RESERVED_ASCII_TOKENS = frozenset(
    {
        "analyze",
        "analysis",
        "and",
        "at",
        "between",
        "calculate",
        "check",
        "column",
        "columns",
        "compute",
        "correlation",
        "current",
        "data",
        "dataset",
        "distribution",
        "field",
        "fields",
        "for",
        "frequency",
        "histogram",
        "look",
        "matrix",
        "missing",
        "null",
        "of",
        "or",
        "overview",
        "profile",
        "rate",
        "sample",
        "show",
        "target",
        "the",
        "variable",
        "variables",
        "versus",
        "vs",
        "with",
    }
)


@dataclass(frozen=True)
class DatasetAnalysisRequest:
    sections: tuple[str, ...]
    columns: tuple[str, ...] | None
    target_col: str | None


@dataclass(frozen=True)
class DatasetAnalysisRequestResult:
    request: DatasetAnalysisRequest | None = None
    clarification: str | None = None


def detect_dataset_analysis_intent(utterance: str | None) -> bool:
    """Return whether the utterance explicitly requests dataset diagnostics."""

    text = _normalized_text(utterance)
    if not text:
        return False
    return any(
        hint in text
        for hint in (
            *_GENERAL_HINTS,
            *_TARGET_HINTS,
            *_MISSING_HINTS,
            *_CORRELATION_HINTS,
            *_DISTRIBUTION_HINTS,
            *_OVERVIEW_HINTS,
        )
    )


def build_dataset_analysis_request(
    utterance: str,
    *,
    columns: Sequence[str],
    target_col: str | None,
    business_names: Mapping[str, str],
) -> DatasetAnalysisRequestResult:
    """Bind an analysis request to the current dataset's known semantics."""

    text = _normalized_text(utterance)
    ordered_columns = tuple(dict.fromkeys(str(column) for column in columns))
    column_set = frozenset(ordered_columns)
    if not ordered_columns:
        return _clarify("当前数据集没有可分析字段。")
    if target_col is not None and target_col not in column_set:
        return _clarify(f"已配置的 target 字段「{target_col}」不在当前数据集中，请先修正字段映射。")

    is_general = any(hint in text for hint in _GENERAL_HINTS)
    requested: set[str] = set()
    if is_general:
        requested.update({"overview", "missing", "distribution", "correlation"})
        if target_col is not None:
            requested.add("target")
    else:
        if any(hint in text for hint in _OVERVIEW_HINTS):
            requested.add("overview")
        if any(hint in text for hint in _TARGET_HINTS):
            requested.add("target")
        if any(hint in text for hint in _MISSING_HINTS):
            requested.add("missing")
        if any(hint in text for hint in _DISTRIBUTION_HINTS):
            requested.add("distribution")
        if any(hint in text for hint in _CORRELATION_HINTS):
            requested.add("correlation")

    if not requested:
        return _clarify("请说明要看数据概况、target 分布、缺失、分布或相关矩阵。")
    if "target" in requested and target_col is None:
        return _clarify("当前样本尚未确认 target 字段，请先在数据语义中配置 target。")

    selected = _mentioned_columns(
        text,
        columns=ordered_columns,
        business_names=business_names,
    )
    unknown = _unknown_ascii_column_mentions(text, column_set)
    if unknown:
        name = unknown[0]
        return _clarify(f"当前数据集中没有字段「{name}」，请使用现有字段名或业务名称。")

    sections = tuple(section for section in _SECTION_ORDER if section in requested)
    return DatasetAnalysisRequestResult(
        request=DatasetAnalysisRequest(
            sections=sections,
            columns=selected or None,
            target_col=target_col,
        )
    )


def _mentioned_columns(
    text: str,
    *,
    columns: tuple[str, ...],
    business_names: Mapping[str, str],
) -> tuple[str, ...]:
    selected: list[str] = []
    for column in columns:
        raw = column.lower()
        if _contains_column_token(text, raw):
            selected.append(column)
            continue
        business_name = str(business_names.get(column) or "").strip().lower()
        if business_name and business_name in text:
            selected.append(column)
    return tuple(selected)


def _contains_column_token(text: str, column: str) -> bool:
    if not column:
        return False
    if column.isascii() and re.fullmatch(r"[a-z_][a-z0-9_]*", column):
        return re.search(rf"(?<![a-z0-9_]){re.escape(column)}(?![a-z0-9_])", text) is not None
    return column in text


def _unknown_ascii_column_mentions(
    text: str,
    known_columns: frozenset[str],
) -> tuple[str, ...]:
    known_lower = {column.lower() for column in known_columns}
    tokens = re.findall(r"[a-z_][a-z0-9_]*", text)
    unknown = [
        token
        for token in tokens
        if token not in known_lower and token not in _RESERVED_ASCII_TOKENS
    ]
    return tuple(dict.fromkeys(unknown))


def _clarify(message: str) -> DatasetAnalysisRequestResult:
    return DatasetAnalysisRequestResult(request=None, clarification=message)


def _normalized_text(value: str | None) -> str:
    return str(value or "").strip().lower()


__all__ = [
    "DatasetAnalysisRequest",
    "DatasetAnalysisRequestResult",
    "build_dataset_analysis_request",
    "detect_dataset_analysis_intent",
]

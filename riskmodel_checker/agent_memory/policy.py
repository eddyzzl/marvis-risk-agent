from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from riskmodel_checker.agent_memory.models import MemoryCandidate


@dataclass(frozen=True)
class MemoryPolicyDecision:
    allowed: bool
    reasons: list[str]


CUSTOMER_DETAIL_PATTERNS = (
    re.compile(r"(?:客户号|身份证|手机号|phone|mobile)\s*[:：=]?\s*[0-9A-Za-z_* -]{6,}"),
    re.compile(r"\b1[3-9]\d{9}\b"),
)
RAW_SAMPLE_ROW_PATTERNS = (
    re.compile(
        r"\b(?:raw\s+row|sample\s+row|样本行|原始样本)\b.*\b(?:score|y|target|apply_month)\s*=",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:age|score|target|y|apply_month|channel)\s*=\s*[^,\s]+.*\b(?:age|score|target|y|apply_month|channel)\s*=",
        re.IGNORECASE,
    ),
)
NOTEBOOK_SOURCE_PATTERNS = (
    re.compile(r"```(?:python|py|ipython)?\s", re.IGNORECASE),
    re.compile(r"\bimport\s+pandas\b"),
    re.compile(r"\bpd\.read_(?:csv|excel|feather|parquet)\b"),
)
PMML_OR_MODEL_PATTERNS = (
    re.compile(r"<\s*PMML\b", re.IGNORECASE),
    re.compile(r"<\s*(?:MiningModel|RegressionModel|TreeModel|NeuralNetwork)\b", re.IGNORECASE),
    re.compile(r"\b(?:pickle|joblib|model_file|pmml file content)\b", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"\b(?:api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
)
DB_CONNECTION_PATTERNS = (
    re.compile(r"\b(?:postgresql|mysql|oracle|sqlite|mongodb)://", re.IGNORECASE),
    re.compile(r"\b(?:jdbc|odbc):", re.IGNORECASE),
)


def classify_memory_candidate(candidate: MemoryCandidate) -> MemoryPolicyDecision:
    text = _candidate_text(candidate)
    reasons: list[str] = []

    if any(pattern.search(text) for pattern in CUSTOMER_DETAIL_PATTERNS):
        reasons.append("customer detail")
    if any(pattern.search(text) for pattern in RAW_SAMPLE_ROW_PATTERNS):
        reasons.append("raw sample row")
    if any(pattern.search(text) for pattern in NOTEBOOK_SOURCE_PATTERNS):
        reasons.append("notebook source")
    if any(pattern.search(text) for pattern in PMML_OR_MODEL_PATTERNS):
        reasons.append("pmml or model content")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        reasons.append("secret")
    if any(pattern.search(text) for pattern in DB_CONNECTION_PATTERNS):
        reasons.append("database connection")
    if _looks_like_long_report_text(candidate.summary):
        reasons.append("long report text")

    return MemoryPolicyDecision(allowed=not reasons, reasons=reasons)


def _candidate_text(candidate: MemoryCandidate) -> str:
    return f"{candidate.summary}\n{_json_text(candidate.payload)}"


def _json_text(value: dict[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _looks_like_long_report_text(summary: str) -> bool:
    text = str(summary or "")
    if len(text) < 300:
        return False
    report_markers = ("模型验证报告", "报告全文", "本报告", "验证报告")
    return any(marker in text for marker in report_markers)

"""Pure payload-shaping helpers for agent-memory capture.

Extracted from marvis/pipeline.py (ARCH-6) -- no behavior change. These are
pure functions (no I/O, no store writes) that shape memory-candidate
payloads and classify failure messages; they are only ever called, never
monkeypatched by name in tests, so moving them here is safe.

The actual capture entry points (_capture_agent_memory_for_metrics_success,
_capture_agent_memory_for_failure) and the MEM-7 negative-feedback helper
(_downgrade_task_memory_on_failure) stay in marvis/pipeline.py itself:
test_memory_policy.py monkeypatches AgentMemoryStore/extract_validation_pitfall/
extract_task_experience directly on the `marvis.pipeline` module object and
then calls pipeline._capture_agent_memory_for_failure(...), so those entry
points must resolve those names through marvis.pipeline's own namespace.
"""
from __future__ import annotations

import json
from pathlib import Path

from marvis.domain import TaskRecord

SCAN_STAGE_FAILURE_PREFIX = "材料扫描失败："


def _read_validation_results_payload(outputs_dir: Path) -> dict:
    path = outputs_dir / "validation_results.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _memory_model_experience_payload(
    *,
    task: TaskRecord,
    results: dict,
) -> dict:
    metrics_row = _memory_preferred_overall_row(results)
    return {
        "task_id": task.id,
        "source_task_id": task.id,
        "model_name": results.get("model_name") or task.model_name,
        "model_version": results.get("model_version") or task.model_version or "未标注",
        "scope": results.get("scope") or f"{task.model_name}验证任务",
        "channel": results.get("channel") or "未标注",
        "month": results.get("month") or _memory_latest_month(results) or "未标注",
        "metrics": {
            "ks": _memory_metric_value(metrics_row, "ks"),
            "auc": _memory_metric_value(metrics_row, "auc"),
            "psi": _memory_metric_value(metrics_row, "psi_vs_train", "psi"),
        },
        "important_feature_sources": _memory_important_feature_sources(results),
    }


def _memory_field_convention_payload(task: TaskRecord) -> dict:
    return {
        "task_id": task.id,
        "target_col": task.target_col,
        "score_col": task.score_col,
        "split_col": task.split_col,
        "time_col": task.time_col,
    }


def _memory_preferred_overall_row(results: dict) -> dict:
    overall = ((results.get("effectiveness") or {}).get("overall") or [])
    rows = [row for row in overall if isinstance(row, dict)]
    for split_name in ("oot", "test", "train"):
        for row in rows:
            if str(row.get("split") or "").strip().lower() == split_name:
                return row
    return rows[0] if rows else {}


def _memory_metric_value(row: dict, *keys: str):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _memory_latest_month(results: dict) -> str:
    monthly_sources = (
        (results.get("effectiveness") or {}).get("monthly_ks") or [],
        (results.get("basic_info") or {}).get("monthly_distribution") or [],
    )
    months: list[str] = []
    for rows in monthly_sources:
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("month") not in (None, ""):
                months.append(str(row["month"]))
    return sorted(months)[-1] if months else ""


def _memory_important_feature_sources(results: dict) -> list[str]:
    feature_importance = (results.get("basic_info") or {}).get("feature_importance") or []
    sources: list[str] = []
    for row in feature_importance:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category") or row.get("类别") or "").strip()
        if category:
            sources.append(category)
    return list(dict.fromkeys(sources)) or ["未标注"]


def _memory_failure_kind(message: str, *, default: str) -> str:
    text = str(message or "").lower()
    if (
        SCAN_STAGE_FAILURE_PREFIX.lower() in text
        or "too many files" in text
        or "too deep" in text
        or "source dir invalid" in text
    ):
        return "scan"
    if "pmml" in text:
        return "pmml"
    if (
        "field" in text
        or "column" in text
        or "字段" in text
        or "split_col" in text
        or "target_col" in text
        or "score_col" in text
        or "time_col" in text
    ):
        return "field"
    if "report" in text or "报告" in text or "word" in text:
        return "report"
    if "notebook" in text or "kernel" in text or "rmc_" in text:
        return "notebook"
    return default

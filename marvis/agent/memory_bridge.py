"""Bridge between the V2 plan-conversation driver (JOIN/FEATURE/MODELING) and
the agent memory subsystem (MEM-1).

The memory subsystem (store/retrieval/distillation) was, until now, wired only
into the V1.1 validation agent. This module gives the V2 driver the write-side
half of that capability, strictly observing INV-4 (memory is read-only with
respect to deterministic behavior) and INV-1 (all metrics still come from the
platform tools): when a V2 modeling/join plan reaches DONE, capture the
champion experiment / join execution result into agent_memory so future tasks
of the same kind have a historical anchor (MEM-1 write direction).

Every entry point here degrades silently to a no-op on any failure (missing
memory policy file, unreadable store, malformed metadata, ...): memory is a
strictly additive convenience, never a hard dependency of the V2 driver.
"""

from __future__ import annotations

import re
from typing import Any

from marvis.agent_memory.extractors import extract_join_experience, extract_model_experience
from marvis.agent_memory.store import AgentMemoryStore
from marvis.domain import TASK_TYPE_DATA_JOIN, TASK_TYPE_MODELING, TaskRecord
from marvis.memory_policy import load_memory_policy


def capture_agent_memory_for_driver_done(
    settings,
    task: TaskRecord,
    *,
    done_message_content: str = "",
    done_message_metadata: dict[str, Any] | None,
) -> None:
    """Write a V2 plan's terminal result into agent memory (MEM-1 write side).

    Called once a driver turn's assistant message is the ``done`` message.
    ``done_message_metadata`` is that message's ``metadata`` dict, exactly as
    built by ``PlanMessageComposer.done_message`` — for modeling this carries
    ``model_delivery`` (from ``build_model_delivery_payload``); for data_join
    the terminal step output is rendered into ``tables`` (per-table match rate)
    plus the ``done_message_content`` text (which carries the overall anchor/
    joined row counts — see ``renderers._render_execute_join``), so this reads
    the join outcome from both. Gated by the auto_distill memory policy flag,
    same as the existing V1.1 capture path (pipeline.py).
    """
    if not load_memory_policy(settings.workspace).auto_distill:
        return
    try:
        if task.task_type == TASK_TYPE_MODELING:
            _capture_model_experience(settings, task, done_message_metadata)
        elif task.task_type == TASK_TYPE_DATA_JOIN:
            _capture_join_experience(settings, task, done_message_content, done_message_metadata)
    except Exception:
        # Memory capture is best-effort; never fail the user-facing turn over it.
        return


def _capture_model_experience(
    settings, task: TaskRecord, metadata: dict[str, Any] | None
) -> None:
    delivery = (metadata or {}).get("model_delivery")
    if not isinstance(delivery, dict) or not delivery:
        return
    metrics = delivery.get("metrics") if isinstance(delivery.get("metrics"), dict) else {}
    recipe = str(delivery.get("recipe") or "").strip()
    if not recipe or not metrics:
        return
    scope = _modeling_scope(task, metadata)
    result = {
        "task_id": task.id,
        "source_task_id": task.id,
        "model_name": recipe,
        "model_version": str(delivery.get("artifact_id") or task.id),
        "scope": scope,
        "channel": "未标注",
        "month": "未标注",
        "metrics": {
            "ks": _first_metric(metrics, ("oot_ks", "test_ks", "ks")),
            "auc": _first_metric(metrics, ("oot_auc", "test_auc", "auc")),
            "psi": _first_metric(metrics, ("psi_oot_vs_train", "psi_test_vs_train", "psi")),
        },
        "important_feature_sources": [str(metrics.get("feature_count") or metrics.get("n_features") or "未标注")],
    }
    candidate = extract_model_experience(result)
    if candidate is None:
        return
    store = AgentMemoryStore(settings.db_path)
    store.create(candidate, task_id=task.id)


def _capture_join_experience(
    settings, task: TaskRecord, content: str, metadata: dict[str, Any] | None
) -> None:
    per_table = _join_per_table_from_tables(metadata)
    if not per_table:
        return
    match_rates = [
        float(row.get("match_rate"))
        for row in per_table
        if isinstance(row, dict) and isinstance(row.get("match_rate"), (int, float))
    ]
    if not match_rates:
        return
    anchor_rows, joined_rows = _join_row_counts_from_content(content)
    if anchor_rows is None or joined_rows is None:
        return
    result = {
        "task_id": task.id,
        "source_task_id": task.id,
        "match_rate": round(sum(match_rates) / len(match_rates), 4),
        "anchor_rows": anchor_rows,
        "joined_rows": joined_rows,
        "feature_table_count": len(per_table),
        "scope": f"data_join:{task.model_name or task.id}",
    }
    candidate = extract_join_experience(result)
    if candidate is None:
        return
    store = AgentMemoryStore(settings.db_path)
    store.create(candidate, task_id=task.id)


def _join_per_table_from_tables(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``done_message`` renders the terminal ``execute_join`` step into a
    structured table (see ``renderers._render_execute_join``): 特征表/命中率/
    新增列/新列缺失率/去重策略. Reconstruct the per-feature-table match rates
    from that rendered table (identified by its 命中率 column)."""
    tables = (metadata or {}).get("tables")
    if not isinstance(tables, list):
        return []
    for table in tables:
        if not isinstance(table, dict):
            continue
        columns = [str(c) for c in (table.get("columns") or [])]
        rows = table.get("rows") or []
        if not rows or "命中率" not in " ".join(columns):
            continue
        match_idx = next((i for i, c in enumerate(columns) if "命中率" in c), None)
        if match_idx is None:
            continue
        per_table = []
        for row in rows:
            cells = list(row) if isinstance(row, (list, tuple)) else []
            if match_idx >= len(cells):
                continue
            cell_text = str(cells[match_idx]).strip()
            try:
                rate = float(cell_text.rstrip("%"))
            except (TypeError, ValueError):
                continue
            rate = rate / 100.0 if "%" in cell_text else rate
            per_table.append({"match_rate": rate})
        if per_table:
            return per_table
    return []


_JOIN_ROW_COUNTS_RE = re.compile(r"锚行\s*(\d+)\s*→\s*拼接后\s*(\d+)\s*行")


def _join_row_counts_from_content(content: str) -> tuple[int | None, int | None]:
    """Parse the anchor/joined row counts out of the done message's rendered
    text (``renderers._render_execute_join``: "锚行 N → 拼接后 M 行"). These
    counts are not carried in structured metadata, only in the rendered text,
    so this is the only place they are available to the memory-capture bridge.
    """
    match = _JOIN_ROW_COUNTS_RE.search(str(content or ""))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _modeling_scope(task: TaskRecord, metadata: dict[str, Any] | None) -> str:
    target_type = str(getattr(task, "target_type", "") or "binary")
    delivery = (metadata or {}).get("model_delivery") if isinstance(metadata, dict) else {}
    scenario = str((delivery or {}).get("target_type") or "").strip() or target_type
    return f"{target_type}:{scenario}:{task.model_name or task.id}"


def _first_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


__all__ = [
    "capture_agent_memory_for_driver_done",
]

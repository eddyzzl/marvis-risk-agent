"""Bridge between the V2 plan-conversation driver (JOIN/FEATURE/MODELING) and
the agent memory subsystem (MEM-1 / MEM-4).

The memory subsystem (store/retrieval/distillation) was, until now, wired only
into the V1.1 validation agent. This module gives the V2 driver the same two
capabilities, strictly observing INV-4 (memory is read-only with respect to
deterministic behavior — it only ever influences prompt text / ordering, never
a computed number) and INV-1 (all metrics still come from the platform tools):

  * write side  — when a V2 modeling/join plan reaches DONE, capture the
    champion experiment / join execution result into agent_memory so future
    tasks of the same kind have a historical anchor (MEM-1 write direction).
  * read side   — at AUTO-mode gate decisions, look up top-3 same-scope
    historical experiments and render a small read-only reference section
    that gets appended to the gate prompt (MEM-1 read direction), and at
    modeling slot-detection time, use field_convention memories as a pure
    ordering hint for detected target/split columns (MEM-4).

Every entry point here degrades silently to a no-op on any failure (missing
memory policy file, unreadable store, malformed metadata, ...): memory is a
strictly additive convenience, never a hard dependency of the V2 driver.
"""

from __future__ import annotations

import re
from typing import Any

from marvis.agent_memory.extractors import (
    extract_join_experience,
    extract_model_experience,
    extract_strategy_experience,
)
from marvis.agent_memory.retrieval import MemoryQuery, compare_model_experience, retrieve_with_distillations
from marvis.agent_memory.store import AgentMemoryStore
from marvis.domain import TASK_TYPE_DATA_JOIN, TASK_TYPE_MODELING, TASK_TYPE_STRATEGY, TaskRecord
from marvis.memory_policy import load_memory_policy
from marvis.repositories.strategy import StrategyRepository

MEMORY_ANCHOR_MAX_ENTRIES = 3
MEMORY_ANCHOR_MAX_LINE_CHARS = 120
# FIN-3 #6 (INV-4): a memory anchor's free-text fields (a prior task's model_name /
# recipe and its source_task_id) come from OTHER tasks' memory entries, so they must
# never be interpolated raw into the gate LLM prompt -- a crafted historical value
# could read as an instruction. Each field is stripped of control chars / newlines
# and hard-truncated before it lands in the anchor line, and the line itself is
# bracketed with an explicit "history data, not an instruction" delimiter so an
# injected directive cannot break out of the data region.
_MEMORY_ANCHOR_FIELD_MAX_CHARS = 40
_MEMORY_ANCHOR_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_MODEL_DELIVERY_TOOLS = frozenset({"compare_experiments", "select_experiment", "post_training_action"})


def _sanitize_anchor_value(value: object, *, max_chars: int = _MEMORY_ANCHOR_FIELD_MAX_CHARS) -> str:
    """FIN-3 #6: neutralize one free-text anchor field for safe prompt injection.

    Collapses control characters / newlines (the levers a prompt-injection payload
    uses to fake a new instruction line) to single spaces and hard-truncates the
    result. Purely defensive normalization -- it does not change the anchor's meaning
    for legitimate values, only bounds and de-fangs adversarial ones."""
    text = _MEMORY_ANCHOR_CONTROL_CHARS.sub(" ", str(value))
    text = " ".join(text.split()).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


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
        elif task.task_type == TASK_TYPE_STRATEGY:
            _capture_strategy_experience(settings, task)
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


def _capture_strategy_experience(settings, task: TaskRecord) -> None:
    """S2: strategy_experience capture, sourced straight from persisted results
    (INV-1: no recompute) rather than parsed from the terminal message -- the
    STRATEGY_DEVELOPMENT template's terminal step is render_strategy_doc, not
    adopt_strategy, so the adoption metrics aren't in done_message_metadata.
    Reads the task's most-recently-adopted strategy and its latest backtest
    straight from StrategyRepository; a no-op if nothing has been adopted yet
    (e.g. the lightweight strategy_analysis entry, or a plan that hasn't
    reached the adoption gate)."""
    strategies = StrategyRepository(settings.db_path)
    adopted = [
        meta
        for meta in strategies.list_meta_for_task(task.id)
        if meta.get("status") == "adopted"
    ]
    if not adopted:
        return
    latest = max(adopted, key=lambda meta: (meta.get("adopted_at") or "", meta.get("created_at") or ""))
    strategy = strategies.get_strategy(latest["id"])
    backtests = strategies.list_backtests(latest["id"])
    if strategy is None or not backtests:
        return
    backtest = backtests[-1]
    result = {
        "task_id": task.id,
        "source_task_id": task.id,
        "strategy_type": strategy.strategy_type,
        "cutoff_summary": _strategy_cutoff_summary(strategy),
        "approval_rate": backtest.approval_rate,
        "approved_bad_rate": backtest.approved_bad_rate,
        "expected_profit": backtest.expected_profit,
        "scope": f"strategy:{strategy.strategy_type}:{task.model_name or task.id}",
    }
    candidate = extract_strategy_experience(result)
    if candidate is None:
        return
    store = AgentMemoryStore(settings.db_path)
    store.create(candidate, task_id=task.id)


def _strategy_cutoff_summary(strategy) -> str:
    conditions = [str(rule.condition) for rule in strategy.rules if getattr(rule, "condition", None)]
    return "；".join(conditions) if conditions else "无规则"


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


def build_memory_anchor(
    settings,
    task: TaskRecord,
    *,
    gate_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Read-side MEM-1: top-3 historical same-scope experiments for the modeling
    "选择实验"/调参 gates, as a read-only reference block. Returns ``None`` when
    memory is disabled, unavailable, or has nothing comparable — callers must
    render nothing in that case (regression: byte-identical to current gate
    payload with no memory).
    """
    if task.task_type != TASK_TYPE_MODELING:
        return None
    meta = gate_metadata if isinstance(gate_metadata, dict) else {}
    tool = _gate_delivery_tool(meta)
    has_modeling_setup = isinstance(meta.get("modeling_setup"), dict) and bool(
        meta["modeling_setup"].get("recipe")
    )
    if tool not in _MODEL_DELIVERY_TOOLS and not has_modeling_setup:
        return None
    if not load_memory_policy(settings.workspace).reference_cross_task:
        return None
    try:
        store = AgentMemoryStore(settings.db_path)
        history = store.list_entries(memory_type="model_experience", limit=200)
    except Exception:
        return None
    if not history:
        return None
    scope = _modeling_scope(task, meta)
    current_payload = {"scope": scope, "model_name": _gate_recipe(meta)}
    try:
        comparison = compare_model_experience(current_payload, history, limit=MEMORY_ANCHOR_MAX_ENTRIES)
    except Exception:
        return None
    packets = [
        packet
        for packet in comparison.get("context_packets", [])
        if str(packet.get("confidence") or "").lower() != "low"
    ][:MEMORY_ANCHOR_MAX_ENTRIES]
    if not packets:
        return None
    lines: list[str] = []
    references: list[dict[str, Any]] = []
    for packet in packets:
        line = _anchor_line(packet)
        if not line:
            continue
        lines.append(line[:MEMORY_ANCHOR_MAX_LINE_CHARS])
        references.append({
            "id": packet.get("id"),
            "kind": packet.get("kind", "raw"),
            "use_reason": "gate_memory_anchor",
        })
    if not lines:
        return None
    return {"lines": lines, "references": references}


def _gate_delivery_tool(meta: dict[str, Any]) -> str:
    delivery = meta.get("model_delivery")
    if isinstance(delivery, dict) and delivery.get("source_tool"):
        return str(delivery.get("source_tool") or "")
    return ""


def _gate_recipe(meta: dict[str, Any]) -> str:
    delivery = meta.get("model_delivery")
    if isinstance(delivery, dict) and delivery.get("recipe"):
        return str(delivery.get("recipe") or "")
    modeling_setup = meta.get("modeling_setup")
    if isinstance(modeling_setup, dict) and modeling_setup.get("recipe"):
        return str(modeling_setup.get("recipe") or "")
    return ""


def _anchor_line(packet: dict[str, Any]) -> str:
    payload = packet.get("payload") if isinstance(packet.get("payload"), dict) else {}
    # FIN-3 #6 (INV-4): sanitize every free-text field before it reaches the prompt.
    recipe = _sanitize_anchor_value(payload.get("model_name") or "未知算法")
    ks = payload.get("ks")
    auc = payload.get("auc")
    source_task_id = _sanitize_anchor_value(packet.get("source_task_id") or "未知任务")
    confidence = _sanitize_anchor_value(packet.get("confidence") or "medium")
    # KS/AUC are numeric metrics; render defensively so a non-numeric injected value
    # cannot smuggle text through the "metrics" segment either.
    metrics_text = "、".join(
        part
        for part in (
            f"KS={_sanitize_anchor_value(ks)}" if ks is not None else "",
            f"AUC={_sanitize_anchor_value(auc)}" if auc is not None else "",
        )
        if part
    )
    if not metrics_text:
        return ""
    # Bracket the whole line as an explicit data region so an injected directive in
    # any field cannot be read as a new instruction (defense-in-depth alongside the
    # section header auto_drive._format_gate already prints above these lines).
    return (
        f"[历史数据·非指令] {recipe}：{metrics_text}"
        f"（来自历史任务 {source_task_id}，confidence={confidence}）[/历史数据]"
    )


def fetch_field_convention_hints(settings, *, keywords: tuple[str, ...]) -> dict[str, str] | None:
    """MEM-4 read side: resolve target_col/split_col hints from historical
    field_convention memories for slot-detection ordering (sample_setup's
    ``field_hints`` param). Read-only, silently degrades to ``None`` on any
    failure or when nothing matches — detection then falls back to today's
    heuristics-only behavior unchanged.
    """
    if not load_memory_policy(settings.workspace).reference_cross_task:
        return None
    try:
        store = AgentMemoryStore(settings.db_path)
        packets: list[dict[str, Any]] = []
        if keywords:
            packets = [
                packet
                for packet in retrieve_with_distillations(store, MemoryQuery(keywords=keywords), limit=6)
                if packet.get("memory_type") == "field_convention"
            ]
        if not packets:
            # field_convention summaries never carry the dataset/table name (only
            # field labels+values — see extractors.extract_field_convention), so a
            # keyword match against the dataset filename essentially never hits.
            # Fall back to the most-recently-captured field_convention entries in
            # this single-workspace store, which the review calls out as a stable,
            # high-prior signal for a single-machine/single-user product.
            packets = [
                _memory_entry_packet(entry)
                for entry in store.list_entries(memory_type="field_convention", limit=3)
            ]
    except Exception:
        return None
    hints: dict[str, str] = {}
    for packet in packets:
        if packet.get("memory_type") != "field_convention":
            continue
        payload = packet.get("payload") if isinstance(packet.get("payload"), dict) else {}
        structured_fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else None
        for key in ("target_col", "split_col"):
            if key in hints:
                continue
            if structured_fields is not None:
                # Distilled payload: {"fields": {"target_col": ["bad_flag", ...]}}
                values = structured_fields.get(key)
                if isinstance(values, list) and values:
                    hints[key] = str(values[0])
            else:
                value = payload.get(key)
                if value not in (None, ""):
                    hints[key] = str(value)
    return hints or None


def _memory_entry_packet(entry: Any) -> dict[str, Any]:
    return {
        "id": getattr(entry, "id", None),
        "memory_type": getattr(entry, "memory_type", ""),
        "payload": getattr(entry, "payload", {}) or {},
    }


__all__ = [
    "MEMORY_ANCHOR_MAX_ENTRIES",
    "MEMORY_ANCHOR_MAX_LINE_CHARS",
    "build_memory_anchor",
    "capture_agent_memory_for_driver_done",
    "fetch_field_convention_hints",
]

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
  * read side   — at meaningful workflow milestones, look up top-3 historical
    results and render a small read-only reference section for both NORMAL and
    AUTO modes (MEM-1 read direction), and at modeling slot-detection time, use
    field_convention memories as a pure ordering hint for detected target/split
    columns (MEM-4).

Every entry point here degrades silently to a no-op on any failure (missing
memory policy file, unreadable store, malformed metadata, ...): memory is a
strictly additive convenience, never a hard dependency of the V2 driver.
"""

from __future__ import annotations

import re
from typing import Any

from marvis.agent_memory.extractors import (
    extract_feature_experience,
    extract_join_experience,
    extract_model_experience,
    extract_risk_analysis_experience,
    extract_strategy_experience,
)
from marvis.agent_memory.distillation import render_structured_distillation_summary
from marvis.agent_memory.api_support import dispatch_memory_after_save
from marvis.agent_memory.retrieval import MemoryQuery, compare_model_experience, retrieve_with_distillations
from marvis.agent_memory.store import AgentMemoryStore
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    TaskRecord,
)
from marvis.memory_policy import load_memory_policy
from marvis.packs.strategy.backtest_compat import approval_backtest_projection
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.strategy import StrategyRepository
from marvis.strategy_lifecycle import ASSET_STATUS_ADOPTED_LOCAL

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
    hook_dispatcher=None,
) -> list[dict[str, Any]]:
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
        return []
    entries = []
    try:
        if task.task_type == TASK_TYPE_MODELING:
            entries.append(_capture_model_experience(settings, task, done_message_metadata))
        elif task.task_type == TASK_TYPE_DATA_JOIN:
            entries.append(_capture_join_experience(settings, task, done_message_content, done_message_metadata))
        elif task.task_type == TASK_TYPE_FEATURE_ANALYSIS:
            entries.append(_capture_feature_experience(settings, task, done_message_metadata))
        elif task.task_type == TASK_TYPE_STRATEGY:
            entries.append(_capture_strategy_experience(settings, task))
        elif task.task_type == TASK_TYPE_VINTAGE:
            entries.append(_capture_risk_analysis_experience(settings, task, done_message_metadata))
    except Exception:
        # Memory capture is best-effort; never fail the user-facing turn over it.
        return []
    receipts = []
    for entry in entries:
        if entry is None or entry.status != "active":
            continue
        dispatch_memory_after_save(
            hook_dispatcher,
            task_id=task.id,
            memory_type=entry.memory_type,
        )
        receipts.append({
            "id": entry.id,
            "memory_type": entry.memory_type,
            "summary": entry.summary,
            "status": entry.status,
        })
    return receipts


def _capture_model_experience(
    settings, task: TaskRecord, metadata: dict[str, Any] | None
) -> Any | None:
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
        return None
    store = AgentMemoryStore(settings.db_path)
    return store.create(candidate, task_id=task.id)


def _capture_join_experience(
    settings, task: TaskRecord, content: str, metadata: dict[str, Any] | None
) -> Any | None:
    per_table = _join_per_table_from_tables(metadata)
    if not per_table:
        return None
    match_rates = [
        float(row.get("match_rate"))
        for row in per_table
        if isinstance(row, dict) and isinstance(row.get("match_rate"), (int, float))
    ]
    if not match_rates:
        return None
    anchor_rows, joined_rows = _join_row_counts_from_content(content)
    if anchor_rows is None or joined_rows is None:
        return None
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
        return None
    store = AgentMemoryStore(settings.db_path)
    return store.create(candidate, task_id=task.id)


def _capture_feature_experience(
    settings, task: TaskRecord, metadata: dict[str, Any] | None
) -> Any | None:
    tables = (metadata or {}).get("tables") if isinstance(metadata, dict) else []
    advice = next(
        (
            table for table in (tables or [])
            if isinstance(table, dict) and table.get("title") == "Agent 特征建议"
        ),
        None,
    )
    if advice is None:
        return None
    columns = [str(item) for item in (advice.get("columns") or [])]
    try:
        feature_idx = columns.index("特征")
        recommendation_idx = columns.index("Agent建议")
    except ValueError:
        return None
    state_idx = columns.index("建议状态") if "建议状态" in columns else -1
    confidence_idx = columns.index("证据置信度") if "证据置信度" in columns else -1
    evidence_idx = columns.index("支持指标") if "支持指标" in columns else -1
    recommended: list[str] = []
    avoid: list[str] = []
    recommendation_evidence: dict[str, dict[str, str]] = {}
    actionable_confidences: list[str] = []
    adverse_tokens = ("不推荐", "剔除", "慎用", "谨慎", "不建议", "不可用")
    positive_states = frozenset({"recommended", "candidate"})
    negative_states = frozenset({"not_recommended", "caution"})
    for source_row in advice.get("rows") or []:
        if not isinstance(source_row, (list, tuple)):
            continue
        row = list(source_row)
        if max(feature_idx, recommendation_idx) >= len(row):
            continue
        feature = str(row[feature_idx] or "").strip()
        recommendation = str(row[recommendation_idx] or "").strip()
        if not feature:
            continue
        state = (
            str(row[state_idx] or "").strip().lower()
            if 0 <= state_idx < len(row)
            else ""
        )
        if state in negative_states or (
            not state and any(token in recommendation for token in adverse_tokens)
        ):
            avoid.append(feature)
        elif state in positive_states or (
            not state and recommendation in {"推荐", "候选"}
        ):
            recommended.append(feature)
        else:
            # "待评估"/unevaluated is intentionally neutral: insufficient
            # evidence must not become a reusable recommendation.
            continue
        confidence = (
            str(row[confidence_idx] or "").strip().lower()
            if 0 <= confidence_idx < len(row)
            else ""
        )
        evidence = (
            str(row[evidence_idx] or "").strip()
            if 0 <= evidence_idx < len(row)
            else ""
        )
        actionable_confidences.append(confidence)
        if evidence and evidence != "-":
            recommendation_evidence[feature] = {
                "state": state or recommendation,
                "confidence": confidence or "unknown",
                "metrics": evidence,
            }
    actionable_features = set(recommended) | set(avoid)
    if not actionable_features:
        return None
    recommendation_confidence = (
        "high"
        if actionable_confidences
        and set(recommendation_evidence) == actionable_features
        and all(value == "high" for value in actionable_confidences)
        else "medium"
        if recommended or avoid
        else "low"
    )
    result = {
        "task_id": task.id,
        "source_task_id": task.id,
        "feature_count": len(advice.get("rows") or []),
        "recommended_features": list(dict.fromkeys(recommended)),
        "avoid_features": list(dict.fromkeys(avoid)),
        "recommendation_confidence": recommendation_confidence,
        "recommendation_evidence": recommendation_evidence,
        "target_col": task.target_col,
        # Group by analytical target instead of the user-entered task name.
        # Equivalent analyses should consolidate even when operators give the
        # tasks different display names.
        "scope": f"feature:target={task.target_col or 'unknown'}",
    }
    candidate = extract_feature_experience(result)
    if candidate is None:
        return None
    return AgentMemoryStore(settings.db_path).create(candidate, task_id=task.id)


def _capture_strategy_experience(settings, task: TaskRecord) -> Any | None:
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
        if meta.get("asset_status") == ASSET_STATUS_ADOPTED_LOCAL
    ]
    if not adopted:
        return None
    latest = max(adopted, key=lambda meta: (meta.get("adopted_at") or "", meta.get("created_at") or ""))
    strategy = strategies.get_strategy(latest["id"])
    backtests = strategies.list_backtests(latest["id"])
    if strategy is None or not backtests:
        return None
    # The existing strategy_experience memory contract is deliberately approval
    # specific.  Limit, pricing and segmentation backtests carry different typed
    # metrics; skipping them is safer than inventing approval-rate aliases.
    if strategy.strategy_type not in {"approval", "reject"}:
        return None
    backtest = backtests[-1]
    memory_metrics = _approval_backtest_memory_metrics(
        backtest,
        strategy_type=strategy.strategy_type,
    )
    if memory_metrics is None:
        return None
    result = {
        "task_id": task.id,
        "source_task_id": task.id,
        "strategy_type": strategy.strategy_type,
        "cutoff_summary": _strategy_cutoff_summary(strategy),
        **memory_metrics,
        "scope": f"strategy:{strategy.strategy_type}:{task.model_name or task.id}",
    }
    candidate = extract_strategy_experience(result)
    if candidate is None:
        return None
    store = AgentMemoryStore(settings.db_path)
    return store.create(candidate, task_id=task.id)


def _capture_risk_analysis_experience(
    settings,
    task: TaskRecord,
    metadata: dict[str, Any] | None,
) -> Any:
    report = (metadata or {}).get("risk_analysis_report")
    if not isinstance(report, dict) or not report:
        return
    result = dict(report)
    # Provenance is owned by the terminal task, never by a user-controlled
    # report field. Overwrite both aliases before governed extraction.
    result["task_id"] = task.id
    result["source_task_id"] = task.id
    candidate = extract_risk_analysis_experience(result)
    if candidate is None:
        return
    store = AgentMemoryStore(settings.db_path)
    return store.create(candidate, task_id=task.id)


def _approval_backtest_memory_metrics(
    backtest: object,
    *,
    strategy_type: str,
) -> dict[str, Any] | None:
    """Read approval memory fields from V2 envelopes or legacy result objects.

    No arithmetic or fallback default is performed here. Approval rate and
    approved bad rate must exist in deterministic backtest evidence; profit is
    optional because a valid backtest may not have received economic assumptions.
    For a versioned envelope, canonical ``metrics``/``economics`` win over any
    top-level compatibility projection.
    """

    if strategy_type not in {"approval", "reject"}:
        return None
    envelope_type = getattr(backtest, "strategy_type", None)
    if envelope_type is not None and envelope_type != strategy_type:
        return None
    try:
        projection = approval_backtest_projection(
            backtest,
            preserve_undefined_rates=True,
        )
    except (AttributeError, StrategyError, TypeError):
        return None
    if projection.get("strategy_id") is None:
        return None
    values = {
        "approval_rate": projection.get("approval_rate"),
        "approved_bad_rate": projection.get("approved_bad_rate"),
        "expected_profit": projection.get("expected_profit"),
    }
    if values["approval_rate"] is None or values["approved_bad_rate"] is None:
        return None
    return values


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
    scope = _modeling_scope(task, meta)
    try:
        store = AgentMemoryStore(settings.db_path)
        history = [
            entry
            for entry in store.list_entries(memory_type="model_experience", limit=200)
            if str(entry.source_task_id or "") != task.id
            and str(entry.payload.get("scope") or "") == scope
        ]
    except Exception:
        return None
    if not history:
        return None
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


def build_workflow_memory_context(
    settings,
    task: TaskRecord,
    *,
    limit: int = 3,
) -> dict[str, Any] | None:
    """Return visible same-workflow memory for Agent interpretation.

    Unlike ``build_memory_anchor`` this covers data join, feature analysis and
    modeling in both NORMAL and AUTO modes.  The context is explanation-only;
    callers must never feed it back into deterministic tool inputs.
    """
    if not load_memory_policy(settings.workspace).reference_cross_task:
        return None
    category = {
        TASK_TYPE_DATA_JOIN: "join_experience",
        TASK_TYPE_FEATURE_ANALYSIS: "feature_experience",
        TASK_TYPE_MODELING: "model_experience",
        TASK_TYPE_STRATEGY: "strategy_experience",
    }.get(task.task_type)
    if category is None:
        return None
    expected_scope = _workflow_memory_scope(task, category)
    if expected_scope is None:
        # A missing analytical target/model scope is not permission to mix
        # unrelated task histories. Stay silent until the current scope is
        # known rather than falling back to category-wide memories.
        return None
    try:
        store = AgentMemoryStore(settings.db_path)
        distillations = [
            item
            for item in store.list_distillations(category=category, limit=200)
            if _distillation_scope_matches(category, expected_scope, item)
            and _distillation_is_actionable(category, item)
        ][:limit]
        entries = [
            entry
            for entry in store.list_entries(memory_type=category, limit=200)
            if str(entry.source_task_id or "") != task.id
            and str(entry.confidence or "").lower() != "low"
            and _memory_scope_matches(
                category,
                expected_scope,
                str(entry.payload.get("scope") or ""),
            )
        ][:limit]
    except Exception:
        return None
    memories: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    lines: list[str] = []
    for item in distillations:
        if item.confidence == "low":
            continue
        source_summary = (
            render_structured_distillation_summary(category, item.structured)
            if category == "feature_experience"
            else item.distilled_summary
        )
        summary = _sanitize_anchor_value(source_summary, max_chars=180)
        if not summary:
            continue
        memories.append({
            "id": item.id,
            "kind": "distillation",
            "summary": summary,
            "confidence": item.confidence,
            "scope": expected_scope,
        })
        references.append({
            "id": item.id,
            "kind": "distillation",
            "scope": expected_scope,
            "use_reason": "workflow_insight",
        })
        lines.append(f"同口径记忆沉淀（{expected_scope}，{item.confidence}）：{summary}")
        if len(memories) >= limit:
            break
    remaining = max(0, limit - len(memories))
    selected_entries = entries[:remaining]
    if selected_entries:
        try:
            found = store.record_retrievals(
                [entry.id for entry in selected_entries],
                task_id=task.id,
            )
        except Exception:
            found = set()
        for entry in selected_entries:
            if entry.id not in found:
                continue
            summary = _sanitize_anchor_value(entry.summary, max_chars=180)
            memories.append({
                "id": entry.id,
                "kind": "raw",
                "summary": summary,
                "confidence": entry.confidence,
                "source_task_id": entry.source_task_id,
                "scope": expected_scope,
            })
            references.append({
                "id": entry.id,
                "kind": "raw",
                "scope": expected_scope,
                "use_reason": "workflow_insight",
            })
            lines.append(
                f"同口径历史任务 {entry.source_task_id}"
                f"（{expected_scope}，{entry.confidence}）：{summary}"
            )
    if not memories:
        return None
    return {"category": category, "lines": lines, "memories": memories, "references": references}


def _workflow_memory_scope(task: TaskRecord, category: str) -> str | None:
    model_name = str(task.model_name or "").strip()
    if category == "feature_experience":
        target_col = str(task.target_col or "").strip()
        return f"feature:target={target_col}" if target_col else None
    if category == "join_experience":
        return f"data_join:{model_name}" if model_name else None
    if category == "model_experience":
        return _modeling_scope(task, None) if model_name else None
    if category == "strategy_experience":
        # Strategy type is decided inside the workflow. The task's governed
        # model name is the common boundary available before that decision.
        return f":{model_name}" if model_name else None
    return None


def _memory_scope_matches(category: str, expected: str, actual: str) -> bool:
    if not actual:
        return False
    if category == "strategy_experience":
        return actual.endswith(expected)
    return actual == expected


def _distillation_scope_matches(category: str, expected: str, item: Any) -> bool:
    structured = item.structured if isinstance(item.structured, dict) else {}
    scopes = structured.get("scopes")
    if isinstance(scopes, list) and any(
        _memory_scope_matches(category, expected, str(scope or ""))
        for scope in scopes
    ):
        return True
    scope_key = str(item.scope_key or "")
    if category == "strategy_experience":
        return scope_key.endswith(expected)
    return scope_key == f"{category}:{expected}"


def _distillation_is_actionable(category: str, item: Any) -> bool:
    if category != "feature_experience":
        return True
    structured = item.structured if isinstance(item.structured, dict) else {}
    return any(
        structured.get(field)
        for field in (
            "recommended_features",
            "avoid_features",
            "inconsistent_features",
        )
    )


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
    "build_workflow_memory_context",
    "capture_agent_memory_for_driver_done",
    "fetch_field_convention_hints",
]

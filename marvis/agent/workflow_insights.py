"""Grounded Agent interpretation for deterministic workflow outputs.

The platform tools remain the only source of metrics.  This module selects a
bounded set of already-computed facts and lets the configured LLM explain them;
when no LLM is configured it renders the same facts with a deterministic
fallback so a workflow never becomes silent or misleading.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from marvis.agent.json_reply import load_json_object
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
)
from marvis.llm_prompts import WORKFLOW_INSIGHT_SYS


_ROW_COUNTS = re.compile(r"锚行\s*(\d+)\s*→\s*拼接后\s*(\d+)\s*行")
_ADVERSE_RECOMMENDATIONS = ("不推荐", "剔除", "慎用", "谨慎", "不建议", "不可用")
_POSITIVE_FEATURE_STATES = frozenset({"recommended", "candidate"})
_NEGATIVE_FEATURE_STATES = frozenset({"not_recommended", "caution"})


def build_workflow_insight_context(
    task_type: str,
    *,
    stage: str,
    metadata: dict[str, Any] | None,
    content: str = "",
) -> dict[str, Any] | None:
    """Extract auditable facts for one meaningful workflow milestone."""
    meta = metadata if isinstance(metadata, dict) else {}
    tables = [item for item in (meta.get("tables") or []) if isinstance(item, dict)]
    if task_type == TASK_TYPE_DATA_JOIN:
        return _join_context(stage, tables, meta, content)
    if task_type == TASK_TYPE_FEATURE_ANALYSIS:
        return _feature_context(stage, tables, meta)
    if task_type == TASK_TYPE_MODELING:
        return _model_context(stage, tables, meta)
    return None


def render_workflow_insight(
    context: dict[str, Any],
    *,
    client=None,
    memory_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return user-facing Agent analysis grounded in ``context`` only."""
    memory = [item for item in (memory_context or []) if isinstance(item, dict)][:3]
    parsed = _llm_insight(client, context, memory)
    generated_by = "llm" if parsed is not None else "deterministic_fallback"
    if parsed is None:
        parsed = {
            "summary": _fallback_summary(context),
            "findings": list(context.get("facts") or []),
            "risks": list(context.get("risks") or []),
            "recommendations": list(context.get("recommendations") or []),
        }
    return {
        "content": _render_sections(context.get("title") or "Agent 分析", parsed),
        "generated_by": generated_by,
        "milestone": context.get("milestone"),
        "evidence": list(context.get("evidence") or []),
        "recommended_features": list(context.get("recommended_features") or []),
        "avoid_features": list(context.get("avoid_features") or []),
        "recommended_params": dict(context.get("recommended_params") or {}),
    }


def _join_context(stage: str, tables: list[dict], meta: dict, content: str) -> dict | None:
    table = _find_table(tables, ("各特征表贡献", "拼接诊断（逐特征表）", "择键建议"))
    if table is None:
        return None
    columns = _columns(table)
    name_idx = _column_index(columns, ("特征表", "文件"), default=0)
    rate_idx = _column_index(columns, ("命中率", "当前命中率", "减后命中率"))
    missing_idx = _column_index(columns, ("新列缺失率", "缺失率"))
    unique_idx = _column_index(columns, ("键唯一", "减后唯一"))
    inflation_idx = _column_index(columns, ("膨胀", "减后膨胀"))
    facts: list[str] = []
    risks: list[str] = []
    rows = [list(row) for row in (table.get("rows") or []) if isinstance(row, (list, tuple))]
    for row in rows:
        name = _cell(row, name_idx) or "未知特征表"
        rate = _number(_cell(row, rate_idx))
        if rate is not None:
            facts.append(f"{name} 命中率 {_pct_text(rate)}")
            if rate < 0.8:
                risks.append(f"{name} 命中率仅 {_pct_text(rate)}，需要复核拼接键、时间口径或覆盖范围。")
        missing = _number(_cell(row, missing_idx))
        if missing is not None and missing > 0.3:
            risks.append(f"{name} 拼入列缺失率 {_pct_text(missing)}，下游使用前应确认缺失机制。")
        unique = _cell(row, unique_idx).lower()
        if unique in {"否", "false", "0", "✗"}:
            risks.append(f"{name} 的拼接键不唯一，存在一对多或重复记录风险。")
        inflation = _cell(row, inflation_idx).lower()
        if inflation in {"是", "true", "1", "⚠是", "⚠️是"}:
            risks.append(f"{name} 检测到行数膨胀，应检查去重策略和键唯一性。")
    count_match = _ROW_COUNTS.search(str(content or ""))
    if count_match:
        anchor_rows, joined_rows = (int(count_match.group(1)), int(count_match.group(2)))
        facts.append(f"锚表 {anchor_rows} 行，拼接结果 {joined_rows} 行。")
        if joined_rows != anchor_rows:
            risks.append(f"拼接前后行数由 {anchor_rows} 变为 {joined_rows}，未保持严格 1:1。")
    if not facts and not risks:
        return None
    return _context(
        title="样本拼接 Agent 解读",
        milestone="join_completed" if stage == "done" else "join_diagnosed",
        facts=facts,
        risks=risks,
        recommendations=(
            ["优先处理低命中、键不唯一和行数膨胀问题，再把结果用于特征分析或建模。"]
            if risks else
            ["当前拼接结果未发现明显结构性异常，可继续核对新增字段及业务口径。"]
        ),
        evidence=[str(table.get("title") or "拼接结果")],
    )


def _feature_context(stage: str, tables: list[dict], meta: dict) -> dict | None:
    advice = _find_table(tables, ("Agent 特征建议",))
    quality = _find_table(tables, ("数据质量",))
    metrics = _find_table(tables, ("特征指标",))
    if advice is None and quality is None and metrics is None:
        return None
    recommended: list[str] = []
    avoid: list[str] = []
    facts: list[str] = []
    risks: list[str] = []
    if advice is not None:
        columns = _columns(advice)
        feature_idx = _column_index(columns, ("特征",), default=0)
        advice_idx = _column_index(columns, ("Agent建议", "建议"), default=1)
        reason_idx = _column_index(columns, ("推荐原因", "原因"), default=2)
        state_idx = _column_index(columns, ("建议状态",), default=-1)
        for source_row in advice.get("rows") or []:
            if not isinstance(source_row, (list, tuple)):
                continue
            row = list(source_row)
            feature = _cell(row, feature_idx)
            label = _cell(row, advice_idx)
            reason = _cell(row, reason_idx)
            state = _cell(row, state_idx) if state_idx >= 0 else ""
            if not feature:
                continue
            if state in _NEGATIVE_FEATURE_STATES or (
                not state and any(token in label for token in _ADVERSE_RECOMMENDATIONS)
            ):
                avoid.append(feature)
                risks.append(f"{feature}：{label}；{reason or '需要进一步复核'}。")
            elif state in _POSITIVE_FEATURE_STATES or (
                not state and label in {"推荐", "候选"}
            ):
                recommended.append(feature)
                facts.append(f"{feature}：{label or '可进一步评估'}；{reason or '平台未提供额外原因'}。")
            else:
                facts.append(f"{feature}：{label or '待评估'}；{reason or '当前证据不足'}。")
    if quality is not None:
        columns = _columns(quality)
        feature_idx = _column_index(columns, ("特征",), default=0)
        for source_row in quality.get("rows") or []:
            if not isinstance(source_row, (list, tuple)):
                continue
            row = list(source_row)
            feature = _cell(row, feature_idx)
            for labels, threshold, label in (
                (("缺失率",), 0.5, "缺失率"),
                (("单一值率",), 0.95, "单一值率"),
                (("零值率",), 0.95, "零值率"),
            ):
                value = _number(_cell(row, _column_index(columns, labels)))
                if value is not None and value >= threshold:
                    risks.append(f"{feature} 的{label}为 {_pct_text(value)}，直接入模可能不稳定或贡献有限。")
                    if feature and feature not in avoid:
                        avoid.append(feature)
    if metrics is not None:
        facts.insert(0, f"本次完成 {len(metrics.get('rows') or [])} 个特征的单变量分析。")
    return _context(
        title="特征分析 Agent 解读",
        milestone="feature_reported" if stage == "done" else "feature_analyzed",
        facts=facts,
        risks=_unique(risks),
        recommendations=(
            ([f"优先评估：{', '.join(recommended[:8])}。"] if recommended else [])
            + ([f"建议暂缓或谨慎使用：{', '.join(avoid[:8])}。"] if avoid else [])
        ),
        evidence=[str(item.get("title") or "") for item in (advice, quality, metrics) if item],
        recommended_features=_unique(recommended),
        avoid_features=_unique(avoid),
    )


def _model_context(stage: str, tables: list[dict], meta: dict) -> dict | None:
    delivery = meta.get("model_delivery") if isinstance(meta.get("model_delivery"), dict) else {}
    metrics = delivery.get("metrics") if isinstance(delivery.get("metrics"), dict) else {}
    params_table = _find_table(tables, ("最优超参", "固定/控制参数"))
    comparison = _find_table(tables, ("候选模型对比", "最终模型指标", "模型指标"))
    candidate = _recommended_model_candidate(delivery, comparison)
    if not metrics and candidate is not None:
        candidate_metrics = candidate.get("metrics")
        if isinstance(candidate_metrics, dict):
            metrics = candidate_metrics
    if not metrics and params_table is None and comparison is None:
        return None
    facts: list[str] = []
    risks: list[str] = []
    recipe = str(
        delivery.get("recipe")
        or (candidate.get("recipe") if candidate is not None else "")
        or ""
    ).strip()
    if recipe:
        facts.append(f"当前推荐模型为 {recipe}。")
    metric_labels = (
        ("train_ks", "Train KS"),
        ("test_ks", "Test KS"),
        ("oot_ks", "OOT KS"),
        ("test_auc", "Test AUC"),
        ("oot_auc", "OOT AUC"),
        ("psi_oot_vs_train", "OOT PSI"),
    )
    for key, label in metric_labels:
        value = _number(metrics.get(key))
        if value is not None:
            facts.append(f"{label}={value:.4f}")
    train_ks = _number(metrics.get("train_ks"))
    test_ks = _number(metrics.get("test_ks"))
    oot_ks = _number(metrics.get("oot_ks"))
    if train_ks is not None and test_ks is not None and train_ks - test_ks > 0.05:
        risks.append(f"Train/Test KS 相差 {train_ks - test_ks:.4f}，存在过拟合迹象。")
    if test_ks is not None and oot_ks is not None and test_ks - oot_ks > 0.05:
        risks.append(f"Test/OOT KS 相差 {test_ks - oot_ks:.4f}，样本外效果有衰减。")
    if oot_ks is None:
        risks.append("当前没有 OOT KS，无法评价时间外推稳定性。")
    params: dict[str, str] = {}
    if params_table is not None:
        for row in params_table.get("rows") or []:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                params[str(row[0])] = str(row[1])
    recommendations = []
    if params:
        recommendations.append("建议参数沿用本轮确定的最优组合，并在报告中保留搜索范围和随机种子。")
    recommendations.append(
        "上线前重点复核 OOT、过拟合差距、稳定性和重要特征的业务合理性。"
        if risks else
        "当前指标未触发明显效果红旗，仍建议结合业务目标和样本外稳定性决定是否交付。"
    )
    return _context(
        title="模型效果 Agent 解读",
        milestone="model_completed" if stage == "done" else "model_review",
        facts=facts,
        risks=risks,
        recommendations=recommendations,
        evidence=[str(item.get("title") or "") for item in (comparison, params_table) if item]
        + (["model_delivery.metrics"] if metrics else []),
        recommended_params=params,
    )


def _recommended_model_candidate(delivery: dict, comparison: dict | None) -> dict | None:
    candidates = [
        item
        for item in (delivery.get("candidates") or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return None
    selected_id = str(delivery.get("selected_experiment_id") or "").strip()
    selected_recipe = str(delivery.get("recipe") or "").strip()
    for candidate in candidates:
        if candidate.get("selected"):
            return candidate
        candidate_id = str(
            candidate.get("id") or candidate.get("experiment_id") or ""
        ).strip()
        if selected_id and candidate_id == selected_id:
            return candidate
        if selected_recipe and str(candidate.get("recipe") or "").strip() == selected_recipe:
            return candidate

    recommended_recipe = ""
    if comparison is not None:
        columns = _columns(comparison)
        recipe_idx = _column_index(columns, ("算法", "配方", "模型"), default=0)
        for source_row in comparison.get("rows") or []:
            if not isinstance(source_row, (list, tuple)):
                continue
            row = list(source_row)
            if any("★" in str(cell) for cell in row):
                recommended_recipe = _cell(row, recipe_idx).replace("★", "").strip()
                break
    if recommended_recipe:
        for candidate in candidates:
            if str(candidate.get("recipe") or "").strip() == recommended_recipe:
                return candidate
    return None


def _context(
    *, title: str, milestone: str, facts: list[str], risks: list[str],
    recommendations: list[str], evidence: list[str],
    recommended_features: list[str] | None = None,
    avoid_features: list[str] | None = None,
    recommended_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "milestone": milestone,
        "facts": _unique(facts)[:16],
        "risks": _unique(risks)[:12],
        "recommendations": _unique(recommendations)[:8],
        "evidence": _unique([item for item in evidence if item]),
        "recommended_features": recommended_features or [],
        "avoid_features": avoid_features or [],
        "recommended_params": recommended_params or {},
    }


def _llm_insight(client, context: dict, memory: list[dict]) -> dict | None:
    # Real configured clients expose a profile. Avoid consuming test doubles used
    # only for gate-decision tests and silently fall back if interpretation fails.
    if client is None or not isinstance(getattr(client, "profile", None), dict):
        return None
    prompt = {
        "milestone": context.get("milestone"),
        "facts": context.get("facts") or [],
        "platform_risks": context.get("risks") or [],
        "platform_recommendations": context.get("recommendations") or [],
        "recommended_features": context.get("recommended_features") or [],
        "avoid_features": context.get("avoid_features") or [],
        "recommended_params": context.get("recommended_params") or {},
        "memory_context": [
            {"summary": str(item.get("summary") or ""), "kind": str(item.get("kind") or "raw")}
            for item in memory
        ],
        "instruction": "只解释上述事实；历史记忆只能作为对比或提醒，不能覆盖本次平台指标。",
    }
    try:
        raw = client.complete(
            system_prompt=WORKFLOW_INSIGHT_SYS.text,
            user_prompt=json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            temperature=0.0,
            response_format={"type": "json_object"},
            stream=False,
            caller="workflow_insight",
            prompt_name=WORKFLOW_INSIGHT_SYS.name,
            prompt_version=WORKFLOW_INSIGHT_SYS.version,
        )
        parsed, _error = load_json_object(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict) or not str(parsed.get("summary") or "").strip():
        return None
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "findings": _string_list(parsed.get("findings")),
        "risks": _string_list(parsed.get("risks")),
        "recommendations": _string_list(parsed.get("recommendations")),
    }


def _render_sections(title: str, payload: dict) -> str:
    lines = [f"**{title}**", str(payload.get("summary") or "").strip()]
    for label, key in (("主要发现", "findings"), ("风险与坑点", "risks"), ("建议", "recommendations")):
        items = _string_list(payload.get(key))
        if items:
            lines.append(f"{label}：")
            lines.extend(f"- {item}" for item in items[:8])
    return "\n".join(line for line in lines if line)


def _fallback_summary(context: dict) -> str:
    if context.get("risks"):
        return "平台结果已经完成，Agent 识别到需要优先复核的风险点。"
    return "平台结果已经完成，当前未识别到明显结构性异常。"


def _find_table(tables: list[dict], titles: tuple[str, ...]) -> dict | None:
    for table in tables:
        title = str(table.get("title") or "")
        if any(candidate in title for candidate in titles):
            return table
    return None


def _columns(table: dict) -> list[str]:
    return [str(item) for item in (table.get("columns") or [])]


def _column_index(columns: list[str], labels: tuple[str, ...], default: int | None = None) -> int | None:
    for index, column in enumerate(columns):
        if any(label in column for label in labels):
            return index
    return default


def _cell(row: list, index: int | None) -> str:
    if index is None or index < 0 or index >= len(row):
        return ""
    return str(row[index] if row[index] is not None else "").strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value or "").strip().replace(",", "")
    if not text or text.lower() in {"n/a", "na", "none", "-"}:
        return None
    percent = text.endswith("%")
    try:
        number = float(text.rstrip("%"))
    except ValueError:
        return None
    if percent:
        number /= 100.0
    return number if math.isfinite(number) else None


def _pct_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique([str(item) for item in value])


__all__ = ["build_workflow_insight_context", "render_workflow_insight"]

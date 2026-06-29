"""Structured payload builders for interactive PlanDriver gates."""

from __future__ import annotations


def build_dedup_payload(confirm_o: dict | None, propose_o: dict | None) -> dict | None:
    """Per-feature dedup picker payload for a join gate (§4). Returns None unless
    ``confirm_join`` left features awaiting a dedup strategy. For each such feature,
    attach the conflict-key count + conflicting columns from the propose-step
    diagnostics so the picker shows *why* a strategy is needed."""
    confirm = confirm_o if isinstance(confirm_o, dict) else {}
    needs = [str(f) for f in (confirm.get("needs_dedup") or [])]
    if not needs:
        return None
    info: dict[str, dict] = {}
    propose = propose_o if isinstance(propose_o, dict) else {}
    for join in propose.get("joins") or []:
        if not isinstance(join, dict):
            continue
        fid = str(join.get("feature_id"))
        diag = join.get("diagnostics") if isinstance(join.get("diagnostics"), dict) else {}
        report = diag.get("conflict_report") if isinstance(diag.get("conflict_report"), dict) else {}
        info[fid] = {
            "conflict_keys": int(report.get("n_conflict_keys") or 0),
            "conflict_columns": [str(c) for c in (report.get("conflict_columns") or [])],
        }
    features = [
        {"feature_id": fid, **info.get(fid, {"conflict_keys": 0, "conflict_columns": []})}
        for fid in needs
    ]
    return {"needs_dedup": needs, "features": features, "strategies": ["first", "last"]}


def screen_known_features(output: dict) -> set:
    """Every feature the screen actually saw — scored, ranked, or bucketed into
    leakage/suspected/unusable. Used to constrain an edited selection so it can only
    re-pick among validated columns (force-selecting a flagged one is allowed)."""
    known: set = set()
    o = output if isinstance(output, dict) else {}
    scores = o.get("scores")
    if isinstance(scores, dict):
        known.update(str(k) for k in scores)
    for key in ("ranked", "leakage", "suspected", "unusable"):
        for item in o.get(key) or []:
            if isinstance(item, (list, tuple)) and item:
                known.add(str(item[0]))
            elif isinstance(item, str):
                known.add(item)
    known.update(str(f) for f in (o.get("selected") or []))
    return known


def build_screen_payload(output: dict, dep) -> dict:
    """Structured screening result for the frontend §4 interactive selection table.

    A pass-through of the screen tool output (ranked KS, per-feature scores, the
    leakage/suspected/unusable buckets with reasons) plus (a) the screen step id —
    so an edited selection can be confirmed back against that exact step — and (b)
    the gating thresholds the screen used, so the table's sliders default to them.
    """
    o = output if isinstance(output, dict) else {}
    inputs = getattr(dep, "inputs", None) or {}

    def _flt(key, default):
        try:
            return float(inputs.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "step_id": getattr(dep, "id", None),
        "step_title": getattr(dep, "title", None),
        "selected": list(o.get("selected") or []),
        "ranked": o.get("ranked") or [],
        "leakage": o.get("leakage") or [],
        "suspected": o.get("suspected") or [],
        "unusable": o.get("unusable") or [],
        "scores": o.get("scores") if isinstance(o.get("scores"), dict) else {},
        "n_screened": o.get("n_screened") or 0,
        "thresholds": {
            "leakage_ks": _flt("leakage_ks", 0.40),
            "max_missing_rate": _flt("max_missing_rate", 0.95),
        },
    }


def build_modeling_setup_payload(
    output: dict,
    dep,
    *,
    split_output: dict | None = None,
) -> dict | None:
    """Interactive modeling setup payload.

    The modeling spec step is usually not a pause point by itself; its output is
    rendered at the next gated step. Keep only the small subset the frontend can
    safely adjust at that gate: sample-weight usage from already-detected
    candidates. Free-form explicit columns are handled earlier by task setup,
    where the full schema is available for validation.
    """
    o = output if isinstance(output, dict) else {}
    if not o:
        return None
    candidates = [str(col) for col in (o.get("sample_weight_candidates") or []) if str(col).strip()]
    selected = str(o.get("sample_weight_col") or "").strip()
    if selected and selected not in candidates:
        candidates.insert(0, selected)
    return {
        "step_id": getattr(dep, "id", None),
        "step_title": getattr(dep, "title", None),
        "target_type": str(o.get("target_type") or "binary"),
        "recipe": str(o.get("recipe") or ""),
        "recipes": [str(item) for item in (o.get("recipes") or [])],
        "feature_count": o.get("feature_count"),
        "n_trials": o.get("n_trials"),
        "metric_policy": str(o.get("metric_policy") or ""),
        "eligible_algorithms": [str(item) for item in (o.get("eligible_algorithms") or [])],
        "disabled_algorithms": [
            dict(item)
            for item in (o.get("disabled_algorithms") or [])
            if isinstance(item, dict)
        ],
        "pmml_supported_algorithms": [
            str(item) for item in (o.get("pmml_supported_algorithms") or [])
        ],
        "warnings": [str(item) for item in (o.get("warnings") or []) if str(item)],
        "reason": str(o.get("reason") or ""),
        "split_summary": _split_summary(split_output),
        "sample_weight_col": selected,
        "sample_weight_candidates": candidates,
        "sample_weight_diagnostics": [
            dict(item)
            for item in (o.get("sample_weight_diagnostics") or [])
            if isinstance(item, dict)
        ],
    }


def build_model_delivery_payload(
    output: dict,
    dep,
    *,
    report_output: dict | None = None,
    report_step=None,
) -> dict | None:
    """Structured modeling comparison/delivery payload for late-stage gates."""
    o = output if isinstance(output, dict) else {}
    if not o:
        return None
    tool = str(getattr(getattr(dep, "tool_ref", None), "tool", "") or "")
    if tool not in {"compare_experiments", "select_experiment", "post_training_action"}:
        return None
    capabilities = _capabilities(o.get("capabilities"))
    actions = _delivery_actions(o.get("actions"))
    candidates = _experiment_candidates(o.get("experiments"))
    selected_id = str(o.get("selected_experiment_id") or o.get("experiment_id") or "")
    report = _report_summary(report_output, report_step)
    if selected_id:
        candidates = _mark_selected_candidate(candidates, selected_id)
    return {
        "step_id": getattr(dep, "id", None),
        "step_title": getattr(dep, "title", None),
        "source_tool": tool,
        "selected_experiment_id": selected_id,
        "artifact_id": str(o.get("artifact_id") or ""),
        "recipe": str(o.get("recipe") or ""),
        "target_type": str(o.get("target_type") or ""),
        "selection_metric": str(o.get("selection_metric") or ""),
        "selection_reason": str(o.get("selection_reason") or ""),
        "metrics": _metrics(o.get("metrics")),
        "capabilities": capabilities,
        "candidates": candidates,
        "actions": actions,
        "native_model_path": str(o.get("native_model_path") or ""),
        "pmml_path": str(o.get("pmml_path") or ""),
        "validation_task_id": str(o.get("validation_task_id") or ""),
        "report": report,
        "readiness": _delivery_readiness(o, capabilities, actions, report=report),
    }


def _split_summary(output: dict | None) -> dict | None:
    if not isinstance(output, dict):
        return None
    analysis = output.get("sample_analysis") if isinstance(output.get("sample_analysis"), dict) else {}
    split_counts = {
        str(key): int(value)
        for key, value in (analysis.get("split_counts") or {}).items()
        if str(key)
    }
    if not split_counts:
        return None
    total_rows = analysis.get("total_rows")
    try:
        total_rows = int(total_rows)
    except (TypeError, ValueError):
        total_rows = sum(split_counts.values())
    holdout_values = [str(item) for item in (output.get("holdout_values") or []) if str(item)]
    warnings: list[str] = []
    lowered = {key.lower(): value for key, value in split_counts.items()}
    if lowered.get("train", 0) <= 0:
        warnings.append("缺少 train 样本。")
    if lowered.get("test", 0) <= 0:
        warnings.append("缺少 test 样本。")
    if lowered.get("oot", 0) <= 0:
        warnings.append("缺少 OOT 样本,上线前建议补充时间外验证。")
    if total_rows and lowered.get("oot", 0) / total_rows < 0.05:
        warnings.append("OOT 占比低于 5%,稳定性结论需谨慎。")
    return {
        "split_col": str(output.get("split_col") or ""),
        "split_counts": split_counts,
        "total_rows": total_rows,
        "holdout_values": holdout_values,
        "warnings": warnings,
    }


def _metrics(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): val
        for key, val in value.items()
        if _is_delivery_metric_key(str(key)) and isinstance(val, (int, float, str, bool))
    }


def _is_delivery_metric_key(key: str) -> bool:
    return key.startswith(("train_", "test_", "oot_")) or key in {
        "ks",
        "auc",
        "rmse",
        "mae",
        "r2",
        "accuracy",
        "logloss",
        "feature_count",
        "n_features",
    }


def _capabilities(value) -> dict:
    caps = value if isinstance(value, dict) else {}
    return {
        "pmml_supported": bool(caps.get("pmml_supported")),
        "handoff_supported": bool(caps.get("handoff_supported")),
        "native_model_supported": bool(caps.get("native_model_supported")),
        "reason": str(caps.get("reason") or ""),
    }


def _experiment_candidates(value) -> list[dict]:
    rows = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "id": str(item.get("id") or item.get("experiment_id") or ""),
            "artifact_id": str(item.get("artifact_id") or ""),
            "recipe": str(item.get("recipe") or ""),
            "metrics": _metrics(item),
            "capabilities": _capabilities(item.get("capabilities")),
            "selected": False,
        })
    return rows


def _mark_selected_candidate(candidates: list[dict], selected_id: str) -> list[dict]:
    marked = []
    for item in candidates:
        row = dict(item)
        row["selected"] = row.get("id") == selected_id
        marked.append(row)
    return marked


def _delivery_actions(value) -> list[dict]:
    actions = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        actions.append({
            "action": str(item.get("action") or ""),
            "status": str(item.get("status") or ""),
            "pmml_path": str(item.get("pmml_path") or ""),
            "validation_task_id": str(item.get("validation_task_id") or ""),
            "reason": str(item.get("reason") or ""),
        })
    return actions


def _action(actions: list[dict], name: str) -> dict:
    return next((item for item in actions if item.get("action") == name), {})


def _report_summary(output: dict | None, dep=None) -> dict | None:
    if not isinstance(output, dict):
        return None
    sections = []
    for item in output.get("section_status") or []:
        if not isinstance(item, dict):
            continue
        available = _report_section_available(item)
        sections.append({
            "section": str(item.get("section") or ""),
            "available": available,
            "reason": str(item.get("reason") or ""),
        })
    total = len(sections)
    available_count = sum(1 for item in sections if item.get("available"))
    report_path = str(output.get("report_path") or "")
    if not report_path:
        status = "missing"
    elif total and available_count < total:
        status = "partial"
    else:
        status = "ready"
    return {
        "step_id": getattr(dep, "id", None),
        "step_title": getattr(dep, "title", None),
        "report_path": report_path,
        "available_sections": available_count,
        "total_sections": total,
        "skipped_sections": max(total - available_count, 0),
        "status": status,
        "sections": sections,
    }


def _report_section_available(section: dict) -> bool:
    if section.get("available") is True:
        return True
    status = str(section.get("status") or "").strip().lower()
    return status in {"ok", "ready", "available", "generated", "succeeded"}


def _delivery_readiness(
    output: dict,
    capabilities: dict,
    actions: list[dict],
    *,
    report: dict | None = None,
) -> list[dict]:
    native_path = str(output.get("native_model_path") or "")
    pmml_path = str(output.get("pmml_path") or "")
    validation_task_id = str(output.get("validation_task_id") or "")
    pmml_action = _action(actions, "export_pmml")
    handoff_action = _action(actions, "handoff_to_validation")
    readiness = [
        {
            "id": "native_model",
            "label": "原生模型",
            "status": "ready" if native_path or capabilities.get("native_model_supported") else "missing",
            "artifact": native_path,
            "reason": "",
        },
        {
            "id": "pmml",
            "label": "PMML",
            "status": pmml_action.get("status")
            or ("ready" if capabilities.get("pmml_supported") else "unsupported"),
            "artifact": pmml_path or pmml_action.get("pmml_path", ""),
            "reason": pmml_action.get("reason") or capabilities.get("reason", ""),
        },
        {
            "id": "validation_handoff",
            "label": "验证移交",
            "status": handoff_action.get("status")
            or ("ready" if capabilities.get("handoff_supported") else "unsupported"),
            "artifact": validation_task_id or handoff_action.get("validation_task_id", ""),
            "reason": handoff_action.get("reason") or capabilities.get("reason", ""),
        },
    ]
    if report is not None:
        total = int(report.get("total_sections") or 0)
        available = int(report.get("available_sections") or 0)
        readiness.insert(1, {
            "id": "model_report",
            "label": "模型报告",
            "status": str(report.get("status") or "missing"),
            "artifact": str(report.get("report_path") or ""),
            "reason": f"报告章节 {available}/{total} 可生成" if total else "",
        })
    return readiness


__all__ = [
    "build_dedup_payload",
    "build_model_delivery_payload",
    "build_modeling_setup_payload",
    "build_screen_payload",
    "screen_known_features",
]

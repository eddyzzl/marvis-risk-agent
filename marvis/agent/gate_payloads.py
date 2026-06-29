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


__all__ = [
    "build_dedup_payload",
    "build_modeling_setup_payload",
    "build_screen_payload",
    "screen_known_features",
]

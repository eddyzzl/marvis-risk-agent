"""Structured payload builders for interactive PlanDriver gates."""

from __future__ import annotations

from marvis.modeling_policy_signals import has_monotonic_policy, monotonic_policy_profile


# UX-6: sample_keys/sample_conflicts are already capped (dedup._SAMPLE_CAP=50) by the
# backend two_level_dedup computation; re-cap here to a much smaller number of *example*
# keys shown at the gate (the picker's job is "show why", not dump the whole sample).
_DEDUP_EXAMPLE_CAP = 3


def _dedup_examples(sample_keys: list, sample_conflicts: list, key_columns: list[str]) -> list[dict]:
    examples = []
    for key_tuple, values in zip(sample_keys, sample_conflicts):
        if not isinstance(values, dict) or not values:
            continue
        key_label = ", ".join(
            f"{col}={val}" for col, val in zip(key_columns, key_tuple)
        ) if key_columns else str(key_tuple)
        examples.append({"key": key_label, "values": values})
        if len(examples) >= _DEDUP_EXAMPLE_CAP:
            break
    return examples


def build_dedup_payload(confirm_o: dict | None, propose_o: dict | None) -> dict | None:
    """Per-feature dedup picker payload for a join gate (§4). Returns None unless
    ``confirm_join`` left features awaiting a dedup strategy. For each such feature,
    attach the conflict-key count + conflicting columns + a few real conflict-value
    examples from the propose-step diagnostics so the picker shows *why* a strategy is
    needed and what first/last would each keep (UX-6: the picker previously showed only
    a bare conflict count, dropping the conflict_columns/sample data the backend already
    computes)."""
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
        key_columns = [str(c) for c in (report.get("key_columns") or [])]
        sample_keys = list(report.get("sample_keys") or [])
        sample_conflicts = list(report.get("sample_conflicts") or [])
        info[fid] = {
            "conflict_keys": int(report.get("n_conflict_keys") or 0),
            "conflict_columns": [str(c) for c in (report.get("conflict_columns") or [])],
            "examples": _dedup_examples(sample_keys, sample_conflicts, key_columns),
        }
    features = [
        {
            "feature_id": fid,
            **info.get(fid, {"conflict_keys": 0, "conflict_columns": [], "examples": []}),
        }
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

    UX-4/VD-7: also passes through the watch-band lists (ks_decay_watch/psi_watch/
    leakage_watch/split_shift) and the categorical-column notes (sentinel_columns/
    excluded_categorical/suspected_categorical) the screen already computes
    (marvis/feature/screen.py, marvis/packs/modeling/tools.py:tool_screen_features)
    but the table previously dropped, so the frontend's watch/category-column chip
    filters have real data instead of only the four hard-cut buckets.
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
        "leakage_watch": o.get("leakage_watch") or [],
        "ks_decay_watch": o.get("ks_decay_watch") or [],
        "psi_watch": o.get("psi_watch") or [],
        "split_shift": o.get("split_shift") or [],
        "sentinel_columns": o.get("sentinel_columns") if isinstance(o.get("sentinel_columns"), dict) else {},
        "excluded_categorical": o.get("excluded_categorical") or [],
        "suspected_categorical": o.get("suspected_categorical") or [],
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
    split_summary = _split_summary(split_output)
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
        "split_summary": split_summary,
        "sample_weight_col": selected,
        "sample_weight_candidates": candidates,
        "sample_weight_diagnostics": [
            dict(item)
            for item in (o.get("sample_weight_diagnostics") or [])
            if isinstance(item, dict)
        ],
        "override_guidance": _modeling_override_guidance(
            o,
            split_summary=split_summary,
            sample_weight_candidates=candidates,
            sample_weight_col=selected,
        ),
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
    selected_candidate = next((item for item in candidates if item.get("selected")), None)
    policy_signals = (
        dict(selected_candidate.get("policy_signals"))
        if isinstance(selected_candidate, dict) and isinstance(selected_candidate.get("policy_signals"), dict)
        else _policy_signals(o)
    )
    policy_decision = _policy_decision(o.get("policy_decision"))
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
        "business_signals": (
            dict(selected_candidate.get("business_signals"))
            if isinstance(selected_candidate, dict) and isinstance(selected_candidate.get("business_signals"), dict)
            else _business_signals(o)
        ),
        "policy_signals": policy_signals,
        "policy_decision": policy_decision,
        "capabilities": capabilities,
        "candidates": candidates,
        "actions": actions,
        "native_model_path": str(o.get("native_model_path") or ""),
        "pmml_path": str(o.get("pmml_path") or ""),
        "validation_task_id": str(o.get("validation_task_id") or ""),
        "challenger_task_id": str(o.get("challenger_task_id") or ""),
        "challenger_package_path": str(o.get("challenger_package_path") or ""),
        "challenger_package_markdown_path": str(o.get("challenger_package_markdown_path") or ""),
        "approval_package_path": str(o.get("approval_package_path") or ""),
        "approval_package_markdown_path": str(o.get("approval_package_markdown_path") or ""),
        "monitoring_policy_path": str(o.get("monitoring_policy_path") or ""),
        "monitoring_policy_markdown_path": str(o.get("monitoring_policy_markdown_path") or ""),
        "monitoring_policy": (
            dict(o.get("monitoring_policy"))
            if isinstance(o.get("monitoring_policy"), dict)
            else {}
        ),
        "model_card_path": str(o.get("model_card_path") or ""),
        "model_card_markdown_path": str(o.get("model_card_markdown_path") or ""),
        "model_card": (
            dict(o.get("model_card"))
            if isinstance(o.get("model_card"), dict)
            else {}
        ),
        "challenger_comparison_path": str(o.get("challenger_comparison_path") or ""),
        "challenger_comparison_markdown_path": str(
            o.get("challenger_comparison_markdown_path") or ""
        ),
        "challenger_comparison": (
            dict(o.get("challenger_comparison"))
            if isinstance(o.get("challenger_comparison"), dict)
            else {}
        ),
        "report": report,
        "readiness": _delivery_readiness(
            o,
            capabilities,
            actions,
            report=report,
            policy_signals=policy_signals,
            policy_decision=policy_decision,
        ),
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
        warnings.append("缺少 OOT 样本，上线前建议补充时间外验证。")
    if total_rows and lowered.get("oot", 0) / total_rows < 0.05:
        warnings.append("OOT 占比低于 5%，稳定性结论需谨慎。")
    return {
        "split_col": str(output.get("split_col") or ""),
        "split_counts": split_counts,
        "total_rows": total_rows,
        "holdout_values": holdout_values,
        "warnings": warnings,
    }


def _modeling_override_guidance(
    output: dict,
    *,
    split_summary: dict | None,
    sample_weight_candidates: list[str],
    sample_weight_col: str,
) -> list[dict]:
    """Business guidance shown before users change modeling setup controls."""
    target_type = str(output.get("target_type") or "binary")
    recipes = [str(item) for item in (output.get("recipes") or []) if str(item)]
    pmml_supported = {str(item) for item in (output.get("pmml_supported_algorithms") or []) if str(item)}
    disabled_algorithms = [
        item
        for item in (output.get("disabled_algorithms") or [])
        if isinstance(item, dict)
    ]
    diagnostics = [
        item
        for item in (output.get("sample_weight_diagnostics") or [])
        if isinstance(item, dict)
    ]
    guidance: list[dict] = []
    target_messages = {
        "binary": "二分类适合好/坏、通过/拒绝等 0/1 风控标签；切换到回归或多分类会同步改变指标、报告章节和可交付物。",
        "continuous": "回归适合金额、额度、损失率等连续目标；不会使用 KS/AUC 作为主评估口径，报告也会走回归指标。",
        "multiclass": "多分类适合风险等级或评级标签；指标、PMML 支持和验证移交通常比二分类更受限制。",
    }
    guidance.append({
        "id": "target_type",
        "label": "目标类型",
        "level": "info",
        "message": target_messages.get(target_type, target_messages["binary"]),
    })
    if recipes:
        non_pmml = [recipe for recipe in recipes if recipe not in pmml_supported]
        if non_pmml:
            guidance.append({
                "id": "recipes",
                "label": "算法组合",
                "level": "warning",
                "message": (
                    f"当前选择包含仅原生模型算法 {', '.join(non_pmml)}；需要 PMML 或验证移交时请确认替代算法或交付方案。"
                ),
            })
        else:
            guidance.append({
                "id": "recipes",
                "label": "算法组合",
                "level": "info",
                "message": (
                    f"当前算法 {', '.join(recipes)} 均可导出 PMML；仍需一起比较效果、稳定性、特征复杂度和交付形态。"
                ),
            })
    if disabled_algorithms:
        guidance.append({
            "id": "disabled_algorithms",
            "label": "不可用算法",
            "level": "review",
            "message": (
                f"有 {len(disabled_algorithms)} 个算法因当前目标或依赖条件不可用；切换目标类型后需要重新确认算法家族和下游报告口径。"
            ),
        })
    n_trials = _safe_int(output.get("n_trials"))
    if n_trials is not None:
        total_rows = _safe_int(split_summary.get("total_rows")) if isinstance(split_summary, dict) else None
        budgets = output.get("n_trials_by_recipe") if isinstance(output.get("n_trials_by_recipe"), dict) else {}
        if len(budgets) > 1:
            # Multi-algorithm comparison (TUNE-1/SEL-2): every recipe now tunes with
            # its own budget — total search cost is the SUM of each recipe's
            # n_trials, so surface that instead of a single scalar's guidance.
            total_budget = sum(int(v) for v in budgets.values())
            budget_note = "、".join(f"{recipe}={value}" for recipe, value in budgets.items())
            guidance.append({
                "id": "n_trials",
                "label": "调参预算",
                "level": "info",
                "message": (
                    f"多算法对比:按算法预算 {budget_note}；多算法总预算=Σ各配方预算={total_budget} 轮"
                    "（树模型 lgb/xgb/catboost 默认各 40 轮两阶段搜索，lr/scorecard/mlp 默认各 12 轮；"
                    "总预算越大，本步运行耗时越长，可按需在本门调整每个算法的预算）。"
                ),
            })
        else:
            guidance.append({
                "id": "n_trials",
                "label": "调参预算",
                **_n_trials_guidance_by_scale(n_trials, total_rows),
            })
        cv_guidance = _cv_folds_guidance_by_scale(output.get("cv_folds"), total_rows)
        if cv_guidance is not None:
            guidance.append({"id": "cv_folds", "label": "交叉验证", **cv_guidance})
    invalid_weights = [
        str(item.get("column") or "")
        for item in diagnostics
        if item.get("valid") is False and str(item.get("column") or "")
    ]
    if sample_weight_col:
        guidance.append({
            "id": "sample_weight",
            "label": "样本权重",
            "level": "review",
            "message": (
                f"权重列 {sample_weight_col} 会改变拟合目标且不会入模；请确认它来自抽样、拒绝推断或业务权重，而不是贷后结果泄漏。"
            ),
        })
    elif sample_weight_candidates:
        guidance.append({
            "id": "sample_weight",
            "label": "样本权重",
            "level": "info",
            "message": (
                f"检测到候选权重列 {', '.join(sample_weight_candidates)}；默认不使用，除非样本抽样、拒绝推断或业务策略明确需要加权。"
            ),
        })
    if invalid_weights:
        guidance.append({
            "id": "sample_weight_quality",
            "label": "权重质量",
            "level": "warning",
            "message": f"权重列 {', '.join(invalid_weights)} 存在非正数、缺失或不可用问题，使用前需要先清洗或重新选择。",
        })
    split_warnings = []
    if isinstance(split_summary, dict):
        split_warnings = [str(item) for item in (split_summary.get("warnings") or []) if str(item)]
    if split_warnings:
        guidance.append({
            "id": "split_quality",
            "label": "样本切分",
            "level": "warning",
            "message": "；".join(split_warnings),
        })
    return guidance


def _n_trials_guidance_by_scale(n_trials: int, total_rows: int | None) -> dict:
    """Data-scale-aware n_trials guidance (TUNE-2): recommend a budget instead of
    discouraging larger ones. Small samples (<50k rows) can afford a thorough
    search cheaply; larger samples get a runtime heads-up, not a ceiling."""
    small_sample = total_rows is not None and total_rows < 50_000
    recommended = "40-80" if small_sample else "40 起，按数据规模和可用时间上调"
    if n_trials < 5:
        return {
            "level": "warning",
            "message": (
                f"当前调参轮数较少，适合快速烟测；正式候选模型建议 {recommended} 轮起步，或记录人工原因。"
            ),
        }
    if small_sample:
        message = (
            f"当前调参轮数 {n_trials}；当前数据规模（<5万行）适合 40-80 轮的两阶段搜索"
            "（粗搜定区域、细搜收敛），充分搜索的运行成本通常可接受。"
        )
    elif total_rows is not None:
        message = (
            f"当前调参轮数 {n_trials}；数据规模较大，扩大搜索预算会显著增加运行耗时，"
            "建议先用 40 轮左右两阶段搜索评估收益，再按可用时间上调。"
        )
    else:
        message = f"当前调参轮数 {n_trials}；两阶段搜索（粗搜+细搜）下可按需扩大预算以换取更充分的收敛。"
    return {"level": "info", "message": message}


def _cv_folds_guidance_by_scale(cv_folds, total_rows: int | None) -> dict | None:
    """TUNE-3: recommend enabling grouped cross-validation on small samples, where
    a single train/test split's KS is noisy enough to make trial selection and
    champion comparison unreliable. Returns None when there is nothing worth
    saying (cv_folds already set and sample isn't tiny)."""
    resolved = _safe_int(cv_folds)
    very_small_sample = total_rows is not None and total_rows < 5_000
    small_sample = total_rows is not None and total_rows < 20_000
    if resolved and resolved >= 2:
        return {
            "level": "info",
            "message": (
                f"已启用 {resolved} 折分组交叉验证；每轮 trial 耗时约为单一切分的 {resolved} 倍，"
                "换来更稳健的选参/选冠军口径（folds 间方差越小说明参数越稳）。"
            ),
        }
    if very_small_sample:
        return {
            "level": "warning",
            "message": (
                "当前数据规模很小（<5000 行），单一 train/test 切分的 KS 抽样噪声可能超过参数差异本身；"
                "建议设置 cv_folds=5（分组感知）换取更可信的选参/选冠军结果，耗时约为不开启时的 5 倍。"
            ),
        }
    if small_sample:
        return {
            "level": "info",
            "message": (
                "当前数据规模较小（<2万行），建议设置 cv_folds=3（分组感知）降低单一切分的选择噪声；"
                "耗时约为不开启时的 3 倍。"
            ),
        }
    return None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_flag",
    }


def _capabilities(value) -> dict:
    caps = value if isinstance(value, dict) else {}
    calibration = caps.get("calibration") if isinstance(caps.get("calibration"), dict) else {}
    limitations = [
        str(item)
        for item in (caps.get("limitations") or [])
        if str(item)
    ]
    return {
        "pmml_supported": bool(caps.get("pmml_supported")),
        "handoff_supported": bool(caps.get("handoff_supported")),
        "native_model_supported": bool(caps.get("native_model_supported")),
        "reason": str(caps.get("reason") or ""),
        "calibrated": bool(caps.get("calibrated")),
        "calibration": dict(calibration),
        "pmml_includes_calibration": bool(caps.get("pmml_includes_calibration", True)),
        "limitations": limitations,
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
            "business_signals": _business_signals(item),
            "policy_signals": _policy_signals(item),
            "selected": False,
        })
    return rows


def _business_signals(row: dict | None) -> dict:
    item = row if isinstance(row, dict) else {}
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else item
    caps = _capabilities(item.get("capabilities"))
    feature_count = _first_number(metrics, ("feature_count", "n_features"))
    if feature_count is None:
        features = item.get("features") or item.get("feature_list")
        feature_count = len(features) if isinstance(features, list) else None
    psi_oot = _first_number(metrics, ("psi_oot_vs_train",))
    psi_test = _first_number(metrics, ("psi_test_vs_train",))
    stability_gap = _stability_gap(metrics)
    overfit_flag = metrics.get("overfit_flag") if isinstance(metrics, dict) else None
    return {
        "feature_count": feature_count,
        "stability": _stability_label(psi_oot, psi_test, stability_gap, overfit_flag),
        "stability_value": psi_oot if psi_oot is not None else psi_test,
        "generalization_gap": stability_gap,
        "overfit_flag": bool(overfit_flag) if overfit_flag is not None else False,
        "calibration": _calibration_label(item),
        "delivery": _delivery_label(caps),
    }


def _first_number(metrics: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _stability_gap(metrics: dict) -> float | None:
    if not isinstance(metrics, dict):
        return None
    pairs = (("test_ks", "oot_ks"), ("test_auc", "oot_auc"), ("test_rmse", "oot_rmse"))
    for left, right in pairs:
        a = metrics.get(left)
        b = metrics.get(right)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b))
    return None


def _stability_label(
    psi_oot: float | None,
    psi_test: float | None,
    gap: float | None,
    overfit_flag,
) -> str:
    if overfit_flag:
        return "需复核"
    psi = psi_oot if psi_oot is not None else psi_test
    if psi is not None:
        if psi >= 0.25:
            return "高风险"
        if psi >= 0.10:
            return "关注"
        return "稳定"
    if gap is not None:
        if gap >= 0.10:
            return "关注"
        return "稳定"
    return "待评估"


def _calibration_label(row: dict) -> str:
    calibration = row.get("calibration") if isinstance(row.get("calibration"), dict) else {}
    if calibration:
        includes_pmml = calibration.get("pmml_includes_calibration")
        return "已校准（PMML不含）" if includes_pmml is False else "已校准"
    caps = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
    reason = str(caps.get("reason") or "")
    if "calibration" in reason.lower() or "校准" in reason:
        return "需说明"
    return "未校准"


def _delivery_label(caps: dict) -> str:
    if caps.get("pmml_supported") and caps.get("handoff_supported"):
        return "可移交"
    if caps.get("native_model_supported"):
        return "仅原生"
    return "不可交付"


def _capability_note(caps: dict) -> str:
    limitations = [
        str(item)
        for item in (caps.get("limitations") or [])
        if str(item)
    ]
    if limitations:
        return " ".join(limitations)
    return str(caps.get("reason") or "")


def _policy_signals(row: dict | None) -> dict:
    item = row if isinstance(row, dict) else {}
    caps = _capabilities(item.get("capabilities"))
    business = _business_signals(item)
    recipe = str(item.get("recipe") or "")
    scorecard_rows = item.get("scorecard_table") if isinstance(item.get("scorecard_table"), list) else []
    scorecard_like = recipe == "scorecard" or bool(scorecard_rows)
    monotonic_profile = monotonic_policy_profile(item, scorecard_rows)
    monotonic_declared = bool(monotonic_profile.get("monotonicity_declared"))
    stability = str(business.get("stability") or "")
    delivery = str(business.get("delivery") or "")
    reasons: list[str] = []

    if scorecard_like:
        scorecard = "评分卡"
        scorecard_status = "ready"
        if scorecard_rows:
            reasons.append(f"评分卡表 {len(scorecard_rows)} 行")
    else:
        scorecard = "非评分卡"
        scorecard_status = "neutral"

    if monotonic_declared:
        monotonicity = "已约束"
        monotonicity_status = "ready"
    elif scorecard_like:
        monotonicity = "需确认"
        monotonicity_status = "warning"
        missing = monotonic_profile.get("monotonicity_missing_features")
        if isinstance(missing, list) and missing:
            reasons.append(f"评分卡部分特征缺少单调性方向证据: {', '.join(str(item) for item in missing[:8])}")
        else:
            reasons.append("评分卡缺少单调性方向证据")
    else:
        monotonicity = "未声明"
        monotonicity_status = "neutral"

    if stability in {"高风险", "需复核"}:
        approval = "需业务复核"
        approval_status = "warning"
        reasons.append("稳定性或过拟合信号需要复核")
    elif delivery == "不可交付":
        approval = "不可审批"
        approval_status = "error"
        reasons.append("缺少可交付模型产物")
    elif caps.get("handoff_supported") and delivery == "可移交" and monotonicity_status != "warning":
        approval = "建议可审批"
        approval_status = "ready"
    elif caps.get("native_model_supported"):
        approval = "仅实验候选"
        approval_status = "warning"
        reasons.append("交付或验证移交能力受限")
    else:
        approval = "待评估"
        approval_status = "neutral"

    return {
        "scorecard": scorecard,
        "scorecard_status": scorecard_status,
        "monotonicity": monotonicity,
        "monotonicity_status": monotonicity_status,
        "approval": approval,
        "approval_status": approval_status,
        "reasons": reasons,
    }


def _has_monotonic_policy(item: dict, scorecard_rows: list) -> bool:
    return has_monotonic_policy(item, scorecard_rows)


def _policy_decision(value) -> dict:
    decision = value if isinstance(value, dict) else {}
    if not decision:
        return {}
    violations = []
    for item in decision.get("violations") or []:
        if not isinstance(item, dict):
            continue
        violations.append({
            "code": str(item.get("code") or ""),
            "message": str(item.get("message") or ""),
        })
    profile = decision.get("profile") if isinstance(decision.get("profile"), dict) else {}
    policy = decision.get("policy") if isinstance(decision.get("policy"), dict) else {}
    return {
        "status": str(decision.get("status") or ""),
        "explicit_selection": bool(decision.get("explicit_selection")),
        "selected_experiment_id": str(decision.get("selected_experiment_id") or ""),
        "policy": {
            str(key): value
            for key, value in policy.items()
            if isinstance(value, (str, int, float, bool))
        },
        "profile": {
            str(key): value
            for key, value in profile.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "violations": violations,
        "override_reason": str(decision.get("override_reason") or ""),
    }


def _policy_decision_readiness(policy_decision: dict) -> tuple[str, str] | None:
    if not isinstance(policy_decision, dict) or not policy_decision:
        return None
    status = str(policy_decision.get("status") or "")
    if status == "accepted":
        return "ready", "策略门控已通过"
    if status == "overridden":
        reason = str(policy_decision.get("override_reason") or "已人工 override")
        return "warning", reason
    if status == "blocked":
        messages = [
            str(item.get("message") or item.get("code") or "")
            for item in (policy_decision.get("violations") or [])
            if isinstance(item, dict)
        ]
        return "error", "; ".join(item for item in messages if item) or "策略门控未通过"
    if status == "not_requested":
        return "neutral", "未启用执行策略"
    return "neutral", status or "待评估"


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
            "challenger_task_id": str(item.get("challenger_task_id") or ""),
            "package_path": str(item.get("package_path") or ""),
            "markdown_path": str(item.get("markdown_path") or ""),
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
    policy_signals: dict | None = None,
    policy_decision: dict | None = None,
) -> list[dict]:
    native_path = str(output.get("native_model_path") or "")
    approval_package_path = str(output.get("approval_package_path") or "")
    approval_package_markdown_path = str(output.get("approval_package_markdown_path") or "")
    monitoring_policy_path = str(output.get("monitoring_policy_path") or "")
    monitoring_policy_markdown_path = str(output.get("monitoring_policy_markdown_path") or "")
    monitoring_policy = output.get("monitoring_policy") if isinstance(output.get("monitoring_policy"), dict) else {}
    model_card_path = str(output.get("model_card_path") or "")
    model_card_markdown_path = str(output.get("model_card_markdown_path") or "")
    model_card = output.get("model_card") if isinstance(output.get("model_card"), dict) else {}
    challenger_comparison_path = str(output.get("challenger_comparison_path") or "")
    challenger_comparison_markdown_path = str(
        output.get("challenger_comparison_markdown_path") or ""
    )
    challenger_comparison = (
        output.get("challenger_comparison")
        if isinstance(output.get("challenger_comparison"), dict)
        else {}
    )
    pmml_path = str(output.get("pmml_path") or "")
    validation_task_id = str(output.get("validation_task_id") or "")
    challenger_task_id = str(output.get("challenger_task_id") or "")
    challenger_package_markdown_path = str(output.get("challenger_package_markdown_path") or "")
    challenger_package_path = str(output.get("challenger_package_path") or "")
    pmml_action = _action(actions, "export_pmml")
    handoff_action = _action(actions, "handoff_to_validation")
    challenger_action = _action(actions, "create_challenger_backtest")
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
            "reason": pmml_action.get("reason") or _capability_note(capabilities),
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
    if challenger_action or challenger_task_id or challenger_package_markdown_path or challenger_package_path:
        readiness.append({
            "id": "challenger_backtest",
            "label": "Challenger/Backtest",
            "status": challenger_action.get("status")
            or ("ready" if capabilities.get("handoff_supported") else "unsupported"),
            "artifact": (
                challenger_task_id
                or challenger_action.get("challenger_task_id", "")
                or challenger_package_markdown_path
                or challenger_package_path
                or challenger_action.get("markdown_path", "")
                or challenger_action.get("package_path", "")
            ),
            "reason": challenger_action.get("reason") or capabilities.get("reason", ""),
        })
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
    if approval_package_path:
        insert_at = 2 if report is not None else 1
        readiness.insert(insert_at, {
            "id": "approval_package",
            "label": "审批包",
            "status": "ready",
            "artifact": approval_package_markdown_path or approval_package_path,
            "reason": "模型审批与交付证据已生成",
        })
    if model_card_path:
        insert_at = next(
            (idx + 1 for idx, item in enumerate(readiness) if item.get("id") == "approval_package"),
            len(readiness),
        )
        readiness.insert(insert_at, {
            "id": "model_card",
            "label": "模型卡",
            "status": str(model_card.get("status") or "ready"),
            "artifact": model_card_markdown_path or model_card_path,
            "reason": "最终模型卡已生成",
        })
    if monitoring_policy_path:
        insert_after = "model_card" if model_card_path else "approval_package"
        insert_at = next(
            (idx + 1 for idx, item in enumerate(readiness) if item.get("id") == insert_after),
            len(readiness),
        )
        readiness.insert(insert_at, {
            "id": "monitoring_policy",
            "label": "监控策略",
            "status": str(monitoring_policy.get("status") or "ready"),
            "artifact": monitoring_policy_markdown_path or monitoring_policy_path,
            "reason": str(monitoring_policy.get("recommendation") or "监控阈值策略已生成"),
        })
    if challenger_comparison_path:
        insert_after = (
            "monitoring_policy"
            if monitoring_policy_path
            else "model_card"
            if model_card_path
            else "approval_package"
        )
        insert_at = next(
            (idx + 1 for idx, item in enumerate(readiness) if item.get("id") == insert_after),
            len(readiness),
        )
        readiness.insert(insert_at, {
            "id": "challenger_comparison",
            "label": "Champion对比",
            "status": str(challenger_comparison.get("status") or "ready"),
            "artifact": challenger_comparison_markdown_path or challenger_comparison_path,
            "reason": str(challenger_comparison.get("recommendation") or "Champion/Challenger 对比已生成"),
        })
    if isinstance(policy_signals, dict) and policy_signals:
        decision_readiness = _policy_decision_readiness(policy_decision or {})
        status = decision_readiness[0] if decision_readiness else str(policy_signals.get("approval_status") or "neutral")
        reason = decision_readiness[1] if decision_readiness else str(policy_signals.get("approval") or "待评估")
        readiness.append({
            "id": "approval_policy",
            "label": "审批策略",
            "status": status,
            "artifact": "",
            "reason": reason,
        })
    return readiness


__all__ = [
    "build_dedup_payload",
    "build_model_delivery_payload",
    "build_modeling_setup_payload",
    "build_screen_payload",
    "screen_known_features",
]

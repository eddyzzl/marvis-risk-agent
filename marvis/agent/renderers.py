"""Tool-output renderers for V2 plan-driver messages.

Each renderer turns a tool's raw output into ``(markdown_text, table_blocks)``.
Keeping this registry outside ``plan_driver.py`` lets the driver focus on the
execution loop and gate controls while task/domain-specific presentation lives
in one small module.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# tool -> table registry (decision #4 in the driver spec)
# Each renderer turns a tool's raw output into (markdown text, [table dicts]).
# Task differences land HERE; the driver loop above stays task-agnostic. A table
# dict is {title, columns, rows} — the frontend maps it onto renderMetricTableSection.
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _names(items) -> list[str]:
    out = []
    for item in items or []:
        out.append(item[0] if isinstance(item, (list, tuple)) and item else item)
    return [str(x) for x in out]


def _pct(value):
    """Missing-rate as a percentage string; ``None`` → n/a."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _range_text(minimum, maximum) -> str:
    if minimum is None and maximum is None:
        return "n/a"
    return f"{_fmt(minimum)} - {_fmt(maximum)}"


def _triple(item):
    """A (feature, ks, reason) row from a leakage/suspected entry, tolerant of shape."""
    if isinstance(item, (list, tuple)):
        feat = str(item[0]) if len(item) > 0 else ""
        ks = item[1] if len(item) > 1 else None
        reason = str(item[2]) if len(item) > 2 else ""
        return feat, ks, reason
    return str(item), None, ""


def _key_label(column, dictionary: dict) -> str:
    """A raw key-column code, appended with "（含义: ...）" when the task's data
    dictionary has a business-name entry for it (GAP-4); the bare code otherwise."""
    name = str(column) if column is not None else "?"
    meaning = dictionary.get(name) if dictionary else None
    return f"{name}（含义:{meaning}）" if meaning else name


def _render_screen(o: dict):
    selected = o.get("selected") or []
    leak = o.get("leakage") or []
    susp = o.get("suspected") or []
    unusable = o.get("unusable") or []
    excluded_categorical = o.get("excluded_categorical") or []
    scores = o.get("scores") if isinstance(o.get("scores"), dict) else {}
    leak_names = _names(leak)
    susp_names = _names(susp)
    n = o.get("n_screened") or o.get("n") or (len(selected) + len(leak) + len(susp))
    text = (
        f"**特征筛选完成**:从 {n} 个候选中提议保留 **{len(selected)}** 个特征。\n"
        f"- 剔除疑似**泄漏** {len(leak_names)} 个"
        + (f"（如 {leak_names[:3]}）" if leak_names else "")
        + "\n"
        f"- 疑似**模型输出/评分**列 {len(susp_names)} 个"
        + (f"（如 {susp_names[:5]}）" if susp_names else "")
        + "\n"
        f"- 剔除**不可用**（常量/稀疏） {len(unusable)} 个"
    )
    if excluded_categorical:
        preview = "、".join(
            f"{item.get('column')}（基数{item.get('cardinality')}）"
            for item in excluded_categorical[:8]
            if isinstance(item, dict)
        )
        more = (
            f" 等共 {len(excluded_categorical)} 个"
            if len(excluded_categorical) > 8
            else ""
        )
        text += (
            f"\n- **{len(excluded_categorical)} 个类别列未入模**:{preview}{more}；"
            "如需使用，请先用 woe_encode_categorical 编码，或改用 catboost（原生支持类别列）。"
        )
    nan_dropped = o.get("nan_labels_dropped") or 0
    if nan_dropped:
        text += (
            f"\n- 🚩 **{nan_dropped} 行标签为空（NaN）已按确认丢弃**，"
            "本次筛选的 KS/IV 与泄漏判定仅基于有标签样本。"
        )
    tables = []
    if selected:
        rows = []
        for feat in selected[:20]:
            s = scores.get(feat) if isinstance(scores.get(feat), dict) else {}
            rows.append(
                [
                    feat,
                    _num(s.get("ks")),
                    _num(s.get("iv")),
                    _pct(s.get("missing_rate")),
                ]
            )
        tables.append(
            {
                "title": "入选特征（前20）",
                "columns": ["特征", "KS", "IV", "缺失率"],
                "rows": rows,
            }
        )
    if leak:
        tables.append(
            {
                "title": f"疑似泄漏（KS≥阈值，共{len(leak)}）",
                "columns": ["特征", "KS", "原因"],
                "rows": [
                    [f, _num(k), r] for f, k, r in (_triple(i) for i in leak[:20])
                ],
            }
        )
    if susp:
        tables.append(
            {
                "title": f"疑似模型输出/评分列（共{len(susp)}）",
                "columns": ["特征", "KS", "原因"],
                "rows": [
                    [f, _num(k), r] for f, k, r in (_triple(i) for i in susp[:20])
                ],
            }
        )
    if unusable:
        rows = []
        for item in unusable[:20]:
            if isinstance(item, (list, tuple)):
                rows.append(
                    [
                        str(item[0]) if item else "",
                        str(item[1]) if len(item) > 1 else "",
                    ]
                )
            else:
                rows.append([str(item), ""])
        tables.append(
            {
                "title": f"剔除·不可用（常量/稀疏，共{len(unusable)}）",
                "columns": ["特征", "原因"],
                "rows": rows,
            }
        )
    return text, tables


def _render_select(o: dict):
    """FS-1 multivariate refinement gate: IV floor + correlation (+ optional VIF) funnel
    between the sanity-level screen and tuning. Surfaces per-stage drop counts/reasons so
    the confirm gate reads as a funnel, not just a final list."""
    selected = o.get("selected") or []
    dropped = o.get("dropped") or []
    scores = o.get("scores") if isinstance(o.get("scores"), dict) else {}
    n_in = len(selected) + len(dropped)
    low_iv = [
        item
        for item in dropped
        if isinstance(item, (list, tuple))
        and len(item) > 1
        and "low" in str(item[1])
        and "IV" in str(item[1])
    ]
    collinear = [
        item
        for item in dropped
        if isinstance(item, (list, tuple))
        and len(item) > 1
        and "collinear" in str(item[1])
    ]
    high_vif = [
        item
        for item in dropped
        if isinstance(item, (list, tuple)) and len(item) > 1 and "VIF" in str(item[1])
    ]
    top_k_dropped = [
        item
        for item in dropped
        if isinstance(item, (list, tuple)) and len(item) > 1 and "top_k" in str(item[1])
    ]
    other = [
        item
        for item in dropped
        if item not in low_iv
        and item not in collinear
        and item not in high_vif
        and item not in top_k_dropped
    ]
    text = (
        f"**精选特征完成**:从 {n_in} 个候选中精选出 **{len(selected)}** 个特征"
        f"（淘汰 {len(dropped)} 个）。\n"
        f"- IV 底线淘汰 {len(low_iv)} 个\n"
        f"- 相关性去冗余淘汰 {len(collinear)} 个\n"
        f"- 高 VIF 淘汰 {len(high_vif)} 个\n"
        f"- 超出 top_k 淘汰 {len(top_k_dropped)} 个"
    )
    if other:
        text += f"\n- 其他原因淘汰 {len(other)} 个"
    fit_rows = o.get("fit_rows")
    fit_split = o.get("fit_split")
    if fit_rows is not None:
        text += f"\n\n统计口径:{fit_split or 'train'} 上 {fit_rows} 行。"
    tables = []
    if selected:
        rows = []
        for feat in selected[:20]:
            s = scores.get(feat) if isinstance(scores.get(feat), dict) else {}
            rows.append([feat, _num(s.get("iv")), _num(s.get("ks"))])
        tables.append(
            {
                "title": f"最终清单（前20，共{len(selected)}）",
                "columns": ["特征", "IV", "KS"],
                "rows": rows,
            }
        )
    if dropped:
        rows = []
        for item in dropped[:30]:
            if isinstance(item, (list, tuple)) and item:
                feat = str(item[0])
                reason = str(item[1]) if len(item) > 1 else ""
            else:
                feat, reason = str(item), ""
            rows.append([feat, reason])
        tables.append(
            {
                "title": f"淘汰清单（前30，共{len(dropped)}）",
                "columns": ["特征", "原因"],
                "rows": rows,
            }
        )
    return text, tables


def _render_choose_modeling_spec(o: dict):
    recipes = [str(item) for item in (o.get("recipes") or [])]
    target_type = str(o.get("target_type") or "binary")
    sample_weight_col = str(o.get("sample_weight_col") or "")
    metric_policy = str(o.get("metric_policy") or "")
    text = (
        f"**建模规格已生成**:目标类型 `{target_type}`，"
        f"算法 {'/'.join(recipes) or '-'}，选择策略 `{metric_policy}`。"
    )
    tables = [
        {
            "title": "建模规格",
            "columns": ["项目", "值"],
            "rows": [
                ["目标类型", target_type],
                ["主调参算法", str(o.get("recipe") or "")],
                ["训练算法", "/".join(recipes)],
                ["样本权重列", sample_weight_col or "不使用"],
                ["候选特征数", _fmt(o.get("feature_count", ""))],
                ["调参轮数", _fmt(o.get("n_trials", ""))],
                ["选择指标", metric_policy],
            ],
        }
    ]
    eligible = o.get("eligible_algorithms") or []
    disabled = [
        item for item in (o.get("disabled_algorithms") or []) if isinstance(item, dict)
    ]
    if eligible or disabled:
        tables.append(
            {
                "title": "算法可用性",
                "columns": ["算法", "状态", "说明"],
                "rows": (
                    [[str(recipe), "可用", ""] for recipe in eligible]
                    + [
                        [
                            str(item.get("recipe", "")),
                            "不可用",
                            str(item.get("reason", "")),
                        ]
                        for item in disabled
                    ]
                ),
            }
        )
    diagnostics = [
        item
        for item in (o.get("sample_weight_diagnostics") or [])
        if isinstance(item, dict)
    ]
    if diagnostics:
        tables.append(
            {
                "title": "样本权重候选诊断",
                "columns": ["列", "状态", "缺失率", "范围", "均值", "说明"],
                "rows": [
                    [
                        str(item.get("column") or ""),
                        "可用" if item.get("valid") else "需检查",
                        _pct(item.get("missing_rate")),
                        _range_text(item.get("min"), item.get("max")),
                        _fmt(item.get("mean")),
                        str(item.get("reason") or "已排除出入模特征"),
                    ]
                    for item in diagnostics
                ],
            }
        )
    warnings = [str(item) for item in (o.get("warnings") or [])]
    if warnings:
        text += "\n" + "\n".join(f"- {warning}" for warning in warnings)
    return text, tables


def _render_configure_tuning(o: dict):
    tune_enabled = bool(o.get("tune_enabled"))
    sample_weight_col = str(o.get("sample_weight_col") or "")
    budgets = (
        o.get("n_trials_by_recipe")
        if isinstance(o.get("n_trials_by_recipe"), dict)
        else {}
    )
    recipes = [str(item) for item in (o.get("recipes") or []) if str(item)]
    total_n_trials = o.get("total_n_trials")
    multi = len(budgets) > 1
    if multi:
        budget_note = "、".join(
            f"{recipe}={budgets[recipe]}" for recipe in recipes if recipe in budgets
        )
        text = (
            f"**调参配置已生成**:候选算法 {'/'.join(recipes)}，"
            f"{'每个算法各自执行' if tune_enabled else '跳过'}两阶段随机搜索"
            f"（按算法预算 {budget_note}；多算法总预算=Σ各配方预算={_fmt(total_n_trials)} 轮）。"
        )
    else:
        text = (
            f"**调参配置已生成**:算法 `{o.get('recipe', '')}`，"
            f"{'执行' if tune_enabled else '跳过'}两阶段随机搜索，"
            f"轮数 {o.get('n_trials', 0)}。"
        )
    rows = [
        ["目标类型", str(o.get("target_type") or "")],
        ["算法", "/".join(recipes) if recipes else str(o.get("recipe") or "")],
        ["随机搜索", "是" if tune_enabled else "否"],
    ]
    if multi:
        rows.append(
            [
                "按算法调参预算（轮数，总预算=Σ各配方预算）",
                "、".join(
                    f"{recipe}={budgets[recipe]}"
                    for recipe in recipes
                    if recipe in budgets
                ),
            ]
        )
        rows.append(["总预算", _fmt(total_n_trials)])
    else:
        rows.append(["调参轮数", _fmt(o.get("n_trials", ""))])
    rows.append(["样本权重列", sample_weight_col or "不使用"])
    rows.append(["说明", str(o.get("reason") or "")])
    tables = [
        {
            "title": "调参配置",
            "columns": ["项目", "值"],
            "rows": rows,
        }
    ]
    params = o.get("params") if isinstance(o.get("params"), dict) else {}
    if params:
        tables.append(
            {
                "title": "固定/控制参数",
                "columns": ["参数", "值"],
                "rows": [[str(key), _fmt(value)] for key, value in params.items()],
            }
        )
    return text, tables


def _render_tune(o: dict):
    best_params = o.get("best_params") or {}
    best_metrics = o.get("best_metrics") or {}
    trials = [t for t in (o.get("trials") or []) if isinstance(t, dict)]
    text = f"**调参完成**:{o.get('n_trials', '?')} 轮搜索，选出最优超参组合。"
    tables = []
    if trials:
        # trials leaderboard (G4): each trial's train/test/oot KS + overfit gap,
        # ranked by the in-time selection score (OOT is the unbiased final metric).
        ranked = sorted(
            trials,
            key=lambda t: (
                t.get("score")
                if isinstance(t.get("score"), (int, float))
                else float("-inf")
            ),
            reverse=True,
        )
        rows = []
        for rank, trial in enumerate(ranked[:15], start=1):
            train_ks, test_ks = trial.get("train_ks"), trial.get("test_ks")
            # overfit gaps: prefer stored values, fall back to deriving train-test.
            gap_tt = trial.get("overfit_gap_tt")
            if (
                gap_tt is None
                and isinstance(train_ks, (int, float))
                and isinstance(test_ks, (int, float))
            ):
                gap_tt = train_ks - test_ks
            rows.append(
                [
                    str(rank),
                    _num(train_ks),
                    _num(test_ks),
                    _num(trial.get("oot_ks")),
                    _num(trial.get("test_auc")),
                    _num(trial.get("oot_auc")),
                    _num(trial.get("lift_head_5")),
                    _num(trial.get("lift_head_10")),
                    _num(trial.get("lift_tail_5")),
                    _num(trial.get("lift_tail_10")),
                    _num(gap_tt),
                    _num(trial.get("overfit_gap_to")),
                ]
            )
        tables.append(
            {
                "title": "trials 排行（按 in-time 选优；前15）",
                "columns": [
                    "#",
                    "train_ks",
                    "test_ks",
                    "oot_ks",
                    "test_auc",
                    "oot_auc",
                    "头部lift5%",
                    "头部lift10%",
                    "尾部lift5%",
                    "尾部lift10%",
                    "过拟合gap（tt）",
                    "过拟合gap（to）",
                ],
                "rows": rows,
            }
        )
    if best_metrics:
        tables.append(
            {
                "title": "最优 trial 指标",
                "columns": ["指标", "值"],
                "rows": [[k, _fmt(v)] for k, v in best_metrics.items()],
            }
        )
    if best_params:
        tables.append(
            {
                "title": "最优超参",
                "columns": ["参数", "值"],
                "rows": [[k, _fmt(v)] for k, v in best_params.items()],
            }
        )
    return text, tables


def _render_train(o: dict):
    metrics = o.get("metrics") or {}
    text = "**训练完成**。"
    tables = []
    if metrics:
        scalar = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))
        }
        if scalar:
            tables.append(
                {
                    "title": "模型指标",
                    "columns": ["指标", "值"],
                    "rows": [[k, _fmt(v)] for k, v in scalar.items()],
                }
            )
    importance = o.get("feature_importance") or []
    rows = []
    for item in importance[:15]:
        if isinstance(item, (list, tuple)) and item:
            rows.append([str(item[0]), _fmt(item[1]) if len(item) > 1 else ""])
    if rows:
        tables.append(
            {"title": "特征重要性（前15）", "columns": ["特征", "重要性"], "rows": rows}
        )
    return text, tables


# C9: champion-selection metric labels emitted by the tool
# (marvis/packs/modeling/train_tools.py). The renderer maps each LABEL to the
# per-experiment value used for selection so the evidence sentence cites the axis
# the champion actually won on, not a hard-coded OOT KS. Kept in sync with
# train_tools._CHAMPION_OVERFIT_PENALTY / BINARY_SELECTION_METRIC /
# RESPONSE_LIFT_SELECTION_METRIC; test_strategy_development guards drift.
_CHAMPION_OVERFIT_PENALTY = 0.5
_BINARY_SELECTION_METRIC = "test_ks(overfit-penalized)"
_RESPONSE_LIFT_SELECTION_METRIC = "test_lift_head_10"


def _penalized_test_ks(metrics: dict):
    """Mirror of train_tools._overfit_penalized_test_ks: ``test_ks - 0.5*max(0,
    train_ks - test_ks)``, weighted-aware. Returns None when test_ks is missing so
    the evidence line falls back to omitting the experiment (INV-1: recomputes only
    the already-computed KS numbers, same formula that drove selection)."""
    test_ks = metrics.get("weighted_test_ks")
    if not isinstance(test_ks, (int, float)):
        test_ks = metrics.get("test_ks")
    if not isinstance(test_ks, (int, float)):
        return None
    train_ks = metrics.get("weighted_train_ks")
    if not isinstance(train_ks, (int, float)):
        train_ks = metrics.get("train_ks")
    gap = (
        float(train_ks) - float(test_ks) if isinstance(train_ks, (int, float)) else 0.0
    )
    return float(test_ks) - _CHAMPION_OVERFIT_PENALTY * max(0.0, gap)


def _key_value(key: str):
    """Per-experiment value extractor that reads a plain metrics dict key."""

    def _extract(metrics: dict):
        value = metrics.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return _extract


def _selection_axis(o: dict):
    """C9: resolve (display_label, value_of, higher_is_better) from the tool's
    emitted ``selection_metric`` LABEL so the evidence sentence renders the true
    selection axis. ``selection_metric`` is a presentation label, not a metrics key,
    so each label maps to the extractor that fetches its per-experiment value.
    Falls back to the per-target_type defaults for legacy outputs that predate the
    selection_metric field (the fallback label no longer claims 'OOT KS' unless OOT
    KS was in fact the basis for that target type)."""
    sel = str(o.get("selection_metric") or "")
    if sel == _BINARY_SELECTION_METRIC:
        return ("按 test KS(过拟合惩罚)", _penalized_test_ks, True)
    if sel == _RESPONSE_LIFT_SELECTION_METRIC:
        return ("按 test 头部10%提升", _key_value("test_lift_head_10"), True)
    if sel == "oot_rmse":
        return ("按 OOT RMSE", _key_value("oot_rmse"), False)
    if sel == "oot_macro_auc":
        return ("按 OOT macro-AUC", _key_value("oot_macro_auc"), True)
    if sel == "oot_logloss":
        return ("按 OOT logloss", _key_value("oot_logloss"), False)
    # Legacy fallback (no selection_metric field): keep today's per-target_type
    # basis so replayed/cached outputs do not crash. This is the ONLY path that may
    # still label 'OOT KS', and only because that was the historical binary default.
    target_type = str(o.get("target_type") or "binary")
    if target_type == "continuous":
        return ("按 OOT RMSE", _key_value("oot_rmse"), False)
    if target_type == "multiclass":
        return ("按 OOT macro-AUC", _key_value("oot_macro_auc"), True)
    return ("按 OOT KS", _key_value("oot_ks"), True)


def _champion_evidence_text(
    experiments, best_id, value_of, selector_label, higher_is_better
) -> str:
    """LT-11 (B.1/B.2) + C9: champion evidence -- the SELECTION metric's champion
    value and the gap to the runner-up algorithm on that SAME axis, both read from
    the experiments' own metrics via ``value_of`` (INV-1: presentation only, the gap
    is a subtraction on existing fields). Because the axis is now the one the
    champion actually won on, the '高'/'低' direction word is truthful by
    construction; a defensive guard emits neutral phrasing if the champion is
    somehow not the extreme. Empty when the champion or a runner-up value is
    unavailable."""

    def _val(exp):
        return value_of(exp.get("metrics") or {})

    champion = next((e for e in experiments if e.get("experiment_id") == best_id), None)
    champion_value = _val(champion) if champion is not None else None
    if champion_value is None:
        return ""
    others = [
        (e, _val(e))
        for e in experiments
        if e.get("experiment_id") != best_id and _val(e) is not None
    ]
    if not others:
        return f"（依据：{selector_label}={champion_value:.4f}，为唯一可比算法）"
    runner_up, runner_value = (
        max(others, key=lambda item: item[1])
        if higher_is_better
        else min(others, key=lambda item: item[1])
    )
    gap = champion_value - runner_value
    champion_leads = gap >= 0 if higher_is_better else gap <= 0
    if not champion_leads:
        # Defensive: champion is not the extreme on this axis. Do not assert '高'/'低';
        # state the values without a false lead claim.
        return (
            f"（依据：{selector_label}={champion_value:.4f}，"
            f"次优 {runner_up.get('recipe', '?')}（{runner_value:.4f}）"
            f"，二者差 {abs(gap):.4f}）"
        )
    return (
        f"（依据：{selector_label}={champion_value:.4f}，"
        f"较次优 {runner_up.get('recipe', '?')}（{runner_value:.4f}）"
        f"{'高' if higher_is_better else '低'} {abs(gap):.4f}）"
    )


def _render_train_models(o: dict):
    experiments = [e for e in (o.get("experiments") or []) if isinstance(e, dict)]
    best_id = o.get("best_experiment_id")
    best_recipe = o.get("best_recipe")
    target_type = str(o.get("target_type") or "binary")
    tables = []
    rows = []
    best_metrics: dict = {}
    if target_type == "continuous":
        metric_columns = [
            "train_rmse",
            "test_rmse",
            "oot_rmse",
            "test_mae",
            "oot_mae",
            "test_r2",
            "oot_r2",
        ]
    elif target_type == "multiclass":
        metric_columns = [
            "train_macro_auc",
            "test_macro_auc",
            "oot_macro_auc",
            "test_logloss",
            "oot_logloss",
            "test_accuracy",
            "oot_accuracy",
        ]
    else:
        metric_columns = ["train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc"]
    # C9: the evidence SENTENCE metric comes from the tool's emitted selection_metric
    # (not a hard-coded per-target_type key). The comparison TABLE columns above are
    # the full metric grid and stay unchanged. selection_axis falls back to the
    # per-target_type default only for legacy outputs lacking selection_metric.
    selector_label, value_of, higher_is_better = _selection_axis(o)
    for exp in experiments:
        metrics = exp.get("metrics") or {}
        is_best = exp.get("experiment_id") == best_id
        if is_best:
            best_metrics = metrics
        rows.append(
            [str(exp.get("recipe", "?")) + (" ★" if is_best else "")]
            + [_num(metrics.get(column)) for column in metric_columns]
        )
    if len(experiments) > 1:
        # LT-11 (B.1/B.2) + C9: the champion choice carries its evidence -- the REAL
        # selection metric it won on (selector_label from selection_metric), the
        # champion's own value on that axis, and the gap to the runner-up algorithm on
        # that SAME axis so the user sees what the champion actually beat. All numbers
        # are the experiments' own already-computed metrics (INV-1: presentation only;
        # the penalized-KS value recomputes the same formula that drove selection).
        evidence = _champion_evidence_text(
            experiments, best_id, value_of, selector_label, higher_is_better
        )
        text = (
            f"**训练完成**:对比 {len(experiments)} 个算法，"
            f"最优 **{best_recipe}**（★；{selector_label}）{evidence}。"
        )
        tables.append(
            {
                "title": "候选模型对比",
                "columns": ["算法", *metric_columns],
                "rows": rows,
            }
        )
    else:
        text = "**训练完成**。"
    # the best model's full metrics (mirrors the single-model 模型指标 table)
    scalar = {
        k: v for k, v in best_metrics.items() if isinstance(v, (int, float, str, bool))
    }
    if scalar:
        tables.append(
            {
                "title": "模型指标",
                "columns": ["指标", "值"],
                "rows": [[k, _fmt(v)] for k, v in scalar.items()],
            }
        )
    return text, tables


def _render_compare(o: dict):
    experiments = o.get("experiments") or []
    rows = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        caps = exp.get("capabilities") or {}
        rows.append(
            [
                exp.get("recipe") or "?",
                "是" if caps.get("pmml_supported") else "否",
                "是" if caps.get("handoff_supported") else "否",
                "是" if caps.get("native_model_supported") else "否",
                caps.get("reason") or "",
            ]
        )
    tables = []
    if rows:
        tables.append(
            {
                "title": "训练后动作能力",
                "columns": ["算法", "PMML", "移交验证", "原生模型", "说明"],
                "rows": rows,
            }
        )
    return f"**实验对比完成**:共 {len(experiments)} 个实验候选。", tables


def _render_select_experiment(o: dict):
    selected = o.get("selected_experiment_id") or ""
    recipe = o.get("recipe") or "?"
    metric = o.get("selection_metric") or ""
    reason = o.get("selection_reason") or ""
    caps = o.get("capabilities") or {}
    text = f"**已选择最终实验**:`{selected}`（{recipe}）；{reason}"
    rows = [
        ["PMML", "是" if caps.get("pmml_supported") else "否"],
        ["移交验证", "是" if caps.get("handoff_supported") else "否"],
        ["原生模型", "是" if caps.get("native_model_supported") else "否"],
    ]
    if caps.get("reason"):
        rows.append(["说明", caps.get("reason")])
    policy = (
        o.get("policy_decision") if isinstance(o.get("policy_decision"), dict) else {}
    )
    if policy:
        rows.append(["策略门控", policy.get("status") or "not_requested"])
        violations = [
            str(item.get("message") or item.get("code") or "")
            for item in (policy.get("violations") or [])
            if isinstance(item, dict)
        ]
        if violations:
            rows.append(["策略说明", "; ".join(item for item in violations if item)])
        if policy.get("override_reason"):
            rows.append(["Override", policy.get("override_reason")])
    tables = [
        {
            "title": f"最终模型交付能力（{metric}）",
            "columns": ["能力", "状态"],
            "rows": rows,
        }
    ]
    metrics = o.get("metrics") or {}
    if metrics:
        tables.append(
            {
                "title": "最终模型指标",
                "columns": ["指标", "值"],
                "rows": [[key, _fmt(value)] for key, value in metrics.items()],
            }
        )
    return text, tables


def _render_report(o: dict):
    path = o.get("report_path") or ""
    sections = [
        section
        for section in (o.get("section_status") or [])
        if isinstance(section, dict)
    ]
    available = sum(1 for section in sections if section.get("available"))
    skipped = len(sections) - available
    text = (
        f"**模型开发报告已生成**:`{path}`"
        f"（业务章节 {available}/{len(sections)} 可生成"
        + (f"，{skipped} 个缺输入/跳过" if skipped else "")
        + "，可在右栏下载）。"
    )
    tables = []
    if sections:
        tables.append(
            {
                "title": "报告章节状态",
                "columns": ["章节", "状态", "说明"],
                "rows": [
                    [
                        str(section.get("section", "")),
                        "可生成" if section.get("available") else "缺输入/跳过",
                        str(section.get("reason") or ""),
                    ]
                    for section in sections
                ],
            }
        )
    calibration_table = _calibration_table(o.get("calibration"))
    if calibration_table:
        tables.append(calibration_table)
    score_band_table = _score_band_table(o.get("score_bands"))
    if score_band_table:
        tables.append(score_band_table)
    return text, tables


# VD-4: calibration/score_bands are already produced by generate_model_report
# (report_tools.py::_artifact_calibration_rows / _score_band_rows) and land in
# the Excel workbook, but never reached the agent-conversation payload at all
# -- these two helpers reshape the same numbers (no new computation, INV-1)
# into the {title, columns, rows, chart} shape agentMessageTablesHtml expects,
# where `chart` carries the coordinate-ready series for the frontend SVG.
def _calibration_table(rows) -> dict | None:
    rows = [row for row in (rows or []) if isinstance(row, dict)]
    if not rows:
        return None
    summary = next((row for row in rows if row.get("score_type") == "summary"), {})
    points = [
        row
        for row in rows
        if row.get("score_type") == "raw" and row.get("avg_predicted_pd") is not None
    ]
    chart = {
        "kind": "calibration_curve",
        "points": [
            {
                "avg_predicted_pd": float(row["avg_predicted_pd"]),
                "observed_bad_rate": float(row["observed_bad_rate"]),
                "sample_count": int(row.get("sample_count") or 0),
                "bin": row.get("bin"),
            }
            for row in points
            if row.get("observed_bad_rate") is not None
        ],
        "brier_raw": summary.get("brier_raw"),
        "brier_calibrated": summary.get("brier_calibrated"),
        "ece_raw": summary.get("ece_raw"),
        "ece_calibrated": summary.get("ece_calibrated"),
    }
    table_rows = [row for row in rows if row.get("score_type") in ("raw", "calibrated")]
    return {
        "title": "概率校准（可靠性曲线）",
        "columns": [
            "类型",
            "分箱",
            "预测概率区间",
            "样本量",
            "预测均值",
            "实际坏率",
            "偏差",
        ],
        "rows": [
            [
                "原始" if row.get("score_type") == "raw" else "校准后",
                _num(row.get("bin")),
                f"{_num(row.get('prob_lower'))} - {_num(row.get('prob_upper'))}",
                _num(row.get("sample_count")),
                _num(row.get("avg_predicted_pd")),
                _num(row.get("observed_bad_rate")),
                _num(row.get("abs_gap")),
            ]
            for row in table_rows
        ],
        "chart": chart,
    }


def _score_band_table(rows) -> dict | None:
    rows = [row for row in (rows or []) if isinstance(row, dict)]
    if not rows:
        return None
    # One split at a time reads clearest as a bar+line combo; oot (or the first
    # split present) mirrors what a risk reviewer checks first for cutoff work.
    preferred_order = ["oot", "test", "train"]
    available_splits = {row.get("split") for row in rows}
    split = next(
        (s for s in preferred_order if s in available_splits), rows[0].get("split")
    )
    split_rows = [row for row in rows if row.get("split") == split]
    split_rows.sort(key=lambda row: row.get("bin") if row.get("bin") is not None else 0)
    has_unscored = any(int(row.get("unscored_count") or 0) > 0 for row in split_rows)
    chart = {
        "kind": "score_band_bars",
        "split": split,
        "bands": [
            {
                "bin": row.get("bin"),
                "score_lower": row.get("score_lower"),
                "score_upper": row.get("score_upper"),
                "sample_count": row.get("sample_count"),
                "bad_rate": row.get("bad_rate"),
            }
            for row in split_rows
        ],
    }
    columns = [
        "分箱",
        "分数区间",
        "样本量",
        "坏率",
        "累计拒绝率",
        "拒绝人群坏率",
        "lift",
    ]
    if has_unscored:
        columns.extend(["评分覆盖率", "未评分数"])
    table_rows = []
    for row in split_rows:
        values = [
            _num(row.get("bin")),
            f"{_num(row.get('score_lower'))} - {_num(row.get('score_upper'))}",
            _num(row.get("sample_count")),
            _num(row.get("bad_rate")),
            _num(row.get("cum_reject_rate", row.get("cum_count_pct"))),
            _num(row.get("cum_bad_rate")),
            _num(row.get("lift")),
        ]
        if has_unscored:
            values.extend(
                [
                    _num(row.get("score_coverage")),
                    _num(row.get("unscored_count")),
                ]
            )
        table_rows.append(values)
    return {
        "title": f"评分分段（{split}）",
        "columns": columns,
        "rows": table_rows,
        "chart": chart,
    }


def _num(value):
    return "n/a" if value is None else _fmt(value)


def _render_feature_metrics(o: dict):
    metrics = [
        metric for metric in (o.get("metrics") or []) if isinstance(metric, dict)
    ]
    # The risk-aware head/tail lift columns show only when that metric was selected
    # (absent keys → not computed); base columns are always present.
    has_head_tail = any("lift_head_5" in metric for metric in metrics)
    has_importance = any("importance" in metric for metric in metrics)
    columns = ["特征", "IV", "KS", "AUC", "PSI", "缺失率", "头部lift"]
    if has_head_tail:
        columns += ["头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%"]
    if has_importance:
        columns += ["重要性"]
    rows = []
    for metric in metrics:
        row = [
            str(metric.get("feature", "?")),
            _num(metric.get("iv")),
            _num(metric.get("ks")),
            _num(metric.get("auc")),
            _num(metric.get("psi")),
            _num(metric.get("missing_rate")),
            _num(metric.get("lift_top_bin")),
        ]
        if has_head_tail:
            row += [
                _num(metric.get("lift_head_5")),
                _num(metric.get("lift_head_10")),
                _num(metric.get("lift_tail_5")),
                _num(metric.get("lift_tail_10")),
            ]
        if has_importance:
            row += [_num(metric.get("importance"))]
        rows.append(row)
    text = (
        f"**特征分析完成**:{len(rows)} 个特征的指标如下"
        "（IV/KS/AUC 越高区分力越强；PSI/缺失率越低越稳）。可在右栏下载分析报告。"
    )
    tables = []
    if rows:
        tables.append(
            {
                "title": "特征指标",
                "columns": columns,
                "rows": rows,
            }
        )
    # Optional collinear / VIF section (computed only when the metric was selected).
    collinear = o.get("collinear")
    if isinstance(collinear, dict):
        vif = collinear.get("vif") or {}
        if vif:
            tables.append(
                {
                    "title": "VIF（共线性）",
                    "columns": ["特征", "VIF"],
                    "rows": [[str(feat), _num(value)] for feat, value in vif.items()],
                }
            )
        pairs = [
            p
            for p in (collinear.get("collinear_pairs") or [])
            if isinstance(p, (list, tuple)) and len(p) >= 3
        ]
        if pairs:
            tables.append(
                {
                    "title": "高相关特征对",
                    "columns": ["特征A", "特征B", "相关系数"],
                    "rows": [[str(p[0]), str(p[1]), _num(p[2])] for p in pairs],
                }
            )
    return text, tables


def _render_feature_report(o: dict):
    # Reuse the metrics wide table (the tool echoes metrics) and append the report link.
    text, tables = _render_feature_metrics(o)
    path = o.get("report_path") or ""
    if path:
        text += f"\n\n**特征分析报告已生成**:`{path}`（可在右栏下载）。"
    return text, tables


def _render_build_strategy(o: dict):
    rules = [rule for rule in (o.get("rules") or []) if isinstance(rule, dict)]
    strategy_type = str(o.get("strategy_type") or "approval")
    default_decision = str(o.get("default_decision") or "")
    score_col = str(o.get("score_col") or "")
    text = (
        f"**策略候选已生成**:`{o.get('strategy_id', '')}`。"
        f"类型 `{strategy_type}`，评分列 `{score_col}`，默认动作 `{default_decision}`。"
    )
    tables = []
    if rules:
        tables.append(
            {
                "title": "策略规则（按顺序命中）",
                "columns": ["#", "条件", "动作", "取值"],
                "rows": [
                    [
                        str(index),
                        str(rule.get("condition", "")),
                        str(rule.get("decision", "")),
                        _fmt(rule.get("value"))
                        if rule.get("value") is not None
                        else "-",
                    ]
                    for index, rule in enumerate(rules, start=1)
                ],
            }
        )
    return text, tables


def _backtest_view(o: dict) -> tuple[str, dict, list[dict], list[dict], dict]:
    """Normalize the versioned V2 envelope and legacy flat approval output.

    The versioned envelope is authoritative whenever it is present.  Top-level
    approval fields may still accompany it as a temporary Tool compatibility
    projection, but presentation must not let those aliases override canonical
    metrics.  Legacy plan outputs have no ``strategy_type``/``metrics`` and keep
    their historical approval interpretation.
    """

    metrics = o.get("metrics")
    strategy_type = o.get("strategy_type")
    if isinstance(metrics, dict) and isinstance(strategy_type, str):
        return (
            strategy_type,
            metrics,
            [row for row in (o.get("breakdown") or []) if isinstance(row, dict)],
            [row for row in (o.get("transitions") or []) if isinstance(row, dict)],
            o.get("economics") if isinstance(o.get("economics"), dict) else {},
        )
    return (
        "approval",
        o,
        [row for row in (o.get("by_segment") or []) if isinstance(row, dict)],
        [],
        {
            "expected_profit": o.get("expected_profit"),
            "profit_note": o.get("profit_note"),
        },
    )


def _render_backtest_strategy(o: dict):
    strategy_type, metrics, breakdown, transitions, economics = _backtest_view(o)
    if strategy_type in {"approval", "reject"}:
        text, tables = _render_decision_backtest(
            o,
            strategy_type=strategy_type,
            metrics=metrics,
            breakdown=breakdown,
            transitions=transitions,
            economics=economics,
        )
    elif strategy_type == "limit":
        text, tables = _render_limit_backtest(o, metrics, breakdown, economics)
    elif strategy_type == "pricing":
        text, tables = _render_pricing_backtest(o, metrics, breakdown, economics)
    elif strategy_type == "segmentation":
        text, tables = _render_segmentation_backtest(o, metrics, breakdown, transitions)
    else:
        text = f"**策略回测完成**:未知策略类型 `{strategy_type}`，请检查结构化结果。"
        tables = []
    return _append_backtest_warnings(text, tables, o)


def _render_design_strategy_candidate(o: dict):
    evidence = o.get("design_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    strategy_type = str(o.get("strategy_type") or evidence.get("strategy_type") or "")
    bands = [band for band in (evidence.get("bands") or []) if isinstance(band, dict)]
    objective = str(evidence.get("objective") or "")
    source_hash = str(o.get("source_dataset_content_hash") or "")
    text = (
        f"**{strategy_type or '非审批'}策略候选已确定性生成**:"
        f"共 {len(bands)} 个有效分箱，目标 `{objective or '-'}`，"
        f"policy `{o.get('candidate_policy_version') or '-'}`。"
        "这是可复核草稿，尚未采纳；采纳仍需人工确认。"
    )
    if source_hash:
        text += f" 数据证据 `{source_hash[:12]}…`。"
    assumptions = [str(item) for item in (evidence.get("assumptions") or [])]
    if assumptions:
        text += "\n" + "\n".join(f"- 口径:{item}" for item in assumptions)
    red_flags = [
        flag for flag in (evidence.get("red_flags") or []) if isinstance(flag, dict)
    ]
    if red_flags:
        text += "\n" + "\n".join(
            f"- {str(flag.get('level') or 'warning').upper()}:"
            f"{flag.get('message') or flag.get('kind') or flag.get('code')}"
            for flag in red_flags
        )

    rows = []
    for band in bands:
        action = band.get("selected_action")
        action = action if isinstance(action, dict) else {}
        lower = "-∞" if band.get("lower") is None else _fmt(band.get("lower"))
        upper = "+∞" if band.get("upper") is None else _fmt(band.get("upper"))
        rows.append(
            [
                str(band.get("band_id") or ""),
                f"{lower} ~ {upper}",
                _fmt(band.get("count")),
                _pct(band.get("population_share")),
                _pct(band.get("bad_rate")),
                _fmt(band.get("risk_estimate")),
                str(action.get("type") or ""),
                _fmt(action.get("value")),
            ]
        )
    tables = []
    if rows:
        tables.append(
            {
                "title": "确定性候选分箱与动作",
                "columns": [
                    "分箱",
                    "范围",
                    "样本数",
                    "占比",
                    "观测坏率",
                    "风险估计",
                    "动作",
                    "值",
                ],
                "rows": rows,
            }
        )
    return text, tables


def _render_analyze_univariate_candidates(o: dict):
    rankings = [item for item in (o.get("rankings") or []) if isinstance(item, dict)]
    red_flags = [str(item) for item in (o.get("red_flags") or [])]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    text = (
        f"**单变量候选分析完成**：已分析 {o.get('feature_count', 0)} 个字段，"
        f"得到 {o.get('available_method_count', 0)} 个可用字段/分箱方法组合。"
        f"候选证据 `{o.get('candidate_id', '')}` 仅处于 "
        "`development / unvalidated`，不代表独立验证、采纳或上线。"
    )
    if rankings:
        top = rankings[0]
        text += (
            f" 当前 IV 排名首位是 `{top.get('feature', '')}` / "
            f"`{top.get('method', '')}`；指标均由平台确定性计算。"
        )
    if o.get("nan_labels_dropped"):
        text += (
            f"\n- 已按你的确认排除 {o['nan_labels_dropped']} 行空标签；"
            "候选证据记录了这一口径。"
        )
    if any(flag.startswith("loan_amount_metrics_unavailable") for flag in red_flags):
        text += "\n- 尚未配置放款金额列；如能提供，我可以补做金额口径影响分析。"
    if any(flag.startswith("overdue_amount_metrics_unavailable") for flag in red_flags):
        text += "\n- 尚未配置逾期金额列；如能提供，我可以补做逾期金额口径分析。"
    links = [
        f"[{str(item.get('filename') or item.get('kind') or '下载')}]"
        f"({str(item.get('download_url'))})"
        for item in artifacts
        if item.get("download_url")
    ]
    if links:
        text += "\n\n**候选报告**：" + "；".join(links)

    tables = []
    if rankings:
        tables.append(
            {
                "title": "单变量候选排名（前20）",
                "columns": ["特征", "分箱方法", "IV", "KS", "AUC"],
                "rows": [
                    [
                        str(item.get("feature") or ""),
                        str(item.get("method") or ""),
                        _num(item.get("iv")),
                        _num(item.get("ks")),
                        _num(item.get("auc")),
                    ]
                    for item in rankings[:20]
                ],
            }
        )
    material_flags = [
        flag
        for flag in red_flags
        if not flag.startswith("loan_amount_metrics_unavailable")
        and not flag.startswith("overdue_amount_metrics_unavailable")
    ]
    if material_flags:
        tables.append(
            {
                "title": "候选分析提示",
                "columns": ["提示"],
                "rows": [[flag] for flag in material_flags[:30]],
            }
        )
    return text, tables


def _render_decision_backtest(
    o: dict,
    *,
    strategy_type: str,
    metrics: dict,
    breakdown: list[dict],
    transitions: list[dict],
    economics: dict,
) -> tuple[str, list[dict]]:
    typed = isinstance(o.get("metrics"), dict)
    approval_rate = (
        metrics.get("approve_rate") if typed else metrics.get("approval_rate")
    )
    approved_count = (
        metrics.get("approve_count") if typed else metrics.get("approved_count")
    )
    approved_bad_rate = (
        metrics.get("approve_bad_rate") if typed else metrics.get("approved_bad_rate")
    )
    rejected_count = (
        metrics.get("reject_count") if typed else metrics.get("rejected_count")
    )
    rejected_bad_rate = (
        metrics.get("reject_bad_rate") if typed else metrics.get("rejected_bad_rate")
    )
    review_count = metrics.get("review_count")
    review_rate = metrics.get("review_rate")
    review_bad_rate = metrics.get("review_bad_rate")
    expected_profit = economics.get("expected_profit")
    profit_note = economics.get("profit_note")
    label = "拒绝策略回测完成" if strategy_type == "reject" else "策略回测完成"
    text = (
        f"**{label}**:"
        f"审批率 {_pct(approval_rate)}，"
        f"通过客群坏率 {_pct(approved_bad_rate)}，"
        f"拒绝客群坏率 {_pct(rejected_bad_rate)}，"
        f"预期利润 {_num(expected_profit)}。"
    )
    if o.get("label_coverage") is not None:
        text += f" 标签覆盖率 {_pct(o.get('label_coverage'))}。"
    if int(review_count or 0):
        text += (
            f" 人工复核 {review_count} 户（{_pct(review_rate)}），"
            f"复核客群坏率 {_pct(review_bad_rate)}。"
        )
    if strategy_type == "reject":
        text += (
            f" 坏客户捕获率 {_pct(metrics.get('bad_capture_rate'))}，"
            f"好客户误拒率 {_pct(metrics.get('good_reject_rate'))}。"
        )
    if profit_note:
        text += f" 利润口径提示：{profit_note}"

    rows = [
        ["审批率", _pct(approval_rate)],
        ["通过人数", _fmt(approved_count)],
        ["通过坏率", _pct(approved_bad_rate)],
        ["拒绝人数", _fmt(rejected_count)],
        ["拒绝坏率", _pct(rejected_bad_rate)],
        ["人工复核人数", _fmt(review_count)],
        ["人工复核率", _pct(review_rate)],
        ["复核客群坏率", _pct(review_bad_rate)],
        ["预期利润", _num(expected_profit)],
    ]
    if profit_note:
        rows.append(["利润口径提示", str(profit_note)])
    if strategy_type == "reject":
        rows.extend(
            [
                ["坏客户捕获率", _pct(metrics.get("bad_capture_rate"))],
                ["好客户误拒率", _pct(metrics.get("good_reject_rate"))],
            ]
        )
    if typed:
        rows.append(["标签覆盖率", _pct(o.get("label_coverage"))])
    else:
        rows.extend(
            [
                ["swap-in", _fmt(metrics.get("swap_in_count"))],
                ["swap-out", _fmt(metrics.get("swap_out_count"))],
                ["标签覆盖率", _pct(o.get("label_coverage"))],
            ]
        )
    tables: list[dict] = [
        {"title": "策略回测摘要", "columns": ["指标", "值"], "rows": rows}
    ]
    if breakdown:
        if typed:
            tables.append(
                {
                    "title": "按决策分组",
                    "columns": ["决策", "样本数", "占比", "有标签数", "坏样本", "坏率"],
                    "rows": [
                        [
                            str(row.get("action", "")),
                            _fmt(row.get("count")),
                            _pct(row.get("rate")),
                            _fmt(row.get("labeled_count")),
                            _fmt(row.get("bad_count")),
                            _pct(row.get("bad_rate")),
                        ]
                        for row in breakdown
                    ],
                }
            )
        else:
            tables.append(
                {
                    "title": "按决策分组",
                    "columns": ["决策", "样本数", "坏样本", "坏率"],
                    "rows": [
                        [
                            str(row.get("decision", "")),
                            _fmt(row.get("count")),
                            _fmt(row.get("bad_count")),
                            _pct(row.get("bad_rate")),
                        ]
                        for row in breakdown
                    ],
                }
            )
    transition_table = _transition_table(strategy_type, transitions)
    if transition_table is not None:
        tables.append(transition_table)
    return text, tables


def _render_limit_backtest(
    o: dict, metrics: dict, breakdown: list[dict], economics: dict
) -> tuple[str, list[dict]]:
    text = (
        "**额度策略回测完成**:"
        f"覆盖 {_fmt(metrics.get('count', o.get('population_count')))} 户，"
        f"总额度 {_num(metrics.get('total_limit'))}，"
        f"户均额度 {_num(metrics.get('mean_limit'))}，"
        f"较基线总额度变化 {_num(metrics.get('total_limit_delta'))}。"
    )
    if o.get("label_coverage") is not None:
        text += f" 标签覆盖率 {_pct(o.get('label_coverage'))}。"
    rows = [
        ["样本数", _fmt(metrics.get("count", o.get("population_count")))],
        ["总额度", _num(metrics.get("total_limit"))],
        ["户均额度", _num(metrics.get("mean_limit"))],
        ["最低额度", _num(metrics.get("min_limit"))],
        ["最高额度", _num(metrics.get("max_limit"))],
        ["提额人数", _num(metrics.get("up_count"))],
        ["降额人数", _num(metrics.get("down_count"))],
        ["额度不变人数", _num(metrics.get("unchanged_count"))],
        ["总额度变化", _num(metrics.get("total_limit_delta"))],
        ["预期 EAD", _num(economics.get("expected_ead"))],
        ["预期损失", _num(economics.get("expected_loss"))],
        ["标签覆盖率", _pct(o.get("label_coverage"))],
    ]
    tables: list[dict] = [
        {"title": "额度策略回测摘要", "columns": ["指标", "值"], "rows": rows}
    ]
    if breakdown:
        tables.append(
            {
                "title": "额度分布",
                "columns": ["额度", "样本数", "占比", "有标签数", "坏样本", "坏率"],
                "rows": [
                    [
                        _num(row.get("assigned_limit")),
                        _fmt(row.get("count")),
                        _pct(row.get("share")),
                        _fmt(row.get("labeled_count")),
                        _fmt(row.get("bad_count")),
                        _pct(row.get("bad_rate")),
                    ]
                    for row in breakdown
                ],
            }
        )
    return text, tables


def _render_pricing_backtest(
    o: dict, metrics: dict, breakdown: list[dict], economics: dict
) -> tuple[str, list[dict]]:
    text = (
        "**定价策略回测完成**:"
        f"覆盖 {_fmt(metrics.get('count', o.get('population_count')))} 户，"
        f"平均年化利率 {_pct(metrics.get('mean_rate'))}，"
        f"预期利润 {_num(economics.get('profit'))}，"
        f"ROA {_pct(economics.get('roa'))}。"
    )
    if o.get("label_coverage") is not None:
        text += f" 标签覆盖率 {_pct(o.get('label_coverage'))}。"
    rows = [
        ["样本数", _fmt(metrics.get("count", o.get("population_count")))],
        ["平均年化利率", _pct(metrics.get("mean_rate"))],
        ["提价人数", _num(metrics.get("repriced_up_count"))],
        ["降价人数", _num(metrics.get("repriced_down_count"))],
        ["价格不变人数", _num(metrics.get("unchanged_count"))],
        ["EAD 加权利率", _pct(economics.get("ead_weighted_rate"))],
        ["预期收入", _num(economics.get("revenue"))],
        ["预期损失", _num(economics.get("expected_loss"))],
        ["资金成本", _num(economics.get("funding_cost"))],
        ["运营成本", _num(economics.get("operating_cost"))],
        ["预期利润", _num(economics.get("profit"))],
        ["ROA", _pct(economics.get("roa"))],
        ["基线利润", _num(economics.get("baseline_profit"))],
        ["较基线利润变化", _num(economics.get("profit_delta_vs_baseline"))],
        ["标签覆盖率", _pct(o.get("label_coverage"))],
    ]
    tables: list[dict] = [
        {"title": "定价策略回测摘要", "columns": ["指标", "值"], "rows": rows}
    ]
    if breakdown:
        tables.append(
            {
                "title": "定价分布",
                "columns": ["年化利率", "样本数", "占比", "有标签数", "坏样本", "坏率"],
                "rows": [
                    [
                        _pct(row.get("assigned_rate")),
                        _fmt(row.get("count")),
                        _pct(row.get("share")),
                        _fmt(row.get("labeled_count")),
                        _fmt(row.get("bad_count")),
                        _pct(row.get("bad_rate")),
                    ]
                    for row in breakdown
                ],
            }
        )
    return text, tables


def _render_segmentation_backtest(
    o: dict, metrics: dict, breakdown: list[dict], transitions: list[dict]
) -> tuple[str, list[dict]]:
    text = (
        "**分群策略回测完成**:"
        f"形成 {_fmt(metrics.get('segment_count'))} 个客群，"
        f"总体坏率 {_pct(metrics.get('overall_bad_rate'))}。"
    )
    if o.get("label_coverage") is not None:
        text += f" 标签覆盖率 {_pct(o.get('label_coverage'))}。"
    rows = [
        [
            str(row.get("segment", "")),
            _fmt(row.get("count")),
            _pct(row.get("share")),
            _fmt(row.get("labeled_count")),
            _fmt(row.get("bad_count")),
            _pct(row.get("bad_rate")),
            _fmt(row.get("lift")),
        ]
        for row in breakdown
    ]
    tables: list[dict] = [
        {
            "title": "客群风险分布",
            "columns": ["客群", "样本数", "占比", "有标签数", "坏样本", "坏率", "Lift"],
            "rows": rows,
        }
    ]
    transition_table = _transition_table("segmentation", transitions)
    if transition_table is not None:
        tables.append(transition_table)
    return text, tables


def _transition_table(strategy_type: str, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    if strategy_type in {"approval", "reject"}:
        return {
            "title": "相对基线的决策迁移",
            "columns": ["原决策", "新决策", "样本数", "原决策内占比", "总体占比"],
            "rows": [
                [
                    str(row.get("from_action", "")),
                    str(row.get("to_action", "")),
                    _fmt(row.get("count")),
                    _pct(row.get("rate")),
                    _pct(row.get("population_share")),
                ]
                for row in rows
            ],
        }
    if strategy_type == "segmentation":
        return {
            "title": "相对基线的客群迁移",
            "columns": ["原客群", "新客群", "样本数", "原客群内占比", "总体占比"],
            "rows": [
                [
                    str(row.get("from_segment", "")),
                    str(row.get("to_segment", "")),
                    _fmt(row.get("count")),
                    _pct(row.get("rate")),
                    _pct(row.get("population_share")),
                ]
                for row in rows
            ],
        }
    return None


def _append_backtest_warnings(
    text: str, tables: list[dict], o: dict
) -> tuple[str, list[dict]]:
    warnings = [str(item) for item in (o.get("warnings") or []) if str(item)]
    if warnings:
        text += " 警告：" + "；".join(warnings) + "。"
        tables.append(
            {
                "title": "回测警告",
                "columns": ["警告"],
                "rows": [[warning] for warning in warnings],
            }
        )
    red_flags = [
        item
        for item in (o.get("red_flags") or [])
        if isinstance(item, dict) and str(item.get("message") or "")
    ]
    if red_flags:
        tables.append(
            {
                "title": "回测风险提示",
                "columns": ["等级", "代码", "说明"],
                "rows": [
                    [
                        str(item.get("level") or ""),
                        str(item.get("code") or ""),
                        str(item.get("message") or ""),
                    ]
                    for item in red_flags
                ],
            }
        )
    return text, tables


def _profit_delta_text(value, other) -> str:
    """预期利润差值文案（推荐 vs 备选），只对已有输出字段做减法（不新增任何计算口径,
    与 compare 渲染器对 tool deltas 取差同源，presentation-only INV-1）。"""
    try:
        diff = float(value) - float(other)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


def _tradeoff_alternatives(
    points: list, recommended: dict | None
) -> tuple[list[list], dict | None]:
    """LT-11 (B.2): the top-2 feasible cutoff alternatives *other than* the
    recommended one (each with its 预期利润 gap vs the recommended point) plus the
    single best alternative point itself, so the caller can also state the
    recommendation's advantage. All numbers are the points' own already-computed
    fields; the gap is a plain subtraction (no new computation口径)."""
    if not recommended:
        return [], None
    reco_cutoff = recommended.get("cutoff")
    feasible = [
        point
        for point in points
        if point.get("feasible", True) and point.get("cutoff") != reco_cutoff
    ]
    # Order by expected_profit desc (the same objective the recommend picked on), so
    # "备选" reads as the runner-up feasible operating points.
    feasible.sort(
        key=lambda p: (
            p.get("expected_profit")
            if isinstance(p.get("expected_profit"), (int, float))
            else float("-inf")
        ),
        reverse=True,
    )
    rows = [
        [
            _fmt(point.get("cutoff")),
            _pct(point.get("approval_rate")),
            _pct(point.get("bad_rate")),
            _num(point.get("expected_profit")),
            _profit_delta_text(
                point.get("expected_profit"), recommended.get("expected_profit")
            ),
        ]
        for point in feasible[:2]
    ]
    return rows, (feasible[0] if feasible else None)


def _render_tradeoff_view(o: dict):
    points = [point for point in (o.get("points") or []) if isinstance(point, dict)]
    recommended = (
        o.get("recommended") if isinstance(o.get("recommended"), dict) else None
    )
    direction_label = (
        "分数越高风险越低"
        if o.get("score_direction") == "higher_is_better"
        else "分数越高风险越高"
    )
    feasible_points = [point for point in points if point.get("feasible", True)]
    alt_rows, best_alt = _tradeoff_alternatives(points, recommended)
    if recommended:
        # LT-11 (B.1): the recommendation carries its evidence -- it is a *feasible*
        # operating point (constraint-satisfying), and how many of the scanned
        # cutoffs were feasible at all; plus (B.2) the profit advantage over the
        # next-best feasible cutoff so 推荐 is not a bare conclusion. Every number is
        # a field already in the tool output (INV-1: presentation only, no
        # re-computation; the advantage is a subtraction of two existing points).
        evidence = f"依据：满足约束的可行点（共 {len(feasible_points)}/{len(points)} 个 cutoff 可行）"
        if best_alt is not None:
            advantage = _profit_delta_text(
                recommended.get("expected_profit"), best_alt.get("expected_profit")
            )
            evidence += f"，且预期利润较次优 cutoff `{_fmt(best_alt.get('cutoff'))}` 高 {advantage}"
        text = (
            f"**策略权衡视图完成**（{direction_label}）："
            f"推荐 cutoff `{_fmt(recommended.get('cutoff'))}`，"
            f"审批率 {_pct(recommended.get('approval_rate'))}，"
            f"坏率 {_pct(recommended.get('bad_rate'))}，"
            f"预期利润 {_num(recommended.get('expected_profit'))}"
            f"（{evidence}）。"
        )
    else:
        text = f"**策略权衡视图完成**（{direction_label}）。"
    red_flags = [flag for flag in (o.get("red_flags") or []) if isinstance(flag, dict)]
    red_items = [flag for flag in red_flags if flag.get("level") == "red"]
    if red_items:
        names = "、".join(str(flag.get("code")) for flag in red_items)
        text += f" 红旗：{names}。"
    tables = []
    reco_cutoff = recommended.get("cutoff") if recommended else None
    if points:
        tables.append(
            {
                "title": "cutoff 权衡点",
                "columns": ["推荐", "cutoff", "审批率", "坏率", "预期利润", "可行"],
                "rows": [
                    [
                        "★"
                        if point.get("cutoff") == reco_cutoff and recommended
                        else "",
                        _fmt(point.get("cutoff")),
                        _pct(point.get("approval_rate")),
                        _pct(point.get("bad_rate")),
                        _num(point.get("expected_profit")),
                        "是" if point.get("feasible", True) else "否",
                    ]
                    for point in points[:20]
                ],
            }
        )
    # LT-11 (B.2): top-2 feasible备选 with the预期利润 gap to推荐, so the user sees
    # what the recommendation gives up relative to the runner-up operating points.
    if alt_rows:
        tables.append(
            {
                "title": "次优可行 cutoff（备选，含与推荐的预期利润差）",
                "columns": ["cutoff", "审批率", "坏率", "预期利润", "与推荐预期利润差"],
                "rows": alt_rows,
            }
        )
    if red_flags:
        tables.append(_red_flag_table(red_flags))
    return text, tables


_STRATEGY_DECISION_LABEL = {"approve": "通过", "review": "复核", "decline": "拒绝"}


def _red_flag_table(red_flags: list) -> dict:
    return {
        "title": "红旗清单",
        "columns": ["等级", "code", "说明"],
        "rows": [
            [
                str(flag.get("level", "")),
                str(flag.get("code", "")),
                str(flag.get("message", "")),
            ]
            for flag in red_flags
            if isinstance(flag, dict)
        ],
    }


def _render_design_cutoff_bands(o: dict):
    bands = [band for band in (o.get("bands") or []) if isinstance(band, dict)]
    red_flags = [flag for flag in (o.get("red_flags") or []) if isinstance(flag, dict)]
    red_items = [flag for flag in red_flags if flag.get("level") == "red"]
    approved = [band for band in bands if band.get("decision") == "approve"]
    rules = [
        rule for rule in (o.get("recommended_rules") or []) if isinstance(rule, dict)
    ]
    rule_text = rules[0].get("condition") if rules else "无"
    # LT-11 (B.1): the recommended cut carries its evidence -- the cumulative bad
    # rate and approval rate *at the approved frontier* (the最后一个 approve 带's own
    # cum_* fields the bands already carry), so 推荐切法 shows why it is safe rather
    # than only naming the rule. Numbers are the bands' own fields (INV-1: no
    # re-computation). Frontier band = the approved band with the widest cumulative
    # approval (the boundary the cut lands on).
    frontier = max(
        approved,
        key=lambda b: (
            b.get("cum_approval_rate")
            if isinstance(b.get("cum_approval_rate"), (int, float))
            else -1.0
        ),
        default=None,
    )
    evidence = ""
    if rules and frontier is not None:
        evidence = (
            f"（依据：通过客群累计坏率 {_pct(frontier.get('cum_bad_rate'))}，"
            f"累计审批率 {_pct(frontier.get('cum_approval_rate'))}，满足约束）"
        )
    text = (
        f"**分数带设计完成**：推荐切法 `{rule_text}`（拒绝规则）{evidence}，"
        f"通过 {len(approved)}/{len(bands)} 个分数带，红旗 {len(red_flags)} 项。"
    )
    if red_items:
        names = "、".join(str(flag.get("code")) for flag in red_items)
        text += f" 红项：{names}。"
    tables = [
        {
            "title": "分数带",
            "columns": [
                "band 区间",
                "样本占比",
                "坏率",
                "累计审批率",
                "累计坏率",
                "决策",
            ],
            "rows": [
                [
                    f"[{_fmt(band.get('lo'))},{_fmt(band.get('hi'))})",
                    _pct(band.get("pop_pct")),
                    _pct(band.get("bad_rate")),
                    _pct(band.get("cum_approval_rate")),
                    _pct(band.get("cum_bad_rate")),
                    _STRATEGY_DECISION_LABEL.get(
                        str(band.get("decision")), str(band.get("decision", ""))
                    ),
                ]
                for band in bands
            ],
        }
    ]
    if red_flags:
        tables.append(_red_flag_table(red_flags))
    return text, tables


def _render_compare_strategies(o: dict):
    if o.get("status") == "no_baseline":
        return (
            "**策略对比未执行**：未提供基线策略；矩阵、差异和标签覆盖率均为 n/a。",
            [],
        )
    matrix = o.get("matrix_2x2") if isinstance(o.get("matrix_2x2"), dict) else {}
    deltas = o.get("deltas") if isinstance(o.get("deltas"), dict) else {}
    red_flags = [flag for flag in (o.get("red_flags") or []) if isinstance(flag, dict)]
    text = f"**策略对比完成**：{o.get('summary_text') or ''}"
    if o.get("label_coverage") is not None:
        text += f" 标签覆盖率 {_pct(o.get('label_coverage'))}。"
    conclusion = _compare_conclusion_line(deltas)
    if conclusion:
        text += f"\n\n{conclusion}"
    red_items = [flag for flag in red_flags if flag.get("level") == "red"]
    if red_items:
        names = "、".join(str(flag.get("code")) for flag in red_items)
        text += f" 红旗：{names}。"

    def _cell(key: str) -> dict:
        return matrix.get(key) if isinstance(matrix.get(key), dict) else {}

    ba, on, ob, bd = (
        _cell("both_approve"),
        _cell("only_new"),
        _cell("only_baseline"),
        _cell("both_decline"),
    )
    # S6: the swap 2×2 is a matrix-heat card — each cell's own approved bad rate (0..1)
    # colors the heat chip (S3 matrix-heat kind reused); the count rides along as text.
    heat_columns = ["", "基线通过", "基线拒绝"]
    heat_rows = [
        ["新策略通过", _heat_cell(ba), _heat_cell(on)],
        ["新策略拒绝", _heat_cell(ob), _heat_cell(bd)],
    ]
    tables = [
        {
            "title": "swap 2×2 坏率热力（含样本数）",
            "columns": heat_columns,
            "rows": heat_rows,
            "column_specs": [
                {"kind": "text"},
                {"kind": "matrix-heat"},
                {"kind": "matrix-heat"},
            ],
        },
        {
            "title": "关键指标并排（挑战者 vs 基线）",
            "columns": ["指标", "挑战者−基线", "方向"],
            "rows": [
                [
                    "审批率",
                    _pct(deltas.get("approval_rate")),
                    _delta_arrow(deltas.get("approval_rate")),
                ],
                [
                    "通过坏率",
                    _pct(deltas.get("approved_bad_rate")),
                    _delta_arrow(deltas.get("approved_bad_rate"), lower_is_better=True),
                ],
                [
                    "预期利润",
                    _num(deltas.get("expected_profit")),
                    _delta_arrow(deltas.get("expected_profit")),
                ],
            ],
        },
    ]
    if red_flags:
        tables.append(_red_flag_table(red_flags))
    return text, tables


def _heat_cell(cell: dict) -> float:
    """matrix-heat value for a swap cell: its approved bad rate (0..1). The count is
    kept in the label the frontend renders alongside the heat chip."""
    try:
        return float(cell.get("bad_rate") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _delta_word(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "持平"
    if number > 0:
        return "上升"
    if number < 0:
        return "下降"
    return "持平"


def _delta_arrow(value, *, lower_is_better: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "持平"
    if number == 0:
        return "持平"
    improved = (number < 0) if lower_is_better else (number > 0)
    direction = "↑" if number > 0 else "↓"
    return f"{direction} {'更优' if improved else '更差'}"


def _compare_conclusion_line(deltas: dict) -> str:
    """Templated Chinese conclusion — every number comes straight from the tool's
    deltas (INV-1: presentation only). Empty when there is no delta to talk about."""
    if not deltas:
        return ""
    approval = deltas.get("approval_rate")
    bad = deltas.get("approved_bad_rate")
    profit = deltas.get("expected_profit")
    if approval is None and bad is None and profit is None:
        return ""
    approval_word = _delta_word(approval)
    bad_word = _delta_word(bad)
    return (
        f"结论：挑战者在通过率{approval_word} {abs(float(approval or 0)) * 100:.1f}pp 下，"
        f"通过客群坏率{bad_word} {abs(float(bad or 0)) * 100:.2f}pp，"
        f"预期利润变动 {float(profit or 0):.2f}。"
    )


def _render_limit_pricing_matrix(o: dict):
    matrix = [cell for cell in (o.get("matrix") or []) if isinstance(cell, dict)]
    recommended = [
        item for item in (o.get("recommended") or []) if isinstance(item, dict)
    ]
    red_flags = [flag for flag in (o.get("red_flags") or []) if isinstance(flag, dict)]
    registered_artifacts = [
        item
        for item in (o.get("artifacts") or [])
        if isinstance(item, dict) and item.get("artifact_id")
    ]
    reco_keys = {
        (str(item.get("band")), _num(item.get("limit")), _num(item.get("rate")))
        for item in recommended
    }
    text = (
        f"**额度×定价矩阵完成**：{len(matrix)} 个 band×额度×定价单元，"
        f"推荐 {len(recommended)} 档（每带利润最大可行档）。"
    )
    red_items = [flag for flag in red_flags if flag.get("level") == "red"]
    if red_items:
        names = "、".join(str(flag.get("code")) for flag in red_items)
        text += f" 红旗：{names}。"
    if registered_artifacts:
        text += f" 已登记 {len(registered_artifacts)} 个文件，可在策略产物卡下载。"

    def _cell_row(cell: dict) -> list:
        key = (str(cell.get("band")), _num(cell.get("limit")), _num(cell.get("rate")))
        recommended_mark = "★" if key in reco_keys else ""
        profit = cell.get("expected_profit")
        # Negative-profit cells are red-染 by prefixing a marker the frontend maps to
        # the warning skin; recommended cells carry a ★ and are hoisted to the top.
        profit_text = _num(profit)
        try:
            if profit is not None and float(profit) < 0:
                profit_text = f"⚠{profit_text}"
        except (TypeError, ValueError):
            pass
        return [
            f"{recommended_mark}{cell.get('band', '')}",
            _num(cell.get("limit")),
            _pct(cell.get("rate")),
            _fmt(cell.get("count")),
            _pct(cell.get("pd")),
            _num(cell.get("el")),
            profit_text,
            _pct(cell.get("roa")),
            "是" if cell.get("feasible") else "否",
        ]

    # Recommended cells first (置顶), then the rest in stable order.
    reco_cells = [
        cell
        for cell in matrix
        if (str(cell.get("band")), _num(cell.get("limit")), _num(cell.get("rate")))
        in reco_keys
    ]
    other_cells = [
        cell
        for cell in matrix
        if (str(cell.get("band")), _num(cell.get("limit")), _num(cell.get("rate")))
        not in reco_keys
    ]
    tables = [
        {
            "title": "额度×定价矩阵（★为推荐档，⚠为负利润）",
            "columns": [
                "band",
                "额度",
                "年化",
                "样本数",
                "PD",
                "EL",
                "预期利润",
                "ROA",
                "可行",
            ],
            "rows": [_cell_row(cell) for cell in [*reco_cells, *other_cells]],
        }
    ]
    if red_flags:
        tables.append(_red_flag_table(red_flags))
    return text, tables


def _render_profit_calc(o: dict):
    results = [row for row in (o.get("results") or []) if isinstance(row, dict)]
    warnings = [
        item for item in (o.get("quality_warnings") or []) if isinstance(item, dict)
    ]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    registered_artifacts = [item for item in artifacts if item.get("artifact_id")]
    total_profit = sum(float(row.get("net_profit") or 0.0) for row in results)
    text = f"**利润分析完成**：{len(results)} 个分群，合计净利润 {_num(total_profit)}。"
    if warnings:
        text += f" {len(warnings)} 条数据质量提示。"
    if registered_artifacts:
        text += f" 已登记 {len(registered_artifacts)} 个文件，可在策略产物卡下载。"
    elif artifacts:
        text += f" 已生成 {len(artifacts)} 个文件，但尚未登记下载。"
    tables = [
        {
            "title": "分群利润结果",
            "columns": [
                "分群",
                "样本数",
                "收入",
                "预期损失",
                "资金成本",
                "运营成本",
                "净利润",
                "ROA",
            ],
            "rows": [
                [
                    str(row.get("segment", "")),
                    _fmt(row.get("count")),
                    _num(row.get("revenue")),
                    _num(row.get("expected_loss")),
                    _num(row.get("funding_cost")),
                    _num(row.get("operating_cost")),
                    _num(row.get("net_profit")),
                    _pct(row.get("roa")),
                ]
                for row in results
            ],
        }
    ]
    if warnings:
        tables.append(
            {
                "title": "数据质量提示",
                "columns": ["代码", "影响行数", "说明"],
                "rows": [
                    [
                        str(item.get("code", "")),
                        _fmt(item.get("count")),
                        str(item.get("message", "")),
                    ]
                    for item in warnings
                ],
            }
        )
    return text, tables


def _render_roll_rate_matrix(o: dict):
    states = [str(state) for state in (o.get("states") or [])]
    matrix = o.get("matrix") or []
    base_counts = o.get("base_counts") or {}
    warnings = [
        item
        for item in (o.get("data_quality_warnings") or [])
        if isinstance(item, dict)
    ]
    artifacts = [item for item in (o.get("artifacts") or []) if isinstance(item, dict)]
    registered_artifacts = [item for item in artifacts if item.get("artifact_id")]
    semantics = str(o.get("observation_semantics") or "adjacent_observation")
    semantics_text = "相邻观测" if semantics == "adjacent_observation" else semantics
    text = f"**Roll-rate 矩阵完成**：{len(states)} 个状态，口径为{semantics_text}。"
    if warnings:
        text += f" {len(warnings)} 条质量提示。"
    if registered_artifacts:
        text += f" 已登记 {len(registered_artifacts)} 个文件，可在策略产物卡下载。"
    elif artifacts:
        text += f" 已生成 {len(artifacts)} 个文件，但尚未登记下载。"

    rows = []
    for index, state in enumerate(states):
        raw_row = (
            matrix[index]
            if index < len(matrix) and isinstance(matrix[index], list)
            else []
        )
        rows.append(
            [
                state,
                _fmt(base_counts.get(state)),
                *[
                    _pct(raw_row[to_index] if to_index < len(raw_row) else None)
                    for to_index in range(len(states))
                ],
            ]
        )
    tables = [
        {
            "title": "相邻观测状态转移率",
            "columns": ["期初状态", "基数", *states],
            "rows": rows,
        }
    ]
    if warnings:
        tables.append(
            {
                "title": "数据质量提示",
                "columns": ["代码", "说明"],
                "rows": [
                    [str(item.get("code", "")), str(item.get("message", ""))]
                    for item in warnings
                ],
            }
        )
    return text, tables


def _render_adopt_strategy(o: dict):
    retired = [str(item) for item in (o.get("retired_strategy_ids") or [])]
    artifacts = [a for a in (o.get("artifacts") or []) if isinstance(a, dict)]
    text = (
        f"**策略已在本地采纳**：`{o.get('strategy_id', '')}` v{o.get('version', '')}，"
        f"资产状态 {o.get('asset_status', 'adopted_local')}（兼容状态 "
        f"{o.get('status', '')}），退役 {len(retired)} 个旧版本，"
        f"生成 {len(artifacts)} 份交付物。本地采纳不代表生产环境已上线。"
    )
    tables = [
        {
            "title": "交付物",
            "columns": ["类型", "路径"],
            "rows": [
                [str(a.get("kind", "")), str(a.get("path", ""))] for a in artifacts
            ],
        }
    ]
    if retired:
        tables.append(
            {
                "title": "退役策略",
                "columns": ["策略 id"],
                "rows": [[item] for item in retired],
            }
        )
    return text, tables


def _render_challenger_report(o: dict):
    status = str(o.get("status") or "")
    artifacts = [a for a in (o.get("artifacts") or []) if isinstance(a, dict)]
    if status == "no_baseline":
        return "**挑战者对比报告**：未提供基线（champion）策略，已跳过报告。", []
    text = (
        f"**挑战者对比报告已生成**：`{o.get('report_path', '')}`，"
        f"登记 {len(artifacts)} 份交付物。"
    )
    tables = [
        {
            "title": "交付物",
            "columns": ["类型", "路径"],
            "rows": [
                [str(a.get("kind", "")), str(a.get("path", ""))] for a in artifacts
            ],
        }
    ]
    return text, tables


def _render_strategy_doc(o: dict):
    sections = [str(item) for item in (o.get("sections") or [])]
    text = f"**策略文档已生成**：`{o.get('doc_path', '')}`，共 {len(sections)} 个章节。"
    tables = [
        {
            "title": "文档章节",
            "columns": ["#", "章节"],
            "rows": [
                [str(index), section] for index, section in enumerate(sections, start=1)
            ],
        }
    ]
    return text, tables


def _render_vintage_curve(o: dict):
    cohorts = [str(item) for item in (o.get("cohorts") or [])]
    mob_axis = list(o.get("mob_axis") or [])
    summary = o.get("summary") if isinstance(o.get("summary"), dict) else {}
    trend = str(summary.get("trend") or "stable")
    text = f"**Vintage 曲线完成**:{len(cohorts)} 个 cohort，趋势 `{trend}`。"
    tables = []
    curves = o.get("curves") if isinstance(o.get("curves"), dict) else {}
    counts = o.get("counts") if isinstance(o.get("counts"), dict) else {}
    if cohorts and mob_axis:
        tables.append(
            {
                "title": "Vintage 累计坏账率",
                "columns": ["cohort", "样本数", *[f"MOB{mob}" for mob in mob_axis]],
                "rows": [
                    [
                        cohort,
                        _fmt(counts.get(cohort, "")),
                        *[
                            _pct(value) if value is not None else "n/a"
                            for value in (curves.get(cohort) or [])[: len(mob_axis)]
                        ],
                    ]
                    for cohort in cohorts
                ],
            }
        )
    at_ref = summary.get("at_ref") if isinstance(summary.get("at_ref"), dict) else {}
    if at_ref:
        tables.append(
            {
                "title": "参考 MOB 坏账率",
                "columns": ["cohort", "坏账率"],
                "rows": [
                    [str(cohort), _pct(value)] for cohort, value in at_ref.items()
                ],
            }
        )
    # A1: surface the vintage kernel's data-quality warnings (e.g. the snapshot-flag
    # red flag when data looks cumulative but was declared incremental) as red flags,
    # mirroring _render_slice_aggregate. De-duplicated: the kernel attaches the same
    # flag to every point, so warnings arrives with one entry per point.
    seen: set[str] = set()
    flag_lines = []
    for warning in o.get("warnings") or []:
        text_warning = str(warning)
        if text_warning and text_warning not in seen:
            seen.add(text_warning)
            flag_lines.append(f"🚩 {text_warning}")
    if flag_lines:
        text += "\n" + "\n".join(flag_lines)
    return text, tables


# S6 ad-hoc slice/aggregate result. INV-1: every number here comes from the
# slice_aggregate tool (a single deterministic DuckDB SQL); this renderer only
# lays the tool's own rows/columns into a table and echoes the confirmed 口径
# (spec_echo) + any red flags, never (re)computing anything.
_SLICE_OP_LABEL = {
    "count": "数量",
    "sum": "求和",
    "mean": "均值",
    "min": "最小值",
    "max": "最大值",
    "bad_rate": "坏率",
    "approval_rate": "通过率",
    "distinct": "去重计数",
}


def _slice_metric_text(metric: dict) -> str:
    op = str(metric.get("op") or "")
    label = _SLICE_OP_LABEL.get(op, op)
    col = metric.get("col")
    return f"{col} 的{label}" if col else label


def _render_slice_aggregate(o: dict):
    columns = [str(c) for c in (o.get("columns") or [])]
    rows = [row for row in (o.get("rows") or []) if isinstance(row, dict)]
    spec = o.get("spec_echo") if isinstance(o.get("spec_echo"), dict) else {}
    group_by = [str(c) for c in (spec.get("group_by") or [])]
    metrics = [m for m in (spec.get("metrics") or []) if isinstance(m, dict)]
    group_text = "、".join(group_by) if group_by else "全体样本"
    metric_text = "、".join(_slice_metric_text(m) for m in metrics) or "—"
    echo_parts = [f"口径:按〔{group_text}〕统计〔{metric_text}〕"]
    if spec.get("month_col") and spec.get("months"):
        echo_parts.append(
            f"，时间〔{'、'.join(str(m) for m in spec.get('months') or [])}〕"
        )
    filters = [f for f in (spec.get("filters") or []) if isinstance(f, dict)]
    if filters:
        filter_text = "、".join(
            f"{f.get('col')}{f.get('op')}{f.get('value')}" for f in filters
        )
        echo_parts.append(f"，筛选〔{filter_text}〕")
    text = (
        "**即席问数结果**（" + str(len(rows)) + " 行）。\n" + "".join(echo_parts) + "。"
    )
    # A4: bad_rate/approval_rate may now be NULL for an all-unlabeled group — render it as
    # "n/a" rather than the literal "None" so the honest "no labeled samples" answer reads
    # cleanly. The companion unlabeled_count_<col> columns flow through automatically.
    tables = [
        {
            "title": "聚合结果",
            "columns": columns,
            "rows": [
                [
                    "n/a" if row.get(col) is None else _fmt(row.get(col))
                    for col in columns
                ]
                for row in rows
            ],
        }
    ]
    red_flags = [f for f in (o.get("red_flags") or []) if isinstance(f, dict)]
    if red_flags:
        text += "\n" + "\n".join(f"🚩 {str(f.get('message') or '')}" for f in red_flags)
    return text, tables


_PROFILE_SECTION_LABELS = {
    "overview": "概览",
    "target": "Target 分布",
    "missing": "缺失",
    "distribution": "分布",
    "correlation": "相关矩阵",
}

_PROFILE_CORRELATION_REASON_LABELS = {
    "insufficient_pairs": "有效样本不足",
    "zero_variance_left": "左侧常量",
    "zero_variance_right": "右侧常量",
    "zero_variance_both": "双侧常量",
    "nonfinite_result": "结果非有限值",
    "unsafe_numeric_precision": "数值精度不安全",
    "unsafe_numeric_precision_left": "左侧数值精度不安全",
    "unsafe_numeric_precision_right": "右侧数值精度不安全",
    "unsafe_numeric_precision_both": "双侧数值精度不安全",
}


def _profile_tagged_value(value) -> str:
    if not isinstance(value, dict):
        return "n/a" if value is None else str(value)
    if value.get("type") == "null":
        return "NULL"
    nonfinite = value.get("nonfinite")
    if nonfinite:
        return {
            "negative_infinity": "-Infinity",
            "positive_infinity": "+Infinity",
            "nan": "NaN",
        }.get(str(nonfinite), str(nonfinite))
    raw = value.get("value")
    return "n/a" if raw is None else str(raw)


def _profile_semantics(o: dict) -> tuple[dict, dict]:
    semantics = o.get("semantics") if isinstance(o.get("semantics"), dict) else {}
    roles = (
        semantics.get("field_roles")
        if isinstance(semantics.get("field_roles"), dict)
        else {}
    )
    names = (
        semantics.get("business_names")
        if isinstance(semantics.get("business_names"), dict)
        else {}
    )
    return roles, names


def _profile_field_label(name, business_names: dict) -> str:
    raw = str(name or "")
    business = str(business_names.get(raw) or "").strip()
    return f"{raw}（{business}）" if business else raw


def _profile_frequency_text(frequency: dict) -> str:
    items = frequency.get("items") if isinstance(frequency.get("items"), list) else []
    parts = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        parts.append(
            f"{_profile_tagged_value(item.get('value'))}:"
            f"{_fmt(item.get('count'))}({_pct(item.get('rate_all'))})"
        )
    other_count = frequency.get("other_count")
    if isinstance(other_count, int) and other_count > 0:
        parts.append(f"其他:{other_count}")
    return "；".join(parts) if parts else "n/a"


def _profile_correlation_cell(value, reason, pair_count) -> str:
    if reason == "ok" and value is not None:
        return _fmt(value)
    label = _PROFILE_CORRELATION_REASON_LABELS.get(str(reason), str(reason or "不可用"))
    return f"n/a（{label}，n={_fmt(pair_count)}）"


def _render_profile_dataset(o: dict):
    """Lay out deterministic profile evidence without calculating new metrics."""

    result = o.get("result") if isinstance(o.get("result"), dict) else {}
    dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
    fields = [
        field for field in (result.get("fields") or []) if isinstance(field, dict)
    ]
    options = o.get("options_echo") if isinstance(o.get("options_echo"), dict) else {}
    sections = [
        str(item) for item in (options.get("sections") or _PROFILE_SECTION_LABELS)
    ]
    requested = set(sections)
    roles, business_names = _profile_semantics(o)
    row_count = o.get("row_count_scanned", dataset.get("row_count", 0))
    text = (
        f"**样本描述分析完成**：数据集 `{str(o.get('dataset_id') or '')}`，"
        f"全量扫描 {_fmt(row_count)} 行；结果已绑定 dataset hash、分析代次和语义版本。"
    )
    if requested != set(_PROFILE_SECTION_LABELS):
        labels = "、".join(
            _PROFILE_SECTION_LABELS.get(section, section) for section in sections
        )
        text += f" 按请求展示：{labels}。"

    tables = []
    if "overview" in requested:
        tables.append(
            {
                "title": "字段概览",
                "columns": [
                    "字段",
                    "角色",
                    "类型",
                    "总行数",
                    "缺失数",
                    "缺失率",
                    "唯一值数",
                ],
                "rows": [
                    [
                        _profile_field_label(field.get("name"), business_names),
                        str(roles.get(str(field.get("name") or "")) or "—"),
                        str(field.get("duckdb_type") or field.get("kind") or "—"),
                        _fmt(field.get("row_count")),
                        _fmt(field.get("null_count")),
                        _pct(field.get("null_rate")),
                        _fmt(field.get("distinct_count")),
                    ]
                    for field in fields
                ],
            }
        )

    if "missing" in requested:
        tables.append(
            {
                "title": "缺失分析",
                "columns": ["字段", "缺失数", "缺失率", "总行数"],
                "rows": [
                    [
                        _profile_field_label(field.get("name"), business_names),
                        _fmt(field.get("null_count")),
                        _pct(field.get("null_rate")),
                        _fmt(field.get("row_count")),
                    ]
                    for field in sorted(
                        fields,
                        key=lambda item: (
                            -int(item.get("null_count") or 0),
                            str(item.get("name") or ""),
                        ),
                    )
                ],
            }
        )

    target = result.get("target_distribution")
    if "target" in requested and isinstance(target, dict):
        frequency = (
            target.get("frequency") if isinstance(target.get("frequency"), dict) else {}
        )
        items = [
            item for item in (frequency.get("items") or []) if isinstance(item, dict)
        ]
        tables.append(
            {
                "title": "Target 分布",
                "columns": ["取值", "样本数", "占比"],
                "rows": [
                    [
                        _profile_tagged_value(item.get("value")),
                        _fmt(item.get("count")),
                        _pct(item.get("rate_all")),
                    ]
                    for item in items
                ],
            }
        )

    if "distribution" in requested:
        tables.append(
            {
                "title": "字段分布",
                "columns": [
                    "字段",
                    "类型",
                    "Min",
                    "P25",
                    "P50",
                    "P75",
                    "Max",
                    "频数摘要",
                ],
                "rows": [
                    [
                        _profile_field_label(field.get("name"), business_names),
                        str(field.get("kind") or field.get("duckdb_type") or "—"),
                        _fmt((field.get("numeric") or {}).get("min"))
                        if isinstance(field.get("numeric"), dict)
                        else "n/a",
                        _fmt((field.get("numeric") or {}).get("p25"))
                        if isinstance(field.get("numeric"), dict)
                        else "n/a",
                        _fmt((field.get("numeric") or {}).get("p50"))
                        if isinstance(field.get("numeric"), dict)
                        else "n/a",
                        _fmt((field.get("numeric") or {}).get("p75"))
                        if isinstance(field.get("numeric"), dict)
                        else "n/a",
                        _fmt((field.get("numeric") or {}).get("max"))
                        if isinstance(field.get("numeric"), dict)
                        else "n/a",
                        _profile_frequency_text(field.get("frequency") or {})
                        if isinstance(field.get("frequency"), dict)
                        else "n/a",
                    ]
                    for field in fields
                ],
            }
        )

    correlations = result.get("correlations")
    if "correlation" in requested and isinstance(correlations, dict):
        columns = [str(item) for item in (correlations.get("columns") or [])]
        values = correlations.get("values") or []
        counts = correlations.get("pair_counts") or []
        reasons = correlations.get("reasons") or []
        rows = []
        for row_index, column in enumerate(columns):
            row = [_profile_field_label(column, business_names)]
            for column_index in range(len(columns)):
                value = values[row_index][column_index]
                reason = reasons[row_index][column_index]
                count = counts[row_index][column_index]
                row.append(_profile_correlation_cell(value, reason, count))
            rows.append(row)
        tables.append(
            {
                "title": "相关矩阵",
                "columns": [
                    "字段",
                    *[_profile_field_label(name, business_names) for name in columns],
                ],
                "rows": rows,
            }
        )
    return text, tables


_TRANSFORM_OPERATION_LABELS = {
    "rename_columns": "重命名字段",
    "drop_columns": "删除字段",
    "cast_columns": "转换字段类型",
    "fill_missing": "填补缺失",
    "filter_rows": "筛选样本",
    "derive_columns": "生成字段",
    "deduplicate": "去重",
}


def _transform_field_list(value) -> str:
    if not isinstance(value, (list, tuple)):
        return "n/a"
    fields = [str(item) for item in value]
    return "、".join(fields) if fields else "无"


def _transform_mapping_text(value) -> str:
    if not isinstance(value, dict) or not value:
        return "无"
    return "；".join(f"{source} → {target}" for source, target in value.items())


def _transform_impact_text(op: str, impact) -> str:
    """Render the kernel's impact evidence without deriving replacement metrics."""

    if not isinstance(impact, dict):
        return "n/a"
    if op == "rename_columns":
        return (
            f"重命名 {_fmt(impact.get('renamed_count'))} 个："
            f"{_transform_mapping_text(impact.get('mapping'))}"
        )
    if op == "drop_columns":
        return (
            f"删除 {_fmt(impact.get('dropped_count'))} 个："
            f"{_transform_field_list(impact.get('columns'))}"
        )
    if op == "cast_columns":
        return (
            f"转换字段：{_transform_field_list(impact.get('columns'))}；"
            f"输入非空 {_fmt(impact.get('non_null_input_count'))}；"
            f"无效转空 {_fmt(impact.get('invalid_to_null_count'))}"
        )
    if op == "fill_missing":
        return (
            f"填补 {_fmt(impact.get('filled_count'))} 个缺失值："
            f"{_transform_field_list(impact.get('columns'))}"
        )
    if op == "filter_rows":
        return (
            f"保留 {_fmt(impact.get('kept_rows'))} 行；"
            f"移除 {_fmt(impact.get('removed_rows'))} 行"
        )
    if op == "derive_columns":
        return (
            f"生成 {_fmt(impact.get('derived_count'))} 个字段："
            f"{_transform_field_list(impact.get('columns'))}"
        )
    if op == "deduplicate":
        return (
            f"按 {_transform_field_list(impact.get('keys'))} 去重；"
            f"保留 {_fmt(impact.get('kept_rows'))} 行；"
            f"移除 {_fmt(impact.get('removed_rows'))} 行"
        )
    scalar_items = [
        f"{key}={_fmt(value)}"
        for key, value in impact.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ]
    return "；".join(scalar_items) if scalar_items else "n/a"


def _render_transform_dataset(o: dict):
    """Present only persisted transform, semantic, workspace and lineage evidence."""

    source_id = str(o.get("source_dataset_id") or "")
    result_id = str(o.get("result_dataset_id") or "")
    text = (
        f"**数据加工完成**：`{source_id}`（{_fmt(o.get('row_count_before'))} 行 / "
        f"{_fmt(o.get('column_count_before'))} 列）→ `{result_id}`"
        f"（{_fmt(o.get('row_count_after'))} 行 / "
        f"{_fmt(o.get('column_count_after'))} 列）。"
    )
    if o.get("cached") is True:
        text += " 本次复用已验证证据。"

    steps = [item for item in (o.get("steps") or []) if isinstance(item, dict)]
    semantic = (
        o.get("semantic_migration")
        if isinstance(o.get("semantic_migration"), dict)
        else {}
    )
    protected = [str(item) for item in (semantic.get("dropped_protected_fields") or [])]
    if protected:
        text += f" 已按明确确认删除受保护字段：{'、'.join(protected)}。"

    tables = []
    if steps:
        tables.append(
            {
                "title": "加工步骤影响",
                "columns": [
                    "步骤",
                    "操作",
                    "加工前行数",
                    "加工后行数",
                    "行变化",
                    "影响证据",
                ],
                "rows": [
                    [
                        _fmt(step.get("step")),
                        _TRANSFORM_OPERATION_LABELS.get(
                            str(step.get("op") or ""),
                            str(step.get("op") or "未知操作"),
                        ),
                        _fmt(step.get("row_count_before")),
                        _fmt(step.get("row_count_after")),
                        _fmt(step.get("row_delta")),
                        _transform_impact_text(
                            str(step.get("op") or ""), step.get("impact")
                        ),
                    ]
                    for step in steps
                ],
            }
        )

    renamed_fields = semantic.get("renamed_fields")
    dropped_fields = [str(item) for item in (semantic.get("dropped_fields") or [])]
    semantic_rows = [
        [
            "语义映射 SHA-256",
            f"{semantic.get('before_hash') or 'n/a'} → {semantic.get('after_hash') or 'n/a'}",
        ],
        ["重命名字段", _transform_mapping_text(renamed_fields)],
        ["删除字段", "、".join(dropped_fields) if dropped_fields else "无"],
    ]
    if protected:
        semantic_rows.append(["已确认删除受保护字段", "、".join(protected)])
    tables.append(
        {
            "title": "字段语义迁移",
            "columns": ["项目", "工具证据"],
            "rows": semantic_rows,
        }
    )

    workspace = o.get("workspace") if isinstance(o.get("workspace"), dict) else {}
    tables.append(
        {
            "title": "Workspace 版本迁移",
            "columns": ["项目", "加工前", "加工后"],
            "rows": [
                [
                    "Revision",
                    _fmt(workspace.get("source_revision")),
                    _fmt(workspace.get("result_revision")),
                ],
                [
                    "分析代次",
                    _fmt(workspace.get("source_analysis_generation")),
                    _fmt(workspace.get("result_analysis_generation")),
                ],
            ],
        }
    )

    lineage = o.get("lineage") if isinstance(o.get("lineage"), dict) else {}
    tables.append(
        {
            "title": "数据血缘",
            "columns": ["父数据集", "子数据集", "关系", "边序号"],
            "rows": [
                [
                    str(lineage.get("parent_dataset_id") or ""),
                    str(lineage.get("child_dataset_id") or ""),
                    str(lineage.get("relation_kind") or ""),
                    _fmt(lineage.get("edge_order")),
                ]
            ],
        }
    )

    tables.append(
        {
            "title": "证据与下载",
            "columns": ["项目", "值"],
            "rows": [
                ["Run ID", str(o.get("run_id") or "")],
                ["结果 SHA-256", str(o.get("result_content_hash") or "")],
                ["证据产物", str(o.get("evidence_artifact_id") or "")],
                ["下载地址", str(o.get("evidence_download_url") or "")],
            ],
        }
    )
    return text, tables


_EXPORT_SAFETY_LABELS = (
    ("formula_cells_escaped", "公式注入转义"),
    ("text_column_cells_written", "文本字段单元格"),
    ("csv_text_cells_coerced", "CSV 文本保护"),
    ("large_integer_cells_as_text", "超长整数按文本"),
    ("decimal_cells_as_text", "小数按文本"),
    ("high_precision_decimal_cells_as_text", "高精度小数按文本"),
    ("non_finite_cells_as_text", "非有限数值按文本"),
    ("xlsx_control_characters_escaped", "Excel 控制字符转义"),
)


def _export_integer(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return "n/a"


def _export_size(value) -> str:
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    if size < 0:
        return "n/a"
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.2f} KB"
    if size < 1024**3:
        return f"{size / 1024**2:.2f} MB"
    return f"{size / 1024**3:.2f} GB"


def _render_export_dataset(o: dict):
    """Render only the persisted export artifact and safety evidence."""

    export_format = str(o.get("format") or "").casefold()
    if export_format == "csv":
        format_label = "CSV"
    elif export_format == "xlsx":
        format_label = "Excel"
    else:
        format_label = export_format.upper() or "未知格式"
    text = (
        f"**数据导出完成**：{format_label}，"
        f"{_export_integer(o.get('row_count'))} 行 / "
        f"{_export_integer(o.get('column_count'))} 列，"
        f"文件大小 {_export_size(o.get('size_bytes'))}。"
    )
    if o.get("cached") is True:
        text += " 本次复用已验证产物。"

    options = o.get("options") if isinstance(o.get("options"), dict) else {}
    text_columns = [str(item) for item in (options.get("text_columns") or [])]
    safety = o.get("safety") if isinstance(o.get("safety"), dict) else {}
    return text, [
        {
            "title": "导出文件",
            "columns": ["项目", "值"],
            "rows": [
                ["格式", format_label],
                ["数据集", str(o.get("dataset_id") or "")],
                ["数据集 SHA-256", str(o.get("dataset_content_hash") or "")],
                ["Workspace Revision", _num(o.get("workspace_revision"))],
                ["分析代次", _num(o.get("analysis_generation"))],
                ["文件大小", _export_size(o.get("size_bytes"))],
                ["文件 SHA-256", str(o.get("content_hash") or "")],
                ["按文本导出的字段", "、".join(text_columns) if text_columns else "无"],
                ["产物", str(o.get("artifact_id") or "")],
                ["下载地址", str(o.get("download_url") or "")],
            ],
        },
        {
            "title": "安全处理",
            "columns": ["保护项目", "处理单元格数"],
            "rows": [
                [label, _export_integer(safety.get(key))]
                for key, label in _EXPORT_SAFETY_LABELS
            ],
        },
    ]


def _render_propose_join(o: dict):
    joins = o.get("joins") or []
    # GAP-4: business-meaning lookup for raw key-column codes (e.g. als_m3_id_nbank_orgnum),
    # present only when the task has a registered data dictionary; {} otherwise, in which
    # case _key_label below degrades to the plain column name (no visible change).
    dictionary = o.get("dictionary") if isinstance(o.get("dictionary"), dict) else {}
    rows = []
    relax_rows = []
    any_conflict = False
    any_fp_mismatch = False
    any_dtype_mismatch = False
    for j in joins:
        diag = j.get("diagnostics") or {}
        match_rate = diag.get("match_rate")
        unique = diag.get("feature_key_unique")
        fan_out = diag.get("fan_out_detected", diag.get("fan_out"))
        # Prefer the friendly file name (features.parquet) over the raw ds_<hash> id.
        fname = str(j.get("feature_name") or j.get("feature_id", "?"))
        key_pairs = j.get("key_pairs") or []
        keys = (
            ", ".join(
                f"{_key_label(p.get('anchor_col'), dictionary)}={_key_label(p.get('feature_col'), dictionary)}"
                for p in key_pairs
            )
            or "?"
        )
        # Dynamic key relaxation proposals (spec §4/§5): low-match keys may match better with
        # one element dropped — surface as suggestions (the user confirms; never auto-applied).
        for alt in diag.get("key_alternatives") or []:
            if not isinstance(alt, dict):
                continue
            alt_keys = ", ".join(f"{a}={f}" for a, f in (alt.get("key_pairs") or []))
            relax_rows.append(
                [
                    fname,
                    _fmt(match_rate) if match_rate is not None else "n/a",
                    f"减「{alt.get('dropped', '?')}」→ {alt_keys}",
                    _fmt(alt.get("match_rate"))
                    if alt.get("match_rate") is not None
                    else "n/a",
                    "是" if alt.get("feature_key_unique") else "否",
                    "⚠️是" if alt.get("fan_out_detected") else "否",
                ]
            )
        # Fingerprint consistency (spec §5 C2 "指纹 raw=md5? ✓/✗"): transform_side == "both"
        # means anchor and feature key share format (both raw or both md5); anything else
        # means one side is raw and the other md5 (键格式不一致), joinable only via a hash
        # transform — surfaced so the user can sanity-check the key before执行.
        fp_consistent = (
            all((p.get("transform_side") or "both") == "both" for p in key_pairs)
            if key_pairs
            else True
        )
        if not fp_consistent:
            any_fp_mismatch = True
        fp_cell = "✓" if fp_consistent else "✗ raw≠md5"
        # T1-B8: key-dtype divergence (one side text, one side float/int) risks a silent miss
        # via precision / leading-zero loss. A "red" (text↔float) divergence forces confirm.
        divergences = [
            d for d in (diag.get("key_dtype_divergences") or []) if isinstance(d, dict)
        ]
        red_divergence = any(d.get("level") == "red" for d in divergences)
        if red_divergence:
            any_dtype_mismatch = True
            dtype_cell = "✗ text≠float"
        elif divergences:
            dtype_cell = "⚠️类型不一致"
        else:
            dtype_cell = "✓"
        # Two-level dedup breakdown (spec §6): safe whole-row dups vs same-key conflicts.
        report = diag.get("conflict_report") or {}
        conflict_keys = int(report.get("n_conflict_keys") or 0)
        safe_dropped = int(report.get("safe_dropped") or 0)
        if conflict_keys:
            any_conflict = True
        dedup_cell = "-" if unique else f"安全{safe_dropped}/⚠️冲突{conflict_keys}"
        rows.append(
            [
                fname,
                keys,
                fp_cell,
                dtype_cell,
                _fmt(match_rate) if match_rate is not None else "n/a",
                "是" if unique else "否",
                "⚠️是" if fan_out else "否",
                dedup_cell,
            ]
        )
    text = (
        f"**拼接诊断完成**:{len(joins)} 张特征表待左连接到锚样本（锚行数 **1:1 保留**）。\n"
        "请核对每张表的命中率/键唯一性/是否膨胀。键不唯一的特征需选去重策略；确认后才会真正执行拼接。"
    )
    if any_conflict:
        text += (
            "\n\n⚠️ 检测到**同键值冲突**（同一键多行但特征值不一致）：这类**不会自动删除**，"
            "请先确认去重策略或清洗数据后再拼接。"
        )
    if any_fp_mismatch:
        text += (
            "\n\n⚠️ 检测到**键指纹不一致**（`✗ raw≠md5`:锚/特征侧一为原文、一为 md5）："
            "系统会自动对齐哈希后再连接，但请确认这是同一标识（避免误配）。"
        )
    if any_dtype_mismatch:
        text += (
            "\n\n⚠️ 检测到**键类型不一致**（一侧文本、一侧浮点/整型）：可能已发生精度丢失/前导零丢失"
            "导致静默漏配，请确认是否为同一标识后再拼接（需先确认才能执行）。"
        )
    if relax_rows:
        text += (
            "\n\n💡 部分特征表命中率偏低，**减一个识别要素**可提高命中（见下「择键建议」）："
            "系统只提议、不会自动改键；若减后**膨胀**需配合去重策略。请确认后再选用。"
        )
    # T3-1: dual-path reconciliation red flags. Each feature's match count is computed two
    # independent ways (DuckDB SQL vs pandas); any divergence beyond tolerance is a BLOCKING
    # red flag showing BOTH path values, so the human sees the disagreement rather than
    # rubber-stamping one number. Silent (no line) when the two paths agree.
    reconcile_summary = (
        o.get("reconcile_summary")
        if isinstance(o.get("reconcile_summary"), dict)
        else {}
    )
    for flag in reconcile_summary.get("red_flags") or []:
        if isinstance(flag, dict) and flag.get("message"):
            text += f"\n\n🚩 {str(flag.get('message'))}"
    tables = []
    if rows:
        tables.append(
            {
                "title": "拼接诊断（逐特征表）",
                "columns": [
                    "特征表",
                    "匹配键",
                    "指纹（raw=md5?）",
                    "键类型",
                    "命中率",
                    "键唯一",
                    "膨胀",
                    "去重（安全/冲突键）",
                ],
                "rows": rows,
            }
        )
    if relax_rows:
        tables.append(
            {
                "title": "择键建议（减要素换更高命中）",
                "columns": [
                    "特征表",
                    "当前命中率",
                    "建议键",
                    "减后命中率",
                    "减后唯一",
                    "减后膨胀",
                ],
                "rows": relax_rows,
            }
        )
    # T3: expandable "数字溯源" detail — the two-path match count + the provenance tuple
    # (dataset fingerprint / code version / params digest / seed) behind each feature's
    # headline number, so the displayed number is auditable back to its inputs.
    trust_rows = _join_trust_rows(joins)
    if trust_rows:
        tables.append(
            {
                "title": "数字溯源（对账 + 血缘）",
                "columns": [
                    "特征表",
                    "匹配行数(权威路)",
                    "匹配行数(独立路)",
                    "对账",
                    "数据指纹",
                    "代码版本",
                    "参数摘要",
                    "seed",
                ],
                "rows": trust_rows,
            }
        )
    return text, tables


def _join_trust_rows(joins) -> list[list[str]]:
    """T3: one row per feature summarizing its match-count reconciliation + provenance
    tuple. Shows both computation paths' values, the reconcile verdict, and the minimal
    lineage (fingerprints truncated for readability). Empty when the trust layer is
    absent (e.g. an older stored plan) so this degrades to no extra table."""
    rows = []
    for j in joins:
        if not isinstance(j, dict):
            continue
        rec = j.get("reconcile") if isinstance(j.get("reconcile"), dict) else None
        prov = j.get("provenance") if isinstance(j.get("provenance"), dict) else None
        if not rec and not prov:
            continue
        fname = str(j.get("feature_name") or j.get("feature_id", "?"))
        primary = rec.get("primary") if rec else None
        secondary = rec.get("secondary") if rec else None
        # T3-2: an honest verdict. A number with no independent second path is "未独立复核",
        # NOT the ✓ 一致 (agree) badge -- a same-path self-comparison must never look verified.
        if rec and rec.get("trust") == "not_independently_verified":
            verdict = "⚠ 未独立复核"
        elif rec and rec.get("consistent"):
            verdict = "✓ 一致"
        elif rec:
            verdict = "🚩 分歧"
        else:
            verdict = "—"
        rows.append(
            [
                fname,
                _fmt(primary) if primary is not None else "n/a",
                _fmt(secondary) if secondary is not None else "n/a",
                verdict,
                _short_digest(prov.get("dataset_fingerprint")) if prov else "—",
                str(prov.get("code_version") or "—") if prov else "—",
                _short_digest(prov.get("params_digest")) if prov else "—",
                str(prov.get("seed")) if prov and prov.get("seed") is not None else "—",
            ]
        )
    return rows


def _short_digest(value) -> str:
    """Truncate a ``sha256:<hex>`` digest to a readable prefix for gate display."""
    text = str(value or "")
    if text.startswith("sha256:"):
        body = text[len("sha256:") :]
        return f"sha256:{body[:12]}…" if len(body) > 12 else text
    return text[:16] + "…" if len(text) > 16 else text


def _render_confirm_join(o: dict):
    # Internal plumbing step (marks engine specs confirmed). It is a dependency of
    # the execute_join gate, but its summary would show "已确认…" before the human
    # actually confirms, which is confusing — so render nothing at the gate…
    # T1-B8: a red key-dtype mismatch blocks confirmation until the user acknowledges it.
    needs_dtype = o.get("needs_dtype_ack") or []
    if needs_dtype:
        labels = o.get("needs_dtype_ack_labels") or {}
        listed = "、".join(f"`{labels.get(f, f)}`" for f in needs_dtype)
        return (
            f"⚠️ 特征 {listed} 的**拼接键类型不一致**（一侧文本、一侧浮点：可能已丢失精度/前导零，"
            "导致静默漏配）。请确认这是同一标识后回复「确认键类型」继续；或以字符串重导入该列后重试。"
        ), []
    needs = o.get("needs_dedup") or []
    if needs:
        # …UNLESS a feature has a same-key conflict (spec §6): surface it so the user knows
        # the join can't execute until they pick a dedup strategy (or exclude the feature).
        labels = o.get("needs_dedup_labels") or {}
        listed = "、".join(f"`{labels.get(f, f)}`" for f in needs)
        return (
            f"⚠️ 特征 {listed} 存在**同键冲突**（同一键多行、特征值不一致），"
            "需先定去重策略才能拼接。回复「去重 first」（保留首条）或「去重 last」（保留末条）解决；"
            "或排除这些特征后重试。"
        ), []
    return "", []


def _render_execute_join(o: dict):
    anchor_rows = o.get("anchor_rows")
    joined_rows = o.get("joined_rows")
    ok = anchor_rows == joined_rows
    text = (
        f"**拼接执行完成**:结果数据集 `{o.get('result_dataset_id', '')}`，"
        f"锚行 {anchor_rows} → 拼接后 {joined_rows} 行"
        + ("（1:1 保持 ✓）" if ok else "（⚠️ 行数发生变化，请检查膨胀）")
    )
    warnings = o.get("warnings") or []
    if warnings:
        text += "\n警告:" + "; ".join(str(w) for w in warnings)
    # §8 per-table contribution summary from real diagnostics.
    tables = []
    per_table = [row for row in (o.get("per_table") or []) if isinstance(row, dict)]
    if per_table:
        tables.append(
            {
                "title": "各特征表贡献",
                "columns": ["特征表", "命中率", "新增列", "新列缺失率", "去重策略"],
                "rows": [
                    [
                        str(row.get("feature_id", "?")),
                        _num(row.get("match_rate")),
                        str(row.get("new_columns", "")),
                        _num(row.get("new_columns_null_rate")),
                        str(row.get("dedup_strategy", "无")),
                    ]
                    for row in per_table
                ],
            }
        )
    return text, tables


def _render_post_training_action(o: dict):
    actions = [item for item in (o.get("actions") or []) if isinstance(item, dict)]
    succeeded = sum(1 for item in actions if item.get("status") == "succeeded")
    skipped = sum(1 for item in actions if item.get("status") == "skipped")
    text = (
        f"**训练后交付动作完成**:成功 {succeeded} 个，跳过 {skipped} 个。"
        if actions
        else "**训练后交付动作完成**。"
    )
    rows = [
        [
            "原生模型",
            "succeeded" if o.get("native_model_path") else "missing",
            o.get("native_model_path") or "",
            "",
        ],
    ]
    if o.get("approval_package_path"):
        rows.append(
            [
                "审批包",
                "succeeded",
                o.get("approval_package_markdown_path")
                or o.get("approval_package_path"),
                "模型审批与交付证据包",
            ]
        )
    if o.get("model_card_path"):
        rows.append(
            [
                "模型卡",
                "succeeded",
                o.get("model_card_markdown_path") or o.get("model_card_path"),
                "最终模型卡",
            ]
        )
    if o.get("monitoring_policy_path"):
        monitoring = (
            o.get("monitoring_policy")
            if isinstance(o.get("monitoring_policy"), dict)
            else {}
        )
        rows.append(
            [
                "监控策略",
                monitoring.get("status") or "succeeded",
                o.get("monitoring_policy_markdown_path")
                or o.get("monitoring_policy_path"),
                monitoring.get("recommendation") or "模型监控阈值策略",
            ]
        )
    for item in actions:
        action = str(item.get("action") or "")
        status = str(item.get("status") or "")
        artifact = (
            item.get("pmml_path")
            or item.get("validation_task_id")
            or item.get("challenger_task_id")
            or item.get("markdown_path")
            or item.get("package_path")
            or ""
        )
        rows.append([action, status, artifact, str(item.get("reason") or "")])
    tables = [
        {
            "title": "训练后交付状态",
            "columns": ["动作", "状态", "产物/任务", "说明"],
            "rows": rows,
        }
    ]
    caps = o.get("capabilities") or {}
    if caps:
        cap_rows = [
            ["PMML", "是" if caps.get("pmml_supported") else "否"],
            ["移交验证", "是" if caps.get("handoff_supported") else "否"],
            ["原生模型", "是" if caps.get("native_model_supported") else "否"],
        ]
        if caps.get("reason"):
            cap_rows.append(["说明", caps.get("reason")])
        tables.append(
            {"title": "最终模型交付能力", "columns": ["能力", "状态"], "rows": cap_rows}
        )
    return text, tables


def _render_make_split(o: dict):
    """G1 split gate: surface the train/test/oot counts + per month/channel distribution so
    the user can sanity-check the split (proportions, OOT-by-time, no cross-group leakage)
    before spending compute on screening/training."""
    analysis = o.get("sample_analysis") or {}
    counts = analysis.get("split_counts") or {}
    total = analysis.get("total_rows")
    rows = [
        [str(split), int(n), _fmt(n / total) if total else "n/a"]
        for split, n in counts.items()
    ]
    text = (
        f"**样本切分完成**:共 {total} 行。请核对 train/test/oot 划分"
        "（占比是否合理、OOT 是否按时间、分组是否防泄漏）后再继续。"
    )
    tables = []
    if rows:
        tables.append(
            {
                "title": "切分计数（train/test/oot）",
                "columns": ["划分", "行数", "占比"],
                "rows": rows,
            }
        )
    for group_col, dist in (analysis.get("group_distributions") or {}).items():
        if not isinstance(dist, dict):
            continue
        group_values = sorted(
            {gv for per in dist.values() if isinstance(per, dict) for gv in per}
        )
        grows = [
            [str(split)] + [int(per.get(gv, 0)) for gv in group_values]
            for split, per in dist.items()
            if isinstance(per, dict)
        ]
        if grows:
            tables.append(
                {
                    "title": f"按「{group_col}」分布（逐划分）",
                    "columns": ["划分", *[str(gv) for gv in group_values]],
                    "rows": grows,
                }
            )
    return text, tables


def _render_score_dataset(o: dict):
    direction_label = (
        "分数越高风险越高"
        if o.get("score_direction") == "higher_is_riskier"
        else "分数越高风险越低"
    )
    text = (
        f"**打分完成**（{direction_label}）:"
        f"{_fmt(o.get('row_count'))} 行,分数列 `{o.get('score_col')}`,"
        f"缺失率 {_pct(o.get('score_missing_rate'))}。"
    )
    rows = [
        ["数据集", o.get("result_dataset_id") or ""],
        ["分数列", o.get("score_col") or ""],
        ["分数缺失率", _pct(o.get("score_missing_rate"))],
    ]
    if o.get("points_col"):
        text += f" 评分卡 points 列 `{o.get('points_col')}`。"
        rows.append(["points 列", o.get("points_col") or ""])
        rows.append(["points 缺失率", _pct(o.get("points_missing_rate"))])
    return text, [{"title": "打分结果摘要", "columns": ["项", "值"], "rows": rows}]


#: S1b/DOM-3: gate copy fragments per monitor_run verdict level, keyed by the
#: same green/amber/red vocabulary tool_monitor_run emits -- the confirmation
#: gate text names which flags fired and the suggested action so a weak LLM (or
#: a manual reviewer) doesn't have to re-derive it from the raw checks table.
_MONITOR_LEVEL_LABEL = {"green": "绿", "amber": "黄", "red": "红"}


def _render_monitor_run(o: dict):
    overall = str(o.get("overall_level") or "")
    checks = [c for c in (o.get("checks") or []) if isinstance(c, dict)]
    red_flags = [c for c in checks if c.get("level") == "red"]
    amber_flags = [c for c in checks if c.get("level") == "amber"]
    label = _MONITOR_LEVEL_LABEL.get(overall, overall)
    text = f"**监控运行完成**:总体判级【{label}】。{o.get('recommendation') or ''}"
    if red_flags:
        names = "、".join(str(c.get("label") or c.get("id")) for c in red_flags)
        text += f" 红旗:{names}。"
    if amber_flags:
        names = "、".join(str(c.get("label") or c.get("id")) for c in amber_flags)
        text += f" 黄旗:{names}。"
    rows = [
        [
            str(c.get("label") or c.get("id") or ""),
            _MONITOR_LEVEL_LABEL.get(str(c.get("level")), str(c.get("level"))),
            _fmt(c.get("value")) if c.get("value") is not None else "n/a",
            str(c.get("message") or ""),
        ]
        for c in checks
    ]
    tables = [
        {
            "title": "监控判级明细",
            "columns": ["检查项", "判级", "值", "说明"],
            "rows": rows,
        }
    ]
    drifted = [
        row for row in (o.get("top_drifted_features") or []) if isinstance(row, dict)
    ]
    if drifted:
        tables.append(
            {
                "title": "特征漂移 Top",
                "columns": ["特征", "CSI"],
                "rows": [
                    [str(row.get("feature") or ""), _fmt(row.get("csi"))]
                    for row in drifted[:10]
                ],
            }
        )
    return text, tables


#: S5: red-light disposition checklist injected into the strategy-monitoring
#: alarm gate copy. The three options map to the driver's disposition keywords
#: (观察 / 调阈值 / 起新版本); the gate reply parser recognises these keywords.
_MONITORING_RED_CHECKLIST = (
    "处置建议(红灯,请三选一并回复关键词):",
    "1. 维持并观察 —— 回复「观察」保持当前策略,加强下一周期监控;",
    "2. 调阈值重跑 —— 回复「调阈值」调整监控计划阈值后重新运行监控;",
    "3. 起新版本策略 —— 回复「起新版本」基于当前策略起一个新版本重走策略开发。",
)


def _render_run_strategy_monitoring(o: dict):
    overall = str(o.get("overall_level") or "")
    checks = [c for c in (o.get("checks") or []) if isinstance(c, dict)]
    red_flags = [c for c in checks if c.get("level") == "red"]
    amber_flags = [c for c in checks if c.get("level") == "amber"]
    label = _MONITOR_LEVEL_LABEL.get(overall, overall)
    text = f"**策略监控完成**:总体判级【{label}】。"
    if red_flags:
        names = "、".join(str(c.get("label") or c.get("id")) for c in red_flags)
        text += f" 红旗:{names}。"
    if amber_flags:
        names = "、".join(str(c.get("label") or c.get("id")) for c in amber_flags)
        text += f" 黄旗:{names}。"
    if overall == "red":
        text += "\n\n" + "\n".join(_MONITORING_RED_CHECKLIST)
    rows = [
        [
            str(c.get("label") or c.get("id") or ""),
            _MONITOR_LEVEL_LABEL.get(str(c.get("level")), str(c.get("level"))),
            _fmt(c.get("value")) if c.get("value") is not None else "n/a",
            str(c.get("message") or ""),
        ]
        for c in checks
    ]
    tables = [
        {
            "title": "监控判级明细",
            "columns": ["检查项", "判级", "值", "说明"],
            "rows": rows,
        }
    ]
    drifted = [
        row for row in (o.get("top_drifted_features") or []) if isinstance(row, dict)
    ]
    if drifted:
        tables.append(
            {
                "title": "特征漂移 Top",
                "columns": ["特征", "CSI"],
                "rows": [
                    [str(row.get("feature") or ""), _fmt(row.get("csi"))]
                    for row in drifted[:10]
                ],
            }
        )
    return text, tables


def _render_apply_monitoring_disposition(o: dict):
    disposition = str(o.get("disposition") or "acknowledge")
    label = {
        "acknowledge": "确认知悉",
        "observe": "维持并观察",
        "adjust_threshold": "调整阈值并重跑",
        "new_version": "创建新版本",
    }.get(disposition, disposition)
    level = str(o.get("overall_level") or "")
    level_label = _MONITOR_LEVEL_LABEL.get(level, level)
    text = f"**监控处置已执行**：{label}"
    if level_label:
        text += f"，处置后判级【{level_label}】"
    resolved_run = o.get("resolved_monitoring_run_id")
    if resolved_run:
        text += f"，证据运行 `{resolved_run}`"
    if disposition == "new_version" and o.get("new_task_id"):
        text += (
            f"。已创建策略任务 `{o['new_task_id']}` 与草案策略 "
            f"`{o.get('new_strategy_id', '')}`"
        )
    if disposition == "adjust_threshold" and o.get("monitoring_plan_revision"):
        text += f"。监控计划已追加为 revision {o['monitoring_plan_revision']}"
    return text + "。", []


def _render_monitoring_report(o: dict):
    timeline = [row for row in (o.get("timeline") or []) if isinstance(row, dict)]
    overall = str(o.get("overall_level") or "")
    label = _MONITOR_LEVEL_LABEL.get(overall, overall) if overall else ""
    head = f"**监控报告已生成**:`{o.get('report_path', '')}`"
    if label:
        head += f",最近总体判级【{label}】"
    head += f",历史监控 {len(timeline)} 次。"
    next_action = (
        o.get("next_action") if isinstance(o.get("next_action"), dict) else None
    )
    if next_action and next_action.get("prompt"):
        head += f"\n\n下一步:{next_action['prompt']}"
    tables = []
    if timeline:
        tables.append(
            {
                "title": "监控判级时间线",
                "columns": ["时间", "总体判级", "样本量"],
                "rows": [
                    [
                        str(row.get("at") or ""),
                        _MONITOR_LEVEL_LABEL.get(
                            str(row.get("overall_level")),
                            str(row.get("overall_level") or ""),
                        ),
                        _fmt(row.get("row_count"))
                        if row.get("row_count") is not None
                        else "",
                    ]
                    for row in timeline
                ],
            }
        )
    return head, tables


def _render_flow_rate(o: dict):
    months = [str(m) for m in (o.get("months") or [])]
    net_flows = [row for row in (o.get("net_flows") or []) if isinstance(row, dict)]
    red_flags = [f for f in (o.get("red_flags") or []) if isinstance(f, dict)]
    text = f"**桶流量分析完成**:{len(months)} 个相邻月对。"
    if red_flags:
        text += f" 红旗 {len(red_flags)} 项。"
    tables = []
    if net_flows:
        tables.append(
            {
                "title": "逐月净流量（进入坏 / 退出坏）",
                "columns": ["月份", "进入坏", "退出坏"],
                "rows": [
                    [
                        str(r.get("month") or ""),
                        _fmt(r.get("into_bad")),
                        _fmt(r.get("out_of_bad")),
                    ]
                    for r in net_flows
                ],
            }
        )
    if red_flags:
        tables.append(_data_quality_flag_table(red_flags))
    return text, tables


def _render_bucket_migration(o: dict):
    states = [str(s) for s in (o.get("states") or [])]
    to_states = [str(s) for s in (o.get("to_states") or [])]
    heat_table = [row for row in (o.get("heat_table") or []) if isinstance(row, dict)]
    red_flags = [f for f in (o.get("red_flags") or []) if isinstance(f, dict)]
    window = [str(m) for m in (o.get("window_months") or [])]
    text = f"**桶迁徙热力完成**:{len(states)} 状态，窗口 {len(window)} 个月。"
    tables = []
    if heat_table and to_states:
        columns = ["from", *to_states]
        rows = [
            [str(row.get("from") or ""), *[_pct(row.get(state)) for state in to_states]]
            for row in heat_table
        ]
        # matrix-heat column_specs: the from label is text, each to-state cell is a
        # heat cell colored from its own 0..1 migration rate (frontend matrix-heat).
        column_specs = [{"kind": "text"}, *[{"kind": "matrix-heat"} for _ in to_states]]
        tables.append(
            {
                "title": "平均迁徙率矩阵",
                "columns": columns,
                "rows": rows,
                "column_specs": column_specs,
            }
        )
    if red_flags:
        tables.append(_data_quality_flag_table(red_flags))
    return text, tables


def _render_segment_profile(o: dict):
    segments = [row for row in (o.get("segments") or []) if isinstance(row, dict)]
    conc = o.get("concentration") if isinstance(o.get("concentration"), dict) else {}
    red_flags = [f for f in (o.get("red_flags") or []) if isinstance(f, dict)]
    text = (
        f"**细分画像完成**:{len(segments)} 个细分，top1 占比 {_pct(conc.get('top1_pct'))}，"
        f"HHI {_fmt(conc.get('hhi'))}。"
    )
    tables = []
    if segments:
        tables.append(
            {
                "title": "细分画像",
                "columns": ["细分", "样本数", "占比", "坏率", "均分", "净利润"],
                "rows": [
                    [
                        str(r.get("segment") or ""),
                        _fmt(r.get("count")),
                        _pct(r.get("pop_pct")),
                        _pct(r.get("bad_rate"))
                        if r.get("bad_rate") is not None
                        else "n/a",
                        _fmt(r.get("avg_score"))
                        if r.get("avg_score") is not None
                        else "n/a",
                        _fmt(r.get("net_profit"))
                        if r.get("net_profit") is not None
                        else "n/a",
                    ]
                    for r in segments
                ],
            }
        )
    if red_flags:
        tables.append(_data_quality_flag_table(red_flags))
    return text, tables


def _render_el_estimate(o: dict):
    chain = [row for row in (o.get("chain") or []) if isinstance(row, dict)]
    el_by_month = [row for row in (o.get("el_by_month") or []) if isinstance(row, dict)]
    red_flags = [f for f in (o.get("red_flags") or []) if isinstance(f, dict)]
    assumptions = o.get("assumptions") if isinstance(o.get("assumptions"), dict) else {}
    ref = assumptions.get("reference_snapshot")
    # total_el is a reference-snapshot口径 (latest month), NOT a cross-month sum;
    # annotate the headline so the user ties合计 EL to a specific month.
    basis_note = f"（参考快照 {ref} 口径）" if ref else ""
    text = f"**预期损失估计完成**:损失态 `{o.get('loss_state', '')}`，合计 EL {_fmt(o.get('total_el'))}{basis_note}。"
    tables = []
    if chain:
        tables.append(
            {
                "title": "各状态到损失态的吸收概率",
                "columns": ["起始状态", "P(损失)"],
                "rows": [
                    [str(r.get("from_state") or ""), _pct(r.get("p_to_loss"))]
                    for r in chain
                ],
            }
        )
    if el_by_month:
        tables.append(
            {
                "title": "逐月预期损失",
                "columns": ["月份", "余额", "预期损失"],
                "rows": [
                    [
                        ("★ " if r.get("is_reference") else "")
                        + str(r.get("month") or ""),
                        _fmt(r.get("balance")),
                        _fmt(r.get("expected_loss")),
                    ]
                    for r in el_by_month
                ],
            }
        )
    if red_flags:
        tables.append(_data_quality_flag_table(red_flags))
    return text, tables


def _render_portfolio_report(o: dict):
    sheets = [str(s) for s in (o.get("sheets") or [])]
    text = (
        f"**组合报告已生成**:`{o.get('report_path', '')}`，含 {len(sheets)} 个 sheet。"
    )
    tables = []
    if sheets:
        tables.append(
            {
                "title": "报告 sheet",
                "columns": ["#", "sheet"],
                "rows": [[str(i), s] for i, s in enumerate(sheets, start=1)],
            }
        )
    return text, tables


def _render_portfolio_gate_summary(o: dict):
    checklist = [str(item) for item in (o.get("checklist") or [])]
    highlights = o.get("highlights") if isinstance(o.get("highlights"), dict) else {}
    text = f"**组合分析汇总**:{o.get('red_flag_count', 0)} 项数据质量红旗，请确认后生成报告。"
    tables = []
    if highlights:
        tables.append(
            {
                "title": "关键数字",
                "columns": ["指标", "值"],
                "rows": [[str(k), _fmt(v)] for k, v in highlights.items()],
            }
        )
    if checklist:
        tables.append(
            {
                "title": "红旗 checklist",
                "columns": ["#", "红旗"],
                "rows": [[str(i), item] for i, item in enumerate(checklist, start=1)],
            }
        )
    return text, tables


def _data_quality_flag_table(red_flags: list[dict]) -> dict:
    return {
        "title": "数据质量红旗",
        "columns": ["类型", "说明"],
        "rows": [
            [str(f.get("kind") or ""), str(f.get("message") or "")] for f in red_flags
        ],
    }


def _render_mine_rules(o: dict):
    rules = [
        rule for rule in (o.get("candidate_rules") or []) if isinstance(rule, dict)
    ]
    red_flags = [flag for flag in (o.get("red_flags") or []) if isinstance(flag, dict)]
    n_rows = o.get("n_rows")
    text = (
        f"**规则挖掘完成**:在 {_fmt(n_rows)} 行样本上提议 **{len(rules)}** 条候选拒绝规则"
        "（按 lift 降序；决策树路径 + 单变量切点两通道）。请回复「选 1,3,5」/「去掉 2」/「全选」选定要采纳的规则集。"
    )
    red_items = [flag for flag in red_flags if flag.get("level") == "red"]
    if red_items:
        names = "、".join(str(flag.get("code")) for flag in red_items)
        text += f" 红旗:{names}。"
    tables = []
    if rules:
        tables.append(
            {
                "title": "候选规则（按 lift 降序）",
                "columns": ["#", "规则", "支持度", "命中坏率", "lift", "来源"],
                "rows": [
                    [
                        str(index),
                        str(rule.get("condition", "")),
                        _pct(rule.get("support")),
                        _pct(rule.get("hit_bad_rate")),
                        _num(rule.get("lift")),
                        str(rule.get("source", "")),
                    ]
                    for index, rule in enumerate(rules, start=1)
                ],
            }
        )
    if red_flags:
        tables.append(_red_flag_table(red_flags))
    return text, tables


def _render_select_rule_set(o: dict):
    selected = [
        rule for rule in (o.get("selected_rules") or []) if isinstance(rule, dict)
    ]
    candidate_count = o.get("candidate_count")
    text = (
        f"**规则集已选定**:从 {_fmt(candidate_count)} 条候选中选定 **{len(selected)}** 条规则"
        "（按选定顺序，首个命中生效）。确认后将评估该规则集并构造策略。"
    )
    tables = []
    if selected:
        tables.append(
            {
                "title": "已选规则（按顺序命中）",
                "columns": ["#", "规则", "命中坏率", "lift", "来源"],
                "rows": [
                    [
                        str(index),
                        str(rule.get("condition", "")),
                        _pct(rule.get("hit_bad_rate"))
                        if rule.get("hit_bad_rate") is not None
                        else "n/a",
                        _num(rule.get("lift"))
                        if rule.get("lift") is not None
                        else "n/a",
                        str(rule.get("source", "")),
                    ]
                    for index, rule in enumerate(selected, start=1)
                ],
            }
        )
    return text, tables


def _render_evaluate_rule_set(o: dict):
    waterfall = [row for row in (o.get("waterfall") or []) if isinstance(row, dict)]
    residual = o.get("residual") if isinstance(o.get("residual"), dict) else {}
    combined = o.get("combined") if isinstance(o.get("combined"), dict) else {}
    red_flags = [flag for flag in (o.get("red_flags") or []) if isinstance(flag, dict)]
    text = (
        "**规则集评估完成**:"
        f"合计拒绝率 {_pct(combined.get('reject_rate'))}，"
        f"拒绝客群坏率 {_pct(combined.get('rejected_bad_rate'))}；"
        f"残余通过率 {_pct(residual.get('approval_rate'))}，"
        f"通过客群坏率 {_pct(residual.get('bad_rate'))}。"
    )
    red_items = [
        flag
        for flag in red_flags
        if flag.get("code") in {"rule_shadowed", "high_overlap"}
    ]
    if red_items:
        names = "、".join(str(flag.get("code")) for flag in red_items)
        text += f" 告警:{names}。"
    tables = []
    if waterfall:
        tables.append(
            {
                "title": "命中瀑布（按顺序，首个命中生效）",
                "columns": [
                    "规则",
                    "增量命中",
                    "增量坏率",
                    "累计拒绝率",
                    "累计拒绝坏率",
                ],
                "rows": [
                    [
                        str(row.get("rule_id", "")),
                        _fmt(row.get("incremental_hits")),
                        _pct(row.get("incremental_bad_rate")),
                        _pct(row.get("cum_reject_rate")),
                        _pct(row.get("cum_reject_bad_rate")),
                    ]
                    for row in waterfall
                ],
            }
        )
    overlap = o.get("overlap_matrix")
    if isinstance(overlap, list) and len(overlap) > 1:
        header = [f"R{index}" for index in range(1, len(overlap) + 1)]
        tables.append(
            {
                "title": "规则重叠矩阵（共同命中占比）",
                "columns": ["", *header],
                "rows": [
                    [
                        f"R{i + 1}",
                        *[_pct(overlap[i][j]) for j in range(len(overlap[i]))],
                    ]
                    for i in range(len(overlap))
                ],
            }
        )
    if red_flags:
        tables.append(_red_flag_table(red_flags))
    return text, tables


_RENDERERS = {
    "make_split": _render_make_split,
    "choose_modeling_spec": _render_choose_modeling_spec,
    "screen_features": _render_screen,
    "select_features": _render_select,
    "configure_tuning": _render_configure_tuning,
    "tune_hyperparameters": _render_tune,
    "train_model": _render_train,
    "train_models": _render_train_models,
    "compare_experiments": _render_compare,
    "select_experiment": _render_select_experiment,
    "post_training_action": _render_post_training_action,
    "generate_model_report": _render_report,
    "propose_join": _render_propose_join,
    "confirm_join": _render_confirm_join,
    "execute_join": _render_execute_join,
    "compute_feature_metrics": _render_feature_metrics,
    "generate_feature_report": _render_feature_report,
    "build_strategy": _render_build_strategy,
    "design_strategy_candidate": _render_design_strategy_candidate,
    "analyze_univariate_candidates": _render_analyze_univariate_candidates,
    "backtest_strategy": _render_backtest_strategy,
    "tradeoff_view": _render_tradeoff_view,
    "design_cutoff_bands": _render_design_cutoff_bands,
    "compare_strategies": _render_compare_strategies,
    "profit_calc": _render_profit_calc,
    "roll_rate_matrix": _render_roll_rate_matrix,
    "limit_pricing_matrix": _render_limit_pricing_matrix,
    "adopt_strategy": _render_adopt_strategy,
    "render_strategy_doc": _render_strategy_doc,
    "render_challenger_report": _render_challenger_report,
    "vintage_curve": _render_vintage_curve,
    "slice_aggregate": _render_slice_aggregate,
    "profile_dataset": _render_profile_dataset,
    "transform_dataset": _render_transform_dataset,
    "export_dataset": _render_export_dataset,
    "score_dataset": _render_score_dataset,
    "monitor_run": _render_monitor_run,
    "run_strategy_monitoring": _render_run_strategy_monitoring,
    "apply_monitoring_disposition": _render_apply_monitoring_disposition,
    "render_monitoring_report": _render_monitoring_report,
    "flow_rate": _render_flow_rate,
    "bucket_migration": _render_bucket_migration,
    "segment_profile": _render_segment_profile,
    "expected_loss_estimate": _render_el_estimate,
    "portfolio_gate_summary": _render_portfolio_gate_summary,
    "portfolio_report": _render_portfolio_report,
    "evaluate_rule_set": _render_evaluate_rule_set,
    "mine_rules": _render_mine_rules,
    "select_rule_set": _render_select_rule_set,
}


def _render_generic(o: dict):
    if not isinstance(o, dict) or not o:
        return "已完成。", []
    scalar = {k: v for k, v in o.items() if isinstance(v, (str, int, float, bool))}
    if scalar:
        head = ", ".join(f"{k}={_fmt(v)}" for k, v in list(scalar.items())[:6])
        return f"已完成:{head}", []
    return "已完成。", []


def render_tool_output(tool: str, output: dict):
    """Render a tool's raw output to (text, tables); falls back to generic."""
    renderer = _RENDERERS.get(tool, _render_generic)
    try:
        return renderer(output or {})
    except Exception:
        return _render_generic(output or {})


__all__ = ["render_tool_output"]

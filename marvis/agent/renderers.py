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


def _render_screen(o: dict):
    selected = o.get("selected") or []
    leak = o.get("leakage") or []
    susp = o.get("suspected") or []
    unusable = o.get("unusable") or []
    scores = o.get("scores") if isinstance(o.get("scores"), dict) else {}
    leak_names = _names(leak)
    susp_names = _names(susp)
    n = o.get("n_screened") or o.get("n") or (len(selected) + len(leak) + len(susp))
    text = (
        f"**特征筛选完成**:从 {n} 个候选中提议保留 **{len(selected)}** 个特征。\n"
        f"- 剔除疑似**泄漏** {len(leak_names)} 个" + (f"(如 {leak_names[:3]})" if leak_names else "") + "\n"
        f"- 疑似**模型输出/评分**列 {len(susp_names)} 个" + (f"(如 {susp_names[:5]})" if susp_names else "") + "\n"
        f"- 剔除**不可用**(常量/稀疏) {len(unusable)} 个"
    )
    tables = []
    if selected:
        rows = []
        for feat in selected[:20]:
            s = scores.get(feat) if isinstance(scores.get(feat), dict) else {}
            rows.append([feat, _num(s.get("ks")), _num(s.get("iv")), _pct(s.get("missing_rate"))])
        tables.append({"title": "入选特征(前20)", "columns": ["特征", "KS", "IV", "缺失率"], "rows": rows})
    if leak:
        tables.append({
            "title": f"疑似泄漏(KS≥阈值,共{len(leak)})",
            "columns": ["特征", "KS", "原因"],
            "rows": [[f, _num(k), r] for f, k, r in (_triple(i) for i in leak[:20])],
        })
    if susp:
        tables.append({
            "title": f"疑似模型输出/评分列(共{len(susp)})",
            "columns": ["特征", "KS", "原因"],
            "rows": [[f, _num(k), r] for f, k, r in (_triple(i) for i in susp[:20])],
        })
    if unusable:
        rows = []
        for item in unusable[:20]:
            if isinstance(item, (list, tuple)):
                rows.append([str(item[0]) if item else "", str(item[1]) if len(item) > 1 else ""])
            else:
                rows.append([str(item), ""])
        tables.append({"title": f"剔除·不可用(常量/稀疏,共{len(unusable)})", "columns": ["特征", "原因"], "rows": rows})
    return text, tables


def _render_choose_modeling_spec(o: dict):
    recipes = [str(item) for item in (o.get("recipes") or [])]
    target_type = str(o.get("target_type") or "binary")
    sample_weight_col = str(o.get("sample_weight_col") or "")
    metric_policy = str(o.get("metric_policy") or "")
    text = (
        f"**建模规格已生成**:目标类型 `{target_type}`,"
        f"算法 {'/'.join(recipes) or '-'},选择策略 `{metric_policy}`。"
    )
    tables = [{
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
    }]
    eligible = o.get("eligible_algorithms") or []
    disabled = [item for item in (o.get("disabled_algorithms") or []) if isinstance(item, dict)]
    if eligible or disabled:
        tables.append({
            "title": "算法可用性",
            "columns": ["算法", "状态", "说明"],
            "rows": (
                [[str(recipe), "可用", ""] for recipe in eligible]
                + [[str(item.get("recipe", "")), "不可用", str(item.get("reason", ""))] for item in disabled]
            ),
        })
    diagnostics = [item for item in (o.get("sample_weight_diagnostics") or []) if isinstance(item, dict)]
    if diagnostics:
        tables.append({
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
        })
    warnings = [str(item) for item in (o.get("warnings") or [])]
    if warnings:
        text += "\n" + "\n".join(f"- {warning}" for warning in warnings)
    return text, tables


def _render_configure_tuning(o: dict):
    tune_enabled = bool(o.get("tune_enabled"))
    sample_weight_col = str(o.get("sample_weight_col") or "")
    budgets = o.get("n_trials_by_recipe") if isinstance(o.get("n_trials_by_recipe"), dict) else {}
    recipes = [str(item) for item in (o.get("recipes") or []) if str(item)]
    total_n_trials = o.get("total_n_trials")
    multi = len(budgets) > 1
    if multi:
        budget_note = "、".join(f"{recipe}={budgets[recipe]}" for recipe in recipes if recipe in budgets)
        text = (
            f"**调参配置已生成**:候选算法 {'/'.join(recipes)},"
            f"{'每个算法各自执行' if tune_enabled else '跳过'}两阶段随机搜索"
            f"(按算法预算 {budget_note};多算法总预算=Σ各配方预算={_fmt(total_n_trials)} 轮)。"
        )
    else:
        text = (
            f"**调参配置已生成**:算法 `{o.get('recipe', '')}`,"
            f"{'执行' if tune_enabled else '跳过'}两阶段随机搜索,"
            f"轮数 {o.get('n_trials', 0)}。"
        )
    rows = [
        ["目标类型", str(o.get("target_type") or "")],
        ["算法", "/".join(recipes) if recipes else str(o.get("recipe") or "")],
        ["随机搜索", "是" if tune_enabled else "否"],
    ]
    if multi:
        rows.append(["按算法调参预算(轮数,总预算=Σ各配方预算)", "、".join(f"{recipe}={budgets[recipe]}" for recipe in recipes if recipe in budgets)])
        rows.append(["总预算", _fmt(total_n_trials)])
    else:
        rows.append(["调参轮数", _fmt(o.get("n_trials", ""))])
    rows.append(["样本权重列", sample_weight_col or "不使用"])
    rows.append(["说明", str(o.get("reason") or "")])
    tables = [{
        "title": "调参配置",
        "columns": ["项目", "值"],
        "rows": rows,
    }]
    params = o.get("params") if isinstance(o.get("params"), dict) else {}
    if params:
        tables.append({
            "title": "固定/控制参数",
            "columns": ["参数", "值"],
            "rows": [[str(key), _fmt(value)] for key, value in params.items()],
        })
    return text, tables


def _render_tune(o: dict):
    best_params = o.get("best_params") or {}
    best_metrics = o.get("best_metrics") or {}
    trials = [t for t in (o.get("trials") or []) if isinstance(t, dict)]
    text = f"**调参完成**:{o.get('n_trials', '?')} 轮搜索,选出最优超参组合。"
    tables = []
    if trials:
        # trials leaderboard (G4): each trial's train/test/oot KS + overfit gap,
        # ranked by the in-time selection score (OOT is the unbiased final metric).
        ranked = sorted(
            trials,
            key=lambda t: t.get("score") if isinstance(t.get("score"), (int, float)) else float("-inf"),
            reverse=True,
        )
        rows = []
        for rank, trial in enumerate(ranked[:15], start=1):
            train_ks, test_ks = trial.get("train_ks"), trial.get("test_ks")
            # overfit gaps: prefer stored values, fall back to deriving train-test.
            gap_tt = trial.get("overfit_gap_tt")
            if gap_tt is None and isinstance(train_ks, (int, float)) and isinstance(test_ks, (int, float)):
                gap_tt = train_ks - test_ks
            rows.append([
                str(rank), _num(train_ks), _num(test_ks), _num(trial.get("oot_ks")),
                _num(trial.get("test_auc")), _num(trial.get("oot_auc")),
                _num(trial.get("lift_head_5")), _num(trial.get("lift_head_10")),
                _num(trial.get("lift_tail_5")), _num(trial.get("lift_tail_10")),
                _num(gap_tt), _num(trial.get("overfit_gap_to")),
            ])
        tables.append({
            "title": "trials 排行(按 in-time 选优;前15)",
            "columns": [
                "#", "train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc",
                "头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%",
                "过拟合gap(tt)", "过拟合gap(to)",
            ],
            "rows": rows,
        })
    if best_metrics:
        tables.append({"title": "最优 trial 指标", "columns": ["指标", "值"], "rows": [[k, _fmt(v)] for k, v in best_metrics.items()]})
    if best_params:
        tables.append({"title": "最优超参", "columns": ["参数", "值"], "rows": [[k, _fmt(v)] for k, v in best_params.items()]})
    return text, tables


def _render_train(o: dict):
    metrics = o.get("metrics") or {}
    text = "**训练完成**。"
    tables = []
    if metrics:
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
        if scalar:
            tables.append({"title": "模型指标", "columns": ["指标", "值"], "rows": [[k, _fmt(v)] for k, v in scalar.items()]})
    importance = o.get("feature_importance") or []
    rows = []
    for item in importance[:15]:
        if isinstance(item, (list, tuple)) and item:
            rows.append([str(item[0]), _fmt(item[1]) if len(item) > 1 else ""])
    if rows:
        tables.append({"title": "特征重要性(前15)", "columns": ["特征", "重要性"], "rows": rows})
    return text, tables


def _render_train_models(o: dict):
    experiments = [e for e in (o.get("experiments") or []) if isinstance(e, dict)]
    best_id = o.get("best_experiment_id")
    best_recipe = o.get("best_recipe")
    target_type = str(o.get("target_type") or "binary")
    tables = []
    rows = []
    best_metrics: dict = {}
    if target_type == "continuous":
        metric_columns = ["train_rmse", "test_rmse", "oot_rmse", "test_mae", "oot_mae", "test_r2", "oot_r2"]
        selector_label = "按 OOT RMSE"
    elif target_type == "multiclass":
        metric_columns = ["train_macro_auc", "test_macro_auc", "oot_macro_auc", "test_logloss", "oot_logloss", "test_accuracy", "oot_accuracy"]
        selector_label = "按 OOT macro-AUC"
    else:
        metric_columns = ["train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc"]
        selector_label = "按 OOT KS"
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
        text = f"**训练完成**:对比 {len(experiments)} 个算法,最优 **{best_recipe}**(★;{selector_label})。"
        tables.append({
            "title": "候选模型对比",
            "columns": ["算法", *metric_columns],
            "rows": rows,
        })
    else:
        text = "**训练完成**。"
    # the best model's full metrics (mirrors the single-model 模型指标 table)
    scalar = {k: v for k, v in best_metrics.items() if isinstance(v, (int, float, str, bool))}
    if scalar:
        tables.append({"title": "模型指标", "columns": ["指标", "值"], "rows": [[k, _fmt(v)] for k, v in scalar.items()]})
    return text, tables


def _render_compare(o: dict):
    experiments = o.get("experiments") or []
    rows = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        caps = exp.get("capabilities") or {}
        rows.append([
            exp.get("recipe") or "?",
            "是" if caps.get("pmml_supported") else "否",
            "是" if caps.get("handoff_supported") else "否",
            "是" if caps.get("native_model_supported") else "否",
            caps.get("reason") or "",
        ])
    tables = []
    if rows:
        tables.append({
            "title": "训练后动作能力",
            "columns": ["算法", "PMML", "移交验证", "原生模型", "说明"],
            "rows": rows,
        })
    return f"**实验对比完成**:共 {len(experiments)} 个实验候选。", tables


def _render_select_experiment(o: dict):
    selected = o.get("selected_experiment_id") or ""
    recipe = o.get("recipe") or "?"
    metric = o.get("selection_metric") or ""
    reason = o.get("selection_reason") or ""
    caps = o.get("capabilities") or {}
    text = f"**已选择最终实验**:`{selected}`({recipe});{reason}"
    rows = [
        ["PMML", "是" if caps.get("pmml_supported") else "否"],
        ["移交验证", "是" if caps.get("handoff_supported") else "否"],
        ["原生模型", "是" if caps.get("native_model_supported") else "否"],
    ]
    if caps.get("reason"):
        rows.append(["说明", caps.get("reason")])
    policy = o.get("policy_decision") if isinstance(o.get("policy_decision"), dict) else {}
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
    tables = [{
        "title": f"最终模型交付能力({metric})",
        "columns": ["能力", "状态"],
        "rows": rows,
    }]
    metrics = o.get("metrics") or {}
    if metrics:
        tables.append({
            "title": "最终模型指标",
            "columns": ["指标", "值"],
            "rows": [[key, _fmt(value)] for key, value in metrics.items()],
        })
    return text, tables


def _render_report(o: dict):
    path = o.get("report_path") or ""
    sections = [section for section in (o.get("section_status") or []) if isinstance(section, dict)]
    available = sum(1 for section in sections if section.get("available"))
    skipped = len(sections) - available
    text = (
        f"**模型开发报告已生成**:`{path}`"
        f"(业务章节 {available}/{len(sections)} 可生成"
        + (f", {skipped} 个缺输入/跳过" if skipped else "")
        + ",可在右栏下载)。"
    )
    tables = []
    if sections:
        tables.append({
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
        })
    return text, tables


def _num(value):
    return "n/a" if value is None else _fmt(value)


def _render_feature_metrics(o: dict):
    metrics = [metric for metric in (o.get("metrics") or []) if isinstance(metric, dict)]
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
        "(IV/KS/AUC 越高区分力越强;PSI/缺失率越低越稳)。可在右栏下载分析报告。"
    )
    tables = []
    if rows:
        tables.append({
            "title": "特征指标",
            "columns": columns,
            "rows": rows,
        })
    # Optional collinear / VIF section (computed only when the metric was selected).
    collinear = o.get("collinear")
    if isinstance(collinear, dict):
        vif = collinear.get("vif") or {}
        if vif:
            tables.append({
                "title": "VIF(共线性)",
                "columns": ["特征", "VIF"],
                "rows": [[str(feat), _num(value)] for feat, value in vif.items()],
            })
        pairs = [p for p in (collinear.get("collinear_pairs") or []) if isinstance(p, (list, tuple)) and len(p) >= 3]
        if pairs:
            tables.append({
                "title": "高相关特征对",
                "columns": ["特征A", "特征B", "相关系数"],
                "rows": [[str(p[0]), str(p[1]), _num(p[2])] for p in pairs],
            })
    return text, tables


def _render_feature_report(o: dict):
    # Reuse the metrics wide table (the tool echoes metrics) and append the report link.
    text, tables = _render_feature_metrics(o)
    path = o.get("report_path") or ""
    if path:
        text += f"\n\n**特征分析报告已生成**:`{path}`(可在右栏下载)。"
    return text, tables


def _render_build_strategy(o: dict):
    rules = [rule for rule in (o.get("rules") or []) if isinstance(rule, dict)]
    strategy_type = str(o.get("strategy_type") or "approval")
    default_decision = str(o.get("default_decision") or "")
    score_col = str(o.get("score_col") or "")
    text = (
        f"**策略候选已生成**:`{o.get('strategy_id', '')}`。"
        f"类型 `{strategy_type}`,评分列 `{score_col}`,默认动作 `{default_decision}`。"
    )
    tables = []
    if rules:
        tables.append({
            "title": "策略规则(按顺序命中)",
            "columns": ["#", "条件", "动作", "取值"],
            "rows": [
                [
                    str(index),
                    str(rule.get("condition", "")),
                    str(rule.get("decision", "")),
                    _fmt(rule.get("value")) if rule.get("value") is not None else "-",
                ]
                for index, rule in enumerate(rules, start=1)
            ],
        })
    return text, tables


def _render_backtest_strategy(o: dict):
    text = (
        "**策略回测完成**:"
        f"审批率 {_pct(o.get('approval_rate'))},"
        f"通过客群坏率 {_pct(o.get('approved_bad_rate'))},"
        f"拒绝客群坏率 {_pct(o.get('rejected_bad_rate'))},"
        f"预期利润 {_num(o.get('expected_profit'))}。"
    )
    rows = [
        ["审批率", _pct(o.get("approval_rate"))],
        ["通过人数", _fmt(o.get("approved_count"))],
        ["通过坏率", _pct(o.get("approved_bad_rate"))],
        ["拒绝坏率", _pct(o.get("rejected_bad_rate"))],
        ["预期利润", _num(o.get("expected_profit"))],
        ["swap-in", _fmt(o.get("swap_in_count"))],
        ["swap-out", _fmt(o.get("swap_out_count"))],
    ]
    tables = [{"title": "策略回测摘要", "columns": ["指标", "值"], "rows": rows}]
    by_segment = [row for row in (o.get("by_segment") or []) if isinstance(row, dict)]
    if by_segment:
        tables.append({
            "title": "按决策分组",
            "columns": ["决策", "样本数", "坏样本", "坏率"],
            "rows": [
                [
                    str(row.get("decision", "")),
                    _fmt(row.get("count")),
                    _fmt(row.get("bad_count")),
                    _pct(row.get("bad_rate")),
                ]
                for row in by_segment
            ],
        })
    return text, tables


def _render_tradeoff_view(o: dict):
    points = [point for point in (o.get("points") or []) if isinstance(point, dict)]
    recommended = o.get("recommended") if isinstance(o.get("recommended"), dict) else None
    if recommended:
        text = (
            "**策略权衡视图完成**:"
            f"推荐 cutoff `{_fmt(recommended.get('cutoff'))}`,"
            f"审批率 {_pct(recommended.get('approval_rate'))},"
            f"坏率 {_pct(recommended.get('bad_rate'))},"
            f"预期利润 {_num(recommended.get('expected_profit'))}。"
        )
    else:
        text = "**策略权衡视图完成**。"
    tables = []
    if points:
        tables.append({
            "title": "cutoff 权衡点",
            "columns": ["cutoff", "审批率", "坏率", "预期利润"],
            "rows": [
                [
                    _fmt(point.get("cutoff")),
                    _pct(point.get("approval_rate")),
                    _pct(point.get("bad_rate")),
                    _num(point.get("expected_profit")),
                ]
                for point in points[:20]
            ],
        })
    return text, tables


def _render_vintage_curve(o: dict):
    cohorts = [str(item) for item in (o.get("cohorts") or [])]
    mob_axis = list(o.get("mob_axis") or [])
    summary = o.get("summary") if isinstance(o.get("summary"), dict) else {}
    trend = str(summary.get("trend") or "stable")
    text = f"**Vintage 曲线完成**:{len(cohorts)} 个 cohort,趋势 `{trend}`。"
    tables = []
    curves = o.get("curves") if isinstance(o.get("curves"), dict) else {}
    counts = o.get("counts") if isinstance(o.get("counts"), dict) else {}
    if cohorts and mob_axis:
        tables.append({
            "title": "Vintage 累计坏账率",
            "columns": ["cohort", "样本数", *[f"MOB{mob}" for mob in mob_axis]],
            "rows": [
                [
                    cohort,
                    _fmt(counts.get(cohort, "")),
                    *[_pct(value) if value is not None else "n/a" for value in (curves.get(cohort) or [])[:len(mob_axis)]],
                ]
                for cohort in cohorts
            ],
        })
    at_ref = summary.get("at_ref") if isinstance(summary.get("at_ref"), dict) else {}
    if at_ref:
        tables.append({
            "title": "参考 MOB 坏账率",
            "columns": ["cohort", "坏账率"],
            "rows": [[str(cohort), _pct(value)] for cohort, value in at_ref.items()],
        })
    return text, tables


def _render_propose_join(o: dict):
    joins = o.get("joins") or []
    rows = []
    relax_rows = []
    any_conflict = False
    any_fp_mismatch = False
    for j in joins:
        diag = j.get("diagnostics") or {}
        match_rate = diag.get("match_rate")
        unique = diag.get("feature_key_unique")
        fan_out = diag.get("fan_out_detected", diag.get("fan_out"))
        # Prefer the friendly file name (features.parquet) over the raw ds_<hash> id.
        fname = str(j.get("feature_name") or j.get("feature_id", "?"))
        key_pairs = j.get("key_pairs") or []
        keys = ", ".join(f"{p.get('anchor_col')}={p.get('feature_col')}" for p in key_pairs) or "?"
        # Dynamic key relaxation proposals (spec §4/§5): low-match keys may match better with
        # one element dropped — surface as suggestions (the user confirms; never auto-applied).
        for alt in (diag.get("key_alternatives") or []):
            if not isinstance(alt, dict):
                continue
            alt_keys = ", ".join(f"{a}={f}" for a, f in (alt.get("key_pairs") or []))
            relax_rows.append([
                fname,
                _fmt(match_rate) if match_rate is not None else "n/a",
                f"减「{alt.get('dropped', '?')}」→ {alt_keys}",
                _fmt(alt.get("match_rate")) if alt.get("match_rate") is not None else "n/a",
                "是" if alt.get("feature_key_unique") else "否",
                "⚠️是" if alt.get("fan_out_detected") else "否",
            ])
        # Fingerprint consistency (spec §5 C2 "指纹 raw=md5? ✓/✗"): transform_side == "both"
        # means anchor and feature key share format (both raw or both md5); anything else
        # means one side is raw and the other md5 (键格式不一致), joinable only via a hash
        # transform — surfaced so the user can sanity-check the key before执行.
        fp_consistent = all((p.get("transform_side") or "both") == "both" for p in key_pairs) if key_pairs else True
        if not fp_consistent:
            any_fp_mismatch = True
        fp_cell = "✓" if fp_consistent else "✗ raw≠md5"
        # Two-level dedup breakdown (spec §6): safe whole-row dups vs same-key conflicts.
        report = diag.get("conflict_report") or {}
        conflict_keys = int(report.get("n_conflict_keys") or 0)
        safe_dropped = int(report.get("safe_dropped") or 0)
        if conflict_keys:
            any_conflict = True
        dedup_cell = "-" if unique else f"安全{safe_dropped}/⚠️冲突{conflict_keys}"
        rows.append([
            fname,
            keys,
            fp_cell,
            _fmt(match_rate) if match_rate is not None else "n/a",
            "是" if unique else "否",
            "⚠️是" if fan_out else "否",
            dedup_cell,
        ])
    text = (
        f"**拼接诊断完成**:{len(joins)} 张特征表待左连接到锚样本(锚行数 **1:1 保留**)。\n"
        "请核对每张表的命中率/键唯一性/是否膨胀。键不唯一的特征需选去重策略;确认后才会真正执行拼接。"
    )
    if any_conflict:
        text += (
            "\n\n⚠️ 检测到**同键值冲突**(同一键多行但特征值不一致):这类**不会自动删除**,"
            "请先确认去重策略或清洗数据后再拼接。"
        )
    if any_fp_mismatch:
        text += (
            "\n\n⚠️ 检测到**键指纹不一致**(`✗ raw≠md5`:锚/特征侧一为原文、一为 md5):"
            "系统会自动对齐哈希后再连接,但请确认这是同一标识(避免误配)。"
        )
    if relax_rows:
        text += (
            "\n\n💡 部分特征表命中率偏低,**减一个识别要素**可提高命中(见下「择键建议」):"
            "系统只提议、不会自动改键;若减后**膨胀**需配合去重策略。请确认后再选用。"
        )
    tables = []
    if rows:
        tables.append({
            "title": "拼接诊断(逐特征表)",
            "columns": ["特征表", "匹配键", "指纹(raw=md5?)", "命中率", "键唯一", "膨胀", "去重(安全/冲突键)"],
            "rows": rows,
        })
    if relax_rows:
        tables.append({
            "title": "择键建议(减要素换更高命中)",
            "columns": ["特征表", "当前命中率", "建议键", "减后命中率", "减后唯一", "减后膨胀"],
            "rows": relax_rows,
        })
    return text, tables


def _render_confirm_join(o: dict):
    # Internal plumbing step (marks engine specs confirmed). It is a dependency of
    # the execute_join gate, but its summary would show "已确认…" before the human
    # actually confirms, which is confusing — so render nothing at the gate…
    needs = o.get("needs_dedup") or []
    if needs:
        # …UNLESS a feature has a same-key conflict (spec §6): surface it so the user knows
        # the join can't execute until they pick a dedup strategy (or exclude the feature).
        labels = o.get("needs_dedup_labels") or {}
        listed = "、".join(f"`{labels.get(f, f)}`" for f in needs)
        return (
            f"⚠️ 特征 {listed} 存在**同键冲突**(同一键多行、特征值不一致),"
            "需先定去重策略才能拼接。回复「去重 first」(保留首条)或「去重 last」(保留末条)解决;"
            "或排除这些特征后重试。"
        ), []
    return "", []


def _render_execute_join(o: dict):
    anchor_rows = o.get("anchor_rows")
    joined_rows = o.get("joined_rows")
    ok = anchor_rows == joined_rows
    text = (
        f"**拼接执行完成**:结果数据集 `{o.get('result_dataset_id', '')}`,"
        f"锚行 {anchor_rows} → 拼接后 {joined_rows} 行"
        + ("(1:1 保持 ✓)" if ok else "(⚠️ 行数发生变化,请检查膨胀)")
    )
    warnings = o.get("warnings") or []
    if warnings:
        text += "\n警告:" + "; ".join(str(w) for w in warnings)
    # §8 per-table contribution summary from real diagnostics.
    tables = []
    per_table = [row for row in (o.get("per_table") or []) if isinstance(row, dict)]
    if per_table:
        tables.append({
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
        })
    return text, tables


def _render_post_training_action(o: dict):
    actions = [item for item in (o.get("actions") or []) if isinstance(item, dict)]
    succeeded = sum(1 for item in actions if item.get("status") == "succeeded")
    skipped = sum(1 for item in actions if item.get("status") == "skipped")
    text = (
        f"**训练后交付动作完成**:成功 {succeeded} 个,跳过 {skipped} 个。"
        if actions else "**训练后交付动作完成**。"
    )
    rows = [
        ["原生模型", "succeeded" if o.get("native_model_path") else "missing", o.get("native_model_path") or "", ""],
    ]
    if o.get("approval_package_path"):
        rows.append([
            "审批包",
            "succeeded",
            o.get("approval_package_markdown_path") or o.get("approval_package_path"),
            "模型审批与交付证据包",
        ])
    if o.get("model_card_path"):
        rows.append([
            "模型卡",
            "succeeded",
            o.get("model_card_markdown_path") or o.get("model_card_path"),
            "最终模型卡",
        ])
    if o.get("monitoring_policy_path"):
        monitoring = o.get("monitoring_policy") if isinstance(o.get("monitoring_policy"), dict) else {}
        rows.append([
            "监控策略",
            monitoring.get("status") or "succeeded",
            o.get("monitoring_policy_markdown_path") or o.get("monitoring_policy_path"),
            monitoring.get("recommendation") or "模型监控阈值策略",
        ])
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
    tables = [{
        "title": "训练后交付状态",
        "columns": ["动作", "状态", "产物/任务", "说明"],
        "rows": rows,
    }]
    caps = o.get("capabilities") or {}
    if caps:
        cap_rows = [
            ["PMML", "是" if caps.get("pmml_supported") else "否"],
            ["移交验证", "是" if caps.get("handoff_supported") else "否"],
            ["原生模型", "是" if caps.get("native_model_supported") else "否"],
        ]
        if caps.get("reason"):
            cap_rows.append(["说明", caps.get("reason")])
        tables.append({"title": "最终模型交付能力", "columns": ["能力", "状态"], "rows": cap_rows})
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
        "(占比是否合理、OOT 是否按时间、分组是否防泄漏)后再继续。"
    )
    tables = []
    if rows:
        tables.append({
            "title": "切分计数(train/test/oot)",
            "columns": ["划分", "行数", "占比"],
            "rows": rows,
        })
    for group_col, dist in (analysis.get("group_distributions") or {}).items():
        if not isinstance(dist, dict):
            continue
        group_values = sorted({gv for per in dist.values() if isinstance(per, dict) for gv in per})
        grows = [
            [str(split)] + [int(per.get(gv, 0)) for gv in group_values]
            for split, per in dist.items() if isinstance(per, dict)
        ]
        if grows:
            tables.append({
                "title": f"按「{group_col}」分布(逐划分)",
                "columns": ["划分", *[str(gv) for gv in group_values]],
                "rows": grows,
            })
    return text, tables


_RENDERERS = {
    "make_split": _render_make_split,
    "choose_modeling_spec": _render_choose_modeling_spec,
    "screen_features": _render_screen,
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
    "backtest_strategy": _render_backtest_strategy,
    "tradeoff_view": _render_tradeoff_view,
    "vintage_curve": _render_vintage_curve,
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

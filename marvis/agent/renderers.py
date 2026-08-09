"""Tool-output renderers for V2 plan-driver messages.

Each renderer turns a Tool's raw output into ``(markdown_text, table_blocks)``.
This facade keeps cross-domain dispatch outside ``plan_driver.py``; large
domain families live in ``marvis.agent.presenters`` and register here without
moving presentation rules back into the execution loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from marvis.agent.presenters.modeling_evidence import (
    MODELING_EVIDENCE_PRESENTERS,
)
from marvis.agent.presenters.runtime import build_trusted_presenter_runtime
from marvis.agent.presenters.strategy_exploration import (
    STRATEGY_EXPLORATION_PRESENTERS,
)
from marvis.agent.presenters._shared import (
    MONITOR_LEVEL_LABEL as _MONITOR_LEVEL_LABEL,
    format_number as _num,
    format_percent as _pct,
    format_value as _fmt,
)
from marvis.agent.presenters import strategy as _strategy_presenters
from marvis.agent.presenters.strategy import (
    STRATEGY_RENDERERS,
    strategy_integrity_failure,
)
from marvis.orchestrator.evidence import artifact_bindings, artifact_refs


_TRUSTED_RUNTIME_PRESENTERS = {
    **MODELING_EVIDENCE_PRESENTERS,
    **STRATEGY_EXPLORATION_PRESENTERS,
}
if len(_TRUSTED_RUNTIME_PRESENTERS) != (
    len(MODELING_EVIDENCE_PRESENTERS) + len(STRATEGY_EXPLORATION_PRESENTERS)
):
    raise RuntimeError("duplicate authenticated runtime presenter refs")


# ---------------------------------------------------------------------------
# tool -> table registry (decision #4 in the driver spec)
# Each renderer turns a tool's raw output into (markdown text, [table dicts]).
# Task differences land HERE; the driver loop above stays task-agnostic. A table
# dict is {title, columns, rows} — the frontend maps it onto renderMetricTableSection.
# ---------------------------------------------------------------------------
def _names(items) -> list[str]:
    out = []
    for item in items or []:
        out.append(item[0] if isinstance(item, (list, tuple)) and item else item)
    return [str(x) for x in out]


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


def _render_reports(o: dict):
    reports = [item for item in (o.get("reports") or []) if isinstance(item, dict)]
    primary_path = str(o.get("report_path") or "")
    generated = [item for item in reports if str(item.get("report_path") or "").strip()]
    text = (
        f"**模型开发报告已生成**：共 {len(generated)} 份"
        + (f"，主报告 `{primary_path}`" if primary_path else "")
        + "。请使用本条结果下方的“下载模型开发报告”按钮。"
    )
    tables = []
    if reports:
        tables.append(
            {
                "title": "候选模型报告",
                "columns": ["实验", "算法", "状态", "报告"],
                "rows": [
                    [
                        str(item.get("experiment_id") or ""),
                        str(item.get("recipe") or ""),
                        "已生成"
                        if str(item.get("report_path") or "").strip()
                        else "缺失",
                        str(item.get("report_path") or ""),
                    ]
                    for item in reports
                ],
            }
        )
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


def _render_feature_metrics(o: dict):
    metrics = [
        metric for metric in (o.get("metrics") or []) if isinstance(metric, dict)
    ]
    # FEATURE §2: every metric is independently selectable. A missing key means
    # "not selected", while a present key with ``None`` means "selected but not
    # computable" and must retain its structured reason.
    has_iv = any("iv" in metric for metric in metrics)
    has_ks = any("ks" in metric for metric in metrics)
    has_auc = any("auc" in metric for metric in metrics)
    has_head_tail = any("lift_head_5" in metric for metric in metrics)
    has_importance = any("importance" in metric for metric in metrics)
    has_quality = any(
        any(
            key in metric
            for key in (
                "coverage",
                "valid_count",
                "missing_rate",
                "mode_rate",
                "zero_rate",
                "unique_count",
                "unique_rate",
            )
        )
        for metric in metrics
    )
    has_distribution = any(
        any(
            key in metric
            for key in ("mean", "std", "median", "min", "q25", "q75", "max")
        )
        for metric in metrics
    )
    has_meaning = any(
        any(
            key in metric
            for key in (
                "business_meaning",
                "expected_direction",
                "actual_direction",
                "meaning_consistency",
                "meaning_consistency_reason",
            )
        )
        for metric in metrics
    )
    columns = ["特征"]
    metric_keys: list[str] = []
    if has_iv:
        columns.append("IV")
        metric_keys.append("iv")
    if has_ks:
        columns.append("KS")
        metric_keys.append("ks")
    if has_auc:
        columns.append("AUC")
        metric_keys.append("auc")
    if has_importance:
        columns.append("重要性")
        metric_keys.append("importance")
    rows = [
        [
            str(metric.get("feature", "?")),
            *[_num(metric.get(key)) for key in metric_keys],
        ]
        for metric in metrics
    ]
    selected_labels = [
        label
        for key, label in (
            ("iv", "IV"),
            ("ks", "KS"),
            ("auc", "AUC"),
            ("coverage", "覆盖/质量"),
            ("psi", "PSI"),
            ("psi_month_first", "月度PSI(首月基准)"),
            ("psi_month_last", "月度PSI(末月基准)"),
            ("psi_month_previous", "月度PSI(逐月环比)"),
            ("psi_split", "样本集PSI"),
            ("vif", "VIF"),
            ("head_tail_lift", "头尾Lift"),
            ("importance", "重要性"),
            ("meaning_consistency", "含义方向一致性"),
        )
        if key in set(o.get("selected_metrics") or [])
    ]
    selected_text = "、".join(selected_labels) if selected_labels else "本次返回的指标"
    text = (
        f"**特征分析完成**:{len(rows)} 个特征，已计算 {selected_text}，并生成 Agent 建议。"
        "完整指标可通过本条结果下方的按钮下载。"
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
        tables.append(
            {
                "title": "Agent 特征建议",
                "columns": [
                    "特征",
                    "Agent建议",
                    "推荐原因",
                    "建议状态",
                    "证据置信度",
                    "支持指标",
                ],
                "rows": [
                    [
                        str(metric.get("feature", "?")),
                        str(metric.get("recommendation") or "待评估"),
                        str(metric.get("recommendation_reason") or "-"),
                        str(metric.get("recommendation_state") or "unevaluated"),
                        str(metric.get("recommendation_confidence") or "none"),
                        "；".join(
                            f"{item.get('metric')}={item.get('value')}"
                            for item in (metric.get("recommendation_evidence") or [])
                            if isinstance(item, dict) and item.get("metric")
                        )
                        or "-",
                    ]
                    for metric in metrics
                ],
            }
        )
        psi_views = [
            ("psi", "PSI"),
            ("psi_month_first", "月度PSI(首月基准)"),
            ("psi_month_last", "月度PSI(末月基准)"),
            ("psi_month_previous", "月度PSI(逐月环比)"),
            ("psi_split", "样本集PSI"),
        ]
        selected_psi_views = [
            (key, label)
            for key, label in psi_views
            if any(key in metric for metric in metrics)
        ]
        if selected_psi_views:
            psi_columns = ["特征"]
            for _key, label in selected_psi_views:
                psi_columns.extend([label, f"{label}说明"])
            tables.append(
                {
                    "title": "PSI 稳定性",
                    "columns": psi_columns,
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            *[
                                value
                                for key, _label in selected_psi_views
                                for value in (
                                    _num(metric.get(key)),
                                    _feature_reason_text(
                                        metric.get(f"{key}_reason"),
                                        fallback="已完成计算"
                                        if metric.get(key) is not None
                                        else "未提供原因",
                                    ),
                                )
                            ],
                        ]
                        for metric in metrics
                    ],
                }
            )
            psi_detail_rows = []
            for metric in metrics:
                for key, label in selected_psi_views:
                    series = metric.get(f"{key}_series")
                    if not isinstance(series, list):
                        continue
                    for point in series:
                        if not isinstance(point, dict):
                            continue
                        psi_detail_rows.append(
                            [
                                str(metric.get("feature", "?")),
                                label,
                                str(point.get("base") or ""),
                                str(point.get("compare") or ""),
                                _num(point.get("psi")),
                            ]
                        )
            if psi_detail_rows:
                tables.append(
                    {
                        "title": "PSI 明细",
                        "columns": ["特征", "口径", "基准", "对比", "PSI"],
                        "rows": psi_detail_rows,
                    }
                )
        if has_quality:
            tables.append(
                {
                    "title": "数据质量",
                    "columns": [
                        "特征",
                        "有效样本",
                        "覆盖率",
                        "缺失率",
                        "单一值率",
                        "零值率",
                    ],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            str(int(metric.get("valid_count") or 0)),
                            _num(metric.get("coverage")),
                            _num(metric.get("missing_rate")),
                            _num(metric.get("mode_rate")),
                            _num(metric.get("zero_rate")),
                        ]
                        for metric in metrics
                    ],
                }
            )
            tables.append(
                {
                    "title": "唯一值检查",
                    "columns": ["特征", "唯一值数", "唯一值率"],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            str(int(metric.get("unique_count") or 0)),
                            _num(metric.get("unique_rate")),
                        ]
                        for metric in metrics
                    ],
                }
            )
        if has_head_tail:
            tables.append(
                {
                    "title": "头尾 Lift（风险方向）",
                    "columns": [
                        "特征",
                        "头部lift5%",
                        "头部lift10%",
                        "尾部lift5%",
                        "尾部lift10%",
                    ],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            _num(metric.get("lift_head_5")),
                            _num(metric.get("lift_head_10")),
                            _num(metric.get("lift_tail_5")),
                            _num(metric.get("lift_tail_10")),
                        ]
                        for metric in metrics
                    ],
                }
            )
            tables.append(
                {
                    "title": "Lift 计算说明",
                    "columns": ["特征", "说明"],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            str(metric.get("lift_reason") or "已完成计算"),
                        ]
                        for metric in metrics
                    ],
                }
            )
        if has_distribution:
            tables.append(
                {
                    "title": "分布统计（中心）",
                    "columns": ["特征", "均值", "标准差", "中位数"],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            _num(metric.get("mean")),
                            _num(metric.get("std")),
                            _num(metric.get("median")),
                        ]
                        for metric in metrics
                    ],
                }
            )
            tables.append(
                {
                    "title": "分布统计（范围）",
                    "columns": ["特征", "最小值", "P25", "P75", "最大值"],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            _num(metric.get("min")),
                            _num(metric.get("q25")),
                            _num(metric.get("q75")),
                            _num(metric.get("max")),
                        ]
                        for metric in metrics
                    ],
                }
            )
        if has_meaning:
            tables.append(
                {
                    "title": "含义方向一致性",
                    "columns": [
                        "特征",
                        "业务含义",
                        "预期方向",
                        "实际方向",
                        "结论",
                        "说明",
                    ],
                    "rows": [
                        [
                            str(metric.get("feature", "?")),
                            str(metric.get("business_meaning") or "-"),
                            str(metric.get("expected_direction") or "-"),
                            str(metric.get("actual_direction") or "-"),
                            str(metric.get("meaning_consistency") or "n/a"),
                            _feature_reason_text(
                                metric.get("meaning_consistency_reason"),
                                fallback="已完成核对"
                                if metric.get("meaning_consistency")
                                else "未提供原因",
                            ),
                        ]
                        for metric in metrics
                    ],
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


def _feature_reason_text(value, *, fallback: str) -> str:
    if isinstance(value, dict):
        return str(value.get("message") or value.get("code") or fallback)
    if value not in (None, ""):
        return str(value)
    return fallback


def _render_feature_report(o: dict):
    # Reuse the metrics wide table (the tool echoes metrics) and append the report link.
    text, tables = _render_feature_metrics(o)
    bin_text, bin_tables = _render_feature_bins(o)
    if bin_text:
        text += f"\n\n{bin_text}"
    tables.extend(bin_tables)
    path = o.get("report_path") or ""
    if path:
        text += f"\n\n**特征分析报告已生成**:`{path}`。请使用本条结果下方的“下载特征分析报告”按钮。"
    return text, tables


def _render_feature_bins(o: dict):
    analyses = [item for item in (o.get("binning") or []) if isinstance(item, dict)]
    if not analyses:
        return "已跳过可选分箱分析。", []
    rows = []
    for analysis in analyses:
        feature = str(analysis.get("feature") or "")
        requested = int(analysis.get("requested_bins") or 0)
        actual = int(analysis.get("actual_bins") or 0)
        for item in analysis.get("rows") or []:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    feature,
                    requested,
                    int(item.get("bin_index") or 0),
                    int(item.get("risk_rank") or 0),
                    str(item.get("interval") or ""),
                    int(item.get("count") or 0),
                    int(item.get("bad_count") or 0),
                    int(item.get("good_count") or 0),
                    _num(item.get("bad_rate")),
                    _num(item.get("cumulative_bad_rate")),
                    _num(item.get("lift")),
                    _num(item.get("cumulative_lift")),
                    _num(item.get("ks")),
                    _num(item.get("woe")),
                    _num(item.get("iv_contribution")),
                    actual,
                    str(analysis.get("direction") or "unknown"),
                    str(analysis.get("degraded_reason") or ""),
                ]
            )
    table = {
        "title": "分箱分析",
        "columns": [
            "特征",
            "请求箱数",
            "箱号",
            "风险排序",
            "区间",
            "样本数",
            "坏样本数",
            "好样本数",
            "坏率",
            "累计坏率",
            "单箱Lift",
            "累计Lift",
            "KS",
            "WOE",
            "IV贡献",
            "实际箱数",
            "风险方向",
            "分箱说明",
        ],
        "rows": rows,
    }
    return f"**分箱分析完成**：已分析 {len(analyses)} 个用户选择的特征。", [table]


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
    has_key_alternatives = False
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
            has_key_alternatives = True
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
    if has_key_alternatives:
        text += (
            "\n\n💡 检测到拼接键存在多种候选。下方按**每张特征表**分别展示可选键；"
            "选择后系统会回到当前步骤重新计算命中率、唯一性和膨胀，不会直接执行拼接。"
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
                _integer_text(primary) if primary is not None else "n/a",
                _integer_text(secondary) if secondary is not None else "n/a",
                verdict,
                _short_digest(prov.get("dataset_fingerprint")) if prov else "—",
                str(prov.get("code_version") or "—") if prov else "—",
                _short_digest(prov.get("params_digest")) if prov else "—",
                str(prov.get("seed")) if prov and prov.get("seed") is not None else "—",
            ]
        )
    return rows


def _integer_text(value) -> str:
    """Counts are discrete even when a computation backend returns float scalars."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return str(value)


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


def _render_make_split(o: dict, *, presentation_state: str = "preview"):
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
    if presentation_state == "adopted":
        text = (
            f"**已采用的样本切分**：共 {total} 行；"
            "后续特征筛选与训练沿用以下 Train/Test/OOT 方案。"
        )
    else:
        text = (
            f"**样本切分预览已生成**:共 {total} 行，尚未进入特征筛选或训练。"
            "请在下方选择是否需要 OOT、切分方式和比例；确认后才会采用该方案继续。"
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


def _render_risk_analysis_report(o: dict):
    kind_label = {
        "vtg_terminal": "VTG终值与年化不良",
        "profitability": "收益测算",
    }.get(str(o.get("analysis_kind") or ""), "风险分析")
    key_points = [str(item) for item in (o.get("key_points") or [])]
    red_flags = [str(item) for item in (o.get("red_flags") or [])]
    assumptions = [str(item) for item in (o.get("assumptions") or [])]
    metrics = (
        o.get("headline_metrics") if isinstance(o.get("headline_metrics"), dict) else {}
    )
    source_row_count = o.get("source_row_count", o.get("row_count", 0))
    text = (
        f"**{kind_label}报告已生成**：源数据 {_fmt(source_row_count)} 行，"
        f"形成 {_fmt(o.get('row_count', 0))} 行结果，"
        "可通过下方“下载报告”查看完整明细、口径与数据质量检查。"
    )
    if key_points:
        text += "\n\n**重要发现**\n" + "\n".join(f"- {item}" for item in key_points)
    if red_flags:
        text += "\n\n**需重点关注**\n" + "\n".join(f"- {item}" for item in red_flags)
    tables = []
    if metrics:
        metric_labels = {
            "annualized_bad_rate": "组合年化不良率",
            "weighted_terminal_bad_rate": "加权VTG终值",
            "portfolio_turnover": "组合周转次数",
            "observed_annualized_bad_rate": "MOB14观察年化不良率",
            "lowest_net_yield": "最低产品净收益率",
            "highest_net_yield": "最高产品净收益率",
            "negative_product_count": "负净收益产品数",
            "max_cost_rate": "最大成本率",
            "largest_scenario_net_yield_spread": "最大场景净收益差",
        }
        tables.append(
            {
                "title": "核心指标",
                "columns": ["指标", "值"],
                "rows": [
                    [
                        metric_labels.get(str(name), str(name)),
                        _pct(value)
                        if str(name).endswith(("_rate", "_yield", "_spread"))
                        else _fmt(value),
                    ]
                    for name, value in metrics.items()
                ],
            }
        )
    if assumptions:
        tables.append(
            {
                "title": "计算口径与假设",
                "columns": ["#", "口径/假设"],
                "rows": [
                    [str(index), item]
                    for index, item in enumerate(assumptions, start=1)
                ],
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
    "generate_model_reports": _render_reports,
    "propose_join": _render_propose_join,
    "confirm_join": _render_confirm_join,
    "execute_join": _render_execute_join,
    "compute_feature_metrics": _render_feature_metrics,
    "analyze_feature_bins": _render_feature_bins,
    "generate_feature_report": _render_feature_report,
    "generate_risk_analysis_report": _render_risk_analysis_report,
    "vintage_curve": _render_vintage_curve,
    "slice_aggregate": _render_slice_aggregate,
    "profile_dataset": _render_profile_dataset,
    "transform_dataset": _render_transform_dataset,
    "export_dataset": _render_export_dataset,
    "score_dataset": _render_score_dataset,
    "monitor_run": _render_monitor_run,
    "flow_rate": _render_flow_rate,
    "bucket_migration": _render_bucket_migration,
    "segment_profile": _render_segment_profile,
    "expected_loss_estimate": _render_el_estimate,
    "portfolio_gate_summary": _render_portfolio_gate_summary,
    "portfolio_report": _render_portfolio_report,
}


def _render_generic(o: dict):
    if not isinstance(o, dict) or not o:
        return "已完成。", []
    scalar = {k: v for k, v in o.items() if isinstance(v, (str, int, float, bool))}
    if scalar:
        head = ", ".join(f"{k}={_fmt(v)}" for k, v in list(scalar.items())[:6])
        return f"已完成:{head}", []
    return "已完成。", []


def _canonical_presenter_integrity_failure() -> tuple[str, list[dict]]:
    return (
        "**结果完整性校验失败**：平台无法从当前任务的实时不可变证据认证该 Tool 输出，"
        "已停止展示；不会把缓存字段降级为普通成功结果。",
        [],
    )


def _trusted_step_evidence_matches(
    tool: str,
    output: object,
    *,
    trusted_output_ref: str | None,
    trusted_step_evidence: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(trusted_step_evidence, Mapping):
        return False
    if (
        not isinstance(trusted_output_ref, str)
        or not trusted_output_ref
        or trusted_step_evidence.get("output_ref") != trusted_output_ref
        or trusted_step_evidence.get("renderer_hint") != tool
    ):
        return False
    run_id = trusted_step_evidence.get("step_run_id")
    input_hash = trusted_step_evidence.get("input_hash")
    if not isinstance(run_id, str) or not run_id:
        return False
    if (
        not isinstance(input_hash, str)
        or not input_hash.startswith("sha256:")
        or len(input_hash) != 71
        or any(
            character not in "0123456789abcdef"
            for character in input_hash.removeprefix("sha256:")
        )
    ):
        return False
    recorded_refs = trusted_step_evidence.get("artifact_refs")
    if not isinstance(recorded_refs, list) or not all(
        isinstance(ref, str) and ref for ref in recorded_refs
    ):
        return False
    live_refs = artifact_refs(output)
    recorded_bindings = trusted_step_evidence.get("artifact_bindings")
    live_bindings = artifact_bindings(output)
    return (
        bool(live_refs)
        and live_refs == recorded_refs
        and isinstance(recorded_bindings, list)
        and bool(live_bindings)
        and live_bindings == recorded_bindings
    )


def has_tool_presenter(tool: str) -> bool:
    """Return whether a Tool has an explicit domain or facade presenter."""

    return (
        tool in _TRUSTED_RUNTIME_PRESENTERS
        or tool in STRATEGY_RENDERERS
        or tool in _RENDERERS
    )


def render_tool_output(
    tool: str,
    output: dict,
    *,
    trusted_task_id: str | None = None,
    trusted_workspace: Path | str | None = None,
    trusted_output_ref: str | None = None,
    trusted_step_evidence: Mapping[str, Any] | None = None,
    trusted_inputs: Mapping[str, Any] | None = None,
    trusted_artifacts: Mapping[str, Any] | None = None,
    presentation_state: str | None = None,
):
    """Render a tool output, with optional gate-specific presentation context."""
    authenticated_presenter = _TRUSTED_RUNTIME_PRESENTERS.get(tool)
    if authenticated_presenter is not None:
        try:
            if not _trusted_step_evidence_matches(
                tool,
                output,
                trusted_output_ref=trusted_output_ref,
                trusted_step_evidence=trusted_step_evidence,
            ):
                return _canonical_presenter_integrity_failure()
            runtime = build_trusted_presenter_runtime(
                tool,
                workspace=trusted_workspace,
                task_id=trusted_task_id,
            )
            if runtime is None:
                return _canonical_presenter_integrity_failure()
            return authenticated_presenter(
                output or {},
                runtime=runtime,
                task_id=trusted_task_id,
                trusted_inputs=trusted_inputs,
                trusted_artifacts=trusted_artifacts,
            )
        except Exception:
            return _canonical_presenter_integrity_failure()
    if tool == "make_split":
        try:
            return _render_make_split(
                output or {},
                presentation_state=presentation_state or "preview",
            )
        except Exception:
            return _render_generic(output or {})
    renderer = STRATEGY_RENDERERS.get(tool) or _RENDERERS.get(
        tool,
        _render_generic,
    )
    try:
        if tool == "build_voting_candidate_from_search":
            return renderer(
                output or {},
                trusted_inputs=trusted_inputs,
            )
        if tool in {
            "export_strategy_delivery",
            "measure_candidate_monthly_stability",
            "measure_strategy_pool_validation",
            "measure_strategy_pool_stability",
        }:
            return renderer(
                output or {},
                trusted_task_id=trusted_task_id,
                trusted_inputs=trusted_inputs,
                trusted_artifacts=trusted_artifacts,
            )
        return renderer(output or {})
    except Exception:
        integrity_failure = strategy_integrity_failure(tool)
        if integrity_failure is not None:
            return integrity_failure
        return _render_generic(output or {})


def __getattr__(name: str):
    """Keep historical private presenter imports working during migration."""

    try:
        return getattr(_strategy_presenters, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


__all__ = ["has_tool_presenter", "render_tool_output"]

"""Strategy lifecycle, analytics, monitoring, and rule presenters."""

from __future__ import annotations


from marvis.agent.presenters._shared import (
    MONITOR_LEVEL_LABEL as _MONITOR_LEVEL_LABEL,
    format_number as _num,
    format_percent as _pct,
    format_value as _fmt,
    red_flag_table as _red_flag_table,
)

_STRATEGY_DECISION_LABEL = {
    "approve": "通过",
    "review": "复核",
    "decline": "拒绝",
}

_MONITORING_RED_CHECKLIST = (
    "处置建议(红灯,请三选一并回复关键词):",
    "1. 维持并观察 —— 回复「观察」保持当前策略,加强下一周期监控;",
    "2. 调阈值重跑 —— 回复「调阈值」调整监控计划阈值后重新运行监控;",
    "3. 起新版本策略 —— 回复「起新版本」基于当前策略起一个新版本重走策略开发。",
)


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


LIFECYCLE_PRESENTERS = {
    "build_strategy": _render_build_strategy,
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
    "run_strategy_monitoring": _render_run_strategy_monitoring,
    "apply_monitoring_disposition": _render_apply_monitoring_disposition,
    "render_monitoring_report": _render_monitoring_report,
    "evaluate_rule_set": _render_evaluate_rule_set,
    "mine_rules": _render_mine_rules,
    "select_rule_set": _render_select_rule_set,
}

INTEGRITY_FAILURES = {}

__all__ = ["LIFECYCLE_PRESENTERS", "INTEGRITY_FAILURES"]

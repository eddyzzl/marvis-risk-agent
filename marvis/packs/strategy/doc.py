from __future__ import annotations


# The strategy doc never recomputes metrics (INV-1): every number here comes from
# the persisted strategy / backtests / band stats passed in, formatted for a
# Chinese-language markdown deliverable.
_STATUS_LABEL = {"draft": "草稿", "adopted": "已采纳", "retired": "已退役"}


def render_strategy_doc_markdown(
    *,
    strategy: dict,
    meta: dict,
    backtests: list[dict],
    artifacts: list[dict],
    band_stats: list[dict],
    red_flags: list[dict] | None = None,
) -> tuple[str, list[str]]:
    strategy_type = str(strategy.get("strategy_type") or "")
    decision_strategy = strategy_type in {"approval", "reject"}
    distribution_section = "分数带" if decision_strategy else "类型化分布"
    sections = [
        "策略概览",
        "规则清单",
        "回测摘要",
        distribution_section,
        "红旗与处置记录",
        "监控计划摘要",
    ]
    lines: list[str] = []
    strategy_id = str(strategy.get("id", ""))
    lines.append(f"# 策略文档 · {strategy_id}")
    lines.append("")

    # 1. Overview
    lines.append("## 策略概览")
    version = meta.get("version", 1)
    status = _STATUS_LABEL.get(str(meta.get("status", "draft")), str(meta.get("status", "draft")))
    parent = meta.get("parent_strategy_id")
    lines.append(f"- 类型：{strategy.get('strategy_type', '')}")
    lines.append(f"- 版本：v{version}")
    lines.append(f"- 状态：{status}")
    lines.append(f"- 谱系父策略：{parent if parent else '无'}")
    if meta.get("adopted_at"):
        lines.append(f"- 采纳时间：{meta.get('adopted_at')}")
    if meta.get("adoption_reason"):
        lines.append(f"- 采纳理由：{meta.get('adoption_reason')}")
    lines.append("")

    # 2. Rules
    lines.append("## 规则清单")
    lines.append("| # | 条件 | 决策 | 取值 |")
    lines.append("| --- | --- | --- | --- |")
    for index, rule in enumerate(strategy.get("rules") or [], start=1):
        value = rule.get("value")
        lines.append(
            f"| {index} | {rule.get('condition', '')} | {rule.get('decision', '')} | "
            f"{'-' if value is None else value} |"
        )
    lines.append(f"| - | 默认动作 | {strategy.get('default_decision', '')} | - |")
    lines.append("")

    # 3. Backtest summary (incl. swap)
    lines.append("## 回测摘要")
    if backtests:
        lines.extend(_backtest_summary_lines(backtests[-1]))
    else:
        lines.append("- 暂无回测结果。")
    lines.append("")

    # 4. Bands
    lines.append(f"## {distribution_section}")
    if not decision_strategy:
        lines.append("- 类型化分布与风险指标已在回测摘要中列示；本类型不使用分数带口径。")
    elif band_stats:
        lines.append("| band 区间 | 样本占比 | 坏率 | 累计审批率 | 累计坏率 | 决策 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for band in band_stats:
            lines.append(
                f"| [{_g(band.get('lo'))},{_g(band.get('hi'))}) | {_pct(band.get('pop_pct'))} | "
                f"{_pct(band.get('bad_rate'))} | {_pct(band.get('cum_approval_rate'))} | "
                f"{_pct(band.get('cum_bad_rate'))} | {band.get('decision', '')} |"
            )
    else:
        lines.append("- 未提供分数带统计。")
    lines.append("")

    # 5. Red flags
    lines.append("## 红旗与处置记录")
    flags = _effective_red_flags(red_flags, backtests)
    if flags:
        lines.append("| 等级 | code | 说明 |")
        lines.append("| --- | --- | --- |")
        for flag in flags:
            lines.append(
                f"| {flag.get('level', '')} | {flag.get('code', '')} | {flag.get('message', '')} |"
            )
    else:
        lines.append("- 无红旗记录。")
    lines.append("")

    # 6. Monitoring plan summary
    lines.append("## 监控计划摘要")
    monitoring = [a for a in artifacts if a.get("kind") == "monitoring_plan_json"]
    if monitoring:
        lines.append(f"- 监控计划已登记：{monitoring[-1].get('path', '')}")
        if decision_strategy:
            lines.append("- 监控指标：通过客群坏率漂移、审批率下滑（S5 闭环消费）。")
        else:
            lines.append("- 监控指标：按该策略类型的结构化回测指标执行。")
    else:
        lines.append("- 尚未生成监控计划。")
    lines.append("")

    return "\n".join(lines), sections


def _effective_red_flags(
    supplied: list[dict] | None,
    backtests: list[dict],
) -> list[dict]:
    flags = [dict(flag) for flag in (supplied or []) if isinstance(flag, dict)]
    latest = backtests[-1] if backtests else {}
    economics = latest.get("economics") if isinstance(latest, dict) else None
    profit_note = (
        economics.get("profit_note")
        if isinstance(economics, dict)
        else latest.get("profit_note")
    )
    if profit_note and not any(
        flag.get("code") == "expected_profit_unavailable" for flag in flags
    ):
        flags.append(
            {
                "level": "amber",
                "code": "expected_profit_unavailable",
                "message": str(profit_note),
            }
        )
    return flags


def _g(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _typed_backtest_view(
    payload: dict,
) -> tuple[str, dict, list[dict], list[dict], dict, list[str]] | None:
    metrics = payload.get("metrics")
    strategy_type = payload.get("strategy_type")
    if not isinstance(metrics, dict) or not isinstance(strategy_type, str):
        return None
    return (
        strategy_type,
        metrics,
        [row for row in (payload.get("breakdown") or []) if isinstance(row, dict)],
        [row for row in (payload.get("transitions") or []) if isinstance(row, dict)],
        payload.get("economics")
        if isinstance(payload.get("economics"), dict)
        else {},
        [str(item) for item in (payload.get("warnings") or []) if str(item)],
    )


def _backtest_summary_lines(payload: dict) -> list[str]:
    """Format persisted metrics without deriving any new business number."""

    typed = _typed_backtest_view(payload)
    if typed is None:
        return _legacy_approval_backtest_lines(payload)
    strategy_type, metrics, breakdown, transitions, economics, warnings = typed
    if strategy_type in {"approval", "reject"}:
        lines = _decision_backtest_lines(
            payload,
            strategy_type=strategy_type,
            metrics=metrics,
            breakdown=breakdown,
            economics=economics,
        )
    elif strategy_type == "limit":
        lines = _limit_backtest_lines(payload, metrics, breakdown, economics)
    elif strategy_type == "pricing":
        lines = _pricing_backtest_lines(payload, metrics, breakdown, economics)
    elif strategy_type == "segmentation":
        lines = _segmentation_backtest_lines(payload, metrics, breakdown)
    else:
        lines = [f"- 未识别的回测类型：{strategy_type}"]
    lines.extend(_backtest_transition_lines(strategy_type, transitions))
    if warnings:
        lines.append("- 回测警告：" + "；".join(warnings))
    return lines


def _legacy_approval_backtest_lines(payload: dict) -> list[str]:
    lines = [
        f"- 审批率：{_pct(payload.get('approval_rate'))}",
        f"- 通过客群坏率：{_pct(payload.get('approved_bad_rate'))}",
        f"- 拒绝客群坏率：{_pct(payload.get('rejected_bad_rate'))}",
    ]
    if int(payload.get("review_count") or 0):
        lines.append(
            f"- 人工复核：{payload.get('review_count')} 户，"
            f"占比 {_pct(payload.get('review_rate'))}，"
            f"坏率 {_pct(payload.get('review_bad_rate'))}"
        )
    lines.extend(
        [
            f"- 预期利润：{_num(payload.get('expected_profit'))}",
            (
                f"- swap-in：{payload.get('swap_in_count', 0)} 户，"
                f"坏率 {_pct(payload.get('swap_in_bad_rate'))}"
            ),
            (
                f"- swap-out：{payload.get('swap_out_count', 0)} 户，"
                f"坏率 {_pct(payload.get('swap_out_bad_rate'))}"
            ),
        ]
    )
    if payload.get("profit_note"):
        lines.append(f"- 利润口径提示：{payload['profit_note']}")
    return lines


def _decision_backtest_lines(
    payload: dict,
    *,
    strategy_type: str,
    metrics: dict,
    breakdown: list[dict],
    economics: dict,
) -> list[str]:
    lines = [
        f"- 回测类型：{'拒绝策略' if strategy_type == 'reject' else '准入策略'}",
        f"- 审批率：{_pct(metrics.get('approve_rate'))}",
        f"- 通过客群坏率：{_pct(metrics.get('approve_bad_rate'))}",
        f"- 拒绝客群坏率：{_pct(metrics.get('reject_bad_rate'))}",
        f"- 人工复核率：{_pct(metrics.get('review_rate'))}",
        f"- 预期利润：{_num(economics.get('expected_profit'))}",
        f"- 标签覆盖率：{_pct(payload.get('label_coverage'))}",
    ]
    if economics.get("profit_note"):
        lines.append(f"- 利润口径提示：{economics['profit_note']}")
    if strategy_type == "reject":
        lines.extend(
            [
                f"- 坏客户捕获率：{_pct(metrics.get('bad_capture_rate'))}",
                f"- 好客户误拒率：{_pct(metrics.get('good_reject_rate'))}",
            ]
        )
    if breakdown:
        lines.extend(
            [
                "",
                "### 决策分布",
                "| 决策 | 样本数 | 占比 | 有标签数 | 坏样本 | 坏率 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in breakdown:
            lines.append(
                f"| {row.get('action', '')} | {row.get('count', '')} | "
                f"{_pct(row.get('rate'))} | {row.get('labeled_count', '')} | "
                f"{row.get('bad_count', '')} | {_pct(row.get('bad_rate'))} |"
            )
    return lines


def _limit_backtest_lines(
    payload: dict, metrics: dict, breakdown: list[dict], economics: dict
) -> list[str]:
    lines = [
        "- 回测类型：额度策略",
        f"- 样本数：{metrics.get('count', payload.get('population_count', ''))}",
        f"- 总额度：{_num(metrics.get('total_limit'))}",
        f"- 户均额度：{_num(metrics.get('mean_limit'))}",
        f"- 最低/最高额度：{_num(metrics.get('min_limit'))} / "
        f"{_num(metrics.get('max_limit'))}",
        f"- 提额/降额/不变人数：{_na(metrics.get('up_count'))} / "
        f"{_na(metrics.get('down_count'))} / {_na(metrics.get('unchanged_count'))}",
        f"- 总额度变化：{_num(metrics.get('total_limit_delta'))}",
        f"- 预期 EAD：{_num(economics.get('expected_ead'))}",
        f"- 预期损失：{_num(economics.get('expected_loss'))}",
        f"- 标签覆盖率：{_pct(payload.get('label_coverage'))}",
    ]
    if breakdown:
        lines.extend(
            [
                "",
                "### 额度分布",
                "| 额度 | 样本数 | 占比 | 有标签数 | 坏样本 | 坏率 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in breakdown:
            lines.append(
                f"| {_num(row.get('assigned_limit'))} | {row.get('count', '')} | "
                f"{_pct(row.get('share'))} | {row.get('labeled_count', '')} | "
                f"{row.get('bad_count', '')} | {_pct(row.get('bad_rate'))} |"
            )
    return lines


def _pricing_backtest_lines(
    payload: dict, metrics: dict, breakdown: list[dict], economics: dict
) -> list[str]:
    lines = [
        "- 回测类型：定价策略",
        f"- 样本数：{metrics.get('count', payload.get('population_count', ''))}",
        f"- 平均年化利率：{_pct(metrics.get('mean_rate'))}",
        f"- 提价/降价/不变人数：{_na(metrics.get('repriced_up_count'))} / "
        f"{_na(metrics.get('repriced_down_count'))} / "
        f"{_na(metrics.get('unchanged_count'))}",
        f"- EAD 加权利率：{_pct(economics.get('ead_weighted_rate'))}",
        f"- 预期收入：{_num(economics.get('revenue'))}",
        f"- 预期损失：{_num(economics.get('expected_loss'))}",
        f"- 预期利润：{_num(economics.get('profit'))}",
        f"- ROA：{_pct(economics.get('roa'))}",
        f"- 较基线利润变化：{_num(economics.get('profit_delta_vs_baseline'))}",
        f"- 标签覆盖率：{_pct(payload.get('label_coverage'))}",
    ]
    if breakdown:
        lines.extend(
            [
                "",
                "### 定价分布",
                "| 年化利率 | 样本数 | 占比 | 有标签数 | 坏样本 | 坏率 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in breakdown:
            lines.append(
                f"| {_pct(row.get('assigned_rate'))} | {row.get('count', '')} | "
                f"{_pct(row.get('share'))} | {row.get('labeled_count', '')} | "
                f"{row.get('bad_count', '')} | {_pct(row.get('bad_rate'))} |"
            )
    return lines


def _segmentation_backtest_lines(
    payload: dict, metrics: dict, breakdown: list[dict]
) -> list[str]:
    lines = [
        "- 回测类型：分群策略",
        f"- 客群数：{metrics.get('segment_count', 'n/a')}",
        f"- 总体坏率：{_pct(metrics.get('overall_bad_rate'))}",
        f"- 标签覆盖率：{_pct(payload.get('label_coverage'))}",
    ]
    if breakdown:
        lines.extend(
            [
                "",
                "### 客群风险分布",
                "| 客群 | 样本数 | 占比 | 有标签数 | 坏样本 | 坏率 | Lift |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in breakdown:
            lines.append(
                f"| {row.get('segment', '')} | {row.get('count', '')} | "
                f"{_pct(row.get('share'))} | {row.get('labeled_count', '')} | "
                f"{row.get('bad_count', '')} | {_pct(row.get('bad_rate'))} | "
                f"{_g(row.get('lift'))} |"
            )
    return lines


def _backtest_transition_lines(
    strategy_type: str,
    transitions: list[dict],
) -> list[str]:
    """Format persisted baseline transitions without deriving summary metrics."""

    if not transitions:
        return []
    if strategy_type in {"approval", "reject"}:
        lines = [
            "",
            "### 相对基线的决策迁移",
            "| 原决策 | 新决策 | 样本数 | 原决策内占比 | 总体占比 | 有标签数 | 坏率 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in transitions:
            lines.append(
                f"| {row.get('from_action', '')} | {row.get('to_action', '')} | "
                f"{row.get('count', '')} | {_pct(row.get('rate'))} | "
                f"{_pct(row.get('population_share'))} | {row.get('labeled_count', '')} | "
                f"{_pct(row.get('bad_rate'))} |"
            )
        return lines
    if strategy_type == "segmentation":
        lines = [
            "",
            "### 相对基线的客群迁移",
            "| 原客群 | 新客群 | 样本数 | 原客群内占比 | 总体占比 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in transitions:
            lines.append(
                f"| {row.get('from_segment', '')} | {row.get('to_segment', '')} | "
                f"{row.get('count', '')} | {_pct(row.get('rate'))} | "
                f"{_pct(row.get('population_share'))} |"
            )
        return lines
    if strategy_type in {"limit", "pricing"}:
        lines = [
            "",
            "### 相对基线的调整方向",
            "| 方向 | 样本数 | 占比 |",
            "| --- | --- | --- |",
        ]
        for row in transitions:
            lines.append(
                f"| {row.get('direction', '')} | {row.get('count', '')} | "
                f"{_pct(row.get('rate'))} |"
            )
        return lines
    return []


def _na(value) -> str:
    return "n/a" if value is None else str(value)


__all__ = ["render_strategy_doc_markdown"]

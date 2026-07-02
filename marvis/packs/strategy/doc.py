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
    sections = [
        "策略概览",
        "规则清单",
        "回测摘要",
        "分数带",
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
        latest = backtests[-1]
        lines.append(f"- 审批率：{_pct(latest.get('approval_rate'))}")
        lines.append(f"- 通过客群坏率：{_pct(latest.get('approved_bad_rate'))}")
        lines.append(f"- 拒绝客群坏率：{_pct(latest.get('rejected_bad_rate'))}")
        lines.append(f"- 预期利润：{_num(latest.get('expected_profit'))}")
        lines.append(
            f"- swap-in：{latest.get('swap_in_count', 0)} 户，坏率 {_pct(latest.get('swap_in_bad_rate'))}"
        )
        lines.append(
            f"- swap-out：{latest.get('swap_out_count', 0)} 户，坏率 {_pct(latest.get('swap_out_bad_rate'))}"
        )
    else:
        lines.append("- 暂无回测结果。")
    lines.append("")

    # 4. Bands
    lines.append("## 分数带")
    if band_stats:
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
    flags = red_flags or []
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
        lines.append("- 监控指标：通过客群坏率漂移、审批率下滑（S5 闭环消费）。")
    else:
        lines.append("- 尚未生成监控计划。")
    lines.append("")

    return "\n".join(lines), sections


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


__all__ = ["render_strategy_doc_markdown"]

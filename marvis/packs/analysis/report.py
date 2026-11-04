"""组合分析汇总门 + 报告拼装工具 (report.py).

- portfolio_gate_summary：纯拼装各 $ref 步骤的 red_flags 与关键数字为 gate payload，
  供决策门（needs_confirmation）展示；红旗清单即门 checklist（先例 MONITORING_RUN 告警门）。
- portfolio_report：把前序步骤已持久化的输出搬进 xlsx（render_portfolio_report），
  登记 artifact 审计行。报告只搬运不重算（INV-1）。
"""

from __future__ import annotations

from pathlib import Path

from marvis.output.portfolio_report import PortfolioReportPayload, render_portfolio_report


def collect_red_flags(*step_outputs) -> list[dict]:
    """把若干步骤输出 dict 里的 red_flags 汇成一个清单（带来源标签，不去重语义）。"""
    flags: list[dict] = []
    for label, output in step_outputs:
        if not isinstance(output, dict):
            continue
        for flag in output.get("red_flags") or []:
            if isinstance(flag, dict):
                flags.append({"source": label, **flag})
    return flags


def _worst_worsening_cell(migration: dict) -> dict | None:
    """The largest pooled rate from a state to a worse one (states run from
    good to bad), read straight off the step's heat table; no recomputation."""
    states = [str(state) for state in migration.get("states") or []]
    order = {state: index for index, state in enumerate(states)}
    best = None
    for row in migration.get("heat_table") or []:
        if not isinstance(row, dict):
            continue
        source = str(row.get("from") or "")
        if source not in order:
            continue
        for target, value in row.items():
            if target == "from" or target not in order or order[target] <= order[source]:
                continue
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            if best is None or value > best["rate"]:
                best = {"from": source, "to": target, "rate": float(value)}
    return best


def gate_summary_payload(
    *,
    flow: dict | None,
    migration: dict | None,
    segment: dict | None,
    trend: dict | None,
    expected_loss: dict | None,
) -> dict:
    """汇总门 payload：关键数字 + 聚合红旗清单（=门 checklist）。"""
    red_flags = collect_red_flags(
        ("flow_rate", flow),
        ("bucket_migration", migration),
        ("segment_profile", segment),
        ("score_stability_trend", trend),
        ("expected_loss_estimate", expected_loss),
    )
    highlights: dict = {}
    if expected_loss:
        highlights["total_el"] = expected_loss.get("total_el")
        # annotate the total_el 口径 (reference-snapshot basis) so the gate headline
        # is self-documenting; pure pass-through of assumptions (INV-1, no recompute).
        el_assumptions = expected_loss.get("assumptions") or {}
        highlights["total_el_basis"] = el_assumptions.get("total_el_basis")
        highlights["reference_snapshot"] = el_assumptions.get("reference_snapshot")
    if segment and isinstance(segment.get("concentration"), dict):
        highlights["hhi"] = segment["concentration"].get("hhi")
        highlights["top1_pct"] = segment["concentration"].get("top1_pct")
        # Which segment that is, looked up from the step's own rows (eval S9:
        # the user asked about concentration and got no sentence about it).
        rows = [row for row in segment.get("segments") or [] if isinstance(row, dict)]
        sized = [row for row in rows if isinstance(row.get("pop_pct"), (int, float))]
        if sized:
            top = max(row["pop_pct"] for row in sized)
            highlights["top1_segment"] = "、".join(
                str(row.get("segment")) for row in sized if row["pop_pct"] == top
            )
        highlights["concentration_flagged"] = any(
            isinstance(flag, dict) and flag.get("kind") == "high_concentration"
            for flag in segment.get("red_flags") or []
        )
    if migration:
        highlights["migration_months"] = migration.get("window_months")
        worst = _worst_worsening_cell(migration)
        if worst is not None:
            highlights["worst_migration"] = worst
    return {
        "highlights": highlights,
        "red_flags": red_flags,
        "red_flag_count": len(red_flags),
        "checklist": [flag.get("message") or flag.get("kind") for flag in red_flags],
    }


def build_report(
    *,
    project_meta: dict | None,
    flow: dict | None,
    migration: dict | None,
    segment: dict | None,
    trend: dict | None,
    expected_loss: dict | None,
    out_path: Path,
) -> tuple[Path, list[str]]:
    red_flags = collect_red_flags(
        ("flow_rate", flow),
        ("bucket_migration", migration),
        ("segment_profile", segment),
        ("score_stability_trend", trend),
        ("expected_loss_estimate", expected_loss),
    )
    payload = PortfolioReportPayload(
        project_meta=dict(project_meta or {}),
        flow=flow,
        migration=migration,
        segment=segment,
        trend=trend,
        expected_loss=expected_loss,
        red_flags=red_flags,
    )
    from marvis.output.portfolio_report import PORTFOLIO_REPORT_SHEETS

    final_path = render_portfolio_report(payload, out_path)
    return final_path, list(PORTFOLIO_REPORT_SHEETS)


__all__ = ["build_report", "collect_red_flags", "gate_summary_payload"]

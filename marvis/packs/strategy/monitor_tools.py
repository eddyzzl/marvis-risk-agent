"""S5: strategy monitoring closure.

``tool_run_strategy_monitoring`` reads an adopted strategy's monitoring plan and
runs one monitoring pass against a fresh dataset:

* if the plan carries an ``experiment_id`` (the strategy is driven by a scoring
  model), it delegates to the modeling ``monitor_run`` kernel unchanged (INV-1),
  passing the plan's threshold overrides through monitor_run's own
  ``monitoring_policy`` channel -> the same PSI/CSI/KS/AUC checks the model
  monitoring surface produces;
* approval/reject strategies keep their strategy-facing approval-rate and
  approved-bad-rate drift checks;
* limit, pricing, and segmentation strategies are applied through the canonical
  vectorized evaluator. Their fresh, directly observable metrics are judged by
  the type-specific threshold specs committed in the adoption plan. Metrics
  requiring inputs that the monitoring request does not carry (for example
  pricing economics) stay explicitly ``n/a``;
* it composes an overall green/amber/red verdict, refreshes the plan's
  ``last_run_at`` (the only write-back field), and writes a ``strategy.monitor``
  audit row.

A pure-rule strategy (no ``experiment_id``) skips PSI/CSI entirely and reports
only the strategy-facing checks. An unadopted strategy raises a typed
``StrategyNotAdoptedError`` -- monitoring is only meaningful against a live
strategy.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository
from marvis.feature.metrics import compute_psi
from marvis.packs.strategy.backtest import strategy_approval_mask
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.errors import StrategyError, StrategyNotAdoptedError
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    load_monitoring_plan,
    save_monitoring_plan,
)
from marvis.repositories.audit import _list_audit_rows
from marvis.settings import build_settings

#: Strategy-facing drift bands (percentage points, configurable). A metric that
#: has moved more than AMBER but at most RED off its adoption baseline is amber;
#: beyond RED is red. Symmetric so both a rising bad rate and a falling approval
#: rate (or the reverse) trip the same bands -- the spec's "approval ±5pp=amber
#: ±10pp=red" made a shared constant for both strategy-facing metrics.
STRATEGY_DRIFT_AMBER_PP = 0.05
STRATEGY_DRIFT_RED_PP = 0.10


def tool_run_strategy_monitoring(inputs: dict, ctx) -> dict:
    runtime = _Runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    dataset_id = str(inputs["dataset_id"])

    meta = _strategy_meta_for_task(runtime, strategy_id, str(ctx.task_id))
    if str(meta.get("status")) != "adopted":
        raise StrategyNotAdoptedError(strategy_id=strategy_id, status=meta.get("status"))

    plan_path = _latest_plan_path(runtime, strategy_id)
    plan = load_monitoring_plan(plan_path)

    strategy = runtime.strategies.get_strategy(strategy_id)
    if strategy is None:
        raise StrategyError(f"strategy not found: {strategy_id}")

    frame = _dataset_frame(runtime, dataset_id, task_id=str(ctx.task_id))
    target_col = _optional_str(inputs.get("target_col"))

    model_checks, top_drifted, model_level = _run_model_monitoring(inputs, ctx, plan)
    if strategy.strategy_type in {"approval", "reject"}:
        strategy_checks, strategy_level = _strategy_drift_checks(
            frame, strategy, plan, target_col=target_col
        )
    else:
        strategy_checks, strategy_level = _typed_strategy_threshold_checks(
            frame,
            strategy,
            plan,
            target_col=target_col,
        )

    checks = [*model_checks, *strategy_checks]
    overall_level = _overall_level([model_level, strategy_level])
    red_flags = [
        {"id": check["id"], "label": check.get("label"), "message": check.get("message")}
        for check in checks
        if check.get("level") == "red"
    ]

    now = datetime.now(UTC).isoformat()
    updated_plan = MonitoringPlan(
        strategy_id=plan.strategy_id,
        version=plan.version,
        cadence_days=plan.cadence_days,
        experiment_id=plan.experiment_id,
        last_run_at=now,
        thresholds=plan.thresholds,
        expectation_baseline=plan.expectation_baseline,
        plan_version=plan.plan_version,
    )
    save_monitoring_plan(plan_path, updated_plan)

    runtime.strategies_repo_write_audit(
        kind="strategy.monitor",
        target_ref=strategy_id,
        detail={
            "task_id": str(ctx.task_id),
            "strategy_id": strategy_id,
            "dataset_id": dataset_id,
            "experiment_id": plan.experiment_id,
            "overall_level": overall_level,
            "row_count": int(len(frame)),
            "last_run_at": now,
        },
    )

    return {
        "strategy_id": strategy_id,
        "dataset_id": dataset_id,
        "experiment_id": plan.experiment_id,
        "overall_level": overall_level,
        "checks": checks,
        "top_drifted_features": top_drifted,
        "red_flags": red_flags,
        "plan_updated": True,
        "last_run_at": now,
        "row_count": int(len(frame)),
    }


def _run_model_monitoring(inputs: dict, ctx, plan: MonitoringPlan):
    """Delegate to the modeling monitor_run kernel when the plan is model-backed.

    Returns (checks, top_drifted_features, level). A pure-rule strategy (no
    experiment_id) skips PSI/CSI and returns ([], [], None)."""
    if not plan.experiment_id:
        return [], [], None
    from marvis.packs.modeling.monitor_tools import tool_monitor_run

    monitor_inputs = {
        "experiment_id": plan.experiment_id,
        "dataset_id": inputs["dataset_id"],
    }
    if inputs.get("score_col"):
        monitor_inputs["score_col"] = inputs["score_col"]
        monitor_inputs["scored_dataset_id"] = inputs["dataset_id"]
        monitor_inputs.pop("dataset_id", None)
    if inputs.get("target_col"):
        monitor_inputs["target_col"] = inputs["target_col"]
    # Plan thresholds override monitor_run's defaults through its own
    # monitoring_policy channel (INV-1: same kernel, plan-supplied thresholds).
    if plan.thresholds:
        monitor_inputs["monitoring_policy"] = {"thresholds": plan.thresholds}

    result = tool_monitor_run(monitor_inputs, ctx)
    checks = [dict(check) for check in (result.get("checks") or []) if isinstance(check, dict)]
    top_drifted = [dict(row) for row in (result.get("top_drifted_features") or []) if isinstance(row, dict)]
    return checks, top_drifted, str(result.get("overall_level") or "green")


def _typed_strategy_threshold_checks(
    frame: pd.DataFrame,
    strategy,
    plan: MonitoringPlan,
    *,
    target_col: str | None,
) -> tuple[list[dict], str]:
    """Evaluate fresh typed-strategy metrics against the adopted plan.

    Only values derivable from the fresh strategy input are populated. In
    particular, limit/pricing economics require PD/LGD/EAD/funding inputs that
    are intentionally absent from ``run_strategy_monitoring``; those plan rows
    are preserved as explicit n/a checks instead of reusing adoption economics.
    """

    evaluation = evaluate_strategy_frame(
        frame,
        strategy.spec or legacy_strategy_to_spec(strategy),
    )
    strategy_type = str(strategy.strategy_type)
    metrics: dict[str, float | None]
    if strategy_type == "limit":
        metrics = {
            "mean_limit": _assigned_numeric_mean(
                evaluation.decisions,
                metric="mean_limit",
            )
        }
    elif strategy_type == "pricing":
        metrics = {
            "mean_rate": _assigned_numeric_mean(
                evaluation.decisions,
                metric="mean_rate",
            )
        }
    elif strategy_type == "segmentation":
        metrics = {
            "overall_bad_rate": _fresh_overall_bad_rate(
                frame,
                target_col=target_col,
            ),
            "segment_share_psi": _segment_share_psi(
                evaluation.decisions,
                baseline=plan.expectation_baseline,
            ),
        }
    else:
        raise StrategyError(
            f"monitoring does not support strategy type: {strategy_type}"
        )

    checks = [
        _plan_threshold_check(check_id, spec, metrics)
        for check_id, spec in plan.thresholds.items()
    ]
    return checks, _overall_level(check["level"] for check in checks)


def _assigned_numeric_mean(decisions: pd.Series, *, metric: str) -> float | None:
    if decisions.empty:
        return None
    numeric = pd.to_numeric(decisions, errors="coerce")
    if bool(numeric.isna().any()):
        raise StrategyError(
            f"typed strategy produced a non-numeric assigned value for {metric}"
        )
    value = float(numeric.mean())
    if not math.isfinite(value):
        raise StrategyError(
            f"typed strategy produced a non-finite assigned value for {metric}"
        )
    return value


def _fresh_overall_bad_rate(
    frame: pd.DataFrame,
    *,
    target_col: str | None,
) -> float | None:
    numeric = _fresh_binary_target(frame, target_col=target_col)
    if numeric is None:
        return None
    labeled = numeric.dropna()
    if labeled.empty:
        return None
    return float(labeled.eq(1).mean())


def _fresh_binary_target(
    frame: pd.DataFrame,
    *,
    target_col: str | None,
) -> pd.Series | None:
    if target_col is None or target_col not in frame.columns:
        return None
    raw = frame[target_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & (numeric.isna() | ~numeric.isin([0, 1]))
    if bool(invalid.any()):
        raise StrategyError("target must contain only 0, 1, or missing")
    return numeric


def _segment_share_psi(
    decisions: pd.Series,
    *,
    baseline: dict,
) -> float | None:
    """PSI of fresh segment shares versus the adoption breakdown.

    The union of baseline and fresh segment ids is used, so a newly appearing or
    disappearing segment is visible through the shared PSI smoothing kernel.
    """

    if decisions.empty:
        return None
    breakdown = baseline.get("breakdown")
    if not isinstance(breakdown, list) or not breakdown:
        return None

    population_count = _finite_float(baseline.get("population_count"))
    baseline_shares: dict[str, float] = {}
    for row in breakdown:
        if not isinstance(row, dict) or "segment" not in row:
            return None
        share = _finite_float(row.get("share"))
        if share is None and population_count not in {None, 0.0}:
            count = _finite_float(row.get("count"))
            if count is not None:
                share = count / float(population_count)
        if share is None or share < 0:
            return None
        token = _segment_token(row["segment"])
        baseline_shares[token] = baseline_shares.get(token, 0.0) + share
    if not baseline_shares or sum(baseline_shares.values()) <= 0:
        return None

    fresh_counts: dict[str, int] = {}
    for value in decisions.tolist():
        token = _segment_token(value)
        fresh_counts[token] = fresh_counts.get(token, 0) + 1
    tokens = sorted(set(baseline_shares) | set(fresh_counts))
    expected = np.asarray([baseline_shares.get(token, 0.0) for token in tokens])
    actual = np.asarray(
        [fresh_counts.get(token, 0) / len(decisions) for token in tokens]
    )
    return float(compute_psi(expected, actual))


def _segment_token(value) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "segmentation decisions must be finite scalar segment ids"
        ) from exc


def _plan_threshold_check(
    check_id,
    raw_spec,
    metrics: dict[str, float | None],
) -> dict:
    spec = raw_spec if isinstance(raw_spec, dict) else {}
    check_id = str(check_id)
    metric = str(spec.get("metric") or check_id)
    label = str(spec.get("label") or check_id)
    direction = str(spec.get("direction") or "max")
    warn = _finite_float(spec.get("warn"))
    fail = _finite_float(spec.get("fail"))
    actual = _finite_float(metrics.get(metric))
    check = {
        "id": check_id,
        "label": label,
        "metric": metric,
        "value": actual,
        "actual": actual,
        "direction": direction,
        "warn": warn,
        "fail": fail,
    }
    if actual is None:
        return {
            **check,
            "level": "n/a",
            "status": "missing",
            "message": (
                f"本次新鲜样本未提供计算 {metric} 所需的确定性输入；"
                "该指标标记为 n/a，未填充或推断数值。"
            ),
        }
    if direction not in {"min", "max"} or (warn is None and fail is None):
        return {
            **check,
            "level": "n/a",
            "status": "needs_policy",
            "message": "监控计划缺少可执行的 direction/warn/fail 阈值，无法自动判级。",
        }

    if direction == "min":
        if fail is not None and actual < fail - _DRIFT_EPS:
            level, status = "red", "fail"
        elif warn is not None and actual < warn - _DRIFT_EPS:
            level, status = "amber", "warn"
        else:
            level, status = "green", "pass"
    elif fail is not None and actual > fail + _DRIFT_EPS:
        level, status = "red", "fail"
    elif warn is not None and actual > warn + _DRIFT_EPS:
        level, status = "amber", "warn"
    else:
        level, status = "green", "pass"
    return {
        **check,
        "level": level,
        "status": status,
        "message": _threshold_message(
            actual,
            direction=direction,
            warn=warn,
            fail=fail,
            level=level,
        ),
    }


def _threshold_message(
    actual: float,
    *,
    direction: str,
    warn: float | None,
    fail: float | None,
    level: str,
) -> str:
    operator = "低于" if direction == "min" else "高于"
    if level == "red" and fail is not None:
        return f"实际 {actual:.6g} {operator} fail 阈值 {fail:.6g}。"
    if level == "amber" and warn is not None:
        return f"实际 {actual:.6g} {operator} warn 阈值 {warn:.6g}。"
    return (
        f"实际 {actual:.6g} 在监控阈值内"
        f"（warn={warn if warn is not None else 'n/a'}, "
        f"fail={fail if fail is not None else 'n/a'}）。"
    )


def _finite_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strategy_drift_checks(
    frame: pd.DataFrame,
    strategy,
    plan: MonitoringPlan,
    *,
    target_col: str | None,
):
    """Evaluate approval/reject metrics using the versioned plan thresholds.

    Approval-facing ids retain the historical ``*_drift`` projection for API
    compatibility, but their level now comes from the absolute warn/fail values
    committed in the monitoring plan.  Reject plans can therefore execute their
    bad-capture and good-reject contracts instead of silently ignoring them.
    """
    baseline = plan.expectation_baseline or {}
    approved = strategy_approval_mask(frame, strategy)
    row_count = int(len(frame))
    metrics: dict[str, float | None] = {
        "approval_rate": (
            float(approved.sum() / row_count) if row_count else None
        ),
        "approved_bad_rate": None,
        "bad_capture_rate": None,
        "good_reject_rate": None,
    }
    target = _fresh_binary_target(frame, target_col=target_col)
    if target is not None:
        approved_labeled = target.loc[approved & target.notna()]
        if not approved_labeled.empty:
            metrics["approved_bad_rate"] = float(approved_labeled.eq(1).mean())
        rejected = ~approved
        bad = target.eq(1)
        good = target.eq(0)
        bad_count = int(bad.sum())
        good_count = int(good.sum())
        if bad_count:
            metrics["bad_capture_rate"] = float((rejected & bad).sum() / bad_count)
        if good_count:
            metrics["good_reject_rate"] = float((rejected & good).sum() / good_count)

    checks = [
        _approval_plan_check(check_id, spec, metrics, baseline=baseline)
        for check_id, spec in plan.thresholds.items()
    ]
    if not checks:
        # Historical plans without threshold specs retain their old drift-only
        # presentation rather than becoming silently healthy.
        for metric in ("approval_rate", "approved_bad_rate"):
            actual = metrics[metric]
            baseline_value = _optional_float(baseline.get(metric))
            if actual is None:
                checks.append(
                    {
                        "id": f"{metric}_drift",
                        "label": metric,
                        "metric": metric,
                        "value": None,
                        "level": "n/a",
                        "baseline": baseline_value,
                        "actual": None,
                        "message": "监控计划或新鲜样本缺少该指标所需证据。",
                    }
                )
            else:
                checks.append(
                    _drift_check(
                        check_id=f"{metric}_drift",
                        label=metric,
                        actual=actual,
                        baseline=baseline_value,
                        metric=metric,
                    )
                )
    level = _overall_level(check["level"] for check in checks)
    return checks, level


def _approval_plan_check(
    check_id,
    spec,
    metrics: dict[str, float | None],
    *,
    baseline: dict,
) -> dict:
    check = _plan_threshold_check(check_id, spec, metrics)
    metric = str(check["metric"])
    if metric not in {"approval_rate", "approved_bad_rate"}:
        return check
    baseline_value = _optional_float(baseline.get(metric))
    actual = check.get("actual")
    drift = (
        None
        if actual is None or baseline_value is None
        else float(actual) - baseline_value
    )
    return {
        **check,
        "id": f"{metric}_drift",
        "value": drift,
        "baseline": baseline_value,
        "actual": actual,
    }


def _drift_check(
    *,
    check_id: str,
    label: str,
    actual: float,
    baseline: float | None,
    metric: str | None = None,
) -> dict:
    if baseline is None:
        return {
            "id": check_id,
            "label": label,
            "metric": metric or check_id,
            "value": None,
            "level": "n/a",
            "baseline": None,
            "actual": float(actual),
            "message": "监控计划缺少该指标的采纳基线，无法比较漂移。",
        }
    drift = float(actual) - float(baseline)
    level = _drift_level(drift)
    return {
        "id": check_id,
        "label": label,
        "metric": metric or check_id,
        "value": drift,
        "level": level,
        "baseline": float(baseline),
        "actual": float(actual),
        "message": (
            f"实际 {actual:.4f} vs 采纳基线 {baseline:.4f}，漂移 {drift:+.4f}（{_drift_gloss(level)}）。"
        ),
    }


#: Float tolerance so a drift that sits exactly on a band boundary (e.g. an
#: approval rate that moved by precisely 10pp, where 0.7 - 0.8 evaluates to
#: -0.1000000000000001 in IEEE-754) grades to the lower/less-severe tier
#: deterministically instead of flipping on binary-float noise.
_DRIFT_EPS = 1e-9


def _drift_level(drift: float) -> str:
    magnitude = abs(float(drift))
    if magnitude > STRATEGY_DRIFT_RED_PP + _DRIFT_EPS:
        return "red"
    if magnitude > STRATEGY_DRIFT_AMBER_PP + _DRIFT_EPS:
        return "amber"
    return "green"


def _drift_gloss(level: str) -> str:
    return {"red": "红灯", "amber": "黄灯", "green": "绿灯"}.get(level, level)


def _overall_level(levels) -> str:
    values = {str(level) for level in levels if level is not None}
    if "red" in values:
        return "red"
    if "amber" in values:
        return "amber"
    if "green" in values:
        return "green"
    return "n/a"


def _latest_plan_path(runtime: "_Runtime", strategy_id: str) -> Path:
    artifacts = [
        artifact
        for artifact in runtime.strategies.list_strategy_artifacts(strategy_id)
        if artifact.get("kind") == "monitoring_plan_json"
    ]
    if not artifacts:
        raise StrategyError(
            f"策略 {strategy_id} 没有登记的监控计划（monitoring_plan_json）；请先采纳该策略。"
        )
    return Path(artifacts[-1]["path"])


#: Recent monitoring runs summarised in the report timeline (audit rows). N is
#: bounded so a long-lived strategy's report shows the recent trend, not its whole
#: history (the audit table stays the source of record).
_REPORT_TIMELINE_LIMIT = 20

_LEVEL_LABEL = {"green": "绿", "amber": "黄", "red": "红", "n/a": "n/a"}


def tool_render_monitoring_report(inputs: dict, ctx) -> dict:
    """S5: render a monitoring report (Markdown) for an adopted strategy.

    Aggregates the strategy's recent ``strategy.monitor`` audit rows into an
    overall-level timeline and renders the latest run's per-check table (passed
    through from the run step's output when available). Registers the report as a
    ``monitoring_report_md`` strategy artifact. When a ``disposition`` is supplied
    (the red-light gate's parsed choice), a ``next_action`` is surfaced for the
    driver -- for "new_version" it names STRATEGY_DEVELOPMENT as the follow-up, but
    never creates a task itself (single-machine, human-in-the-loop)."""
    runtime = _Runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    meta = _strategy_meta_for_task(runtime, strategy_id, str(ctx.task_id))

    checks = [dict(c) for c in (inputs.get("checks") or []) if isinstance(c, dict)]
    overall_level = _optional_str(inputs.get("overall_level"))

    timeline = _monitoring_timeline(runtime, strategy_id)
    markdown = _render_report_markdown(
        strategy_id=strategy_id,
        version=int(meta.get("version", 1)),
        overall_level=overall_level,
        checks=checks,
        timeline=timeline,
    )

    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    report_path = strategy_dir / f"monitoring_report_{strategy_id}_v{int(meta.get('version', 1))}.md"
    report_path.write_text(markdown, encoding="utf-8")

    runtime.strategies.save_strategy_artifact(
        strategy_id, kind="monitoring_report_md", path=str(report_path)
    )
    runtime.strategies_repo_write_audit(
        kind="strategy.artifact",
        target_ref=strategy_id,
        detail={"task_id": str(ctx.task_id), "kind": "monitoring_report_md", "path": str(report_path)},
    )

    result = {
        "strategy_id": strategy_id,
        "report_path": str(report_path),
        "overall_level": overall_level,
        "timeline": timeline,
    }
    next_action = monitoring_next_action(_optional_str(inputs.get("disposition")), strategy_id=strategy_id)
    if next_action is not None:
        result["next_action"] = next_action
    return result


def _monitoring_timeline(runtime: "_Runtime", strategy_id: str) -> list[dict]:
    rows = _list_audit_rows(
        runtime.settings.db_path,
        kind="strategy.monitor",
        target_ref=strategy_id,
        limit=_REPORT_TIMELINE_LIMIT,
    )
    timeline: list[dict] = []
    for row in rows:
        detail = row.get("detail") or {}
        timeline.append({
            "at": row.get("at"),
            "overall_level": detail.get("overall_level"),
            "dataset_id": detail.get("dataset_id"),
            "row_count": detail.get("row_count"),
        })
    return timeline


def _render_report_markdown(
    *,
    strategy_id: str,
    version: int,
    overall_level: str | None,
    checks: list[dict],
    timeline: list[dict],
) -> str:
    lines = [
        f"# 策略监控报告 — {strategy_id} v{version}",
        "",
    ]
    if overall_level:
        lines.append(f"- 最近一次总体判级：**{_LEVEL_LABEL.get(overall_level, overall_level)}**")
    lines.append(f"- 历史监控次数：{len(timeline)}")
    lines.append("")
    if checks:
        lines.append("## 最近一次监控明细")
        lines.append("")
        lines.append("| 检查项 | 判级 | 值 | 说明 |")
        lines.append("| --- | --- | --- | --- |")
        for check in checks:
            value = check.get("value")
            value_text = "n/a" if value is None else f"{float(value):+.4f}" if isinstance(value, (int, float)) else str(value)
            lines.append(
                f"| {check.get('label') or check.get('id')} "
                f"| {_LEVEL_LABEL.get(str(check.get('level')), check.get('level'))} "
                f"| {value_text} | {check.get('message') or ''} |"
            )
        lines.append("")
    if timeline:
        lines.append("## 监控判级时间线")
        lines.append("")
        lines.append("| 时间 | 总体判级 | 样本量 |")
        lines.append("| --- | --- | --- |")
        for entry in timeline:
            lines.append(
                f"| {entry.get('at') or ''} "
                f"| {_LEVEL_LABEL.get(str(entry.get('overall_level')), entry.get('overall_level'))} "
                f"| {entry.get('row_count') if entry.get('row_count') is not None else ''} |"
            )
        lines.append("")
    return "\n".join(lines)


#: Red-light disposition keyword codes and the driver next_action each maps to.
#: "observe" / "adjust_threshold" stay in-place (no follow-up task); "new_version"
#: names STRATEGY_DEVELOPMENT as the suggested follow-up (via new_version_from) but
#: never auto-creates it -- the driver surfaces the prompt for the user to accept.
DISPOSITION_OBSERVE = "observe"
DISPOSITION_ADJUST_THRESHOLD = "adjust_threshold"
DISPOSITION_NEW_VERSION = "new_version"


def monitoring_next_action(disposition: str | None, *, strategy_id: str) -> dict | None:
    if disposition == DISPOSITION_NEW_VERSION:
        return {
            "kind": "suggest_template",
            "template_id": "strategy_development",
            "parent_strategy_id": strategy_id,
            "prompt": (
                f"监控红灯，建议基于策略 {strategy_id} 起一个新版本（new_version_from）"
                f"并重新走一遍策略开发流程。是否开始？"
            ),
        }
    if disposition == DISPOSITION_ADJUST_THRESHOLD:
        return {
            "kind": "note",
            "prompt": "已选择「调阈值重跑」：请调整监控计划阈值后重新运行策略监控。",
        }
    if disposition == DISPOSITION_OBSERVE:
        return {
            "kind": "note",
            "prompt": "已选择「维持并观察」：保持当前策略，加强下一周期监控。",
        }
    return None


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.strategies = StrategyRepository(self.settings.db_path)

    def strategies_repo_write_audit(self, *, kind: str, target_ref: str, detail: dict) -> None:
        from marvis.db_schema import connect
        from marvis.repositories.strategy import _write_audit_row

        with connect(self.settings.db_path) as conn:
            _write_audit_row(
                conn,
                kind=kind,
                target_ref=target_ref,
                outcome="succeeded",
                detail=detail,
            )


def _dataset_frame(
    runtime: _Runtime,
    dataset_id: str,
    *,
    task_id: str,
) -> pd.DataFrame:
    try:
        dataset = runtime.registry.get(dataset_id)
    except KeyError:
        raise StrategyError(f"dataset not found: {dataset_id}") from None
    if str(dataset.task_id) != str(task_id):
        raise StrategyError(f"dataset not found: {dataset_id}")
    return runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id))


def _strategy_meta_for_task(
    runtime: _Runtime, strategy_id: str, task_id: str
) -> dict:
    metadata = runtime.strategies.get_strategy_meta(strategy_id)
    if metadata is None or str(metadata["task_id"]) != str(task_id):
        raise StrategyError(f"strategy not found: {strategy_id}")
    return metadata


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DISPOSITION_ADJUST_THRESHOLD",
    "DISPOSITION_NEW_VERSION",
    "DISPOSITION_OBSERVE",
    "STRATEGY_DRIFT_AMBER_PP",
    "STRATEGY_DRIFT_RED_PP",
    "monitoring_next_action",
    "tool_render_monitoring_report",
    "tool_run_strategy_monitoring",
]

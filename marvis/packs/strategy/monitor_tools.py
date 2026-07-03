"""S5: strategy monitoring closure.

``tool_run_strategy_monitoring`` reads an adopted strategy's monitoring plan and
runs one monitoring pass against a fresh dataset:

* if the plan carries an ``experiment_id`` (the strategy is driven by a scoring
  model), it delegates to the modeling ``monitor_run`` kernel unchanged (INV-1),
  passing the plan's threshold overrides through monitor_run's own
  ``monitoring_policy`` channel -> the same PSI/CSI/KS/AUC checks the model
  monitoring surface produces;
* it always computes the *strategy-facing* drift: apply the adopted strategy to
  the fresh dataset, measure approval rate (always) and approved bad rate (only
  when the sample carries labels) and compare them against the plan's
  ``expectation_baseline`` (the approval/bad rate committed at adoption), graded
  into green/amber/red on fixed percentage-point drift bands;
* it composes an overall green/amber/red verdict, refreshes the plan's
  ``last_run_at`` (the only write-back field), and writes a ``strategy.monitor``
  audit row.

A pure-rule strategy (no ``experiment_id``) skips PSI/CSI entirely and reports
only the strategy-facing checks. An unadopted strategy raises a typed
``StrategyNotAdoptedError`` -- monitoring is only meaningful against a live
strategy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository
from marvis.packs.strategy.errors import StrategyError, StrategyNotAdoptedError
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    load_monitoring_plan,
    save_monitoring_plan,
)
from marvis.packs.strategy.strategy import apply_strategy
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

    meta = runtime.strategies.get_strategy_meta(strategy_id)
    if meta is None:
        raise StrategyError(f"strategy not found: {strategy_id}")
    if str(meta.get("status")) != "adopted":
        raise StrategyNotAdoptedError(strategy_id=strategy_id, status=meta.get("status"))

    plan_path = _latest_plan_path(runtime, strategy_id)
    plan = load_monitoring_plan(plan_path)

    strategy = runtime.strategies.get_strategy(strategy_id)
    if strategy is None:
        raise StrategyError(f"strategy not found: {strategy_id}")

    frame = _dataset_frame(runtime, dataset_id)
    target_col = _optional_str(inputs.get("target_col"))

    model_checks, top_drifted, model_level = _run_model_monitoring(inputs, ctx, plan)
    strategy_checks, strategy_level = _strategy_drift_checks(
        frame, strategy, plan, target_col=target_col
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


def _strategy_drift_checks(
    frame: pd.DataFrame,
    strategy,
    plan: MonitoringPlan,
    *,
    target_col: str | None,
):
    """Strategy-facing drift: approval-rate drift (always) and approved-bad-rate
    drift (labels only) vs the adoption expectation_baseline."""
    baseline = plan.expectation_baseline or {}
    decision = apply_strategy(frame, strategy)
    approved = decision.astype(str) != "reject"
    row_count = int(len(frame))
    approval_rate = float(approved.sum() / row_count) if row_count else 0.0

    approval_check = _drift_check(
        check_id="approval_rate_drift",
        label="审批率漂移",
        actual=approval_rate,
        baseline=_optional_float(baseline.get("approval_rate")),
    )

    approved_bad_rate = None
    if target_col and target_col in frame.columns and int(approved.sum()) > 0:
        target = pd.to_numeric(frame[target_col], errors="coerce")
        approved_target = target.loc[approved].dropna()
        if not approved_target.empty:
            approved_bad_rate = float((approved_target == 1).mean())

    if approved_bad_rate is None:
        bad_rate_check = {
            "id": "approved_bad_rate_drift",
            "label": "通过客群坏率漂移",
            "metric": "approved_bad_rate",
            "value": None,
            "level": "n/a",
            "baseline": _optional_float(baseline.get("approved_bad_rate")),
            "actual": None,
            "message": "本次样本无有效标签，无法计算通过客群坏率漂移；这是正常的监控场景，不代表数据质量问题。",
        }
    else:
        bad_rate_check = _drift_check(
            check_id="approved_bad_rate_drift",
            label="通过客群坏率漂移",
            actual=approved_bad_rate,
            baseline=_optional_float(baseline.get("approved_bad_rate")),
            metric="approved_bad_rate",
        )

    checks = [approval_check, bad_rate_check]
    level = _overall_level(check["level"] for check in checks)
    return checks, level


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
    return "green"


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


def _dataset_frame(runtime: _Runtime, dataset_id: str) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    return runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id))


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
    "STRATEGY_DRIFT_AMBER_PP",
    "STRATEGY_DRIFT_RED_PP",
    "tool_run_strategy_monitoring",
]

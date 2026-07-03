"""S5: the monitoring plan single source of truth.

A monitoring plan is the small JSON contract an adopted strategy commits to at
adoption time: the cadence it should be re-monitored on, the drift thresholds to
judge against, and the expectation baseline (approval/bad rate from the adoption
backtest) the strategy-facing drift checks compare against. S2 writes it at the
adopt gate; S5 reads it back and drives ``run_strategy_monitoring`` off it, so the
read/write path lives here (one module) rather than being duplicated across the
S2 write point and the S5 read point.

The on-disk shape is deliberately forgiving: ``load_monitoring_plan`` tolerates
unknown fields (forward compatibility with plans written by a newer version) and
raises a typed ``StrategyError`` only when a required field is missing or the file
cannot be parsed. ``thresholds`` mirrors the modeling MONITOR_RUN_THRESHOLDS shape
(label/metric/direction/warn/fail) so monitor_run can consume a strategy plan's
overrides through its own ``monitoring_policy`` channel unchanged (INV-1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from marvis.packs.strategy.errors import StrategyError

#: Current on-disk plan schema version. Bumped only on a breaking field change;
#: ``load_monitoring_plan`` reads older versions best-effort (unknown fields are
#: dropped, missing optional fields default), so a version mismatch is not by
#: itself an error.
PLAN_VERSION = 1

#: Default re-monitoring cadence (days) when a plan does not pin its own.
DEFAULT_CADENCE_DAYS = 30


@dataclass(frozen=True)
class MonitoringPlan:
    """Parsed monitoring plan. ``experiment_id`` is None for a pure-rule strategy
    (no scoring model -> PSI/CSI are skipped and only the strategy-facing
    approval/bad-rate drift checks run). ``last_run_at`` is the only field
    ``run_strategy_monitoring`` writes back (the plan is otherwise immutable after
    adoption)."""

    strategy_id: str
    version: int
    cadence_days: int = DEFAULT_CADENCE_DAYS
    experiment_id: str | None = None
    last_run_at: str | None = None
    thresholds: dict = field(default_factory=dict)
    expectation_baseline: dict = field(default_factory=dict)
    plan_version: int = PLAN_VERSION

    def to_dict(self) -> dict:
        """Serialize to the on-disk JSON shape. Deterministic key order via the
        json.dumps(sort_keys=True) the writer uses."""
        return {
            "plan_version": int(self.plan_version),
            "strategy_id": self.strategy_id,
            "version": int(self.version),
            "cadence_days": int(self.cadence_days),
            "experiment_id": self.experiment_id,
            "last_run_at": self.last_run_at,
            "thresholds": dict(self.thresholds),
            "expectation_baseline": dict(self.expectation_baseline),
        }


def build_monitoring_plan(
    *,
    strategy_id: str,
    version: int,
    approved_bad_rate: float,
    approval_rate: float,
    experiment_id: str | None = None,
    cadence_days: int = DEFAULT_CADENCE_DAYS,
    source_backtest_id: str | None = None,
    bad_rate_warn_delta: float = 0.02,
    bad_rate_fail_delta: float = 0.05,
    approval_warn_delta: float = 0.05,
    approval_fail_delta: float = 0.10,
    thresholds: dict | None = None,
) -> dict:
    """Build the adoption-time monitoring plan dict (S2 write point).

    ``thresholds`` mirrors MONITOR_RUN_THRESHOLDS (label/metric/direction/warn/
    fail); the defaults are derived from the adoption backtest's approval/bad rate
    plus the delta bands, but a caller may pass an explicit ``thresholds`` override
    (the spec's "采纳时可覆盖默认"). ``expectation_baseline`` snapshots the
    approval/bad rate the strategy committed to at adoption, which the S5
    strategy-facing drift checks compare a fresh run against."""
    resolved_thresholds = (
        dict(thresholds)
        if thresholds
        else {
            "approved_bad_rate": {
                "label": "通过客群坏率漂移",
                "metric": "approved_bad_rate",
                "direction": "max",
                "warn": float(approved_bad_rate + bad_rate_warn_delta),
                "fail": float(approved_bad_rate + bad_rate_fail_delta),
            },
            "approval_rate": {
                "label": "审批率下滑",
                "metric": "approval_rate",
                "direction": "min",
                "warn": float(approval_rate - approval_warn_delta),
                "fail": float(approval_rate - approval_fail_delta),
            },
        }
    )
    plan = MonitoringPlan(
        strategy_id=str(strategy_id),
        version=int(version),
        cadence_days=int(cadence_days),
        experiment_id=str(experiment_id) if experiment_id else None,
        last_run_at=None,
        thresholds=resolved_thresholds,
        expectation_baseline={
            "approval_rate": float(approval_rate),
            "approved_bad_rate": float(approved_bad_rate),
            "source_backtest_id": str(source_backtest_id) if source_backtest_id else None,
        },
    )
    return plan.to_dict()


def load_monitoring_plan(artifact_path: str | Path) -> MonitoringPlan:
    """Parse a monitoring plan file into a MonitoringPlan.

    Unknown fields are tolerated (forward compat); a missing required field
    (strategy_id / version) or an unreadable/non-object file raises StrategyError
    with a specific message so the caller surfaces a typed failure rather than a
    KeyError deep in the stack."""
    path = Path(artifact_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrategyError(f"无法读取监控计划文件 {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StrategyError(f"监控计划文件 {path} 不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StrategyError(f"监控计划文件 {path} 顶层不是 JSON 对象。")
    return _plan_from_dict(payload, source=str(path))


def save_monitoring_plan(artifact_path: str | Path, plan: MonitoringPlan | dict) -> Path:
    """Write a monitoring plan to disk (deterministic key order). Accepts either a
    MonitoringPlan or an already-built plan dict (the S2 write point builds the
    dict via build_monitoring_plan, then persists it here -- single write path)."""
    path = Path(artifact_path)
    payload = plan.to_dict() if isinstance(plan, MonitoringPlan) else dict(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _plan_from_dict(payload: dict, *, source: str) -> MonitoringPlan:
    strategy_id = payload.get("strategy_id")
    if not strategy_id:
        raise StrategyError(f"监控计划文件 {source} 缺少必填字段 strategy_id。")
    if "version" not in payload:
        raise StrategyError(f"监控计划文件 {source} 缺少必填字段 version。")
    try:
        version = int(payload["version"])
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"监控计划文件 {source} 的 version 不是整数: {payload['version']!r}") from exc
    thresholds = payload.get("thresholds")
    expectation = payload.get("expectation_baseline")
    return MonitoringPlan(
        strategy_id=str(strategy_id),
        version=version,
        cadence_days=int(payload.get("cadence_days") or DEFAULT_CADENCE_DAYS),
        experiment_id=(str(payload["experiment_id"]) if payload.get("experiment_id") else None),
        last_run_at=(str(payload["last_run_at"]) if payload.get("last_run_at") else None),
        thresholds=dict(thresholds) if isinstance(thresholds, dict) else {},
        expectation_baseline=dict(expectation) if isinstance(expectation, dict) else {},
        plan_version=int(payload.get("plan_version") or PLAN_VERSION),
    )


__all__ = [
    "DEFAULT_CADENCE_DAYS",
    "MonitoringPlan",
    "PLAN_VERSION",
    "build_monitoring_plan",
    "load_monitoring_plan",
    "save_monitoring_plan",
]

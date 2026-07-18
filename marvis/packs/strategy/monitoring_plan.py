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

from copy import deepcopy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from marvis.packs.strategy.errors import StrategyError

#: Current on-disk plan schema version. V2 adds an immutable revision identity
#: and safe economics bindings. ``load_monitoring_plan`` continues to read V1
#: artifacts by supplying revision-1 defaults.
PLAN_VERSION = 2
PLAN_SCHEMA_VERSION = "strategy.monitoring_plan.v2"
LEGACY_PLAN_VERSION = 1

#: Default re-monitoring cadence (days) when a plan does not pin its own.
DEFAULT_CADENCE_DAYS = 30


@dataclass(frozen=True)
class MonitoringPlan:
    """Parsed monitoring plan. ``experiment_id`` is None for a pure-rule strategy
    (no scoring model -> PSI/CSI are skipped and only the strategy-facing
    approval/bad-rate drift checks run). ``last_run_at`` remains readable only
    for legacy artifact compatibility; immutable ledger plans keep it empty and
    monitoring timestamps live on run receipts."""

    strategy_id: str
    version: int
    cadence_days: int = DEFAULT_CADENCE_DAYS
    experiment_id: str | None = None
    last_run_at: str | None = None
    thresholds: dict = field(default_factory=dict)
    expectation_baseline: dict = field(default_factory=dict)
    plan_version: int = PLAN_VERSION
    monitoring_plan_id: str | None = None
    revision: int = 1
    supersedes_plan_id: str | None = None
    economics_bindings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.strategy_id).strip():
            raise StrategyError("monitoring plan strategy_id must be non-empty")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise StrategyError("monitoring plan strategy version must be an integer")
        if isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int):
            raise StrategyError("monitoring plan plan_version must be an integer")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise StrategyError("monitoring plan revision must be a positive integer")
        monitoring_plan_id = _optional_non_empty_text(
            self.monitoring_plan_id, field_name="monitoring_plan_id"
        )
        supersedes_plan_id = _optional_non_empty_text(
            self.supersedes_plan_id, field_name="supersedes_plan_id"
        )
        if self.revision == 1 and supersedes_plan_id is not None:
            raise StrategyError("monitoring plan revision 1 cannot supersede another plan")
        if self.revision > 1 and supersedes_plan_id is None:
            raise StrategyError("monitoring plan revision above 1 requires supersedes_plan_id")
        object.__setattr__(self, "monitoring_plan_id", monitoring_plan_id)
        object.__setattr__(self, "supersedes_plan_id", supersedes_plan_id)
        object.__setattr__(
            self,
            "thresholds",
            deepcopy(dict(self.thresholds)) if isinstance(self.thresholds, Mapping) else {},
        )
        object.__setattr__(
            self,
            "expectation_baseline",
            (
                deepcopy(dict(self.expectation_baseline))
                if isinstance(self.expectation_baseline, Mapping)
                else {}
            ),
        )
        object.__setattr__(
            self,
            "economics_bindings",
            _normalize_economics_bindings(self.economics_bindings),
        )

    def to_dict(self) -> dict:
        """Serialize to the on-disk JSON shape. Deterministic key order via the
        json.dumps(sort_keys=True) the writer uses."""
        return {
            "plan_version": int(self.plan_version),
            "monitoring_plan_id": self.monitoring_plan_id,
            "strategy_id": self.strategy_id,
            "version": int(self.version),
            "revision": int(self.revision),
            "supersedes_plan_id": self.supersedes_plan_id,
            "cadence_days": int(self.cadence_days),
            "experiment_id": self.experiment_id,
            "last_run_at": self.last_run_at,
            "thresholds": deepcopy(dict(self.thresholds)),
            "expectation_baseline": deepcopy(dict(self.expectation_baseline)),
            "economics_bindings": deepcopy(dict(self.economics_bindings)),
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
    monitoring_plan_id: str | None = None,
    revision: int = 1,
    supersedes_plan_id: str | None = None,
    economics_bindings: dict | None = None,
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
        monitoring_plan_id=monitoring_plan_id,
        revision=revision,
        supersedes_plan_id=supersedes_plan_id,
        economics_bindings=dict(economics_bindings or {}),
    )
    return plan.to_dict()


def canonical_monitoring_plan_json(plan: MonitoringPlan | Mapping[str, Any]) -> str:
    """Return the canonical JSON used by the immutable plan ledger."""

    payload = plan.to_dict() if isinstance(plan, MonitoringPlan) else dict(plan)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"monitoring plan is not canonical JSON: {exc}") from exc


def canonical_monitoring_plan_hash(plan: MonitoringPlan | Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_monitoring_plan_json(plan).encode("utf-8")).hexdigest()


def canonical_economics_bindings_hash(bindings: Mapping[str, Any] | None) -> str:
    normalized = _normalize_economics_bindings(bindings or {})
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def monitoring_plan_from_dict(
    payload: Mapping[str, Any], *, source: str = "<memory>"
) -> MonitoringPlan:
    """Parse an in-memory payload through the same compatibility boundary."""

    if not isinstance(payload, Mapping):
        raise StrategyError(f"监控计划 {source} 顶层不是对象。")
    return _plan_from_dict(dict(payload), source=source)


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
    try:
        return MonitoringPlan(
            strategy_id=str(strategy_id),
            version=version,
            cadence_days=int(payload.get("cadence_days") or DEFAULT_CADENCE_DAYS),
            experiment_id=(
                str(payload["experiment_id"]) if payload.get("experiment_id") else None
            ),
            last_run_at=(
                str(payload["last_run_at"]) if payload.get("last_run_at") else None
            ),
            thresholds=dict(thresholds) if isinstance(thresholds, dict) else {},
            expectation_baseline=(
                dict(expectation) if isinstance(expectation, dict) else {}
            ),
            # A file with no explicit plan_version predates V2.
            plan_version=int(payload.get("plan_version") or LEGACY_PLAN_VERSION),
            monitoring_plan_id=(
                str(payload["monitoring_plan_id"])
                if payload.get("monitoring_plan_id")
                else None
            ),
            revision=int(payload.get("revision") or 1),
            supersedes_plan_id=(
                str(payload["supersedes_plan_id"])
                if payload.get("supersedes_plan_id")
                else None
            ),
            economics_bindings=(
                dict(payload["economics_bindings"])
                if isinstance(payload.get("economics_bindings"), dict)
                else {}
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"监控计划文件 {source} 字段格式无效: {exc}") from exc


def _optional_non_empty_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"monitoring plan {field_name} must be non-empty")
    return value.strip()


def _normalize_economics_bindings(bindings: Mapping[str, Any]) -> dict[str, dict]:
    if not isinstance(bindings, Mapping):
        raise StrategyError("monitoring plan economics_bindings must be an object")
    normalized: dict[str, dict] = {}
    for raw_name, raw_binding in sorted(bindings.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise StrategyError("economics binding name must be non-empty")
        name = raw_name.strip()
        if not isinstance(raw_binding, Mapping):
            raise StrategyError(f"economics binding {name} must be an object")
        kind = raw_binding.get("kind")
        if kind == "scalar":
            if set(raw_binding) != {"kind", "value"}:
                raise StrategyError(
                    f"economics scalar binding {name} may contain only kind and value"
                )
            value = raw_binding["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StrategyError(f"economics scalar binding {name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise StrategyError(f"economics scalar binding {name} must be finite")
            normalized[name] = {"kind": "scalar", "value": numeric}
        elif kind == "column":
            if set(raw_binding) != {"kind", "column"}:
                raise StrategyError(
                    f"economics column binding {name} may contain only kind and column"
                )
            column = raw_binding["column"]
            if not isinstance(column, str) or not column.strip():
                raise StrategyError(f"economics column binding {name} must be non-empty")
            normalized[name] = {"kind": "column", "column": column.strip()}
        else:
            raise StrategyError(f"economics binding {name} has unsupported kind {kind!r}")
    return normalized


__all__ = [
    "DEFAULT_CADENCE_DAYS",
    "MonitoringPlan",
    "PLAN_VERSION",
    "PLAN_SCHEMA_VERSION",
    "build_monitoring_plan",
    "canonical_economics_bindings_hash",
    "canonical_monitoring_plan_hash",
    "canonical_monitoring_plan_json",
    "load_monitoring_plan",
    "monitoring_plan_from_dict",
    "save_monitoring_plan",
]

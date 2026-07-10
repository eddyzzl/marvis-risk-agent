"""S5 Commit 1: tool_run_strategy_monitoring (strategy monitoring closure).

Covers the strategy-facing monitoring path -- no scoring model, so PSI/CSI are
skipped and only the approval-rate / approved-bad-rate drift-vs-baseline checks
run. Every drift value is hand-computed from row counts so the three-tier grading
(green/amber/red) is verified against exact numbers, not the tool's own math.

Setup builds a pure-rule strategy (`score < 500` -> reject), backtests it on a
baseline dataset to fix the expectation_baseline, adopts it (which writes the
monitoring plan), then runs monitoring on a fresh dataset whose approval / bad
rates are engineered to land in a chosen drift band.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.monitoring_plan import load_monitoring_plan
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.audit import _list_audit_rows
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="S5 策略监控",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
            score_col="score",
        )
    )
    return runner, registry, task, settings


def _register(registry, tmp_path, frame: pd.DataFrame, name: str, task_id: str):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def _baseline_frame() -> pd.DataFrame:
    # 20 rows, rule `score < 500` rejects the 4 lowest scores -> approval_rate=0.80.
    # Of the 16 approved (score>=500) exactly 1 is bad -> approved_bad_rate=0.0625.
    scores = list(range(100, 2100, 100))
    bad = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return pd.DataFrame({"score": scores, "bad": bad})


def _adopt_pure_rule_strategy(runner, registry, task, tmp_path):
    """Build -> backtest -> adopt a pure-rule strategy; return its id and the
    baseline backtest (approval_rate/approved_bad_rate become the plan baseline)."""
    baseline_ds = _register(registry, tmp_path, _baseline_frame(), "baseline", task.id)
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {
            "strategy_type": "approval",
            "rules": [{"condition": "score < 500", "decision": "reject"}],
            "score_col": "score",
            "default_decision": "approve",
        },
        task_id=task.id,
    )
    assert built.ok, built.error
    strategy_id = built.output["strategy_id"]

    bt = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {"dataset_id": baseline_ds.id, "strategy_id": strategy_id, "target_col": "bad"},
        task_id=task.id,
    )
    assert bt.ok, bt.error
    assert round(bt.output["approval_rate"], 4) == 0.8
    assert round(bt.output["approved_bad_rate"], 4) == 0.0625

    adopted = runner.invoke(
        ToolRef("strategy", "adopt_strategy"),
        {
            "strategy_id": strategy_id,
            "backtest_id": bt.output["backtest_id"],
            "adoption_reason": "committee sign-off",
        },
        task_id=task.id,
    )
    assert adopted.ok, adopted.error
    return strategy_id, bt.output


def _fresh_frame(*, n_reject: int, n_approve_good: int, n_approve_bad: int) -> pd.DataFrame:
    """Fresh monitoring sample under rule `score < 500`. Reject rows get score<500,
    approved rows get score>=500 split into good/bad by the `bad` column."""
    scores = []
    bad = []
    for _ in range(n_reject):
        scores.append(100)
        bad.append(1)
    for _ in range(n_approve_good):
        scores.append(900)
        bad.append(0)
    for _ in range(n_approve_bad):
        scores.append(900)
        bad.append(1)
    return pd.DataFrame({"score": scores, "bad": bad})


@pytest.mark.slow
def test_pure_rule_monitoring_green(tmp_path):
    runner, registry, task, _ = _runtime(tmp_path)
    strategy_id, _bt = _adopt_pure_rule_strategy(runner, registry, task, tmp_path)

    # 100 rows: 20 reject, 80 approved (5 bad) -> approval=0.80 (drift 0.00),
    # approved_bad_rate=5/80=0.0625 (drift 0.0000). Both within +-5pp -> green.
    fresh = _fresh_frame(n_reject=20, n_approve_good=75, n_approve_bad=5)
    ds = _register(registry, tmp_path, fresh, "fresh_green", task.id)
    res = runner.invoke(
        ToolRef("strategy", "run_strategy_monitoring"),
        {"strategy_id": strategy_id, "dataset_id": ds.id, "target_col": "bad"},
        task_id=task.id,
    )
    assert res.ok, res.error
    o = res.output
    assert o["experiment_id"] is None  # pure rule -> no model monitoring
    checks = {c["id"]: c for c in o["checks"]}
    # No PSI/CSI checks for a pure-rule strategy.
    assert "score_psi" not in checks
    assert set(checks) == {"approval_rate_drift", "approved_bad_rate_drift"}
    assert round(checks["approval_rate_drift"]["value"], 4) == 0.0
    assert checks["approval_rate_drift"]["level"] == "green"
    assert round(checks["approved_bad_rate_drift"]["value"], 4) == 0.0
    assert checks["approved_bad_rate_drift"]["level"] == "green"
    assert o["overall_level"] == "green"


@pytest.mark.slow
def test_pure_rule_monitoring_amber(tmp_path):
    runner, registry, task, _ = _runtime(tmp_path)
    strategy_id, _bt = _adopt_pure_rule_strategy(runner, registry, task, tmp_path)

    # 100 rows: 27 reject, 73 approved -> approval=0.73, drift 0.73-0.80=-0.07.
    # |0.07| in (0.05, 0.10] -> amber. Approved bad = 5/73=0.0685, drift +0.006 -> green.
    fresh = _fresh_frame(n_reject=27, n_approve_good=68, n_approve_bad=5)
    ds = _register(registry, tmp_path, fresh, "fresh_amber", task.id)
    res = runner.invoke(
        ToolRef("strategy", "run_strategy_monitoring"),
        {"strategy_id": strategy_id, "dataset_id": ds.id, "target_col": "bad"},
        task_id=task.id,
    )
    assert res.ok, res.error
    checks = {c["id"]: c for c in res.output["checks"]}
    assert round(checks["approval_rate_drift"]["value"], 4) == -0.07
    assert checks["approval_rate_drift"]["level"] == "amber"
    assert checks["approved_bad_rate_drift"]["level"] == "green"
    assert res.output["overall_level"] == "amber"


@pytest.mark.slow
def test_pure_rule_monitoring_red(tmp_path):
    runner, registry, task, _ = _runtime(tmp_path)
    strategy_id, _bt = _adopt_pure_rule_strategy(runner, registry, task, tmp_path)

    # 100 rows: 30 reject, 70 approved with 20 bad -> approval=0.70 (drift -0.10 -> amber),
    # approved_bad_rate=20/70=0.2857, drift 0.2857-0.0625=+0.2232 (>0.10) -> red.
    fresh = _fresh_frame(n_reject=30, n_approve_good=50, n_approve_bad=20)
    ds = _register(registry, tmp_path, fresh, "fresh_red", task.id)
    res = runner.invoke(
        ToolRef("strategy", "run_strategy_monitoring"),
        {"strategy_id": strategy_id, "dataset_id": ds.id, "target_col": "bad"},
        task_id=task.id,
    )
    assert res.ok, res.error
    o = res.output
    checks = {c["id"]: c for c in o["checks"]}
    assert round(checks["approval_rate_drift"]["value"], 4) == -0.10
    assert checks["approval_rate_drift"]["level"] == "amber"
    assert round(checks["approved_bad_rate_drift"]["value"], 4) == 0.2232
    assert checks["approved_bad_rate_drift"]["level"] == "red"
    assert o["overall_level"] == "red"
    assert any(f["id"] == "approved_bad_rate_drift" for f in o["red_flags"])


@pytest.mark.slow
def test_monitoring_no_label_is_na(tmp_path):
    runner, registry, task, _ = _runtime(tmp_path)
    strategy_id, _bt = _adopt_pure_rule_strategy(runner, registry, task, tmp_path)

    # Fresh sample with NO label column -> approved_bad_rate_drift is n/a, approval still graded.
    fresh = pd.DataFrame({"score": [100] * 20 + [900] * 80})
    ds = _register(registry, tmp_path, fresh, "fresh_nolabel", task.id)
    res = runner.invoke(
        ToolRef("strategy", "run_strategy_monitoring"),
        {"strategy_id": strategy_id, "dataset_id": ds.id},
        task_id=task.id,
    )
    assert res.ok, res.error
    checks = {c["id"]: c for c in res.output["checks"]}
    assert checks["approved_bad_rate_drift"]["level"] == "n/a"
    assert checks["approved_bad_rate_drift"]["value"] is None
    # approval 0.80 vs baseline 0.80 -> green; overall ignores n/a.
    assert checks["approval_rate_drift"]["level"] == "green"
    assert res.output["overall_level"] == "green"


@pytest.mark.slow
def test_monitoring_writes_back_last_run_at_and_audit(tmp_path):
    runner, registry, task, settings = _runtime(tmp_path)
    strategy_id, _bt = _adopt_pure_rule_strategy(runner, registry, task, tmp_path)

    strategies = StrategyRepository(settings.db_path)
    plan_path = Path(
        [a for a in strategies.list_strategy_artifacts(strategy_id)
         if a["kind"] == "monitoring_plan_json"][-1]["path"]
    )
    before = json.loads(plan_path.read_text(encoding="utf-8"))
    assert before["last_run_at"] is None

    fresh = _fresh_frame(n_reject=20, n_approve_good=75, n_approve_bad=5)
    ds = _register(registry, tmp_path, fresh, "fresh_wb", task.id)
    res = runner.invoke(
        ToolRef("strategy", "run_strategy_monitoring"),
        {"strategy_id": strategy_id, "dataset_id": ds.id, "target_col": "bad"},
        task_id=task.id,
    )
    assert res.ok, res.error

    # last_run_at written back (the only mutated field); rest unchanged.
    plan = load_monitoring_plan(plan_path)
    assert plan.last_run_at == res.output["last_run_at"]
    assert plan.last_run_at is not None
    after = json.loads(plan_path.read_text(encoding="utf-8"))
    assert after["expectation_baseline"] == before["expectation_baseline"]
    assert after["thresholds"] == before["thresholds"]

    # strategy.monitor audit row with overall_level.
    rows = _list_audit_rows(settings.db_path, kind="strategy.monitor", target_ref=strategy_id)
    assert len(rows) == 1
    assert rows[0]["detail"]["overall_level"] == "green"


@pytest.mark.slow
def test_monitoring_unadopted_strategy_typed_error(tmp_path):
    runner, registry, task, _ = _runtime(tmp_path)
    # Build (but do not adopt) a strategy.
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {
            "strategy_type": "approval",
            "rules": [{"condition": "score < 500", "decision": "reject"}],
            "score_col": "score",
            "default_decision": "approve",
        },
        task_id=task.id,
    )
    assert built.ok, built.error
    fresh = _fresh_frame(n_reject=20, n_approve_good=75, n_approve_bad=5)
    ds = _register(registry, tmp_path, fresh, "fresh_unadopted", task.id)
    res = runner.invoke(
        ToolRef("strategy", "run_strategy_monitoring"),
        {"strategy_id": built.output["strategy_id"], "dataset_id": ds.id, "target_col": "bad"},
        task_id=task.id,
    )
    assert res.ok is False
    assert res.error_kind == "strategy_not_adopted"


# ---------------------------------------------------------------------------
# S5 Commit 2: due derivation, disposition parsing, next_action, renderer.
# ---------------------------------------------------------------------------
def _adopt_with_plan(db_path, tmp_path, *, cadence_days, last_run_at, adopted_at="2026-01-01T00:00:00Z"):
    from marvis.packs.strategy.contracts import Strategy, StrategyRule
    from marvis.packs.strategy.monitoring_plan import build_monitoring_plan, save_monitoring_plan

    repo = StrategyRepository(db_path)
    strategy = Strategy(
        id=f"s-{cadence_days}-{last_run_at or 'none'}",
        strategy_type="approval",
        rules=(StrategyRule(condition="score < 500", decision="reject", value=None),),
        score_col="score",
        default_decision="approve",
        description="due-test",
    )
    repo.create_strategy("task-1", strategy, created_at=adopted_at)
    repo.adopt_strategy_with_audit(
        strategy.id,
        reason="seed",
        audit={"kind": "strategy.adopt", "target_ref": strategy.id, "outcome": "succeeded", "detail": {}},
        adopted_at=adopted_at,
    )
    plan = build_monitoring_plan(
        strategy_id=strategy.id, version=1, approved_bad_rate=0.05, approval_rate=0.8, cadence_days=cadence_days
    )
    plan["last_run_at"] = last_run_at
    plan_path = tmp_path / f"plan_{strategy.id}.json"
    save_monitoring_plan(plan_path, plan)
    repo.save_strategy_artifact(strategy.id, kind="monitoring_plan_json", path=str(plan_path))
    return strategy.id


def test_list_monitoring_due_uses_adopted_at_when_no_last_run(tmp_path):
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    # No last_run_at -> due is measured from adopted_at + cadence.
    sid = _adopt_with_plan(
        settings.db_path, tmp_path, cadence_days=30, last_run_at=None, adopted_at="2026-01-01T00:00:00Z"
    )
    now = datetime(2026, 3, 1, tzinfo=UTC)  # ~59 days after adoption, 30d cadence -> overdue ~29d
    due = StrategyRepository(settings.db_path).list_monitoring_due(now=now)
    assert [d["strategy_id"] for d in due] == [sid]
    assert due[0]["last_run_at"] is None
    assert round(due[0]["overdue_days"]) == 29


def test_list_monitoring_due_boundary_not_yet_due(tmp_path):
    from datetime import UTC, datetime, timedelta

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    last_run = datetime(2026, 6, 1, tzinfo=UTC)
    _adopt_with_plan(
        settings.db_path, tmp_path, cadence_days=30, last_run_at=last_run.isoformat()
    )
    # Exactly at due (last_run + 30d): overdue_seconds == 0 -> not returned.
    at_due = last_run + timedelta(days=30)
    assert StrategyRepository(settings.db_path).list_monitoring_due(now=at_due) == []
    # One day past due -> returned.
    past = at_due + timedelta(days=1)
    due = StrategyRepository(settings.db_path).list_monitoring_due(now=past)
    assert len(due) == 1
    assert round(due[0]["overdue_days"]) == 1


def test_list_monitoring_due_skips_non_adopted_and_planless(tmp_path):
    from datetime import UTC, datetime

    from marvis.packs.strategy.contracts import Strategy, StrategyRule

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = StrategyRepository(settings.db_path)
    # A draft strategy (never adopted) + an adopted one with no plan artifact.
    draft = Strategy(id="draft-1", strategy_type="approval",
                     rules=(StrategyRule(condition="score < 1", decision="reject", value=None),),
                     score_col="score", default_decision="approve", description="d")
    repo.create_strategy("task-1", draft, created_at="2026-01-01T00:00:00Z")
    adopted_noplan = Strategy(id="adopted-noplan", strategy_type="reject",
                              rules=(StrategyRule(condition="score < 1", decision="reject", value=None),),
                              score_col="score", default_decision="approve", description="d")
    repo.create_strategy("task-1", adopted_noplan, created_at="2026-01-01T00:00:00Z")
    repo.adopt_strategy_with_audit(
        adopted_noplan.id, reason="x",
        audit={"kind": "strategy.adopt", "target_ref": adopted_noplan.id, "outcome": "succeeded", "detail": {}},
        adopted_at="2026-01-01T00:00:00Z",
    )
    now = datetime(2027, 1, 1, tzinfo=UTC)
    assert StrategyRepository(settings.db_path).list_monitoring_due(now=now) == []


def test_list_monitoring_due_crosses_month_and_year_boundaries(tmp_path):
    """Due date is anchor + cadence via UTC timedelta arithmetic, so it crosses
    month/year boundaries correctly (Jan 20 + 30d = Feb 19)."""
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    sid = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=30,
        last_run_at="2026-01-20T00:00:00+00:00",
    )
    due = StrategyRepository(settings.db_path).list_monitoring_due(
        now=datetime(2026, 2, 25, tzinfo=UTC)
    )
    assert [d["strategy_id"] for d in due] == [sid]
    assert due[0]["due_at"] == "2026-02-19T00:00:00+00:00"
    assert round(due[0]["overdue_days"]) == 6


def test_list_monitoring_due_handles_leap_day(tmp_path):
    """Feb 28 (leap year) + 1d resolves to Feb 29, not Mar 1 -- calendar-aware
    UTC arithmetic, no manual day math."""
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=1,
        last_run_at="2028-02-28T00:00:00+00:00",
    )
    due = StrategyRepository(settings.db_path).list_monitoring_due(
        now=datetime(2028, 3, 1, tzinfo=UTC)
    )
    assert len(due) == 1
    assert due[0]["due_at"] == "2028-02-29T00:00:00+00:00"


def test_list_monitoring_due_is_dst_immune_because_timestamps_are_utc(tmp_path):
    """All timestamps are UTC (adopted_at/last_run_at via _now() = datetime.now
    (UTC); _parse_iso normalizes naive to UTC), and cadence is added as a UTC
    timedelta, so a 30-day cadence spanning a wall-clock DST transition is
    exactly 30*86400 seconds -- no ambiguity, no off-by-one-hour."""
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    # 2026-03-01 -> +30d spans the US spring-forward (2026-03-08).
    _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=30,
        last_run_at="2026-03-01T00:00:00+00:00",
    )
    repo = StrategyRepository(settings.db_path)
    # Exactly 30 days later to the second: still not due (boundary is > 0).
    assert repo.list_monitoring_due(now=datetime(2026, 3, 31, tzinfo=UTC)) == []
    due = repo.list_monitoring_due(now=datetime(2026, 3, 31, 0, 0, 1, tzinfo=UTC))
    assert len(due) == 1
    assert due[0]["due_at"] == "2026-03-31T00:00:00+00:00"


def test_list_monitoring_due_pins_cadence_zero_and_negative_semantics(tmp_path):
    """Pin the current cadence-edge behavior (no reject/clamp is introduced):
    cadence_days=0 falls back to the 30-day default (0 is falsy), and a negative
    cadence places the due date before the anchor so the strategy reads as
    perpetually overdue."""
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = StrategyRepository(settings.db_path)

    # cadence 0 -> defaults to 30: Jan 1 + 30d = Jan 31.
    sid_zero = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=0,
        last_run_at="2026-01-01T00:00:00+00:00",
    )
    due_zero = repo.list_monitoring_due(now=datetime(2026, 2, 15, tzinfo=UTC))
    entry_zero = next(d for d in due_zero if d["strategy_id"] == sid_zero)
    assert entry_zero["cadence_days"] == 30
    assert entry_zero["due_at"] == "2026-01-31T00:00:00+00:00"

    # negative cadence -> due before the anchor -> always overdue.
    sid_neg = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=-5,
        last_run_at="2026-06-01T00:00:00+00:00",
    )
    due_neg = repo.list_monitoring_due(now=datetime(2026, 6, 1, 12, tzinfo=UTC))
    entry_neg = next(d for d in due_neg if d["strategy_id"] == sid_neg)
    assert entry_neg["cadence_days"] == -5
    assert entry_neg["due_at"] == "2026-05-27T00:00:00+00:00"


def test_parse_monitoring_disposition_three_keywords():
    from marvis.agent.plan_driver import _parse_monitoring_disposition as parse

    assert parse("起新版本") == "new_version"
    assert parse("基于当前策略新版本重做") == "new_version"
    assert parse("new version please") == "new_version"
    assert parse("调阈值重跑") == "adjust_threshold"
    assert parse("adjust threshold and rerun") == "adjust_threshold"
    assert parse("维持并观察") == "observe"
    assert parse("先保持观察") == "observe"
    # More than one choice is ambiguous, even when one keyword is more specific.
    assert parse("先观察，不行就起新版本") is None
    # a plain confirm names no disposition.
    assert parse("确认") is None
    assert parse("") is None


@pytest.mark.parametrize(
    "text",
    [
        "raise the threshold to 3%",
        "stakeholder feedback",
        "this drift remains unobserved",
        "new versioning notes",
        "保持报告简洁，先解释下红灯",
        "阈值保持不变",
        "要观察吗？",
        "不要观察",
        "暂不调阈值",
        "不起新版本",
        "不建议调整阈值",
        "没有必要起新版本",
        "我不认为应该观察",
        "不考虑调阈值",
        "我不同意起新版本",
        "我反对调整阈值",
        "拒绝起新版本",
        "不赞成观察",
        "暂缓观察",
        "观察还是调阈值？",
    ],
)
def test_parse_monitoring_disposition_rejects_non_explicit_choices(text):
    from marvis.agent.plan_driver import _parse_monitoring_disposition as parse

    assert parse(text) is None


def test_monitoring_next_action_new_version_points_at_development():
    from marvis.packs.strategy.monitor_tools import monitoring_next_action

    action = monitoring_next_action("new_version", strategy_id="s-1")
    assert action is not None
    assert action["template_id"] == "strategy_development"
    assert action["parent_strategy_id"] == "s-1"
    assert "s-1" in action["prompt"]
    # observe / adjust are notes (no follow-up template); None disposition -> no action.
    assert monitoring_next_action("observe", strategy_id="s-1")["kind"] == "note"
    assert monitoring_next_action("adjust_threshold", strategy_id="s-1")["kind"] == "note"
    assert monitoring_next_action(None, strategy_id="s-1") is None


def test_render_run_strategy_monitoring_red_injects_checklist():
    from marvis.agent.renderers import render_tool_output

    text, tables = render_tool_output("run_strategy_monitoring", {
        "overall_level": "red",
        "checks": [
            {"id": "approved_bad_rate_drift", "label": "通过客群坏率漂移", "level": "red", "value": 0.22, "message": "x"},
            {"id": "approval_rate_drift", "label": "审批率漂移", "level": "green", "value": 0.0, "message": "y"},
        ],
    })
    assert "总体判级【红】" in text
    assert "起新版本" in text  # red-light checklist injected
    assert "维持并观察" in text
    assert "调阈值" in text
    assert tables[0]["columns"] == ["检查项", "判级", "值", "说明"]


def test_render_monitoring_report_surfaces_next_action():
    from marvis.agent.renderers import render_tool_output

    text, tables = render_tool_output("render_monitoring_report", {
        "report_path": "/w/tasks/t/strategy/monitoring_report_s1_v1.md",
        "overall_level": "red",
        "timeline": [{"at": "2026-07-01T00:00:00Z", "overall_level": "red", "row_count": 100}],
        "next_action": {"kind": "suggest_template", "prompt": "监控红灯，建议起新版本。"},
    })
    assert "监控报告已生成" in text
    assert "监控红灯，建议起新版本。" in text
    assert tables[0]["title"] == "监控判级时间线"


def test_monitoring_report_gate_declares_disposition_schema():
    """LT-3 (A.3): the render_monitoring_report gate adapter declares its `disposition`
    control as a JSON enum schema (observe/adjust_threshold/new_version), surfaced on
    the gate payload as editable_input_schema (the LT-4 frontend key)."""
    from marvis.agent.gates.adapters import gate_editable_input_schema
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.plugins.manifest import ToolRef

    gate = PlanStep(
        id="rep", plan_id="p", index=0, title="监控报告",
        tool_ref=ToolRef("strategy", "render_monitoring_report"),
        inputs={"disposition": None}, depends_on=[], post_checks=[],
    )
    plan = Plan(
        id="p", task_id="t", goal="g", source="template", template_id="strategy_monitoring",
        autonomy_level=1, steps=[gate], status=PlanStatus.AWAITING_CONFIRM,
    )
    schema = gate_editable_input_schema(plan, gate, lambda sid: None)
    disposition = schema["properties"]["disposition"]
    assert disposition["type"] == "string"
    assert disposition["enum"] == ["observe", "adjust_threshold", "new_version"]

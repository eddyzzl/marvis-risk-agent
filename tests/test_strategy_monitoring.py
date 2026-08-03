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
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    canonical_economics_bindings_hash,
    load_monitoring_plan,
)
from marvis.plugins.loader import load_manifest
from marvis.plugins.manifest import GovernancePolicy, ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_monitoring import StrategyMonitoringRepository
from marvis.repositories.audit import _list_audit_rows
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    _register_policy_neutral_strategy_pack(plugin_registry, packs_root)
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


def _register_policy_neutral_strategy_pack(plugin_registry, packs_root):
    """Register real monitoring kernels behind a policy-neutral test manifest."""
    manifest = load_manifest(packs_root / "strategy", builtin=True)
    neutral_manifest = replace(
        manifest,
        tools=tuple(
            replace(tool, policy=GovernancePolicy()) for tool in manifest.tools
        ),
    )
    plugin_registry.register(neutral_manifest, enabled=True)


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
def test_monitoring_keeps_plan_immutable_and_persists_run_and_audit(tmp_path):
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

    # The adopted plan artifact is immutable. Runtime timestamps and results live
    # in the append-only monitoring-run ledger instead of mutating the plan.
    plan = load_monitoring_plan(plan_path)
    assert plan.last_run_at is None
    after = json.loads(plan_path.read_text(encoding="utf-8"))
    assert after == before

    run = StrategyMonitoringRepository(settings.db_path).get_run(
        res.output["monitoring_run_id"]
    )
    assert run is not None
    assert run.created_at == res.output["last_run_at"]
    assert run.overall_level == "green"
    assert run.monitoring_plan_id == res.output["monitoring_plan_id"]

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


def _add_ledger_plan(
    db_path,
    strategy_id,
    *,
    cadence_days,
    created_at,
    plan_id,
):
    ledger = StrategyMonitoringRepository(db_path)
    latest = ledger.latest_plan(strategy_id)
    revision = 1 if latest is None else latest.revision + 1
    return ledger.create_plan(
        MonitoringPlan(
            strategy_id=strategy_id,
            version=1,
            cadence_days=cadence_days,
            monitoring_plan_id=plan_id,
            revision=revision,
            supersedes_plan_id=None if latest is None else latest.id,
        ),
        expected_revision=0 if latest is None else latest.revision,
        expected_payload_hash=None if latest is None else latest.payload_hash,
        plan_id=plan_id,
        created_at=created_at,
    )


def _add_ledger_run(
    db_path,
    strategy_id,
    plan,
    *,
    task_id,
    run_id,
    created_at,
):
    import hashlib

    dataset_hash = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    dataset = Dataset(
        id=f"dataset-{run_id}",
        task_id=task_id,
        role="strategy.monitoring",
        source_path=f"ledger/{run_id}.parquet",
        format="parquet",
        sheet=None,
        row_count=1,
        columns=(),
        has_target=False,
        target_col=None,
        created_at=created_at,
        content_hash=dataset_hash,
    )
    DatasetRepository(db_path).create_dataset(dataset)
    return StrategyMonitoringRepository(db_path).create_run(
        strategy_id=strategy_id,
        monitoring_plan_id=plan.id,
        expected_plan_revision=plan.revision,
        expected_plan_payload_hash=plan.payload_hash,
        dataset_id=dataset.id,
        dataset_content_hash=dataset_hash,
        strategy_effect_hash=hashlib.sha256(b"strategy-effect").hexdigest(),
        economics_binding_hash=canonical_economics_bindings_hash(
            plan.plan.economics_bindings
        ),
        result={
            "overall_level": "green",
            "checks": [{"id": "approval_rate", "level": "green"}],
        },
        overall_level="green",
        run_id=run_id,
        created_at=created_at,
    )


def test_list_monitoring_due_prefers_fresh_ledger_plan_over_stale_artifact(tmp_path):
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    sid = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=1,
        last_run_at="2026-01-01T00:00:00+00:00",
        adopted_at="2026-01-01T00:00:00+00:00",
    )
    _add_ledger_plan(
        settings.db_path,
        sid,
        cadence_days=30,
        created_at="2026-02-25T00:00:00+00:00",
        plan_id="ledger-plan-fresh",
    )
    repo = StrategyRepository(settings.db_path)

    # The old artifact is long overdue, but a newly-created immutable plan gets
    # its own full cadence before the first ledger run is due.
    assert repo.list_monitoring_due(now=datetime(2026, 3, 1, tzinfo=UTC)) == []
    due = repo.list_monitoring_due(now=datetime(2026, 3, 28, tzinfo=UTC))

    assert [item["strategy_id"] for item in due] == [sid]
    assert due[0]["due_at"] == "2026-03-27T00:00:00+00:00"
    assert due[0]["last_run_at"] is None
    assert due[0]["cadence_days"] == 30


def test_list_monitoring_due_no_run_anchors_at_later_of_plan_and_adoption(tmp_path):
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    sid = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=1,
        last_run_at=None,
        adopted_at="2026-03-01T00:00:00+00:00",
    )
    _add_ledger_plan(
        settings.db_path,
        sid,
        cadence_days=30,
        created_at="2026-02-01T00:00:00+00:00",
        plan_id="ledger-plan-backdated",
    )
    repo = StrategyRepository(settings.db_path)

    assert repo.list_monitoring_due(now=datetime(2026, 3, 15, tzinfo=UTC)) == []
    due = repo.list_monitoring_due(now=datetime(2026, 4, 1, tzinfo=UTC))
    assert due[0]["due_at"] == "2026-03-31T00:00:00+00:00"


def test_list_monitoring_due_uses_latest_run_for_current_ledger_plan(tmp_path):
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    sid = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=365,
        last_run_at="2026-12-31T00:00:00+00:00",
        adopted_at="2026-01-01T00:00:00+00:00",
    )
    plan = _add_ledger_plan(
        settings.db_path,
        sid,
        cadence_days=30,
        created_at="2026-01-01T00:00:00+00:00",
        plan_id="ledger-plan-runs",
    )
    _add_ledger_run(
        settings.db_path,
        sid,
        plan,
        task_id="task-1",
        run_id="run-old",
        created_at="2026-02-01T00:00:00+00:00",
    )
    _add_ledger_run(
        settings.db_path,
        sid,
        plan,
        task_id="task-1",
        run_id="run-latest",
        created_at="2026-02-10T00:00:00+00:00",
    )

    due = StrategyRepository(settings.db_path).list_monitoring_due(
        now=datetime(2026, 3, 13, tzinfo=UTC)
    )

    assert [item["strategy_id"] for item in due] == [sid]
    assert due[0]["last_run_at"] == "2026-02-10T00:00:00+00:00"
    assert due[0]["due_at"] == "2026-03-12T00:00:00+00:00"
    assert round(due[0]["overdue_days"]) == 1


def test_list_monitoring_due_does_not_reuse_run_from_superseded_plan(tmp_path):
    from datetime import UTC, datetime

    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    sid = _adopt_with_plan(
        settings.db_path,
        tmp_path,
        cadence_days=1,
        last_run_at="2026-01-01T00:00:00+00:00",
    )
    first = _add_ledger_plan(
        settings.db_path,
        sid,
        cadence_days=1,
        created_at="2026-02-01T00:00:00+00:00",
        plan_id="ledger-plan-1",
    )
    _add_ledger_run(
        settings.db_path,
        sid,
        first,
        task_id="task-1",
        run_id="run-plan-1",
        created_at="2026-02-10T00:00:00+00:00",
    )
    _add_ledger_plan(
        settings.db_path,
        sid,
        cadence_days=30,
        created_at="2026-03-01T00:00:00+00:00",
        plan_id="ledger-plan-2",
    )

    # The newest plan has no run, so its own creation timestamp is the anchor;
    # neither the old plan's run nor the mutable artifact can make it overdue.
    assert StrategyRepository(settings.db_path).list_monitoring_due(
        now=datetime(2026, 3, 15, tzinfo=UTC)
    ) == []


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
        adopted_noplan.id, reason="approved",
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
        "next_action": {
            "kind": "completed",
            "action": "new_version",
            "prompt": "新版本任务、策略和数据集均已创建。",
        },
    })
    assert "监控报告已生成" in text
    assert "新版本任务、策略和数据集均已创建。" in text
    assert tables[0]["title"] == "监控判级时间线"


def test_monitoring_disposition_gate_declares_real_action_schema():
    """The evidence-bound disposition gate exposes the action, reason, and patch."""
    from marvis.agent.gate_param_schema import gate_param_schema
    from marvis.agent.gates.adapters import gate_editable_input_schema
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.plugins.manifest import ToolRef

    gate = PlanStep(
        id="disposition", plan_id="p", index=0, title="处置监控结果",
        tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
        inputs={"disposition": None, "reason": None, "threshold_patch": None},
        depends_on=[], post_checks=[],
    )
    plan = Plan(
        id="p", task_id="t", goal="g", source="template", template_id="strategy_monitoring",
        autonomy_level=1, steps=[gate], status=PlanStatus.AWAITING_CONFIRM,
    )
    schema = gate_editable_input_schema(plan, gate, lambda sid: None)
    disposition = schema["properties"]["disposition"]
    assert disposition["type"] == "string"
    assert disposition["enum"] == ["observe", "adjust_threshold", "new_version"]
    assert schema["properties"]["reason"]["type"] == "string"
    assert schema["properties"]["threshold_patch"]["type"] == "object"
    routed = gate_param_schema(plan, gate, editable_input_schema=schema)
    assert [item["name"] for item in routed] == [
        "disposition",
        "reason",
        "threshold_patch",
    ]
    assert all("expected_plan" not in item["name"] for item in routed)


@pytest.mark.parametrize(
    ("overall_level", "requires_structured_input"),
    (("green", False), ("amber", False), ("red", True)),
)
def test_monitoring_gate_message_carries_authoritative_input_requirement(
    overall_level,
    requires_structured_input,
):
    from marvis.agent.plan_message_composer import PlanMessageComposer
    from marvis.orchestrator.contracts import (
        Plan,
        PlanStatus,
        PlanStep,
        StepStatus,
    )

    run = PlanStep(
        id="run",
        plan_id="p",
        index=0,
        title="执行策略监控",
        tool_ref=ToolRef("strategy", "run_strategy_monitoring"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    gate = PlanStep(
        id="disposition",
        plan_id="p",
        index=1,
        title="处置监控结果",
        tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
        inputs={"disposition": None, "reason": None, "threshold_patch": None},
        depends_on=[run.id],
        post_checks=[],
        status=StepStatus.AWAITING_CONFIRM,
    )
    plan = Plan(
        id="p",
        task_id="t",
        goal="g",
        source="template",
        template_id="strategy_monitoring",
        autonomy_level=1,
        steps=[run, gate],
        status=PlanStatus.AWAITING_CONFIRM,
    )
    output = {
        "overall_level": overall_level,
        "checks": [],
        "adjustable_threshold_ids": [],
    }

    message = PlanMessageComposer(
        load_output=lambda step_id: output if step_id == run.id else None,
    ).gate_message(plan, gate, run_seq=1)

    assert message.metadata["monitoring_disposition"] == {
        "overall_level": overall_level,
        "requires_structured_input": requires_structured_input,
    }


@pytest.mark.parametrize(
    ("adjustable_ids", "expected_ids"),
    (
        (["approval_floor"], ["approval_floor"]),
        (["score_psi"], ["score_psi"]),
    ),
)
def test_monitoring_disposition_gate_uses_plan_threshold_ids_only(
    adjustable_ids,
    expected_ids,
):
    """Display metric aliases and non-adjustable checks cannot become patch keys."""
    from marvis.agent.gates.adapters import gate_editable_input_schema
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.plugins.manifest import ToolRef

    run = PlanStep(
        id="run",
        plan_id="p",
        index=0,
        title="执行策略监控",
        tool_ref=ToolRef("strategy", "run_strategy_monitoring"),
        inputs={},
        depends_on=[],
        post_checks=[],
    )
    gate = PlanStep(
        id="disposition",
        plan_id="p",
        index=1,
        title="处置监控结果",
        tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
        inputs={"disposition": None, "reason": None, "threshold_patch": None},
        depends_on=[run.id],
        post_checks=[],
    )
    plan = Plan(
        id="p",
        task_id="t",
        goal="g",
        source="template",
        template_id="strategy_monitoring",
        autonomy_level=1,
        steps=[run, gate],
        status=PlanStatus.AWAITING_CONFIRM,
    )
    output = {
        "overall_level": "red",
        "adjustable_threshold_ids": adjustable_ids,
        "checks": [
            {
                "id": "approval_floor",
                "metric": "approval_rate",
                "level": "red",
            },
            {
                "id": "feature_csi:age",
                "metric": "feature_csi",
                "level": "red",
            },
        ],
    }
    schema = gate_editable_input_schema(
        plan,
        gate,
        lambda step_id: output if step_id == run.id else None,
    )

    patch_schema = schema["properties"]["threshold_patch"]
    assert patch_schema["propertyNames"]["enum"] == expected_ids
    assert "approval_rate" not in patch_schema["propertyNames"]["enum"]
    assert "feature_csi" not in patch_schema["propertyNames"]["enum"]


def test_monitoring_disposition_gate_fails_closed_without_threshold_receipt():
    from marvis.agent.gates.adapters import gate_editable_input_schema
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.plugins.manifest import ToolRef

    run = PlanStep(
        id="run",
        plan_id="p",
        index=0,
        title="执行策略监控",
        tool_ref=ToolRef("strategy", "run_strategy_monitoring"),
        inputs={},
        depends_on=[],
        post_checks=[],
    )
    gate = PlanStep(
        id="disposition",
        plan_id="p",
        index=1,
        title="处置监控结果",
        tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
        inputs={},
        depends_on=[run.id],
        post_checks=[],
    )
    plan = Plan(
        id="p",
        task_id="t",
        goal="g",
        source="template",
        template_id="strategy_monitoring",
        autonomy_level=1,
        steps=[run, gate],
        status=PlanStatus.AWAITING_CONFIRM,
    )
    schema = gate_editable_input_schema(
        plan,
        gate,
        lambda _step_id: {"overall_level": "red", "checks": []},
    )

    assert schema["properties"]["threshold_patch"]["maxProperties"] == 0


def test_red_monitoring_gate_rejects_plain_confirm_without_complete_disposition():
    from marvis.agent.gates.adapters import monitoring_plain_confirm_error
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.plugins.manifest import ToolRef

    run = PlanStep(
        id="run",
        plan_id="p",
        index=0,
        title="执行策略监控",
        tool_ref=ToolRef("strategy", "run_strategy_monitoring"),
        inputs={},
        depends_on=[],
        post_checks=[],
    )
    gate = PlanStep(
        id="disposition",
        plan_id="p",
        index=1,
        title="处置监控结果",
        tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
        inputs={"disposition": None, "threshold_patch": None},
        depends_on=[run.id],
        post_checks=[],
    )
    plan = Plan(
        id="p",
        task_id="t",
        goal="g",
        source="template",
        template_id="strategy_monitoring",
        autonomy_level=1,
        steps=[run, gate],
        status=PlanStatus.AWAITING_CONFIRM,
    )
    red_output = {"overall_level": "red", "checks": []}

    def load_red(step_id):
        return red_output if step_id == run.id else None

    assert "不能只回复" in monitoring_plain_confirm_error(plan, gate, load_red)
    gate.inputs["disposition"] = "adjust_threshold"
    assert "还没有具体" in monitoring_plain_confirm_error(plan, gate, load_red)
    gate.inputs["threshold_patch"] = {"approval_rate": {"warn": 0.6}}
    assert monitoring_plain_confirm_error(plan, gate, load_red) is None
    gate.inputs = {"disposition": None, "threshold_patch": None}

    def load_green(step_id):
        return {"overall_level": "green"} if step_id == run.id else None

    assert monitoring_plain_confirm_error(plan, gate, load_green) is None

    def load_amber(step_id):
        return {"overall_level": "amber"} if step_id == run.id else None

    assert monitoring_plain_confirm_error(plan, gate, load_amber) is None

    def load_missing(_step_id):
        return None

    assert "缺少可信" in monitoring_plain_confirm_error(plan, gate, load_missing)

    def load_unknown(step_id):
        return {"overall_level": "blue"} if step_id == run.id else None

    assert "缺少可信" in monitoring_plain_confirm_error(plan, gate, load_unknown)


def test_monitoring_structured_control_cannot_rebind_frozen_evidence():
    from marvis.agent.gate_response_adapter import (
        GateControlValidationError,
        validate_gate_control,
    )
    from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
    from marvis.plugins.manifest import ToolRef

    gate = PlanStep(
        id="disposition",
        plan_id="p",
        index=0,
        title="处置监控结果",
        tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
        inputs={
            "expected_plan_id": "plan-1",
            "disposition": None,
            "reason": None,
            "threshold_patch": None,
        },
        depends_on=[],
        post_checks=[],
    )
    plan = Plan(
        id="p",
        task_id="t",
        goal="g",
        source="template",
        template_id="strategy_monitoring",
        autonomy_level=1,
        steps=[gate],
        status=PlanStatus.AWAITING_CONFIRM,
    )

    with pytest.raises(GateControlValidationError, match="不可修改冻结"):
        validate_gate_control(
            plan,
            gate,
            expected_step_id=gate.id,
            selection=None,
            dedup_strategies=None,
            adjust_params={"expected_plan_id": "attacker-plan"},
        )

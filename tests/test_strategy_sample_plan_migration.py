"""Fail-closed migration guard for serialized strategy plans."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marvis.agent.driver_turn import DriverTurn
from marvis.agent.turn_handlers import (
    _STRATEGY_SPEC,
    _TurnHandlerSpec,
    _run_driver_turn,
    _stale_strategy_sample_steps,
    run_strategy_driver_turn,
)
from marvis.db import PlanRepository, TaskRepository, init_db
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS, TASK_TYPE_STRATEGY, TaskCreate
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef


SAMPLE_DESIGN_REF = {
    "artifact_id": "a" * 64,
    "artifact_content_hash": "b" * 64,
    "sample_design_id": "strategy-sample-design-1",
    "sample_design_content_hash": "c" * 64,
    "partition": "development",
}


def _repositories(tmp_path, *, task_type: str = TASK_TYPE_STRATEGY):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = task_repo.create_task(
        TaskCreate(
            model_name="serialized-plan-migration",
            model_version="v2",
            validator="qa",
            source_dir=str(tmp_path),
            task_type=task_type,
        )
    )
    return task_repo, PlanRepository(db_path), task


def _plan(
    task_id: str,
    *,
    template_id: str = "strategy_development",
    plan_status: PlanStatus = PlanStatus.VALIDATED,
    step_status: StepStatus = StepStatus.PENDING,
    tool: str = "backtest_strategy",
    sample_design_ref: object = None,
) -> Plan:
    inputs = {"dataset_id": "dataset-1", "target_col": "bad"}
    if sample_design_ref is not None:
        inputs["sample_design_ref"] = sample_design_ref
    return Plan(
        id=f"plan-{template_id}-{plan_status.value}",
        task_id=task_id,
        goal="continue serialized strategy plan",
        source="builtin",
        template_id=template_id,
        steps=[
            PlanStep(
                id=f"step-{template_id}",
                plan_id=f"plan-{template_id}-{plan_status.value}",
                index=0,
                title="sample-bound analysis",
                tool_ref=ToolRef("strategy", tool),
                inputs=inputs,
                depends_on=[],
                post_checks=[],
                status=step_status,
            )
        ],
        autonomy_level=1,
        status=plan_status,
    )


@pytest.mark.parametrize(
    "template_id",
    [
        "strategy_analysis",
        "strategy_development",
        "rule_strategy",
        "deterministic_strategy_candidate_development",
        "typed_strategy_evaluation",
        "stored_strategy_evaluation",
        "stored_strategy_adoption",
    ],
)
def test_old_active_backtest_plans_fail_closed_before_driver_resume(
    tmp_path,
    monkeypatch,
    template_id,
):
    task_repo, plan_repo, task = _repositories(tmp_path)
    plan = _plan(task.id, template_id=template_id)
    plan_repo.create_plan(plan)

    def _driver_must_not_be_built(_runtime):
        raise AssertionError("driver must not be built for a stale serialized plan")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._driver",
        _driver_must_not_be_built,
    )

    response = run_strategy_driver_turn(
        SimpleNamespace(plan_repo=plan_repo),
        task_repo,
        task,
        user_text="继续",
    )

    assert response["status"] == "clarification_required"
    assert response["code"] == "strategy_plan_sample_design_stale"
    assert response["plan_status"] == PlanStatus.FAILED.value
    assert plan_repo.load_plan(plan.id).status == PlanStatus.FAILED
    message = response["messages"][-1]
    assert message["metadata"]["code"] == "strategy_plan_sample_design_stale"
    assert "成熟样本设计" in message["content"]
    audits = plan_repo.list_audit(kind="strategy.plan.sample_design_stale")
    assert len(audits) == 1
    assert audits[0]["outcome"] == "blocked"
    assert audits[0]["detail"]["stale_steps"] == [
        {
            "step_id": f"step-{template_id}",
            "tool": "backtest_strategy",
            "step_status": "pending",
        }
    ]


@pytest.mark.parametrize(
    "tool",
    [
        "analyze_univariate_candidates",
        "build_automatic_tree_candidate",
        "compare_strategies",
        "design_cutoff_bands",
        "design_strategy_candidate",
        "evaluate_rule_set",
        "limit_pricing_matrix",
        "measure_pool_impact",
        "mine_rules",
        "tradeoff_view",
    ],
)
def test_old_active_candidate_read_steps_also_require_exact_sample_ref(
    tmp_path,
    monkeypatch,
    tool,
):
    task_repo, plan_repo, task = _repositories(tmp_path)
    plan = _plan(
        task.id,
        template_id=f"legacy-{tool}",
        tool=tool,
    )
    plan_repo.create_plan(plan)
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._driver",
        lambda _runtime: pytest.fail("stale plan reached the driver"),
    )

    response = run_strategy_driver_turn(
        SimpleNamespace(plan_repo=plan_repo),
        task_repo,
        task,
        user_text="继续",
    )

    assert response["code"] == "strategy_plan_sample_design_stale"
    assert plan_repo.load_plan(plan.id).status == PlanStatus.FAILED


def test_awaiting_confirmation_stale_plan_uses_legal_cancel_transition(
    tmp_path,
    monkeypatch,
):
    task_repo, plan_repo, task = _repositories(tmp_path)
    plan = _plan(task.id, plan_status=PlanStatus.AWAITING_CONFIRM)
    plan_repo.create_plan(plan)
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._driver",
        lambda _runtime: pytest.fail("stale plan reached the driver"),
    )

    response = run_strategy_driver_turn(
        SimpleNamespace(plan_repo=plan_repo),
        task_repo,
        task,
        user_text="确认",
    )

    assert response["plan_status"] == PlanStatus.CANCELLED.value
    assert plan_repo.load_plan(plan.id).status == PlanStatus.CANCELLED
    status_audits = plan_repo.list_audit(kind="plan.status")
    assert status_audits[-1]["detail"] == {
        "from": PlanStatus.AWAITING_CONFIRM.value,
        "to": PlanStatus.CANCELLED.value,
    }


@pytest.mark.parametrize(
    "invalid_ref",
    [
        {},
        {**SAMPLE_DESIGN_REF, "partition": "oot"},
        {**SAMPLE_DESIGN_REF, "unexpected": "field"},
        {**SAMPLE_DESIGN_REF, "artifact_content_hash": "not-a-hash"},
    ],
)
def test_partial_or_malformed_sample_reference_is_stale(invalid_ref):
    plan = _plan("task-1", sample_design_ref=invalid_ref)

    assert [step.id for step in _stale_strategy_sample_steps(plan)] == [
        "step-strategy_development"
    ]


def test_new_exactly_bound_plan_resumes_normally(tmp_path, monkeypatch):
    task_repo, plan_repo, task = _repositories(tmp_path)
    plan = _plan(task.id, sample_design_ref=dict(SAMPLE_DESIGN_REF))
    plan_repo.create_plan(plan)
    calls = []

    class _Driver:
        def resume(self, **kwargs):
            calls.append(kwargs)
            return DriverTurn(plan_id=plan.id, status=plan.status.value)

    monkeypatch.setattr("marvis.agent.turn_handlers._driver", lambda _runtime: _Driver())

    response = run_strategy_driver_turn(
        SimpleNamespace(plan_repo=plan_repo, settings=None),
        task_repo,
        task,
        user_text="继续",
    )

    assert response["status"] == "ok"
    assert calls == [
        {
            "plan_id": plan.id,
            "user_text": "继续",
            "selection": None,
            "dedup_strategies": None,
            "adjust_params": None,
            "expected_step_id": None,
            "confirmation_source": "human",
        }
    ]
    assert plan_repo.list_audit(kind="strategy.plan.sample_design_stale") == []


def test_completed_sample_step_and_completed_plan_are_not_migration_stale():
    completed_step = _plan(
        "task-1",
        step_status=StepStatus.DONE,
    )
    completed_plan = _plan("task-1", plan_status=PlanStatus.DONE)

    assert _stale_strategy_sample_steps(completed_step) == []
    assert _stale_strategy_sample_steps(completed_plan) == []


def test_non_strategy_turn_is_not_subject_to_strategy_plan_migration_guard(
    tmp_path,
    monkeypatch,
):
    task_repo, plan_repo, task = _repositories(
        tmp_path,
        task_type=TASK_TYPE_FEATURE_ANALYSIS,
    )
    plan = _plan(task.id, template_id="feature_analysis")
    plan_repo.create_plan(plan)
    calls = []

    class _Driver:
        def resume(self, **kwargs):
            calls.append(kwargs)
            return DriverTurn(plan_id=plan.id, status=plan.status.value)

    monkeypatch.setattr("marvis.agent.turn_handlers._driver", lambda _runtime: _Driver())
    feature_spec = _TurnHandlerSpec(
        intent="feature",
        setup_error_types=(),
        error_label="feature",
        run_setup=lambda *_args: pytest.fail("active plan should resume"),
        format_user_display=lambda value: value,
        pass_memory_kwargs=False,
    )

    response = _run_driver_turn(
        feature_spec,
        SimpleNamespace(plan_repo=plan_repo),
        task_repo,
        task,
        user_text="继续",
        selection=None,
        dedup_strategies=None,
        adjust_params=None,
        expected_step_id=None,
        expected_plan_id=None,
        confirmation_source="human",
    )

    assert response["status"] == "ok"
    assert calls
    assert plan_repo.load_plan(plan.id).status == PlanStatus.VALIDATED
    assert plan_repo.list_audit(kind="strategy.plan.sample_design_stale") == []


def test_strategy_spec_identity_remains_strategy():
    assert _STRATEGY_SPEC.intent == "strategy"

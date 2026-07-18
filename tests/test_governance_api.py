import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import PlanRepository, StrategyRepository, TaskRepository, connect
from marvis.domain import TaskCreate
from marvis.governance.errors import ApprovalBindingError
from marvis.governance.service import _governance_payload_hash
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.plugins.manifest import (
    EffectTargetPolicy,
    GovernancePolicy,
    ToolRef,
)


class _NoopExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, plan_id: str):
        self.calls.append(plan_id)
        return type("Result", (), {"status": PlanStatus.AWAITING_CONFIRM})()


def _seed_protected_plan(app) -> tuple[str, str]:
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="strategy",
            model_version="v1",
            validator="owner",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )
    strategy = Strategy(
        id="strategy-1",
        strategy_type="approval",
        rules=(
            StrategyRule(
                condition="score >= 700",
                decision="decline",
                value=None,
            ),
        ),
        score_col="score",
        default_decision={"action": "approve"},
        description="candidate",
    )
    StrategyRepository(app.state.settings.db_path).create_strategy(task.id, strategy)
    policy = GovernancePolicy(
        human_decision_gate="required",
        effect_authorization="required",
        effect_target=EffectTargetPolicy(
            kind="strategy",
            id_input="strategy_id",
            expected_statuses=("draft",),
            result_status="adopted",
        ),
    )
    plan = Plan(
        id="plan-1",
        task_id=task.id,
        goal="adopt candidate",
        source="template",
        template_id="strategy_development",
        steps=[
            PlanStep(
                id="step-evidence",
                plan_id="plan-1",
                index=0,
                title="Evidence",
                tool_ref=ToolRef("strategy", "backtest_strategy"),
                inputs={},
                depends_on=[],
                post_checks=[],
                status=StepStatus.DONE,
            ),
            PlanStep(
                id="step-adopt",
                plan_id="plan-1",
                index=1,
                title="Adopt",
                tool_ref=ToolRef("strategy", "adopt_strategy"),
                inputs={
                    "strategy_id": strategy.id,
                    "backtest_id": "backtest-1",
                    "adoption_reason": "Reviewed backtest and operating constraints",
                },
                depends_on=["step-evidence"],
                post_checks=[],
                needs_confirmation=True,
                policy=policy,
                status=StepStatus.AWAITING_CONFIRM,
            )
        ],
        autonomy_level=1,
        status=PlanStatus.AWAITING_CONFIRM,
    )
    plans = PlanRepository(app.state.settings.db_path)
    plans.create_plan(plan)
    plans.store_step_output(
        "step-evidence",
        {"backtest_id": "backtest-1"},
        evidence={"dataset_hash": "dataset-v1", "metric_hash": "metrics-v1"},
    )
    return task.id, strategy.id


def _seed_human_only_plan(app) -> str:
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="decision",
            model_version="v1",
            validator="owner",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )
    plan = Plan(
        id="plan-human",
        task_id=task.id,
        goal="review candidate",
        source="template",
        template_id="human_only_test",
        steps=[
            PlanStep(
                id="step-human",
                plan_id="plan-human",
                index=0,
                title="Review candidate",
                tool_ref=ToolRef("_sample", "echo"),
                inputs={"message": "approved candidate"},
                depends_on=[],
                post_checks=[],
                needs_confirmation=True,
                policy=GovernancePolicy(human_decision_gate="required"),
                status=StepStatus.AWAITING_CONFIRM,
            )
        ],
        autonomy_level=1,
        status=PlanStatus.AWAITING_CONFIRM,
    )
    PlanRepository(app.state.settings.db_path).create_plan(plan)
    return task.id


def test_local_session_principal_cookie_is_server_issued_and_http_only(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/health")

    cookie = response.headers.get("set-cookie", "")
    assert "marvis_local_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    with connect(app.state.settings.db_path) as conn:
        row = conn.execute(
            "SELECT session_token_hash, status FROM local_principals"
        ).fetchone()
    assert row is not None
    assert row["status"] == "active"
    assert client.cookies.get("marvis_local_session") not in row["session_token_hash"]


def test_required_effect_step_uses_dedicated_human_decision_endpoint(tmp_path):
    app = create_app(tmp_path)
    app.state.plan_executor = _NoopExecutor()
    client = TestClient(app)
    task_id, _strategy_id = _seed_protected_plan(app)

    ordinary = client.post("/api/plans/plan-1/steps/step-adopt/confirm")
    authorized = client.post(
        "/api/plans/plan-1/steps/step-adopt/decisions",
        json={
            "decision": "approve",
            "reason": "I reviewed the backtest and authorize local adoption",
            "expected_plan_revision": 0,
        },
    )

    assert ordinary.status_code == 409
    assert "dedicated human decision" in ordinary.json()["detail"]
    assert authorized.status_code == 202, authorized.text
    payload = authorized.json()
    assert payload["decision_id"]
    assert payload["approval_id"]
    assert payload["principal_id"]
    assert payload["plan_id"] == "plan-1"
    assert payload["step_id"] == "step-adopt"
    assert payload["task_id"] == task_id
    assert app.state.plan_repo.is_step_confirmed("step-adopt") is True
    adopted_step = next(
        step
        for step in app.state.plan_repo.load_plan("plan-1").steps
        if step.id == "step-adopt"
    )
    assert adopted_step.inputs["adoption_reason"] == (
        "I reviewed the backtest and authorize local adoption"
    )
    assert app.state.plan_executor.calls == ["plan-1"]
    with connect(app.state.settings.db_path) as conn:
        decision = conn.execute("SELECT * FROM decision_records").fetchone()
        approval = conn.execute("SELECT * FROM approval_records").fetchone()
    assert decision["principal_id"] == payload["principal_id"]
    assert approval["principal_id"] == payload["principal_id"]
    assert approval["status"] == "issued"


def test_decision_binding_canonicalizes_nonfinite_resolved_band_evidence(tmp_path):
    app = create_app(tmp_path)
    app.state.plan_executor = _NoopExecutor()
    client = TestClient(app)
    _seed_protected_plan(app)
    app.state.plan_repo.store_step_output(
        "step-evidence",
        {
            "backtest_id": "backtest-1",
            "band_edges": [float("-inf"), 700.0, float("inf")],
        },
        evidence={"dataset_hash": "dataset-v1", "metric_hash": "metrics-v1"},
    )
    plan = app.state.plan_repo.load_plan("plan-1")
    step = next(item for item in plan.steps if item.id == "step-adopt")
    step.inputs["band_stats"] = "$ref:step-evidence.output"
    app.state.plan_repo.update_step(step)

    response = client.post(
        "/api/plans/plan-1/steps/step-adopt/decisions",
        json={
            "decision": "approve",
            "reason": "I reviewed the open-ended score bands",
            "expected_plan_revision": 0,
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["approval_id"]
    assert app.state.plan_repo.is_step_confirmed("step-adopt") is True


def test_nonfinite_float_canonical_tags_cannot_collide_with_user_payloads():
    negative_infinity = _governance_payload_hash({"value": float("-inf")})
    lookalikes = (
        {"value": "-inf"},
        {"value": ["float", "-inf"]},
        {"value": {"type": "float", "value": "-inf"}},
    )

    assert all(
        negative_infinity != _governance_payload_hash(payload)
        for payload in lookalikes
    )
    assert negative_infinity != _governance_payload_hash({"value": float("inf")})
    assert negative_infinity != _governance_payload_hash({"value": float("nan")})


def test_human_only_decision_executes_through_live_binding_without_effect_approval(
    tmp_path,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    _seed_human_only_plan(app)

    response = client.post(
        "/api/plans/plan-human/steps/step-human/decisions",
        json={
            "decision": "approve",
            "reason": "I reviewed and selected this candidate",
            "expected_plan_revision": 0,
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["decision_id"]
    assert response.json()["approval_id"] is None
    plan = app.state.plan_repo.load_plan("plan-human")
    assert plan.status == PlanStatus.DONE
    assert plan.steps[0].status == StepStatus.DONE
    assert app.state.plan_repo.load_step_output("step-human") == {
        "echoed": "approved candidate"
    }


def test_app_injects_governance_service_as_main_and_restricted_authorizer(tmp_path):
    app = create_app(tmp_path)

    assert app.state.plan_executor._authorizer is app.state.governance_service
    restricted = app.state.subagent_dispatcher._executor_factory(
        app.state.tool_registry
    )
    assert restricted._authorizer is app.state.governance_service


def test_decision_endpoint_rejects_client_supplied_actor_and_stale_revision(tmp_path):
    app = create_app(tmp_path)
    app.state.plan_executor = _NoopExecutor()
    client = TestClient(app)
    _seed_protected_plan(app)

    spoofed = client.post(
        "/api/plans/plan-1/steps/step-adopt/decisions",
        json={
            "decision": "approve",
            "reason": "spoof",
            "expected_plan_revision": 0,
            "principal_id": "attacker-controlled",
        },
    )
    stale = client.post(
        "/api/plans/plan-1/steps/step-adopt/decisions",
        json={
            "decision": "approve",
            "reason": "stale revision",
            "expected_plan_revision": 99,
        },
    )

    assert spoofed.status_code == 422
    assert stale.status_code == 409
    assert app.state.plan_repo.is_step_confirmed("step-adopt") is False


@pytest.mark.parametrize(
    "drift",
    ["inputs", "evidence", "revision", "target", "strategy_spec"],
)
def test_issued_effect_approval_is_fenced_by_live_binding_drift(tmp_path, drift):
    app = create_app(tmp_path)
    app.state.plan_executor = _NoopExecutor()
    client = TestClient(app)
    _seed_protected_plan(app)
    approved = client.post(
        "/api/plans/plan-1/steps/step-adopt/decisions",
        json={
            "decision": "approve",
            "reason": "approve the frozen evidence and target snapshot",
            "expected_plan_revision": 0,
        },
    )
    assert approved.status_code == 202, approved.text
    approved_plan = app.state.plan_repo.load_plan("plan-1")
    approved_step = next(
        item for item in approved_plan.steps if item.id == "step-adopt"
    )
    context = app.state.governance_service.execution_context_for(
        plan=approved_plan,
        step=approved_step,
        inputs=approved_step.inputs,
    )
    assert context is not None

    if drift == "inputs":
        plan = app.state.plan_repo.load_plan("plan-1")
        step = next(item for item in plan.steps if item.id == "step-adopt")
        step.inputs = {**step.inputs, "adoption_reason": "changed after approval"}
        app.state.plan_repo.update_step(step)
    elif drift == "evidence":
        app.state.plan_repo.store_step_output(
            "step-evidence",
            {"backtest_id": "backtest-2"},
            evidence={"dataset_hash": "dataset-v2", "metric_hash": "metrics-v2"},
        )
    elif drift == "revision":
        with connect(app.state.settings.db_path) as conn:
            conn.execute("UPDATE plans SET replan_count = 1 WHERE id = 'plan-1'")
    elif drift == "target":
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE strategies SET status = 'adopted' WHERE id = 'strategy-1'"
            )
    else:
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE strategies SET description = 'changed after approval' "
                "WHERE id = 'strategy-1'"
            )

    plan = app.state.plan_repo.load_plan("plan-1")
    step = next(item for item in plan.steps if item.id == "step-adopt")
    manifest, tool = app.state.tool_registry.resolve_with_manifest(step.tool_ref)
    if drift in {"revision", "target"}:
        with pytest.raises(ApprovalBindingError):
            app.state.governance_service.resolve_binding(
                task_id=plan.task_id,
                ref=step.tool_ref,
                inputs=step.inputs,
                execution_context=context,
                manifest=manifest,
                tool=tool,
            )
        return

    live_binding = app.state.governance_service.resolve_binding(
        task_id=plan.task_id,
        ref=step.tool_ref,
        inputs=step.inputs,
        execution_context=context,
        manifest=manifest,
        tool=tool,
    )
    with pytest.raises(ApprovalBindingError):
        app.state.governance_repo.reserve_effect(context, live_binding)

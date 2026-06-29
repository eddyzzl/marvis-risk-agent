from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import PlanRepository, TaskRepository, connect, init_db
from marvis.domain import TaskCreate
from marvis.orchestrator.contracts import (
    AgentStatus,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
    SubAgent,
)
from marvis.routers.plans import router
from marvis.plugins.manifest import ToolRef


class FakeIntentRouter:
    def __init__(self, kind="template"):
        self.kind = kind

    def route(self, _goal, task_context):
        return SimpleNamespace(
            kind=self.kind,
            template_id="sample_echo" if self.kind == "template" else None,
            slots={"message": task_context.get("message", "hi")},
        )


class FakePlanner:
    def __init__(self):
        self.generated = []
        self.from_template_calls = []

    def from_template(self, template, slots, task_id, *, autonomy=None):
        self.from_template_calls.append((template.id, slots, task_id, autonomy))
        return _plan(task_id=task_id, autonomy=autonomy or 1)

    def generate(
        self,
        goal,
        task_id,
        *,
        memory_context,
        task_context,
        tier=None,
        novel_mode="plan_ahead",
    ):
        self.generated.append((goal, task_id, memory_context, task_context, tier, novel_mode))
        return _plan(task_id=task_id, source="generated", template_id=None)


class FakeValidator:
    def __init__(self, problems=None):
        self.problems = problems or []

    def validate(self, _plan):
        return list(self.problems)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def run(self, plan_id):
        self.calls.append(plan_id)
        return SimpleNamespace(status=PlanStatus.DONE)


class FakeHookDispatcher:
    def __init__(self):
        self.calls = []

    def dispatch(self, event, payload, *, task_id):
        self.calls.append((event, payload, task_id))
        return []


def _client(tmp_path, *, validator=None, intent=None):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    app = FastAPI()
    app.include_router(router)
    app.state.plan_repo = PlanRepository(db_path)
    app.state.intent_router = intent or FakeIntentRouter()
    app.state.planner = FakePlanner()
    app.state.plan_validator = validator or FakeValidator()
    app.state.plan_executor = FakeExecutor()
    return TestClient(app)


def _plan(
    *,
    plan_id="plan-1",
    task_id="task-1",
    status=PlanStatus.DRAFT,
    autonomy=1,
    source="template",
    template_id="sample_echo",
):
    return Plan(
        id=plan_id,
        task_id=task_id,
        goal="finish",
        source=source,
        template_id=template_id,
        steps=[
            PlanStep(
                id="step-1",
                plan_id=plan_id,
                index=0,
                title="Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs={"message": "hi"},
                depends_on=[],
                post_checks=[],
            )
        ],
        autonomy_level=autonomy,
        status=status,
    )


def test_create_plan_endpoint_routes_template_and_persists_validated_plan(tmp_path):
    from marvis.orchestrator.templates import load_builtin_templates

    load_builtin_templates()
    client = _client(tmp_path)

    response = client.post(
        "/api/tasks/task-1/plans",
        json={"goal": "echo", "autonomy_level": 2, "slots": {"message": "hello"}},
    )

    assert response.status_code == 201
    payload = response.json()["plan"]
    assert payload["status"] == "validated"
    assert payload["autonomy_level"] == 2
    assert client.app.state.plan_repo.load_plan("plan-1").status == PlanStatus.VALIDATED


def test_create_plan_endpoint_returns_validator_problems(tmp_path):
    from marvis.orchestrator.templates import load_builtin_templates

    load_builtin_templates()
    client = _client(tmp_path, validator=FakeValidator(["missing tool"]))

    response = client.post("/api/tasks/task-1/plans", json={"goal": "echo"})

    assert response.status_code == 422
    assert response.json()["detail"] == {"problems": ["missing tool"]}


def test_create_plan_endpoint_uses_generated_path_for_novel_goal(tmp_path):
    client = _client(tmp_path, intent=FakeIntentRouter(kind="novel"))

    response = client.post("/api/tasks/task-1/plans", json={"goal": "custom analysis"})

    assert response.status_code == 201
    assert response.json()["plan"]["source"] == "generated"
    assert client.app.state.planner.generated


def test_create_plan_endpoint_passes_capability_tier_and_novel_mode(tmp_path):
    client = _client(tmp_path, intent=FakeIntentRouter(kind="novel"))

    response = client.post(
        "/api/tasks/task-1/plans",
        json={
            "goal": "custom analysis",
            "tier": "autonomous",
            "novel_mode": "explore",
        },
    )

    assert response.status_code == 201
    call = client.app.state.planner.generated[0]
    assert call[4].name == "autonomous"
    assert call[5] == "explore"


def test_capability_tiers_endpoint_lists_defaults(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/capability-tiers")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default"] == "balanced"
    assert {tier["name"] for tier in payload["tiers"]} == {
        "conservative",
        "balanced",
        "autonomous",
    }


def test_step_output_endpoint_returns_stored_structured_output(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    repo.create_plan(_plan(status=PlanStatus.VALIDATED))
    repo.store_step_output("step-1", {"auc": 0.74, "notes": ["ok"]})
    repo.store_step_output("step-1", {"auc": 0.81, "notes": ["new"]})

    response = client.get("/api/step-outputs/step-1")
    versioned = client.get("/api/step-outputs/step-1:v1")
    missing = client.get("/api/step-outputs/missing-step")

    assert response.status_code == 200
    assert response.json() == {"auc": 0.81, "notes": ["new"]}
    assert versioned.status_code == 200
    assert versioned.json() == {"auc": 0.74, "notes": ["ok"]}
    assert missing.status_code == 404


def test_get_plan_payload_includes_sub_agents_for_v2_panel(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    repo.create_plan(_plan(status=PlanStatus.RUNNING))
    repo.upsert_sub_agent(
        SubAgent(
            id="sub-1",
            parent_task_id="task-1",
            parent_step_id="step-1",
            scope="Review join",
            granted_tools=[ToolRef("_sample", "echo")],
            context_budget=8192,
            status=AgentStatus.RETURNED,
            result_ref="artifact:summary",
        )
    )

    response = client.get("/api/plans/plan-1")

    assert response.status_code == 200
    assert response.json()["plan"]["sub_agents"] == [
        {
            "id": "sub-1",
            "parent_task_id": "task-1",
            "parent_step_id": "step-1",
            "scope": "Review join",
            "granted_tools": [{"plugin": "_sample", "tool": "echo", "version": ""}],
            "context_budget": 8192,
            "status": "returned",
            "result_ref": "artifact:summary",
        }
    ]


def test_plan_confirm_run_step_confirm_and_cancel_endpoints(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    task_id = _create_task(repo.db_path)
    repo.create_plan(_plan(status=PlanStatus.VALIDATED, task_id=task_id))

    confirmed = client.post("/api/plans/plan-1/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["plan"]["status"] == "confirmed"

    run = client.post("/api/plans/plan-1/run")
    assert run.status_code == 202
    assert run.json()["job_id"]
    assert client.app.state.plan_executor.calls == ["plan-1"]
    assert _job_statuses(repo.db_path) == ["succeeded"]

    step_confirm = client.post("/api/plans/plan-1/steps/step-1/confirm")
    assert step_confirm.status_code == 202
    assert step_confirm.json()["job_id"]
    assert client.app.state.plan_executor.calls == ["plan-1", "plan-1"]
    assert repo.is_step_confirmed("step-1") is True
    assert _job_statuses(repo.db_path) == ["succeeded", "succeeded"]

    cancel_client = _client(tmp_path / "cancel")
    cancel_repo = cancel_client.app.state.plan_repo
    cancel_task_id = _create_task(cancel_repo.db_path)
    cancel_repo.create_plan(_plan(status=PlanStatus.CONFIRMED, task_id=cancel_task_id))
    cancelled = cancel_client.post("/api/plans/plan-1/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["plan"]["status"] == "cancelled"


def test_plan_retry_failed_step_endpoint_reopens_plan_and_runs(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    task_id = _create_task(repo.db_path)
    plan = _plan(status=PlanStatus.FAILED, task_id=task_id)
    plan.steps[0].status = StepStatus.FAILED
    plan.steps[0].error = "interrupted during running before output was persisted"
    repo.create_plan(plan)

    response = client.post("/api/plans/plan-1/steps/step-1/retry")

    assert response.status_code == 202
    assert response.json()["reset_step_ids"] == ["step-1"]
    assert response.json()["job_id"]
    assert client.app.state.plan_executor.calls == ["plan-1"]
    assert repo.load_plan("plan-1").steps[0].status == StepStatus.PENDING
    assert repo.load_plan("plan-1").status == PlanStatus.RUNNING
    assert _job_statuses(repo.db_path) == ["succeeded"]


def test_plan_retry_failed_step_endpoint_accepts_replacement_inputs(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    task_id = _create_task(repo.db_path)
    plan = _plan(status=PlanStatus.FAILED, task_id=task_id)
    plan.steps[0].status = StepStatus.FAILED
    repo.create_plan(plan)

    response = client.post(
        "/api/plans/plan-1/steps/step-1/retry",
        json={"inputs": {"message": "retry after threshold edit"}},
    )

    assert response.status_code == 202
    loaded = repo.load_plan("plan-1")
    assert loaded.steps[0].inputs == {"message": "retry after threshold edit"}
    assert loaded.status == PlanStatus.RUNNING


def test_plan_retry_failed_step_endpoint_rejects_non_object_inputs(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    task_id = _create_task(repo.db_path)
    plan = _plan(status=PlanStatus.FAILED, task_id=task_id)
    plan.steps[0].status = StepStatus.FAILED
    repo.create_plan(plan)

    response = client.post(
        "/api/plans/plan-1/steps/step-1/retry",
        json={"inputs": ["not", "an", "object"]},
    )

    assert response.status_code == 422


def test_plan_confirm_dispatches_plan_confirmed_hook(tmp_path):
    client = _client(tmp_path)
    dispatcher = FakeHookDispatcher()
    client.app.state.hook_dispatcher = dispatcher
    repo = client.app.state.plan_repo
    task_id = _create_task(repo.db_path)
    repo.create_plan(_plan(status=PlanStatus.VALIDATED, task_id=task_id))

    response = client.post("/api/plans/plan-1/confirm")

    assert response.status_code == 200
    assert dispatcher.calls == [
        (
            "plan.confirmed",
            {"task_id": task_id, "plan_id": "plan-1"},
            task_id,
        )
    ]


def test_plan_run_rejects_active_task_job(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    task_id = _create_task(repo.db_path)
    repo.create_plan(_plan(status=PlanStatus.CONFIRMED, task_id=task_id))
    TaskRepository(repo.db_path).start_job(task_id, "plan")

    response = client.post("/api/plans/plan-1/run")

    assert response.status_code == 409
    assert response.json()["detail"] == "task already has an active job"
    assert client.app.state.plan_executor.calls == []


def test_create_app_wires_plan_runtime_and_router(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/plans/missing")

    assert response.status_code == 404
    assert hasattr(app.state, "plan_repo")
    assert hasattr(app.state, "plan_executor")
    assert hasattr(app.state, "intent_router")


def test_create_app_can_create_standard_modeling_template_plan_from_goal(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _create_task(app.state.plan_repo.db_path)

    response = client.post(
        f"/api/tasks/{task_id}/plans",
        json={
            "goal": "请帮我建模，训练一个A卡模型",
            "slots": {
                "dataset_id": "dataset-1",
                "target_col": "bad_flag",
                "feature_cols": ["income", "age"],
                "split_col": "split",
                "split_values": {"train": "train", "test": "test", "oot": "oot"},
                "recipe": "lr",
                "seed": 7,
            },
        },
    )

    assert response.status_code == 201, response.json()
    plan = response.json()["plan"]
    assert plan["template_id"] == "standard_modeling"
    assert plan["status"] == "validated"
    tools = [step["tool_ref"]["tool"] for step in plan["steps"]]
    assert "generate_model_report" in tools
    assert plan["steps"][-1]["tool_ref"] == {"plugin": "modeling", "tool": "post_training_action", "version": ""}
    assert plan["steps"][-1]["needs_confirmation"] is True


def test_create_app_can_create_model_validation_plan_from_task_goal(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _create_task(app.state.plan_repo.db_path)

    response = client.post(
        f"/api/tasks/{task_id}/plans",
        json={"goal": "请验证模型"},
    )

    assert response.status_code == 201, response.json()
    plan = response.json()["plan"]
    assert plan["template_id"] == "model_validation"
    assert plan["status"] == "validated"
    assert plan["steps"][0]["tool_ref"] == {"plugin": "v1_compat", "tool": "scan_materials", "version": ""}
    assert plan["steps"][0]["inputs"] == {"task_id": task_id}
    assert plan["steps"][-1]["needs_confirmation"] is True


def test_create_app_can_create_feature_derivation_plan_from_goal(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _create_task(app.state.plan_repo.db_path)

    response = client.post(
        f"/api/tasks/{task_id}/plans",
        json={
            "goal": "做特征衍生和特征交叉",
            "slots": {
                "dataset_id": "dataset-1",
                "target_col": "bad_flag",
                "feature_cols": ["income", "age"],
                "derivation_recipe": [{"kind": "ratio", "num": "income", "den": "age"}],
            },
        },
    )

    assert response.status_code == 201, response.json()
    plan = response.json()["plan"]
    assert plan["template_id"] == "feature_derivation"
    assert [step["tool_ref"]["tool"] for step in plan["steps"]] == [
        "compute_feature_metrics",
        "cross_features",
        "compute_feature_metrics",
        "screen_features",  # FEAT-3: derivation ends in a leakage-aware screening step
    ]
    assert [step["title"] for step in plan["steps"] if step["decision_point"]] == ["衍生特征"]


def test_create_app_can_create_strategy_analysis_plan_from_goal(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id = _create_task(app.state.plan_repo.db_path)

    response = client.post(
        f"/api/tasks/{task_id}/plans",
        json={
            "goal": "做策略回测并看风险收益权衡",
            "slots": {
                "dataset_id": "dataset-1",
                "target_col": "bad_flag",
                "score_col": "score",
                "strategy_type": "approval",
                "rules": [{"condition": "score < 600", "decision": "reject"}],
                "default_decision": "approve",
            },
        },
    )

    assert response.status_code == 201, response.json()
    plan = response.json()["plan"]
    assert plan["template_id"] == "strategy_analysis"
    assert [step["tool_ref"]["tool"] for step in plan["steps"]] == [
        "build_strategy",
        "backtest_strategy",
        "tradeoff_view",
    ]
    assert [step["title"] for step in plan["steps"] if step["needs_confirmation"]] == ["回测策略"]
    assert [step["title"] for step in plan["steps"] if step["decision_point"]] == ["回测策略"]


def _job_statuses(db_path):
    with connect(db_path) as conn:
        rows = conn.execute("SELECT status FROM jobs ORDER BY created_at, id").fetchall()
    return [row["status"] for row in rows]


def _create_task(db_path) -> str:
    task = TaskRepository(db_path).create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(db_path.parent),
        )
    )
    return task.id

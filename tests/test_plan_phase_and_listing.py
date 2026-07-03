"""Step 1 (C4) backend contracts: PlanStep.phase threading + per-task plan listing.

Covers the two backend gaps the V2 plan review confirmed:
  - PlanStep / StepTemplate gained a display-only `phase` field that must survive
    template->plan build, JSON (de)serialization, and SQLite round-trips.
  - PlanRepository.list_plans_for_task + GET /tasks/{task_id}/plans let the right
    rail resume an existing task's plan (create only returns a single plan_id).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.db import PlanRepository, init_db
from marvis.orchestrator.contracts import (
    Plan,
    PlanStatus,
    PlanStep,
    PostCheck,
    plan_to_dict,
)
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef
from marvis.routers.plans import router as plans_router


def _phased_plan(
    *,
    plan_id: str = "plan-1",
    task_id: str = "task-1",
    created_at: str = "2026-06-25T00:00:00+00:00",
) -> Plan:
    return Plan(
        id=plan_id,
        task_id=task_id,
        goal="join then feature",
        source="template",
        template_id="data_join",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        created_at=created_at,
        updated_at=created_at,
        steps=[
            PlanStep(
                id=f"{plan_id}-s1",
                plan_id=plan_id,
                index=0,
                title="Infer schema",
                tool_ref=ToolRef("data_ops", "infer_schema"),
                inputs={},
                depends_on=[],
                post_checks=[PostCheck("schema", {"required": ["columns"]})],
                phase="数据准备",
            ),
            PlanStep(
                id=f"{plan_id}-s2",
                plan_id=plan_id,
                index=1,
                title="Propose join",
                tool_ref=ToolRef("data_ops", "propose_join"),
                inputs={},
                depends_on=[f"{plan_id}-s1"],
                post_checks=[],
                needs_confirmation=True,
                phase="数据准备",
            ),
            PlanStep(
                id=f"{plan_id}-s3",
                plan_id=plan_id,
                index=2,
                title="Compute metrics",
                tool_ref=ToolRef("feature", "compute_feature_metrics"),
                inputs={},
                depends_on=[f"{plan_id}-s2"],
                post_checks=[],
                phase=None,  # ungrouped step stays None
            ),
        ],
    )


def test_phase_survives_create_and_load_round_trip(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _phased_plan()

    repo.create_plan(plan)
    loaded = repo.load_plan("plan-1")

    assert loaded == plan  # full equality incl. phase (default None unaffected)
    assert [s.phase for s in loaded.steps] == ["数据准备", "数据准备", None]


def test_plan_to_dict_emits_phase(tmp_path):
    payload = plan_to_dict(_phased_plan())
    assert [s["phase"] for s in payload["steps"]] == ["数据准备", "数据准备", None]


def test_update_step_preserves_and_updates_phase(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_phased_plan())

    plan = repo.load_plan("plan-1")
    step = plan.steps[2]
    assert step.phase is None
    step.phase = "特征"
    repo.update_step(step)

    reloaded = repo.load_plan("plan-1")
    assert reloaded.steps[2].phase == "特征"
    # untouched steps keep their phase
    assert reloaded.steps[0].phase == "数据准备"


def test_from_template_threads_phase_onto_steps(tmp_path):
    template = WorkflowTemplate(
        id="phased_demo",
        title="Phased Demo",
        goal_patterns=("demo",),
        slots=(SlotSpec("message", False, "user", "Message"),),
        steps=(
            StepTemplate(
                title="Step A",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={},
                depends_on_titles=(),
                post_checks=(),
                phase="建模",
            ),
            StepTemplate(
                title="Step B",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={},
                depends_on_titles=("Step A",),
                post_checks=(),
            ),
        ),
    )
    # from_template does not touch the tool registry / validator, so None is fine.
    planner = Planner(None, None, None)
    plan = planner.from_template(template, {}, task_id="task-9")

    assert plan.steps[0].phase == "建模"
    assert plan.steps[1].phase is None


def test_list_plans_for_task_returns_task_plans_oldest_first(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(
        _phased_plan(plan_id="p-a", task_id="task-A", created_at="2026-06-25T01:00:00+00:00")
    )
    repo.create_plan(
        _phased_plan(plan_id="p-b", task_id="task-A", created_at="2026-06-25T02:00:00+00:00")
    )
    repo.create_plan(
        _phased_plan(plan_id="p-c", task_id="task-B", created_at="2026-06-25T03:00:00+00:00")
    )

    task_a = repo.list_plans_for_task("task-A")
    assert [p.id for p in task_a] == ["p-a", "p-b"]
    assert [p.id for p in repo.list_plans_for_task("task-B")] == ["p-c"]
    assert repo.list_plans_for_task("task-unknown") == []


def _plans_client(tmp_path) -> TestClient:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    app = FastAPI()
    app.include_router(plans_router)
    app.state.plan_repo = PlanRepository(db_path)
    return TestClient(app)


def test_list_task_plans_endpoint(tmp_path):
    client = _plans_client(tmp_path)
    repo = client.app.state.plan_repo
    repo.create_plan(
        _phased_plan(plan_id="p-1", task_id="task-X", created_at="2026-06-25T01:00:00+00:00")
    )
    repo.create_plan(
        _phased_plan(plan_id="p-2", task_id="task-X", created_at="2026-06-25T02:00:00+00:00")
    )

    resp = client.get("/api/tasks/task-X/plans")
    assert resp.status_code == 200, resp.text
    plans = resp.json()["plans"]
    assert [p["id"] for p in plans] == ["p-1", "p-2"]
    assert [s["phase"] for s in plans[0]["steps"]] == ["数据准备", "数据准备", None]
    assert "sub_agents" in plans[0]


def test_list_task_plans_endpoint_empty_for_unknown_task(tmp_path):
    client = _plans_client(tmp_path)
    resp = client.get("/api/tasks/none/plans")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"plans": []}


def test_list_task_plans_endpoint_supports_limit_offset_and_reports_total(tmp_path):
    # LT-13: limit/offset are opt-in -- omitting them (covered above) must keep
    # returning the full per-task plan history with no pagination keys.
    client = _plans_client(tmp_path)
    repo = client.app.state.plan_repo
    for index in range(3):
        repo.create_plan(
            _phased_plan(
                plan_id=f"p-{index}",
                task_id="task-X",
                created_at=f"2026-06-25T0{index}:00:00+00:00",
            )
        )

    first_page = client.get("/api/tasks/task-X/plans", params={"limit": 2, "offset": 0})
    assert first_page.status_code == 200, first_page.text
    body = first_page.json()
    assert [p["id"] for p in body["plans"]] == ["p-0", "p-1"]
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert body["has_more"] is True

    second_page = client.get("/api/tasks/task-X/plans", params={"limit": 2, "offset": 2})
    body2 = second_page.json()
    assert [p["id"] for p in body2["plans"]] == ["p-2"]
    assert body2["total"] == 3
    assert body2["has_more"] is False


def test_list_task_plans_endpoint_limit_is_bounded_at_maximum(tmp_path):
    client = _plans_client(tmp_path)
    repo = client.app.state.plan_repo
    repo.create_plan(_phased_plan(plan_id="p-only", task_id="task-Y"))

    response = client.get("/api/tasks/task-Y/plans", params={"limit": 999999})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["limit"] == 500
    assert [p["id"] for p in body["plans"]] == ["p-only"]

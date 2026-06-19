import json
from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import PlanStatus, PostCheck, StepStatus
from marvis.orchestrator.planner import PLAN_SYS, REPLAN_SYS, Planner, PlanningError, ReplanError
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(registry)


def _planner(tmp_path: Path, llm: FakeLLM) -> Planner:
    tool_registry = _tool_registry(tmp_path)
    return Planner(tool_registry, lambda: llm, PlanValidator(tool_registry))


def _template() -> WorkflowTemplate:
    return WorkflowTemplate(
        id="two_step_echo",
        title="Two Step Echo",
        goal_patterns=("echo twice",),
        slots=(SlotSpec("message", True, "user", "Message"),),
        steps=(
            StepTemplate(
                title="First Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={"message": "{slot:message}"},
                depends_on_titles=(),
                post_checks=(PostCheck("nonempty", {"field": "echoed"}),),
            ),
            StepTemplate(
                title="Second Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={"message": "$ref:First Echo.output.echoed"},
                depends_on_titles=("First Echo",),
                post_checks=(PostCheck("nonempty", {"field": "echoed"}),),
            ),
        ),
    )


def _generated_plan(tool: dict | None = None) -> str:
    return json.dumps({
        "autonomy_level": 1,
        "steps": [
            {
                "title": "Echo",
                "tool": tool or {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": "hi"},
                "depends_on": [],
                "post_checks": [{"kind": "nonempty", "spec": {"field": "echoed"}}],
            }
        ],
    })


def _replanned_steps(tool: dict | None = None) -> str:
    return json.dumps({
        "steps": [
            {
                "id": "step-3",
                "title": "Revised Echo",
                "tool": tool or {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": "$ref:step-1.output.echoed"},
                "depends_on": ["step-1"],
                "post_checks": [{"kind": "nonempty", "spec": {"field": "echoed"}}],
            }
        ],
    })


def test_planner_from_template_fills_slots_and_rewrites_refs(tmp_path):
    llm = FakeLLM([])

    plan = _planner(tmp_path, llm).from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )

    assert plan.status == PlanStatus.DRAFT
    assert plan.source == "template"
    assert plan.template_id == "two_step_echo"
    assert [step.plan_id for step in plan.steps] == [plan.id, plan.id]
    assert plan.steps[0].inputs == {"message": "hello"}
    assert plan.steps[1].depends_on == [plan.steps[0].id]
    assert plan.steps[1].inputs == {"message": f"$ref:{plan.steps[0].id}.output.echoed"}
    assert llm.calls == []


def test_planner_from_template_rejects_missing_required_slots(tmp_path):
    with pytest.raises(PlanningError, match="missing required slots"):
        _planner(tmp_path, FakeLLM([])).from_template(_template(), {}, task_id="task-1")


def test_planner_generate_accepts_valid_llm_plan(tmp_path):
    llm = FakeLLM([_generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "echo once",
        "task-1",
        memory_context={},
        task_context={},
    )

    assert "不计算任何指标" in PLAN_SYS
    assert plan.source == "generated"
    assert plan.steps[0].tool_ref == ToolRef("_sample", "echo")
    assert llm.calls[0]["response_format"] == {"type": "json_object"}


def test_planner_generate_retries_after_invalid_json(tmp_path):
    llm = FakeLLM(["not json", _generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "echo once",
        "task-1",
        memory_context={},
        task_context={},
        max_retries=1,
    )

    assert plan.steps[0].title == "Echo"
    assert len(llm.calls) == 2
    assert "not json" in llm.calls[1]["user_prompt"]


def test_planner_generate_retries_validator_failures_and_then_raises(tmp_path):
    llm = FakeLLM([
        _generated_plan({"plugin": "missing", "tool": "echo"}),
        _generated_plan({"plugin": "missing", "tool": "echo"}),
    ])

    with pytest.raises(PlanningError, match="could not generate valid plan"):
        _planner(tmp_path, llm).generate(
            "echo once",
            "task-1",
            memory_context={},
            task_context={},
            max_retries=1,
        )

    assert len(llm.calls) == 2
    assert "missing" in llm.calls[1]["user_prompt"]


def test_planner_replan_replaces_remaining_steps_and_preserves_done(tmp_path):
    llm = FakeLLM([_replanned_steps()])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    plan.steps[0].status = StepStatus.DONE
    plan.steps[0].output_ref = "metrics:step-1"

    replanned = planner.replan(
        plan,
        completed_summaries={"step-1": {"echoed": "hello"}},
        observation={"echoed": "hello"},
        reason="decision_point",
        tier=resolve_tier("balanced"),
    )

    assert "剩余步骤" in REPLAN_SYS
    assert replanned.id == plan.id
    assert replanned.replan_count == 1
    assert replanned.tier == "balanced"
    assert [step.title for step in replanned.steps] == ["First Echo", "Revised Echo"]
    assert replanned.steps[0].status == StepStatus.DONE
    assert replanned.steps[1].depends_on == ["step-1"]
    assert replanned.steps[1].inputs == {"message": "$ref:step-1.output.echoed"}
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
    assert "Two Step Echo" in llm.calls[0]["user_prompt"]
    assert "decision_point" in llm.calls[0]["user_prompt"]


def test_planner_replan_retries_after_validator_failure(tmp_path):
    llm = FakeLLM([
        _replanned_steps({"plugin": "missing", "tool": "echo"}),
        _replanned_steps(),
    ])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    plan.steps[0].status = StepStatus.DONE

    replanned = planner.replan(
        plan,
        completed_summaries={"step-1": {"echoed": "hello"}},
        observation={"error_kind": "execution"},
        reason="failure",
        tier=resolve_tier("balanced"),
    )

    assert replanned.steps[1].tool_ref == ToolRef("_sample", "echo")
    assert len(llm.calls) == 2
    assert "missing" in llm.calls[1]["user_prompt"]


def test_planner_replan_rejects_exhausted_budget_without_llm_call(tmp_path):
    llm = FakeLLM([_replanned_steps()])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    plan.replan_count = resolve_tier("balanced").max_replan_iterations

    with pytest.raises(ReplanError, match="replan budget exhausted"):
        planner.replan(
            plan,
            completed_summaries={},
            observation={},
            reason="decision_point",
            tier=resolve_tier("balanced"),
        )

    assert llm.calls == []

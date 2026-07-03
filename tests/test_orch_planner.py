import json
from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import PlanStatus, PostCheck, StepStatus
from marvis.orchestrator.planner import (
    EXPLORE_SYS,
    PLAN_SYS,
    REPLAN_SYS,
    build_plan_prompt,
    compact_catalog_for_prompt,
    Planner,
    PlanningError,
    ReplanError,
)
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


def _replanned_steps(tool: dict | None = None, ref_id: str = "step-1") -> str:
    return json.dumps({
        "steps": [
            {
                "id": "step-3",
                "title": "Revised Echo",
                "tool": tool or {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": f"$ref:{ref_id}.output.echoed"},
                "depends_on": [ref_id],
                "post_checks": [{"kind": "nonempty", "spec": {"field": "echoed"}}],
            }
        ],
    })


def _multi_step_plan(count: int) -> str:
    return json.dumps({
        "steps": [
            {
                "id": f"step-{index + 1}",
                "title": f"Echo {index + 1}",
                "tool": {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": f"message-{index + 1}"},
                "depends_on": [],
                "post_checks": [{"kind": "nonempty", "spec": {"field": "echoed"}}],
            }
            for index in range(count)
        ],
    })


def _explore_response(*, done: bool = False, ref_id: str = "step-1") -> str:
    return json.dumps({
        "done": done,
        "steps": [] if done else [
            {
                "id": "step-3",
                "title": "Explore Echo",
                "tool": {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": f"$ref:{ref_id}.output.echoed"},
                "depends_on": [ref_id],
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


def test_plan_prompt_uses_compact_catalog_and_ref_examples():
    catalog = [
        {
            "plugin": "_sample",
            "tool": "echo",
            "version": "0.1.0",
            "summary": "Echo a message",
            "determinism": "deterministic",
            "input_schema": {
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"},
                    "seconds": {"type": "number"},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {"echoed": {"type": "string"}},
            },
        }
    ]

    payload = json.loads(
        build_plan_prompt(
            "echo once",
            catalog,
            memory_context={},
            task_context={},
            last_error=None,
        )
    )

    tool = payload["available_tools"][0]
    assert "input_schema" not in tool
    assert "output_schema" not in tool
    assert tool["required_inputs"] == ["message"]
    assert tool["input_fields"][0]["name"] == "message"
    assert tool["output_fields"] == [{"name": "echoed", "type": "string"}]
    assert "$ref:train-step.output.experiment_id" in json.dumps(
        payload["planning_examples"],
        ensure_ascii=False,
    )


def test_compact_catalog_truncates_large_schema_fields():
    catalog = [
        {
            "plugin": "wide",
            "tool": "tool",
            "input_schema": {
                "type": "object",
                "properties": {f"field_{index}": {"type": "string"} for index in range(14)},
            },
            "output_schema": {},
        }
    ]

    compact = compact_catalog_for_prompt(catalog)

    assert len(compact[0]["input_fields"]) == 13
    assert compact[0]["input_fields"][-1]["name"] == "..."
    assert compact[0]["input_fields"][-1]["type"] == "truncated"


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


def test_planner_generate_accepts_plan_wrapped_in_json_fence(tmp_path):
    """AGT-10: _parse_plan_json now goes through load_json_object, so a reply
    wrapped in ```json fences parses on the FIRST attempt (no retry needed) —
    unlike a bare json.loads, which would reject it outright."""
    llm = FakeLLM([f"这是计划:\n```json\n{_generated_plan()}\n```\n"])

    plan = _planner(tmp_path, llm).generate(
        "echo once",
        "task-1",
        memory_context={},
        task_context={},
        max_retries=1,
    )

    assert plan.steps[0].title == "Echo"
    assert len(llm.calls) == 1  # parsed on the first attempt, no retry consumed


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


def test_planner_generate_explore_limits_first_segment_and_sets_mode(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([_multi_step_plan(5)])

    plan = _planner(tmp_path, llm).generate(
        "explore echo",
        "task-1",
        memory_context={},
        task_context={},
        tier=tier,
        novel_mode="explore",
    )

    assert plan.novel_mode == "explore"
    assert plan.tier == "balanced"
    assert len(plan.steps) == tier.explore_segment_size
    assert "explore" in llm.calls[0]["user_prompt"]
    assert str(tier.explore_segment_size) in llm.calls[0]["user_prompt"]


def test_planner_generate_conservative_reverts_explore_to_plan_ahead(tmp_path):
    tier = resolve_tier("conservative")
    llm = FakeLLM([_multi_step_plan(2)])

    plan = _planner(tmp_path, llm).generate(
        "explore echo",
        "task-1",
        memory_context={},
        task_context={},
        tier=tier,
        novel_mode="explore",
    )

    assert plan.novel_mode == "plan_ahead"
    assert plan.tier == "conservative"
    assert len(plan.steps) == 2


def test_planner_replan_replaces_remaining_steps_and_preserves_done(tmp_path):
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    plan.steps[0].output_ref = f"metrics:{done_id}"
    llm.responses = [_replanned_steps(ref_id=done_id)]

    replanned = planner.replan(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
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
    assert replanned.steps[1].depends_on == [done_id]
    assert replanned.steps[1].inputs == {"message": f"$ref:{done_id}.output.echoed"}
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
    assert "Two Step Echo" in llm.calls[0]["user_prompt"]
    assert "decision_point" in llm.calls[0]["user_prompt"]


def test_planner_replan_accepts_steps_wrapped_in_json_fence(tmp_path):
    """AGT-10: replan's MAX_REPLAN_PARSE_RETRY=1 budget means a fence issue
    appearing twice used to raise ReplanError outright; _parse_steps_json now
    falls back to load_json_object, so a single fenced reply parses on the
    first attempt with no retry consumed."""
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    llm.responses = [f"```json\n{_replanned_steps(ref_id=done_id)}\n```"]

    replanned = planner.replan(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        observation={"echoed": "hello"},
        reason="decision_point",
        tier=resolve_tier("balanced"),
    )

    assert [step.title for step in replanned.steps] == ["First Echo", "Revised Echo"]
    assert len(llm.calls) == 1


def test_planner_replan_retries_after_validator_failure(tmp_path):
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    llm.responses = [
        _replanned_steps({"plugin": "missing", "tool": "echo"}, ref_id=done_id),
        _replanned_steps(ref_id=done_id),
    ]

    replanned = planner.replan(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
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


def test_planner_next_explore_segment_returns_valid_segment(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.novel_mode = "explore"
    plan.steps[0].status = StepStatus.DONE
    llm.responses = [_explore_response(ref_id=done_id)]

    segment, done = planner.next_explore_segment(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        tier=tier,
    )

    assert "下一小段" in EXPLORE_SYS
    assert done is False
    assert len(segment) == 1
    assert segment[0].index == 2
    assert segment[0].depends_on == [done_id]
    assert "Two Step Echo" in llm.calls[0]["user_prompt"]


def test_planner_next_explore_segment_accepts_steps_wrapped_in_json_fence(tmp_path):
    """AGT-10: explore's first parse (_parse_json_object, for the {done: bool}
    check) and its steps parse (_parse_steps_json) both tolerate a ```json
    fenced reply on the first attempt."""
    tier = resolve_tier("balanced")
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.novel_mode = "explore"
    plan.steps[0].status = StepStatus.DONE
    llm.responses = [f"```json\n{_explore_response(ref_id=done_id)}\n```"]

    segment, done = planner.next_explore_segment(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        tier=tier,
    )

    assert done is False
    assert len(segment) == 1
    assert len(llm.calls) == 1


def test_planner_next_explore_segment_done_and_budget_exhaustion(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([_explore_response(done=True)])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )

    segment, done = planner.next_explore_segment(
        plan,
        completed_summaries={},
        tier=tier,
    )

    assert segment == []
    assert done is True
    assert len(llm.calls) == 1

    plan.replan_count = tier.max_replan_iterations
    exhausted_segment, exhausted_done = planner.next_explore_segment(
        plan,
        completed_summaries={},
        tier=tier,
    )

    assert exhausted_segment == []
    assert exhausted_done is True
    assert len(llm.calls) == 1

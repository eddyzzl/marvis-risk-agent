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
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.orchestrator.templates.feature import FEATURE_ANALYSIS
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


def _feature_replan_without_binning() -> str:
    return json.dumps({
        "steps": [
            {
                "id": "feature-metrics-revised",
                "title": "特征指标",
                "tool": {
                    "plugin": "feature",
                    "tool": "compute_feature_metrics",
                },
                "inputs": {
                    "dataset_id": "dataset-1",
                    "features": ["sig1", "sig2"],
                    "target_col": "bad_flag",
                    "metrics": ["iv", "ks", "auc", "coverage"],
                    "meaning_directions": {},
                    "bins": 10,
                },
                "depends_on": [],
                "post_checks": [
                    {"kind": "nonempty", "spec": {"field": "metrics"}},
                ],
            },
            {
                "id": "feature-report-revised",
                "title": "生成特征分析报告",
                "tool": {
                    "plugin": "feature",
                    "tool": "generate_feature_report",
                },
                "inputs": {
                    "metrics": "$ref:feature-metrics-revised.output.metrics",
                    "collinear": "$ref:feature-metrics-revised.output.collinear",
                },
                "depends_on": ["feature-metrics-revised"],
                "post_checks": [
                    {"kind": "nonempty", "spec": {"field": "report_path"}},
                ],
            },
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
    assert llm.calls[0]["max_tokens"] == 4096


def test_planner_applies_manifest_governance_when_llm_omits_policy(tmp_path):
    llm = FakeLLM([
        json.dumps({
            "steps": [
                {
                    "title": "Adopt strategy",
                    "tool": {"plugin": "strategy", "tool": "adopt_strategy"},
                    "inputs": {
                        "strategy_id": "strategy-1",
                        "backtest_id": "backtest-1",
                        "adoption_reason": "Human reviewed the backtest",
                    },
                    "depends_on": [],
                    "post_checks": [],
                }
            ]
        })
    ])

    plan = _planner(tmp_path, llm).generate(
        "adopt strategy",
        "task-1",
        memory_context={},
        task_context={},
    )

    step = plan.steps[0]
    assert step.needs_confirmation is True
    assert step.policy.human_decision_gate == "required"
    assert step.policy.effect_authorization == "required"
    assert step.policy.effect_target is not None
    assert step.policy.effect_target.kind == "strategy"


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("autonomy_level", "auto"),
        ("inputs", 7),
    ],
)
def test_planner_generate_wraps_malformed_field_types_and_retries(
    tmp_path,
    field,
    value,
):
    malformed = json.loads(_generated_plan())
    if field == "inputs":
        malformed["steps"][0][field] = value
    else:
        malformed[field] = value
    llm = FakeLLM([json.dumps(malformed), _generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "echo once",
        "task-1",
        memory_context={},
        task_context={},
        max_retries=1,
    )

    assert plan.steps[0].title == "Echo"
    assert len(llm.calls) == 2
    assert "invalid plan fields" in llm.calls[1]["user_prompt"]


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


def test_planner_replan_retries_after_validator_failure(tmp_path, caplog):
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
        _replanned_steps(
            {"plugin": "missing\nraw-marker", "tool": "echo"},
            ref_id=done_id,
        ),
        _replanned_steps(ref_id=done_id),
    ]

    with caplog.at_level("WARNING", logger="marvis.orchestrator.planner"):
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
    assert (
        "replan attempt rejected stage=validator attempt=1 "
        "error_type=PlanValidationError "
        "message=step Revised Echo: missing raw-marker"
    ) in caplog.text
    assert "\nraw-marker" not in caplog.text


def test_planner_replan_retries_invalid_llm_governance_policy(tmp_path):
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    malformed = json.loads(_replanned_steps(ref_id=done_id))
    malformed["steps"][0]["policy"] = {
        "human_decision_gate": "sometimes",
    }
    llm.responses = [
        json.dumps(malformed),
        _replanned_steps(ref_id=done_id),
    ]

    replanned = planner.replan(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        observation={"error_kind": "user_instruction"},
        reason="user_instruction",
        tier=resolve_tier("balanced"),
        instruction="keep the second echo",
    )

    assert replanned.steps[1].title == "Revised Echo"
    assert len(llm.calls) == 2
    assert "human_decision_gate" in llm.calls[1]["user_prompt"]


def test_planner_replan_retries_truncated_feature_plan_with_explicit_output_budget(
    tmp_path,
):
    llm = FakeLLM([
        '{"steps":[{"title":"特征指标"',
        _feature_replan_without_binning(),
    ])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        FEATURE_ANALYSIS,
        {
            "dataset_id": "dataset-1",
            "target_col": "bad_flag",
            "features": ["sig1", "sig2"],
            "metrics": ["iv", "ks", "auc", "coverage"],
            "meaning_directions": {},
        },
        task_id="task-1",
    )
    plan.steps[0].granted_tools = [ToolRef("data_ops", "profile_dataset")]
    plan.steps[0].output_ref = "private-runtime-output"
    plan.steps[0].error = "private-runtime-error"
    plan.steps[0].sub_agent_id = "private-runtime-agent"

    replanned = planner.replan(
        plan,
        completed_summaries={},
        observation={
            "reason": "user_instruction",
            "instruction": "删掉分箱分析，只保留特征指标和最终报告。",
        },
        reason="user_instruction",
        tier=resolve_tier("balanced"),
        instruction="删掉分箱分析，只保留特征指标和最终报告。",
    )

    assert [step.title for step in replanned.steps] == [
        "特征指标",
        "生成特征分析报告",
    ]
    assert replanned.steps[1].depends_on == ["feature-metrics-revised"]
    assert "binning" not in replanned.steps[1].inputs
    assert [call["max_tokens"] for call in llm.calls] == [4096, 4096]
    replan_prompt = json.loads(llm.calls[0]["user_prompt"])
    available_tools = replan_prompt["available_tools"]
    available_refs = {
        (item["plugin"], item["tool"])
        for item in available_tools
    }
    assert ("feature", "screen_features") in available_refs
    assert ("data_ops", "profile_dataset") in available_refs
    assert {
        tool
        for plugin, tool in available_refs
        if plugin == "data_ops"
    } == {"profile_dataset"}
    assert {plugin for plugin, _tool in available_refs} == {"feature", "data_ops"}
    source_metrics, source_binning, source_report = replan_prompt["remaining_steps"]
    assert source_metrics["tool"] == {
        "plugin": "feature",
        "tool": "compute_feature_metrics",
        "version": "",
    }
    assert source_metrics["inputs"]["dataset_id"] == "dataset-1"
    assert source_metrics["inputs"]["features"] == ["sig1", "sig2"]
    assert source_metrics["inputs"]["metrics"] == ["iv", "ks", "auc", "coverage"]
    assert source_binning["depends_on"] == [source_metrics["id"]]
    assert source_report["depends_on"] == [
        source_metrics["id"],
        source_binning["id"],
    ]
    assert source_report["inputs"]["metrics"] == (
        f"$ref:{source_metrics['id']}.output.metrics"
    )
    assert "output_ref" not in source_metrics
    assert "review_verdicts" not in source_metrics
    assert "error" not in source_metrics
    assert "sub_agent_id" not in source_metrics
    assert "private-runtime" not in llm.calls[0]["user_prompt"]
    assert "copy retained steps" in replan_prompt["instruction"]
    assert "depends_on edge and $ref" in replan_prompt["instruction"]
    assert "full revised remaining plan" in replan_prompt["instruction"]
    assert "steps must be non-empty" in replan_prompt["instruction"]


def test_planner_replan_wraps_malformed_field_types_and_retries(tmp_path):
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-1",
    )
    done_id = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    malformed = json.loads(_replanned_steps(ref_id=done_id))
    malformed["steps"][0]["inputs"] = 7
    llm.responses = [json.dumps(malformed), _replanned_steps(ref_id=done_id)]

    replanned = planner.replan(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        observation={"error_kind": "execution"},
        reason="failure",
        tier=resolve_tier("balanced"),
    )

    assert replanned.steps[1].tool_ref == ToolRef("_sample", "echo")
    assert len(llm.calls) == 2
    assert "invalid plan fields" in llm.calls[1]["user_prompt"]


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
    assert llm.calls[0]["max_tokens"] == 4096


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

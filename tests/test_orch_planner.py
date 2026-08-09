import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import marvis.orchestrator.planner as planner_module
from marvis.db import PluginRepository, init_db
from marvis.llm_client import estimate_tokens
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import Plan, PlanStatus, PostCheck, StepStatus
from marvis.orchestrator.planner import (
    ContextBudgetExhaustedError,
    EXPLORE_SYS,
    PLAN_SYS,
    REPLAN_SYS,
    build_plan_prompt,
    compact_catalog_for_prompt,
    Planner,
    PlannerConstraints,
    PlanningError,
    ReplanError,
    RequiredLiteralInput,
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
    def __init__(self, responses, *, profile=None):
        self.responses = list(responses)
        self.calls = []
        self.profile = dict(profile or {})

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class CatalogOverrideRegistry:
    def __init__(self, base, catalog):
        self._base = base
        self._catalog = list(catalog)

    def catalog_for_planner(self):
        return list(self._catalog)

    def resolve(self, ref):
        return self._base.resolve(ref)

    def resolve_with_manifest(self, ref):
        return self._base.resolve_with_manifest(ref)


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


def _strategy_build_plan() -> str:
    return json.dumps({
        "steps": [
            {
                "id": "build-current-strategy",
                "title": "Build current strategy",
                "tool": {"plugin": "strategy", "tool": "build_strategy"},
                "inputs": {
                    "strategy_type": "approval",
                    "rules": [
                        {"condition": "score >= 700", "decision": "approve"},
                    ],
                    "default_decision": "reject",
                },
                "depends_on": [],
                "post_checks": [],
            },
        ],
    })


def _strategy_adopt_plan() -> str:
    return json.dumps({
        "steps": [
            {
                "id": "adopt-stale-strategy",
                "title": "Adopt stale strategy",
                "tool": {"plugin": "strategy", "tool": "adopt_strategy"},
                "inputs": {
                    "strategy_id": "strategy-stale",
                    "backtest_id": "backtest-stale",
                    "adoption_reason": "use prior evidence",
                },
                "depends_on": [],
                "post_checks": [],
            },
        ],
    })


def _schema_inspection_plan() -> str:
    return json.dumps({
        "autonomy_level": 1,
        "steps": [
            {
                "title": "Inspect registered schema",
                "tool": {"plugin": "data_ops", "tool": "infer_schema"},
                "inputs": {"dataset_id": "fixture://catalog-ranking"},
                "depends_on": [],
                "post_checks": [],
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
    assert llm.calls[0]["max_tokens"] == 16384
    assert llm.calls[0]["temperature"] == 0


def test_planner_constraints_retain_stale_required_and_exclude_adopt(tmp_path):
    llm = FakeLLM(
        [_strategy_build_plan()],
        profile={"context_window": 32_768},
    )
    constraints = PlannerConstraints(
        required_tool_refs=(ToolRef("strategy", "build_strategy"),),
        forbidden_tool_refs=(ToolRef("strategy", "adopt_strategy"),),
        required_literal_inputs=(
            RequiredLiteralInput(
                ToolRef("strategy", "build_strategy"),
                {"strategy_type": "approval"},
            ),
        ),
    )

    plan = _planner(tmp_path, llm).generate(
        "Refresh stale strategy evidence; do not adopt it",
        "task-stale-strategy",
        memory_context={},
        task_context={"required_tools": ["strategy.adopt_strategy"]},
        constraints=constraints,
    )

    prompt = json.loads(llm.calls[0]["user_prompt"])
    available_refs = {
        f"{item['plugin']}.{item['tool']}"
        for item in prompt["available_tools"]
    }
    assert "strategy.build_strategy" in available_refs
    assert "strategy.adopt_strategy" not in available_refs
    assert prompt["planner_constraints"] == {
        "forbidden_tool_refs": [
            {"plugin": "strategy", "tool": "adopt_strategy", "version": "0.19.0"},
        ],
        "required_literal_inputs": [
            {
                "tool_ref": {
                    "plugin": "strategy",
                    "tool": "build_strategy",
                    "version": "0.19.0",
                },
                "inputs": {"strategy_type": "approval"},
            },
        ],
        "required_tool_refs": [
            {"plugin": "strategy", "tool": "build_strategy", "version": "0.19.0"},
        ],
    }
    assert plan.steps[0].tool_ref == ToolRef("strategy", "build_strategy")


def test_planner_constraints_reject_forbidden_output_after_retry(tmp_path):
    llm = FakeLLM([_strategy_adopt_plan(), _strategy_adopt_plan()])

    with pytest.raises(
        PlanningError,
        match="forbidden tool strategy.adopt_strategy was selected",
    ):
        _planner(tmp_path, llm).generate(
            "Do not adopt stale strategy evidence",
            "task-stale-strategy",
            memory_context={},
            task_context={},
            max_retries=1,
            constraints=PlannerConstraints(
                forbidden_tool_refs=(ToolRef("strategy", "adopt_strategy"),),
            ),
        )

    assert len(llm.calls) == 2
    assert "forbidden tool strategy.adopt_strategy" in llm.calls[1]["user_prompt"]


def test_planner_constraints_reject_forbidden_subagent_grant_after_retry(tmp_path):
    forbidden_grant = json.loads(_generated_plan())
    forbidden_grant["steps"][0].update({
        "id": "echo-with-forbidden-grant",
        "sub_agent_scope": "echo through a restricted sub-agent",
        "granted_tools": [
            {"plugin": "data_ops", "tool": "infer_schema"},
        ],
    })
    llm = FakeLLM([json.dumps(forbidden_grant), _generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "echo without schema access",
        "task-forbidden-grant",
        memory_context={},
        task_context={},
        max_retries=1,
        constraints=PlannerConstraints(
            forbidden_tool_refs=(ToolRef("data_ops", "infer_schema"),),
        ),
    )

    assert plan.steps[0].granted_tools == []
    assert len(llm.calls) == 2
    assert "forbidden tool data_ops.infer_schema" in llm.calls[1]["user_prompt"]


def test_planner_constraints_retry_required_literal_input_mismatch(tmp_path):
    corrected = json.loads(_generated_plan())
    corrected["steps"][0]["id"] = "echo-current"
    corrected["steps"][0]["inputs"] = {"message": "current"}
    llm = FakeLLM([_generated_plan(), json.dumps(corrected)])

    plan = _planner(tmp_path, llm).generate(
        "echo current",
        "task-current-input",
        memory_context={},
        task_context={},
        max_retries=1,
        constraints=PlannerConstraints(
            required_literal_inputs=(
                RequiredLiteralInput(
                    ToolRef("_sample", "echo"),
                    {"message": "current"},
                ),
            ),
        ),
    )

    assert plan.steps[0].inputs == {"message": "current"}
    assert len(llm.calls) == 2
    assert "required literal inputs mismatch" in llm.calls[1]["user_prompt"]


def test_planner_required_literal_input_matches_any_repeated_tool_step(tmp_path):
    repeated_echo = json.dumps({
        "steps": [
            {
                "id": "echo-prior",
                "title": "Echo prior message",
                "tool": {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": "prior"},
                "depends_on": [],
                "post_checks": [
                    {"kind": "nonempty", "spec": {"field": "echoed"}},
                ],
            },
            {
                "id": "echo-current",
                "title": "Echo current message",
                "tool": {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": "current"},
                "depends_on": [],
                "post_checks": [
                    {"kind": "nonempty", "spec": {"field": "echoed"}},
                ],
            },
        ],
    })

    plan = _planner(tmp_path, FakeLLM([repeated_echo])).generate(
        "echo both messages",
        "task-repeated-tool",
        memory_context={},
        task_context={},
        constraints=PlannerConstraints(
            required_literal_inputs=(
                RequiredLiteralInput(
                    ToolRef("_sample", "echo"),
                    {"message": "prior"},
                ),
                RequiredLiteralInput(
                    ToolRef("_sample", "echo"),
                    {"message": "current"},
                ),
            ),
        ),
    )

    assert [step.inputs["message"] for step in plan.steps] == ["prior", "current"]


def test_planner_explore_defers_literal_match_until_completion(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([_generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "explore prior and current messages",
        "task-explore-literal-later",
        memory_context={},
        task_context={},
        tier=tier,
        novel_mode="explore",
        max_retries=0,
        constraints=PlannerConstraints(
            required_literal_inputs=(
                RequiredLiteralInput(
                    ToolRef("_sample", "echo"),
                    {"message": "current"},
                ),
            ),
        ),
    )

    assert plan.steps[0].inputs == {"message": "hi"}
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("constraints", "expected_error"),
    [
        (
            PlannerConstraints(
                required_tool_refs=(ToolRef("missing", "echo"),),
            ),
            "unknown tool missing.echo",
        ),
        (
            PlannerConstraints(
                required_tool_refs=(ToolRef("_sample", "echo"),),
                forbidden_tool_refs=(ToolRef("_sample", "echo"),),
            ),
            "conflict for tool _sample.echo",
        ),
    ],
)
def test_planner_constraints_fail_closed_before_llm(
    tmp_path,
    constraints,
    expected_error,
):
    llm = FakeLLM([_generated_plan()])

    with pytest.raises(PlanningError, match=expected_error):
        _planner(tmp_path, llm).generate(
            "echo once",
            "task-invalid-constraints",
            memory_context={},
            task_context={},
            constraints=constraints,
        )

    assert llm.calls == []


def test_planner_constraints_fail_closed_when_required_catalog_exceeds_budget(
    tmp_path,
):
    base = _tool_registry(tmp_path)
    oversized_echo = {
        "plugin": "_sample",
        "tool": "echo",
        "version": "0.1.0",
        "summary": "echo",
        "determinism": "deterministic",
        "input_schema": {
            "type": "object",
            "properties": {
                f"field_{index}": {
                    "type": "string",
                    "description": "required catalog metadata " * 20,
                }
                for index in range(12)
            },
        },
        "output_schema": {"type": "object", "properties": {}},
    }
    registry = CatalogOverrideRegistry(base, [oversized_echo])
    llm = FakeLLM([_generated_plan()], profile={"context_window": 512})
    planner = Planner(registry, lambda: llm, PlanValidator(base))

    with pytest.raises(PlanningError, match="required tools exceed catalog token budget"):
        planner.generate(
            "echo once",
            "task-catalog-budget",
            memory_context={},
            task_context={},
            constraints=PlannerConstraints(
                required_tool_refs=(ToolRef("_sample", "echo"),),
            ),
        )

    assert llm.calls == []


def test_planner_accepts_step_id_only_when_id_is_missing(tmp_path):
    response = json.loads(_generated_plan())
    response["steps"][0]["step_id"] = "legacy-step-id"

    plan = _planner(tmp_path, FakeLLM([json.dumps(response)])).generate(
        "echo once",
        "task-step-id-alias",
        memory_context={},
        task_context={},
    )

    assert plan.steps[0].id == "legacy-step-id"


def test_planner_rejects_conflicting_id_and_step_id(tmp_path):
    response = json.loads(_generated_plan())
    response["steps"][0]["id"] = "canonical-id"
    response["steps"][0]["step_id"] = "different-id"

    with pytest.raises(PlanningError, match="conflicting id and step_id"):
        _planner(tmp_path, FakeLLM([json.dumps(response)])).generate(
            "echo once",
            "task-step-id-conflict",
            memory_context={},
            task_context={},
            max_retries=0,
        )


def test_planner_rejects_explicit_null_optional_step_field(tmp_path):
    response = json.loads(_generated_plan())
    response["steps"][0]["sub_agent_scope"] = None

    with pytest.raises(PlanningError, match="must be omitted rather than null"):
        _planner(tmp_path, FakeLLM([json.dumps(response)])).generate(
            "echo once",
            "task-null-field",
            memory_context={},
            task_context={},
            max_retries=0,
        )


def test_planner_rejects_explicit_null_optional_plan_field(tmp_path):
    response = json.loads(_generated_plan())
    response["success_criteria"] = None

    with pytest.raises(PlanningError, match="must be omitted rather than null"):
        _planner(tmp_path, FakeLLM([json.dumps(response)])).generate(
            "echo once",
            "task-null-plan-field",
            memory_context={},
            task_context={},
            max_retries=0,
        )


def test_planner_generate_omits_stale_context_before_reducing_json_budget(tmp_path):
    stale_history = "stale business note | " * 50_000
    llm = FakeLLM(
        [_generated_plan()],
        profile={"context_window": 32_768, "max_output_tokens": 16_384},
    )

    _planner(tmp_path, llm).generate(
        "Generate the active vintage report under the confirmed constraint.",
        "task-context-budget",
        memory_context={"user_constraint": "keep the approved split"},
        task_context={
            "workflow_family": "fixed",
            "entrypoint": "vintage",
            "dataset_id": "fixture://risk-analysis/current-snapshot",
            "active_dataset_revision": 23,
            "target_col": "bad_flag",
            "split": {"train": "2024-01", "test": "2024-02"},
            "features": ["mob", "balance"],
            "historical_context": stale_history,
            "stale_dataset_id": "fixture://risk-analysis/previous-snapshot",
        },
    )

    call = llm.calls[0]
    payload = json.loads(call["user_prompt"])
    task_context = payload["task_context"]
    omission = task_context["historical_context"]
    assert omission["__omitted_context__"] == "context_budget"
    assert omission["original_type"] == "str"
    assert set(omission) == {"__omitted_context__", "original_type"}
    assert stale_history[:100] not in call["user_prompt"]
    assert task_context["dataset_id"] == "fixture://risk-analysis/current-snapshot"
    assert task_context["active_dataset_revision"] == 23
    assert task_context["target_col"] == "bad_flag"
    assert task_context["split"] == {"train": "2024-01", "test": "2024-02"}
    assert task_context["features"] == ["mob", "balance"]
    assert payload["memory_context"]["user_constraint"] == "keep the approved split"
    assert call["max_tokens"] == 16_384
    assert call["truncated"] is True


def test_planner_generate_recomputes_completion_from_fitted_prompt(tmp_path):
    context_window = 8_192
    llm = FakeLLM(
        [_schema_inspection_plan()],
        profile={"context_window": context_window},
    )

    _planner(tmp_path, llm).generate(
        "Inspect the active dataset schema.",
        "task-dynamic-completion",
        memory_context={},
        task_context={"dataset_id": "fixture://active"},
    )

    call = llm.calls[0]
    prompt_tokens = estimate_tokens(PLAN_SYS) + estimate_tokens(call["user_prompt"])
    available_tokens = context_window - prompt_tokens - 512

    assert call["max_tokens"] == min(16_384, available_tokens)
    assert 4_096 <= call["max_tokens"] < 16_384
    assert prompt_tokens + call["max_tokens"] + 512 <= context_window


def test_planner_generate_honors_profile_output_cap(tmp_path):
    llm = FakeLLM(
        [_generated_plan()],
        profile={"context_window": 32_768, "max_output_tokens": 8_192},
    )

    _planner(tmp_path, llm).generate(
        "echo once",
        "task-profile-cap",
        memory_context={},
        task_context={},
    )

    assert llm.calls[0]["max_tokens"] == 8_192


def test_planner_generate_fails_typed_when_protected_context_exhausts_json_floor(
    tmp_path,
):
    llm = FakeLLM(
        [_generated_plan()],
        profile={"context_window": 8_192, "max_output_tokens": 4_096},
    )
    protected_goal = "confirmed-active-constraint " * 2_000

    with pytest.raises(ContextBudgetExhaustedError) as caught:
        _planner(tmp_path, llm).generate(
            protected_goal,
            "task-exhausted-context",
            memory_context={},
            task_context={
                "dataset_id": "fixture://active",
                "active_dataset_revision": 7,
                "target_col": "bad_flag",
            },
        )

    error = caught.value
    assert error.error_kind == "context_budget_exhausted"
    assert error.context_window == 8_192
    assert error.available_tokens < error.required_tokens == 4_096
    assert protected_goal[:100] not in str(error)
    assert llm.calls == []


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
    assert "unique `id`" in payload["instruction"]
    assert "instead of null" in payload["instruction"]
    assert "hard server-authored constraints" in payload["instruction"]


@pytest.mark.parametrize("system_prompt", [PLAN_SYS, REPLAN_SYS, EXPLORE_SYS])
def test_planner_system_prompts_define_canonical_step_shape(system_prompt):
    assert "`id`" in system_prompt
    assert "`step_id`" in system_prompt
    assert "null" in system_prompt
    assert "hard" in system_prompt.casefold()


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


@pytest.mark.parametrize(
    ("goal", "task_context", "expected_ref"),
    [
        (
            "Inventory the submitted notebook and model validation materials.",
            {"workflow_family": "fixed", "entrypoint": "validation"},
            ("v1_compat", "scan_materials"),
        ),
        (
            "Build and train a credit-risk model from the approved dataset.",
            {"workflow_family": "fixed", "entrypoint": "modeling"},
            ("modeling", "train_model"),
        ),
        (
            "Generate the VTG terminal risk-analysis report.",
            {"workflow_family": "fixed", "entrypoint": "vintage"},
            ("risk_analysis", "generate_risk_analysis_report"),
        ),
        (
            "Backtest a cutoff strategy and compare its tradeoffs.",
            {"workflow_family": "adaptive", "entrypoint": "strategy"},
            ("strategy", "backtest_strategy"),
        ),
        (
            "Continue the explicitly selected workflow safely.",
            {
                "workflow": {
                    "plugin": "v1_compat",
                    "tool": "scan_materials",
                }
            },
            ("v1_compat", "scan_materials"),
        ),
        (
            "请基于已确认的数据训练信用风险模型。",
            {"workflow_family": "fixed", "entrypoint": "modeling"},
            ("modeling", "train_model"),
        ),
    ],
)
def test_planner_retrieves_relevant_tail_tools_with_bounded_catalog(
    tmp_path,
    goal,
    task_context,
    expected_ref,
):
    registry = _tool_registry(tmp_path)
    full_catalog = registry.catalog_for_planner()
    full_refs = [(item["plugin"], item["tool"]) for item in full_catalog]
    assert full_refs.index(expected_ref) >= 49

    llm = FakeLLM(
        [_schema_inspection_plan()],
        profile={"context_window": 8192},
    )
    planner = Planner(registry, lambda: llm, PlanValidator(registry))

    planner.generate(
        goal,
        "task-catalog-ranking",
        memory_context={},
        task_context={
            **task_context,
            "dataset_id": "fixture://catalog-ranking",
        },
    )

    call = llm.calls[0]
    available = json.loads(call["user_prompt"])["available_tools"]
    available_refs = [
        (item["plugin"], item["tool"])
        for item in available
    ]
    available_by_ref = {
        (item["plugin"], item["tool"]): item
        for item in available
    }
    assert expected_ref in available_refs[:5]
    assert len(available_by_ref) < len(full_catalog)
    assert estimate_tokens(
        json.dumps(available, ensure_ascii=False, sort_keys=True)
    ) <= 4096
    assert call["truncated"] is True

    universal_core = {
        ("data_ops", "infer_schema"),
        ("data_ops", "profile_dataset"),
        ("data_ops", "propose_join"),
    }
    assert universal_core <= available_by_ref.keys()

    policy_gated = {
        (item["plugin"], item["tool"])
        for item in full_catalog
        if item["policy"]["human_decision_gate"] == "required"
        or item["policy"]["effect_authorization"] == "required"
    }
    shown_policy_gated = policy_gated.intersection(available_by_ref)
    assert shown_policy_gated
    assert len(shown_policy_gated) <= 8
    for ref in shown_policy_gated:
        original = full_catalog[full_refs.index(ref)]
        assert available_by_ref[ref]["policy"] == original["policy"]


def test_planner_bounds_safety_catalog_reservation_and_keeps_shown_policy(tmp_path):
    base = _tool_registry(tmp_path)
    policy = {
        "schema_version": "tool-policy.v1",
        "human_decision_gate": "required",
        "effect_authorization": "none",
    }
    synthetic_safety = [
        {
            "plugin": "synthetic_safety",
            "tool": f"governed_action_{index}",
            "version": "1.0.0",
            "summary": f"Governed optional action {index}",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {}},
            "determinism": "deterministic",
            "policy": dict(policy),
        }
        for index in range(200)
    ]
    registry = CatalogOverrideRegistry(
        base,
        [*base.catalog_for_planner(), *synthetic_safety],
    )
    llm = FakeLLM(
        [_schema_inspection_plan()],
        profile={"context_window": 8192},
    )

    Planner(registry, lambda: llm, PlanValidator(registry)).generate(
        "Inspect schema, then consider governed action 199.",
        "task-many-gated-tools",
        memory_context={},
        task_context={"dataset_id": "fixture://catalog-ranking"},
    )

    call = llm.calls[0]
    available = json.loads(call["user_prompt"])["available_tools"]
    shown_safety = [
        item
        for item in available
        if item["plugin"] == "synthetic_safety"
    ]
    assert 1 <= len(shown_safety) <= 8
    assert all(item["policy"] == policy for item in shown_safety)
    assert estimate_tokens(
        json.dumps(available, ensure_ascii=False, sort_keys=True)
    ) <= 4096
    assert call["truncated"] is True


def test_catalog_query_traversal_is_bounded_for_deep_large_context():
    nested: object = "tail"
    for _index in range(1_500):
        nested = {"node": nested}
    large = {
        f"field_{index}": f"unique_term_{index}"
        for index in range(2_000)
    }

    weights = planner_module._catalog_query_weights(
        "训练模型",
        {
            "entrypoint": "modeling",
            "deep": nested,
            "large": large,
        },
    )

    assert {"model", "modeling"} <= weights.keys()
    assert len(weights) <= 512


def test_catalog_query_term_order_is_hash_seed_stable():
    script = """
import json
from marvis.orchestrator.planner import _catalog_query_weights
weights = _catalog_query_weights(
    "训练模型",
    {
        "entrypoint": "modeling",
        "signals": {"zeta", "alpha", "训练", "模型"},
    },
)
print(json.dumps(list(weights), ensure_ascii=False))
"""
    outputs = []
    for seed in (1, 2, 3):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(seed)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[1],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert len(set(outputs)) == 1


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


def test_planner_generate_retries_after_invalid_governance_policy(tmp_path):
    invalid = json.loads(_generated_plan())
    invalid["steps"][0]["policy"] = {"human_decision_gate": "maybe"}
    llm = FakeLLM([json.dumps(invalid), _generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "echo once",
        "task-invalid-policy",
        memory_context={},
        task_context={},
    )

    assert plan.steps[0].tool_ref == ToolRef("_sample", "echo")
    assert len(llm.calls) == 2
    assert "human_decision_gate" in llm.calls[1]["user_prompt"]


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


def test_planner_generate_explore_allows_required_tool_in_later_segment(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([_generated_plan()])

    plan = _planner(tmp_path, llm).generate(
        "explore data before schema inspection",
        "task-explore-initial-segment",
        memory_context={},
        task_context={},
        tier=tier,
        novel_mode="explore",
        constraints=PlannerConstraints(
            required_tool_refs=(ToolRef("data_ops", "infer_schema"),),
        ),
    )

    assert plan.novel_mode == "explore"
    assert [step.tool_ref for step in plan.steps] == [ToolRef("_sample", "echo")]
    assert len(llm.calls) == 1


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
    assert llm.calls[0]["temperature"] == 0
    assert "Two Step Echo" in llm.calls[0]["user_prompt"]
    assert "decision_point" in llm.calls[0]["user_prompt"]


def test_planner_replan_constraints_extend_scoped_catalog(tmp_path):
    llm = FakeLLM([])
    planner = _planner(tmp_path, llm)
    plan = planner.from_template(
        _template(),
        {"message": "hello"},
        task_id="task-replan-constraints",
    )
    done_id = plan.steps[0].id
    plan.steps[0].status = StepStatus.DONE
    llm.responses = [json.dumps({
        "steps": [
            {
                "id": "inspect-current-dataset",
                "title": "Inspect current dataset",
                "tool": {"plugin": "data_ops", "tool": "infer_schema"},
                "inputs": {"dataset_id": "dataset-current"},
                "depends_on": [],
                "post_checks": [],
            },
        ],
    })]

    replanned = planner.replan(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        observation={"dataset_id": "dataset-current"},
        reason="stale_evidence",
        tier=resolve_tier("balanced"),
        constraints=PlannerConstraints(
            required_tool_refs=(ToolRef("data_ops", "infer_schema"),),
            required_literal_inputs=(
                RequiredLiteralInput(
                    ToolRef("data_ops", "infer_schema"),
                    {"dataset_id": "dataset-current"},
                ),
            ),
        ),
    )

    prompt = json.loads(llm.calls[0]["user_prompt"])
    assert "data_ops.infer_schema" in {
        f"{item['plugin']}.{item['tool']}"
        for item in prompt["available_tools"]
    }
    assert replanned.steps[-1].tool_ref == ToolRef("data_ops", "infer_schema")


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


def test_planner_replan_retries_after_scoped_catalog_failure(tmp_path, caplog):
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
        "replan attempt rejected stage=parse attempt=1 "
        "error_type=PlanningError "
        "message=tools outside the allowed replan catalog: missing raw-marker.echo"
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
    assert [call["max_tokens"] for call in llm.calls] == [16384, 16384]
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
    assert ("strategy", "adopt_strategy") not in available_refs
    assert ("analysis", "portfolio_gate_summary") not in available_refs
    assert ("labeling", "define_label") not in available_refs
    assert ("modeling", "train_model") not in available_refs
    assert ("v1_compat", "scan_materials") not in available_refs
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


def test_planner_replan_rejects_globally_valid_tool_outside_scoped_catalog(tmp_path):
    unauthorized_echo = json.dumps({
        "steps": [
            {
                "id": "unauthorized-echo",
                "title": "Unrelated echo",
                "tool": {"plugin": "_sample", "tool": "echo"},
                "inputs": {"message": "globally valid but not replan-authorized"},
                "depends_on": [],
                "post_checks": [
                    {"kind": "nonempty", "spec": {"field": "echoed"}},
                ],
            }
        ],
    })
    llm = FakeLLM([unauthorized_echo, _feature_replan_without_binning()])
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
        task_id="task-replan-scope",
    )
    plan.steps[0].granted_tools = [ToolRef("data_ops", "profile_dataset")]

    replanned = planner.replan(
        plan,
        completed_summaries={},
        observation={"reason": "user_instruction"},
        reason="user_instruction",
        tier=resolve_tier("balanced"),
        instruction="删掉分箱分析，只保留特征指标和最终报告。",
    )

    assert len(llm.calls) == 2
    assert "outside the allowed replan catalog" in llm.calls[1]["user_prompt"]
    assert [step.tool_ref for step in replanned.steps] == [
        ToolRef("feature", "compute_feature_metrics"),
        ToolRef("feature", "generate_feature_report"),
    ]


def test_planner_replan_rejects_grant_outside_scoped_catalog(tmp_path):
    out_of_scope_grant = json.loads(_feature_replan_without_binning())
    out_of_scope_grant["steps"][0].update({
        "sub_agent_scope": "feature work only",
        "granted_tools": [
            {"plugin": "_sample", "tool": "echo"},
        ],
    })
    llm = FakeLLM([
        json.dumps(out_of_scope_grant),
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
        task_id="task-replan-grant-scope",
    )

    replanned = planner.replan(
        plan,
        completed_summaries={},
        observation={"reason": "user_instruction"},
        reason="user_instruction",
        tier=resolve_tier("balanced"),
        instruction="删掉分箱分析，只保留特征指标和最终报告。",
    )

    assert len(llm.calls) == 2
    assert "outside the allowed replan catalog" in llm.calls[1]["user_prompt"]
    assert all(not step.granted_tools for step in replanned.steps)


def test_planner_replan_rejects_tool_removed_by_final_request_fit(
    tmp_path,
    monkeypatch,
):
    removed_tool_plan = json.dumps({
        "steps": [
            {
                "id": "removed-normalize",
                "title": "Normalize outside the final prompt scope",
                "tool": {"plugin": "feature", "tool": "normalize"},
                "inputs": {
                    "dataset_id": "dataset-1",
                    "columns": ["sig1"],
                    "method": "zscore",
                },
                "depends_on": [],
                "post_checks": [],
            },
        ],
    })
    llm = FakeLLM([removed_tool_plan, _feature_replan_without_binning()])
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
        task_id="task-replan-final-fit-scope",
    )
    original_fit = planner_module._fit_planner_request

    def fit_without_normalize(client, *, system_prompt, user_prompt):
        fitted_prompt, max_tokens, truncated = original_fit(
            client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        payload = json.loads(fitted_prompt)
        payload["available_tools"] = [
            item
            for item in payload["available_tools"]
            if (item["plugin"], item["tool"]) != ("feature", "normalize")
        ]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True), max_tokens, True

    monkeypatch.setattr(
        planner_module,
        "_fit_planner_request",
        fit_without_normalize,
    )

    replanned = planner.replan(
        plan,
        completed_summaries={},
        observation={"reason": "user_instruction"},
        reason="user_instruction",
        tier=resolve_tier("balanced"),
        instruction="删掉分箱分析，只保留特征指标和最终报告。",
    )

    assert len(llm.calls) == 2
    assert all(
        (item["plugin"], item["tool"]) != ("feature", "normalize")
        for call in llm.calls
        for item in json.loads(call["user_prompt"])["available_tools"]
    )
    assert "outside the allowed replan catalog" in llm.calls[1]["user_prompt"]
    assert [step.tool_ref for step in replanned.steps] == [
        ToolRef("feature", "compute_feature_metrics"),
        ToolRef("feature", "generate_feature_report"),
    ]


def test_planner_constraints_fail_closed_on_empty_replan_scope(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([_generated_plan()])
    empty_plan = Plan(
        id="empty-replan",
        task_id="task-empty-replan",
        goal="replan without an authorized source scope",
        source="generated",
        template_id=None,
        steps=[],
        autonomy_level=tier.default_autonomy_level,
        tier=tier.name,
    )

    with pytest.raises(ReplanError, match="replan catalog scope is empty"):
        _planner(tmp_path, llm).replan(
            empty_plan,
            completed_summaries={},
            observation={},
            reason="user_instruction",
            tier=tier,
            constraints=PlannerConstraints(
                forbidden_tool_refs=(ToolRef("strategy", "adopt_strategy"),),
            ),
        )

    assert llm.calls == []


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
    llm.responses = [
        json.dumps(malformed),
        json.dumps(malformed),
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
    assert len(llm.calls) == 3
    assert "invalid plan fields" in llm.calls[1]["user_prompt"]
    assert "invalid plan fields" in llm.calls[2]["user_prompt"]


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
    assert llm.calls[0]["max_tokens"] == 16384
    assert llm.calls[0]["temperature"] == 0


def test_planner_explore_rejects_done_until_required_tool_is_planned(tmp_path):
    tier = resolve_tier("balanced")
    llm = FakeLLM([
        _explore_response(done=True),
        json.dumps({
            "done": False,
            "steps": [
                {
                    "id": "required-echo",
                    "title": "Required echo",
                    "tool": {"plugin": "_sample", "tool": "echo"},
                    "inputs": {"message": "current"},
                    "depends_on": [],
                    "post_checks": [
                        {"kind": "nonempty", "spec": {"field": "echoed"}},
                    ],
                },
            ],
        }),
    ])
    plan = Plan(
        id="explore-plan",
        task_id="task-explore-constraints",
        goal="explore then echo",
        source="generated",
        template_id=None,
        steps=[],
        autonomy_level=tier.default_autonomy_level,
        novel_mode="explore",
        tier=tier.name,
    )

    segment, done = _planner(tmp_path, llm).next_explore_segment(
        plan,
        completed_summaries={},
        tier=tier,
        task_context={},
        constraints=PlannerConstraints(
            required_tool_refs=(ToolRef("_sample", "echo"),),
            required_literal_inputs=(
                RequiredLiteralInput(
                    ToolRef("_sample", "echo"),
                    {"message": "current"},
                ),
            ),
        ),
    )

    assert done is False
    assert [step.tool_ref for step in segment] == [ToolRef("_sample", "echo")]
    assert len(llm.calls) == 2
    assert "required tool _sample.echo is missing" in llm.calls[1]["user_prompt"]


def test_planner_explore_retries_segment_without_required_constraint_progress(
    tmp_path,
):
    tier = resolve_tier("balanced")
    planner = _planner(tmp_path, FakeLLM([]))
    plan = planner.from_template(
        _template(),
        {"message": "research material"},
        task_id="task-explore-progress",
    )
    plan.novel_mode = "explore"
    for step in plan.steps:
        step.status = StepStatus.DONE
    prior_id = plan.steps[0].id
    llm = FakeLLM([
        _explore_response(ref_id=prior_id),
        json.dumps({
            "done": False,
            "steps": [
                {
                    "id": "inspect-current-schema",
                    "title": "Inspect current schema",
                    "tool": {"plugin": "data_ops", "tool": "infer_schema"},
                    "inputs": {"dataset_id": "dataset-current"},
                    "depends_on": [],
                    "post_checks": [],
                },
            ],
        }),
    ])

    segment, done = _planner(tmp_path, llm).next_explore_segment(
        plan,
        completed_summaries={},
        tier=tier,
        task_context={},
        constraints=PlannerConstraints(
            required_tool_refs=(ToolRef("data_ops", "infer_schema"),),
        ),
    )

    assert done is False
    assert [step.tool_ref for step in segment] == [
        ToolRef("data_ops", "infer_schema"),
    ]
    assert len(llm.calls) == 2
    assert "made no progress toward unmet planner constraints" in (
        llm.calls[1]["user_prompt"]
    )
    assert "required tool data_ops.infer_schema is missing" in (
        llm.calls[1]["user_prompt"]
    )
    first_prompt = json.loads(llm.calls[0]["user_prompt"])
    assert "must advance at least one currently unmet required" in (
        first_prompt["instruction"]
    )


def test_planner_explore_retries_explicit_null_done_field(tmp_path):
    tier = resolve_tier("balanced")
    plan = _planner(tmp_path, FakeLLM([])).from_template(
        _template(),
        {"message": "hello"},
        task_id="task-explore-null-done",
    )
    done_id = plan.steps[0].id
    plan.novel_mode = "explore"
    plan.steps[0].status = StepStatus.DONE
    invalid = json.loads(_explore_response(ref_id=done_id))
    invalid["done"] = None
    llm = FakeLLM([
        json.dumps(invalid),
        _explore_response(ref_id=done_id),
    ])

    segment, done = _planner(tmp_path, llm).next_explore_segment(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        tier=tier,
    )

    assert done is False
    assert len(segment) == 1
    assert len(llm.calls) == 2
    assert "done must be omitted rather than null" in llm.calls[1]["user_prompt"]


def test_planner_explore_retries_after_invalid_governance_policy(tmp_path):
    tier = resolve_tier("balanced")
    plan = _planner(tmp_path, FakeLLM([])).from_template(
        _template(),
        {"message": "hello"},
        task_id="task-explore-invalid-policy",
    )
    done_id = plan.steps[0].id
    plan.novel_mode = "explore"
    plan.steps[0].status = StepStatus.DONE
    invalid = json.loads(_explore_response(ref_id=done_id))
    invalid["steps"][0]["policy"] = {"human_decision_gate": "maybe"}
    llm = FakeLLM([
        json.dumps(invalid),
        _explore_response(ref_id=done_id),
    ])

    segment, done = _planner(tmp_path, llm).next_explore_segment(
        plan,
        completed_summaries={done_id: {"echoed": "hello"}},
        tier=tier,
    )

    assert done is False
    assert len(segment) == 1
    assert len(llm.calls) == 2
    assert "human_decision_gate" in llm.calls[1]["user_prompt"]


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

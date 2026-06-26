from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import Plan, PlanStep, PostCheck
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef, parse_manifest
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    registry.register(_metrics_manifest(), enabled=True)
    registry.register(_join_manifest(), enabled=True)
    return ToolRegistry(registry)


def _metrics_manifest():
    return parse_manifest(
        {
            "name": "metrics_pack",
            "version": "0.1.0",
            "display_name": "Metrics Pack",
            "description": "Validator test metrics",
            "module": "metrics_pack.tools",
            "tools": [
                {
                    "name": "score_metrics",
                    "summary": "Compute metrics",
                    "input_schema": {
                        "type": "object",
                        "properties": {"dataset": {"type": "string"}},
                        "required": ["dataset"],
                        "additionalProperties": False,
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "ks": {"type": "number"},
                            "auc": {"type": "number"},
                        },
                        "required": ["ks", "auc"],
                        "additionalProperties": False,
                    },
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_score_metrics",
                },
                {
                    "name": "nested_metrics",
                    "summary": "Compute nested metrics",
                    "input_schema": {
                        "type": "object",
                        "properties": {"dataset": {"type": "string"}},
                        "required": ["dataset"],
                        "additionalProperties": False,
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "metrics": {
                                "type": "object",
                                "properties": {"ks": {"type": "number"}},
                                "required": ["ks"],
                                "additionalProperties": False,
                            },
                            "bins": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"total_iv": {"type": "number"}},
                                    "required": ["total_iv"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["metrics", "bins"],
                        "additionalProperties": False,
                    },
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_nested_metrics",
                }
            ],
            "hooks": [],
            "permissions": [],
        },
        builtin=True,
    )


def _join_manifest():
    return parse_manifest(
        {
            "name": "join_pack",
            "version": "0.1.0",
            "display_name": "Join Pack",
            "description": "Validator test joins",
            "module": "join_pack.tools",
            "tools": [
                {
                    "name": "execute_join",
                    "summary": "Execute a join",
                    "input_schema": {
                        "type": "object",
                        "properties": {"left": {"type": "string"}, "right": {"type": "string"}},
                        "required": ["left", "right"],
                        "additionalProperties": False,
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"joined_rows": {"type": "integer"}},
                        "required": ["joined_rows"],
                        "additionalProperties": False,
                    },
                    "determinism": "deterministic",
                    "timeout_seconds": 10,
                    "failure_policy": "fail",
                    "entrypoint": "tool_execute_join",
                }
            ],
            "hooks": [],
            "permissions": [],
        },
        builtin=True,
    )


def _plan(*steps: PlanStep) -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="validate",
        source="template",
        template_id="test",
        steps=list(steps),
        autonomy_level=1,
    )


def _step(
    step_id: str,
    tool_ref: ToolRef,
    inputs: dict,
    *,
    depends_on: list[str] | None = None,
    post_checks: list[PostCheck] | None = None,
    needs_confirmation: bool = False,
    sub_agent_scope: str | None = None,
    granted_tools: list[ToolRef] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        plan_id="plan-1",
        index=int(step_id.rsplit("-", 1)[-1]),
        title=step_id,
        tool_ref=tool_ref,
        inputs=inputs,
        depends_on=depends_on or [],
        post_checks=post_checks or [],
        needs_confirmation=needs_confirmation,
        sub_agent_scope=sub_agent_scope,
        granted_tools=granted_tools or [],
    )


def _validator(tmp_path: Path) -> PlanValidator:
    return PlanValidator(_tool_registry(tmp_path))


def test_plan_validator_accepts_basic_echo_plan(tmp_path):
    step = _step("step-1", ToolRef("_sample", "echo"), {"message": "hi"})

    assert _validator(tmp_path).validate(_plan(step)) == []


def test_plan_validator_reports_unknown_tools(tmp_path):
    step = _step("step-1", ToolRef("missing", "echo"), {"message": "hi"})

    problems = _validator(tmp_path).validate(_plan(step))

    assert any("missing" in problem for problem in problems)


def test_plan_validator_checks_literal_inputs_but_skips_deferred_inputs(tmp_path):
    bad = _step("step-1", ToolRef("_sample", "echo"), {"message": 123})
    deferred = _step("step-1", ToolRef("_sample", "echo"), {"message": "{slot:message}"})

    assert any("schema" in problem for problem in _validator(tmp_path).validate(_plan(bad)))
    assert _validator(tmp_path).validate(_plan(deferred)) == []


def test_plan_validator_checks_dangling_dependencies_and_cycles(tmp_path):
    dangling = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        depends_on=["missing"],
    )
    first = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        depends_on=["step-2"],
    )
    second = _step(
        "step-2",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        depends_on=["step-1"],
    )

    dangling_problems = _validator(tmp_path).validate(_plan(dangling))
    cycle_problems = _validator(tmp_path).validate(_plan(first, second))

    assert any("dangling" in problem for problem in dangling_problems)
    assert any("cycle" in problem for problem in cycle_problems)


def test_plan_validator_checks_ref_compatibility(tmp_path):
    upstream = _step("step-1", ToolRef("_sample", "echo"), {"message": "hi"})
    valid = _step(
        "step-2",
        ToolRef("_sample", "echo"),
        {"message": "$ref:step-1.output.echoed"},
        depends_on=["step-1"],
    )
    missing_field = _step(
        "step-2",
        ToolRef("_sample", "echo"),
        {"message": "$ref:step-1.output.missing"},
        depends_on=["step-1"],
    )
    missing_edge = _step(
        "step-2",
        ToolRef("_sample", "echo"),
        {"message": "$ref:step-1.output.echoed"},
    )

    assert _validator(tmp_path).validate(_plan(upstream, valid)) == []
    assert any(
        "upstream output" in problem
        for problem in _validator(tmp_path).validate(_plan(upstream, missing_field))
    )
    assert any(
        "dependency" in problem
        for problem in _validator(tmp_path).validate(_plan(upstream, missing_edge))
    )


def test_plan_validator_requires_join_confirmation(tmp_path):
    join = _step(
        "step-1",
        ToolRef("join_pack", "execute_join"),
        {"left": "a", "right": "b"},
    )
    confirmed = _step(
        "step-1",
        ToolRef("join_pack", "execute_join"),
        {"left": "a", "right": "b"},
        needs_confirmation=True,
    )

    assert any("join" in problem for problem in _validator(tmp_path).validate(_plan(join)))
    assert _validator(tmp_path).validate(_plan(confirmed)) == []


def test_plan_validator_requires_draft_run_confirmation(tmp_path):
    draft_run = _step(
        "step-1",
        ToolRef("drafts", "run_draft"),
        {"draft_id": "draft-1", "inputs": {}},
    )
    confirmed = _step(
        "step-1",
        ToolRef("drafts", "run_draft"),
        {"draft_id": "draft-1", "inputs": {}},
        needs_confirmation=True,
    )

    problems = _validator(tmp_path).validate(_plan(draft_run))

    assert any("draft" in problem and "confirmation" in problem for problem in problems)
    assert _validator(tmp_path).validate(_plan(confirmed)) == []


def test_plan_validator_requires_range_checks_for_metric_fields(tmp_path):
    missing_checks = _step(
        "step-1",
        ToolRef("metrics_pack", "score_metrics"),
        {"dataset": "sample"},
    )
    checked = _step(
        "step-1",
        ToolRef("metrics_pack", "score_metrics"),
        {"dataset": "sample"},
        post_checks=[
            PostCheck("range", {"field": "ks", "min": 0.0, "max": 1.0}),
            PostCheck("range", {"field": "auc", "min": 0.0, "max": 1.0}),
        ],
    )

    problems = _validator(tmp_path).validate(_plan(missing_checks))

    assert any("ks" in problem for problem in problems)
    assert any("auc" in problem for problem in problems)
    assert _validator(tmp_path).validate(_plan(checked)) == []


def test_plan_validator_requires_range_checks_for_nested_metric_fields(tmp_path):
    missing_checks = _step(
        "step-1",
        ToolRef("metrics_pack", "nested_metrics"),
        {"dataset": "sample"},
    )
    checked = _step(
        "step-1",
        ToolRef("metrics_pack", "nested_metrics"),
        {"dataset": "sample"},
        post_checks=[
            PostCheck("range", {"field": "metrics.ks", "min": 0.0, "max": 1.0}),
            PostCheck("range", {"field": "bins.0.total_iv", "min": 0.0}),
        ],
    )

    problems = _validator(tmp_path).validate(_plan(missing_checks))

    assert any("metrics.ks" in problem for problem in problems)
    assert any("bins.0.total_iv" in problem for problem in problems)
    assert _validator(tmp_path).validate(_plan(checked)) == []


def test_plan_validator_requires_subagent_tool_grants(tmp_path):
    no_grants = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        sub_agent_scope="summarize",
    )
    bad_grant = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        sub_agent_scope="summarize",
        granted_tools=[ToolRef("missing", "echo")],
    )
    ok = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        sub_agent_scope="summarize",
        granted_tools=[ToolRef("_sample", "echo")],
    )

    assert any("empty granted_tools" in problem for problem in _validator(tmp_path).validate(_plan(no_grants)))
    assert any("granted tool" in problem for problem in _validator(tmp_path).validate(_plan(bad_grant)))
    assert _validator(tmp_path).validate(_plan(ok)) == []


def test_plan_validator_rejects_unknown_post_check_kind(tmp_path):
    step = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        post_checks=[PostCheck("mystery", {"field": "echoed"})],
    )

    problems = _validator(tmp_path).validate(_plan(step))

    assert any("unknown post_check kind mystery" in problem for problem in problems)


def test_plan_validator_accepts_one_of_post_check_kind(tmp_path):
    step = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        post_checks=[PostCheck("one_of", {"field": "status", "values": ["ok"]})],
    )

    assert _validator(tmp_path).validate(_plan(step)) == []


def test_plan_validator_rejects_decision_point_on_safety_steps(tmp_path):
    metric_step = _step(
        "step-1",
        ToolRef("_sample", "echo"),
        {"message": "hi"},
        post_checks=[PostCheck("range", {"field": "ks", "max": 1.0})],
    )
    metric_step.decision_point = True
    join_step = _step(
        "step-2",
        ToolRef("join_pack", "execute_join"),
        {"left": "a", "right": "b"},
        needs_confirmation=True,
    )
    join_step.decision_point = True

    problems = _validator(tmp_path).validate(_plan(metric_step, join_step))

    assert any("safety step step-1" in problem for problem in problems)
    assert any("safety step step-2" in problem for problem in problems)

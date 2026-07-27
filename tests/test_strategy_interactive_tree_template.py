from __future__ import annotations

from pathlib import Path

from marvis.agent.strategy_request_compiler import (
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_INTERACTIVE_TREE_REVISION,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


AUTOMATIC_SOURCE = "candidate-asset-" + "a" * 32
NODE_ID = "node-" + "1" * 20
REASON = "人工确认该子树颗粒度过细"


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply

    def complete(self, **_kwargs):
        return self.reply


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def _workflow_request(*, reason: str | None = REASON) -> dict:
    inputs: dict[str, object] = {
        "source_tree_id": AUTOMATIC_SOURCE,
        "node_id": NODE_ID,
        "operation": "prune_subtree",
    }
    if reason is not None:
        inputs["reason"] = reason
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_revision",
        "workflow_inputs": inputs,
    }


def test_interactive_tree_revision_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_INTERACTIVE_TREE_REVISION

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_interactive_tree_revision"
    assert template == get_template(template.id)
    assert len(template.steps) == 1

    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert set(slot_sources) == {
        "source_tree_id",
        "node_id",
        "operation",
        "feature",
        "threshold",
        "reason",
    }
    assert set(slot_sources.values()) == {"user"}
    assert {slot.name for slot in template.slots if slot.required} == {
        "source_tree_id",
        "node_id",
        "operation",
    }

    [step] = template.steps
    assert step.tool_ref == ToolRef("strategy", "revise_interactive_tree")
    assert step.inputs_template == {
        "source_tree_id": "{slot:source_tree_id}",
        "node_id": "{slot:node_id}",
        "operation": "{slot:operation}",
        "feature": "{slot:feature}",
        "threshold": "{slot:threshold}",
        "reason": "{slot:reason}",
    }
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.policy.human_decision_gate == "none"
    assert step.policy.effect_authorization == "none"
    assert step.policy.effect_target is None
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "revision_id"}),
        PostCheck("nonempty", {"field": "revision_hash"}),
        PostCheck("nonempty", {"field": "semantic_tree_id"}),
        PostCheck("nonempty", {"field": "tree_hash"}),
        PostCheck(
            "one_of",
            {
                "field": "replay.exactly_once",
                "values": [True],
            },
        ),
        PostCheck("nonempty", {"field": "replay.result_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_interactive_tree_revision_template_matches_the_closed_tool_contract(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    tool = tools.resolve(ToolRef("strategy", "revise_interactive_tree"))

    assert set(tool.input_schema["properties"]) == {
        "source_tree_id",
        "node_id",
        "operation",
        "feature",
        "threshold",
        "reason",
    }
    assert set(tool.input_schema["required"]) == {
        "source_tree_id",
        "node_id",
        "operation",
    }
    assert tool.input_schema["additionalProperties"] is False
    assert tool.entrypoint == "tool_revise_interactive_tree"
    assert tool.determinism == "deterministic"
    assert tool.policy.human_decision_gate == "none"
    assert tool.policy.effect_authorization == "none"
    assert tool.policy.effect_target is None
    assert set(tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }

    planner = Planner(tools, lambda: None, PlanValidator(tools))
    plan = planner.from_template(
        get_template("strategy_interactive_tree_revision"),
        _workflow_request()["workflow_inputs"],
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.template_id == "strategy_interactive_tree_revision"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_ref == ToolRef(
        "strategy",
        "revise_interactive_tree",
    )
    assert plan.steps[0].inputs == _workflow_request()["workflow_inputs"]
    assert plan.steps[0].needs_confirmation is False
    assert plan.steps[0].policy.human_decision_gate == "none"
    assert plan.steps[0].policy.effect_authorization == "none"


def test_interactive_tree_revision_template_omits_optional_reason(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    inputs = _workflow_request(reason=None)["workflow_inputs"]

    plan = planner.from_template(
        get_template("strategy_interactive_tree_revision"),
        inputs,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == inputs
    assert "reason" not in plan.steps[0].inputs


def test_manual_and_natural_language_requests_reach_the_same_template_inputs(
    tmp_path: Path,
) -> None:
    request = _workflow_request()
    manual = validate_strategy_request(request, allowed_columns=())
    natural = compile_strategy_request(
        (
            f"对交互式树 {AUTOMATIC_SOURCE} 的节点 {NODE_ID} 执行 "
            f"prune_subtree；理由：{REASON}。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(request),
    )
    assert manual.draft is not None
    assert natural.draft is not None
    assert manual.draft.to_dict() == natural.draft.to_dict()

    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    template = get_template("strategy_interactive_tree_revision")
    manual_plan = planner.from_template(
        template,
        manual.draft.to_dict()["workflow_inputs"],
        task_id="task-1",
    )
    natural_plan = planner.from_template(
        template,
        natural.draft.to_dict()["workflow_inputs"],
        task_id="task-1",
    )

    assert manual_plan.template_id == natural_plan.template_id == template.id
    assert manual_plan.steps[0].tool_ref == natural_plan.steps[0].tool_ref
    assert manual_plan.steps[0].inputs == natural_plan.steps[0].inputs

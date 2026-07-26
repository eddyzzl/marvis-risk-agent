from __future__ import annotations

from pathlib import Path
import re

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_INTERACTIVE_TREE_FRONTIER_MATERIALIZATION,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


REVISION_ID = "interactive-tree-revision-" + "1" * 32
SOURCE_NODE_ID = "node-" + "2" * 20
REASON = "人工确认这个前沿节点进入候选池准备区"


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def _inputs(*, reason: str | None = REASON) -> dict[str, str]:
    inputs = {
        "revision_id": REVISION_ID,
        "source_node_id": SOURCE_NODE_ID,
    }
    if reason is not None:
        inputs["selection_reason"] = reason
    return inputs


def test_interactive_tree_frontier_materialization_is_one_nongated_builtin_step() -> (
    None
):
    template = STRATEGY_INTERACTIVE_TREE_FRONTIER_MATERIALIZATION

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_interactive_tree_frontier_materialization"
    assert template == get_template(template.id)
    assert len(template.steps) == 1

    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert slot_sources == {
        "revision_id": "user",
        "source_node_id": "user",
        "selection_reason": "user",
    }
    assert {slot.name for slot in template.slots if slot.required} == {
        "revision_id",
        "source_node_id",
    }

    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy",
        "materialize_interactive_tree_frontier_selection",
    )
    assert step.inputs_template == {
        "revision_id": "{slot:revision_id}",
        "source_node_id": "{slot:source_node_id}",
        "selection_reason": "{slot:selection_reason}",
    }
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.policy.human_decision_gate == "none"
    assert step.policy.effect_authorization == "none"
    assert step.policy.effect_target is None
    assert step.post_checks == tuple(
        PostCheck("nonempty", {"field": field})
        for field in (
            "selection_id",
            "selection_hash",
            "revision_id",
            "semantic_tree_id",
            "tree_hash",
            "source_node_id",
            "leaf_id",
            "fragment_id",
            "fragment_hash",
            "rule_id",
            "effect_id",
            "artifacts",
        )
    )


def test_interactive_tree_frontier_template_matches_closed_tool_contract(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    tool = tools.resolve(
        ToolRef(
            "strategy",
            "materialize_interactive_tree_frontier_selection",
        )
    )

    assert set(tool.input_schema["properties"]) == {
        "revision_id",
        "source_node_id",
        "selection_reason",
    }
    assert set(tool.input_schema["required"]) == {
        "revision_id",
        "source_node_id",
    }
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["properties"]["revision_id"] == {
        "type": "string",
        "pattern": "^interactive-tree-revision-[0-9a-f]{32}$",
    }
    assert tool.input_schema["properties"]["source_node_id"] == {
        "type": "string",
        "pattern": "^(?:node|leaf)-[0-9a-f]{20}$",
    }
    assert tool.entrypoint == "tool_materialize_interactive_tree_frontier_selection"
    assert tool.determinism == "deterministic"
    assert tool.policy.human_decision_gate == "none"
    assert tool.policy.effect_authorization == "none"
    assert tool.policy.effect_target is None
    assert set(tool.side_effects) == {
        "read:task",
        "read:dataset",
        "write:artifact",
    }

    output = tool.output_schema
    assert output["additionalProperties"] is False
    assert output["properties"]["schema_version"] == {
        "const": (
            "strategy.materialize-interactive-tree-frontier-selection-tool.v1"
        )
    }
    assert set(output["required"]) == {
        "schema_version",
        "selection_id",
        "selection_hash",
        "selection_reason",
        "revision_id",
        "semantic_tree_id",
        "tree_hash",
        "source_node_id",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
        "artifacts",
    }
    assert output["properties"]["artifacts"]["minItems"] == 1
    assert output["properties"]["artifacts"]["maxItems"] == 1

    planner = Planner(tools, lambda: None, PlanValidator(tools))
    plan = planner.from_template(
        get_template("strategy_interactive_tree_frontier_materialization"),
        _inputs(),
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.template_id == "strategy_interactive_tree_frontier_materialization"
    assert len(plan.steps) == 1
    assert plan.steps[0].inputs == _inputs()
    assert plan.steps[0].needs_confirmation is False


def test_interactive_tree_frontier_template_omits_optional_reason(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    inputs = _inputs(reason=None)

    plan = planner.from_template(
        get_template("strategy_interactive_tree_frontier_materialization"),
        inputs,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == inputs
    assert "selection_reason" not in plan.steps[0].inputs


def test_pool_manifest_accepts_only_candidate_or_interactive_tree_asset_ids(
    tmp_path: Path,
) -> None:
    tools = _tool_registry(tmp_path)
    pool_tool = tools.resolve(ToolRef("strategy", "add_candidate_to_pool"))
    pattern = pool_tool.input_schema["properties"]["expected_asset_id"]["pattern"]

    assert pattern == "^(?:candidate-asset|interactive-tree)-[0-9a-f]{32}$"
    compiled = re.compile(pattern)
    assert compiled.fullmatch("candidate-asset-" + "a" * 32)
    assert compiled.fullmatch("interactive-tree-" + "b" * 32)
    assert compiled.fullmatch("interactive-tree-revision-" + "c" * 32) is None
    assert compiled.fullmatch("arbitrary-asset-id") is None

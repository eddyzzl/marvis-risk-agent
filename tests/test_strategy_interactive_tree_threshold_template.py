"""The existing interactive-tree workflow transports one threshold edit."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


SOURCE_ID = "candidate-asset-" + "a" * 32
NODE_ID = "node-" + "1" * 20


def _tools(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def test_interactive_tree_template_transports_threshold_without_evidence() -> None:
    load_builtin_templates()
    template = get_template("strategy_interactive_tree_revision")

    slots = {slot.name: slot for slot in template.slots}
    assert set(slots) == {
        "source_tree_id",
        "node_id",
        "operation",
        "threshold",
        "reason",
    }
    assert slots["operation"].required is True
    assert slots["threshold"].required is False
    assert template.steps[0].inputs_template == {
        "source_tree_id": "{slot:source_tree_id}",
        "node_id": "{slot:node_id}",
        "operation": "{slot:operation}",
        "threshold": "{slot:threshold}",
        "reason": "{slot:reason}",
    }
    assert template.steps[0].post_checks == (
        PostCheck("nonempty", {"field": "revision_id"}),
        PostCheck("nonempty", {"field": "revision_hash"}),
        PostCheck("nonempty", {"field": "semantic_tree_id"}),
        PostCheck("nonempty", {"field": "tree_hash"}),
        PostCheck(
            "one_of",
            {"field": "replay.exactly_once", "values": [True]},
        ),
        PostCheck("nonempty", {"field": "replay.result_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_threshold_template_plan_injects_only_user_owned_tool_inputs(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tools(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    workflow_inputs = {
        "source_tree_id": SOURCE_ID,
        "node_id": NODE_ID,
        "operation": "adjust_split_threshold",
        "threshold": 1.5,
        "reason": "人工复核后调整风险切分",
    }

    plan = planner.from_template(
        get_template("strategy_interactive_tree_revision"),
        workflow_inputs,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_ref == ToolRef(
        "strategy",
        "revise_interactive_tree",
    )
    assert plan.steps[0].inputs == workflow_inputs

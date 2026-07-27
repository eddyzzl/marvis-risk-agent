from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_INTERACTIVE_TREE_SPLIT_SEARCH,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


SOURCE_ID = "candidate-asset-" + "a" * 32
NODE_ID = "node-" + "1" * 20


def test_split_search_is_one_nongated_builtin_tool_step(tmp_path: Path) -> None:
    template = STRATEGY_INTERACTIVE_TREE_SPLIT_SEARCH
    assert template in BUILTIN_TEMPLATES
    assert template == get_template("strategy_interactive_tree_split_search")
    assert len(template.steps) == 1
    assert {slot.name for slot in template.slots if slot.required} == {
        "source_tree_id",
        "node_id",
        "mode",
        "max_thresholds_per_feature",
        "max_row_evaluations",
    }
    assert {slot.source for slot in template.slots} == {"user"}
    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy",
        "search_interactive_tree_split_candidates",
    )
    assert step.needs_confirmation is False
    assert step.policy.human_decision_gate == "none"
    assert step.policy.effect_authorization == "none"

    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    inputs = {
        "source_tree_id": SOURCE_ID,
        "node_id": NODE_ID,
        "mode": "all_features",
        "max_thresholds_per_feature": 10,
        "max_row_evaluations": 2_000_000,
    }
    plan = Planner(tools, lambda: None, PlanValidator(tools)).from_template(
        template,
        inputs,
        task_id="task-1",
    )
    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == inputs
    assert "features" not in plan.steps[0].inputs


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)

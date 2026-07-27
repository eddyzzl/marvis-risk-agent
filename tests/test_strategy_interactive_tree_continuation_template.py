from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_INTERACTIVE_TREE_AUTO_CONTINUATION,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def test_continuation_is_one_nongated_builtin_tool_step(
    tmp_path: Path,
) -> None:
    template = STRATEGY_INTERACTIVE_TREE_AUTO_CONTINUATION
    assert template in BUILTIN_TEMPLATES
    assert template == get_template(
        "strategy_interactive_tree_auto_continuation"
    )
    assert len(template.steps) == 1
    assert {slot.source for slot in template.slots} == {"user"}
    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy",
        "auto_continue_interactive_tree",
    )
    assert step.needs_confirmation is False

    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    inputs = {
        "search_id": "interactive-tree-split-search-" + "a" * 32,
        "candidate_id": "interactive-tree-split-candidate-" + "b" * 32,
        "max_additional_depth": 3,
        "min_gini_gain": 0.01,
        "max_generated_nodes": 31,
        "max_thresholds_per_feature": 10,
        "max_row_evaluations": 2_000_000,
        "objective": "max_gini_gain",
        "tie_break": "eligible_gain_feature_threshold_candidate_id",
    }
    plan = Planner(tools, lambda: None, PlanValidator(tools)).from_template(
        template,
        inputs,
        task_id="task-1",
    )
    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == inputs


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)

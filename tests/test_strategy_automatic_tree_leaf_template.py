from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_AUTOMATIC_TREE_LEAF_MATERIALIZATION,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


_PLATFORM_SLOTS = {
    "source_artifact_id",
    "expected_artifact_content_hash",
    "expected_asset_id",
    "expected_asset_hash",
    "expected_tree_result_hash",
}


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def _platform_slots() -> dict[str, str]:
    return {
        "source_artifact_id": "artifact-tree-json",
        "expected_artifact_content_hash": "a" * 64,
        "expected_asset_id": "candidate-asset-" + "b" * 32,
        "expected_asset_hash": "c" * 64,
        "expected_tree_result_hash": "d" * 64,
    }


def test_automatic_tree_leaf_materialization_is_one_nongated_builtin_step() -> None:
    template = STRATEGY_AUTOMATIC_TREE_LEAF_MATERIALIZATION

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_automatic_tree_leaf_materialization"
    assert template == get_template(template.id)
    assert len(template.steps) == 1

    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == _PLATFORM_SLOTS
    assert {name for name, source in slot_sources.items() if source == "user"} == {
        "leaf_id",
        "selection_reason",
    }
    assert {slot.name for slot in template.slots if slot.required} == {
        *_PLATFORM_SLOTS,
        "leaf_id",
    }
    assert "tree_asset_id" not in slot_sources

    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy", "materialize_automatic_tree_leaf_fragment"
    )
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert "tree_asset_id" not in step.inputs_template
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "selection_id"}),
        PostCheck("nonempty", {"field": "selection_hash"}),
        PostCheck("nonempty", {"field": "tree_asset_id"}),
        PostCheck("nonempty", {"field": "tree_asset_hash"}),
        PostCheck("nonempty", {"field": "tree_result_hash"}),
        PostCheck("nonempty", {"field": "leaf_id"}),
        PostCheck("nonempty", {"field": "fragment_id"}),
        PostCheck("nonempty", {"field": "fragment_hash"}),
        PostCheck("nonempty", {"field": "rule_id"}),
        PostCheck("nonempty", {"field": "effect_id"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_automatic_tree_leaf_materialization_binds_exact_tool_inputs(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    slots = {
        **_platform_slots(),
        "leaf_id": "leaf-node-7",
        "selection_reason": "用户明确选择该叶节点供后续评审",
    }

    plan = planner.from_template(
        get_template("strategy_automatic_tree_leaf_materialization"),
        slots,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == slots


def test_automatic_tree_leaf_materialization_omits_optional_reason(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    slots = {**_platform_slots(), "leaf_id": "leaf-node-7"}

    plan = planner.from_template(
        get_template("strategy_automatic_tree_leaf_materialization"),
        slots,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == slots

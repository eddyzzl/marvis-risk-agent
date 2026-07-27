from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_AUTOMATIC_TREE_APPLY
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
    "dataset_id",
    "expected_content_hash",
    "workspace_revision",
    "analysis_generation",
    "semantic_mapping_hash",
}


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _platform_slots() -> dict[str, object]:
    return {
        "source_artifact_id": "artifact-tree-json",
        "expected_artifact_content_hash": "a" * 64,
        "expected_asset_id": "candidate-asset-" + "b" * 32,
        "expected_asset_hash": "c" * 64,
        "expected_tree_result_hash": "d" * 64,
        "dataset_id": "dataset-1",
        "expected_content_hash": "e" * 64,
        "workspace_revision": 4,
        "analysis_generation": 5,
        "semantic_mapping_hash": "f" * 64,
    }


def test_automatic_tree_apply_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_AUTOMATIC_TREE_APPLY

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_automatic_tree_apply"
    assert template == get_template(template.id)
    assert len(template.steps) == 1

    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == _PLATFORM_SLOTS
    assert {name for name, source in slot_sources.items() if source == "user"} == {
        "leaf_id_column",
        "rule_id_column",
    }
    assert {slot.name for slot in template.slots if slot.required} == _PLATFORM_SLOTS
    assert "tree_asset_id" not in slot_sources

    [step] = template.steps
    assert step.tool_ref == ToolRef("strategy", "apply_automatic_tree")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.inputs_template["activate_result"] is False
    assert "tree_asset_id" not in step.inputs_template
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "run_id"}),
        PostCheck("nonempty", {"field": "source.asset_id"}),
        PostCheck("nonempty", {"field": "result.dataset_id"}),
        PostCheck("nonempty", {"field": "result.dataset_content_hash"}),
        PostCheck("nonempty", {"field": "columns.leaf_id"}),
        PostCheck("nonempty", {"field": "columns.rule_id"}),
        PostCheck("nonempty", {"field": "evidence.artifact_id"}),
        PostCheck("nonempty", {"field": "evidence.content_hash"}),
    )


def test_automatic_tree_apply_omits_user_columns_for_tool_defaults(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))

    plan = planner.from_template(
        get_template("strategy_automatic_tree_apply"),
        _platform_slots(),
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        **_platform_slots(),
        "activate_result": False,
    }


def test_automatic_tree_apply_passes_only_explicit_output_columns(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    slots = {
        **_platform_slots(),
        "leaf_id_column": "tree_leaf_bucket",
        "rule_id_column": "tree_rule_bucket",
    }

    plan = planner.from_template(
        get_template("strategy_automatic_tree_apply"),
        slots,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {**slots, "activate_result": False}

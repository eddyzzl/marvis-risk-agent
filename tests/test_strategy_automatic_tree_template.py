from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_AUTOMATIC_TREE_CANDIDATE_BUILD,
)
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


_PLATFORM_SLOTS = {
    "dataset_id",
    "expected_content_hash",
    "workspace_revision",
    "analysis_generation",
    "semantic_mapping_hash",
    "target_col",
    "sample_design_ref",
}
_OPTIONAL_CONTROL_SLOTS = {
    "drop_nan_labels",
    "sample_weight_col",
    "directions",
    "max_depth",
    "min_leaf_count",
    "min_weight_fraction_leaf",
    "seed",
    "loan_amount_col",
    "overdue_amount_col",
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


def _platform_slots() -> dict[str, object]:
    return {
        "dataset_id": "dataset-1",
        "expected_content_hash": "a" * 64,
        "workspace_revision": 0,
        "analysis_generation": 0,
        "semantic_mapping_hash": "b" * 64,
        "target_col": "bad",
        "sample_design_ref": {
            "artifact_id": "c" * 64,
            "artifact_content_hash": "d" * 64,
            "sample_design_id": "strategy-sample-design-1",
            "sample_design_content_hash": "e" * 64,
            "partition": "development",
        },
    }


def test_automatic_tree_candidate_build_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_AUTOMATIC_TREE_CANDIDATE_BUILD

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_automatic_tree_candidate_build"
    assert template == get_template(template.id)
    assert len(template.steps) == 1

    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == (_PLATFORM_SLOTS)
    assert {name for name, source in slot_sources.items() if source == "user"} == {
        "features",
        *_OPTIONAL_CONTROL_SLOTS,
    }
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_content_hash",
            "semantic_mapping_hash",
            "target_col",
            "sample_design_ref",
            "features",
    }

    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "build_automatic_tree_candidate")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "summary.asset_id"}),
        PostCheck("nonempty", {"field": "summary.asset_hash"}),
        PostCheck("nonempty", {"field": "summary.tree_id"}),
        PostCheck("nonempty", {"field": "summary.tree_result_hash"}),
        PostCheck("nonempty", {"field": "leaf_index"}),
        PostCheck("nonempty", {"field": "artifacts"}),
        PostCheck(
            "schema",
            {
                "schema": {
                    "type": "object",
                    "properties": {"report_info_gaps": {"type": "array"}},
                    "required": ["report_info_gaps"],
                }
            },
        ),
    )


def test_automatic_tree_candidate_build_omits_optional_controls_for_tool_defaults(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))

    plan = planner.from_template(
        get_template("strategy_automatic_tree_candidate_build"),
        {**_platform_slots(), "features": ["score", "age"]},
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        **_platform_slots(),
        "features": ["score", "age"],
    }


def test_automatic_tree_candidate_build_passes_only_manifest_supported_controls(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    controls = {
        "drop_nan_labels": True,
        "sample_weight_col": "sample_weight",
        "directions": {"score": "decreasing", "age": "unordered"},
        "max_depth": 5,
        "min_leaf_count": 100,
        "min_weight_fraction_leaf": 0.01,
        "seed": 17,
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
    }

    plan = planner.from_template(
        get_template("strategy_automatic_tree_candidate_build"),
        {
            **_platform_slots(),
            "features": ["score", "age"],
            **controls,
        },
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        **_platform_slots(),
        "features": ["score", "age"],
        **controls,
    }
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("strategy", "build_automatic_tree_candidate")
    ]

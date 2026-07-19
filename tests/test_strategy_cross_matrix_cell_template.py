from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_CROSS_MATRIX_CELL_SELECTION,
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
    "expected_candidate_id",
    "expected_evidence_hash",
}


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _platform_slots() -> dict[str, str]:
    return {
        "source_artifact_id": "artifact-cross-json",
        "expected_artifact_content_hash": "a" * 64,
        "expected_asset_id": "candidate-asset-" + "b" * 32,
        "expected_asset_hash": "c" * 64,
        "expected_candidate_id": "candidate-" + "d" * 32,
        "expected_evidence_hash": "e" * 64,
    }


def test_cross_cell_selection_is_one_nongated_builtin_step() -> None:
    template = STRATEGY_CROSS_MATRIX_CELL_SELECTION

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_cross_matrix_cell_selection"
    assert template == get_template(template.id)
    assert len(template.steps) == 1

    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == _PLATFORM_SLOTS
    assert {name for name, source in slot_sources.items() if source == "user"} == {
        "cell_ids",
        "selection_reason",
    }
    assert {slot.name for slot in template.slots if slot.required} == {
        *_PLATFORM_SLOTS,
        "cell_ids",
    }
    assert "cross_asset_id" not in slot_sources

    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy", "materialize_cross_matrix_cell_selection"
    )
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert "cross_asset_id" not in step.inputs_template
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "selection_id"}),
        PostCheck("nonempty", {"field": "selection_hash"}),
        PostCheck("nonempty", {"field": "group_id"}),
        PostCheck("nonempty", {"field": "cell_ids"}),
        PostCheck("nonempty", {"field": "source_asset_id"}),
        PostCheck("nonempty", {"field": "source_asset_hash"}),
        PostCheck("nonempty", {"field": "source_candidate_id"}),
        PostCheck("nonempty", {"field": "source_evidence_hash"}),
        PostCheck("nonempty", {"field": "fragment_id"}),
        PostCheck("nonempty", {"field": "rule_id"}),
        PostCheck("nonempty", {"field": "effect_id"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_cross_cell_selection_binds_exact_tool_inputs(tmp_path: Path) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    slots = {
        **_platform_slots(),
        "cell_ids": ["cross-cell-" + "1" * 32, "cross-cell-" + "2" * 32],
        "selection_reason": "人工确认用于风险复核",
    }

    plan = planner.from_template(
        get_template("strategy_cross_matrix_cell_selection"),
        slots,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == slots


def test_cross_cell_selection_omits_optional_reason(tmp_path: Path) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    slots = {
        **_platform_slots(),
        "cell_ids": ["cross-cell-" + "1" * 32],
    }

    plan = planner.from_template(
        get_template("strategy_cross_matrix_cell_selection"),
        slots,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == slots

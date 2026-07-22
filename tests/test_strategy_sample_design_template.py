from __future__ import annotations

from pathlib import Path

import pytest

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner, PlanningError
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_SAMPLE_DESIGN
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


_PLATFORM_SLOTS = {
    "dataset_id",
    "expected_dataset_content_hash",
    "workspace_revision",
    "workspace_generation",
    "semantic_mapping_hash",
    "target_col",
}
_USER_SLOTS = {
    "performance_window_status",
    "performance_window_days",
    "observation_window_status",
    "observation_start",
    "observation_end",
    "maturity_status",
    "target_bad_value",
    "split_col",
    "development_values",
    "validation_values",
    "oot_values",
    "month_col",
    "weight_col",
    "loan_amount_col",
    "overdue_amount_col",
    "drop_nan_labels",
}


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _base_slots() -> dict[str, object]:
    return {
        "dataset_id": "dataset-1",
        "expected_dataset_content_hash": "a" * 64,
        "workspace_revision": 0,
        "workspace_generation": 0,
        "semantic_mapping_hash": "b" * 64,
        "target_col": "bad",
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_start": "2025-01-01",
        "observation_end": "2025-12-31",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": False,
    }


def test_sample_design_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_SAMPLE_DESIGN

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_sample_design"
    assert template == get_template(template.id)
    assert len(template.steps) == 1
    slot_sources = {slot.name: slot.source for slot in template.slots}
    assert {name for name, source in slot_sources.items() if source == "task_context"} == (
        _PLATFORM_SLOTS
    )
    assert {name for name, source in slot_sources.items() if source == "user"} == (
        _USER_SLOTS
    )
    assert {slot.name for slot in template.slots if slot.required} == {
        "dataset_id",
        "expected_dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "target_col",
        "performance_window_status",
        "observation_window_status",
        "maturity_status",
        "target_bad_value",
        "drop_nan_labels",
    }
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "materialize_sample_design")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "sample_design_id"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "bundle"}),
        PostCheck("nonempty", {"field": "artifact"}),
    )


def test_sample_design_template_maps_user_dates_to_tool_window_fields(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))

    plan = planner.from_template(
        get_template("strategy_sample_design"),
        _base_slots(),
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        "dataset_id": "dataset-1",
        "expected_dataset_content_hash": "a" * 64,
        "workspace_revision": 0,
        "workspace_generation": 0,
        "semantic_mapping_hash": "b" * 64,
        "target_col": "bad",
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_window_start": "2025-01-01",
        "observation_window_end": "2025-12-31",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": False,
    }


def test_sample_design_template_passes_exact_optional_split_and_metric_columns(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    controls = {
        "split_col": "sample_role",
        "development_values": ["dev"],
        "validation_values": ["validation"],
        "oot_values": ["oot"],
        "month_col": "month",
        "weight_col": "weight",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
    }

    plan = planner.from_template(
        get_template("strategy_sample_design"),
        {**_base_slots(), **controls},
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        **{
            key: value
            for key, value in _base_slots().items()
            if key not in {"observation_start", "observation_end"}
        },
        "observation_window_start": "2025-01-01",
        "observation_window_end": "2025-12-31",
        **controls,
    }


@pytest.mark.parametrize(
    "field",
    [
        "workspace_revision",
        "workspace_generation",
        "target_bad_value",
        "drop_nan_labels",
    ],
)
def test_sample_design_required_slots_use_presence_not_truthiness(
    tmp_path: Path,
    field: str,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    slots = _base_slots()
    slots.pop(field)

    with pytest.raises(PlanningError, match=field):
        planner.from_template(
            get_template("strategy_sample_design"),
            slots,
            task_id="task-1",
        )

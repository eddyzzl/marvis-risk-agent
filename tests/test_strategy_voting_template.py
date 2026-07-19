from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_VOTING_CANDIDATE_BUILD
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _slots() -> dict[str, object]:
    return {
        "strategy_type": "approval",
        "expected_pool_revision": 4,
        "expected_pool_snapshot_hash": "a" * 64,
        "selected_entry_ids": ["pool-entry-1", "pool-entry-2", "pool-entry-3"],
        "n": 2,
    }


def test_voting_candidate_build_is_one_nongated_builtin_tool_step() -> None:
    template = STRATEGY_VOTING_CANDIDATE_BUILD

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_voting_candidate_build"
    assert template == get_template(template.id)
    assert len(template.steps) == 1
    assert {slot.name for slot in template.slots if slot.source == "task_context"} == {
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "selected_entry_ids",
    }
    assert {slot.name for slot in template.slots if slot.source == "user"} == {
        "strategy_type",
        "n",
    }
    assert all(slot.required for slot in template.slots)

    [step] = template.steps
    assert step.tool_ref == ToolRef("strategy", "build_voting_candidate")
    assert step.inputs_template == {
        key: f"{{slot:{key}}}" for key in _slots()
    }
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "asset_id"}),
        PostCheck("nonempty", {"field": "asset_hash"}),
        PostCheck("nonempty", {"field": "candidate_id"}),
        PostCheck("nonempty", {"field": "evidence_hash"}),
        PostCheck("nonempty", {"field": "fragment_id"}),
        PostCheck("nonempty", {"field": "effect_id"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_voting_candidate_template_instantiates_against_real_manifest(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_voting_candidate_build"),
        _slots(),
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert plan.steps[0].inputs == _slots()
    assert plan.steps[0].tool_ref == ToolRef("strategy", "build_voting_candidate")

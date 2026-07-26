"""Plan template contract for exact Voting search-result materialization."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_VOTING_CANDIDATE_BUILD_FROM_SEARCH,
)
from marvis.plugins.manifest import ToolRef
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.orchestrator.validator import PlanValidator


SEARCH_ID = "voting-search-" + "a" * 32
COMBO_ID = "voting-combo-" + "b" * 32


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def test_voting_search_selection_is_one_pointer_only_builtin_tool_step() -> None:
    template = STRATEGY_VOTING_CANDIDATE_BUILD_FROM_SEARCH

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_voting_candidate_build_from_search"
    assert template == get_template(template.id)
    assert {slot.name for slot in template.slots if slot.source == "user"} == {
        "search_id",
        "combo_id",
        "strategy_type",
    }
    assert {slot.name for slot in template.slots if slot.required} == {
        "search_id",
        "combo_id",
    }
    assert all(slot.source == "user" for slot in template.slots)

    [step] = template.steps
    assert step.tool_ref == ToolRef(
        "strategy",
        "build_voting_candidate_from_search",
    )
    assert step.inputs_template == {
        "search_id": "{slot:search_id}",
        "combo_id": "{slot:combo_id}",
        "strategy_type": "{slot:strategy_type}",
    }
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "voting_candidate.asset_id"}),
        PostCheck("nonempty", {"field": "voting_candidate.asset_hash"}),
        PostCheck("nonempty", {"field": "voting_candidate.candidate_id"}),
        PostCheck("nonempty", {"field": "voting_candidate.evidence_hash"}),
        PostCheck("nonempty", {"field": "voting_candidate.fragment_id"}),
        PostCheck("nonempty", {"field": "voting_candidate.effect_id"}),
        PostCheck("nonempty", {"field": "source_search_selection.search_id"}),
        PostCheck("nonempty", {"field": "source_search_selection.combo_id"}),
        PostCheck("nonempty", {"field": "voting_candidate.artifacts"}),
    )


def test_voting_search_selection_template_instantiates_pointer_only_inputs(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    planner = Planner(tools, lambda: None, validator)
    template = get_template("strategy_voting_candidate_build_from_search")

    without_type = planner.from_template(
        template,
        {"search_id": SEARCH_ID, "combo_id": COMBO_ID},
        task_id="task-1",
    )
    with_type = planner.from_template(
        template,
        {
            "search_id": SEARCH_ID,
            "combo_id": COMBO_ID,
            "strategy_type": "approval",
        },
        task_id="task-1",
    )

    assert validator.validate(without_type) == []
    assert validator.validate(with_type) == []
    assert without_type.steps[0].inputs == {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
    }
    assert with_type.steps[0].inputs == {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
        "strategy_type": "approval",
    }
    assert {
        "member_rule_ids",
        "selected_entry_ids",
        "n",
        "rank",
        "pool_ref",
    }.isdisjoint(with_type.steps[0].inputs)

"""Plan template contract for read-only Voting candidate search."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_VOTING_CANDIDATE_SEARCH,
)
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
        "pool_ref": {
            "artifact_id": "a" * 64,
            "expected_artifact_content_hash": "b" * 64,
            "expected_pool_id": "strategy-pool-task-1-approval",
            "expected_revision": 4,
            "expected_revision_id": "strategy-pool-revision-4",
            "expected_snapshot_hash": "c" * 64,
        },
        "member_count": 3,
        "n": 2,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
        "constraints": [],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 10_000,
    }


def test_voting_search_is_one_read_only_builtin_tool_step() -> None:
    template = STRATEGY_VOTING_CANDIDATE_SEARCH

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_voting_candidate_search"
    assert template == get_template(template.id)
    assert len(template.steps) == 1
    assert {slot.name for slot in template.slots if slot.source == "task_context"} == {
        "pool_ref",
    }
    assert {slot.name for slot in template.slots if slot.source == "user"} == {
        "strategy_type",
        "member_count",
        "n",
        "objective",
        "constraints",
        "include_rule_ids",
        "exclude_rule_ids",
        "max_combinations",
    }
    assert all(slot.required for slot in template.slots)

    [step] = template.steps
    assert step.tool_ref == ToolRef("strategy", "search_voting_candidates")
    assert step.inputs_template == {key: f"{{slot:{key}}}" for key in _slots()}
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "search_id"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_voting_search_template_validates_against_real_manifest(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_voting_candidate_search"),
        _slots(),
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert plan.steps[0].inputs == _slots()
    assert plan.steps[0].tool_ref == ToolRef(
        "strategy",
        "search_voting_candidates",
    )

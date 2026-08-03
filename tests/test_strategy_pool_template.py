"""Narrow Strategy Pool Workflow template contracts."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def test_strategy_pool_templates_are_registered_with_one_narrow_tool_each() -> None:
    load_builtin_templates()
    expected = {
        "strategy_pool_add_candidate": "add_candidate_to_pool",
        "strategy_pool_remove_entry": "remove_pool_entry",
        "strategy_pool_set_action": "set_pool_entry_action",
        "strategy_pool_reorder": "reorder_strategy_pool",
        "strategy_pool_compile": "compile_strategy_pool",
        "strategy_pool_materialize": "materialize_strategy_from_pool",
    }

    for template_id, tool_name in expected.items():
        template = get_template(template_id)
        assert len(template.steps) == 1
        assert template.steps[0].tool_ref == ToolRef("strategy", tool_name)


def test_pool_draft_mutations_and_compile_do_not_require_human_confirmation() -> None:
    load_builtin_templates()
    mutations = (
        "strategy_pool_add_candidate",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
    )
    assert all(
        get_template(item).steps[0].needs_confirmation is False for item in mutations
    )
    assert get_template("strategy_pool_compile").steps[0].needs_confirmation is False
    assert (
        get_template("strategy_pool_materialize").steps[0].needs_confirmation
        is False
    )


def test_pool_templates_bind_only_platform_resolved_integrity_fields() -> None:
    load_builtin_templates()
    add = get_template("strategy_pool_add_candidate").steps[0].inputs_template
    assert add["source_artifact_id"] == "{slot:source_artifact_id}"
    assert add["expected_artifact_content_hash"] == (
        "{slot:expected_artifact_content_hash}"
    )
    assert add["expected_asset_id"] == "{slot:expected_asset_id}"
    assert add["expected_asset_hash"] == "{slot:expected_asset_hash}"

    for template_id in (
        "strategy_pool_add_candidate",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_compile",
        "strategy_pool_materialize",
    ):
        inputs = get_template(template_id).steps[0].inputs_template
        assert inputs["expected_pool_revision"] == "{slot:expected_pool_revision}"
        assert inputs["expected_pool_snapshot_hash"] == (
            "{slot:expected_pool_snapshot_hash}"
        )


def test_pool_templates_instantiate_against_the_real_tool_manifests(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    tools = ToolRegistry(plugins)
    validator = PlanValidator(tools)
    planner = Planner(tools, lambda: None, validator)
    common = {
        "strategy_type": "approval",
        "expected_pool_revision": 2,
        "expected_pool_snapshot_hash": "a" * 64,
    }
    slots_by_template = {
        "strategy_pool_add_candidate": {
            **common,
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": "9" * 64,
            "source_artifact_id": "artifact-1",
            "expected_artifact_content_hash": "b" * 64,
            "expected_asset_id": "scorecard-band-asset-" + "c" * 32,
            "expected_asset_hash": "d" * 64,
            "default_action": {"type": "approval", "value": "approve"},
            "action": {"type": "reject", "value": "reject"},
            "placement_mode": "append",
        },
        "strategy_pool_remove_entry": {
            **common,
            "rule_id": "candidate-rule-" + "e" * 32,
        },
        "strategy_pool_set_action": {
            **common,
            "rule_id": "candidate-rule-" + "e" * 32,
            "action": {"type": "review", "value": "review"},
        },
        "strategy_pool_reorder": {
            **common,
            "ordered_rule_ids": ["candidate-rule-" + "e" * 32],
        },
        "strategy_pool_compile": common,
        "strategy_pool_materialize": {
            **common,
            "expected_pool_artifact_id": "b" * 64,
            "expected_pool_artifact_content_hash": "c" * 64,
            "expected_design_hash": "d" * 64,
        },
    }

    for template_id, slots in slots_by_template.items():
        plan = planner.from_template(
            get_template(template_id),
            slots,
            task_id="task-1",
        )
        assert validator.validate(plan) == []
        assert plan.steps[0].needs_confirmation is False

    unknown_asset = planner.from_template(
        get_template("strategy_pool_add_candidate"),
        {
            **slots_by_template["strategy_pool_add_candidate"],
            "expected_asset_id": "unknown-asset-" + "f" * 32,
        },
        task_id="task-1",
    )
    assert any(
        "schema validation failed" in problem
        for problem in validator.validate(unknown_asset)
    )

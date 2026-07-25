"""Builtin Workflow contract for StrategyReportBundle V2."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_REPORT_BUNDLE_V2
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs, load_manifest
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def _report_slots(*, candidate_stability_ref: dict | None) -> dict[str, object]:
    return {
        "title": "策略迭代评审报告",
        "status": "partial",
        "project_context_ref": {
            "artifact_id": "a" * 64,
            "expected_artifact_content_hash": "b" * 64,
            "expected_revision": 1,
            "expected_revision_id": "strategy-project-context-revision-" + "c" * 24,
            "expected_state_hash": "d" * 64,
        },
        "sample_design_ref": {
            "membership_artifact_id": "e" * 64,
            "expected_membership_artifact_content_hash": "f" * 64,
            "bundle_artifact_id": "1" * 64,
            "expected_bundle_artifact_content_hash": "2" * 64,
            "expected_bundle_id": "sample-design-bundle-1",
            "expected_sample_design_id": "sample-design-1",
            "expected_sample_design_content_hash": "3" * 64,
        },
        "candidate_pool_ref": {
            "strategy_type": "approval",
            "expected_pool_revision": 1,
            "expected_pool_snapshot_hash": "4" * 64,
            "expected_artifact_id": "5" * 64,
            "expected_artifact_content_hash": "6" * 64,
        },
        "impact_cube_ref": {
            "artifact_id": "7" * 64,
            "expected_artifact_content_hash": "8" * 64,
            "expected_cube_id": "strategy-impact-cube-" + "9" * 24,
            "expected_cube_content_hash": "a" * 64,
        },
        "candidate_stability_ref": candidate_stability_ref,
        "report_revision": 1,
        "generated_at": "2026-07-25T00:00:00Z",
    }


def test_report_bundle_is_one_reversible_nongated_tool_step() -> None:
    template = STRATEGY_REPORT_BUNDLE_V2

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_report_bundle_v2"
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "build_report_bundle_v2")
    assert step.depends_on_titles == ()
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert step.post_checks == (
        PostCheck("nonempty", {"field": "report_id"}),
        PostCheck("nonempty", {"field": "report_revision"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )


def test_report_bundle_exposes_only_title_and_status_as_user_slots() -> None:
    template = STRATEGY_REPORT_BUNDLE_V2
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert {
        name for name, source in slot_sources.items() if source == "user"
    } == {"title", "status"}
    assert {
        name for name, source in slot_sources.items() if source == "task_context"
    } == {
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
        "pool_impact_ref",
        "impact_cube_ref",
        "candidate_stability_ref",
        "report_revision",
        "previous_report_id",
        "previous_report_content_hash",
        "generated_at",
        "strategy_identity",
        "model_evidence_ref",
        "training_evidence_ref",
        "score_evidence_ref",
    }
    assert set(template.steps[0].inputs_template) == set(slot_sources)
    assert not {
        "metrics",
        "strategy_spec",
        "create_strategy",
        "adopt",
        "deploy",
    } & set(template.steps[0].inputs_template)


def test_report_bundle_manifest_requires_one_impact_source_and_five_types() -> None:
    tool = next(
        item
        for item in load_manifest(
            Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
            builtin=True,
        ).tools
        if item.name == "build_report_bundle_v2"
    )
    schema = tool.input_schema

    assert schema["anyOf"] == [
        {"required": ["impact_cube_ref"]},
        {"required": ["pool_impact_ref"]},
    ]
    assert set(
        schema["$defs"]["candidate_pool_ref"]["properties"][
            "strategy_type"
        ]["enum"]
    ) == {"approval", "reject", "limit", "pricing", "segmentation"}
    assert (
        schema["$defs"]["candidate_stability_ref"]["properties"][
            "expected_stability_id"
        ]["pattern"]
        == "^candidate-stability-[0-9a-f]{24}$"
    )
    assert "pool_impact_ref" not in schema["required"]
    assert "impact_cube_ref" not in schema["required"]


def test_report_bundle_passes_exact_candidate_stability_task_context(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    candidate_stability_ref = {
        "artifact_id": "b" * 64,
        "expected_artifact_content_hash": "c" * 64,
        "expected_stability_id": "candidate-stability-" + ("1" * 24),
        "expected_stability_content_hash": "d" * 64,
    }
    slots = _report_slots(candidate_stability_ref=candidate_stability_ref)

    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_report_bundle_v2"),
        slots,
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert plan.steps[0].inputs["candidate_stability_ref"] == candidate_stability_ref


def test_report_bundle_omits_none_candidate_stability_task_context(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)

    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_report_bundle_v2"),
        _report_slots(candidate_stability_ref=None),
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert "candidate_stability_ref" not in plan.steps[0].inputs

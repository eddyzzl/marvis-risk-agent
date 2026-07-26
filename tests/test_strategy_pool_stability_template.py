"""Builtin Workflow contract for current-Pool cross-partition stability."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_POOL_STABILITY
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
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


def _slots() -> dict[str, object]:
    return {
        "strategy_type": "pricing",
        "pool_ref": {
            "artifact_id": "1" * 64,
            "expected_artifact_content_hash": "2" * 64,
            "expected_pool_id": "strategy-pool-" + ("3" * 32),
            "expected_revision": 7,
            "expected_revision_id": (
                "strategy-pool-revision-" + ("4" * 32)
            ),
            "expected_snapshot_hash": "5" * 64,
        },
        "sample_design_ref": {
            "membership_artifact_id": "6" * 64,
            "expected_membership_artifact_content_hash": "7" * 64,
            "bundle_artifact_id": "8" * 64,
            "expected_bundle_artifact_content_hash": "9" * 64,
            "expected_bundle_id": "sample-bundle-1",
            "expected_sample_design_id": "sample-design-1",
            "expected_sample_design_content_hash": "a" * 64,
        },
        "partitions": ["development", "validation", "oot"],
    }


def test_pool_stability_is_one_exact_two_step_read_only_workflow() -> None:
    template = STRATEGY_POOL_STABILITY

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_pool_stability"
    assert template.default_autonomy == 1
    assert len(template.steps) == 2

    cube_step, stability_step = template.steps
    assert cube_step.title == "生成策略池稳定性基准"
    assert cube_step.tool_ref == ToolRef(
        "strategy",
        "measure_strategy_impact_cube",
    )
    assert cube_step.depends_on_titles == ()
    assert cube_step.needs_confirmation is False
    assert cube_step.decision_point is False
    assert cube_step.post_checks == (
        PostCheck("nonempty", {"field": "cube_id"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifact.artifact_id"}),
    )

    assert stability_step.title == "测量策略池跨分区稳定性"
    assert stability_step.tool_ref == ToolRef(
        "strategy",
        "measure_strategy_pool_stability",
    )
    assert stability_step.depends_on_titles == ("生成策略池稳定性基准",)
    assert stability_step.needs_confirmation is False
    assert stability_step.decision_point is False
    assert stability_step.post_checks == (
        PostCheck("nonempty", {"field": "stability_id"}),
        PostCheck("nonempty", {"field": "content_hash"}),
        PostCheck("nonempty", {"field": "artifact.artifact_id"}),
        PostCheck("nonempty", {"field": "comparison_partitions"}),
    )


def test_pool_stability_freezes_sources_then_passes_only_direct_step_refs() -> None:
    template = STRATEGY_POOL_STABILITY
    slot_sources = {slot.name: slot.source for slot in template.slots}

    assert slot_sources == {
        "strategy_type": "user",
        "pool_ref": "task_context",
        "sample_design_ref": "task_context",
        "partitions": "task_context",
    }
    cube_step, stability_step = template.steps
    assert cube_step.inputs_template == {
        "strategy_type": "{slot:strategy_type}",
        "pool_ref": "{slot:pool_ref}",
        "sample_design_ref": "{slot:sample_design_ref}",
        "partitions": "{slot:partitions}",
        "population": "risk",
        "dimension_bindings": {
            "month_col": None,
            "group_col": None,
            "segment_col": None,
        },
    }
    assert stability_step.inputs_template == {
        "artifact_id": (
            "$ref:生成策略池稳定性基准.output.artifact.artifact_id"
        ),
        "expected_artifact_content_hash": (
            "$ref:生成策略池稳定性基准.output.artifact.content_hash"
        ),
        "expected_cube_id": (
            "$ref:生成策略池稳定性基准.output.cube_id"
        ),
        "expected_cube_content_hash": (
            "$ref:生成策略池稳定性基准.output.content_hash"
        ),
    }
    assert not {
        "latest_artifact",
        "raw_cube",
        "metrics",
        "thresholds",
        "create_strategy",
        "adopt",
        "promote",
        "deploy",
    } & set(stability_step.inputs_template)


def test_pool_stability_two_step_refs_validate_against_real_manifests(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)

    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_pool_stability"),
        _slots(),
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert len(plan.steps) == 2
    cube_step, stability_step = plan.steps
    assert cube_step.inputs["pool_ref"] == _slots()["pool_ref"]
    assert cube_step.inputs["dimension_bindings"] == {
        "month_col": None,
        "group_col": None,
        "segment_col": None,
    }
    assert stability_step.inputs == {
        "artifact_id": (
            f"$ref:{cube_step.id}.output.artifact.artifact_id"
        ),
        "expected_artifact_content_hash": (
            f"$ref:{cube_step.id}.output.artifact.content_hash"
        ),
        "expected_cube_id": f"$ref:{cube_step.id}.output.cube_id",
        "expected_cube_content_hash": (
            f"$ref:{cube_step.id}.output.content_hash"
        ),
    }
    assert stability_step.depends_on == [cube_step.id]

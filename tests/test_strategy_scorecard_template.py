"""Builtin one-step Workflow templates for scorecard candidate operations."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry


SCORE_REF = {
    "evidence_artifact_id": "1" * 64,
    "expected_evidence_artifact_content_hash": "2" * 64,
    "score_vector_artifact_id": "3" * 64,
    "expected_score_vector_artifact_content_hash": "4" * 64,
}
SAMPLE_REF = {
    "membership_artifact_id": "5" * 64,
    "expected_membership_artifact_content_hash": "6" * 64,
    "bundle_artifact_id": "7" * 64,
    "expected_bundle_artifact_content_hash": "8" * 64,
    "expected_bundle_id": "sample-bundle-" + "a" * 32,
    "expected_sample_design_id": "sample-design-" + "b" * 32,
    "expected_sample_design_content_hash": "9" * 64,
}


def _tools(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    return ToolRegistry(plugins)


def test_scorecard_templates_are_registered_as_narrow_nongated_steps() -> None:
    load_builtin_templates()
    expected = {
        "strategy_scorecard_band_build": (
            "build_scorecard_band_asset",
            (
                PostCheck("nonempty", {"field": "asset_id"}),
                PostCheck("nonempty", {"field": "asset_hash"}),
                PostCheck("nonempty", {"field": "scorecard_band_asset"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
        ),
        "strategy_scorecard_cutoff_selection": (
            "materialize_scorecard_cutoff_selection",
            (
                PostCheck("nonempty", {"field": "selection_id"}),
                PostCheck("nonempty", {"field": "selection_hash"}),
                PostCheck("nonempty", {"field": "cutoff_id"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
        ),
    }

    for template_id, (tool_name, post_checks) in expected.items():
        template = get_template(template_id)
        assert template in BUILTIN_TEMPLATES
        assert len(template.steps) == 1
        [step] = template.steps
        assert step.tool_ref == ToolRef("strategy", tool_name)
        assert step.post_checks == post_checks
        assert step.needs_confirmation is False
        assert step.decision_point is False


def test_scorecard_model_score_evidence_template_chains_governed_tools(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tools(tmp_path)
    template = get_template("strategy_scorecard_model_score_evidence_build")

    assert template in BUILTIN_TEMPLATES
    assert [step.tool_ref for step in template.steps] == [
        ToolRef("modeling", "train_model_with_evidence_v2"),
        ToolRef("modeling", "materialize_model_score_evidence_v2"),
    ]
    assert template.steps[1].depends_on_titles == ("训练受治理 Scorecard",)
    assert all(not step.needs_confirmation for step in template.steps)

    plan = Planner(tools, lambda: None, PlanValidator(tools)).from_template(
        template,
        {
            "sample_design_ref": SAMPLE_REF,
            "features": ["age", "income"],
            "params": {"max_iter": 200, "scorecard_max_bins": 4},
            "seed": 23,
        },
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs["recipe"] == "scorecard"
    assert plan.steps[0].inputs["sample_design_ref"] == SAMPLE_REF
    training_ref = plan.steps[1].inputs["training_evidence_ref"]
    assert training_ref["sample_design_ref"] == (
        f"$ref:{plan.steps[0].id}.output.sample_design_ref"
    )
    assert training_ref["expected_experiment_id"] == (
        f"$ref:{plan.steps[0].id}.output.experiment_id"
    )


def test_scorecard_band_template_omits_default_banding_and_maps_bin_count(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tools(tmp_path)
    planner = Planner(tools, lambda: None, PlanValidator(tools))
    template = get_template("strategy_scorecard_band_build")

    default_plan = planner.from_template(
        template,
        {
            "score_evidence_ref": SCORE_REF,
            "sample_design_ref": SAMPLE_REF,
        },
        task_id="task-1",
    )
    explicit_plan = planner.from_template(
        template,
        {
            "score_evidence_ref": SCORE_REF,
            "sample_design_ref": SAMPLE_REF,
            "banding": {"method": "equal_frequency", "bin_count": 7},
        },
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(default_plan) == []
    assert default_plan.steps[0].inputs == {
        "score_evidence_ref": SCORE_REF,
        "sample_design_ref": SAMPLE_REF,
    }
    assert PlanValidator(tools).validate(explicit_plan) == []
    assert explicit_plan.steps[0].inputs["banding"] == {
        "method": "equal_frequency",
        "bin_count": 7,
    }


def test_scorecard_band_template_passes_manual_edges_exclusively(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tools(tmp_path)
    plan = Planner(tools, lambda: None, PlanValidator(tools)).from_template(
        get_template("strategy_scorecard_band_build"),
        {
            "score_evidence_ref": SCORE_REF,
            "sample_design_ref": SAMPLE_REF,
            "raw_pd_band_edges": [0.0, 0.2, 0.6, 1.0],
        },
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == {
        "score_evidence_ref": SCORE_REF,
        "sample_design_ref": SAMPLE_REF,
        "raw_pd_band_edges": [0.0, 0.2, 0.6, 1.0],
    }


def test_scorecard_cutoff_template_passes_only_server_bound_source(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tools(tmp_path)
    slots = {
        "source_artifact_id": "a" * 64,
        "expected_source_artifact_content_hash": "b" * 64,
        "expected_asset_id": "scorecard-band-asset-" + "c" * 32,
        "expected_asset_hash": "d" * 64,
        "cutoff_id": "scorecard-cutoff-" + "e" * 32,
        "reason": "人工确认进入后续影响评审",
    }
    plan = Planner(tools, lambda: None, PlanValidator(tools)).from_template(
        get_template("strategy_scorecard_cutoff_selection"),
        slots,
        task_id="task-1",
    )

    assert PlanValidator(tools).validate(plan) == []
    assert plan.steps[0].inputs == slots

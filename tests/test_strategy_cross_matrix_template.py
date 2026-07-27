from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_CROSS_MATRIX_ANALYSIS
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
        "dataset_id": "dataset-1",
        "expected_content_hash": "a" * 64,
        "workspace_revision": 2,
        "analysis_generation": 3,
        "semantic_mapping_hash": "b" * 64,
        "target_col": "bad",
        "sample_design_ref": {
            "artifact_id": "c" * 64,
            "artifact_content_hash": "d" * 64,
            "sample_design_id": "strategy-sample-design-1",
            "sample_design_content_hash": "e" * 64,
            "partition": "development",
        },
        "drop_nan_labels": False,
        "features": ["age", "score"],
        "methods": ["equal_frequency", "equal_width"],
        "bin_count": 5,
        "min_bin_pct": 0.02,
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "sentinel_values": [],
        "x_feature": "age",
        "x_method": "equal_frequency",
        "y_feature": "score",
        "y_method": "equal_width",
    }


def test_cross_matrix_is_one_governed_two_step_builtin_workflow() -> None:
    template = STRATEGY_CROSS_MATRIX_ANALYSIS

    assert template in BUILTIN_TEMPLATES
    assert template.id == "strategy_cross_matrix_analysis"
    assert template == get_template(template.id)
    assert len(template.steps) == 2
    analyze, cross = template.steps
    assert analyze.tool_ref == ToolRef("strategy", "analyze_univariate_candidates")
    assert cross.tool_ref == ToolRef("strategy", "build_cross_matrix_candidate")
    assert cross.depends_on_titles == ("分析单变量候选",)
    assert cross.inputs_template == {
        "source_artifact_id": "$ref:分析单变量候选.output.artifacts.0.artifact_id",
        "expected_artifact_content_hash": (
            "$ref:分析单变量候选.output.artifacts.0.content_hash"
        ),
        "expected_candidate_id": "$ref:分析单变量候选.output.candidate_id",
        "expected_evidence_hash": "$ref:分析单变量候选.output.evidence_hash",
        "x_feature": "{slot:x_feature}",
        "x_method": "{slot:x_method}",
        "y_feature": "{slot:y_feature}",
        "y_method": "{slot:y_method}",
    }
    assert cross.post_checks == (
        PostCheck("nonempty", {"field": "asset_id"}),
        PostCheck("nonempty", {"field": "asset_hash"}),
        PostCheck("nonempty", {"field": "cell_count"}),
        PostCheck("nonempty", {"field": "artifacts"}),
    )
    assert cross.needs_confirmation is False


def test_cross_matrix_template_instantiates_against_real_manifest(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    plan = Planner(tools, lambda: None, validator).from_template(
        get_template("strategy_cross_matrix_analysis"),
        _slots(),
        task_id="task-1",
    )

    assert validator.validate(plan) == []
    assert [step.tool_ref for step in plan.steps] == [
        ToolRef("strategy", "analyze_univariate_candidates"),
        ToolRef("strategy", "build_cross_matrix_candidate"),
    ]

"""Template wiring for V2 SampleDesign and ModelEvidence workflows."""

from __future__ import annotations

from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import (
    STRATEGY_MODEL_EVIDENCE_V2,
    STRATEGY_SAMPLE_DESIGN,
    STRATEGY_SAMPLE_DESIGN_V2,
    STRATEGY_SAMPLE_DESIGN_V2_NATIVE,
)
from marvis.plugins.manifest import ToolRef
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.orchestrator.validator import PlanValidator


def _tool_registry(tmp_path: Path) -> ToolRegistry:
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    return ToolRegistry(plugins)


def _eq(column: str, value: str) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _policy() -> dict:
    return {
        "minimum_partition_count": 1,
        "minimum_bad_count": 1,
        "minimum_label_coverage": 0.8,
        "minimum_historical_score_coverage": 0.8,
        "maximum_group_coverage_gap": 0.2,
        "diagnostic_severities": {
            "entity_overlap": "fail",
            "temporal_oot": "fail",
            "risk_outside_approval": "fail",
            "maturity": "fail",
            "label_coverage": "fail",
            "historical_score_coverage": "warn",
            "group_coverage_gap": "warn",
            "sufficiency": "fail",
        },
    }


def test_four_sample_and_evidence_templates_are_registered_without_replacing_v1() -> None:
    load_builtin_templates()

    assert STRATEGY_SAMPLE_DESIGN in BUILTIN_TEMPLATES
    assert STRATEGY_SAMPLE_DESIGN_V2 in BUILTIN_TEMPLATES
    assert STRATEGY_SAMPLE_DESIGN_V2_NATIVE in BUILTIN_TEMPLATES
    assert STRATEGY_MODEL_EVIDENCE_V2 in BUILTIN_TEMPLATES
    assert get_template("strategy_sample_design") == STRATEGY_SAMPLE_DESIGN
    assert get_template("strategy_sample_design_v2") == STRATEGY_SAMPLE_DESIGN_V2
    assert (
        get_template("strategy_sample_design_v2_native")
        == STRATEGY_SAMPLE_DESIGN_V2_NATIVE
    )
    assert get_template("strategy_model_evidence_v2") == STRATEGY_MODEL_EVIDENCE_V2


def test_v2_sample_template_uses_v1_anchor_then_v2_materialization() -> None:
    template = STRATEGY_SAMPLE_DESIGN_V2

    assert len(template.steps) == 2
    anchor, v2 = template.steps
    assert anchor.tool_ref == ToolRef("strategy", "materialize_sample_design")
    assert anchor.depends_on_titles == ()
    assert v2.tool_ref == ToolRef("strategy", "materialize_sample_design_v2")
    assert v2.depends_on_titles == (anchor.title,)
    assert anchor.inputs_template["dataset_id"] == "{slot:dataset_id}"
    assert anchor.inputs_template["target_bad_value"] == "{slot:target_bad_value}"
    assert anchor.inputs_template["split_col"] == "{slot:compatibility_split_col}"
    assert v2.inputs_template["relationship"] == "{slot:relationship}"
    assert v2.inputs_template["scope"] == "{slot:scope}"
    assert v2.inputs_template["policy"] == "{slot:policy}"


def test_v2_native_sample_template_materializes_directly_from_active_dataset() -> None:
    template = STRATEGY_SAMPLE_DESIGN_V2_NATIVE

    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef(
        "strategy",
        "materialize_sample_design_v2_native",
    )
    assert step.depends_on_titles == ()
    assert step.inputs_template["source_mode"] == "native_active_dataset"
    assert step.inputs_template["dataset_id"] == "{slot:dataset_id}"
    assert (
        step.inputs_template["expected_dataset_content_hash"]
        == "{slot:expected_dataset_content_hash}"
    )
    assert step.inputs_template["relationship"] == "{slot:relationship}"
    assert "legacy_sample_design_ref" not in step.inputs_template
    assert PostCheck(
        "nonempty",
        {"field": "source_binding.source_mode"},
    ) in step.post_checks
    assert PostCheck(
        "nonempty",
        {"field": "source_binding.development_partition"},
    ) in step.post_checks


def test_v2_sample_legacy_ref_has_four_anchor_refs_and_fixed_partition() -> None:
    anchor, v2 = STRATEGY_SAMPLE_DESIGN_V2.steps
    legacy = v2.inputs_template["legacy_sample_design_ref"]

    assert set(legacy) == {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }
    assert legacy == {
        "artifact_id": f"$ref:{anchor.title}.output.artifact.artifact_id",
        "artifact_content_hash": f"$ref:{anchor.title}.output.artifact.content_hash",
        "sample_design_id": f"$ref:{anchor.title}.output.sample_design_id",
        "sample_design_content_hash": f"$ref:{anchor.title}.output.content_hash",
        "partition": "development",
    }
    assert all(
        legacy[key].startswith(f"$ref:{anchor.title}.output.")
        for key in (
            "artifact_id",
            "artifact_content_hash",
            "sample_design_id",
            "sample_design_content_hash",
        )
    )


def test_v2_sample_post_checks_cover_identities_without_trusting_v2_artifact_ids() -> None:
    _, v2 = STRATEGY_SAMPLE_DESIGN_V2.steps

    assert PostCheck("nonempty", {"field": "bundle_id"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "sample_design_id"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "membership_id"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "membership_content_hash"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "artifacts.membership.kind"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "artifacts.membership.filename"}) in v2.post_checks
    assert (
        PostCheck("nonempty", {"field": "artifacts.membership.content_hash"})
        not in v2.post_checks
    )
    assert PostCheck("nonempty", {"field": "artifacts.bundle.kind"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "artifacts.bundle.filename"}) in v2.post_checks
    assert PostCheck("nonempty", {"field": "artifacts.bundle.content_hash"}) in v2.post_checks
    assert all("artifact_id" not in str(check.spec.get("field")) for check in v2.post_checks)
    assert all("download_url" not in str(check.spec.get("field")) for check in v2.post_checks)


def test_v2_sample_user_and_platform_slot_ownership_is_explicit() -> None:
    sources = {slot.name: slot.source for slot in STRATEGY_SAMPLE_DESIGN_V2.slots}
    user_fields = {
        "target_bad_value",
        "drop_nan_labels",
        "relationship",
        "approval_population",
        "risk_population",
        "partitioning",
        "maturity",
        "performance_window",
        "observation_window",
        "field_bindings",
        "historical_score",
    }

    assert {name for name, source in sources.items() if source == "user"} == user_fields
    assert sources["relationship"] == "user"
    assert sources["scope"] == "task_context"
    assert sources["policy"] == "task_context"
    assert sources["dataset_id"] == "task_context"
    assert all(
        sources[name] == "task_context"
        for name in sources
        if name.startswith("compatibility_")
    )


def test_model_evidence_template_has_only_task_context_refs_and_one_tool_step() -> None:
    template = STRATEGY_MODEL_EVIDENCE_V2

    assert [(slot.name, slot.source) for slot in template.slots] == [
        ("sample_design_ref", "task_context"),
        ("univariate_sources", "task_context"),
        ("expected_registry_token", "task_context"),
    ]
    assert all(slot.required for slot in template.slots)
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref == ToolRef("strategy", "materialize_model_evidence_v2")
    assert step.inputs_template == {
        "sample_design_ref": "{slot:sample_design_ref}",
        "univariate_sources": "{slot:univariate_sources}",
        "expected_registry_token": "{slot:expected_registry_token}",
    }
    assert step.depends_on_titles == ()
    assert PostCheck("nonempty", {"field": "bundle_id"}) in step.post_checks
    assert PostCheck("nonempty", {"field": "sample_design_id"}) in step.post_checks
    assert PostCheck("nonempty", {"field": "source_artifacts"}) in step.post_checks
    assert PostCheck("nonempty", {"field": "univariate_only"}) in step.post_checks


def test_planner_rewrites_v2_anchor_refs_and_validates_model_template(tmp_path: Path) -> None:
    load_builtin_templates()
    tools = _tool_registry(tmp_path)
    validator = PlanValidator(tools)
    planner = Planner(tools, lambda: None, validator)
    partitioning = {
        "method": "predicate_ast",
        "selectors": {
            "development": _eq("sample_role", "dev"),
            "validation": _eq("sample_role", "valid"),
            "oot": _eq("sample_role", "oot"),
        },
    }
    field_bindings = {
        "entity_field": "customer_id",
        "time_field": "apply_date",
        "group_field": "channel",
        "month_field": "apply_month",
        "weight_field": "weight",
        "loan_amount_field": "loan_amount",
        "overdue_amount_field": "overdue_amount",
    }
    sample_plan = planner.from_template(
        STRATEGY_SAMPLE_DESIGN_V2,
        {
            "dataset_id": "dataset-1",
            "expected_dataset_content_hash": "a" * 64,
            "workspace_revision": 0,
            "workspace_generation": 0,
            "semantic_mapping_hash": "b" * 64,
            "target_col": "bad",
            "relationship": "nested_same_cohort",
            "scope": "strategy_development",
            "policy": _policy(),
            "compatibility_performance_window_status": "provided",
            "compatibility_performance_window_days": 30,
            "compatibility_observation_window_status": "provided",
            "compatibility_observation_start": "2026-01-01",
            "compatibility_observation_end": "2026-04-30",
            "compatibility_maturity_status": "confirmed_matured",
            "compatibility_split_col": "sample_role",
            "compatibility_development_values": ["dev"],
            "compatibility_validation_values": ["valid"],
            "compatibility_oot_values": ["oot"],
            "compatibility_month_col": "apply_month",
            "compatibility_weight_col": "weight",
            "compatibility_loan_amount_col": "loan_amount",
            "compatibility_overdue_amount_col": "overdue_amount",
            "target_bad_value": 1,
            "drop_nan_labels": True,
            "approval_population": {"inclusion": None, "exclusion": None},
            "risk_population": {"inclusion": None, "exclusion": None},
            "partitioning": partitioning,
            "maturity": {
                "status": "confirmed_matured",
                "performance_window_days": 30,
                "cutoff_date": "2026-04-30",
                "reason": None,
            },
            "performance_window": {"status": "provided", "days": 30},
            "observation_window": {
                "status": "provided",
                "start": "2026-01-01",
                "end": "2026-04-30",
            },
            "field_bindings": field_bindings,
            "historical_score": {
                "status": "available",
                "column": "legacy_score",
                "direction": "higher_is_riskier",
                "reason": None,
            },
        },
        task_id="task-1",
    )

    anchor, v2 = sample_plan.steps
    assert v2.depends_on == [anchor.id]
    assert v2.inputs["legacy_sample_design_ref"] == {
        "artifact_id": f"$ref:{anchor.id}.output.artifact.artifact_id",
        "artifact_content_hash": f"$ref:{anchor.id}.output.artifact.content_hash",
        "sample_design_id": f"$ref:{anchor.id}.output.sample_design_id",
        "sample_design_content_hash": f"$ref:{anchor.id}.output.content_hash",
        "partition": "development",
    }
    assert validator.validate(sample_plan) == []

    model_plan = planner.from_template(
        STRATEGY_MODEL_EVIDENCE_V2,
        {
            "sample_design_ref": {
                "membership_artifact_id": "c" * 64,
                "expected_membership_artifact_content_hash": "d" * 64,
                "bundle_artifact_id": "e" * 64,
                "expected_bundle_artifact_content_hash": "f" * 64,
                "expected_bundle_id": "strategy-sample-design-bundle-" + "1" * 24,
                "expected_sample_design_id": "strategy-sample-design-" + "2" * 24,
                "expected_sample_design_content_hash": "3" * 64,
            },
            "univariate_sources": [
                {
                    "artifact_id": "4" * 64,
                    "expected_artifact_content_hash": "5" * 64,
                    "expected_candidate_id": "candidate-" + "6" * 32,
                    "expected_evidence_hash": "7" * 64,
                }
            ],
            "expected_registry_token": "8" * 64,
        },
        task_id="task-1",
    )

    assert validator.validate(model_plan) == []

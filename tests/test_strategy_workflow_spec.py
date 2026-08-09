from __future__ import annotations

import ast
import inspect
from types import MappingProxyType

import pytest

from marvis.agent import strategy_request_compiler, turn_handlers
from marvis.agent.strategy_workflows import (
    FRESH_STANDARD_STRATEGY_WORKFLOWS,
    LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS,
    MANUAL_STANDARD_STRATEGY_WORKFLOWS,
    REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowResolutionMode,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    prepare_strategy_plan,
    resolve_strategy_request,
)
from marvis.agent.strategy_workflows._catalog import build_catalog
from marvis.orchestrator.templates import get_template, load_builtin_templates


def _profit_inputs() -> dict[str, object]:
    return {
        "segment_col": "segment",
        "ead_col": "ead",
        "pd_col": "pd_12m",
        "profit_params": {
            "annual_rate": 0.18,
            "funding_rate": 0.04,
            "lgd": 0.5,
            "operating_cost_per_loan": 12,
            "term_months": 12,
        },
    }


def _roll_rate_inputs() -> dict[str, object]:
    return {
        "id_col": "customer_id",
        "time_col": "month",
        "status_col": "status",
        "states": ["M0", "M1", "M2+"],
        "balance_col": "balance",
        "observation_semantics": "adjacent_observation",
    }


def _pricing_inputs() -> dict[str, object]:
    return {
        "score_col": "score",
        "target_col": "bad",
        "n_bands": 5,
        "limit_grid": [1000, 2000],
        "rate_grid": [0.12, 0.18],
        "lgd": 0.5,
        "funding_rate": 0.04,
        "term_months": 12,
        "cost_per_loan": 10,
        "el_ead_max": 0.2,
        "drop_nan_labels": True,
    }


def _sample_v2_inputs() -> dict[str, object]:
    def eq(value: str) -> dict[str, object]:
        return {
            "op": "eq",
            "left": {"column": "sample_role"},
            "right": {"literal": value},
        }

    return {
        "target_bad_value": 1,
        "drop_nan_labels": False,
        "relationship": "nested_same_cohort",
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": eq("dev"),
                "validation": eq("valid"),
                "oot": eq("oot"),
            },
        },
        "maturity": {
            "status": "unknown",
            "performance_window_days": None,
            "cutoff_date": None,
            "reason": "尚未获得成熟度口径",
        },
        "performance_window": {"status": "unavailable", "days": None},
        "observation_window": {
            "status": "unavailable",
            "start": None,
            "end": None,
        },
        "field_bindings": {
            "entity_field": None,
            "time_field": None,
            "group_field": None,
            "month_field": None,
            "weight_field": None,
            "loan_amount_field": None,
            "overdue_amount_field": None,
        },
        "historical_score": {
            "status": "not_applicable",
            "column": None,
            "direction": None,
            "reason": "当前没有历史评分",
        },
    }


def test_catalog_is_the_unique_source_for_fresh_replay_and_manual_surfaces() -> None:
    assert len(FRESH_STANDARD_STRATEGY_WORKFLOWS) == len(
        set(FRESH_STANDARD_STRATEGY_WORKFLOWS)
    )
    assert LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS == (
        "strategy_sample_design",
    )
    assert REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS == (
        *FRESH_STANDARD_STRATEGY_WORKFLOWS,
        *LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS,
    )
    assert set(MANUAL_STANDARD_STRATEGY_WORKFLOWS) < set(
        FRESH_STANDARD_STRATEGY_WORKFLOWS
    )
    assert {
        "cross_rule_search",
        "cross_rule_candidate_build_from_search",
        "interactive_tree_split_search",
        "interactive_tree_auto_continuation",
    } <= set(FRESH_STANDARD_STRATEGY_WORKFLOWS)


@pytest.mark.parametrize(
    ("workflow", "inputs", "columns", "confirmation_fragment"),
    [
        (
            "profit_calc",
            _profit_inputs(),
            ("segment", "ead", "pd_12m"),
            "标准利润分析 Workflow",
        ),
        (
            "roll_rate_matrix",
            _roll_rate_inputs(),
            ("customer_id", "month", "status", "balance"),
            "标准滚动率矩阵 Workflow",
        ),
        (
            "limit_pricing_matrix",
            _pricing_inputs(),
            ("score",),
            "标准额度定价矩阵 Workflow",
        ),
    ],
)
def test_first_family_resolves_to_frozen_canonical_inputs_and_confirmation(
    workflow: str,
    inputs: dict[str, object],
    columns: tuple[str, ...],
    confirmation_fragment: str,
) -> None:
    resolved = resolve_strategy_request(
        workflow,
        inputs,
        context=StrategyWorkflowResolutionContext(
            allowed_columns=columns,
            target_col="bad",
        ),
    )

    assert resolved.workflow_id == workflow
    assert isinstance(resolved.workflow_inputs, MappingProxyType)
    assert confirmation_fragment in resolved.confirmation
    assert "所有数字由平台确定性计算" in resolved.confirmation

    inputs.clear()
    assert resolved.workflow_inputs


def test_first_family_validation_fails_closed_with_typed_error() -> None:
    invalid = _profit_inputs()
    invalid["ead_col"] = "ghost"

    with pytest.raises(StrategyWorkflowValidationError) as exc_info:
        resolve_strategy_request(
            "profit_calc",
            invalid,
            context=StrategyWorkflowResolutionContext(
                allowed_columns=("segment", "ead", "pd_12m"),
                target_col="bad",
            ),
        )

    assert exc_info.value.code == "invalid_strategy_request"
    assert "ghost" in str(exc_info.value)


@pytest.mark.parametrize(
    ("workflow", "inputs", "template_id", "needs_sample"),
    [
        ("profit_calc", _profit_inputs(), "strategy_profit_analysis", False),
        (
            "roll_rate_matrix",
            _roll_rate_inputs(),
            "strategy_roll_rate_analysis",
            False,
        ),
        (
            "limit_pricing_matrix",
            _pricing_inputs(),
            "strategy_limit_pricing_analysis",
            True,
        ),
    ],
)
def test_first_family_prepares_plan_without_executing(
    workflow: str,
    inputs: dict[str, object],
    template_id: str,
    needs_sample: bool,
) -> None:
    sample_calls: list[bool] = []

    def bind_sample_design(allow_native_risk_development: bool):
        sample_calls.append(allow_native_risk_development)
        return MappingProxyType(
            {
                "artifact_id": "artifact-sample",
                "content_hash": "a" * 64,
            }
        )

    prepared = prepare_strategy_plan(
        workflow,
        inputs,
        context=StrategyWorkflowPreparationContext(
            dataset_id="dataset-1",
            drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            bind_sample_design=bind_sample_design,
        ),
    )

    assert prepared.workflow_id == workflow
    assert prepared.template_id == template_id
    assert prepared.slots["dataset_id"] == "dataset-1"
    assert isinstance(prepared.slots, MappingProxyType)
    runtime_slots = prepared.to_runtime_slots()
    assert isinstance(runtime_slots, dict)
    assert sample_calls == ([True] if needs_sample else [])
    if needs_sample:
        assert prepared.slots["sample_design_ref"]["artifact_id"] == "artifact-sample"
        assert isinstance(runtime_slots["sample_design_ref"], dict)
    if workflow == "roll_rate_matrix":
        assert isinstance(runtime_slots["states"], list)


def test_limit_pricing_preparation_requires_authenticated_sample_binding() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as exc_info:
        prepare_strategy_plan(
            "limit_pricing_matrix",
            _pricing_inputs(),
            context=StrategyWorkflowPreparationContext(
                dataset_id="dataset-1",
                drop_nan_labels=True,
            ),
        )

    assert exc_info.value.code == "strategy_sample_design_required"


def test_all_strategy_families_are_canonical_catalog_specs() -> None:
    expected = {
        "strategy_project_context",
        "strategy_sample_design_v2",
        "strategy_model_evidence_v2",
        "strategy_dsl_delivery",
        "strategy_report_bundle_v2",
        "strategy_sample_design",
        "automatic_tree_candidate_build",
        "automatic_tree_apply",
        "automatic_tree_leaf_materialization",
        "interactive_tree_split_search",
        "interactive_tree_auto_continuation",
        "interactive_tree_revision",
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "voting_candidate_build",
        "cross_matrix_candidate_search",
        "cross_matrix_candidate_build_from_search",
        "cross_rule_search",
        "cross_rule_candidate_build_from_search",
        "cross_matrix_analysis",
        "cross_matrix_cell_selection",
        "strategy_pool_add_candidate",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_compile",
        "strategy_pool_materialize",
        "strategy_pool_apply",
        "strategy_pool_validation",
        "strategy_pool_impact",
        "strategy_impact_cube",
        "strategy_pool_stability",
    }
    specs = {spec.workflow_id: spec for spec in build_catalog()}

    assert all(specs[workflow_id].migrated for workflow_id in expected)
    assert sum(spec.migrated for spec in specs.values()) == 44


def test_resolution_mode_preserves_fresh_v2_safety_and_normalized_replay() -> None:
    inputs = _sample_v2_inputs()
    raw_population_ast = {
        "op": "eq",
        "left": {"column": "channel"},
        "right": {"literal": "app"},
    }
    inputs["approval_population"]["inclusion"] = raw_population_ast
    context = StrategyWorkflowResolutionContext(
        allowed_columns=("sample_role", "channel"),
        target_col="bad",
    )

    with pytest.raises(StrategyWorkflowValidationError) as fresh_error:
        resolve_strategy_request(
            "strategy_sample_design_v2",
            inputs,
            context=context,
            mode=StrategyWorkflowResolutionMode.FRESH,
        )
    replayed = resolve_strategy_request(
        "strategy_sample_design_v2",
        inputs,
        context=context,
        mode=StrategyWorkflowResolutionMode.REPLAY,
    )

    assert fresh_error.value.code == (
        "strategy_sample_design_v2_population_dto_required"
    )
    assert replayed.workflow_inputs["approval_population"]["inclusion"] == (
        raw_population_ast
    )


def test_migrated_specs_declare_exact_template_presenter_refs() -> None:
    load_builtin_templates()
    specs = [spec for spec in build_catalog() if spec.migrated]

    assert specs
    for spec in specs:
        actual_refs = tuple(
            dict.fromkeys(
                step.tool_ref.tool
                for template_id in spec.template_ids
                for step in get_template(template_id).steps
            )
        )
        assert spec.presenter_refs == actual_refs, spec.workflow_id


def test_compiler_has_no_family_validator_or_confirmation_shadows() -> None:
    compiler_tree = ast.parse(inspect.getsource(strategy_request_compiler))
    family_validators = sorted(
        node.name
        for node in compiler_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_validate_")
        and node.name.endswith("_inputs")
    )
    confirmation = next(
        node
        for node in compiler_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_standard_workflow_confirmation_text"
    )

    assert family_validators == []
    assert not any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "workflow"
        for node in ast.walk(confirmation)
    )


def test_turn_router_has_one_canonical_standard_workflow_entry() -> None:
    router = ast.parse(
        inspect.getsource(turn_handlers._run_validated_strategy_request)
    )
    standard_type_checks = [
        node
        for node in ast.walk(router)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "StandardWorkflowRequestDraft"
    ]

    assert len(standard_type_checks) == 1

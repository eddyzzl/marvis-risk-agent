"""Characterize canonical automatic/interactive tree workflow boundaries."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from marvis.agent.strategy_workflows._tree_workflows import TREE_WORKFLOW_SPECS
from marvis.agent.strategy_workflows.contracts import (
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_thaw,
)


ASSET_ID = "candidate-asset-" + "a" * 32
LEAF_ID = "leaf-" + "b" * 20
SOURCE_ID = "interactive-tree-revision-" + "c" * 32
NODE_ID = "node-" + "d" * 20
SEARCH_ID = "interactive-tree-split-search-" + "e" * 32
CANDIDATE_ID = "interactive-tree-split-candidate-" + "f" * 32

RESOLUTION = StrategyWorkflowResolutionContext(
    allowed_columns=("score", "age", "weight", "loan", "overdue", "bad"),
    target_col="bad",
)

BUILD_INPUTS = {
    "features": ["score", "age"],
    "sample_weight_col": "weight",
    "directions": {"score": "increasing", "age": "unordered"},
    "max_depth": 4,
    "min_leaf_count": 20,
    "min_weight_fraction_leaf": 0.025,
    "seed": 42,
    "loan_amount_col": "loan",
    "overdue_amount_col": "overdue",
}
APPLY_INPUTS = {
    "tree_asset_id": ASSET_ID,
    "leaf_id_column": "tree_leaf",
    "rule_id_column": "tree_rule",
}
LEAF_INPUTS = {
    "tree_asset_id": ASSET_ID,
    "leaf_id": LEAF_ID,
    "selection_reason": "  人工\n确认  ",
}
SPLIT_INPUTS = {
    "source_tree_id": SOURCE_ID,
    "node_id": NODE_ID,
    "mode": "selected_features",
    "features": ["score", "age"],
    "max_thresholds_per_feature": 10,
    "max_row_evaluations": 100_000,
}
CONTINUATION_INPUTS = {
    "search_id": SEARCH_ID,
    "candidate_id": CANDIDATE_ID,
    "max_additional_depth": 2,
    "min_gini_gain": 0.01,
    "max_generated_nodes": 31,
    "max_thresholds_per_feature": 10,
    "max_row_evaluations": 100_000,
    "objective": "max_gini_gain",
    "tie_break": "eligible_gain_feature_threshold_candidate_id",
    "reason": " 继续\n探索 ",
}
REVISION_INPUTS = {
    "source_tree_id": SOURCE_ID,
    "node_id": NODE_ID,
    "operation": "replace_split_feature",
    "feature": "age",
    "threshold": 12,
    "reason": "  专家\n调整  ",
}

BUILD_EVIDENCE = {
    "dataset_id": "dataset-1",
    "expected_content_hash": "1" * 64,
    "workspace_revision": 7,
    "analysis_generation": 3,
    "semantic_mapping_hash": "2" * 64,
    "target_col": "bad",
    "sample_design_ref": {
        "artifact_id": "sample-artifact-1",
        "content_hash": "3" * 64,
    },
}
APPLY_EVIDENCE = {
    "source_artifact_id": "artifact-1",
    "expected_artifact_content_hash": "4" * 64,
    "expected_asset_id": ASSET_ID,
    "expected_asset_hash": "5" * 64,
    "expected_tree_result_hash": "6" * 64,
    "dataset_id": "dataset-1",
    "expected_content_hash": "7" * 64,
    "workspace_revision": 7,
    "analysis_generation": 3,
    "semantic_mapping_hash": "8" * 64,
}
LEAF_EVIDENCE = {
    key: value
    for key, value in APPLY_EVIDENCE.items()
    if key
    in {
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
    }
}


def _spec(workflow_id: str):
    return next(spec for spec in TREE_WORKFLOW_SPECS if spec.workflow_id == workflow_id)


def _validate(workflow_id: str, inputs: Mapping[str, object]) -> dict[str, object]:
    validator = _spec(workflow_id).validator
    assert validator is not None
    return validator(inputs, RESOLUTION)


def test_tree_workflow_specs_expose_exact_migrated_metadata() -> None:
    expected = {
        "automatic_tree_candidate_build": (
            True,
            StrategyWorkflowRequirements(True, True, True),
            "strategy_automatic_tree_candidate_build",
        ),
        "automatic_tree_apply": (
            False,
            StrategyWorkflowRequirements(True, False, False),
            "strategy_automatic_tree_apply",
        ),
        "automatic_tree_leaf_materialization": (
            False,
            StrategyWorkflowRequirements(False, False, False),
            "strategy_automatic_tree_leaf_materialization",
        ),
        "interactive_tree_split_search": (
            True,
            StrategyWorkflowRequirements(False, False, False),
            "strategy_interactive_tree_split_search",
        ),
        "interactive_tree_auto_continuation": (
            True,
            StrategyWorkflowRequirements(False, False, False),
            "strategy_interactive_tree_auto_continuation",
        ),
        "interactive_tree_revision": (
            True,
            StrategyWorkflowRequirements(False, False, False),
            "strategy_interactive_tree_revision",
        ),
    }

    assert {
        spec.workflow_id: (
            spec.manual,
            spec.requirements,
            spec.template_ids[0],
        )
        for spec in TREE_WORKFLOW_SPECS
    } == expected
    assert all(
        spec.fresh and spec.replayable and spec.migrated for spec in TREE_WORKFLOW_SPECS
    )
    assert all(len(spec.template_ids) == 1 for spec in TREE_WORKFLOW_SPECS)


@pytest.mark.parametrize("spec", TREE_WORKFLOW_SPECS)
def test_tree_workflow_specs_define_complete_adapter_triples(spec) -> None:
    assert callable(spec.validator)
    assert callable(spec.confirmation)
    assert callable(spec.preparer)


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "expected"),
    [
        (
            "automatic_tree_candidate_build",
            BUILD_INPUTS,
            BUILD_INPUTS,
        ),
        (
            "automatic_tree_apply",
            {
                "tree_asset_id": f" {ASSET_ID} ",
                "leaf_id_column": " leaf_col ",
                "rule_id_column": " rule_col ",
            },
            {
                "tree_asset_id": ASSET_ID,
                "leaf_id_column": "leaf_col",
                "rule_id_column": "rule_col",
            },
        ),
        (
            "automatic_tree_leaf_materialization",
            LEAF_INPUTS,
            {
                "tree_asset_id": ASSET_ID,
                "leaf_id": LEAF_ID,
                "selection_reason": "人工 确认",
            },
        ),
        (
            "interactive_tree_split_search",
            SPLIT_INPUTS,
            {
                **SPLIT_INPUTS,
                "features": ["age", "score"],
            },
        ),
        (
            "interactive_tree_auto_continuation",
            CONTINUATION_INPUTS,
            {
                **CONTINUATION_INPUTS,
                "min_gini_gain": 0.01,
                "reason": "继续 探索",
            },
        ),
        (
            "interactive_tree_revision",
            REVISION_INPUTS,
            {
                **REVISION_INPUTS,
                "threshold": 12.0,
                "reason": "专家 调整",
            },
        ),
    ],
)
def test_tree_validators_preserve_legacy_normalization(
    workflow_id: str,
    inputs: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert _validate(workflow_id, inputs) == expected


@pytest.mark.parametrize(
    ("workflow_id", "inputs"),
    [
        (
            "automatic_tree_candidate_build",
            {**BUILD_INPUTS, "dataset_id": "user-injected"},
        ),
        (
            "automatic_tree_apply",
            {**APPLY_INPUTS, "source_artifact_id": "user-injected"},
        ),
        (
            "automatic_tree_leaf_materialization",
            {**LEAF_INPUTS, "expected_asset_hash": "user-injected"},
        ),
        (
            "interactive_tree_split_search",
            {**SPLIT_INPUTS, "sample_design_ref": {}},
        ),
        (
            "interactive_tree_auto_continuation",
            {**CONTINUATION_INPUTS, "expected_content_hash": "user-injected"},
        ),
        (
            "interactive_tree_revision",
            {**REVISION_INPUTS, "parent_revision": {}},
        ),
    ],
)
def test_tree_validators_reject_every_platform_owned_slot(
    workflow_id: str,
    inputs: dict[str, object],
) -> None:
    with pytest.raises(StrategyWorkflowValidationError):
        _validate(workflow_id, inputs)


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "field"),
    [
        (
            "automatic_tree_candidate_build",
            {**BUILD_INPUTS, "features": ["score", "score"]},
            "features",
        ),
        (
            "automatic_tree_candidate_build",
            {**BUILD_INPUTS, "features": ["bad"]},
            "features",
        ),
        (
            "automatic_tree_candidate_build",
            {**BUILD_INPUTS, "sample_weight_col": "score"},
            "sample_weight_col",
        ),
        (
            "automatic_tree_candidate_build",
            {**BUILD_INPUTS, "directions": {"loan": "increasing"}},
            "directions",
        ),
        (
            "automatic_tree_candidate_build",
            {**BUILD_INPUTS, "max_depth": True},
            "max_depth",
        ),
        (
            "automatic_tree_apply",
            {**APPLY_INPUTS, "tree_asset_id": "candidate-asset-short"},
            "tree_asset_id",
        ),
        (
            "automatic_tree_apply",
            {**APPLY_INPUTS, "rule_id_column": "TREE_LEAF"},
            "rule_id_column",
        ),
        (
            "automatic_tree_leaf_materialization",
            {**LEAF_INPUTS, "leaf_id": "leaf-short"},
            "leaf_id",
        ),
        (
            "automatic_tree_leaf_materialization",
            {**LEAF_INPUTS, "selection_reason": "bad\x00reason"},
            "selection_reason",
        ),
        (
            "interactive_tree_split_search",
            {**SPLIT_INPUTS, "mode": "all_features"},
            "features",
        ),
        (
            "interactive_tree_split_search",
            {
                **SPLIT_INPUTS,
                "mode": "selected_features",
                "features": [],
            },
            "features",
        ),
        (
            "interactive_tree_split_search",
            {**SPLIT_INPUTS, "max_row_evaluations": 20_000_001},
            "max_row_evaluations",
        ),
        (
            "interactive_tree_auto_continuation",
            {**CONTINUATION_INPUTS, "candidate_id": "candidate-short"},
            "candidate_id",
        ),
        (
            "interactive_tree_auto_continuation",
            {**CONTINUATION_INPUTS, "min_gini_gain": float("nan")},
            "min_gini_gain",
        ),
        (
            "interactive_tree_auto_continuation",
            {**CONTINUATION_INPUTS, "objective": "best_available"},
            "objective",
        ),
        (
            "interactive_tree_revision",
            {
                "source_tree_id": SOURCE_ID,
                "node_id": NODE_ID,
                "operation": "adjust_split_threshold",
            },
            "threshold",
        ),
        (
            "interactive_tree_revision",
            {
                "source_tree_id": SOURCE_ID,
                "node_id": NODE_ID,
                "operation": "prune_subtree",
                "threshold": 0.5,
            },
            "threshold",
        ),
        (
            "interactive_tree_revision",
            {**REVISION_INPUTS, "threshold": 2**53},
            "threshold",
        ),
        (
            "interactive_tree_revision",
            {**REVISION_INPUTS, "reason": "x" * 501},
            "reason",
        ),
    ],
)
def test_tree_validators_fail_closed_on_invalid_or_ambiguous_controls(
    workflow_id: str,
    inputs: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _validate(workflow_id, inputs)
    assert field in captured.value.fields


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "phrases"),
    [
        (
            "automatic_tree_candidate_build",
            BUILD_INPUTS,
            ("自动决策树候选构建", "LLM 不得填写", "明确 leaf"),
        ),
        (
            "automatic_tree_apply",
            APPLY_INPUTS,
            (ASSET_ID, "不会激活或替换", "development / unvalidated"),
        ),
        (
            "automatic_tree_leaf_materialization",
            {**LEAF_INPUTS, "selection_reason": "人工 确认"},
            (LEAF_ID, "不可变 pointer", "不会加入 Strategy Pool"),
        ),
        (
            "interactive_tree_split_search",
            {**SPLIT_INPUTS, "features": ["age", "score"]},
            (NODE_ID, "SampleDesign", "不会选择胜者"),
        ),
        (
            "interactive_tree_auto_continuation",
            {**CONTINUATION_INPUTS, "reason": "继续 探索"},
            (CANDIDATE_ID, "不会自动挑选种子候选", "不可变修订"),
        ),
        (
            "interactive_tree_revision",
            {**REVISION_INPUTS, "threshold": 12.0, "reason": "专家 调整"},
            (SOURCE_ID, "完整自动树、父 revision", "不会修改原树"),
        ),
    ],
)
def test_tree_confirmations_keep_governance_and_non_action_boundaries(
    workflow_id: str,
    inputs: dict[str, object],
    phrases: tuple[str, ...],
) -> None:
    confirmation = _spec(workflow_id).confirmation
    assert confirmation is not None
    rendered = confirmation(inputs)
    assert all(phrase in rendered for phrase in phrases)


def test_automatic_tree_build_preparer_binds_sample_and_drop_nan_once() -> None:
    calls: list[tuple[str, Mapping[str, object]]] = []

    def bind(workflow_id: str, inputs: Mapping[str, object]):
        calls.append((workflow_id, inputs))
        assert not isinstance(inputs, dict)
        assert inputs["features"] == ("score", "age")
        return dict(BUILD_EVIDENCE)

    normalized = _validate("automatic_tree_candidate_build", BUILD_INPUTS)
    preparer = _spec("automatic_tree_candidate_build").preparer
    assert preparer is not None
    plan = preparer(
        normalized,
        StrategyWorkflowPreparationContext(
            dataset_id="dataset-1",
            drop_nan_labels=True,
            bind_workflow_evidence=bind,
        ),
    )

    assert len(calls) == 1
    assert plan.template_id == "strategy_automatic_tree_candidate_build"
    assert plan.success_criteria == ()
    assert plan.to_runtime_slots() == {
        **BUILD_INPUTS,
        **BUILD_EVIDENCE,
        "drop_nan_labels": True,
    }


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "evidence", "absent_pointer", "expected_slots"),
    [
        (
            "automatic_tree_apply",
            APPLY_INPUTS,
            APPLY_EVIDENCE,
            "tree_asset_id",
            {
                "leaf_id_column": "tree_leaf",
                "rule_id_column": "tree_rule",
                **APPLY_EVIDENCE,
            },
        ),
        (
            "automatic_tree_leaf_materialization",
            {**LEAF_INPUTS, "selection_reason": "人工 确认"},
            LEAF_EVIDENCE,
            "tree_asset_id",
            {
                "leaf_id": LEAF_ID,
                "selection_reason": "人工 确认",
                **LEAF_EVIDENCE,
            },
        ),
    ],
)
def test_tree_pointer_preparers_replace_user_lookup_with_verified_evidence(
    workflow_id: str,
    inputs: dict[str, object],
    evidence: dict[str, object],
    absent_pointer: str,
    expected_slots: dict[str, object],
) -> None:
    calls: list[tuple[str, Mapping[str, object]]] = []

    def bind(called_workflow: str, called_inputs: Mapping[str, object]):
        calls.append((called_workflow, called_inputs))
        return dict(evidence)

    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    plan = preparer(
        inputs,
        StrategyWorkflowPreparationContext(
            dataset_id=("dataset-1" if workflow_id == "automatic_tree_apply" else None),
            bind_workflow_evidence=bind,
        ),
    )

    assert len(calls) == 1
    assert calls[0][0] == workflow_id
    assert calls[0][1][absent_pointer] == ASSET_ID
    assert absent_pointer not in plan.slots
    assert plan.to_runtime_slots() == expected_slots
    assert plan.success_criteria == ()


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "template_id"),
    [
        (
            "interactive_tree_split_search",
            {**SPLIT_INPUTS, "features": ["age", "score"]},
            "strategy_interactive_tree_split_search",
        ),
        (
            "interactive_tree_auto_continuation",
            {**CONTINUATION_INPUTS, "reason": "继续 探索"},
            "strategy_interactive_tree_auto_continuation",
        ),
        (
            "interactive_tree_revision",
            {**REVISION_INPUTS, "threshold": 12.0, "reason": "专家 调整"},
            "strategy_interactive_tree_revision",
        ),
        (
            "interactive_tree_revision",
            {
                "source_tree_id": SOURCE_ID,
                "node_id": NODE_ID,
                "operation": "prune_subtree",
            },
            "strategy_interactive_tree_revision",
        ),
    ],
)
def test_interactive_tree_preparers_require_currentness_callback_without_slots(
    workflow_id: str,
    inputs: dict[str, object],
    template_id: str,
) -> None:
    calls: list[tuple[str, Mapping[str, object]]] = []

    def bind(called_workflow: str, called_inputs: Mapping[str, object]):
        calls.append((called_workflow, called_inputs))
        return {}

    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    plan = preparer(
        inputs,
        StrategyWorkflowPreparationContext(
            dataset_id=None,
            bind_workflow_evidence=bind,
        ),
    )

    assert len(calls) == 1
    assert calls[0][0] == workflow_id
    assert deep_thaw(calls[0][1]) == inputs
    assert plan.template_id == template_id
    assert plan.to_runtime_slots() == inputs
    assert plan.success_criteria == ()


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "dataset_id"),
    [
        ("automatic_tree_candidate_build", BUILD_INPUTS, "dataset-1"),
        ("automatic_tree_apply", APPLY_INPUTS, "dataset-1"),
        ("automatic_tree_leaf_materialization", LEAF_INPUTS, None),
        ("interactive_tree_split_search", SPLIT_INPUTS, None),
        ("interactive_tree_auto_continuation", CONTINUATION_INPUTS, None),
        ("interactive_tree_revision", REVISION_INPUTS, None),
    ],
)
def test_every_tree_preparer_fails_closed_without_evidence_callback(
    workflow_id: str,
    inputs: dict[str, object],
    dataset_id: str | None,
) -> None:
    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            inputs,
            StrategyWorkflowPreparationContext(dataset_id=dataset_id),
        )
    assert captured.value.code == "strategy_workflow_evidence_binding_required"


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "dataset_id", "evidence"),
    [
        (
            "automatic_tree_candidate_build",
            BUILD_INPUTS,
            "dataset-1",
            {
                key: value
                for key, value in BUILD_EVIDENCE.items()
                if key != "sample_design_ref"
            },
        ),
        (
            "automatic_tree_candidate_build",
            BUILD_INPUTS,
            "dataset-1",
            {**BUILD_EVIDENCE, "platform_budget": 10},
        ),
        (
            "automatic_tree_apply",
            APPLY_INPUTS,
            "dataset-1",
            {**APPLY_EVIDENCE, "leaf_id_column": "platform_override"},
        ),
        (
            "automatic_tree_leaf_materialization",
            LEAF_INPUTS,
            None,
            {**LEAF_EVIDENCE, "leaf_id": "platform_override"},
        ),
        (
            "interactive_tree_split_search",
            SPLIT_INPUTS,
            None,
            {"dataset_id": "platform-injected"},
        ),
        (
            "interactive_tree_auto_continuation",
            CONTINUATION_INPUTS,
            None,
            {"candidate_id": CANDIDATE_ID},
        ),
    ],
)
def test_tree_preparers_reject_missing_extra_or_conflicting_evidence_slots(
    workflow_id: str,
    inputs: dict[str, object],
    dataset_id: str | None,
    evidence: dict[str, object],
) -> None:
    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id=dataset_id,
                bind_workflow_evidence=lambda *_args: evidence,
            ),
        )
    assert captured.value.code in {
        "strategy_workflow_evidence_invalid",
        "strategy_workflow_evidence_conflict",
    }


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "dataset_id", "evidence"),
    [
        (
            "automatic_tree_apply",
            APPLY_INPUTS,
            "dataset-1",
            {**APPLY_EVIDENCE, "expected_asset_id": "candidate-asset-" + "9" * 32},
        ),
        (
            "automatic_tree_leaf_materialization",
            LEAF_INPUTS,
            None,
            {**LEAF_EVIDENCE, "expected_asset_id": "candidate-asset-" + "9" * 32},
        ),
        (
            "automatic_tree_candidate_build",
            BUILD_INPUTS,
            "dataset-other",
            BUILD_EVIDENCE,
        ),
        (
            "automatic_tree_apply",
            APPLY_INPUTS,
            "dataset-other",
            APPLY_EVIDENCE,
        ),
    ],
)
def test_tree_evidence_must_match_user_pointer_and_current_dataset(
    workflow_id: str,
    inputs: dict[str, object],
    dataset_id: str | None,
    evidence: dict[str, object],
) -> None:
    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id=dataset_id,
                bind_workflow_evidence=lambda *_args: evidence,
            ),
        )
    assert captured.value.code == "strategy_workflow_evidence_conflict"


@pytest.mark.parametrize(
    ("workflow_id", "inputs"),
    [
        ("automatic_tree_candidate_build", BUILD_INPUTS),
        ("automatic_tree_apply", APPLY_INPUTS),
    ],
)
def test_dataset_workflows_require_live_preparation_dataset(
    workflow_id: str,
    inputs: dict[str, object],
) -> None:
    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=lambda *_args: {},
            ),
        )
    assert captured.value.code == "strategy_workflow_dataset_binding_required"


def test_drop_nan_slot_is_absent_without_explicit_confirmation() -> None:
    preparer = _spec("automatic_tree_candidate_build").preparer
    assert preparer is not None
    plan = preparer(
        BUILD_INPUTS,
        StrategyWorkflowPreparationContext(
            dataset_id="dataset-1",
            drop_nan_labels=False,
            bind_workflow_evidence=lambda *_args: dict(BUILD_EVIDENCE),
        ),
    )
    assert "drop_nan_labels" not in plan.slots

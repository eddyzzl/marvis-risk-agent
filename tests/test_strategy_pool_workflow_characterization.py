"""Characterize canonical Strategy Pool workflow boundaries."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from marvis.agent.strategy_workflows._pool_workflows import POOL_WORKFLOW_SPECS
from marvis.agent.strategy_workflows.contracts import (
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_thaw,
)


ASSET_ID = "candidate-asset-" + "a" * 32
SELECTION_ID = "automatic-tree-leaf-selection-" + "b" * 32
SCORECARD_SELECTION_ID = "scorecard-cutoff-selection-" + "b" * 32
SCORECARD_BAND_ASSET_ID = "scorecard-band-asset-" + "a" * 32
RULE_ID = "candidate-rule-" + "c" * 32
RULE_ID_2 = "candidate-rule-" + "d" * 32
ENTRY_ID = "pool-entry-" + "e" * 32
BASELINE_ID = "strategy-baseline-1"
CURRENT_STRATEGY_ID = "strategy-current-1"

APPROVAL_ACTION = {
    "type": "approval",
    "value": "approve",
    "reason_code": None,
    "stop": True,
}
REJECT_ACTION = {
    "type": "reject",
    "value": "reject",
    "reason_code": "RISK",
    "stop": True,
}

RESOLUTION = StrategyWorkflowResolutionContext(
    allowed_columns=("month", "group", "segment", "loan", "overdue", "ead", "bad"),
    target_col="bad",
)

POOL_IDENTITY = {
    "expected_pool_revision": 7,
    "expected_pool_snapshot_hash": "1" * 64,
}
ADD_EVIDENCE = {
    **POOL_IDENTITY,
    "source_artifact_id": "artifact-1",
    "expected_artifact_content_hash": "2" * 64,
    "expected_asset_id": ASSET_ID,
    "expected_asset_hash": "3" * 64,
    "placement_mode": "append",
}
POOL_REF = {
    "artifact_id": "4" * 64,
    "expected_artifact_content_hash": "5" * 64,
    "expected_pool_id": "strategy-pool-1",
    "expected_revision": 7,
    "expected_revision_id": "strategy-pool-revision-1",
    "expected_snapshot_hash": "6" * 64,
}
V2_SAMPLE_REF = {
    "membership_artifact_id": "7" * 64,
    "expected_membership_artifact_content_hash": "8" * 64,
    "bundle_artifact_id": "9" * 64,
    "expected_bundle_artifact_content_hash": "a" * 64,
    "expected_bundle_id": "sample-bundle-1",
    "expected_sample_design_id": "sample-design-1",
    "expected_sample_design_content_hash": "b" * 64,
}
LEGACY_SAMPLE_REF = {
    "artifact_id": "c" * 64,
    "artifact_content_hash": "d" * 64,
    "sample_design_id": "strategy-sample-design-1",
    "sample_design_content_hash": "e" * 64,
    "partition": "development",
}
MATERIALIZE_EVIDENCE = {
    **POOL_IDENTITY,
    "expected_pool_artifact_id": "f" * 64,
    "expected_pool_artifact_content_hash": "0" * 64,
    "expected_design_hash": "1" * 64,
}
VALIDATION_EVIDENCE = {
    "pool_ref": POOL_REF,
    "sample_design_ref": V2_SAMPLE_REF,
    "population": "risk",
    "comparison_mode": "absolute",
}
IMPACT_CUBE_EVIDENCE = {
    "pool_ref": POOL_REF,
    "sample_design_ref": V2_SAMPLE_REF,
    "partitions": ["development", "oot"],
    "population": "risk",
    "dimension_bindings": {
        "month_col": "month",
        "group_col": "group",
        "segment_col": "segment",
    },
    "current_strategy_ref": {
        "strategy_id": CURRENT_STRATEGY_ID,
        "expected_strategy_spec_hash": "2" * 64,
    },
    "economics_inputs": {
        "ead": {"kind": "column", "column": "ead"},
        "term_months": {"kind": "scalar", "value": 12},
    },
}
STABILITY_EVIDENCE = {
    "pool_ref": POOL_REF,
    "sample_design_ref": V2_SAMPLE_REF,
    "partitions": ["development", "validation"],
}
POOL_IMPACT_EVIDENCE = {
    **POOL_IDENTITY,
    "dataset_id": "dataset-1",
    "expected_dataset_content_hash": "3" * 64,
    "workspace_revision": 4,
    "workspace_generation": 9,
    "semantic_mapping_hash": "4" * 64,
    "target_col": "bad",
    "sample_design_ref": LEGACY_SAMPLE_REF,
    "drop_nan_labels": True,
    "month_col": "month",
    "loan_amount_col": "loan",
}


def _spec(workflow_id: str):
    return next(spec for spec in POOL_WORKFLOW_SPECS if spec.workflow_id == workflow_id)


def _validate(workflow_id: str, inputs: Mapping[str, object]) -> dict[str, object]:
    validator = _spec(workflow_id).validator
    assert validator is not None
    return validator(inputs, RESOLUTION)


def _prepare(
    workflow_id: str,
    inputs: Mapping[str, object],
    evidence: Mapping[str, object],
    *,
    dataset_id: str | None = None,
    drop_nan_labels: bool = False,
):
    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    return preparer(
        inputs,
        StrategyWorkflowPreparationContext(
            dataset_id=dataset_id,
            drop_nan_labels=drop_nan_labels,
            bind_workflow_evidence=lambda *_args: evidence,
        ),
    )


def test_pool_workflow_specs_expose_exact_complete_metadata() -> None:
    impact_requirements = StrategyWorkflowRequirements(True, True, True)
    no_requirements = StrategyWorkflowRequirements(False, False, False)
    expected = {
        "strategy_pool_add_candidate": no_requirements,
        "strategy_pool_remove_entry": no_requirements,
        "strategy_pool_set_action": no_requirements,
        "strategy_pool_reorder": no_requirements,
        "strategy_pool_compile": no_requirements,
        "strategy_pool_materialize": no_requirements,
        "strategy_pool_apply": no_requirements,
        "strategy_pool_validation": no_requirements,
        "strategy_pool_impact": impact_requirements,
        "strategy_impact_cube": no_requirements,
        "strategy_pool_stability": no_requirements,
    }

    assert {
        spec.workflow_id: spec.requirements for spec in POOL_WORKFLOW_SPECS
    } == expected
    assert all(
        spec.fresh and spec.replayable and spec.manual and spec.migrated
        for spec in POOL_WORKFLOW_SPECS
    )
    assert all(spec.template_ids == (spec.workflow_id,) for spec in POOL_WORKFLOW_SPECS)


@pytest.mark.parametrize("spec", POOL_WORKFLOW_SPECS)
def test_pool_workflow_specs_define_complete_adapter_triples(spec) -> None:
    assert callable(spec.validator)
    assert callable(spec.confirmation)
    assert callable(spec.preparer)


def test_pool_add_rejects_noncanonical_selection_id_with_legacy_guidance() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _validate(
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "selection_id": "automatic-tree-leaf-selection-" + "A" * 32,
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

    assert "32 位小写十六进制字符" in str(captured.value)
    assert captured.value.fields == ("selection_id",)


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "expected"),
    [
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": {"type": "approval"},
                "action": {
                    "type": "reject",
                    "reason_code": "RISK",
                },
                "reason": "  人工确认  ",
            },
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
                "reason": "人工确认",
            },
        ),
        (
            "strategy_pool_remove_entry",
            {
                "strategy_type": "approval",
                "entry_id": f" {ENTRY_ID} ",
            },
            {"strategy_type": "approval", "entry_id": ENTRY_ID},
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "action": {"type": "review"},
            },
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "action": {
                    "type": "review",
                    "value": "review",
                    "reason_code": None,
                    "stop": True,
                },
            },
        ),
        (
            "strategy_pool_reorder",
            {
                "strategy_type": "approval",
                "ordered_ids": [ENTRY_ID, RULE_ID],
            },
            {
                "strategy_type": "approval",
                "ordered_ids": [ENTRY_ID, RULE_ID],
            },
        ),
        (
            "strategy_pool_compile",
            {"strategy_type": "pricing"},
            {"strategy_type": "pricing"},
        ),
        (
            "strategy_pool_materialize",
            {"strategy_type": "limit"},
            {"strategy_type": "limit"},
        ),
        (
            "strategy_pool_apply",
            {"strategy_type": "segmentation", "output_prefix": " decision_ "},
            {"strategy_type": "segmentation", "output_prefix": "decision_"},
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "reject", "partition": "oot"},
            {"strategy_type": "reject", "partition": "oot"},
        ),
        (
            "strategy_pool_impact",
            {"strategy_type": "approval"},
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "reject",
                "comparison_mode": "vs_baseline",
                "baseline_strategy_id": BASELINE_ID,
                "month_col": "month",
                "loan_amount_col": "loan",
                "overdue_amount_col": "overdue",
                "drop_nan_labels": True,
            },
            {
                "strategy_type": "reject",
                "comparison_mode": "vs_baseline",
                "baseline_strategy_id": BASELINE_ID,
                "month_col": "month",
                "loan_amount_col": "loan",
                "overdue_amount_col": "overdue",
                "drop_nan_labels": True,
            },
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "approval",
                "partitions": ["oot", "development"],
                "month_col": "month",
                "group_col": "group",
                "segment_col": "segment",
                "current_strategy_id": CURRENT_STRATEGY_ID,
                "economics_inputs": {
                    "term_months": {"kind": "scalar", "value": 12},
                    "ead": {"kind": "column", "column": "ead"},
                },
            },
            {
                "strategy_type": "approval",
                "partitions": ["development", "oot"],
                "month_col": "month",
                "group_col": "group",
                "segment_col": "segment",
                "current_strategy_id": CURRENT_STRATEGY_ID,
                "economics_inputs": {
                    "ead": {"kind": "column", "column": "ead"},
                    "term_months": {"kind": "scalar", "value": 12},
                },
            },
        ),
        (
            "strategy_pool_stability",
            {"strategy_type": "pricing"},
            {"strategy_type": "pricing"},
        ),
    ],
)
def test_pool_validators_preserve_legacy_normalization(
    workflow_id: str,
    inputs: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert _validate(workflow_id, inputs) == expected


@pytest.mark.parametrize(
    "workflow_id",
    [spec.workflow_id for spec in POOL_WORKFLOW_SPECS],
)
def test_pool_validators_reject_platform_owned_slots(workflow_id: str) -> None:
    valid_inputs = {
        "strategy_pool_add_candidate": {
            "strategy_type": "approval",
            "candidate_asset_id": ASSET_ID,
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
        "strategy_pool_remove_entry": {
            "strategy_type": "approval",
            "rule_id": RULE_ID,
        },
        "strategy_pool_set_action": {
            "strategy_type": "approval",
            "rule_id": RULE_ID,
            "action": {"type": "review"},
        },
        "strategy_pool_reorder": {
            "strategy_type": "approval",
            "ordered_ids": [RULE_ID],
        },
        "strategy_pool_compile": {"strategy_type": "approval"},
        "strategy_pool_materialize": {"strategy_type": "approval"},
        "strategy_pool_apply": {"strategy_type": "approval"},
        "strategy_pool_validation": {
            "strategy_type": "approval",
            "partition": "validation",
        },
        "strategy_pool_impact": {"strategy_type": "approval"},
        "strategy_impact_cube": {"strategy_type": "approval"},
        "strategy_pool_stability": {"strategy_type": "approval"},
    }
    inputs = {
        **valid_inputs[workflow_id],
        "expected_pool_revision": 7,
    }
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _validate(workflow_id, inputs)
    assert "expected_pool_revision" in captured.value.fields


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "field"),
    [
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "selection_id": SELECTION_ID,
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
            "selection_id",
        ),
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": "candidate-short",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
            "candidate_asset_id",
        ),
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": {"type": "limit", "value": 1000},
                "action": {"type": "reject"},
            },
            "default_action",
        ),
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
                "placement_mode": "append",
            },
            "placement_mode",
        ),
        (
            "strategy_pool_remove_entry",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "entry_id": ENTRY_ID,
            },
            "entry_id",
        ),
        (
            "strategy_pool_set_action",
            {"strategy_type": "approval", "rule_id": RULE_ID},
            "action",
        ),
        (
            "strategy_pool_reorder",
            {
                "strategy_type": "approval",
                "ordered_ids": [RULE_ID, RULE_ID],
            },
            "ordered_ids",
        ),
        (
            "strategy_pool_remove_entry",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "reason": "x" * 501,
            },
            "reason",
        ),
        (
            "strategy_pool_apply",
            {"strategy_type": "approval", "output_prefix": "9bad"},
            "output_prefix",
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "approval", "partition": "development"},
            "partition",
        ),
        (
            "strategy_pool_impact",
            {"strategy_type": "limit"},
            "strategy_type",
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "comparison_mode": "vs_baseline",
            },
            "baseline_strategy_id",
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "baseline_strategy_id": BASELINE_ID,
            },
            "baseline_strategy_id",
        ),
        (
            "strategy_pool_impact",
            {"strategy_type": "approval", "month_col": "ghost"},
            "month_col",
        ),
        (
            "strategy_pool_impact",
            {"strategy_type": "approval", "drop_nan_labels": 1},
            "drop_nan_labels",
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "approval",
                "partitions": ["development", "development"],
            },
            "partitions",
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "approval",
                "month_col": "month",
                "group_col": "month",
            },
            "group_col",
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "segmentation",
                "economics_inputs": {"ead": {"kind": "column", "column": "ead"}},
            },
            "economics_inputs",
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "approval",
                "economics_inputs": {"term_months": {"kind": "scalar", "value": 0}},
            },
            "economics_inputs",
        ),
    ],
)
def test_pool_validators_fail_closed_on_invalid_or_ambiguous_controls(
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
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
            },
            (ASSET_ID, "development / unvalidated", "可逆 draft Pool revision"),
        ),
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "entry_id": ENTRY_ID},
            (ENTRY_ID, "旧 revision 保持不可变", "不会采纳或部署"),
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "action": REJECT_ACTION,
            },
            (RULE_ID, "新动作", "revision/hash"),
        ),
        (
            "strategy_pool_reorder",
            {"strategy_type": "approval", "ordered_ids": [RULE_ID, RULE_ID_2]},
            ("完整顺序", "遗漏不会被当作删除", RULE_ID_2),
        ),
        (
            "strategy_pool_compile",
            {"strategy_type": "pricing"},
            ("canonical StrategySpec", "草案预览", "不会创建已采纳策略"),
        ),
        (
            "strategy_pool_materialize",
            {"strategy_type": "limit"},
            ("完整认证当前非空 Pool", "design hash", "不采纳、不部署"),
        ),
        (
            "strategy_pool_apply",
            {"strategy_type": "segmentation"},
            ("CAS revision/hash", "不可变派生数据集", "默认前缀"),
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "reject", "partition": "oot"},
            ("独立样本回放验证", "StrategySampleDesign V2", "不声称 PSI"),
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
            ("活动 DataWorkspace", "只读影响证据", "不默认从风险分母排除"),
        ),
        (
            "strategy_impact_cube",
            {"strategy_type": "approval"},
            ("StrategySampleDesign V2", "全部可用且非空", "聚合、可下载、可复核"),
        ),
        (
            "strategy_pool_stability",
            {"strategy_type": "pricing"},
            ("development", "exact ImpactCube", "不等于独立效果验证"),
        ),
    ],
)
def test_pool_confirmations_preserve_evidence_and_lifecycle_boundaries(
    workflow_id: str,
    inputs: dict[str, object],
    phrases: tuple[str, ...],
) -> None:
    confirmation = _spec(workflow_id).confirmation
    assert confirmation is not None
    rendered = confirmation(inputs)
    assert all(phrase in rendered for phrase in phrases)
    assert rendered.endswith(
        "请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。"
    )


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "evidence", "expected_slots", "absent"),
    [
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
                "reason": "人工确认",
            },
            ADD_EVIDENCE,
            {
                "strategy_type": "approval",
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
                "reason": "人工确认",
                **ADD_EVIDENCE,
            },
            {"candidate_asset_id", "selection_id"},
        ),
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "entry_id": ENTRY_ID},
            {**POOL_IDENTITY, "rule_id": RULE_ID},
            {"strategy_type": "approval", **POOL_IDENTITY, "rule_id": RULE_ID},
            {"entry_id"},
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "action": REJECT_ACTION,
            },
            {**POOL_IDENTITY, "rule_id": RULE_ID},
            {
                "strategy_type": "approval",
                "action": REJECT_ACTION,
                **POOL_IDENTITY,
                "rule_id": RULE_ID,
            },
            set(),
        ),
        (
            "strategy_pool_reorder",
            {
                "strategy_type": "approval",
                "ordered_ids": [ENTRY_ID, RULE_ID],
            },
            {**POOL_IDENTITY, "ordered_rule_ids": [RULE_ID_2, RULE_ID]},
            {
                "strategy_type": "approval",
                **POOL_IDENTITY,
                "ordered_rule_ids": [RULE_ID_2, RULE_ID],
            },
            {"ordered_ids"},
        ),
        (
            "strategy_pool_compile",
            {"strategy_type": "pricing"},
            POOL_IDENTITY,
            {"strategy_type": "pricing", **POOL_IDENTITY},
            set(),
        ),
        (
            "strategy_pool_materialize",
            {"strategy_type": "limit"},
            MATERIALIZE_EVIDENCE,
            {"strategy_type": "limit", **MATERIALIZE_EVIDENCE},
            set(),
        ),
        (
            "strategy_pool_apply",
            {"strategy_type": "segmentation", "output_prefix": "decision_"},
            POOL_IDENTITY,
            {
                "strategy_type": "segmentation",
                "output_prefix": "decision_",
                **POOL_IDENTITY,
            },
            set(),
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "reject", "partition": "oot"},
            VALIDATION_EVIDENCE,
            {
                "strategy_type": "reject",
                "partition": "oot",
                **VALIDATION_EVIDENCE,
            },
            set(),
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "approval",
                "partitions": ["development", "oot"],
                "month_col": "month",
                "group_col": "group",
                "segment_col": "segment",
                "current_strategy_id": CURRENT_STRATEGY_ID,
                "economics_inputs": IMPACT_CUBE_EVIDENCE["economics_inputs"],
            },
            IMPACT_CUBE_EVIDENCE,
            {"strategy_type": "approval", **IMPACT_CUBE_EVIDENCE},
            {
                "current_strategy_id",
                "month_col",
                "group_col",
                "segment_col",
            },
        ),
        (
            "strategy_pool_stability",
            {"strategy_type": "pricing"},
            STABILITY_EVIDENCE,
            {"strategy_type": "pricing", **STABILITY_EVIDENCE},
            set(),
        ),
    ],
)
def test_pool_preparers_bind_current_platform_evidence_without_pointer_leaks(
    workflow_id: str,
    inputs: dict[str, object],
    evidence: dict[str, object],
    expected_slots: dict[str, object],
    absent: set[str],
) -> None:
    calls: list[tuple[str, Mapping[str, object]]] = []

    def bind(called_workflow: str, called_inputs: Mapping[str, object]):
        calls.append((called_workflow, called_inputs))
        return evidence

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
    assert plan.workflow_id == workflow_id
    assert plan.template_id == workflow_id
    assert plan.success_criteria == ()
    assert plan.to_runtime_slots() == expected_slots
    assert not absent & set(plan.slots)


def test_pool_impact_preparer_uses_confirmed_drop_nan_and_resolved_roles() -> None:
    inputs = {
        "strategy_type": "approval",
        "comparison_mode": "absolute",
        "drop_nan_labels": False,
    }
    plan = _prepare(
        "strategy_pool_impact",
        inputs,
        POOL_IMPACT_EVIDENCE,
        dataset_id="dataset-1",
        drop_nan_labels=True,
    )

    assert plan.to_runtime_slots() == {
        "strategy_type": "approval",
        "comparison_mode": "absolute",
        **POOL_IMPACT_EVIDENCE,
    }
    assert plan.slots["drop_nan_labels"] is True
    assert "overdue_amount_col" not in plan.slots
    assert plan.success_criteria == ()


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "dataset_id"),
    [
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
            },
            None,
        ),
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "rule_id": RULE_ID},
            None,
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "action": REJECT_ACTION,
            },
            None,
        ),
        (
            "strategy_pool_reorder",
            {"strategy_type": "approval", "ordered_ids": [RULE_ID]},
            None,
        ),
        ("strategy_pool_compile", {"strategy_type": "approval"}, None),
        ("strategy_pool_materialize", {"strategy_type": "approval"}, None),
        ("strategy_pool_apply", {"strategy_type": "approval"}, None),
        (
            "strategy_pool_validation",
            {"strategy_type": "approval", "partition": "validation"},
            None,
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
            "dataset-1",
        ),
        ("strategy_impact_cube", {"strategy_type": "approval"}, None),
        ("strategy_pool_stability", {"strategy_type": "approval"}, None),
    ],
)
def test_every_pool_preparer_requires_platform_evidence_callback(
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
    ("workflow_id", "inputs", "evidence", "dataset_id"),
    [
        (
            "strategy_pool_add_candidate",
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
            },
            {
                key: value
                for key, value in ADD_EVIDENCE.items()
                if key != "expected_asset_hash"
            },
            None,
        ),
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "rule_id": RULE_ID},
            {**POOL_IDENTITY, "rule_id": RULE_ID, "pool_id": "user-visible"},
            None,
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "rule_id": RULE_ID,
                "action": REJECT_ACTION,
            },
            {**POOL_IDENTITY, "rule_id": RULE_ID, "action": REJECT_ACTION},
            None,
        ),
        (
            "strategy_pool_reorder",
            {"strategy_type": "approval", "ordered_ids": [RULE_ID]},
            {**POOL_IDENTITY, "ordered_ids": [RULE_ID]},
            None,
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "approval", "partition": "validation"},
            {**VALIDATION_EVIDENCE, "partition": "validation"},
            None,
        ),
        (
            "strategy_impact_cube",
            {"strategy_type": "approval"},
            {**IMPACT_CUBE_EVIDENCE, "strategy_type": "approval"},
            None,
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": True,
            },
            {**POOL_IMPACT_EVIDENCE, "comparison_mode": "absolute"},
            "dataset-1",
        ),
    ],
)
def test_pool_preparers_reject_missing_extra_or_conflicting_evidence(
    workflow_id: str,
    inputs: dict[str, object],
    evidence: dict[str, object],
    dataset_id: str | None,
) -> None:
    preparer = _spec(workflow_id).preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id=dataset_id,
                drop_nan_labels=bool(inputs.get("drop_nan_labels")),
                bind_workflow_evidence=lambda *_args: evidence,
            ),
        )
    assert captured.value.code in {
        "strategy_workflow_evidence_invalid",
        "strategy_workflow_evidence_conflict",
    }


def test_pool_preparer_propagates_stale_reference_callback_failure() -> None:
    def stale(*_args):
        raise StrategyWorkflowValidationError(
            "Pool changed during confirmation",
            code="strategy_pool_context_changed",
        )

    preparer = _spec("strategy_pool_compile").preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            {"strategy_type": "approval"},
            StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=stale,
            ),
        )
    assert captured.value.code == "strategy_pool_context_changed"


def test_scorecard_selection_full_band_asset_id_reaches_pool_plan() -> None:
    inputs = {
        "strategy_type": "reject",
        "selection_id": SCORECARD_SELECTION_ID,
        "default_action": APPROVAL_ACTION,
        "action": REJECT_ACTION,
    }
    evidence = {
        **ADD_EVIDENCE,
        "expected_asset_id": SCORECARD_BAND_ASSET_ID,
    }

    plan = _prepare("strategy_pool_add_candidate", inputs, evidence)

    assert plan.to_runtime_slots()["expected_asset_id"] == SCORECARD_BAND_ASSET_ID
    assert "selection_id" not in plan.to_runtime_slots()


@pytest.mark.parametrize(
    ("inputs", "evidence", "field"),
    [
        (
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
            },
            {
                **ADD_EVIDENCE,
                "expected_asset_id": "candidate-asset-" + "9" * 32,
            },
            "candidate_asset_id",
        ),
        (
            {
                "strategy_type": "approval",
                "candidate_asset_id": ASSET_ID,
                "default_action": APPROVAL_ACTION,
                "action": REJECT_ACTION,
                "placement_mode": "before_selected_members",
            },
            ADD_EVIDENCE,
            "placement_mode",
        ),
    ],
)
def test_pool_add_evidence_must_match_user_asset_and_voting_placement(
    inputs: dict[str, object],
    evidence: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _prepare("strategy_pool_add_candidate", inputs, evidence)
    assert captured.value.code == "strategy_workflow_evidence_conflict"
    assert field in captured.value.fields


@pytest.mark.parametrize(
    ("workflow_id", "inputs", "evidence", "field"),
    [
        (
            "strategy_pool_compile",
            {"strategy_type": "approval"},
            {**POOL_IDENTITY, "expected_pool_revision": 0},
            "expected_pool_revision",
        ),
        (
            "strategy_pool_apply",
            {"strategy_type": "approval"},
            {**POOL_IDENTITY, "expected_pool_snapshot_hash": "short"},
            "expected_pool_snapshot_hash",
        ),
        (
            "strategy_pool_materialize",
            {"strategy_type": "approval"},
            {**MATERIALIZE_EVIDENCE, "expected_design_hash": "short"},
            "expected_design_hash",
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "approval", "partition": "validation"},
            {
                **VALIDATION_EVIDENCE,
                "sample_design_ref": {
                    key: value
                    for key, value in V2_SAMPLE_REF.items()
                    if key != "bundle_artifact_id"
                },
            },
            "sample_design_ref",
        ),
        (
            "strategy_pool_validation",
            {"strategy_type": "approval", "partition": "validation"},
            {**VALIDATION_EVIDENCE, "population": "approval"},
            "population",
        ),
        (
            "strategy_pool_stability",
            {"strategy_type": "approval"},
            {**STABILITY_EVIDENCE, "partitions": ["development"]},
            "partitions",
        ),
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "entry_id": ENTRY_ID},
            {**POOL_IDENTITY, "rule_id": ENTRY_ID},
            "rule_id",
        ),
        (
            "strategy_pool_reorder",
            {"strategy_type": "approval", "ordered_ids": [ENTRY_ID]},
            {**POOL_IDENTITY, "ordered_rule_ids": [ENTRY_ID]},
            "ordered_rule_ids",
        ),
    ],
)
def test_pool_evidence_schemas_reject_incomplete_or_noncurrent_bindings(
    workflow_id: str,
    inputs: dict[str, object],
    evidence: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _prepare(workflow_id, inputs, evidence)
    assert captured.value.code == "strategy_workflow_evidence_invalid"
    assert field in captured.value.fields


@pytest.mark.parametrize(
    ("inputs", "evidence", "field"),
    [
        (
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": True,
            },
            {**POOL_IMPACT_EVIDENCE, "dataset_id": "dataset-other"},
            "dataset_id",
        ),
        (
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
            POOL_IMPACT_EVIDENCE,
            "drop_nan_labels",
        ),
        (
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "month_col": "month",
                "drop_nan_labels": True,
            },
            {**POOL_IMPACT_EVIDENCE, "month_col": "group"},
            "month_col",
        ),
    ],
)
def test_pool_impact_evidence_matches_workspace_policy_and_explicit_roles(
    inputs: dict[str, object],
    evidence: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _prepare(
            "strategy_pool_impact",
            inputs,
            evidence,
            dataset_id="dataset-1",
            drop_nan_labels=bool(inputs["drop_nan_labels"]),
        )
    assert captured.value.code == "strategy_workflow_evidence_conflict"
    assert field in captured.value.fields


def test_pool_impact_requires_live_dataset_before_evidence_binding() -> None:
    preparer = _spec("strategy_pool_impact").preparer
    assert preparer is not None
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        preparer(
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
            StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=lambda *_args: POOL_IMPACT_EVIDENCE,
            ),
        )
    assert captured.value.code == "strategy_workflow_dataset_binding_required"


@pytest.mark.parametrize(
    ("inputs", "evidence", "field"),
    [
        (
            {"strategy_type": "approval", "partitions": ["development"]},
            IMPACT_CUBE_EVIDENCE,
            "partitions",
        ),
        (
            {"strategy_type": "approval", "month_col": "month"},
            {
                **IMPACT_CUBE_EVIDENCE,
                "current_strategy_ref": None,
                "economics_inputs": None,
                "dimension_bindings": {
                    "month_col": "group",
                    "group_col": None,
                    "segment_col": None,
                },
            },
            "month_col",
        ),
        (
            {
                "strategy_type": "approval",
                "current_strategy_id": CURRENT_STRATEGY_ID,
            },
            {
                **IMPACT_CUBE_EVIDENCE,
                "economics_inputs": None,
                "current_strategy_ref": {
                    "strategy_id": "strategy-other",
                    "expected_strategy_spec_hash": "2" * 64,
                },
            },
            "current_strategy_id",
        ),
        (
            {
                "strategy_type": "approval",
                "economics_inputs": {"ead": {"kind": "column", "column": "ead"}},
            },
            {
                **IMPACT_CUBE_EVIDENCE,
                "current_strategy_ref": None,
                "economics_inputs": None,
            },
            "economics_inputs",
        ),
    ],
)
def test_impact_cube_evidence_matches_explicit_controls_and_refs(
    inputs: dict[str, object],
    evidence: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        _prepare("strategy_impact_cube", inputs, evidence)
    assert captured.value.code == "strategy_workflow_evidence_conflict"
    assert field in captured.value.fields

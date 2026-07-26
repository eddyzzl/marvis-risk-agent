from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import operator

import pandas as pd
import pytest

from marvis.packs.strategy.apply_projection import (
    DEFAULT_POOL_APPLY_PREFIX,
    deterministic_rule_counts,
    deterministic_string_counts,
    resolve_apply_output_columns,
    serialize_strategy_decisions,
)
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    compile_strategy_pool,
)
from marvis.packs.strategy.pool_apply import apply_strategy_pool


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


def _action(
    action_type: str,
    value: object,
    reason_code: str | None,
) -> dict[str, object]:
    return {
        "type": action_type,
        "value": value,
        "reason_code": reason_code,
        "stop": True,
    }


def _fragment(
    *,
    suffix: str,
    rule_id: str,
    condition: dict[str, object],
) -> dict[str, object]:
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{suffix}",
            "artifact_kind": "strategy_test_candidate_json",
            "artifact_schema_version": "strategy.test-candidate-artifact.v1",
            "artifact_content_hash": suffix * 64,
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "strategy.test-candidate.v1",
            "asset_id": f"asset-{suffix}",
            "asset_hash": suffix * 64,
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=rule_id,
        condition=condition,
        requirements=[],
        effect_id=f"effect-{suffix}",
        evidence_id="evidence-1",
        evidence_hash=_HASH_C,
        evidence_identity={
            "dataset_id": "dataset-1",
            "dataset_content_hash": _HASH_A,
            "workspace_revision": 3,
            "workspace_generation": 2,
            "semantic_mapping_hash": _HASH_B,
            "sample_context_hash": _HASH_D,
        },
    )


def _pool() -> dict[str, object]:
    pool = add_verified_candidate_fragment(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_action("approval", "approve", None),
        verified_candidate_fragment=_fragment(
            suffix="1",
            rule_id="rule-low-score",
            condition={
                "op": "compare",
                "field": "virtual_risk_score",
                "operator": "<",
                "value": 500,
                "missing": "no_match",
            },
        ),
        action=_action("review", "review", "LOW_SCORE_REVIEW"),
    )
    return add_verified_candidate_fragment(
        pool,
        task_id="task-1",
        strategy_type="approval",
        default_action=_action("approval", "approve", None),
        verified_candidate_fragment=_fragment(
            suffix="2",
            rule_id="rule-high-score",
            condition={
                "op": "compare",
                "field": "virtual_risk_score",
                "operator": ">=",
                "value": 800,
                "missing": "no_match",
            },
        ),
        action=_action("reject", "reject", "HIGH_SCORE_REJECT"),
    )


def test_apply_pool_projects_six_decision_columns_without_virtual_fields() -> None:
    original = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-2", "C-3"],
            "raw_score": pd.Series([10, 20, 30], dtype="int64"),
        }
    )
    execution = original.copy(deep=True)
    execution["virtual_risk_score"] = [100.0, 700.0, 900.0]
    pool = _pool()

    result = apply_strategy_pool(
        original,
        execution,
        pool,
        compiled_design=compile_strategy_pool(pool),
    )

    assert result.derived_frame.columns.tolist() == [
        "customer_id",
        "raw_score",
        "strategy_pool_action",
        "strategy_pool_value",
        "strategy_pool_value_type",
        "strategy_pool_rule_id",
        "strategy_pool_entry_id",
        "strategy_pool_reason_code",
    ]
    assert "virtual_risk_score" not in result.derived_frame
    assert result.derived_frame["customer_id"].tolist() == ["C-1", "C-2", "C-3"]
    assert result.derived_frame["raw_score"].dtype == original["raw_score"].dtype
    assert result.derived_frame["strategy_pool_action"].tolist() == [
        "review",
        "approval",
        "reject",
    ]
    assert result.derived_frame["strategy_pool_value"].tolist() == [
        "review",
        "approve",
        "reject",
    ]
    assert result.derived_frame["strategy_pool_value_type"].tolist() == [
        "string",
        "string",
        "string",
    ]
    assert result.derived_frame["strategy_pool_rule_id"].tolist() == [
        "rule-low-score",
        None,
        "rule-high-score",
    ]
    entry_ids = {
        entry["rule_id"]: entry["entry_id"] for entry in pool["entries"]
    }
    assert result.derived_frame["strategy_pool_entry_id"].tolist() == [
        entry_ids["rule-low-score"],
        None,
        entry_ids["rule-high-score"],
    ]
    assert result.derived_frame["strategy_pool_reason_code"].tolist() == [
        "LOW_SCORE_REVIEW",
        None,
        "HIGH_SCORE_REJECT",
    ]
    assert result.source_row_count == 3
    assert dict(result.action_counts) == {"approval": 1, "reject": 1, "review": 1}
    assert dict(result.rule_counts) == {
        "rule-high-score": 1,
        "rule-low-score": 1,
    }
    assert dict(result.entry_counts) == {
        entry_ids["rule-high-score"]: 1,
        entry_ids["rule-low-score"]: 1,
    }
    assert result.default_count == 1
    assert len(result.strategy_spec_hash) == 64
    assert result.design_hash == compile_strategy_pool(pool)["design_hash"]
    assert len(result.assignment_hash) == 64


def test_apply_pool_rejects_more_than_one_million_source_rows() -> None:
    oversized = pd.DataFrame(index=pd.RangeIndex(1_000_001))

    with pytest.raises(ValueError, match="row budget"):
        apply_strategy_pool(oversized, oversized.copy(), _pool())


def test_apply_pool_rejects_more_than_five_hundred_source_columns() -> None:
    oversized = pd.DataFrame(
        {f"source_{index}": pd.Series(dtype="float64") for index in range(501)}
    )

    with pytest.raises(ValueError, match="source column budget"):
        apply_strategy_pool(oversized, oversized.copy(), _pool())


def test_apply_pool_rejects_compiled_design_from_another_snapshot() -> None:
    original = pd.DataFrame({"customer_id": ["C-1"], "raw_score": [10]})
    execution = original.assign(virtual_risk_score=[100.0])
    pool = _pool()
    stale_design = deepcopy(compile_strategy_pool(pool))
    stale_design["design_hash"] = "f" * 64

    with pytest.raises(ValueError, match="does not match Strategy Pool"):
        apply_strategy_pool(
            original,
            execution,
            pool,
            compiled_design=stale_design,
        )


@pytest.mark.parametrize(
    ("original", "execution", "message"),
    [
        (
            pd.DataFrame({"raw_score": [10]}, index=[1]),
            pd.DataFrame(
                {"raw_score": [10], "virtual_risk_score": [100.0]},
                index=[1],
            ),
            "zero-based",
        ),
        (
            pd.DataFrame({"raw_score": [10, 20]}),
            pd.DataFrame(
                {
                    "raw_score": [20, 10],
                    "virtual_risk_score": [100.0, 900.0],
                }
            ),
            "changed source values",
        ),
        (
            pd.DataFrame({"raw_score": pd.Series([10], dtype="int64")}),
            pd.DataFrame(
                {
                    "raw_score": pd.Series([10.0], dtype="float64"),
                    "virtual_risk_score": [100.0],
                }
            ),
            "changed source values",
        ),
    ],
)
def test_apply_pool_rejects_source_row_or_dtype_drift(
    original: pd.DataFrame,
    execution: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_strategy_pool(original, execution, _pool())


def test_apply_pool_requires_safe_non_overwriting_output_prefix() -> None:
    original = pd.DataFrame(
        {
            "raw_score": [10],
            "STRATEGY_POOL_ACTION": ["existing"],
        }
    )
    execution = original.assign(virtual_risk_score=[100.0])

    with pytest.raises(ValueError, match="already exist"):
        apply_strategy_pool(original, execution, _pool())
    with pytest.raises(ValueError, match="ASCII"):
        apply_strategy_pool(
            original.drop(columns=["STRATEGY_POOL_ACTION"]),
            execution.drop(columns=["STRATEGY_POOL_ACTION"]),
            _pool(),
            output_prefix="策略_",
        )


def test_projection_helpers_keep_typed_values_and_sorted_counts() -> None:
    decisions = pd.Series(
        [None, {"b": 1, "a": 2}, [1, "x"], True, 7, 0.25, "S"],
        dtype="object",
    )

    serialized = serialize_strategy_decisions(
        decisions,
        strategy_type="segmentation",
    )

    assert serialized.values.tolist() == [
        "null",
        '{"a":2,"b":1}',
        '[1,"x"]',
        "true",
        "7",
        "0.25",
        "S",
    ]
    assert serialized.value_types.tolist() == [
        "null",
        "object",
        "array",
        "boolean",
        "integer",
        "number",
        "string",
    ]
    counted = pd.Series(["z", None, "a", "z"], dtype="object")
    assert deterministic_string_counts(counted) == {
        "None": 1,
        "a": 1,
        "z": 2,
    }
    assert deterministic_rule_counts(counted) == {"a": 1, "z": 2}

    numeric = serialize_strategy_decisions(
        pd.Series([1000, 1250.5], dtype="object"),
        strategy_type="limit",
    )
    assert numeric.values.tolist() == [1000, 1250.5]


def test_output_column_helper_supports_persisted_and_pool_shapes() -> None:
    persisted = resolve_apply_output_columns(
        ["source"],
        output_columns={"action": "DecisionAction"},
        default_prefix="strategy_",
        include_entry_id=False,
    )
    assert persisted.as_dict() == {
        "action": "DecisionAction",
        "value": "strategy_value",
        "value_type": "strategy_value_type",
        "rule_id": "strategy_rule_id",
        "reason_code": "strategy_reason_code",
    }
    pool = resolve_apply_output_columns(
        ["source"],
        default_prefix=DEFAULT_POOL_APPLY_PREFIX,
        include_entry_id=True,
    )
    assert pool.entry_id == "strategy_pool_entry_id"

    with pytest.raises(ValueError, match="case-insensitively unique"):
        resolve_apply_output_columns(
            ["source"],
            output_columns={
                "action": "Decision",
                "value": "decision",
            },
            include_entry_id=False,
        )


def test_pool_apply_result_is_frozen_and_summary_mappings_are_read_only() -> None:
    original = pd.DataFrame({"raw_score": [10]})
    execution = original.assign(virtual_risk_score=[100.0])
    result = apply_strategy_pool(original, execution, _pool())

    with pytest.raises(FrozenInstanceError):
        setattr(result, "default_count", 99)
    with pytest.raises(TypeError):
        operator.setitem(result.action_counts, "forged", 1)
    changed_frame = result.derived_frame.assign(forged=True)
    assert replace(result, derived_frame=changed_frame) == result

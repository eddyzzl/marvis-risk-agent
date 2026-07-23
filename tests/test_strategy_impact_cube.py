from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from marvis.packs.strategy.candidate_fragment import build_verified_candidate_fragment
from marvis.packs.strategy.dsl import (
    StrategyAction,
    StrategyRuleSpec,
    StrategySpec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube import (
    STRATEGY_IMPACT_CUBE_SCHEMA_VERSION,
    STRATEGY_IMPACT_SLICE_SCHEMA_VERSION,
    build_strategy_impact_cube,
    canonical_strategy_impact_cube_json,
    validate_strategy_impact_cube,
)
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    compile_strategy_pool,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _typed_action(strategy_type: str, value) -> dict:
    if strategy_type in {"approval", "reject"}:
        action_type = "approval" if value == "approve" else str(value)
        reason = None if value == "approve" else str(value).upper()
    elif strategy_type == "segmentation":
        action_type = "segment"
        reason = None
    else:
        action_type = strategy_type
        reason = None
    return {
        "type": action_type,
        "value": value,
        "reason_code": reason,
        "stop": True,
    }


def _actions(strategy_type: str) -> tuple[object, object, object]:
    return {
        "approval": ("approve", "reject", "review"),
        "reject": ("approve", "reject", "review"),
        "limit": (1000.0, 2000.0, 500.0),
        "pricing": (0.10, 0.18, 0.14),
        "segmentation": ("C", "A", "B"),
    }[strategy_type]


def _condition(operator: str, value: int) -> dict:
    return {
        "op": "compare",
        "field": "x",
        "operator": operator,
        "value": value,
        "missing": "no_match",
    }


def _fragment(index: int, condition: dict) -> dict:
    suffix = f"{index:064x}"
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{index}",
            "artifact_kind": "test_candidate_json",
            "artifact_schema_version": "test.candidate-artifact.v1",
            "artifact_content_hash": suffix,
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "test.candidate.v1",
            "asset_id": f"candidate-asset-{index}",
            "asset_hash": suffix,
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=f"candidate-rule-{index}",
        condition=condition,
        requirements=[],
        effect_id=f"candidate-effect-{index}",
        evidence_id="candidate-evidence-1",
        evidence_hash=HASH_D,
        evidence_identity={
            "dataset_id": "dataset-1",
            "dataset_content_hash": HASH_A,
            "workspace_revision": 3,
            "workspace_generation": 1,
            "semantic_mapping_hash": HASH_B,
            "sample_context_hash": HASH_C,
        },
    )


def _pool(strategy_type: str) -> dict:
    default, first, second = _actions(strategy_type)
    result = None
    for index, (condition, action) in enumerate(
        (
            (_condition("<", 2), first),
            (_condition("<", 4), second),
        ),
        start=1,
    ):
        result = add_verified_candidate_fragment(
            result,
            task_id="task-1",
            strategy_type=strategy_type,
            default_action=_typed_action(strategy_type, default),
            verified_candidate_fragment=_fragment(index, condition),
            action=_typed_action(strategy_type, action),
        )
    assert result is not None
    return result


def _current_spec(strategy_type: str) -> dict:
    default, first, _second = _actions(strategy_type)
    return StrategySpec(
        strategy_type=strategy_type,
        default_action=StrategyAction.from_dict(
            _typed_action(strategy_type, default)
        ),
        rules=(
            StrategyRuleSpec(
                rule_id="current-rule-1",
                priority=1,
                condition=_condition("<", 3),
                action=StrategyAction.from_dict(
                    _typed_action(strategy_type, first)
                ),
            ),
        ),
    ).to_dict()


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "bad": [0, 1, None, 0, 1, 0],
            "month": ["202601", "202601", "202602", "202602", "202603", "202603"],
            "group": ["g1", "g1", None, "g2", "g2", None],
            "segment": ["s1", None, "s1", "s2", "s2", None],
            "ead": [1000.0, 2000.0, 1500.0, 1000.0, 500.0, 2500.0],
            "pd": [0.01, 0.20, 0.10, 0.03, 0.30, 0.02],
            "utilization": [0.5, 0.4, 0.7, 0.2, 0.9, 0.6],
            "loan_amount": [100.0, 200.0, None, 400.0, 500.0, 600.0],
            "overdue_amount": [0.0, 20.0, 30.0, None, 50.0, 60.0],
        }
    )


def _economics(strategy_type: str) -> dict | None:
    if strategy_type in {"approval", "reject"}:
        return {
            "ead": {"kind": "column", "column": "ead"},
            "pd": {"kind": "column", "column": "pd"},
            "annual_rate": {"kind": "scalar", "value": 0.20},
            "funding_rate": {"kind": "scalar", "value": 0.05},
            "lgd": {"kind": "scalar", "value": 0.5},
            "operating_cost_per_loan": {"kind": "scalar", "value": 10.0},
            "term_months": {"kind": "scalar", "value": 12},
        }
    if strategy_type == "limit":
        return {
            "pd": {"kind": "column", "column": "pd"},
            "lgd": {"kind": "scalar", "value": 0.5},
            "utilization": {"kind": "column", "column": "utilization"},
        }
    if strategy_type == "pricing":
        return {
            "ead": {"kind": "column", "column": "ead"},
            "pd": {"kind": "column", "column": "pd"},
            "lgd": {"kind": "scalar", "value": 0.5},
            "funding_rate": {"kind": "scalar", "value": 0.05},
            "term_months": {"kind": "scalar", "value": 12},
            "operating_cost_per_loan": {"kind": "scalar", "value": 10.0},
        }
    return None


def _pool_artifact_ref() -> dict:
    return {
        "artifact_id": "1" * 64,
        "artifact_content_hash": "2" * 64,
    }


def _sample_design_v2_ref(
    *,
    approval_counts: dict[str, int] | None = None,
    risk_counts: dict[str, int] | None = None,
) -> dict:
    risk = risk_counts or {"development": 3, "validation": 3}
    approval = approval_counts or dict(risk)
    return {
        "membership_artifact_id": "3" * 64,
        "membership_artifact_content_hash": "4" * 64,
        "membership_id": "strategy-sample-membership-" + "5" * 24,
        "membership_content_hash": "6" * 64,
        "bundle_artifact_id": "7" * 64,
        "bundle_artifact_content_hash": "8" * 64,
        "bundle_id": "strategy-sample-design-v2-bundle-" + "9" * 24,
        "bundle_content_hash": "a" * 64,
        "sample_design_id": "strategy-sample-design-v2-" + "b" * 24,
        "sample_design_content_hash": "c" * 64,
        "analysis_universe_row_count": 6,
        "partition_counts": dict(risk),
        "population_partition_counts": {
            "approval": dict(approval),
            "risk": dict(risk),
        },
    }


def _dataset_binding() -> dict:
    return {
        "task_id": "task-1",
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "dataset_source_path": "task-1/sample.parquet",
        "dataset_registry_metadata_hash": "d" * 64,
        "workspace_revision": 3,
        "workspace_generation": 1,
        "semantic_mapping_hash": HASH_B,
    }


def _legacy_ref() -> dict:
    return {
        "artifact_id": "e" * 64,
        "artifact_content_hash": "f" * 64,
        "sample_design_id": "strategy-sample-design-" + "1" * 24,
        "sample_design_content_hash": "2" * 64,
        "partition": "development",
    }


def _build(
    strategy_type: str,
    *,
    economics_bindings: dict | None | object = ...,
    month_col: str | None = "month",
    group_col: str | None = "group",
    segment_col: str | None = "segment",
    include_current: bool = True,
    loan_amount_col: str | None = "loan_amount",
    overdue_amount_col: str | None = "overdue_amount",
) -> dict:
    frame = _frame()
    current_spec = _current_spec(strategy_type) if include_current else None
    economics = (
        _economics(strategy_type)
        if economics_bindings is ...
        else economics_bindings
    )
    return build_strategy_impact_cube(
        pool=_pool(strategy_type),
        approval_partition_frames={
            "development": frame.iloc[:3].reset_index(drop=True),
            "validation": frame.iloc[3:].reset_index(drop=True),
        },
        partition_frames={
            "development": frame.iloc[:3].reset_index(drop=True),
            "validation": frame.iloc[3:].reset_index(drop=True),
        },
        pool_artifact_ref=_pool_artifact_ref(),
        sample_design_v2_ref=_sample_design_v2_ref(),
        dataset_binding=_dataset_binding(),
        legacy_development_ref=_legacy_ref(),
        target_col="bad",
        target_bad_value=1,
        month_col=month_col,
        group_col=group_col,
        segment_col=segment_col,
        current_strategy_spec=current_spec,
        current_strategy_ref=(
            {
                "strategy_id": f"current-{strategy_type}",
                "strategy_type": strategy_type,
                "strategy_spec_hash": strategy_spec_hash(current_spec),
            }
            if include_current
            else None
        ),
        economics_bindings=economics,
        loan_amount_col=loan_amount_col,
        overdue_amount_col=overdue_amount_col,
    )


def _slices(
    cube: dict,
    *,
    family: str,
    partition: str = "development",
    population_role: str = "risk",
) -> list[dict]:
    return [
        item
        for item in cube["slices"]
        if item["family"] == family
        and item["dimensions"]["partition"]["value"] == partition
        and item["population_role"] == population_role
    ]


def _overall(
    cube: dict,
    partition: str = "development",
    *,
    population_role: str = "risk",
) -> dict:
    rows = _slices(
        cube,
        family="overall",
        partition=partition,
        population_role=population_role,
    )
    assert len(rows) == 1
    return rows[0]


def test_impact_cube_keeps_approval_and_risk_denominators_separate() -> None:
    frame = _frame()
    cube = build_strategy_impact_cube(
        pool=_pool("approval"),
        approval_partition_frames={
            "development": frame.iloc[:4].reset_index(drop=True),
            "validation": frame.iloc[4:].reset_index(drop=True),
        },
        partition_frames={
            "development": frame.iloc[:3].reset_index(drop=True),
            "validation": frame.iloc[4:].reset_index(drop=True),
        },
        pool_artifact_ref=_pool_artifact_ref(),
        sample_design_v2_ref=_sample_design_v2_ref(
            approval_counts={"development": 4, "validation": 2},
            risk_counts={"development": 3, "validation": 2},
        ),
        dataset_binding=_dataset_binding(),
        legacy_development_ref=_legacy_ref(),
        target_col="bad",
        target_bad_value=1,
        month_col=None,
        group_col=None,
        segment_col=None,
        current_strategy_spec=None,
        current_strategy_ref=None,
        economics_bindings=None,
    )

    approval = _slices(
        cube,
        family="overall",
        population_role="approval",
    )[0]
    risk = _slices(cube, family="overall", population_role="risk")[0]
    assert approval["population"]["value"]["count"] == 4
    assert approval["new"]["value"]["population_count"] == 4
    assert approval["population"]["value"]["risk"]["availability"] == (
        "not_applicable"
    )
    assert risk["population"]["value"]["count"] == 3
    assert risk["new"]["value"]["population_count"] == 3
    assert risk["population"]["value"]["risk"]["availability"] == "present"
    assert {
        (row["role"], row["population_key"], row["row_count"])
        for row in cube["partitions"]
    } == {
        ("approval", "approval/development", 4),
        ("approval", "approval/validation", 2),
        ("risk", "risk/development", 3),
        ("risk", "risk/validation", 2),
    }


def test_impact_cube_preserves_amount_coverage_sums_and_waterfall_conservation() -> None:
    cube = _build("approval")
    approval = _slices(
        cube,
        family="overall",
        population_role="approval",
    )[0]
    amounts = approval["population"]["value"]["amounts"]
    assert amounts["loan_amount"]["value"] == {
        "column": "loan_amount",
        "covered_count": 2,
        "coverage": pytest.approx(2 / 3),
        "sum": 300.0,
    }
    assert amounts["overdue_amount"]["value"] == {
        "column": "overdue_amount",
        "covered_count": 3,
        "coverage": 1.0,
        "sum": 50.0,
    }
    waterfall = approval["waterfall"]["value"]
    reached = [
        row["incremental"]["amounts"]["loan_amount"]["value"]["sum"]
        for row in waterfall["entries"]
    ]
    default_sum = waterfall["default_unmatched"]["effect"]["amounts"][
        "loan_amount"
    ]["value"]["sum"]
    assert sum(reached) + default_sum == pytest.approx(300.0)

    unavailable = _build(
        "approval",
        loan_amount_col=None,
        overdue_amount_col=None,
    )
    unavailable_amounts = _slices(
        unavailable,
        family="overall",
        population_role="approval",
    )[0]["population"]["value"]["amounts"]
    assert unavailable_amounts["loan_amount"] == {
        "availability": "unavailable",
        "reason": "loan_amount_field_not_bound",
        "value": None,
    }


def test_new_action_bucket_must_match_waterfall_action_identity() -> None:
    cube = copy.deepcopy(_build("approval"))
    row = next(
        item
        for item in _slices(cube, family="new_action")
        if item["dimensions"]["new_action_bucket"]["value"]["type"]
        == "reject"
    )
    row["dimensions"]["new_action_bucket"]["value"]["reason_code"] = (
        "TAMPERED_REASON"
    )
    _rehash_slice(row)
    _sort_slices(cube)
    _rehash_cube(cube)

    with pytest.raises(StrategyError, match="new_action"):
        validate_strategy_impact_cube(cube)


def test_pricing_profit_identity_and_dimension_economics_rollup_fail_closed() -> None:
    cube = copy.deepcopy(_build("pricing"))
    overall = _slices(
        cube,
        family="overall",
        partition="validation",
        population_role="approval",
    )[0]
    values = overall["economics"]["value"]
    values["new"]["profit"] += 1.0
    values["new"]["profit_delta_vs_baseline"] += 1.0
    values["new"]["roa"] = (
        values["new"]["profit"] / values["new"]["total_ead"]
    )
    values["delta"]["profit"] += 1.0
    values["delta"]["roa"] = (
        values["new"]["roa"] - values["current"]["roa"]
    )
    _rehash_slice(overall)
    _rehash_cube(cube)
    with pytest.raises(StrategyError, match="profit identity"):
        validate_strategy_impact_cube(cube)

    cube = copy.deepcopy(_build("pricing"))
    child = _slices(
        cube,
        family="group",
        partition="validation",
        population_role="approval",
    )[0]
    values = child["economics"]["value"]
    values["new"]["revenue"] += 1.0
    values["new"]["profit"] += 1.0
    values["new"]["profit_delta_vs_baseline"] += 1.0
    values["new"]["roa"] = (
        values["new"]["profit"] / values["new"]["total_ead"]
    )
    values["delta"]["revenue"] += 1.0
    values["delta"]["profit"] += 1.0
    values["delta"]["roa"] = (
        values["new"]["roa"] - values["current"]["roa"]
    )
    _rehash_slice(child)
    _rehash_cube(cube)
    with pytest.raises(StrategyError, match="economics.*roll"):
        validate_strategy_impact_cube(cube)


def test_impact_cube_redacts_small_dimension_cells_without_raw_values() -> None:
    cube = _build("approval", segment_col=None)

    group_rows = _slices(cube, family="group")
    assert group_rows
    assert all(row["population"]["value"]["count"] >= 2 for row in group_rows)
    assert any(
        row["dimensions"]["group"] == {"kind": "redacted", "value": None}
        for row in group_rows
    )
    canonical = canonical_strategy_impact_cube_json(cube)
    assert '"group":{"kind":"null","value":null}' not in canonical
    assert cube["source_bindings"]["privacy"]["minimum_group_size"] == 2
    assert any(
        flag["code"] == "dimension_cells_redacted"
        for flag in cube["red_flags"]
    )


def test_impact_cube_rejects_single_row_population_partitions() -> None:
    frame = _frame().iloc[:1].reset_index(drop=True)
    sample_ref = _sample_design_v2_ref(
        approval_counts={"development": 1},
        risk_counts={"development": 1},
    )
    sample_ref["analysis_universe_row_count"] = 1

    with pytest.raises(StrategyError, match="minimum group size"):
        build_strategy_impact_cube(
            pool=_pool("approval"),
            approval_partition_frames={"development": frame},
            partition_frames={"development": frame},
            pool_artifact_ref=_pool_artifact_ref(),
            sample_design_v2_ref=sample_ref,
            dataset_binding=_dataset_binding(),
            legacy_development_ref=_legacy_ref(),
            target_col="bad",
            target_bad_value=1,
            month_col=None,
            group_col=None,
            segment_col=None,
            current_strategy_spec=None,
            current_strategy_ref=None,
            economics_bindings=None,
        )


def _rehash_slice(row: dict) -> None:
    body = {
        key: value
        for key, value in row.items()
        if key not in {"slice_id", "content_hash"}
    }
    row["slice_id"] = "strategy-impact-slice-" + hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()[:24]
    without_hash = {
        key: value for key, value in row.items() if key != "content_hash"
    }
    row["content_hash"] = hashlib.sha256(
        json.dumps(
            without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _rehash_cube(cube: dict) -> None:
    body = {
        key: value
        for key, value in cube.items()
        if key not in {"cube_id", "content_hash"}
    }
    cube["cube_id"] = "strategy-impact-cube-" + hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()[:24]
    without_hash = {
        key: value for key, value in cube.items() if key != "content_hash"
    }
    cube["content_hash"] = hashlib.sha256(
        json.dumps(
            without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sort_slices(cube: dict) -> None:
    cube["slices"].sort(
        key=lambda item: (
            ["approval", "risk"].index(item["population_role"]),
            ["development", "validation", "oot"].index(
                item["dimensions"]["partition"]["value"]
            ),
            [
                "overall",
                "month",
                "group",
                "segment",
                "group_month",
                "segment_month",
                "new_action",
            ].index(item["family"]),
            json.dumps(
                item["dimensions"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    )


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
def test_impact_cube_preserves_all_five_typed_strategy_semantics(
    strategy_type: str,
) -> None:
    cube = _build(strategy_type)
    overall = _overall(cube)

    assert cube["schema_version"] == STRATEGY_IMPACT_CUBE_SCHEMA_VERSION
    assert cube["identity"]["strategy_type"] == strategy_type
    assert overall["schema_version"] == STRATEGY_IMPACT_SLICE_SCHEMA_VERSION
    assert overall["availability"] == "present"
    assert overall["new"]["availability"] == "present"
    assert overall["new"]["value"]["strategy_type"] == strategy_type
    assert overall["current"]["availability"] == "present"
    assert overall["transition"]["availability"] == "present"
    assert sum(
        row["effect"]["count"]
        for row in overall["transition"]["value"]["rows"]
    ) == overall["population"]["value"]["count"]
    assert overall["waterfall"]["availability"] == "present"
    assert len(overall["waterfall"]["value"]["entries"]) == 2
    assert overall["conservation"] == {
        "waterfall_incremental_plus_default_equals_population": True,
        "waterfall_standalone_equals_incremental_plus_shadowed": True,
        "transition_equals_population": True,
    }
    assert sum(
        row["population"]["value"]["count"]
        for row in _slices(cube, family="new_action")
    ) == overall["population"]["value"]["count"]
    assert validate_strategy_impact_cube(cube) == cube


def test_impact_cube_unifies_month_group_segment_null_buckets_and_group_month() -> None:
    cube = _build("approval")
    overall = _overall(cube)

    assert cube["slice_families"]["month"] == {
        "availability": "present",
        "reason": None,
    }
    assert cube["slice_families"]["group"] == {
        "availability": "present",
        "reason": None,
    }
    assert cube["slice_families"]["segment"] == {
        "availability": "present",
        "reason": None,
    }
    assert any(
        row["dimensions"]["group"] == {"kind": "redacted", "value": None}
        for row in _slices(cube, family="group")
    )
    assert any(
        row["dimensions"]["segment"] == {
            "kind": "redacted",
            "value": None,
        }
        for row in _slices(cube, family="segment")
    )
    for family in ("month", "group", "segment", "group_month"):
        assert sum(
            row["population"]["value"]["count"]
            for row in _slices(cube, family=family)
        ) == overall["population"]["value"]["count"]
    assert all(
        row["dimensions"]["group"]["kind"] != "all"
        and row["dimensions"]["month"]["kind"] != "all"
        for row in _slices(cube, family="group_month")
    )


def test_missing_external_dimensions_emit_typed_unavailable_slices_not_zero() -> None:
    cube = _build(
        "approval",
        month_col=None,
        group_col=None,
        segment_col=None,
    )

    for family in ("month", "group", "segment", "group_month"):
        assert cube["slice_families"][family]["availability"] == "unavailable"
        rows = _slices(cube, family=family)
        assert len(rows) == 1
        assert rows[0]["availability"] == "unavailable"
        assert rows[0]["population"]["availability"] == "unavailable"
        assert rows[0]["population"]["value"] is None
        assert rows[0]["new"]["value"] is None
        assert rows[0]["waterfall"]["value"] is None


@pytest.mark.parametrize("strategy_type", ["approval", "reject", "limit", "pricing"])
def test_economics_use_only_bound_deterministic_inputs_without_row_details(
    strategy_type: str,
) -> None:
    cube = _build(strategy_type)
    economics = _overall(
        cube,
        population_role="approval",
    )["economics"]

    assert economics["availability"] == "present"
    assert economics["reason"] is None
    assert economics["value"]["new"]
    assert economics["value"]["current"]
    assert "by_row" not in economics["value"]["new"]
    assert "by_row" not in economics["value"]["current"]
    canonical = canonical_strategy_impact_cube_json(cube)
    assert '"by_row"' not in canonical


@pytest.mark.parametrize("strategy_type", ["approval", "reject"])
def test_decision_economics_bind_current_profit_as_new_baseline(
    strategy_type: str,
) -> None:
    economics = _overall(
        _build(strategy_type),
        partition="validation",
        population_role="approval",
    )["economics"]["value"]

    assert economics["current"]["profit"] != 0
    assert economics["new"]["profit"] != economics["current"]["profit"]
    assert (
        economics["new"]["baseline_profit"]
        == economics["current"]["profit"]
    )
    assert (
        economics["new"]["profit_delta_vs_baseline"]
        == economics["delta"]["profit"]
    )


@pytest.mark.parametrize("strategy_type", ["approval", "reject", "limit", "pricing"])
def test_missing_economics_are_typed_unavailable_never_zero(
    strategy_type: str,
) -> None:
    cube = _build(strategy_type, economics_bindings=None)
    economics = _overall(
        cube,
        population_role="approval",
    )["economics"]

    assert economics == {
        "availability": "unavailable",
        "reason": "economics_inputs_not_provided",
        "value": None,
    }


def test_explicit_empty_economics_bundle_is_distinct_and_validated() -> None:
    cube = _build("approval", economics_bindings={})

    assert _overall(
        cube,
        population_role="approval",
    )["economics"]["availability"] == "unavailable"
    assert _overall(
        cube,
        population_role="approval",
    )["economics"]["reason"].startswith(
        "missing_economics_inputs:"
    )
    assert validate_strategy_impact_cube(cube) == cube


def test_segmentation_economics_are_typed_not_applicable() -> None:
    assert _overall(
        _build("segmentation"),
        population_role="approval",
    )["economics"] == {
        "availability": "not_applicable",
        "reason": "segmentation_has_no_economic_contract",
        "value": None,
    }


def test_current_transition_is_typed_unavailable_when_current_is_not_bound() -> None:
    overall = _overall(_build("approval", include_current=False))

    assert overall["current"] == {
        "availability": "unavailable",
        "reason": "current_strategy_not_bound",
        "value": None,
    }
    assert overall["transition"] == {
        "availability": "unavailable",
        "reason": "current_strategy_not_bound",
        "value": None,
    }


def test_action_buckets_and_transitions_preserve_complete_typed_action_identity() -> None:
    pool = _pool("approval")
    new_spec = compile_strategy_pool(pool)["strategy_spec"]
    current_spec = copy.deepcopy(new_spec)
    current_spec["rules"][0]["action"]["reason_code"] = "LEGACY_REJECT"
    frame = _frame()

    cube = build_strategy_impact_cube(
        pool=pool,
        partition_frames={
            "development": frame.iloc[:3].reset_index(drop=True),
            "validation": frame.iloc[3:].reset_index(drop=True),
        },
        pool_artifact_ref=_pool_artifact_ref(),
        sample_design_v2_ref=_sample_design_v2_ref(),
        dataset_binding=_dataset_binding(),
        legacy_development_ref=_legacy_ref(),
        target_col="bad",
        target_bad_value=1,
        month_col="month",
        group_col="group",
        segment_col="segment",
        current_strategy_spec=current_spec,
        current_strategy_ref={
            "strategy_id": "legacy-current",
            "strategy_type": "approval",
            "strategy_spec_hash": strategy_spec_hash(current_spec),
        },
        economics_bindings=_economics("approval"),
    )

    action_values = {
        json.dumps(
            row["dimensions"]["new_action_bucket"]["value"],
            sort_keys=True,
        )
        for row in _slices(cube, family="new_action")
    }
    assert all('"reason_code"' in value and '"stop"' in value for value in action_values)
    transition_rows = _overall(cube)["transition"]["value"]["rows"]
    assert any(
        row["from_bucket"]["reason_code"] == "LEGACY_REJECT"
        and row["to_bucket"]["reason_code"] == "REJECT"
        and row["direction"] == "changed"
        for row in transition_rows
    )


def test_impact_cube_is_content_addressed_deterministic_and_rejects_tampering() -> None:
    first = _build("pricing")
    second = _build("pricing")

    assert first == second
    assert canonical_strategy_impact_cube_json(first) == (
        canonical_strategy_impact_cube_json(second)
    )
    assert len(first["content_hash"]) == 64
    assert first["cube_id"].startswith("strategy-impact-cube-")

    tampered = copy.deepcopy(first)
    _overall(tampered)["population"]["value"]["count"] += 1
    with pytest.raises(StrategyError, match="content_hash|population"):
        validate_strategy_impact_cube(tampered)

    coherently_rehashed = copy.deepcopy(first)
    _overall(coherently_rehashed)["new"]["value"]["population_count"] += 1
    slice_row = _overall(coherently_rehashed)
    slice_body = {
        key: value
        for key, value in slice_row.items()
        if key not in {"slice_id", "content_hash"}
    }
    slice_row["slice_id"] = "strategy-impact-slice-" + hashlib.sha256(
        json.dumps(
            slice_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()[:24]
    slice_without_hash = {
        key: value for key, value in slice_row.items() if key != "content_hash"
    }
    slice_row["content_hash"] = hashlib.sha256(
        json.dumps(
            slice_without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    cube_body = {
        key: value
        for key, value in coherently_rehashed.items()
        if key not in {"cube_id", "content_hash"}
    }
    coherently_rehashed["cube_id"] = "strategy-impact-cube-" + hashlib.sha256(
        json.dumps(
            cube_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()[:24]
    cube_without_hash = {
        key: value
        for key, value in coherently_rehashed.items()
        if key != "content_hash"
    }
    coherently_rehashed["content_hash"] = hashlib.sha256(
        json.dumps(
            cube_without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(StrategyError, match="population|label coverage"):
        validate_strategy_impact_cube(coherently_rehashed)


def test_impact_cube_fails_closed_when_slice_budget_is_exceeded() -> None:
    frame = _frame()
    frame["group"] = [f"group-{index}" for index in range(len(frame))]
    frames = {
        "development": pd.concat([frame] * 100, ignore_index=True),
    }
    frames["development"]["group"] = [
        f"group-{index}" for index in range(len(frames["development"]))
    ]
    sample_ref = _sample_design_v2_ref()
    sample_ref["analysis_universe_row_count"] = len(frames["development"])
    sample_ref["partition_counts"] = {
        "development": len(frames["development"])
    }
    sample_ref["population_partition_counts"] = {
        role: {"development": len(frames["development"])}
        for role in ("approval", "risk")
    }

    with pytest.raises(
        StrategyError,
        match="slice budget|high-cardinality",
    ):
        build_strategy_impact_cube(
            pool=_pool("approval"),
            partition_frames=frames,
            pool_artifact_ref=_pool_artifact_ref(),
            sample_design_v2_ref=sample_ref,
            dataset_binding=_dataset_binding(),
            legacy_development_ref=_legacy_ref(),
            target_col="bad",
            target_bad_value=1,
            month_col="month",
            group_col="group",
            segment_col="segment",
            current_strategy_spec=None,
            current_strategy_ref=None,
            economics_bindings=None,
        )


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("population_share", "share"),
        ("projection_extra_field", "metrics"),
        ("family_dimension", "dimensions"),
        ("economics_delta", "economics delta"),
        (
            "economics_current_binding",
            "current economics|baseline_profit",
        ),
        ("projection_value_leak", "overall_bad_rate"),
        ("transition_direction", "transition direction"),
        ("waterfall_source", "source_ref"),
    ],
)
def test_coherently_rehashed_semantic_tampering_is_rejected(
    tamper: str,
    error: str,
) -> None:
    cube = copy.deepcopy(_build("pricing"))
    if tamper == "family_dimension":
        row = next(
            item
            for item in cube["slices"]
            if item["family"] == "group"
            and item["dimensions"]["partition"]["value"] == "development"
        )
        row["dimensions"]["group"] = {"kind": "all", "value": None}
    else:
        row = _overall(cube, population_role="approval")
        if tamper == "population_share":
            row["population"]["value"]["share"] = 0.5
        elif tamper == "projection_extra_field":
            row["new"]["value"]["metrics"]["raw_rows"] = [1, 2, 3]
        elif tamper == "economics_delta":
            key = next(iter(row["economics"]["value"]["delta"]))
            row["economics"]["value"]["delta"][key] += 1.0
        elif tamper == "economics_current_binding":
            cube = copy.deepcopy(
                _build("pricing", include_current=False)
            )
            row = _overall(cube, population_role="approval")
            row["economics"]["value"]["current"] = copy.deepcopy(
                row["economics"]["value"]["new"]
            )
            row["economics"]["value"]["delta"] = {
                key: 0.0
                for key, value in row["economics"]["value"]["new"].items()
                if isinstance(value, int | float)
                and not isinstance(value, bool)
            }
        elif tamper == "projection_value_leak":
            row["new"]["value"]["metrics"]["overall_bad_rate"] = {
                "raw_customer_ids": ["customer-1"]
            }
        elif tamper == "transition_direction":
            row["transition"]["value"]["rows"][0]["direction"] = "forged"
        else:
            row["waterfall"]["value"]["entries"][0]["source_ref"][
                "raw_rows"
            ] = [1, 2, 3]
    _rehash_slice(row)
    _sort_slices(cube)
    _rehash_cube(cube)

    with pytest.raises(StrategyError, match=error):
        validate_strategy_impact_cube(cube)


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("current_reason", "current strategy"),
        ("family_reason", "slice families"),
        ("economics_availability", "economics"),
        ("lineage_dataset", "lineage dataset_id"),
        ("red_flags", "red_flags"),
    ],
)
def test_coherently_rehashed_cube_level_contract_drift_is_rejected(
    tamper: str,
    error: str,
) -> None:
    if tamper == "family_reason":
        cube = copy.deepcopy(_build("approval", group_col=None))
        cube["slice_families"]["group"]["reason"] = "forged"
        row = _slices(cube, family="group")[0]
        row["unavailable_reason"] = "forged"
        for field in (
            "population",
            "new",
            "current",
            "transition",
            "waterfall",
            "economics",
        ):
            row[field]["reason"] = "forged"
        _rehash_slice(row)
    elif tamper == "economics_availability":
        cube = copy.deepcopy(_build("pricing", economics_bindings=None))
        row = _overall(cube)
        row["economics"] = {
            "availability": "not_applicable",
            "reason": "forged",
            "value": None,
        }
        _rehash_slice(row)
    elif tamper == "lineage_dataset":
        cube = copy.deepcopy(_build("approval"))
        cube["source_bindings"]["development_lineage"]["sample_binding"][
            "dataset_id"
        ] = "forged-dataset"
    else:
        cube = copy.deepcopy(_build("approval", include_current=False))
        if tamper == "current_reason":
            cube["source_bindings"]["current_strategy"]["reason"] = "forged"
            for row in cube["slices"]:
                if row["availability"] != "present":
                    continue
                row["current"]["reason"] = "forged"
                row["transition"]["reason"] = "forged"
                _rehash_slice(row)
        else:
            cube["red_flags"].append(
                {
                    "code": "forged",
                    "level": "info",
                    "message": "forged",
                }
            )
    _rehash_cube(cube)

    with pytest.raises(StrategyError, match=error):
        validate_strategy_impact_cube(cube)


def test_rehashed_undeclared_partitions_and_duplicate_buckets_are_rejected() -> None:
    undeclared = copy.deepcopy(_build("approval"))
    extra = copy.deepcopy(_overall(undeclared))
    extra["dimensions"]["partition"]["value"] = "oot"
    _rehash_slice(extra)
    undeclared["slices"].append(extra)
    _sort_slices(undeclared)
    _rehash_cube(undeclared)
    with pytest.raises(StrategyError, match="undeclared partition"):
        validate_strategy_impact_cube(undeclared)

    duplicated = copy.deepcopy(_build("approval"))
    action_rows = _slices(duplicated, family="new_action")
    null_row, value_row = action_rows[:2]
    null_row["dimensions"]["new_action_bucket"] = copy.deepcopy(
        value_row["dimensions"]["new_action_bucket"]
    )
    _rehash_slice(null_row)
    _sort_slices(duplicated)
    _rehash_cube(duplicated)
    with pytest.raises(StrategyError, match="dimension buckets are duplicated"):
        validate_strategy_impact_cube(duplicated)

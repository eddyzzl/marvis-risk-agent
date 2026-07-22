from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from marvis.packs.strategy.candidate_fragment import build_verified_candidate_fragment
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    reorder_strategy_pool,
)
from marvis.packs.strategy.pool_impact import (
    STRATEGY_POOL_IMPACT_SCHEMA_VERSION,
    build_strategy_pool_impact_assessment,
    canonical_strategy_pool_impact_json,
    validate_strategy_pool_impact_assessment,
)
import marvis.packs.strategy.pool_impact as pool_impact_module


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _action(action_type: str) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": None if action_type == "approval" else action_type.upper(),
        "stop": True,
    }


def _identity() -> dict:
    return {
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 1,
        "semantic_mapping_hash": HASH_B,
        "sample_context_hash": HASH_C,
    }


def _sample_binding() -> dict:
    return {"task_id": "task-1", **_identity()}


def _sample_design_ref() -> dict:
    return {
        "artifact_id": "e" * 64,
        "artifact_content_hash": "f" * 64,
        "sample_design_id": "strategy-sample-design-" + "1" * 24,
        "sample_design_content_hash": "2" * 64,
        "partition": "development",
    }


def _condition(operator: str, value: int) -> dict:
    return {
        "op": "compare",
        "field": "score",
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
        evidence_identity=_identity(),
    )


def _pool(*, broad_first: bool = False, strategy_type: str = "approval") -> dict:
    if strategy_type in {"approval", "reject"}:
        default = _action("approval")
        actions = (_action("reject"), _action("review"))
    else:
        default = {
            "type": "limit",
            "value": 1000,
            "reason_code": None,
            "stop": True,
        }
        actions = (
            {"type": "limit", "value": 500, "reason_code": None, "stop": True},
            {"type": "limit", "value": 800, "reason_code": None, "stop": True},
        )
    conditions = (
        (_condition("<", 8), _condition("<", 5))
        if broad_first
        else (_condition("<", 5), _condition("<", 8))
    )
    result = None
    for index, (condition, action) in enumerate(zip(conditions, actions, strict=True), 1):
        result = add_verified_candidate_fragment(
            result,
            task_id="task-1",
            strategy_type=strategy_type,
            default_action=default,
            verified_candidate_fragment=_fragment(index, condition),
            action=action,
        )
    assert result is not None
    return result


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bad": [1, 1, 0, 1, 0, 1, None, 0, 0, 1],
            "month": [
                "202601",
                "202601",
                "202601",
                "202601",
                "202601",
                "202602",
                "202602",
                "202602",
                "202602",
                "202602",
            ],
            "loan": [100, 100, 100, 100, None, 200, 200, 200, 200, 200],
            "overdue": [10, 0, 0, 5, 0, 20, None, 0, 0, 10],
        }
    )


def _build(**overrides) -> dict:
    values = {
        "pool": _pool(),
        "frame": _frame(),
        "sample_binding": _sample_binding(),
        "sample_design_ref": _sample_design_ref(),
        "target_col": "bad",
        "target_bad_value": 1,
        "month_col": "month",
        "loan_amount_col": "loan",
        "overdue_amount_col": "overdue",
    }
    values.update(overrides)
    return build_strategy_pool_impact_assessment(**values)


def _rehash(document: dict) -> dict:
    body = {
        key: value
        for key, value in document.items()
        if key not in {"assessment_id", "content_hash"}
    }
    canonical_body = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    document["assessment_id"] = (
        "strategy-impact-assessment-"
        + hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()[:24]
    )
    without_hash = {key: value for key, value in document.items() if key != "content_hash"}
    canonical_document = json.dumps(
        without_hash,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    document["content_hash"] = hashlib.sha256(
        canonical_document.encode("utf-8")
    ).hexdigest()
    return document


def test_pool_impact_first_match_waterfall_and_population_conserve() -> None:
    impact = _build()

    assert impact["schema_version"] == STRATEGY_POOL_IMPACT_SCHEMA_VERSION
    assert impact["population"] == {
        "population_count": 10,
        "labelled_count": 9,
        "unlabelled_count": 1,
        "label_coverage": 0.9,
    }
    first, second = impact["waterfall"]
    assert first["standalone"]["population_count"] == 4
    assert first["incremental"]["population_count"] == 4
    assert first["shadowed"]["population_count"] == 0
    assert second["standalone"]["population_count"] == 7
    assert second["incremental"]["population_count"] == 3
    assert second["shadowed"]["population_count"] == 4
    assert impact["default_unmatched"]["effect"]["population_count"] == 3
    assert sum(row["incremental"]["population_count"] for row in impact["waterfall"]) + 3 == 10
    assert impact["overall"]["actions"]["metrics"]["approve_count"] == 3
    assert impact["overall"]["actions"]["metrics"]["reject_count"] == 4
    assert impact["overall"]["actions"]["metrics"]["review_count"] == 3
    assert impact["overall"]["actions"]["metrics"]["bad_capture_rate"] == pytest.approx(
        3 / 5
    )
    assert impact["bindings"]["sample_design_ref"] == _sample_design_ref()
    assert impact["bindings"]["target_bad_value"] == 1
    assert all(
        row["source_ref"]["sample_design_ref"] == _sample_design_ref()
        for row in impact["waterfall"]
    )


def test_pool_impact_normalizes_explicit_reverse_target_polarity() -> None:
    bad_one = _build()
    reversed_frame = _frame()
    reversed_frame["bad"] = reversed_frame["bad"].map(
        lambda value: None if pd.isna(value) else 1 - int(value)
    )
    bad_zero = _build(frame=reversed_frame, target_bad_value=0)

    assert bad_zero["population"] == bad_one["population"]
    assert bad_zero["overall"] == bad_one["overall"]
    assert bad_zero["waterfall"] == bad_one["waterfall"]
    assert bad_zero["monthly"] == bad_one["monthly"]
    assert bad_zero["bindings"]["target_bad_value"] == 0


def test_pool_impact_sample_design_reference_is_canonical_and_tamper_evident() -> None:
    impact = _build()
    forged = copy.deepcopy(impact)
    forged["bindings"]["sample_design_ref"]["partition"] = "validation"
    _rehash(forged)

    with pytest.raises(StrategyError, match="partition|sample design"):
        validate_strategy_pool_impact_assessment(forged)

    mismatched_source = copy.deepcopy(impact)
    mismatched_source["waterfall"][0]["source_ref"]["sample_design_ref"][
        "sample_design_content_hash"
    ] = HASH_A
    _rehash(mismatched_source)
    with pytest.raises(StrategyError, match="waterfall source sample design"):
        validate_strategy_pool_impact_assessment(mismatched_source)

    malformed = _sample_design_ref()
    malformed["sample_design_content_hash"] = "not-a-hash"
    with pytest.raises(StrategyError, match="sample design"):
        _build(sample_design_ref=malformed)


def test_pool_impact_reports_fully_shadowed_rule_and_reorder_changes_reach() -> None:
    pool = _pool(broad_first=True)
    broad = _build(pool=pool)
    assert broad["waterfall"][1]["incremental"]["population_count"] == 0
    assert any(flag["code"] == "rule_fully_shadowed" for flag in broad["red_flags"])

    reversed_pool = reorder_strategy_pool(
        pool, [pool["entries"][1]["entry_id"], pool["entries"][0]["entry_id"]]
    )
    reordered = _build(pool=reversed_pool)
    assert reordered["waterfall"][0]["incremental"]["population_count"] == 4
    assert reordered["waterfall"][1]["incremental"]["population_count"] == 3


def test_pool_impact_monthly_and_amounts_roll_to_overall() -> None:
    impact = _build()
    monthly = impact["monthly"]

    assert monthly["status"] == "available"
    assert [row["period"] for row in monthly["periods"]] == ["202601", "202602"]
    assert sum(row["effect"]["population_count"] for row in monthly["periods"]) == 10
    overall_amounts = impact["overall"]["effect"]["amounts"]
    assert overall_amounts["loan_amount"]["coverage_count"] == 9
    assert overall_amounts["loan_amount"]["sum"] == 1400.0
    assert overall_amounts["overdue_amount"]["coverage_count"] == 9
    assert overall_amounts["paired"]["coverage_count"] == 8
    assert overall_amounts["paired"]["loan_amount_sum"] == 1200.0
    assert overall_amounts["paired"]["overdue_amount_sum"] == 45.0
    assert overall_amounts["paired"]["overdue_rate"] == pytest.approx(45 / 1200)


def test_pool_impact_optional_columns_are_unavailable_not_zero() -> None:
    impact = _build(
        month_col=None,
        loan_amount_col=None,
        overdue_amount_col=None,
    )

    assert impact["monthly"] == {
        "status": "unavailable",
        "reason": "month_column_not_provided",
        "periods": [],
    }
    amounts = impact["overall"]["effect"]["amounts"]
    assert amounts["loan_amount"]["sum"] is None
    assert amounts["overdue_amount"]["coverage_count"] is None
    assert amounts["paired"]["overdue_rate"] is None


def test_pool_impact_rejects_invalid_month_and_amount_values() -> None:
    bad_month = _frame()
    bad_month.loc[0, "month"] = None
    with pytest.raises(StrategyError, match="unparseable"):
        _build(frame=bad_month)

    bad_amount = _frame()
    bad_amount.loc[0, "loan"] = -1
    with pytest.raises(StrategyError, match="non-negative"):
        _build(frame=bad_amount)


@pytest.mark.parametrize("invalid_amount", [True, 1 + 2j])
def test_pool_impact_rejects_boolean_or_complex_amounts(invalid_amount) -> None:
    frame = _frame()
    frame["loan"] = frame["loan"].astype("object")
    frame.loc[0, "loan"] = invalid_amount

    with pytest.raises(StrategyError, match="finite real numbers"):
        _build(frame=frame)


def _baseline_spec() -> dict:
    return StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(
            StrategyRuleSpec(
                rule_id="baseline-reject",
                priority=10,
                condition=_condition("<", 3),
                action=StrategyAction(type="reject", value="reject"),
            ),
        ),
    ).to_dict()


def test_pool_impact_baseline_deltas_are_explicit_and_monthly() -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    baseline = _baseline_spec()
    impact = _build(
        comparison_mode="vs_baseline",
        baseline_spec=baseline,
        baseline_binding={
            "strategy_id": "strategy-baseline",
            "strategy_type": "approval",
            "spec_hash": strategy_spec_hash(baseline),
        },
    )

    assert impact["baseline"]["status"] == "available"
    deltas = impact["baseline"]["overall"]["metric_deltas"]
    assert deltas["reject_count"] == 2
    assert deltas["approve_count"] == -5
    assert deltas["review_count"] == 3
    amount_deltas = impact["baseline"]["overall"]["amount_deltas"]
    assert amount_deltas["reject"]["loan_amount"]["sum"] == 200.0
    assert amount_deltas["reject"]["overdue_amount"]["sum"] == 5.0
    assert amount_deltas["approve"]["loan_amount"]["sum"] == -600.0
    assert impact["baseline"]["monthly"]["status"] == "available"
    assert len(impact["baseline"]["monthly"]["periods"]) == 2


def test_pool_impact_baseline_cannot_silently_degrade_or_change_type() -> None:
    with pytest.raises(StrategyError, match="requires baseline_spec"):
        _build(comparison_mode="vs_baseline")

    other = StrategySpec(
        strategy_type="reject",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(),
    ).to_dict()
    from marvis.packs.strategy.dsl import strategy_spec_hash

    with pytest.raises(StrategyError, match="type must match"):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=other,
            baseline_binding={
                "strategy_id": "strategy-other",
                "strategy_type": "reject",
                "spec_hash": strategy_spec_hash(other),
            },
        )

    with pytest.raises(StrategyError, match="must not provide"):
        _build(
            baseline_spec=_baseline_spec(),
            baseline_binding={
                "strategy_id": "strategy-baseline",
                "strategy_type": "approval",
                "spec_hash": HASH_A,
            },
        )

    baseline = _baseline_spec()
    with pytest.raises(StrategyError, match="spec_hash"):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=baseline,
            baseline_binding={
                "strategy_id": "strategy-baseline",
                "strategy_type": "approval",
                "spec_hash": HASH_A,
            },
        )


def test_pool_impact_rejects_unsupported_type_and_sample_drift() -> None:
    with pytest.raises(StrategyError, match="approval/reject only"):
        _build(pool=_pool(strategy_type="limit"))

    sample = _sample_binding()
    sample["semantic_mapping_hash"] = HASH_D
    with pytest.raises(StrategyError, match="sample binding"):
        _build(sample_binding=sample)


def test_pool_impact_is_deterministic_canonical_and_tamper_evident() -> None:
    first = _build()
    second = _build()

    assert first == second
    raw = canonical_strategy_pool_impact_json(first)
    assert json.loads(raw) == first
    assert validate_strategy_pool_impact_assessment(first) == first

    forged = copy.deepcopy(first)
    forged["population"]["population_count"] += 1
    with pytest.raises(StrategyError, match="content_hash"):
        validate_strategy_pool_impact_assessment(forged)

    malformed = copy.deepcopy(first)
    malformed["population"]["population_count"] = "10"
    _rehash(malformed)
    with pytest.raises(StrategyError, match="population_count"):
        validate_strategy_pool_impact_assessment(malformed)


def test_pool_impact_rejects_duplicate_or_conflicting_column_bindings() -> None:
    duplicate = _frame()
    duplicate.columns = ["score", "bad", "month", "loan", "loan"]
    with pytest.raises(StrategyError, match="duplicate columns"):
        _build(frame=duplicate)

    with pytest.raises(StrategyError, match="bindings must be distinct"):
        _build(month_col="bad")


def test_pool_impact_reject_strategy_and_zero_denominators_are_explicit() -> None:
    impact = _build(pool=_pool(strategy_type="reject"))
    assert impact["identity"]["strategy_type"] == "reject"

    all_bad = _frame()
    all_bad["bad"] = 1
    no_good = _build(frame=all_bad)
    assert no_good["overall"]["actions"]["metrics"]["good_reject_rate"] is None


def test_pool_impact_monthly_work_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(pool_impact_module, "MAX_IMPACT_MONTHLY_WORK", 39)
    with pytest.raises(StrategyError, match="monthly work budget"):
        _build()


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    [
        ("MAX_IMPACT_RULES", 2, "comparison exceeds the row or rule budget"),
        ("MAX_IMPACT_WORK", 45, "expression evaluation work budget"),
        ("MAX_IMPACT_MONTHLY_WORK", 50, "monthly work budget"),
    ],
)
def test_pool_impact_baseline_is_included_in_resource_budgets(
    monkeypatch,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    baseline = _baseline_spec()
    monkeypatch.setattr(pool_impact_module, limit_name, limit)

    with pytest.raises(StrategyError, match=message):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=baseline,
            baseline_binding={
                "strategy_id": "strategy-baseline",
                "strategy_type": "approval",
                "spec_hash": strategy_spec_hash(baseline),
            },
        )


def test_pool_impact_compound_expression_cost_is_bounded(monkeypatch) -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    compound = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(
            StrategyRuleSpec(
                rule_id="baseline-compound",
                priority=10,
                condition={
                    "op": "or",
                    "args": [_condition(">", 10_000) for _ in range(30)],
                },
                action=StrategyAction(type="reject", value="reject"),
            ),
        ),
    ).to_dict()
    monkeypatch.setattr(pool_impact_module, "MAX_IMPACT_WORK", 100)

    with pytest.raises(StrategyError, match="expression evaluation work budget"):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=compound,
            baseline_binding={
                "strategy_id": "strategy-compound",
                "strategy_type": "approval",
                "spec_hash": strategy_spec_hash(compound),
            },
        )


def test_pool_impact_expression_nodes_and_depth_are_bounded(monkeypatch) -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    too_many_nodes = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(
            StrategyRuleSpec(
                rule_id="baseline-wide",
                priority=10,
                condition={
                    "op": "or",
                    "args": [_condition(">", 10_000) for _ in range(3)],
                },
                action=StrategyAction(type="reject", value="reject"),
            ),
        ),
    ).to_dict()
    monkeypatch.setattr(pool_impact_module, "MAX_IMPACT_EXPRESSION_NODES", 5)
    with pytest.raises(StrategyError, match="comparison exceeds the expression node"):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=too_many_nodes,
            baseline_binding={
                "strategy_id": "strategy-wide",
                "strategy_type": "approval",
                "spec_hash": strategy_spec_hash(too_many_nodes),
            },
        )

    too_deep = StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval", value="approve"),
        rules=(
            StrategyRuleSpec(
                rule_id="baseline-deep",
                priority=10,
                condition={
                    "op": "not",
                    "arg": {"op": "not", "arg": _condition(">", 10_000)},
                },
                action=StrategyAction(type="reject", value="reject"),
            ),
        ),
    ).to_dict()
    monkeypatch.setattr(pool_impact_module, "MAX_IMPACT_EXPRESSION_NODES", 10_000)
    monkeypatch.setattr(pool_impact_module, "MAX_IMPACT_EXPRESSION_DEPTH", 2)
    with pytest.raises(StrategyError, match="expression depth budget"):
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=too_deep,
            baseline_binding={
                "strategy_id": "strategy-deep",
                "strategy_type": "approval",
                "spec_hash": strategy_spec_hash(too_deep),
            },
        )


def test_pool_impact_rejects_swapped_monthly_actions_after_rehash() -> None:
    forged = copy.deepcopy(
        _build(loan_amount_col=None, overdue_amount_col=None)
    )
    january, february = forged["monthly"]["periods"]
    january["actions"], february["actions"] = (
        february["actions"],
        january["actions"],
    )
    _rehash(forged)

    with pytest.raises(StrategyError, match="action summary does not match Pool effects"):
        validate_strategy_pool_impact_assessment(forged)


def test_pool_impact_red_flags_are_canonical_derived_evidence() -> None:
    forged = copy.deepcopy(_build())
    assert forged["red_flags"]
    forged["red_flags"] = []
    _rehash(forged)

    with pytest.raises(StrategyError, match="red_flags do not match"):
        validate_strategy_pool_impact_assessment(forged)


def test_pool_impact_rejects_baseline_amount_delta_tampering_after_rehash() -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    baseline = _baseline_spec()
    forged = copy.deepcopy(
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=baseline,
            baseline_binding={
                "strategy_id": "strategy-baseline",
                "strategy_type": "approval",
                "spec_hash": strategy_spec_hash(baseline),
            },
        )
    )
    forged["baseline"]["overall"]["amount_deltas"]["reject"]["loan_amount"][
        "sum"
    ] += 1.0
    _rehash(forged)

    with pytest.raises(StrategyError, match="amount deltas"):
        validate_strategy_pool_impact_assessment(forged)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["waterfall"][0].__setitem__(
                "remaining_after", copy.deepcopy(payload["overall"]["effect"])
            ),
            "remaining partition",
        ),
        (
            lambda payload: payload["bindings"].__setitem__("month_col", None),
            "month binding",
        ),
        (
            lambda payload: payload["monthly"]["periods"][0].__setitem__(
                "period", "garbage"
            ),
            "canonical YYYYMM",
        ),
        (
            lambda payload: payload["bindings"].__setitem__(
                "loan_amount_col", "fake_loan"
            ),
            "column binding",
        ),
        (
            lambda payload: payload["monthly"]["periods"][1]["rule_incremental"][
                0
            ]["effect"]["amounts"]["loan_amount"].__setitem__("sum", 1.0),
            "sum requires covered rows",
        ),
    ],
)
def test_pool_impact_rehashed_relational_tampering_is_rejected(
    mutation,
    message: str,
) -> None:
    forged = copy.deepcopy(_build())
    mutation(forged)
    _rehash(forged)

    with pytest.raises(StrategyError, match=message):
        validate_strategy_pool_impact_assessment(forged)


def test_pool_impact_rejects_monthly_rule_overlap_after_rehash() -> None:
    forged = copy.deepcopy(_build())
    january = forged["monthly"]["periods"][0]["rule_incremental"][1]["effect"]
    february = forged["monthly"]["periods"][1]["rule_incremental"][1]["effect"]
    january.update(
        {
            "population_count": 2,
            "population_share": 0.2,
            "labelled_count": 1,
            "label_coverage": 0.5,
        }
    )
    january["amounts"]["loan_amount"].update(
        {"coverage_count": 1, "coverage_rate": 0.5, "sum": 200.0}
    )
    january["amounts"]["overdue_amount"]["coverage_rate"] = 0.5
    february.update(
        {
            "population_count": 1,
            "population_share": 0.1,
            "label_coverage": 1.0,
        }
    )
    february["amounts"]["loan_amount"].update(
        {"coverage_count": 1, "coverage_rate": 1.0, "sum": 200.0}
    )
    february["amounts"]["overdue_amount"]["coverage_rate"] = 1.0
    february["amounts"]["paired"]["coverage_rate"] = 1.0
    _rehash(forged)

    with pytest.raises(StrategyError, match="rule increments population_count exceeds"):
        validate_strategy_pool_impact_assessment(forged)


def test_pool_impact_rejects_first_rule_shadowed_rows_after_rehash() -> None:
    forged = copy.deepcopy(
        _build(loan_amount_col=None, overdue_amount_col=None)
    )
    first = forged["waterfall"][0]
    first["shadowed"].update(
        {
            "population_count": 1,
            "population_share": 0.1,
            "labelled_count": 1,
            "label_coverage": 1.0,
            "bad_count": 0,
            "bad_rate": 0.0,
        }
    )
    first["standalone"].update(
        {
            "population_count": 5,
            "population_share": 0.5,
            "labelled_count": 5,
            "label_coverage": 1.0,
            "bad_count": 3,
            "bad_rate": 0.6,
        }
    )
    _rehash(forged)

    with pytest.raises(StrategyError, match="shadowed population_count exceeds"):
        validate_strategy_pool_impact_assessment(forged)


def test_pool_impact_rejects_baseline_target_total_tampering_after_rehash() -> None:
    from marvis.packs.strategy.dsl import strategy_spec_hash

    spec = _baseline_spec()
    forged = copy.deepcopy(
        _build(
            comparison_mode="vs_baseline",
            baseline_spec=spec,
            baseline_binding={
                "strategy_id": "strategy-baseline",
                "strategy_type": "approval",
                "spec_hash": strategy_spec_hash(spec),
            },
        )
    )
    baseline = forged["baseline"]["overall"]["baseline"]
    baseline["breakdown"][0]["bad_count"] = 2
    baseline["breakdown"][0]["bad_rate"] = 2 / 7
    metrics = baseline["metrics"]
    metrics.update(
        {
            "approve_bad_count": 2,
            "approve_bad_rate": 2 / 7,
            "overall_bad_count": 4,
            "overall_bad_rate": 4 / 9,
            "bad_capture_rate": 0.5,
        }
    )
    current = forged["baseline"]["overall"]["current"]["metrics"]
    deltas = forged["baseline"]["overall"]["metric_deltas"]
    for field in (
        "approve_bad_count",
        "approve_bad_rate",
        "overall_bad_count",
        "overall_bad_rate",
        "bad_capture_rate",
    ):
        deltas[field] = current[field] - metrics[field]
    _rehash(forged)

    with pytest.raises(StrategyError, match="bad_count differs on the same sample"):
        validate_strategy_pool_impact_assessment(forged)

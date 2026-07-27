"""Deterministic non-approval candidate design contract tests."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from marvis.packs.strategy.candidate_design import (
    CANDIDATE_DESIGN_SCHEMA_VERSION,
    CANDIDATE_POLICY_VERSION,
    CandidateDesignError,
    design_strategy_candidate,
    normalize_candidate_design,
)
from marvis.packs.strategy.dsl import parse_strategy_spec, strategy_spec_hash
from marvis.packs.strategy.strategy import apply_strategy, build_strategy_from_spec


def _limit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [100, 200, 300, 400, 500, 600, 700, 800],
            "bad": [0, 0, 0, 1, 0, 1, 1, 1],
            "pd_12m": [0.01, 0.02, 0.03, 0.04, 0.10, 0.12, 0.14, 0.16],
            "utilization": [0.5] * 8,
        }
    )


def _limit_design() -> dict:
    return {
        "method": "score_band_limit",
        "score_col": "score",
        "n_bands": 2,
        "limit_grid": [1000, 2000, 4000],
        "max_expected_loss_per_account": 100,
    }


def _limit_economics(*, utilization: float | None = None) -> dict:
    return {
        "pd_col": "pd_12m",
        "lgd_value": 0.5,
        **(
            {"utilization_col": "utilization"}
            if utilization is None
            else {"utilization_value": utilization}
        ),
    }


def _pricing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [100, 200, 300, 400, 500, 600],
            "bad": [0, 0, 0, 1, 1, 1],
            "ead": [100, 200, 300, 1000, 2000, 3000],
            "pd_12m": [0.01, 0.02, 0.03, 0.20, 0.25, 0.30],
        }
    )


def _pricing_design() -> dict:
    return {
        "method": "score_band_pricing",
        "score_col": "score",
        "n_bands": 2,
        "rate_grid": [0.10, 0.20, 0.40],
        "min_roa": 0.0,
    }


def _pricing_economics() -> dict:
    return {
        "ead_col": "ead",
        "pd_col": "pd_12m",
        "lgd_value": 0.5,
        "funding_rate_value": 0.02,
        "term_months_value": 12,
        "operating_cost_per_loan_value": 0,
    }


def test_limit_candidate_is_deterministic_canonical_and_covers_missing_values() -> None:
    frame = _limit_frame()
    frame.loc[len(frame)] = [None, 1, 0.5, 0.5]
    kwargs = {
        "strategy_type": "limit",
        "target_col": "bad",
        "candidate_design": _limit_design(),
        "economics_inputs": _limit_economics(),
        "dataset_id": "dataset-1",
        "source_dataset_content_hash": "a" * 64,
    }

    first = design_strategy_candidate(frame, **kwargs)
    second = design_strategy_candidate(frame.copy(), **copy.deepcopy(kwargs))

    assert first.to_dict() == second.to_dict()
    parsed = parse_strategy_spec(first.strategy_spec)
    assert first.strategy_effect_hash == strategy_spec_hash(parsed)
    assert parsed.metadata["lineage"] == {
        "source": "deterministic_candidate_design",
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "candidate_design_input_hash": first.design_evidence[
            "candidate_design_input_hash"
        ],
        "method": "score_band_limit",
        "dataset_id": "dataset-1",
        "source_dataset_content_hash": "a" * 64,
    }
    strategy = build_strategy_from_spec(parsed)
    assigned = apply_strategy(frame, strategy)
    assert assigned.notna().all()
    assert assigned.iloc[-1] == 0.0
    assert first.design_evidence["missing_policy"] == "zero_limit"
    assert first.design_evidence["missing_count"] == 1
    assert first.design_evidence["objective"] == (
        "maximize_limit_under_expected_loss_budget"
    )
    assert any("不是额度与定价联合利润优化" in item for item in first.design_evidence["assumptions"])


def test_limit_selection_uses_explicit_utilization_in_expected_loss_budget() -> None:
    frame = _limit_frame()

    half_used = design_strategy_candidate(
        frame,
        strategy_type="limit",
        target_col="bad",
        candidate_design=_limit_design(),
        economics_inputs=_limit_economics(utilization=0.5),
    )
    fully_used = design_strategy_candidate(
        frame,
        strategy_type="limit",
        target_col="bad",
        candidate_design=_limit_design(),
        economics_inputs=_limit_economics(utilization=1.0),
    )

    half_high_risk = half_used.design_evidence["bands"][1]
    full_high_risk = fully_used.design_evidence["bands"][1]
    assert half_high_risk["selected_action"] == {"type": "limit", "value": 2000.0}
    assert full_high_risk["selected_action"] == {"type": "limit", "value": 1000.0}
    assert (
        full_high_risk["candidate_scores"][0]["expected_ead"]
        == 2 * half_high_risk["candidate_scores"][0]["expected_ead"]
    )


def test_limit_candidate_fails_if_any_populated_band_has_no_feasible_action() -> None:
    design = {**_limit_design(), "max_expected_loss_per_account": 0}

    with pytest.raises(CandidateDesignError) as exc_info:
        design_strategy_candidate(
            _limit_frame(),
            strategy_type="limit",
            target_col="bad",
            candidate_design=design,
            economics_inputs=_limit_economics(),
        )

    assert exc_info.value.code == "candidate_band_infeasible"


def test_pricing_candidate_scores_real_row_level_ead_and_pd() -> None:
    result = design_strategy_candidate(
        _pricing_frame(),
        strategy_type="pricing",
        target_col="bad",
        candidate_design=_pricing_design(),
        economics_inputs=_pricing_economics(),
    )

    low_band, high_band = result.design_evidence["bands"]
    assert low_band["candidate_scores"][0]["total_ead"] == 600.0
    assert high_band["candidate_scores"][0]["total_ead"] == 6000.0
    assert low_band["risk_estimate"] == pytest.approx(0.02)
    assert high_band["risk_estimate"] == pytest.approx(0.25)
    assert (
        high_band["candidate_scores"][0]["expected_loss"]
        > low_band["candidate_scores"][0]["expected_loss"]
    )
    assert result.economics_inputs["ead_col"] == "ead"
    assert result.economics_inputs["pd_col"] == "pd_12m"
    assert result.design_evidence["objective"] == "maximize_static_expected_profit"
    assert any(
        flag["code"] == "price_elasticity_not_modeled"
        for flag in result.design_evidence["red_flags"]
    )
    assert "最高风险分箱" in result.design_evidence["default_action_rationale"]


@pytest.mark.parametrize(
    "economics_update",
    [
        {"ead_value": 1000},
        {"pd_value": 0.1},
    ],
)
def test_pricing_candidate_rejects_fixed_ead_or_pd(economics_update: dict) -> None:
    economics = _pricing_economics()
    if "ead_value" in economics_update:
        economics.pop("ead_col")
    else:
        economics.pop("pd_col")
    economics.update(economics_update)

    with pytest.raises(CandidateDesignError) as exc_info:
        design_strategy_candidate(
            _pricing_frame(),
            strategy_type="pricing",
            target_col="bad",
            candidate_design=_pricing_design(),
            economics_inputs=economics,
        )

    assert exc_info.value.code == "candidate_requires_observed_economics"
    assert exc_info.value.fields == ("ead_col", "pd_col")


def test_segmentation_uses_fixed_binning_stable_risk_labels_and_missing_segment() -> None:
    frame = pd.DataFrame(
        {
            "income": [10, 10, 20, 20, 30, 30, 40, 40, None],
            "bad": [1, 1, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    design = {
        "method": "single_variable_segmentation",
        "feature_col": "income",
        "n_bands": 4,
    }

    result = design_strategy_candidate(
        frame,
        strategy_type="segmentation",
        target_col="bad",
        candidate_design=design,
    )
    repeated = design_strategy_candidate(
        frame.copy(),
        strategy_type="segmentation",
        target_col="bad",
        candidate_design=design,
    )

    assert result.to_dict() == repeated.to_dict()
    bands = result.design_evidence["bands"]
    observed_labels = [band["selected_action"]["value"] for band in bands]
    assert sorted(observed_labels) == [f"R{index}" for index in range(1, len(bands) + 1)]
    assigned = apply_strategy(frame, build_strategy_from_spec(result.strategy_spec))
    assert assigned.notna().all()
    assert assigned.iloc[-1] == "UNASSIGNED"
    assert result.design_evidence["missing_policy"] == "separate_segment"


@pytest.mark.parametrize(
    ("strategy_type", "candidate_design", "message"),
    [
        (
            "limit",
            {**_limit_design(), "recommended": {"B01": 4000}},
            "不得提交推荐值",
        ),
        (
            "pricing",
            {**_pricing_design(), "strategy_spec": {}},
            "不得提交推荐值",
        ),
        (
            "segmentation",
            {
                "method": "score_band_pricing",
                "feature_col": "income",
            },
            "method",
        ),
        (
            "limit",
            {**_limit_design(), "score_col": "ghost"},
            "不存在",
        ),
    ],
)
def test_candidate_input_contract_rejects_results_mismatched_methods_and_fake_columns(
    strategy_type: str,
    candidate_design: dict,
    message: str,
) -> None:
    with pytest.raises(CandidateDesignError, match=message):
        normalize_candidate_design(
            strategy_type,
            candidate_design,
            allowed_columns=["score", "income"],
        )


def test_collection_candidate_fails_closed_instead_of_becoming_segmentation() -> None:
    with pytest.raises(CandidateDesignError) as exc_info:
        normalize_candidate_design(
            "collection",
            {
                "method": "single_variable_segmentation",
                "feature_col": "delinquency_bucket",
            },
        )

    assert exc_info.value.code == "collection_strategy_unsupported"
    assert "不能映射为分群或拒绝策略" in str(exc_info.value)


def test_result_is_immutable_and_to_dict_is_defensive() -> None:
    result = design_strategy_candidate(
        _limit_frame(),
        strategy_type="limit",
        target_col="bad",
        candidate_design=_limit_design(),
        economics_inputs=_limit_economics(),
    )

    with pytest.raises(TypeError):
        result.strategy_spec["rules"][0]["priority"] = 999
    with pytest.raises(TypeError):
        result.design_evidence["bands"][0]["count"] = 0
    detached = result.to_dict()
    detached["strategy_spec"]["rules"][0]["priority"] = 999
    detached["design_evidence"]["bands"][0]["count"] = 0
    assert result.strategy_spec["rules"][0]["priority"] == 0
    assert result.design_evidence["bands"][0]["count"] > 0
    assert result.strategy_spec["schema_version"] == "strategy.dsl.v1"
    assert CANDIDATE_DESIGN_SCHEMA_VERSION in {
        _limit_design().get("schema_version", CANDIDATE_DESIGN_SCHEMA_VERSION)
    }

import json

import numpy as np
import pandas as pd
import pytest

from marvis.feature.derive import (
    CROSS_SYS,
    CrossRecommendation,
    aggregate_feature,
    cross_arithmetic,
    derive_batch,
    derive_date_features,
    evaluate_crosses,
    recommend_feature_crosses,
)
from marvis.feature.errors import FeatureError


def test_cross_arithmetic_handles_division_by_zero():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 4.0]})

    derived, cols = cross_arithmetic(frame, "a", "b", ["add", "div", "ratio"])

    assert cols == ["a_add_b", "a_div_b", "a_ratio_b"]
    assert derived["a_add_b"].tolist() == [1.0, 6.0]
    assert np.isnan(derived["a_div_b"].iloc[0])
    assert derived["a_div_b"].iloc[1] == 0.5
    assert np.isnan(derived["a_ratio_b"].iloc[0])


def test_aggregate_feature_left_join_does_not_expand_rows():
    frame = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "amount": [10.0, 30.0, 5.0],
    })

    # min_group_size=1 -> every group (however small) keeps its own statistic,
    # same as the platform's pre-PREP-10 behavior.
    derived, cols = aggregate_feature(frame, "user_id", "amount", ["mean", "count"], min_group_size=1)

    assert cols == ["amount_by_user_id_mean", "amount_by_user_id_count"]
    assert len(derived) == len(frame)
    assert derived["amount_by_user_id_mean"].tolist() == [20.0, 20.0, 5.0]
    assert derived["amount_by_user_id_count"].tolist() == [2, 2, 1]


def test_aggregate_feature_small_groups_fall_back_to_global_stat(tmp_path=None):
    """PREP-10: a group below min_group_size gets the *global* fit-frame statistic
    instead of its own noisy small-sample value."""
    frame = pd.DataFrame({
        "user_id": ["u1"] * 2 + ["u2"] * 1,
        "amount": [10.0, 30.0, 5.0],
    })

    derived, _cols = aggregate_feature(frame, "user_id", "amount", ["mean"], min_group_size=30)

    # Both groups have fewer than 30 rows -> both fall back to the global mean.
    assert derived["amount_by_user_id_mean"].tolist() == [15.0, 15.0, 15.0]


def test_aggregate_feature_rejects_target_as_value_col():
    frame = pd.DataFrame({"user_id": ["u1", "u2"], "y": [0, 1]})

    with pytest.raises(FeatureError, match="target column"):
        aggregate_feature(frame, "user_id", "y", ["mean"], target_col="y", min_group_size=1)


def test_aggregate_feature_fits_only_on_non_holdout_rows_and_falls_back_for_unseen_groups():
    """PREP-10: fit_mask restricts which rows compute the group statistic (train-only
    discipline); a group present only outside fit_mask falls back to the global
    fit-frame statistic instead of leaking its own out-of-fit value."""
    frame = pd.DataFrame({
        "city": ["A", "A", "A", "B", "B", "B", "C"],
        "income": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 999.0],
    })
    fit_mask = np.array([True, True, True, True, True, True, False])  # "C" row excluded from fit

    derived, _cols = aggregate_feature(
        frame, "city", "income", ["mean"], fit_mask=fit_mask, min_group_size=1
    )

    assert derived.loc[derived["city"] == "A", "income_by_city_mean"].tolist() == [20.0] * 3
    assert derived.loc[derived["city"] == "B", "income_by_city_mean"].tolist() == [200.0] * 3
    # "C" never appears in the fit frame -> falls back to the global fit-frame mean.
    global_mean = frame.loc[fit_mask, "income"].mean()
    assert derived.loc[derived["city"] == "C", "income_by_city_mean"].tolist() == [global_mean]


def test_derive_batch_applies_cross_agg_and_ratio_recipes():
    frame = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 4.0, 6.0],
    })
    recipe = [
        {"kind": "cross", "a": "a", "b": "b", "ops": ["mul"]},
        {"kind": "agg", "group": "user_id", "value": "a", "aggs": ["sum"], "allow_full_fit": True, "min_group_size": 1},
        {"kind": "ratio", "num": "a", "den": "b"},
    ]

    derived, cols = derive_batch(frame, recipe)

    assert cols == ["a_mul_b", "a_by_user_id_sum", "a_ratio_b"]
    assert derived["a_mul_b"].tolist() == [2.0, 8.0, 18.0]
    assert derived["a_by_user_id_sum"].tolist() == [3.0, 3.0, 3.0]
    assert derived["a_ratio_b"].tolist() == [0.5, 0.5, 0.5]


def test_recommend_feature_crosses_filters_invalid_llm_output_and_ignores_metrics():
    class FakeLLM:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return json.dumps({
                "recommendations": [
                    {
                        "col_a": "used_limit",
                        "col_b": "credit_limit",
                        "ops": ["ratio", "bogus"],
                        "rationale": "utilization has clear credit meaning",
                        "confidence": "high",
                        "iv": 999,
                        "ks": 999,
                    },
                    {"col_a": "made_up", "col_b": "credit_limit", "ops": ["mul"]},
                ]
            })

    llm = FakeLLM()
    recs = recommend_feature_crosses(
        {
            "used_limit": {"meaning": "used credit"},
            "credit_limit": {"meaning": "approved credit"},
        },
        {"used_limit": {"iv": 0.1, "ks": 0.2}},
        llm_factory=lambda: llm,
        max_candidates=10,
    )

    assert CROSS_SYS in llm.calls[0]["system_prompt"]
    assert "Do not calculate" in llm.calls[0]["user_prompt"]
    assert recs == [
        CrossRecommendation(
            col_a="used_limit",
            col_b="credit_limit",
            ops=("ratio",),
            rationale="utilization has clear credit meaning",
            confidence="high",
        )
    ]
    assert not hasattr(recs[0], "iv")


def test_recommend_feature_crosses_returns_empty_on_llm_failure():
    class BrokenLLM:
        def complete(self, **_kwargs):
            raise RuntimeError("unavailable")

    assert recommend_feature_crosses({}, {}, llm_factory=BrokenLLM) == []


def test_evaluate_crosses_only_scores_selected_pairs_with_platform_metrics():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 2.0, 2.0],
        "c": [4.0, 3.0, 2.0, 1.0],
    })
    target = np.array([0, 0, 1, 1])
    recs = [
        CrossRecommendation("a", "b", ("add",), "sum proxy", "medium"),
        CrossRecommendation("c", "b", ("mul",), "unused", "low"),
    ]

    derived, results = evaluate_crosses(frame, target, recs, selected_pairs=[("a", "b")])

    assert "a_add_b" in derived.columns
    assert "c_mul_b" not in derived.columns
    assert results[0]["new_col"] == "a_add_b"
    assert results[0]["from"] == ("a", "b")
    assert results[0]["op"] == "add"
    assert 0.0 <= results[0]["ks"] <= 1.0
    assert isinstance(results[0]["iv"], float)


def test_derive_rejects_invalid_recipes_and_conflicts():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "a_add_b": [3.0]})

    with pytest.raises(FeatureError, match="already exist"):
        cross_arithmetic(frame, "a", "b", ["add"])

    with pytest.raises(FeatureError, match="unsupported"):
        derive_batch(frame, [{"kind": "unknown"}])

def test_derive_date_features_datediff_month_and_tenure():
    frame = pd.DataFrame({
        "apply_date": ["2024-01-15", "2024-03-01", None],
        "open_date": ["2023-01-15", "2023-01-01", "2023-06-01"],
    })
    recipe = [
        {"kind": "datediff", "col": "apply_date", "anchor": "open_date", "unit": "days"},
        {"kind": "month", "col": "apply_date"},
        {"kind": "tenure_months", "col": "apply_date", "anchor": "open_date"},
    ]

    derived, cols = derive_date_features(frame, recipe)

    assert cols == [
        "apply_date__days_since_open_date",
        "apply_date__month",
        "apply_date__months_on_book",
    ]
    assert derived["apply_date__days_since_open_date"].tolist()[:2] == [365.0, 425.0]
    assert pd.isna(derived["apply_date__days_since_open_date"].iloc[2])
    assert derived["apply_date__month"].tolist()[:2] == [1.0, 3.0]
    assert derived["apply_date__months_on_book"].tolist()[:2] == [11.0, 13.0]


def test_derive_date_features_supports_literal_anchor_date():
    frame = pd.DataFrame({"apply_date": ["2024-01-01", "2024-02-01"]})
    recipe = [{"kind": "datediff", "col": "apply_date", "anchor": "2024-01-01", "unit": "days"}]

    derived, cols = derive_date_features(frame, recipe)

    assert cols == ["apply_date__days_since_ref"]
    assert derived["apply_date__days_since_ref"].tolist() == [0.0, 31.0]


def test_derive_date_features_rejects_invalid_kind_conflicts_and_bad_anchor():
    frame = pd.DataFrame({"apply_date": ["2024-01-01"], "apply_date__month": [1]})

    with pytest.raises(FeatureError, match="unsupported date derive kind"):
        derive_date_features(frame, [{"kind": "unknown", "col": "apply_date"}])

    with pytest.raises(FeatureError, match="already exist"):
        derive_date_features(frame, [{"kind": "month", "col": "apply_date"}])

    with pytest.raises(FeatureError, match="not a column or a parseable date"):
        derive_date_features(
            frame, [{"kind": "datediff", "col": "apply_date", "anchor": "not-a-date"}]
        )

    with pytest.raises(FeatureError, match="anchor"):
        derive_date_features(frame, [{"kind": "datediff", "col": "apply_date"}])

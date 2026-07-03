import numpy as np
import pandas as pd
import pytest

from marvis.feature.contracts import WOEResult
from marvis.feature.encode import (
    apply_categorical_woe,
    categorical_woe_encode,
    label_encode,
    onehot_encode,
    woe_encode,
)
from marvis.feature.errors import FeatureError


def test_onehot_encode_returns_mapping_and_rejects_high_cardinality():
    frame = pd.DataFrame({
        "id": [1, 2, 3],
        "segment": ["prime", "subprime", np.nan],
    })

    encoded, mapping = onehot_encode(frame, ["segment"], max_categories=2)

    assert mapping == {"segment": ["prime", "subprime"]}
    assert list(encoded.columns) == ["id", "segment_prime", "segment_subprime"]
    assert encoded["segment_prime"].tolist() == [1, 0, 0]
    assert encoded["segment_subprime"].tolist() == [0, 1, 0]

    with pytest.raises(FeatureError, match="too many categories"):
        onehot_encode(frame, ["segment"], max_categories=1)


def test_label_encode_maps_missing_to_negative_one():
    series = pd.Series(["a", "b", np.nan, "a"], name="grade")

    encoded, mapping = label_encode(series)

    assert mapping == {"a": 0, "b": 1}
    assert encoded.name == "grade"
    assert encoded.tolist() == [0, 1, -1, 0]


def test_woe_encode_uses_training_edges_and_na_woe():
    frame = pd.DataFrame({"score": [5.0, 15.0, np.nan]})
    woe = WOEResult(
        feature="score",
        edges=(-np.inf, 10.0, np.inf),
        woe_by_bin=(-0.25, 0.75),
        na_woe=0.1,
    )

    encoded = woe_encode(frame, "score", woe)

    assert encoded.name == "score_woe"
    assert encoded.tolist() == [-0.25, 0.75, 0.1]


def test_woe_encode_rejects_invalid_mapping_shape():
    frame = pd.DataFrame({"score": [5.0, 15.0]})
    woe = WOEResult(
        feature="score",
        edges=(-np.inf, 10.0, np.inf),
        woe_by_bin=(0.2,),
        na_woe=None,
    )

    with pytest.raises(FeatureError, match="woe_by_bin length"):
        woe_encode(frame, "score", woe)


def test_categorical_woe_encode_matches_hand_computed_smoothed_woe():
    # Group A: 10 rows, 5 bad. Group B: 10 rows, 2 bad. Both above min_count=5 so
    # neither is pooled into __rare__.
    series = pd.Series(["A"] * 10 + ["B"] * 10)
    target = np.array([1] * 5 + [0] * 5 + [1] * 2 + [0] * 8, dtype=float)

    woe = categorical_woe_encode(series, target, feature="chan", min_count=5, smoothing=0.5)

    assert woe.rare_categories == ()
    by_category = {item.category: item for item in woe.categories}
    total_bad, total_good = 7, 13
    n_groups = 2
    # WOE_i = ln(good_dist_i / bad_dist_i), Laplace-smoothed — same convention as
    # marvis.feature.iv.compute_woe_iv.
    expected_a = np.log(((5 + 0.5) / (total_good + 0.5 * n_groups)) / ((5 + 0.5) / (total_bad + 0.5 * n_groups)))
    expected_b = np.log(((8 + 0.5) / (total_good + 0.5 * n_groups)) / ((2 + 0.5) / (total_bad + 0.5 * n_groups)))
    assert by_category["A"].woe == pytest.approx(expected_a)
    assert by_category["B"].woe == pytest.approx(expected_b)
    assert by_category["A"].count == 10
    assert by_category["A"].bad_count == 5
    assert woe.total_iv > 0.0


def test_categorical_woe_encode_pools_rare_categories_and_shares_one_woe():
    # A/B are frequent (10 each); C/D/E/F/G each appear once (below min_count=5) and
    # should be pooled into a single __rare__ bucket with one shared WOE.
    series = pd.Series(["A"] * 10 + ["B"] * 10 + ["C", "D", "E", "F", "G"])
    target = np.array([1] * 5 + [0] * 5 + [1] * 2 + [0] * 8 + [1, 0, 1, 0, 1], dtype=float)

    woe = categorical_woe_encode(series, target, feature="chan", min_count=5)

    assert woe.rare_categories == ("C", "D", "E", "F", "G")
    by_category = {item.category: item for item in woe.categories}
    assert "C" not in by_category and "D" not in by_category
    rare = by_category["__rare__"]
    assert rare.count == 5
    assert rare.bad_count == 3

    frame = pd.DataFrame({"chan": ["C", "D", "F"]})
    encoded = apply_categorical_woe(frame, "chan", woe)
    assert encoded.tolist() == pytest.approx([rare.woe, rare.woe, rare.woe])


def test_categorical_woe_encode_unseen_category_falls_back_to_global_prior():
    series = pd.Series(["A"] * 10 + ["B"] * 10)
    target = np.array([1] * 5 + [0] * 5 + [1] * 2 + [0] * 8, dtype=float)
    woe = categorical_woe_encode(series, target, feature="chan", min_count=5)

    frame = pd.DataFrame({"chan": ["A", "NEVER_SEEN", "ALSO_UNSEEN"]})
    encoded = apply_categorical_woe(frame, "chan", woe)

    by_category = {item.category: item for item in woe.categories}
    assert encoded.iloc[0] == pytest.approx(by_category["A"].woe)
    # An unseen category is treated as "no information": WOE 0 (population-average
    # good:bad ratio), never a leak-prone re-fit against new data.
    assert encoded.iloc[1] == pytest.approx(woe.default_woe)
    assert encoded.iloc[2] == pytest.approx(woe.default_woe)
    assert woe.default_woe == pytest.approx(0.0)


def test_categorical_woe_encode_nan_gets_its_own_bucket_and_encode_time_nan_uses_it():
    series = pd.Series(["A"] * 10 + ["B"] * 10 + [None, None])
    target = np.array([1] * 5 + [0] * 5 + [1] * 2 + [0] * 8 + [1, 0], dtype=float)

    woe = categorical_woe_encode(series, target, feature="chan", min_count=5)

    assert woe.na_woe is not None
    frame = pd.DataFrame({"chan": ["A", None, np.nan]})
    encoded = apply_categorical_woe(frame, "chan", woe)
    assert encoded.iloc[1] == pytest.approx(woe.na_woe)
    assert encoded.iloc[2] == pytest.approx(woe.na_woe)


def test_categorical_woe_encode_default_min_count_scales_with_row_count():
    # max(30, 0.5% of fit rows); with 10,000 rows that is max(30, 50) = 50.
    rows = 10_000
    series = pd.Series((["A"] * (rows - 40)) + ["B"] * 40)
    target = np.array(([1, 0] * ((rows - 40) // 2)) + [1, 0] * 20, dtype=float)

    woe = categorical_woe_encode(series, target, feature="chan")

    assert woe.min_count == 50
    assert woe.rare_categories == ("B",)


def test_categorical_woe_encode_rejects_non_binary_or_single_class_target():
    series = pd.Series(["a", "b", "a", "b"])
    with pytest.raises(FeatureError, match="target must be binary"):
        categorical_woe_encode(series, np.array([0.0, 1.0, 2.0, 1.0]), feature="chan")
    with pytest.raises(FeatureError, match="both good and bad"):
        categorical_woe_encode(series, np.array([1.0, 1.0, 1.0, 1.0]), feature="chan")

import numpy as np
import pandas as pd
import pytest

from marvis.feature.contracts import WOEResult
from marvis.feature.encode import label_encode, onehot_encode, woe_encode
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

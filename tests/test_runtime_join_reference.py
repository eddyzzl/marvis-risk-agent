import hashlib

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_join_reference import join_reference


def _spec(method="exact", strategy=None, side="both"):
    return {"keys": [{"anchor_col": "id", "feature_col": "id", "match_method": method,
                      "transform_side": side}], "dedup_strategy": strategy}


def test_left_rows_nulls_and_collision_names_are_preserved():
    left = pd.DataFrame({"id": ["a", "b", " ", None], "x": [1, 2, 3, 4], "feature_x": [0]*4})
    right = pd.DataFrame({"id": [None, "", "a"], "x": [90, 80, 10], "feature_x": [9, 8, 7]})
    output, stages = join_reference(left, [right], [_spec()])
    assert list(output.columns) == ["id", "x", "feature_x", "feature_feature_x", "feature_x_2"]
    assert output["feature_x_2"].iloc[0] == 10
    assert output["feature_x_2"].iloc[1:].isna().all()
    assert stages[0]["members"] == [{"output_row": i, "left_row": i, "right_rows": [2] if i == 0 else None} for i in range(4)]


@pytest.mark.parametrize("strategy,value,members", [("first", 2, [0]), ("last", 8, [1]),
    ("agg_mean", 5, [0, 1]), ("agg_max", 8, [0, 1])])
def test_dedup_applies_after_transform_with_exact_physical_members(strategy, value, members):
    output, stages = join_reference(pd.DataFrame({"id": [" a "]}),
        [pd.DataFrame({"id": ["A", "a"], "x": [2, 8]})], [_spec("exact_lower", strategy)])
    assert output["x"].tolist() == [value]
    assert stages[0]["members"][0]["right_rows"] == members


@pytest.mark.parametrize("method,left,right,side", [
    ("exact", 1.0, "1.00", "both"), ("date", "2024/02/29", "2024-02-29 12:00:00", "both"),
    ("hash:sha256", "value", hashlib.sha256(b"value").hexdigest().upper(), "anchor"),
    ("hash:md5", hashlib.md5(b"value").hexdigest(), "value", "feature")])
def test_key_transform_methods(method, left, right, side):
    output, _ = join_reference(pd.DataFrame({"id": [left]}), [pd.DataFrame({"id": [right], "x": [7]})],
                               [_spec(method, side=side)])
    assert output["x"].tolist() == [7]


def test_no_dedup_decision_cannot_certify_nonunique_keys():
    with pytest.raises(ValueError, match="nonunique"):
        join_reference(pd.DataFrame({"id": ["a"]}), [pd.DataFrame({"id": ["a", "a"]})], [_spec()])


def test_multi_feature_stages_are_sequential_and_preserve_memberships():
    left = pd.DataFrame({"id": ["a", "b"]})
    one = pd.DataFrame({"id": ["b", "a"], "x": [2, 1]})
    two = pd.DataFrame({"id": ["a"], "x": [3]})
    output, stages = join_reference(left, [one, two], [_spec(), _spec()])
    assert output["x"].tolist() == [1, 2]
    assert output["feature_x"].iloc[0] == 3
    assert pd.isna(output["feature_x"].iloc[1])
    assert stages[0]["members"][0]["right_rows"] == [1]
    assert stages[1]["members"][1]["right_rows"] is None

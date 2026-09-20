"""Evaluation rows never become fitting rows by exclusion or missing identity."""
import numpy as np
import pandas as pd
import pytest

from marvis.feature.derive import _agg_fit_mask
from marvis.feature.errors import FeatureError
from marvis.feature.preprocessing import apply_preprocessing_steps, read_preprocessing_chain
from marvis.packs.feature.tools import _stat_fit_mask, _woe_fit_frame
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime


def _mask(frame, inputs, lane):
    if lane == "woe":
        selected, scope = _woe_fit_frame(frame, inputs, "dataset-1")
        return frame.index.isin(selected.index), scope
    if lane == "aggregate":
        return _agg_fit_mask(frame, inputs, dataset_id="dataset-1")
    return _stat_fit_mask(frame, inputs, "normalize", "dataset-1")


@pytest.mark.parametrize("lane", ["woe", "aggregate", "stat"])
@pytest.mark.parametrize("unassigned", [None, "", "future"])
def test_missing_or_unknown_partition_must_not_enter_fit(lane, unassigned):
    frame = pd.DataFrame({"split": ["train", "test", unassigned], "x": [1., 2., 999.]})
    with pytest.raises(FeatureError, match="partition|分区"):
        _mask(frame, {"split_col": "split"}, lane)


@pytest.mark.parametrize("lane", ["woe", "aggregate", "stat"])
def test_omitting_test_from_holdout_does_not_add_test_to_training(lane):
    frame = pd.DataFrame({"split": ["train", "test", "oot"], "x": [1., 999., 9999.]})
    mask, scope = _mask(frame, {"split_col": "split", "holdout_values": ["oot"]}, lane)
    assert np.asarray(mask).tolist() == [True, False, False]
    assert scope == "train"


@pytest.mark.parametrize("lane", ["woe", "aggregate", "stat"])
def test_custom_training_identity_is_explicit_and_cannot_name_evaluation(lane):
    frame = pd.DataFrame({"split": ["development", "test", "holdout"]})
    mask, scope = _mask(frame, {"split_col": "split", "train_values": ["development"]}, lane)
    assert np.asarray(mask).tolist() == [True, False, False]
    assert scope == "train"
    with pytest.raises(FeatureError, match="evaluation|评估"):
        _mask(frame, {"split_col": "split", "train_values": ["test"]}, lane)


def test_onehot_tool_fits_training_categories_and_replays_unknown_as_zero(tmp_path):
    runner, registry, _, backend = _runtime(tmp_path)
    frame = pd.DataFrame({"split": ["train", "train", "test", "oot"],
                          "cat": ["a", "b", "only_test", "only_oot"]})
    source = tmp_path / "categories.csv"
    frame.to_csv(source, index=False)
    dataset = registry.register_from_upload("task-feature", source, role="sample")
    result = runner.invoke(ToolRef("feature", "onehot_encode"),
                           {"dataset_id": dataset.id, "columns": ["cat"], "split_col": "split"},
                           task_id="task-feature")
    assert result.ok, result.error
    assert result.output["mapping"] == {"cat": ["a", "b"]}
    assert result.output["fit_rows"] == 2 and result.output["fit_split"] == "train"
    path = registry.resolve_path(result.output["result_dataset_id"])
    transformed = backend.read_frame(path)
    assert not any("only_" in col for col in transformed.columns)
    assert transformed.loc[2:, ["cat_a", "cat_b"]].to_numpy().sum() == 0
    replay = apply_preprocessing_steps(frame, read_preprocessing_chain(path))
    pd.testing.assert_frame_equal(transformed, replay, check_dtype=False)

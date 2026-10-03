"""Feature selection must use the same declared fit boundary as preprocessing."""

import pandas as pd
import pytest

from marvis.feature.metrics import feature_metrics
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime


def _select(tmp_path, partitions, **overrides):
    runner, registry, _, _ = _runtime(tmp_path)
    count = 20
    frame = pd.DataFrame({
        "partition": [value for value in partitions for _ in range(count)],
        "x": [float(i % 7) for i in range(count * len(partitions))],
        "y": [i % 2 for i in range(count * len(partitions))],
    })
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")
    result = runner.invoke(
        ToolRef("modeling", "select_features"),
        {
            "dataset_id": dataset.id, "features": ["x"], "target_col": "y",
            "split_col": "partition", "iv_min": 0.0,
            "corr_max": 0.99, "vif_max": 100.0, "seed": 0,
            **overrides,
        },
        task_id="task-feature",
    )
    return result, frame


@pytest.mark.parametrize("holdout", [None, ["oot"]])
def test_selection_fits_only_train_despite_other_evaluation_names(tmp_path, holdout):
    result, frame = _select(
        tmp_path, ["train", "test", "oot", "validation", "valid", "holdout"],
        **({"holdout_values": holdout} if holdout else {}),
    )
    assert result.ok, result.error
    expected = frame[frame.partition == "train"]
    metrics = feature_metrics(expected.x.to_numpy(), expected.y.to_numpy(), feature="x")
    assert result.output["fit_rows"] == len(expected)
    assert result.output["fit_split"] == "train"
    assert result.output["scores"]["x"]["iv"] == pytest.approx(metrics.iv)


@pytest.mark.parametrize("value", ["test", "oot", "validation", "VALID", "holdout"])
def test_explicit_evaluation_partition_is_not_reported_as_training(tmp_path, value):
    result, _ = _select(tmp_path, ["train", value], split_value=value)
    assert not result.ok
    assert "evaluation partition cannot be used for fitting" in str(result.error)


@pytest.mark.parametrize("value", ["future", None, ""])
def test_unknown_or_missing_partition_fails_before_selection(tmp_path, value):
    result, _ = _select(tmp_path, ["train", "test", value])
    assert not result.ok
    assert "partition" in str(result.error)


@pytest.mark.parametrize("train", ["train", "development"])
def test_explicit_real_training_partition_remains_supported(tmp_path, train):
    result, _ = _select(tmp_path, [train, "test", "oot"], split_value=train)
    assert result.ok, result.error
    assert result.output["fit_rows"] == 20
    assert result.output["fit_split"] == "train"


def test_declared_custom_holdout_does_not_become_a_training_member(tmp_path):
    result, _ = _select(
        tmp_path, ["development", "future"], split_value="development",
        holdout_values=["future"],
    )
    assert result.ok, result.error
    assert result.output["fit_rows"] == 20


def test_explicit_legacy_numeric_training_alias_remains_supported(tmp_path):
    result, _ = _select(tmp_path, [0, 1, 2], split_value=0, holdout_values=["1", "2"])
    assert result.ok, result.error
    assert result.output["fit_rows"] == 20


def test_boolean_split_value_cannot_silently_select_a_numeric_partition(tmp_path):
    result, _ = _select(tmp_path, [0, 1], split_value=False, holdout_values=["1"])
    assert not result.ok
    assert "explicit training partition" in str(result.error)

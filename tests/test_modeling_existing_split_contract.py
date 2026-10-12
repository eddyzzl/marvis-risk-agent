"""Actual native outputs preserve the source development/OOT membership."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from marvis.agent.adjust_specs import adjust_param_error, normalize_adjust_params
from marvis.modeling_split import SplitConfigurationError, validate_split_configuration
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.prepare import _make_split, prepare_modeling_frame
from marvis.plugins.manifest import ToolRef
from marvis.db import PluginRepository
from tests.test_modeling_pack import _runtime
from tests.test_modeling_prepare import _register_frame


def _frame(column="split", labels=("train", "test", "oot")):
    return pd.DataFrame({
        "signal": range(600), "affordability": [i % 7 for i in range(600)],
        "noise": [i % 3 for i in range(600)], "y": [0, 1] * 300,
        column: [labels[0]] * 360 + [labels[1]] * 120 + [labels[2]] * 120,
    })


@pytest.mark.parametrize("tool", ["make_split", "prepare_modeling_frame"])
@pytest.mark.parametrize("config", [{}, {"method": "existing"}])
@pytest.mark.parametrize("column,labels", [
    ("split", ("train", "test", "oot")),
    ("cohort", ("train", "test", "oot")),
    ("cohort", ("Training", "validation", "out_of_time")),
])
def test_native_existing_split_preserves_exact_membership_and_output_binding(tmp_path, tool, config, column, labels):
    runner, _plugins, registry, backend, settings, task = _runtime(tmp_path)
    source = _frame(column, labels)
    path = tmp_path / "source-existing.parquet"
    source.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {
        "dataset_id": dataset.id, "target_col": "y",
        "feature_cols": ["signal", "affordability", "noise"],
        "split_col": column, "split_config": config, "seed": 23,
    }
    result = runner.invoke(ToolRef("modeling", tool), inputs, task_id=task.id)
    assert result.ok, result.error
    assert inputs["split_config"] == config
    expected_input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    audit = [row for row in PluginRepository(settings.db_path).list_audit(kind="tool.invoke") if row["target_ref"] == f"modeling.{tool}"]
    assert audit and audit[-1]["inputs_hash"] == expected_input_hash
    output = result.output
    assert output["split_col"] == column
    assert output["split_values"] == dict(zip(("train", "test", "oot"), labels, strict=True))
    assert output["holdout_values"] == [labels[2]]
    counts = output["sample_analysis"]["split_counts"] if tool == "make_split" else output["split_counts"]
    assert counts == dict(zip(labels, (360, 120, 120), strict=True))
    derived = backend.read_frame(registry.resolve_path(output["result_dataset_id"]))
    pd.testing.assert_frame_equal(derived[source.columns], source)
    for label in labels:
        assert derived.loc[derived[column] == label, "signal"].tolist() == source.loc[source[column] == label, "signal"].tolist()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert "split" not in derived if column == "cohort" else True


@pytest.mark.parametrize("config", [
    {"oot_by_time": True}, {"oot_by_time": False}, {"oot_by_time": "True"},
    {"method": "unknown"}, {"method": "existing", "test_size": .3},
    {"unrecognized": True}, {"test_size": True}, {"random_oot": "True"},
    {"rules": [{"when": [{"col": True, "op": "eq", "val": 1}], "assign": "test"}]},
])
def test_invalid_split_configuration_creates_no_native_artifact(tmp_path, config):
    backend, registry, dataset = _register_frame(tmp_path, _frame())
    before_files = set(tmp_path.rglob("*.parquet"))
    before_datasets = registry.list_for_task(dataset.task_id)
    with pytest.raises(ModelingError):
        prepare_modeling_frame(registry, backend, dataset.id, target_col="y", feature_cols=["signal"],
                               split_col="split", split_config=config, seed=23)
    assert set(tmp_path.rglob("*.parquet")) == before_files
    assert registry.list_for_task(dataset.task_id) == before_datasets


@pytest.mark.parametrize("split_col", [None, "missing", True, False])
def test_existing_requires_an_actual_typed_source_partition_column(tmp_path, split_col):
    backend, registry, dataset = _register_frame(tmp_path, _frame().drop(columns="split"))
    before = set(tmp_path.rglob("*.parquet"))
    with pytest.raises(ModelingError):
        prepare_modeling_frame(registry, backend, dataset.id, target_col="y", feature_cols=["signal"],
                               split_col=split_col, split_config={"method": "existing"}, seed=23)
    assert set(tmp_path.rglob("*.parquet")) == before


@pytest.mark.parametrize("bad_label", [None, "future", "training"])
def test_unknown_null_or_ambiguous_existing_labels_cannot_disappear_from_analysis(tmp_path, bad_label):
    source = _frame()
    source.loc[0, "split"] = bad_label
    backend, registry, dataset = _register_frame(tmp_path, source)
    before = set(tmp_path.rglob("*.parquet"))
    with pytest.raises(ModelingError, match="现有切分"):
        prepare_modeling_frame(registry, backend, dataset.id, target_col="y", feature_cols=["signal"],
                               split_col="split", split_config={"method": "existing"}, seed=23)
    assert set(tmp_path.rglob("*.parquet")) == before


def test_existing_kernel_and_adjustment_keep_configuration_semantics():
    source = _frame()
    pd.testing.assert_frame_equal(_make_split(source, {"method": "existing"}, seed=23), source)
    params = normalize_adjust_params({"split_config": {"method": "existing"}})
    assert params == {"split_config": {"method": "existing"}}
    assert adjust_param_error(params) is None
    for invalid in ({"oot_by_time": True}, {"method": "time", "date_col": True}, {"method": "random", "oot_by_time": True}):
        params = normalize_adjust_params({"split_config": invalid})
        assert adjust_param_error(params)
    with pytest.raises(SplitConfigurationError):
        validate_split_configuration({"method": "existing", "random_oot": False})


def test_boolean_time_binding_is_rejected_even_if_literal_true_column_exists(tmp_path):
    source = _frame().assign(**{"True": range(600)})
    backend, registry, dataset = _register_frame(tmp_path, source)
    params = normalize_adjust_params({"split_config": {"oot_by_time": True}})
    assert params["split_config"]["oot_by_time"] is True
    with pytest.raises(ModelingError, match="列名字符串"):
        prepare_modeling_frame(registry, backend, dataset.id, target_col="y", feature_cols=["signal"],
                               split_col="split", split_config=params["split_config"], seed=23)


def test_explicit_zero_time_oot_has_no_member_in_oot_actual_native_tool(tmp_path):
    runner, _plugins, registry, backend, _settings, task = _runtime(tmp_path)
    source = pd.DataFrame({"signal": range(100), "date": range(100), "y": [0, 1] * 50})
    path = tmp_path / "zero-time-oot.parquet"
    source.to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")
    result = runner.invoke(ToolRef("modeling", "make_split"), {
        "dataset_id": dataset.id, "target_col": "y", "feature_cols": ["signal"],
        "split_config": {"oot_by_time": "date", "oot_size": 0, "test_size": .2}, "seed": 7,
    }, task_id=task.id)
    assert result.ok, result.error
    assert result.output["sample_analysis"]["split_counts"] == {"train": 80, "test": 20}
    assert result.output["holdout_values"] == []
    output = backend.read_frame(registry.resolve_path(result.output["result_dataset_id"]))
    assert output["signal"].tolist() == list(range(100))
    assert not output["split"].eq("oot").any()

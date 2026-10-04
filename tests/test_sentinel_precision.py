"""Native masking and scoring replay must agree at numeric precision boundaries."""

import numpy as np
import pandas as pd
import pytest

from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.feature.preprocessing import apply_preprocessing_steps
from marvis.feature.transform import mask_sentinel_values
from marvis.plugins.manifest import ToolRef
from tests.test_modeling_special_values import _runtime


@pytest.mark.parametrize("dtype,values,declared,expected", [
    ("int64", [-999, 0, 1, -999], -999.5, []),
    ("float32", [-999, 0, 1, -999], -999.00001, []),
    ("int64", [-999, 0, 1, -999], -999.0, [0, 3]),
    ("float32", [-999, 0, 1, -999], -999.0, [0, 3]),
    ("int64", [2**60, 2**60 + 1, 2**60 + 2, 0], float(2**60), [0]),
    ("uint64", [2**63, 2**63 + 1, 2**63 + 2, 0], float(2**63), [0]),
    ("Int64", [2**60, 2**60 + 1, None, 0], float(2**60), [0, 2]),
    ("UInt64", [2**63, 2**63 + 1, None, 0], float(2**63), [0, 2]),
    ("int64", [2**60, 2**60 + 1, 2**60 + 2, 0], 2**60 + 1, [1]),
    ("uint64", [2**63, 2**63 + 1, 2**63 + 2, 0], 2**63 + 1, [1]),
    ("Int64", [2**60, 2**60 + 1, None, 0], 2**60 + 1, [1, 2]),
    ("UInt64", [2**63, 2**63 + 1, None, 0], 2**63 + 1, [1, 2]),
])
def test_native_mask_and_replay_keep_exact_numeric_identity(tmp_path, dtype, values, declared, expected):
    runner, registry, _, task = _runtime(tmp_path)
    original = pd.DataFrame({"x1": pd.Series(values, dtype=dtype), "y": [0, 1, 0, 1],
                             "split": ["train", "train", "test", "oot"]})
    path = tmp_path / "precision.parquet"
    original.to_parquet(path, index=False)
    dataset = registry.register_from_upload(task.id, path, role="sample")
    result = runner.invoke(ToolRef("modeling", "resolve_special_values"), {
        "dataset_id": dataset.id, "features": ["x1"],
        "sentinel_columns": {"x1": [[declared, .5]]},
        "decisions": {"x1": {"action": "mask", "values": [declared]}},
    }, task_id=task.id)
    assert result.ok, result.error
    state = load_preprocessing_state(registry, result.output["result_dataset_id"])
    native = registry.read_authenticated_parquet_snapshot(result.output["result_dataset_id"])
    replay = apply_preprocessing_steps(original, state.steps)
    assert np.flatnonzero(native.x1.isna()).tolist() == expected
    assert np.flatnonzero(replay.x1.isna()).tolist() == expected
    assert state.assurance == "row_local"
    assert state.steps[-1]["params"]["x1"] == [declared]
    assert result.output["governance"]["x1"]["detected_values"] == [declared]
    if pd.api.types.is_integer_dtype(original.x1.dtype):
        for index in set(range(len(original))) - set(expected):
            assert int(replay.x1.iloc[index]) == int(original.x1.iloc[index])
    # Check the stored Arrow values directly: pandas may legitimately promote
    # a nullable int64 column to float64 when loading an output with nulls.
    import pyarrow.parquet as pq
    stored = pq.read_table(registry.resolve_path(result.output["result_dataset_id"])).column("x1").to_pylist()
    assert stored == [None if i in expected else value for i, value in enumerate(original.x1.tolist())]


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
def test_nullable_integer_replay_does_not_round_adjacent_values(dtype):
    series = pd.Series([2**60, 2**60 + 1, None], dtype=dtype)
    result = mask_sentinel_values(series, [float(2**60)])
    assert result.isna().tolist() == [True, False, True]
    assert int(result.iloc[1]) == 2**60 + 1


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64"])
def test_integer_declaration_cannot_match_a_rounded_float(dtype):
    series = pd.Series([float(2**60), 0], dtype=dtype)
    assert not mask_sentinel_values(series, [2**60 + 1]).isna().any()
    assert mask_sentinel_values(series, [2**60]).isna().tolist() == [True, False]


def test_fractional_imputation_after_exact_integer_mask_still_replays():
    from marvis.feature.transform import impute_missing

    original = pd.DataFrame({"x": [1, 2, -999]})
    filled, value = impute_missing(original.x, strategy="mean", sentinel_values=[-999])
    replay = apply_preprocessing_steps(original, [
        {"kind": "sentinel", "columns": ["x"], "params": {"x": [-999]}},
        {"kind": "impute", "columns": ["x"], "params": {"x": value}},
    ])
    assert filled.tolist() == replay.x.tolist() == [1.0, 2.0, 1.5]


@pytest.mark.parametrize("dtype", ["uint8", "UInt64"])
def test_unsigned_negative_fill_promotes_safely_and_replays(tmp_path, dtype):
    from tests.test_feature_pack import _runtime as feature_runtime

    runner, registry, _, _ = feature_runtime(tmp_path)
    original = pd.DataFrame({"x": pd.Series([255, 1, 2, 255], dtype=dtype)})
    if dtype == "UInt64":
        original.loc[2, "x"] = pd.NA
    path = tmp_path / "unsigned-fill.parquet"
    original.to_parquet(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    result = runner.invoke(ToolRef("feature", "impute_missing"), {
        "dataset_id": source.id, "columns": ["x"], "allow_full_fit": True,
        "sentinel_values": [255], "strategy": "constant", "fill_value": -1,
    }, task_id="task-feature")
    assert result.ok, result.error
    state = load_preprocessing_state(registry, result.output["result_dataset_id"])
    replay = apply_preprocessing_steps(original, state.steps)
    stored = registry.read_authenticated_parquet_snapshot(result.output["result_dataset_id"])
    assert stored.x.tolist() == replay.x.tolist() == [-1, 1, -1 if dtype == "UInt64" else 2, -1]


@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("tool_name,extra", [
    ("cap_outliers", {"method": "quantile", "lower_q": 0.0, "upper_q": 1.0}),
    ("normalize", {"method": "minmax"}),
    ("impute_missing", {"strategy": "constant", "fill_value": 0, "add_indicators": True}),
])
def test_feature_tool_and_replay_preserve_integer_declarations(tmp_path, unsigned, tool_name, extra):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tests.test_feature_pack import _runtime as feature_runtime

    runner, registry, _, _ = feature_runtime(tmp_path)
    base = 2**63 if unsigned else 2**60
    values = [base, base + 1, None, 0, 10, base, base + 1, None]
    path = tmp_path / "native-integers.parquet"
    pq.write_table(pa.table({"x": pa.array(values, type=pa.uint64() if unsigned else pa.int64()),
        "split": ["train"] * 5 + ["test"] * 3}), path)
    source = registry.register_from_upload("task-feature", path, role="sample")
    result = runner.invoke(ToolRef("feature", tool_name), {
        "dataset_id": source.id, "columns": ["x"], "split_col": "split",
        "sentinel_values": [base + 1], **extra,
    }, task_id="task-feature")
    assert result.ok, result.error
    state = load_preprocessing_state(registry, result.output["result_dataset_id"])
    assert state.steps[0]["params"]["x"] == [base + 1]
    original = pd.DataFrame({"x": pd.Series(values, dtype="UInt64" if unsigned else "Int64")})
    replay = apply_preprocessing_steps(original, state.steps)
    stored = registry.read_authenticated_parquet_snapshot(result.output["result_dataset_id"],
                                                         preserve_integer_values=True)
    if tool_name == "impute_missing":
        assert stored.x.tolist() == replay.x.tolist() == [base, 0, 0, 0, 10, base, 0, 0]
        assert stored.x__was_missing.tolist() == replay.x__was_missing.tolist() == [0, 1, 1, 0, 0, 0, 1, 1]
    else:
        assert stored.x.isna().tolist() == replay.x.isna().tolist() == [False, True, True, False, False, False, True, True]
        if tool_name == "cap_outliers":
            assert result.output["bounds"]["x"]["upper"] == float(base)
        else:
            assert stored.x.iloc[0] == replay.x.iloc[0] == 1.0


def test_training_rejects_adjacent_integer_sentinel_evidence():
    from marvis.packs.modeling.errors import SpecialValueDecisionRequiredError
    from marvis.packs.modeling.special_value_tools import SPECIAL_VALUE_POLICY_VERSION, special_value_decision_fingerprint
    from marvis.packs.modeling.train_tools import _assert_sentinel_preprocessing_governed

    base = 2**60
    evidence = {"policy_version": SPECIAL_VALUE_POLICY_VERSION, "column": "x", "action": "mask",
        "detected_values": [base + 1], "confirmed": False, "reason": "", "source_dataset_id": "source",
        "source_dataset_content_hash": "abc", "resolved_dataset_id": "derived"}
    evidence["decision_fingerprint"] = special_value_decision_fingerprint(evidence)
    with pytest.raises(SpecialValueDecisionRequiredError, match="特殊值"):
        _assert_sentinel_preprocessing_governed(dataset_id="derived", dataset_content_hash="def",
            features=["x"], sentinel_columns={"x": [[base + 1, .5]]}, governance={"x": evidence},
            preprocessing_steps=[{"kind": "sentinel", "columns": ["x"], "params": {"x": [base]}}])

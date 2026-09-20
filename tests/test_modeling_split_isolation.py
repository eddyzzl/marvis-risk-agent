"""Explicit entity isolation must not silently degrade into row sampling."""
import pandas as pd
import pytest

from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.prepare import _make_split, prepare_modeling_frame
from tests.test_modeling_prepare import _register_frame


@pytest.mark.parametrize("groups", [["missing"], ["person", "missing"], "missing"])
def test_missing_declared_group_column_is_rejected(groups):
    frame = pd.DataFrame({"person": list(range(20)), "x": list(range(20))})
    with pytest.raises(ModelingError, match="分组列.*missing"):
        _make_split(frame, {"group_cols": groups, "test_size": .3}, seed=7)
    assert "split" not in frame


@pytest.mark.parametrize("missing", [None, float("nan"), "", "  "])
def test_unresolved_entity_identity_cannot_claim_group_isolation(missing):
    frame = pd.DataFrame({"person": [missing, *range(1, 20)], "x": range(20)})
    with pytest.raises(ModelingError, match="分组列.*缺失"):
        _make_split(frame, {"group_cols": ["person"], "test_size": .3}, seed=7)


def test_time_oot_cannot_split_one_declared_entity_between_partitions():
    frame = pd.DataFrame({"person": ["repeated", "b", "c", "d", "e", "repeated"],
                          "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]})
    with pytest.raises(ModelingError, match="跨分区"):
        _make_split(frame, {"group_cols": ["person"], "oot_by_time": "month",
                            "oot_size": .2, "test_size": 0}, seed=1)


def test_explicit_rules_cannot_override_entity_isolation():
    frame = pd.DataFrame({"person": ["same", "same", "other"], "channel": ["a", "b", "c"]})
    rules = [{"when": [{"col": "channel", "op": "eq", "val": "a"}], "assign": "train"},
             {"when": [{"col": "channel", "op": "eq", "val": "b"}], "assign": "test"}]
    with pytest.raises(ModelingError, match="跨分区"):
        _make_split(frame, {"group_cols": ["person"], "rules": rules, "test_size": 0}, seed=1)


def test_grouped_time_split_keeps_valid_entities_and_existing_determinism():
    frame = pd.DataFrame({"person": [i for i in range(20) for _ in range(2)],
                          "month": [f"2025-{i // 2 + 1:02d}" for i in range(20) for _ in range(2)]})
    config = {"group_cols": ["person"], "oot_by_time": "month", "oot_size": .2, "test_size": .3}
    first = _make_split(frame, config, seed=7)
    pd.testing.assert_frame_equal(first, _make_split(frame, config, seed=7))
    assert set(first["split"]) == {"train", "test", "oot"}
    assert first.groupby("person")["split"].nunique().eq(1).all()


def test_prepare_rejects_missing_isolation_column_before_writing_artifacts(tmp_path):
    frame = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    backend, registry, dataset = _register_frame(tmp_path, frame)
    before = set(tmp_path.rglob("*.parquet"))
    with pytest.raises(ModelingError, match="分组列.*customer_id"):
        prepare_modeling_frame(registry, backend, dataset.id, target_col="y", feature_cols=["x"],
                               split_col=None, split_config={"group_cols": ["customer_id"]}, seed=7)
    assert set(tmp_path.rglob("*.parquet")) == before

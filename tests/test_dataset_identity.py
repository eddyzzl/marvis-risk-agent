"""NaN profile equality must not mask any real registered-dataset drift."""
from dataclasses import replace
import math

import pandas as pd
import pytest

from marvis.data.dataset_identity import dataset_identity_equal
from marvis.data.errors import DatasetContentDriftError
from marvis.data import registry as registry_module
from tests.test_feature_pack import _runtime


@pytest.fixture
def empty_dataset(tmp_path):
    _, registry, repo, _ = _runtime(tmp_path)
    path = registry.datasets_root / "empty.parquet"
    pd.DataFrame({"id": pd.Series(dtype="int64"), "x": pd.Series(dtype="float64")}).to_parquet(path, index=False)
    with repo.transaction() as conn:
        dataset = registry.register_existing_on_connection(conn, path, task_id="task-feature", role="sample")
    return registry, dataset


def test_real_empty_metadata_is_stable_without_repin_or_replacing_nan(empty_dataset):
    registry, dataset = empty_dataset
    before = registry.get(dataset.id)
    assert math.isnan(before.columns[0].null_rate)
    assert registry.authenticated_parquet_column_names(dataset.id) == ("id", "x")
    after = registry.get(dataset.id)
    assert dataset_identity_equal(before, after)
    assert after.source_path == before.source_path
    assert math.isnan(after.columns[0].null_rate)


@pytest.mark.parametrize(("field", "value"), [
    ("role", "changed"), ("source_path", "elsewhere.parquet"),
    ("content_hash", "0" * 64), ("row_count", 1), ("target_col", "x"),
])
def test_real_empty_metadata_rejects_concurrent_identity_changes(empty_dataset, monkeypatch, field, value):
    registry, dataset = empty_dataset
    original = registry_module.read_authenticated_parquet_metadata

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with registry.transaction() as conn:
            # field comes from this fixed test parameter list, never user input.
            conn.execute(f"UPDATE datasets SET {field}=? WHERE id=?", (value, dataset.id))
        return result

    monkeypatch.setattr(registry_module, "read_authenticated_parquet_metadata", mutate)
    with pytest.raises(DatasetContentDriftError, match="metadata differs"):
        registry.authenticated_parquet_column_names(dataset.id)


def test_only_paired_nan_is_equal_and_all_nested_profile_fields_remain_bound(empty_dataset):
    registry, dataset = empty_dataset
    repeated = registry.get(dataset.id)
    assert dataset_identity_equal(dataset, repeated)
    for column in (
        replace(repeated.columns[0], null_rate=0.0),
        replace(repeated.columns[0], dtype="object"),
        replace(repeated.columns[0], fingerprint=replace(repeated.columns[0].fingerprint, value_kind="changed")),
    ):
        assert not dataset_identity_equal(dataset, replace(repeated, columns=(column, *repeated.columns[1:])))
    assert not dataset_identity_equal(dataset, None)


def test_empty_metadata_still_authenticates_file_bytes(empty_dataset):
    registry, dataset = empty_dataset
    pd.DataFrame({"id": [1], "x": [2.0]}).to_parquet(registry.resolve_path(dataset.id), index=False)
    with pytest.raises(DatasetContentDriftError):
        registry.authenticated_parquet_column_names(dataset.id)

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.errors import DataBackendError
from marvis.data.profiler import profile_dataset
from marvis.data.sampler import sample_dataset


def test_sample_dataset_supports_head_and_random(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"id": range(10), "target": [0, 1] * 5}).to_csv(path, index=False)
    backend = DataBackend(tmp_path)

    head = sample_dataset(backend, path, 3, strategy="head")
    random_sample = sample_dataset(backend, path, 4, strategy="random", seed=123)

    assert head["id"].tolist() == [0, 1, 2]
    assert random_sample.shape == (4, 2)
    assert set(random_sample.columns) == {"id", "target"}


def test_sample_dataset_stratified_keeps_each_class_represented(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({
        "id": range(10),
        "segment": ["major"] * 8 + ["minor"] * 2,
    }).to_csv(path, index=False)
    backend = DataBackend(tmp_path)

    sample = sample_dataset(
        backend,
        path,
        4,
        strategy="stratified",
        stratify_col="segment",
        seed=7,
    )

    assert sample.shape[0] == 4
    assert set(sample["segment"]) == {"major", "minor"}
    assert sample["segment"].value_counts()["major"] >= sample["segment"].value_counts()["minor"]


def test_sample_dataset_rejects_invalid_strategies_and_columns(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(path, index=False)
    backend = DataBackend(tmp_path)

    with pytest.raises(DataBackendError):
        sample_dataset(backend, path, 0)
    with pytest.raises(DataBackendError):
        sample_dataset(backend, path, 1, strategy="missing")
    with pytest.raises(DataBackendError):
        sample_dataset(backend, path, 1, strategy="stratified", stratify_col="segment")


def test_profile_dataset_uses_backend_sample_and_schema_inference(tmp_path):
    path = tmp_path / "profile.parquet"
    pd.DataFrame({
        "mobile": ["13800138000", "13900139000", "13700137000"],
        "apply_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "bad_flag": [0, 1, 0],
    }).to_parquet(path, index=False)
    backend = DataBackend(tmp_path)

    profiles = {profile.name: profile for profile in profile_dataset(backend, path)}

    assert set(profiles) == {"mobile", "apply_date", "bad_flag"}
    assert profiles["mobile"].semantic_role == "phone"
    assert profiles["apply_date"].semantic_role == "date"
    assert profiles["bad_flag"].semantic_role == "target"

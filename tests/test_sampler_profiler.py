from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.errors import DataBackendError
from marvis.data.profiler import profile_dataset
from marvis.data.sampler import sample_dataset


def test_sample_dataset_warns_with_supported_replacements(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(path, index=False)
    backend = DataBackend(tmp_path)

    with pytest.warns(
        DeprecationWarning,
        match=(
            r"sample_dataset\(\) is deprecated.*"
            r"DataBackend\.read_frame.*DataBackend\.sample_rows.*"
            r"DataBackend\.sample_rows_stratified"
        ),
    ) as captured:
        sample_dataset(backend, path, 1, strategy="head")

    assert Path(captured[0].filename).resolve() == Path(__file__).resolve()


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


def test_sample_dataset_stratified_keeps_requested_size_when_n_less_than_strata(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({
        "id": range(10),
        "segment": [f"s{index}" for index in range(10)],
    }).to_csv(path, index=False)
    backend = DataBackend(tmp_path)

    sample = sample_dataset(
        backend,
        path,
        3,
        strategy="stratified",
        stratify_col="segment",
        seed=7,
    )

    assert sample.shape[0] == 3
    assert sample["segment"].nunique() == 3


def test_backend_exposes_deterministic_stratified_sampling_replacement(tmp_path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({
        "id": range(12),
        "segment": ["major"] * 8 + ["minor"] * 3 + [None],
    }).to_parquet(path, index=False)
    backend = DataBackend(tmp_path)

    first = backend.sample_rows_stratified(
        path,
        6,
        stratify_col="segment",
        seed=17,
    )
    second = backend.sample_rows_stratified(
        path,
        6,
        stratify_col="segment",
        seed=17,
    )

    assert first.equals(second)
    assert first.shape == (6, 2)
    assert set(first["segment"].dropna()) == {"major", "minor"}
    assert first["segment"].isna().any()


def test_compatibility_sampler_delegates_stratified_sampling_to_backend(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"id": [1], "segment": ["only"]}).to_csv(path, index=False)
    backend = DataBackend(tmp_path)
    expected = pd.DataFrame({"id": [99], "segment": ["delegated"]})
    calls = []

    def sample_rows_stratified(path, n, *, stratify_col, seed):
        calls.append((path, n, stratify_col, seed))
        return expected

    monkeypatch.setattr(backend, "sample_rows_stratified", sample_rows_stratified)

    actual = sample_dataset(
        backend,
        path,
        1,
        strategy="stratified",
        stratify_col="segment",
        seed=23,
    )

    assert actual is expected
    assert calls == [(path, 1, "segment", 23)]


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

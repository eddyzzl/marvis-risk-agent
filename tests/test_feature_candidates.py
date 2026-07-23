import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.feature.candidates import (
    candidate_numeric_features,
    excluded_numeric_columns,
    suspected_categorical_columns,
)


def _write(tmp_path, frame: pd.DataFrame):
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_suspected_categorical_columns_flags_low_cardinality_integer_code(tmp_path):
    """PREP-5: a numeric column with a code-like name, few distinct values, and
    all-integer values should be flagged as suspected_categorical -- informational
    only, candidate_numeric_features is untouched."""
    backend = DataBackend(tmp_path / "datasets")
    frame = pd.DataFrame({
        "region_code": [1, 2, 3, 1, 2, 3, 1, 2] * 4,
        "amount": np.linspace(10.0, 100.0, 32),
        "y": [0, 1] * 16,
    })
    path = _write(tmp_path, frame)

    suspected = suspected_categorical_columns(backend, path, target_col="y")

    columns = {item.column: item.cardinality for item in suspected}
    assert columns == {"region_code": 3}


def test_suspected_categorical_columns_ignores_high_cardinality_or_non_integer_or_plain_names(tmp_path):
    backend = DataBackend(tmp_path / "datasets")
    frame = pd.DataFrame({
        # Code-like name but > max_cardinality distinct values -> not flagged.
        "zip_code": list(range(1, 33)),
        # Code-like name, low cardinality, but non-integer values -> not flagged.
        "type_score": [1.5, 2.5, 3.5, 1.5] * 8,
        # Low cardinality integer values but a plain business name -> not flagged
        # (no ordinal-distance-free naming signal).
        "tenure_years": [1, 2, 3, 1, 2, 3, 1, 2] * 4,
        "y": [0, 1] * 16,
    })
    path = _write(tmp_path, frame)

    suspected = suspected_categorical_columns(backend, path, target_col="y")

    assert suspected == []


def test_suspected_categorical_columns_excludes_target_and_split_columns(tmp_path):
    backend = DataBackend(tmp_path / "datasets")
    frame = pd.DataFrame({
        "channel_code": [1, 2, 3, 1, 2, 3, 1, 2] * 4,
        "y": [1, 2, 3, 1, 2, 3, 1, 2] * 4,  # low-cardinality int target with a code-like alias name
        "split_code": [0, 1] * 16,
    })
    path = _write(tmp_path, frame)

    suspected = suspected_categorical_columns(
        backend, path, target_col="y", split_col="split_code"
    )

    columns = {item.column for item in suspected}
    assert columns == {"channel_code"}


def test_candidate_numeric_features_excludes_compact_join_date_keys(tmp_path):
    """Joined samples often carry compact technical keys such as applydt/usedate.
    They are dates used to align tables, not model features, even when stored as
    YYYYMMDD integers without an underscore separator."""
    backend = DataBackend(tmp_path / "datasets")
    frame = pd.DataFrame({
        "applydt": [20260101, 20260102, 20260103, 20260104],
        "usedate": [20260101, 20260102, 20260103, 20260104],
        "x1": [0.1, 0.2, 0.8, 0.9],
        "x2": [1.0, 0.0, 1.0, 0.0],
        "y": [0, 0, 1, 1],
    })
    path = _write(tmp_path, frame)

    assert candidate_numeric_features(backend, path, target_col="y") == ["x1", "x2"]


def test_business_prefixes_are_not_silently_classified_as_metadata(tmp_path):
    backend = DataBackend(tmp_path / "datasets")
    frame = pd.DataFrame({
        "loan_amount": [10, 20, 30, 40],
        "apply_count": [1, 2, 3, 4],
        "mobile_age": [5, 6, 7, 8],
        "cust_income": [100, 200, 300, 400],
        "cust_id": [101, 102, 103, 104],
        "applydt": [20260101, 20260102, 20260103, 20260104],
        "sample_weight": [1.0, 1.0, 1.0, 1.0],
        "y": [0, 0, 1, 1],
    })
    path = _write(tmp_path, frame)

    candidates = candidate_numeric_features(backend, path, target_col="y")
    excluded = {
        item.column: item.reason
        for item in excluded_numeric_columns(backend, path, target_col="y")
    }

    assert {"loan_amount", "apply_count", "mobile_age", "cust_income"} <= set(candidates)
    assert excluded == {
        "cust_id": "结构化标识字段",
        "applydt": "结构化日期/时间字段",
        "sample_weight": "样本权重字段",
    }


def test_explicit_feature_override_can_include_numeric_technical_column(tmp_path):
    backend = DataBackend(tmp_path / "datasets")
    frame = pd.DataFrame({
        "cust_id": [101, 102, 103, 104],
        "x1": [0.1, 0.2, 0.8, 0.9],
        "y": [0, 0, 1, 1],
    })
    path = _write(tmp_path, frame)

    candidates = candidate_numeric_features(
        backend,
        path,
        target_col="y",
        include_columns=["cust_id"],
    )

    assert candidates == ["cust_id", "x1"]

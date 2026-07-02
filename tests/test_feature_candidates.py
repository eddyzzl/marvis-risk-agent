import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.feature.candidates import suspected_categorical_columns


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

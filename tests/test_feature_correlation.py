import math

import numpy as np
import pandas as pd
import pytest

from marvis.feature.correlation import (
    correlation_matrix,
    correlation_report,
    find_collinear_pairs,
    safe_correlation,
    vif,
)


def test_safe_correlation_returns_zero_for_nan_and_zero_variance():
    assert safe_correlation(np.array([1, 1, 1]), np.array([1, 2, 3])) == 0.0
    assert safe_correlation(np.array([1, np.nan, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)


def test_correlation_matrix_and_collinear_pairs():
    frame = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [2, 4, 6, 8],
        "x3": [4, 3, 2, 1],
    })
    matrix = correlation_matrix(frame, ["x1", "x2", "x3"])
    pairs = find_collinear_pairs(matrix, ["x1", "x2", "x3"], threshold=0.9)

    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1.0
    assert ("x1", "x2", 1.0) in pairs
    assert ("x1", "x3", -1.0) in pairs


def test_vif_and_correlation_report_shape():
    frame = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [2, 4, 6, 8],
        "x3": [1, 1, 2, 3],
    })

    values = vif(frame, ["x1", "x2", "x3"])
    report = correlation_report(frame, ["x1", "x2", "x3"], threshold=0.9)

    # Perfect collinearity (x2 = 2*x1) → VIF is capped at a large FINITE sentinel, not
    # inf, so it stays JSON-safe (responses serialize with allow_nan=False).
    assert math.isfinite(values["x1"]) and values["x1"] >= 1e6
    assert math.isfinite(values["x2"]) and values["x2"] >= 1e6
    assert report.features == ("x1", "x2", "x3")
    assert len(report.matrix) == 3
    assert any(pair[:2] == ("x1", "x2") for pair in report.collinear_pairs)


def test_vif_is_json_safe_under_perfect_collinearity():
    """A perfectly-collinear pair must serialize with allow_nan=False (the Starlette
    response setting) — i.e. no Infinity leaks to the HTTP/browser layer."""
    import json

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    values = vif(frame, ["a", "b"])
    assert all(math.isfinite(v) for v in values.values())
    json.dumps(values, allow_nan=False)  # would raise ValueError if any value were inf


def test_vif_all_nan_column_does_not_zero_other_features():
    """One entirely-NaN feature must not collapse the listwise-dropped frame and zero
    every other feature's VIF — the NaN column is excluded from the design instead."""
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],          # perfectly collinear with a
        "dead": [np.nan, np.nan, np.nan, np.nan],
    })
    values = vif(frame, ["a", "b", "dead"])
    assert values["dead"] == 0.0            # unusable column → 0, not an error
    assert values["a"] >= 1e6 and values["b"] >= 1e6  # real collinearity still detected

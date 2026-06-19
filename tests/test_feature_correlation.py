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

    assert math.isinf(values["x1"])
    assert math.isinf(values["x2"])
    assert report.features == ("x1", "x2", "x3")
    assert len(report.matrix) == 3
    assert any(pair[:2] == ("x1", "x2") for pair in report.collinear_pairs)

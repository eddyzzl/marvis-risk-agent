"""Tests for the deterministic synthetic sample-data generator (marvis.sample_data)."""

from __future__ import annotations

import pandas as pd

from marvis.sample_data import generate_sample_frame


def test_generate_sample_frame_is_deterministic():
    first = generate_sample_frame()
    second = generate_sample_frame()
    pd.testing.assert_frame_equal(first, second)


def test_generate_sample_frame_shape_and_columns():
    frame = generate_sample_frame()
    assert len(frame) == 1500
    assert "apply_month" in frame.columns
    assert "y" in frame.columns
    assert set(frame["y"].unique()) <= {0, 1}
    feature_cols = [c for c in frame.columns if c not in {"apply_month", "y"}]
    assert len(feature_cols) == 6
    # Realistic-ish bad rate: a demo sample that's ~50/50 doesn't read as credible
    # to a credit-risk reviewer, and it also isn't exercising a leakage/imbalance
    # screen the way a skewed real portfolio would.
    bad_rate = frame["y"].mean()
    assert 0.05 < bad_rate < 0.35

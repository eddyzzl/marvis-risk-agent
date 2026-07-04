"""T4-1 self-test: the dirty-shape injector really produces the claimed shapes.

A generator that silently drifts (e.g. stops injecting the sentinel, or emits a
non-monotone "snapshot" panel) would let the regression net pass on clean data
and give false confidence. These tests assert the STRUCTURE of each generated
shape and that generation is deterministic — they are the injector's own guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from support import dirty_shapes as ds


@pytest.mark.parametrize("name", sorted(ds.DIRTY_SHAPES))
def test_every_shape_is_deterministic(name):
    """Same call -> byte-identical frame (fixed seed, no random/wall-clock)."""
    first = ds.build(name, seed=7)
    second = ds.build(name, seed=7)
    pd.testing.assert_frame_equal(first.frame, second.frame)


@pytest.mark.parametrize("name", sorted(ds.DIRTY_SHAPES))
def test_every_shape_declares_its_roles_against_real_columns(name):
    shape = ds.build(name)
    assert shape.name == name
    for role, column in shape.roles.items():
        assert column in shape.frame.columns, f"role {role!r} -> missing column {column!r}"
    assert shape.note  # human description present


def test_sentinel_shape_injects_isolated_low_outlier():
    shape = ds.build("sentinel_numeric_column", sentinel=-999.0)
    col = shape.frame[shape.role("feature")]
    sentinel = shape.extra["sentinel"]
    share = float((col == sentinel).mean())
    assert share == pytest.approx(shape.extra["share"], abs=1e-9)
    assert share >= 0.01  # above detect_sentinel_values min_share
    # sentinel is the column minimum, far below the clean [600,700] band.
    assert col.min() == sentinel
    assert col[col != sentinel].min() >= 600.0


def test_snapshot_panel_is_monotone_ever_bad_per_cohort():
    shape = ds.build("snapshot_vintage_panel")
    frame = shape.frame
    for (_cohort, _loan), grp in frame.groupby([shape.role("cohort"), "loan"]):
        seq = grp.sort_values(shape.role("mob"))[shape.role("bad")].tolist()
        # once bad, stays bad (non-decreasing 0..1 within a loan)
        assert seq == sorted(seq)
    # and per-(cohort, mob) bad_count is non-decreasing across ascending mob
    for _cohort, grp in frame.groupby(shape.role("cohort")):
        by_mob = grp.groupby(shape.role("mob"))[shape.role("bad")].sum().sort_index().tolist()
        assert all(later >= earlier for earlier, later in zip(by_mob, by_mob[1:]))


def test_null_label_shape_has_the_declared_nan_count():
    shape = ds.build("null_and_illegal_labels")
    values = pd.to_numeric(shape.frame[shape.role("target")], errors="coerce")
    assert int(values.isna().sum()) == shape.extra["n_nan"]
    assert shape.extra["n_nan"] > 0


def test_string_yn_labels_are_non_numeric():
    shape = ds.build("string_yn_labels")
    col = shape.frame[shape.role("target")]
    assert set(col.unique()) <= {"Y", "N"}
    with pytest.raises((ValueError, TypeError)):
        pd.to_numeric(col, errors="raise")


def test_float64_long_id_keys_are_float_and_lose_precision_in_naive_cast():
    shape = ds.build("float64_long_id_keys")
    anchor = shape.extra["anchor_frame"]
    key = shape.role("anchor_key")
    assert anchor[key].dtype == np.float64
    # a naive str() of the float renders scientific notation -> the exact trap.
    assert "e+" in repr(float(anchor[key].iloc[0]))


@pytest.mark.parametrize("name", ["blank_join_keys"])
def test_blank_keys_present_on_both_sides(name):
    shape = ds.build(name)
    anchor = shape.extra["anchor_frame"][shape.role("anchor_key")]
    feature = shape.extra["feature_frame"][shape.role("feature_key")]
    assert (anchor.str.strip() == "").any()  # a blank/whitespace anchor key exists
    assert (feature.str.strip() == "").any()  # a blank feature key exists


def test_zero_padded_keys_keep_leading_zeros():
    shape = ds.build("zero_padded_keys")
    keys = shape.extra["anchor_frame"][shape.role("anchor_key")].tolist()
    assert "007" in keys and "012" in keys


def test_custom_split_vocabulary_avoids_standard_tokens():
    shape = ds.build("custom_split_vocabulary")
    tokens = set(shape.frame[shape.role("split")].unique())
    assert tokens.isdisjoint({"train", "test", "oot"})
    # the mapping still names all three modeling roles
    assert set(shape.extra["split_values"]) == {"train", "test", "oot"}


def test_duplicate_anchor_keys_actually_duplicated():
    shape = ds.build("duplicate_anchor_keys")
    keys = shape.extra["anchor_frame"][shape.role("anchor_key")]
    assert keys.duplicated().any()
    assert shape.extra["expected_rows"] == len(shape.extra["anchor_frame"])


def test_slice_null_labels_mixes_null_and_nonbinary():
    shape = ds.build("slice_null_labels")
    col = shape.frame[shape.role("target")]
    numeric = pd.to_numeric(col, errors="coerce")
    unlabeled = int((~numeric.isin([0, 1])).sum())
    assert unlabeled == shape.extra["expected_unlabeled"]


def test_time_panel_spans_multiple_months():
    shape = ds.build("time_column_oot_panel")
    months = shape.frame[shape.role("time")].unique()
    assert len(months) >= 3
    # the time column name is deliberately NOT a common alias
    assert shape.role("time") not in {"apply_month", "observation_date", "month"}

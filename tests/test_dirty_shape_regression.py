"""T4-1: the dirty-shape regression net.

This is the safety net that binds every T1 semantic-correctness fix to a
reproducible adversarial input from :mod:`support.dirty_shapes`. Each test feeds
one generated dirty shape and asserts the *fixed* behavior — vintage snapshot
flags don't double-count, sentinel values reach the replay chain, NULL labels are
dropped from denominators, float64 long ids still match, blank keys don't
mis-match, and so on.

If any T1 fix is ever reverted, the corresponding test here goes red. The
generators live in one place (``support.dirty_shapes``) and the concrete
per-fix behaviour is asserted here; where the platform already ships a
hand-written failure-shape test for a fix, this net exercises the SAME public
API against the SAME shape rather than re-deriving the logic, so the two stay in
lock-step.

Fast tier: no ``@slow`` marker — every case runs in well under a second.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from support import dirty_shapes as ds

from marvis.data.backend import DataBackend
from marvis.data.contracts import ColumnFingerprint, KeyPair
from marvis.data.errors import NanLabelNotConfirmedError


# --------------------------------------------------------------------------- #
# helpers shared with the platform's own backend tests                          #
# --------------------------------------------------------------------------- #
def _fingerprint() -> ColumnFingerprint:
    return ColumnFingerprint(
        value_kind="categorical",
        length_mode=None,
        regex_pattern=None,
        is_hashed=False,
        hash_type=None,
        hex_case=None,
        date_format=None,
    )


def _exact_key_pair() -> KeyPair:
    return KeyPair(
        anchor_col="id",
        feature_col="id",
        match_method="exact",
        transform_side="both",
        match_rate=1.0,
        resolved_by="test",
    )


def _write_join_pair(tmp_path, shape: ds.DirtyShape, suffix: str = "parquet"):
    anchor_path = tmp_path / f"anchor.{suffix}"
    feature_path = tmp_path / f"feature.{suffix}"
    anchor = shape.extra["anchor_frame"]
    feature = shape.extra["feature_frame"]
    if suffix == "csv":
        anchor.to_csv(anchor_path, index=False)
        feature.to_csv(feature_path, index=False)
    else:
        anchor.to_parquet(anchor_path, index=False)
        feature.to_parquet(feature_path, index=False)
    return anchor_path, feature_path


# --------------------------------------------------------------------------- #
# T1-A1 — vintage snapshot flag must not double-count                           #
# --------------------------------------------------------------------------- #
def test_snapshot_vintage_does_not_inflate_cum_bad_rate():
    """T1-A1 (validation/vintage.compute_vintage_curve): a snapshot/ever-bad flag
    read with ``label_semantics='snapshot'`` reports the per-MOB marginal rate and
    never re-accumulates, so cum_bad_rate stays bounded at the true rate. Reading
    the SAME shape as 'incremental' both inflates it AND raises the advisory flag."""
    from marvis.validation.vintage import compute_vintage_curve, vintage_curve_wide

    shape = ds.build("snapshot_vintage_panel")
    frame = shape.frame
    kw = dict(cohort_col=shape.role("cohort"), mob_col=shape.role("mob"), target_col=shape.role("bad"))

    snapshot_points = compute_vintage_curve(frame, label_semantics="snapshot", **kw)
    incremental_points = compute_vintage_curve(frame, label_semantics="incremental", **kw)

    snap_wide = vintage_curve_wide(snapshot_points, metric="cum_bad_rate")
    inc_wide = vintage_curve_wide(incremental_points, metric="cum_bad_rate")

    # snapshot: cum_bad_rate at the last MOB equals the marginal (ever-bad) rate,
    # which is a real fraction <= 1 and does NOT keep climbing by accumulation.
    for cohort, series in snap_wide.items():
        last = series[-1]
        assert last is not None and 0.0 <= last <= 1.0
    # the double-count is real and detectable: for at least one cohort the naive
    # incremental reading sits STRICTLY above the correct snapshot reading.
    inflated = any(
        inc_wide[cohort][-1] > snap_wide[cohort][-1] + 1e-9
        for cohort in snap_wide
        if snap_wide[cohort][-1] is not None
    )
    assert inflated, "expected the incremental reading to over-count the snapshot flag"

    # and the incremental reading raises the advisory data-quality red flag.
    warnings = [w for p in incremental_points for w in p.data_quality_warnings]
    assert any("SNAPSHOT" in w for w in warnings)
    assert not any(p.data_quality_warnings for p in snapshot_points)


def test_undeclared_vintage_semantics_raises_typed_gate():
    """T1-A1 gate: the strategy vintage path recognises this snapshot-shaped panel
    (via the shared kernel heuristic) and, when label_semantics is undeclared, hands
    a typed stop with structured diagnostics — never a silent guess. Binds the exact
    heuristic + typed error the ``tool_vintage_curve`` gate uses, without standing up
    the full pack runtime."""
    from marvis.data.errors import LabelSemanticsNotDeclaredError
    from marvis.packs.strategy.tools import (
        _vintage_cohort_count,
        _vintage_looks_like_snapshot,
    )

    shape = ds.build("snapshot_vintage_panel")
    frame = shape.frame
    cohort_col, mob_col, bad_col = shape.role("cohort"), shape.role("mob"), shape.role("bad")

    # the shared heuristic recognises the monotone ever-bad panel as snapshot-shaped
    monotone = _vintage_looks_like_snapshot(frame, cohort_col, mob_col, bad_col)
    assert monotone is True

    # and the gate the tool raises when label_semantics is omitted carries the right
    # structured diagnostics (this is exactly how tool_vintage_curve constructs it).
    error = LabelSemanticsNotDeclaredError(
        target_col=bad_col,
        n_cohorts=_vintage_cohort_count(frame, cohort_col),
        monotone_heuristic=monotone,
    )
    detail = error.to_detail()
    assert detail["kind"] == "label_semantics_not_declared"
    assert detail["monotone_heuristic"] is True
    assert detail["n_cohorts"] == shape.extra["n_cohorts"]


# --------------------------------------------------------------------------- #
# T1-A3 — sentinel masking enters the preprocessing replay chain                #
# --------------------------------------------------------------------------- #
def test_sentinel_is_detected_and_masked_before_stats():
    """T1-A3: the sentinel injected by the generator is (a) detected by the
    deterministic detector and (b) masked to NaN by ``mask_sentinel_values`` so it
    never contaminates a downstream statistic — the exact behaviour the replay
    ``sentinel`` step reproduces at serve time."""
    from marvis.feature.transform import detect_sentinel_values, mask_sentinel_values

    shape = ds.build("sentinel_numeric_column", sentinel=-999.0)
    col = shape.frame[shape.role("feature")]

    detected = detect_sentinel_values(col.to_numpy(dtype=float))
    assert any(value == shape.extra["sentinel"] for value, _share in detected)

    masked = mask_sentinel_values(col, [shape.extra["sentinel"]])
    assert masked.isna().sum() == shape.extra["n_sentinel"]
    # the sentinel is gone from the finite values -> the column min is back in-band.
    assert masked.dropna().min() >= 600.0


def test_sentinel_replay_step_masks_on_reapply():
    """T1-A3: the emitted preprocessing ``sentinel`` step, replayed via
    ``apply_preprocessing_steps``, masks the sentinel on fresh data (train/serve
    parity)."""
    from marvis.feature.preprocessing import apply_preprocessing_steps

    shape = ds.build("sentinel_numeric_column", sentinel=-999.0)
    feature = shape.role("feature")
    step = {"kind": "sentinel", "columns": [feature], "params": {feature: [shape.extra["sentinel"]]}}
    out = apply_preprocessing_steps(shape.frame.copy(), [step])
    assert out[feature].isna().sum() == shape.extra["n_sentinel"]
    assert not (out[feature] == shape.extra["sentinel"]).any()


# --------------------------------------------------------------------------- #
# T1-A4 — slice bad_rate drops NULL / non-binary labels from the denominator    #
# --------------------------------------------------------------------------- #
def test_slice_bad_rate_excludes_unlabeled_rows():
    """T1-A4 (data_ops._metric_expr): NULL / '' / non-binary labels must fall out
    of the bad_rate denominator (never counted as good=0), and the unlabeled_count
    companion must report exactly how many were excluded. Bound directly against
    the SQL the tool emits, run on DuckDB."""
    from marvis.packs.data_ops.tools import _metric_expr, _unlabeled_count_expr

    shape = ds.build("slice_null_labels")
    target = shape.role("target")
    allowed = set(shape.frame.columns)
    con = duckdb.connect()
    con.register("t", shape.frame)

    bad_rate_expr = _metric_expr("bad_rate", target, allowed)
    unlabeled_expr = _unlabeled_count_expr("bad_rate", target, allowed)
    bad_rate = con.execute(f"SELECT {bad_rate_expr} FROM t").fetchone()[0]
    unlabeled = con.execute(f"SELECT {unlabeled_expr} FROM t").fetchone()[0]

    assert bad_rate == pytest.approx(shape.extra["expected_bad_rate"], abs=1e-9)
    assert int(unlabeled) == shape.extra["expected_unlabeled"]


# --------------------------------------------------------------------------- #
# T1-A5 — blank / whitespace join keys are missing, not matchable               #
# --------------------------------------------------------------------------- #
def test_blank_join_keys_do_not_mismatch(tmp_path):
    """T1-A5: a blank-keyed anchor row falls through to NULL feature columns
    instead of attaching to a blank-keyed feature row, and the match-rate
    diagnostic agrees with the executed join."""
    shape = ds.build("blank_join_keys")
    anchor_path, feature_path = _write_join_pair(tmp_path, shape)
    out_path = tmp_path / "joined.parquet"
    backend = DataBackend(tmp_path)
    fp = _fingerprint()

    matched, sampled = backend.match_rate_for_method(
        anchor_path, ["id"], feature_path, ["id"],
        method="exact", key_fingerprints=[(fp, fp)], sample_n=10, seed=0,
    )
    rows = backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    joined = pd.read_parquet(out_path)
    realized = int(joined["limit"].notna().sum())

    assert rows == len(shape.extra["anchor_frame"])  # LEFT JOIN preserves anchor rows
    assert realized == shape.extra["expected_matches"] == 1  # only 'A1' matches
    assert matched == realized  # diagnostic prediction equals executed join


# --------------------------------------------------------------------------- #
# T1-A6 — float64-stored long ids match and normalize consistently              #
# --------------------------------------------------------------------------- #
def test_float64_long_ids_match_across_join(tmp_path):
    """T1-A6: 18-digit ids stored as float64 on both sides still match — no
    scientific-notation drift between the SQL and Python normalizers."""
    shape = ds.build("float64_long_id_keys")
    anchor_path, feature_path = _write_join_pair(tmp_path, shape)
    out_path = tmp_path / "joined.parquet"
    backend = DataBackend(tmp_path)

    rows = backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    joined = pd.read_parquet(out_path).sort_values("score").reset_index(drop=True)
    assert rows == shape.extra["expected_matches"] == 3
    assert joined["limit"].tolist() == [100, 200, 300]


def test_float64_long_id_sql_and_python_normalizer_agree():
    """T1-A6: the DuckDB and Python key normalizers render the SAME string for a
    float64-stored long id, so SQL-vs-Python key comparisons cannot diverge."""
    from marvis.data.backend import _sql_value_text, _value_text

    shape = ds.build("float64_long_id_keys")
    value = float(shape.extra["anchor_frame"][shape.role("anchor_key")].iloc[0])
    con = duckdb.connect()
    con.register("t", pd.DataFrame({"x": [value]}))
    key_expr = _sql_value_text('a."x"')
    sql_result = con.execute(f"SELECT {key_expr} FROM t a").fetchone()[0]
    assert sql_result == _value_text(value)
    assert "e+" not in str(sql_result)  # no scientific notation leaked


# --------------------------------------------------------------------------- #
# T1-B7 — zero-padded keys: diagnostics and execution agree                     #
# --------------------------------------------------------------------------- #
def test_zero_padded_keys_diagnostic_matches_execution(tmp_path):
    """T1-B7: a zero-padded CSV key must produce a match-rate diagnostic equal to
    the realized left join (both read through the same typed reader)."""
    shape = ds.build("zero_padded_keys")
    anchor_path, feature_path = _write_join_pair(tmp_path, shape, suffix="csv")
    out_path = tmp_path / "joined.parquet"
    backend = DataBackend(tmp_path)
    fp = _fingerprint()

    matched, _sampled = backend.match_rate_for_method(
        anchor_path, ["id"], feature_path, ["id"],
        method="exact", key_fingerprints=[(fp, fp)], sample_n=10, seed=0,
    )
    backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    realized = int(pd.read_parquet(out_path)["val"].notna().sum())
    assert matched == realized == shape.extra["expected_matches"] == 2


# --------------------------------------------------------------------------- #
# T1-B8 — text<->float key dtype divergence is RED (forces confirmation)        #
# --------------------------------------------------------------------------- #
def test_text_vs_float_key_is_red_divergence():
    """T1-B8: a float64 anchor key vs a zero-padded-text feature key is classified
    'red' — the leading-zero/precision-loss case that the join engine blocks until
    the caller acknowledges it."""
    from marvis.data.align import _divergence_level, _dtype_family_from_str

    shape = ds.build("text_vs_float_key")
    anchor_dtype = str(shape.extra["anchor_frame"][shape.role("anchor_key")].dtype)
    feature_dtype = str(shape.extra["feature_frame"][shape.role("feature_key")].dtype)

    assert _dtype_family_from_str(anchor_dtype) == "float"
    assert _dtype_family_from_str(feature_dtype) == "text"
    assert _divergence_level(anchor_dtype, feature_dtype) == "red"


# --------------------------------------------------------------------------- #
# T1-D13 — NaN label gate on screening                                          #
# --------------------------------------------------------------------------- #
def test_screen_nan_labels_gate_raises_then_drops(tmp_path):
    """T1-D13: NaN labels stop feature screening with a typed gate by default and
    are dropped (with an audit count) only on explicit ``drop_nan_labels=True`` —
    never coerced to a class."""
    from marvis.feature.screen import screen_features

    shape = ds.build("null_and_illegal_labels")
    dataset_path = tmp_path / "dev.parquet"
    shape.frame.to_parquet(dataset_path, index=False)
    backend = DataBackend(tmp_path)
    feature = shape.role("feature")
    target = shape.role("target")

    with pytest.raises(NanLabelNotConfirmedError) as excinfo:
        screen_features(backend, dataset_path, features=[feature], target_col=target)
    assert excinfo.value.to_detail()["n_nan"] == shape.extra["n_nan"]

    result = screen_features(
        backend, dataset_path, features=[feature], target_col=target, drop_nan_labels=True
    )
    assert result.nan_labels_dropped == shape.extra["n_nan"]


def test_string_yn_labels_are_hard_error():
    """T1 label contract: a 'Y'/'N' text target is a hard error (non-numeric), a
    distinct failure from the NaN confirmation gate."""
    from marvis.validation.checks import validate_binary_target

    shape = ds.build("string_yn_labels")
    with pytest.raises(ValueError):
        validate_binary_target(shape.frame, shape.role("target"))


# --------------------------------------------------------------------------- #
# T1-D11/holdout — custom split vocabulary still splits by the mapping          #
# --------------------------------------------------------------------------- #
def test_custom_split_vocabulary_resolves_by_mapping(tmp_path):
    """T1 split handling: a custom split vocabulary (build/holdout/future) still
    resolves train/test/oot through the caller-supplied ``split_values`` mapping
    rather than a hard-coded literal, so training runs end-to-end on it and the
    holdout ('future'/oot) rows are excluded from the fit."""
    from marvis.packs.modeling.contracts import TrainConfig
    from marvis.packs.modeling.recipes.common import split_modeling_frame

    shape = ds.build("custom_split_vocabulary")
    config = TrainConfig(
        dataset_id="dirty",
        features=(shape.role("feature"),),
        target_col=shape.role("target"),
        split_col=shape.role("split"),
        split_values=shape.extra["split_values"],
        params={},
        seed=0,
        early_stopping_rounds=None,
        recipe_id="lr",
    )
    train, test, oot = split_modeling_frame(shape.frame, config)
    # every split resolves and is non-empty; the mapping (not the literal) drove it.
    assert not train.empty and not test.empty and oot is not None and not oot.empty
    # no standard-vocabulary rows leaked into the frames.
    for part in (train, test, oot):
        assert set(part[shape.role("split")].unique()).issubset({"build", "holdout", "future"})


# --------------------------------------------------------------------------- #
# duplicate anchor keys — left join preserves every anchor row                  #
# --------------------------------------------------------------------------- #
def test_duplicate_anchor_keys_preserved(tmp_path):
    """A duplicated anchor key must not drop or fan out the anchor side: the left
    join result has exactly the anchor row count and both duplicate rows carry the
    matched feature."""
    shape = ds.build("duplicate_anchor_keys")
    anchor_path, feature_path = _write_join_pair(tmp_path, shape)
    out_path = tmp_path / "joined.parquet"
    backend = DataBackend(tmp_path)

    rows = backend.left_join(
        anchor_path, feature_path, [_exact_key_pair()], dedup_strategy=None, out_path=out_path
    )
    joined = pd.read_parquet(out_path)
    assert rows == shape.extra["expected_rows"] == 3
    a1 = joined[joined["id"] == "A1"]
    assert len(a1) == 2  # both duplicate anchor rows survive
    assert (a1["limit"] == 100).all()  # both carry the matched feature


# --------------------------------------------------------------------------- #
# T1-D12 — a non-alias time column drives a time-based OOT split                #
# --------------------------------------------------------------------------- #
def test_non_alias_time_column_is_honored_for_oot():
    """T1-D12 (modeling_setup._resolve_effective_time_col): an explicit time column
    that is present but NOT a known alias name is honored, so it can drive the
    time-OOT split."""
    from marvis.agent.modeling_setup import _resolve_effective_time_col

    shape = ds.build("time_column_oot_panel")
    time_col = shape.role("time")
    resolved = _resolve_effective_time_col(time_col, list(shape.frame.columns), business_columns={})
    assert resolved == time_col


def test_time_column_produces_oot_split_with_latest_months():
    """T1-D12: ``oot_by_time`` on the same non-alias time column actually carves an
    OOT split holding the most recent slice of the timeline (proving the column
    drives the split, not just resolves)."""
    from marvis.packs.modeling.prepare import _make_split

    shape = ds.build("time_column_oot_panel")
    time_col = shape.role("time")
    split = _make_split(
        shape.frame,
        {"oot_by_time": time_col, "oot_size": 0.2, "test_size": 0.25},
        seed=0,
    )
    assert "oot" in set(split["split"])
    oot_months = set(split.loc[split["split"] == "oot", time_col])
    non_oot_months = set(split.loc[split["split"] != "oot", time_col])
    # the OOT slice is the timeline TAIL: every OOT month is strictly later than every
    # kept (train/test) month, and it holds the newest month in the data.
    assert min(oot_months) > max(non_oot_months)
    assert max(oot_months) == max(set(shape.frame[time_col]))


# --------------------------------------------------------------------------- #
# registry coverage guard — every generated shape is bound somewhere            #
# --------------------------------------------------------------------------- #
def test_every_dirty_shape_has_a_binding():
    """Guard: each registered dirty shape must be referenced by at least one
    regression test in this module, so a newly-added shape can't be silently
    unbound. Kept as an explicit allow-list mapping shape -> the fix it guards."""
    bound = {
        "sentinel_numeric_column",
        "snapshot_vintage_panel",
        "null_and_illegal_labels",
        "string_yn_labels",
        "float64_long_id_keys",
        "text_vs_float_key",
        "blank_join_keys",
        "zero_padded_keys",
        "custom_split_vocabulary",
        "duplicate_anchor_keys",
        "slice_null_labels",
        "time_column_oot_panel",
    }
    registered = set(ds.DIRTY_SHAPES)
    missing = registered - bound
    assert not missing, f"dirty shapes without a regression binding: {sorted(missing)}"
    stale = bound - registered
    assert not stale, f"bindings reference unknown shapes: {sorted(stale)}"

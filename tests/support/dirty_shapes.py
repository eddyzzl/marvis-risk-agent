"""T4-1: deterministic adversarial "dirty shape" injector.

Every known dirty data shape from the T1 semantic-correctness fix pack gets ONE
factory function here. Each factory is:

- **Deterministic** — no ``random``/wall-clock; any variation comes from a fixed
  ``seed`` fed to ``numpy.random.default_rng`` so the same call always returns the
  same bytes. The dirty *shape* (blank keys, sentinel spikes, snapshot flags, ...)
  is always structurally present regardless of seed.
- **Small** — a few dozen rows, enough to exercise the fix, cheap enough for the
  CI fast tier.
- **Self-describing** — returns a :class:`DirtyShape` carrying the frame plus the
  column roles and a human note, so the regression net can bind to it without
  hard-coding column names.

The factories are registered in :data:`DIRTY_SHAPES` (name -> builder). Adding a new
dirty shape is a new ``@register`` factory — the regression net and the self-test
iterate the registry, so a new shape is picked up without touching either.

These generators intentionally mirror the exact shapes the hand-written T1 tests
already use (``tests/test_data_backend.py``, ``tests/test_strategy_vintage.py``,
``tests/validation/test_vintage.py``, ``tests/test_data_ops_slice_aggregate.py``)
so the consolidated net exercises the SAME adversarial inputs those fixes were
written against, from one reusable place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirtyShape:
    """A generated dirty dataset plus the roles a test needs to bind to it.

    ``roles`` maps a semantic role name (e.g. ``"anchor_key"``, ``"target"``,
    ``"cohort"``) to the concrete column name in :attr:`frame`, so a regression
    test never hard-codes a column literal. ``extra`` carries any structured
    truth a test needs to assert against (e.g. the injected sentinel value, or
    the expected number of unlabeled rows).
    """

    name: str
    frame: pd.DataFrame
    roles: dict[str, str]
    note: str
    extra: dict = field(default_factory=dict)

    # Some shapes ship an anchor + a feature table for a join. When present, the
    # secondary frame lives in ``extra["feature_frame"]`` / ``extra["anchor_frame"]``.
    def role(self, name: str) -> str:
        return self.roles[name]


DirtyShapeFactory = Callable[..., DirtyShape]

DIRTY_SHAPES: dict[str, DirtyShapeFactory] = {}


def register(name: str) -> Callable[[DirtyShapeFactory], DirtyShapeFactory]:
    """Register a dirty-shape factory under ``name`` (extensibility seam)."""

    def _wrap(factory: DirtyShapeFactory) -> DirtyShapeFactory:
        if name in DIRTY_SHAPES:
            raise ValueError(f"duplicate dirty shape: {name}")
        DIRTY_SHAPES[name] = factory
        factory.dirty_shape_name = name  # type: ignore[attr-defined]
        return factory

    return _wrap


# --------------------------------------------------------------------------- #
# 1. sentinel values mixed into a numeric column                               #
# --------------------------------------------------------------------------- #
@register("sentinel_numeric_column")
def sentinel_numeric_column(*, seed: int = 0, sentinel: float = -999.0) -> DirtyShape:
    """A numeric feature whose values sit in a plausible band, with a sentinel
    (default -999) spiked in at ~15% share, far below the real distribution.

    Mirrors PREP-4 / T1-A3: a bureau feed encoding "no hit" as -999. The clean
    values are pushed to [600, 700] so the sentinel is a clear low-side outlier
    (``detect_sentinel_values`` requires the candidate at an extreme of the range,
    isolated by a large gap)."""
    rng = np.random.default_rng(seed)
    n_clean = 170
    n_sentinel = 30  # 15% share, comfortably above the 1% min_share threshold
    clean = rng.uniform(600.0, 700.0, size=n_clean).round(2)
    values = np.concatenate([clean, np.full(n_sentinel, float(sentinel))])
    # Interleave deterministically so the sentinel isn't a trailing block.
    order = rng.permutation(len(values))
    feature = values[order]
    target = rng.integers(0, 2, size=len(values))
    frame = pd.DataFrame({"score": feature, "y": target})
    return DirtyShape(
        name="sentinel_numeric_column",
        frame=frame,
        roles={"feature": "score", "target": "y"},
        note=f"{n_sentinel} sentinel {sentinel} values ({n_sentinel/len(values):.0%}) in an otherwise [600,700] column",
        extra={"sentinel": float(sentinel), "n_sentinel": int(n_sentinel), "share": n_sentinel / len(values)},
    )


# --------------------------------------------------------------------------- #
# 2. snapshot (ever-bad) vintage panel                                          #
# --------------------------------------------------------------------------- #
@register("snapshot_vintage_panel")
def snapshot_vintage_panel(*, seed: int = 0) -> DirtyShape:
    """A vintage panel where ``bad`` is a SNAPSHOT / ever-bad flag: a loan that
    goes bad stays 1 for every later MOB. Per-MOB bad_count is non-decreasing
    within each cohort, so an ``incremental`` reading would re-accumulate and
    virtually inflate cum_bad_rate.

    Structure (mirrors ``_snapshot_flag_frame`` in the T1 vintage tests): two
    cohorts, MOB 1..3, each loan flagged from the MOB it first goes bad onward.
    """
    # cohort 2025-01: loan a bad from mob2; loan b bad from mob3; loan c never bad.
    # cohort 2025-02: loan d bad from mob1; loan e never bad; loan f bad from mob2.
    rows: list[dict] = []

    def emit(cohort: str, loan: str, first_bad_mob: int | None) -> None:
        for mob in (1, 2, 3):
            bad = 1 if (first_bad_mob is not None and mob >= first_bad_mob) else 0
            rows.append({"cohort": cohort, "loan": loan, "mob": mob, "bad": bad})

    emit("2025-01", "a", 2)
    emit("2025-01", "b", 3)
    emit("2025-01", "c", None)
    emit("2025-02", "d", 1)
    emit("2025-02", "e", None)
    emit("2025-02", "f", 2)
    frame = pd.DataFrame(rows)
    return DirtyShape(
        name="snapshot_vintage_panel",
        frame=frame,
        roles={"cohort": "cohort", "mob": "mob", "bad": "bad"},
        note="ever-bad snapshot flag: bad_count non-decreasing across every cohort/MOB",
        extra={"n_cohorts": 2, "mob_max": 3},
    )


# --------------------------------------------------------------------------- #
# 3. NULL / illegal labels                                                      #
# --------------------------------------------------------------------------- #
@register("null_and_illegal_labels")
def null_and_illegal_labels(*, seed: int = 0) -> DirtyShape:
    """A target column carrying NaN (empty) alongside legal 0/1. The empty cells
    become NaN under ``pd.to_numeric`` — the NaN-label confirmation gate must fire
    (or drop them on opt-in), never coerce them to a class.

    ``extra["n_nan"]`` is the count the gate should report."""
    rng = np.random.default_rng(seed)
    n = 40
    features = rng.normal(0.0, 1.0, size=n).round(3)
    labels: list[object] = []
    n_nan = 0
    for i in range(n):
        if i % 10 == 0:  # 4 NaN labels
            labels.append(np.nan)
            n_nan += 1
        else:
            labels.append(int(rng.integers(0, 2)))
    frame = pd.DataFrame({"x1": features, "y": pd.Series(labels, dtype="float64")})
    return DirtyShape(
        name="null_and_illegal_labels",
        frame=frame,
        roles={"feature": "x1", "target": "y"},
        note=f"{n_nan} NaN labels mixed into 0/1 target",
        extra={"n_nan": n_nan, "n_total": n},
    )


@register("string_yn_labels")
def string_yn_labels(*, seed: int = 0) -> DirtyShape:
    """A target column stored as 'Y'/'N' text — a non-numeric label that the
    binary-target contract must reject as a hard error (distinct from the NaN
    case, which is a confirmation gate)."""
    rng = np.random.default_rng(seed)
    n = 30
    features = rng.normal(0.0, 1.0, size=n).round(3)
    labels = np.where(rng.integers(0, 2, size=n) == 1, "Y", "N")
    frame = pd.DataFrame({"x1": features, "y": labels})
    return DirtyShape(
        name="string_yn_labels",
        frame=frame,
        roles={"feature": "x1", "target": "y"},
        note="target stored as 'Y'/'N' strings (non-numeric, must hard-error)",
        extra={},
    )


# --------------------------------------------------------------------------- #
# 4. float64-stored long integer id keys (scientific notation trap)             #
# --------------------------------------------------------------------------- #
@register("float64_long_id_keys")
def float64_long_id_keys(*, seed: int = 0) -> DirtyShape:
    """An 18-digit national-id join key stored as float64 on both sides. Under a
    naive ``CAST(... AS VARCHAR)`` it renders in scientific notation
    (``1.2345678901234568e+17``), so a float-vs-float or float-vs-string join
    silently mismatches. Provides an anchor + feature frame keyed on ``id``.

    Mirrors ``test_left_join_matches_long_float_ids_both_sides_float``."""
    ids = [123456789012345678.0, 223456789012345678.0, 323456789012345678.0]
    anchor = pd.DataFrame({"id": ids, "score": [0.1, 0.2, 0.3]})
    feature = pd.DataFrame({"id": ids, "limit": [100, 200, 300]})
    return DirtyShape(
        name="float64_long_id_keys",
        frame=anchor,
        roles={"anchor_key": "id", "feature_key": "id"},
        note="18-digit ids stored as float64 (scientific-notation cast trap)",
        extra={"anchor_frame": anchor, "feature_frame": feature, "expected_matches": 3},
    )


# --------------------------------------------------------------------------- #
# 5b. text-on-one-side / float64-on-the-other join key (dtype divergence)       #
# --------------------------------------------------------------------------- #
@register("text_vs_float_key")
def text_vs_float_key(*, seed: int = 0) -> DirtyShape:
    """A join key stored as float64 on the anchor and as leading-zero text on the
    feature ('001234'). This text<->float divergence is a RED dtype mismatch:
    casting the text to float would drop leading zeros and can silently mis-match
    rows, so the join engine must force an explicit acknowledgement
    (``KeyDtypeMismatchError``) before executing.

    Mirrors T1-B8: the leading-zero short-code that the float side cannot represent.
    """
    anchor = pd.DataFrame({"id": [1234.0, 5678.0, 9012.0], "score": [0.1, 0.2, 0.3]})
    feature = pd.DataFrame({"id": ["001234", "005678", "009012"], "limit": [10, 20, 30]})
    return DirtyShape(
        name="text_vs_float_key",
        frame=anchor,
        roles={"anchor_key": "id", "feature_key": "id"},
        note="anchor key float64 vs feature key zero-padded text (RED dtype divergence)",
        extra={"anchor_frame": anchor, "feature_frame": feature},
    )


# --------------------------------------------------------------------------- #
# 5. blank / whitespace-only join keys                                          #
# --------------------------------------------------------------------------- #
@register("blank_join_keys")
def blank_join_keys(*, seed: int = 0) -> DirtyShape:
    """Anchor and feature frames with blank ('') and whitespace-only ('   ') join
    keys. A blank key is MISSING, not matchable: a blank-keyed anchor row must
    fall through to NULL feature columns, never attach to a blank-keyed feature
    row. Only the real 'A1' key matches.

    Mirrors ``test_left_join_treats_blank_keys_as_missing_not_matchable``."""
    anchor = pd.DataFrame({"id": ["", "   ", "A1"], "score": [0.1, 0.2, 0.3]})
    feature = pd.DataFrame({"id": ["", "A1"], "limit": [999, 100]})
    return DirtyShape(
        name="blank_join_keys",
        frame=anchor,
        roles={"anchor_key": "id", "feature_key": "id"},
        note="blank '' and whitespace '   ' join keys (must be treated as missing)",
        extra={"anchor_frame": anchor, "feature_frame": feature, "expected_matches": 1},
    )


# --------------------------------------------------------------------------- #
# 6. zero-padded join keys                                                       #
# --------------------------------------------------------------------------- #
@register("zero_padded_keys")
def zero_padded_keys(*, seed: int = 0) -> DirtyShape:
    """Zero-padded string codes ('007', '012') as join keys. A typed reader that
    coerces '007' -> int 7 while the other side stays '007' text would diverge
    diagnostics from execution. Match-rate diagnostic must equal the realized
    join (both '007' and '012' match; '099' does not).

    Mirrors ``test_match_rate_matches_left_join_for_zero_padded_csv_keys``."""
    anchor = pd.DataFrame({"id": ["007", "012", "099"]})
    feature = pd.DataFrame({"id": ["007", "012"], "val": [1, 2]})
    return DirtyShape(
        name="zero_padded_keys",
        frame=anchor,
        roles={"anchor_key": "id", "feature_key": "id"},
        note="zero-padded '007'/'012' keys (leading zeros must survive)",
        extra={"anchor_frame": anchor, "feature_frame": feature, "expected_matches": 2},
    )


# --------------------------------------------------------------------------- #
# 7. custom split vocabulary (non train/test/oot)                               #
# --------------------------------------------------------------------------- #
@register("custom_split_vocabulary")
def custom_split_vocabulary(*, seed: int = 0) -> DirtyShape:
    """A modeling frame whose split column uses a non-standard vocabulary
    ('build'/'holdout'/'future' instead of 'train'/'test'/'oot'). The split
    machinery must key off the caller-supplied ``split_values`` mapping, not a
    hard-coded 'train'/'test'/'oot' literal, so a custom vocabulary still splits.
    """
    rng = np.random.default_rng(seed)
    n = 120
    x = rng.normal(0.0, 1.0, size=n)
    logit = 0.8 * x - 0.3
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)
    idx = np.arange(n)
    split = np.empty(n, dtype=object)
    split[idx % 5 < 3] = "build"
    split[idx % 5 == 3] = "holdout"
    split[idx % 5 == 4] = "future"
    frame = pd.DataFrame({"x1": x.round(3), "y": y, "phase": split})
    return DirtyShape(
        name="custom_split_vocabulary",
        frame=frame,
        roles={"feature": "x1", "target": "y", "split": "phase"},
        note="split column uses build/holdout/future, not train/test/oot",
        extra={"split_values": {"train": "build", "test": "holdout", "oot": "future"}},
    )


# --------------------------------------------------------------------------- #
# 8. duplicate anchor keys                                                       #
# --------------------------------------------------------------------------- #
@register("duplicate_anchor_keys")
def duplicate_anchor_keys(*, seed: int = 0) -> DirtyShape:
    """An anchor (sample) table with a duplicated key ('A1' twice). A left join
    must PRESERVE every anchor row — the row count of the result equals the anchor
    row count even with the duplicate — so a duplicated anchor key never silently
    drops or fans out the anchor side.

    (Fan-out control is a feature-side concern; the anchor may legitimately carry
    duplicates and both must survive the join.)"""
    anchor = pd.DataFrame({"id": ["A1", "A1", "B2"], "score": [0.1, 0.2, 0.3]})
    feature = pd.DataFrame({"id": ["A1", "B2"], "limit": [100, 200]})
    return DirtyShape(
        name="duplicate_anchor_keys",
        frame=anchor,
        roles={"anchor_key": "id", "feature_key": "id"},
        note="anchor key 'A1' appears twice (both rows must survive the left join)",
        extra={"anchor_frame": anchor, "feature_frame": feature, "expected_rows": 3},
    )


# --------------------------------------------------------------------------- #
# 9. slice bad_rate with NULL / non-binary labels                               #
# --------------------------------------------------------------------------- #
@register("slice_null_labels")
def slice_null_labels(*, seed: int = 0) -> DirtyShape:
    """A slice/aggregate frame where the ``bad`` column carries NULL/empty and a
    non-binary token ('x') alongside legal 0/1. bad_rate must drop the unlabeled
    rows from the denominator (never count them as good=0), and an unlabeled_count
    companion must expose exactly how many were excluded.

    Mirrors ``_partial_label_frame`` group G in the slice_aggregate tests: labels
    [1,1,0,'',x] -> labeled bad_rate 2/3, 2 unlabeled."""
    frame = pd.DataFrame(
        {
            "channel": ["G", "G", "G", "G", "G"],
            "bad": ["1", "1", "0", "", "x"],
        }
    )
    return DirtyShape(
        name="slice_null_labels",
        frame=frame,
        roles={"group": "channel", "target": "bad"},
        note="bad column mixes 0/1 with NULL/'' and non-binary 'x'",
        extra={"expected_bad_rate": 2.0 / 3.0, "expected_unlabeled": 2, "n_total": 5},
    )


# --------------------------------------------------------------------------- #
# 10. time-column OOT panel                                                     #
# --------------------------------------------------------------------------- #
@register("time_column_oot_panel")
def time_column_oot_panel(*, seed: int = 0) -> DirtyShape:
    """A modeling frame with a real (non-alias-named) time column ``apply_period``
    spanning several months. A time-driven OOT split must be derivable from this
    column — the latest month(s) become OOT — even though the column name isn't a
    known alias like 'apply_month' or 'observation_date'."""
    rng = np.random.default_rng(seed)
    months = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    n = 180
    period = rng.choice(months, size=n)
    x = rng.normal(0.0, 1.0, size=n)
    y = (rng.uniform(size=n) < (1.0 / (1.0 + np.exp(-(0.7 * x - 0.2))))).astype(int)
    frame = pd.DataFrame({"apply_period": period, "x1": x.round(3), "y": y})
    return DirtyShape(
        name="time_column_oot_panel",
        frame=frame,
        roles={"time": "apply_period", "feature": "x1", "target": "y"},
        note="non-alias time column 'apply_period' spanning 6 months for time-OOT",
        extra={"months": months},
    )


def build(name: str, **kwargs) -> DirtyShape:
    """Build a dirty shape by registered name."""
    if name not in DIRTY_SHAPES:
        raise KeyError(f"unknown dirty shape {name!r}; known: {sorted(DIRTY_SHAPES)}")
    return DIRTY_SHAPES[name](**kwargs)


def all_shapes(**kwargs) -> dict[str, DirtyShape]:
    """Build every registered dirty shape (used by the registry self-test)."""
    return {name: factory(**kwargs) for name, factory in DIRTY_SHAPES.items()}


__all__ = ["DIRTY_SHAPES", "DirtyShape", "all_shapes", "build", "register"]
